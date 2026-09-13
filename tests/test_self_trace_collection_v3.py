"""Focused synthetic checks for the current-source user-only V3 gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_self_trace_collection import source_fixture as v1_source_fixture
from test_self_trace_corpus import _rebind_acceptance, write

from evals.fleet import opencode_self_hosted as fleet
from training import dense, rl_data, skyrl_episode
from training import self_trace_collection as v1
from training import self_trace_collection_v2 as v2
from training import self_trace_collection_v3 as v3
from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
REQUEST = ROOT / "configs/qualification/qwen38-self-trace-collection-request-v3.json"
V2_REQUEST = ROOT / "configs/qualification/qwen38-self-trace-collection-request-v2.json"
PARITY = ROOT / "docs/evidence/qwen38-study/2026-09-13-self-trace-recorder-dense-parity-v3.json"
CATALOG = ROOT / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json"
PRIVATE = "SYNTHETIC_PRIVATE_USER_REQUEST_MUST_NOT_ESCAPE"


def read_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def reseal(value: dict) -> dict:
    value["sha256"] = digest_json({key: item for key, item in value.items() if key != "sha256"})
    return value


def bound(path: Path, value: dict) -> dict:
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "document_sha256": value["sha256"],
    }


def source_closure() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in v3.SOURCE_CLOSURE_PATHS.items()}


def v2_source_closure() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in v2.SOURCE_CLOSURE_PATHS.items()}


def frozen_model() -> dict:
    return json.loads(V2_REQUEST.read_text(encoding="utf-8"))["model"]


def collector_receipt() -> dict:
    model = frozen_model()
    image = "registry.example/fleet/skyrl-direct@sha256:" + "a" * 64
    receipt = {
        "schema": v2.COLLECTOR_QUALIFICATION_SCHEMA,
        "status": "qualified",
        "observed_at": "2026-09-12T00:00:00Z",
        "image": {
            "reference": image,
            "manifest_digest": "sha256:" + "a" * 64,
            "runtime_image_id": image,
            "platform": "linux/amd64",
        },
        "bindings": {
            "source_closure_sha256": digest_json(source_closure()),
            "backend": "skyrl_direct",
            "native_helper_sha256": dense.NATIVE_HELPER_SHA,
            "model_repo": model["repo"],
            "model_revision": model["revision"],
            "runtime_chat_template_sha256": v2.CHAT_TEMPLATE_SHA256,
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": v2.TOOL_CATALOG_SHA256,
            "prompt_policy_sha256": v2.PROMPT_POLICY_SHA256,
            "template_invocations_sha256": v2.TEMPLATE_INVOCATIONS_SHA256,
            "rendered_request_contract_sha256": v2.RENDERED_REQUEST_CONTRACT_SHA256,
        },
        "qualification": {
            "local_synthetic_exit_code": 0,
            "clean_pull_by_digest": True,
            "zero_gpu_dev": {
                "pod_uid": "00000000-0000-4000-8000-000000000001",
                "image_pull_policy": "Always",
                "runtime_image_id_matches": True,
                "gpu_requests": 0,
                "exit_code": 0,
                "restart_count": 0,
                "deleted": True,
                "absence_confirmed": True,
            },
            "target_model_weights_loaded": False,
            "task_or_scoring_requests": 0,
            "registry_publications": 0,
        },
    }
    return reseal(receipt)


def route_receipt(collector: dict) -> dict:
    model = frozen_model()
    receipt = {
        "schema": v2.ROUTE_CERTIFICATE_SCHEMA,
        "status": "qualified",
        "observed_at": "2026-09-12T00:01:00Z",
        "route": {
            "served_model_id": "qwen3.8-27b",
            "serving_block_kind": "shared",
            "object_identity_sha256": "sha256:" + "b" * 64,
            "normalized_server_arguments_sha256": "sha256:" + "c" * 64,
            "serving_image_digest": "sha256:" + "d" * 64,
            "ready_replicas": 1,
        },
        "model_artifact": {
            "repo": model["repo"],
            "revision": model["revision"],
            "lock_file_sha256": model["lock_file_sha256"],
            "weights_manifest_sha256": v2.WEIGHTS_MANIFEST_SHA256,
            "tokenizer_manifest_sha256": v2.TOKENIZER_MANIFEST_SHA256,
            "chat_template_sha256": v2.CHAT_TEMPLATE_SHA256,
            "payload_rehashed": True,
            "symlinks_absent": True,
        },
        "collector": {
            "qualification_sha256": collector["sha256"],
            "image": collector["image"]["reference"],
        },
        "interface": copy.deepcopy(v2.PARITY_INTERFACE),
        "native_transport": {
            "endpoint": "/inference/v1/generate",
            "request_contract": "prompt_token_ids_sampling_params",
            "response_contract": "responses_response_ids_response_logprobs_stop_reasons",
            "native_prompt_token_ids": True,
            "native_response_token_ids": True,
            "native_response_logprobs": True,
            "collector_rendered_bytes_verified": True,
            "prompt_or_tool_rewriting": False,
        },
        "fresh_observation": {
            "route_ready": True,
            "model_revision_readback": True,
            "runtime_identity_readback": True,
            "collector_to_route_synthetic_probe_passed": True,
            "target_task_prompt_used": False,
            "task_or_scoring_requests": 0,
        },
    }
    return reseal(receipt)


def qualified_request(tmp_path: Path) -> dict:
    request = read_request()
    collector = collector_receipt()
    collector_path = tmp_path / "collector.json"
    write(collector_path, collector)
    route = route_receipt(collector)
    route_path = tmp_path / "route.json"
    write(route_path, route)
    request["runtime"]["collector_qualification"] = bound(collector_path, collector)
    request["runtime"]["direct_route_certificate"] = bound(route_path, route)
    request["status"] = "qualified"
    return reseal(request)


def test_v3_preserves_v2_and_rebinds_only_current_source(capsys):
    request = read_request()
    predecessor = json.loads(V2_REQUEST.read_text(encoding="utf-8"))
    assert predecessor["sha256"] == v3.PREDECESSOR_SHA256
    assert file_sha256(V2_REQUEST) == v3.PREDECESSOR_FILE_SHA256
    with pytest.raises(v2.CollectionError, match="frozen v1"):
        v2.validate_request(predecessor, relative_to=V2_REQUEST.parent)

    validated = v3.validate_request(request, relative_to=REQUEST.parent)
    assert request["sha256"] == digest_json(
        {key: item for key, item in request.items() if key != "sha256"}
    )
    assert validated["base_request"]["schema"] == v1.REQUEST_SCHEMA
    assert validated["base_request"]["sha256"] == predecessor["base_request"]["document_sha256"]
    assert validated["blockers"] == [
        "missing_immutable_collector_qualification",
        "missing_exact_direct_route_certificate",
    ]
    assert len(validated["tasks"]) == 74
    assert len(validated["roster"]) == 296
    assert validated["predecessor"] == predecessor
    assert predecessor["interface"]["prompt_policy"] == v2.PROMPT_POLICY
    assert request["execution"]["launchable"] is False
    with pytest.raises(v3.CollectionError, match="not ready"):
        v3.validate_request(request, relative_to=REQUEST.parent, require_ready=True)

    assert v3.main(["--request", str(REQUEST)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["blockers"] == validated["blockers"]
    assert output["cluster_or_api_mutations_performed"] is False
    assert output["job_submission_performed"] is False
    assert output["launchable"] is False


@pytest.mark.parametrize(
    "defect",
    ["bare_image", "collector_policy", "route_collector", "route_transport"],
)
def test_runtime_requires_self_digesting_cross_bound_artifacts(tmp_path, defect):
    if defect == "bare_image":
        request = read_request()
        request["runtime"]["collector_qualification"] = (
            "registry.example/collector@sha256:" + "a" * 64
        )
        reseal(request)
        with pytest.raises(v3.CollectionError, match="artifact binding"):
            v3.validate_request(request, relative_to=REQUEST.parent)
        return

    request = qualified_request(tmp_path)
    if defect == "collector_policy":
        path = Path(request["runtime"]["collector_qualification"]["path"])
        artifact = json.loads(path.read_text())
        artifact["bindings"]["prompt_policy_sha256"] = "sha256:" + "0" * 64
    else:
        path = Path(request["runtime"]["direct_route_certificate"]["path"])
        artifact = json.loads(path.read_text())
        if defect == "route_collector":
            artifact["collector"]["qualification_sha256"] = "sha256:" + "0" * 64
        else:
            artifact["native_transport"]["native_response_token_ids"] = False
    reseal(artifact)
    write(path, artifact)
    key = "collector_qualification" if defect == "collector_policy" else "direct_route_certificate"
    request["runtime"][key] = bound(path, artifact)
    reseal(request)
    with pytest.raises(v3.CollectionError, match="qualification|certificate"):
        v3.validate_request(request, relative_to=REQUEST.parent)


def test_synthetic_artifacts_close_only_the_two_external_runtime_slots(tmp_path):
    request = qualified_request(tmp_path)
    validated = v3.validate_request(request, relative_to=REQUEST.parent, require_ready=True)
    assert validated["blockers"] == []
    assert validated["collector_image"].endswith("@sha256:" + "a" * 64)
    assert request["execution"]["launchable"] is False
    assert not hasattr(v3, "launch")


def test_v2_source_projection_and_attempt_preparation_remain_covered(monkeypatch, tmp_path):
    request = qualified_request(tmp_path)
    current = v3.validate_request(request, relative_to=REQUEST.parent, require_ready=True)
    validated = {
        **current,
        "request": current["predecessor"],
        "source_closure": v2_source_closure(),
    }
    attempt = validated["roster"][0]
    expectation = v2.source_expectation(validated, attempt["attempt_id"])
    v2._validate_expectation(expectation)
    assert expectation["interface"] == v2.PARITY_INTERFACE
    assert expectation["runtime_artifacts"] == {
        "collector_qualification_sha256": current["collector_qualification"]["sha256"],
        "collector_image": current["collector_image"],
        "direct_route_certificate_sha256": current["route_certificate"]["sha256"],
    }

    row = validated["inventory"][(attempt["task_key"], attempt["task_version_id"])]
    prompt_sha = fleet.sha256(PRIVATE.encode())
    task_binding = {
        "key": attempt["task_key"],
        "version_id": attempt["task_version_id"],
        "prompt_sha256": prompt_sha,
        "env_variables_sha256": "sha256:" + "1" * 64,
        "output_json_schema_sha256": "sha256:" + "2" * 64,
        "cyber_contract": rl_data.AUTHORITY["required_cyber_contract"],
    }
    environment = {
        **{key: v1._normalized_sha(value) for key, value in row["environment"].items()},
        "ttl_seconds": validated["base_request"]["limits"]["ttl_seconds"],
    }
    verifier = {
        **{key: v1._normalized_sha(value) for key, value in row["verifier"].items()},
        "function_name": "verify",
    }
    monkeypatch.setattr(
        v2.fleet,
        "bind_task",
        lambda task, selected: (task_binding, environment, verifier),
    )
    monkeypatch.setattr(
        v2,
        "prepare_private_request",
        lambda **kwargs: {
            "tools": [],
            "rendered": "synthetic",
            "tokens": [1],
            "rendered_sha256": fleet.sha256(b"synthetic"),
            "tokens_sha256": digest_json([1]),
        },
    )
    prepared = v2.prepare_attempt(
        validated,
        attempt["attempt_id"],
        task={"prompt": PRIVATE},
        tokenizer=NS(chat_template="unused"),
    )
    assert prepared["request_messages"] is None
    assert prepared["config"]["task"]["prompt_sha256"] == prompt_sha
    assert prepared["config"]["initial_prompt_sha256"] == fleet.sha256(b"synthetic")
    assert prepared["config"]["initial_prompt_tokens_sha256"] == digest_json([1])


class Tokenizer:
    chat_template = "synthetic-template"
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": PRIVATE}]
        assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
            "bash",
            "submit_report",
        ]
        assert kwargs["add_generation_prompt"] is True
        assert not ({"enable_thinking", "reasoning_effort", "preserve_thinking"} & set(kwargs))
        if kwargs["tokenize"]:
            assert kwargs["return_dict"] is False
            return [1, 2, 3]
        assert "return_dict" not in kwargs
        return "synthetic-rendered-prefix"

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        if text == "\n":
            return [10]
        assert text == "synthetic-rendered-prefix"
        return [1, 2, 3]


class Engine:
    model_name = "/synthetic-model"

    def __init__(self):
        self.turn = 0

    async def generate(self, request):
        assert set(request) == {"prompt_token_ids", "sampling_params"}
        assert request["prompt_token_ids"]
        assert request["sampling_params"]["max_tokens"] <= 8
        replies = [
            (
                '<tool_call>{"name":"bash","arguments":{"script":"synthetic"}}</tool_call>',
                [20, 9],
            ),
            ('<tool_call>{"name":"submit_report","arguments":{}}</tool_call>', [30, 9]),
        ]
        text, tokens = replies[self.turn]
        self.turn += 1
        return {
            "responses": [text],
            "response_ids": [tokens],
            "response_logprobs": [[-0.1] * len(tokens)],
            "stop_reasons": ["stop"],
        }


class ReviewTokenizer(Tokenizer):
    def apply_chat_template(self, messages, **kwargs):
        assert [message["role"] for message in messages] == ["user"]
        assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
            "bash",
            "submit_report",
        ]
        assert kwargs["add_generation_prompt"] is True
        if kwargs["tokenize"]:
            assert kwargs["return_dict"] is False
            return [1, 2, 3]
        assert "return_dict" not in kwargs
        return "synthetic-rendered-prefix"


def test_private_builder_is_user_only_and_binds_rendered_bytes_and_tokens():
    catalog = json.loads(CATALOG.read_text())
    prepared = v2.prepare_private_request(
        task_prompt=PRIVATE,
        tool_catalog=catalog,
        tokenizer=Tokenizer(),
        task_prompt_sha256=fleet.sha256(PRIVATE.encode()),
        runtime_chat_template_sha256=fleet.sha256(Tokenizer.chat_template.encode()),
    )
    assert prepared["messages"] == [{"role": "user", "content": PRIVATE}]
    assert prepared["rendered_sha256"] == fleet.sha256(b"synthetic-rendered-prefix")
    assert prepared["tokens_sha256"] == digest_json([1, 2, 3])
    assert prepared["task_prompt_sha256"] == fleet.sha256(PRIVATE.encode())
    assert prepared["prompt_policy_sha256"] == v2.PROMPT_POLICY_SHA256
    assert prepared["template_invocations_sha256"] == v2.TEMPLATE_INVOCATIONS_SHA256
    assert prepared["rendered_request_contract_sha256"] == v2.RENDERED_REQUEST_CONTRACT_SHA256

    with pytest.raises(v2.CollectionError, match="binding"):
        v2.prepare_private_request(
            task_prompt=PRIVATE,
            tool_catalog=catalog,
            tokenizer=Tokenizer(),
            task_prompt_sha256="sha256:" + "0" * 64,
            runtime_chat_template_sha256=fleet.sha256(Tokenizer.chat_template.encode()),
        )


@pytest.mark.asyncio
async def test_user_only_native_recorder_matches_checked_dense_parity(monkeypatch, tmp_path):
    helpers = NS(
        __globals__={
            "get_generation_prompt_ids": lambda tokenizer: [12],
            "encode_messages_subset": lambda messages, tokenizer: [11],
        }
    )
    monkeypatch.setattr(skyrl_episode, "native_helper", lambda path: helpers)
    tokenizer, engine = Tokenizer(), Engine()
    catalog = json.loads(CATALOG.read_text())
    prepared = v2.prepare_private_request(
        task_prompt=PRIVATE,
        tool_catalog=catalog,
        tokenizer=tokenizer,
        task_prompt_sha256=fleet.sha256(PRIVATE.encode()),
        runtime_chat_template_sha256=fleet.sha256(tokenizer.chat_template.encode()),
    )
    config = {
        "model": {
            "repo": v2.MODEL_REPO,
            "root": engine.model_name,
            "runtime_chat_template_sha256": fleet.sha256(tokenizer.chat_template.encode()),
        },
        "rl": {"context_tokens": 32, "max_tokens_per_turn": 8},
        "initial_prompt_tokens_sha256": prepared["tokens_sha256"],
    }
    recorder = skyrl_episode.Recorder(
        config,
        tokenizer,
        engine,
        {"temperature": 1.0},
        16,
        tmp_path / "synthetic-helper.py",
    )

    class Session:
        async def call_tool(self, name, arguments):
            return NS(content=[NS(type="text", text=PRIVATE)], is_error=False)

    from training.rl_episode import _agent

    messages, reason, elapsed = await _agent(
        recorder,
        Session(),
        prepared["messages"],
        prepared["tools"],
        {"max_turns": 4, "tool_seconds": 2, "tool_result_chars": 10000},
        skyrl_episode.parse,
    )
    assert reason == "report_submitted"
    assert messages[0]["role"] == "user"
    assert all(message["role"] != "system" for message in messages)
    sample = recorder.finalize(1.0, {"synthetic": True}, elapsed)[0]
    tokens = [1, 2, 3, 20, 9, 10, 11, 12, 30, 9, 10, 11, 12]
    mask = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    receipt = v3.recorder_dense_parity(
        sample,
        {"messages": messages},
        dense_reference_tokens=tokens,
        dense_reference_loss_mask=mask,
        model={
            "repo": v2.MODEL_REPO,
            "revision": v2.MODEL_REVISION,
            "runtime_chat_template_sha256": v2.CHAT_TEMPLATE_SHA256,
        },
        interface=v2.PARITY_INTERFACE,
        source_closure=source_closure(),
        max_length=16384,
        context_tokens=4096,
    )
    assert receipt == json.loads(PARITY.read_text())
    assert PRIVATE not in json.dumps(receipt)
    assert receipt["interface"]["explicit_system_message"] == "absent"


def test_private_reviewer_accepts_user_only_and_emits_only_bound_digests(monkeypatch, tmp_path):
    old, episode, index_path, config, _tokenizer, catalog = v1_source_fixture(tmp_path, monkeypatch)
    conversation = json.loads((episode / "conversation.json").read_text())
    assert [row["role"] for row in conversation["messages"][:2]] == ["system", "user"]
    conversation["messages"].pop(0)
    write(episode / "conversation.json", conversation)
    _rebind_acceptance(episode, index_path, config)
    expectation = {
        "schema": v2.EXPECTATION_SCHEMA,
        "collection_request_sha256": old["collection_request_sha256"],
        "producer_plan_sha256": old["producer_plan_sha256"],
        "attempt": old["attempt"],
        "model": old["model"],
        "interface": copy.deepcopy(v2.PARITY_INTERFACE),
        "runtime_artifacts": {
            "collector_qualification_sha256": "sha256:" + "a" * 64,
            "collector_image": "registry.example/collector@sha256:" + "b" * 64,
            "direct_route_certificate_sha256": "sha256:" + "c" * 64,
        },
        "runtime_binding": old["runtime_binding"],
        "limits": old["limits"],
        "sampling": old["sampling"],
        "source_closure": v2_source_closure(),
        "qualification": old["qualification"],
        "required_private_digests": copy.deepcopy(v2.RENDERED_REQUEST_CONTRACT),
    }
    reseal(expectation)
    receipt = v2.review_source(
        expectation,
        episode,
        tokenizer=ReviewTokenizer(),
        tool_catalog=catalog,
    )
    assert receipt["schema"] == v2.SOURCE_RECEIPT_SCHEMA
    assert receipt["request"]["roles"] == ["user"]
    assert receipt["request"]["explicit_system_message"] == "absent"
    assert (
        receipt["request"]["raw_task_prompt_sha256"]
        == json.loads((episode / "binding.json").read_text())["task"]["prompt_sha256"]
    )
    assert receipt["runtime_artifacts"] == expectation["runtime_artifacts"]
    rendered = json.dumps(receipt)
    assert PRIVATE not in rendered
    assert "private task" not in rendered

    for defect in ("producer_plan", "runtime_binding"):
        broken = copy.deepcopy(expectation)
        if defect == "producer_plan":
            broken["producer_plan_sha256"] = "not-a-digest"
        else:
            broken["runtime_binding"] = {
                "environment": {},
                "verifier": {},
                "current_binding_sha256": "not-a-digest",
            }
        reseal(broken)
        with pytest.raises(v2.CollectionError, match="expectation"):
            v2.review_source(
                broken,
                episode,
                tokenizer=ReviewTokenizer(),
                tool_catalog=catalog,
            )

    conversation["messages"].insert(0, {"role": "system", "content": "synthetic"})
    write(episode / "conversation.json", conversation)
    _rebind_acceptance(episode, index_path, config)
    with pytest.raises(v2.CollectionError, match="differs"):
        v2.review_source(
            expectation,
            episode,
            tokenizer=ReviewTokenizer(),
            tool_catalog=catalog,
        )
