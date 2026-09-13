"""Synthetic-only checks for the blocked direct self-trace collection rail."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_self_trace_corpus import (  # synthetic fixture only
    SYSTEM_PROMPT,
    _rebind_acceptance,
    write,
)
from test_self_trace_corpus import fixture as corpus_fixture

from evals.fleet import opencode_self_hosted as fleet
from training import dense, rl_episode, skyrl_episode
from training import self_trace_collection as collection
from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
REQUEST = ROOT / "configs/qualification/qwen38-self-trace-collection-request-v1.json"
PARITY = ROOT / "docs/evidence/qwen38-study/2026-09-12-self-trace-recorder-dense-parity-v1.json"
PRIVATE = "SYNTHETIC_PRIVATE_REQUEST_CONTENT_MUST_NOT_ESCAPE"


def read_request() -> dict:
    return json.loads(REQUEST.read_text(encoding="utf-8"))


def reseal(value: dict) -> dict:
    value["sha256"] = digest_json({key: item for key, item in value.items() if key != "sha256"})
    return value


def source_closure() -> dict[str, str]:
    return {name: file_sha256(path) for name, path in collection.SOURCE_CLOSURE_PATHS.items()}


def test_checked_v1_request_is_immutable_and_fails_closed_after_source_change():
    request = read_request()
    assert request["sha256"] == digest_json(
        {key: item for key, item in request.items() if key != "sha256"}
    )
    assert request["source_closure"]["rl_episode.py"]["sha256"] != file_sha256(
        collection.SOURCE_CLOSURE_PATHS["rl_episode.py"]
    )
    assert request["execution"] == {
        "kind": "offline_qualification_not_a_launcher",
        "cluster_or_api_mutations_performed": False,
        "job_submission_performed": False,
        "credentials_required": False,
    }
    assert "PENDING" not in REQUEST.read_text(encoding="utf-8")
    with pytest.raises(collection.CollectionError, match="source file digest"):
        collection.validate_request(request, relative_to=REQUEST.parent)


def test_frozen_v1_preflight_cli_rejects_without_private_output(capsys):
    assert collection.main(["--request", str(REQUEST)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "status": "rejected",
        "reason": "collection_request_not_qualified",
    }
    rendered = json.dumps(result)
    assert PRIVATE not in rendered


@pytest.mark.parametrize(
    "defect",
    ["tool_order", "roster", "model", "status", "source", "sampling", "prefix_policy"],
)
def test_request_identity_or_policy_drift_fails_closed(defect):
    request = read_request()
    if defect == "tool_order":
        request["interface"]["required_task_tools"].reverse()
    elif defect == "roster":
        request["selection"]["attempt_roster_sha256"] = "sha256:" + "0" * 64
    elif defect == "model":
        request["model"]["revision"] = "0" * 40
    elif defect == "status":
        request["status"] = "qualified"
    elif defect == "source":
        request["source_closure"]["rl_episode.py"]["sha256"] = "sha256:" + "0" * 64
    elif defect == "sampling":
        request["sampling"]["temperature"] = float("nan")
    else:
        request["interface"]["request_prefix_policy"]["tool_rewriting"] = "allowed"
    reseal(request)
    with pytest.raises(collection.CollectionError):
        collection.validate_request(request, relative_to=REQUEST.parent)


@pytest.mark.parametrize("defect", ["runtime_scope", "dense_policy", "token_hash"])
def test_parity_receipt_cannot_overclaim_or_break_cross_field_bindings(defect):
    parity = json.loads(PARITY.read_text(encoding="utf-8"))
    if defect == "runtime_scope":
        parity["qualification_scope"]["target_model_weights_loaded"] = True
    elif defect == "dense_policy":
        parity["dense_policy"]["max_length"] = 32
    else:
        parity["dense"]["reference_tokens_sha256"] = "sha256:" + "f" * 64
    reseal(parity)
    request = read_request()
    with pytest.raises(collection.CollectionError, match="parity receipt fields differ"):
        collection._validate_parity_receipt(
            parity,
            model=request["model"],
            interface=request["interface"],
            source_closure={
                name: reference["sha256"] for name, reference in request["source_closure"].items()
            },
            qualification=request["qualification"],
        )


class Tokenizer:
    chat_template = "synthetic-template"
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert [message["role"] for message in messages] == ["system", "user"]
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
        assert request["sampling_params"]["max_tokens"] <= 8
        replies = [
            ('<tool_call>{"name":"bash","arguments":{"script":"synthetic"}}</tool_call>', [20, 9]),
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


def test_private_request_builder_rejects_rewritten_or_unbound_interface():
    catalog = json.loads(
        (ROOT / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json").read_text()
    )
    with pytest.raises(collection.CollectionError, match="binding"):
        collection.prepare_private_request(
            system_prompt=PRIVATE,
            task_prompt=PRIVATE,
            tool_catalog=list(reversed(catalog)),
            tokenizer=Tokenizer(),
            system_prompt_sha256=fleet.sha256(PRIVATE.encode()),
            task_prompt_sha256=fleet.sha256(PRIVATE.encode()),
            runtime_chat_template_sha256=fleet.sha256(Tokenizer.chat_template.encode()),
        )
    with pytest.raises(collection.CollectionError, match="binding"):
        collection.prepare_private_request(
            system_prompt=PRIVATE,
            task_prompt=PRIVATE,
            tool_catalog=catalog,
            tokenizer=Tokenizer(),
            system_prompt_sha256="sha256:" + "0" * 64,
            task_prompt_sha256=fleet.sha256(PRIVATE.encode()),
            runtime_chat_template_sha256=fleet.sha256(Tokenizer.chat_template.encode()),
        )


def test_episode_collector_preserves_exact_system_user_prefix_by_copy():
    supplied = [
        {"role": "system", "content": PRIVATE},
        {"role": "user", "content": "synthetic task"},
    ]
    copied = rl_episode._direct_request_messages("synthetic task", supplied)
    assert copied == supplied and copied is not supplied
    copied[0]["content"] = "changed copy"
    assert supplied[0]["content"] == PRIVATE
    assert rl_episode._direct_request_messages("synthetic task", None) == [
        {"role": "user", "content": "synthetic task"}
    ]
    for invalid in (
        supplied[:1],
        list(reversed(supplied)),
        [supplied[0], {"role": "user", "content": "another task"}],
        [supplied[0], {**supplied[1], "extra": True}],
    ):
        with pytest.raises(rl_episode.InvalidEpisode, match="direct_request_prefix_invalid"):
            rl_episode._direct_request_messages("synthetic task", invalid)


@pytest.mark.asyncio
async def test_synthetic_native_recorder_and_dense_adapter_match_across_two_direct_turns(
    monkeypatch, tmp_path
):
    helpers = NS(
        __globals__={
            "get_generation_prompt_ids": lambda tokenizer: [12],
            "encode_messages_subset": lambda messages, tokenizer: [11],
        }
    )
    monkeypatch.setattr(skyrl_episode, "native_helper", lambda path: helpers)
    tokenizer, engine = Tokenizer(), Engine()
    tool_catalog = json.loads(
        (ROOT / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json").read_text()
    )
    prepared = collection.prepare_private_request(
        system_prompt=PRIVATE,
        task_prompt=PRIVATE,
        tool_catalog=tool_catalog,
        tokenizer=tokenizer,
        system_prompt_sha256=fleet.sha256(PRIVATE.encode()),
        task_prompt_sha256=fleet.sha256(PRIVATE.encode()),
        runtime_chat_template_sha256=fleet.sha256(tokenizer.chat_template.encode()),
    )
    messages, tools = prepared["messages"], prepared["tools"]
    config = {
        "model": {
            "repo": collection.MODEL_REPO,
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
            return NS(
                content=[NS(type="text", text=PRIVATE)],
                is_error=False,
            )

    from training.rl_episode import _agent

    messages, reason, elapsed = await _agent(
        recorder,
        Session(),
        messages,
        tools,
        {"max_turns": 4, "tool_seconds": 2, "tool_result_chars": 10000},
        skyrl_episode.parse,
    )
    assert reason == "report_submitted"
    sample = recorder.finalize(1.0, {"synthetic": True}, elapsed)[0]
    dense_reference_tokens = [1, 2, 3, 20, 9, 10, 11, 12, 30, 9, 10, 11, 12]
    dense_reference_loss_mask = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
    assert sample.tokens == dense_reference_tokens
    assert [0, 0, 0, *sample.loss_mask] == dense_reference_loss_mask
    receipt = collection.recorder_dense_parity(
        sample,
        {"messages": messages},
        dense_reference_tokens=dense_reference_tokens,
        dense_reference_loss_mask=dense_reference_loss_mask,
        model={
            "repo": collection.MODEL_REPO,
            "revision": collection.MODEL_REVISION,
            "runtime_chat_template_sha256": collection.CHAT_TEMPLATE_SHA256,
        },
        interface={
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": collection.TOOL_CATALOG_SHA256,
            "compaction": "disabled",
        },
        source_closure=source_closure(),
        max_length=16384,
        context_tokens=4096,
    )
    assert receipt["status"] == "qualified"
    assert receipt["first_observation"] == {
        "tokens": 3,
        "native_dense_token_ids_equal": True,
        "native_dense_loss_masks_equal": True,
        "all_observation_tokens_masked": True,
    }
    assert receipt["native"]["assistant_targets"] == 2
    assert (
        receipt["native"]["recorded_tokens_sha256"] == receipt["dense"]["reference_tokens_sha256"]
    )
    assert (
        receipt["native"]["recorded_loss_mask_sha256"]
        == receipt["dense"]["reference_loss_mask_sha256"]
    )
    assert receipt["dense_policy"] == {
        "format": dense.FORMAT,
        "max_length": 16384,
        "context_tokens": 4096,
    }
    assert receipt["qualification_scope"] == {
        "kind": "skyrl_recorder_adapter_to_dense_synthetic_fixture",
        "proves": "recorded_token_and_loss_mask_preservation_across_two_direct_turns",
        "target_model_weights_loaded": False,
        "target_runtime_chat_template_loaded": False,
        "pinned_native_helper_loaded": False,
        "collector_image_or_base_route_used": False,
    }
    assert receipt["sha256"] == digest_json(
        {key: item for key, item in receipt.items() if key != "sha256"}
    )
    frozen = json.loads(PARITY.read_text(encoding="utf-8"))
    assert receipt["native"] == frozen["native"]
    assert receipt["dense"] == frozen["dense"]
    assert receipt["source_closure_sha256"] != frozen["source_closure_sha256"]
    rendered = json.dumps(receipt)
    assert PRIVATE not in rendered
    assert not ({"prompt", "trace", "reward", "score", "flag", "credential"} & set(receipt))

    broken = copy.deepcopy(sample)
    broken.loss_mask[2] = 1
    with pytest.raises(collection.CollectionError, match="align|parity"):
        collection.recorder_dense_parity(
            broken,
            {"messages": messages},
            dense_reference_tokens=dense_reference_tokens,
            dense_reference_loss_mask=dense_reference_loss_mask,
            model=receipt["model"],
            interface=receipt["interface"],
            source_closure=source_closure(),
            max_length=16384,
            context_tokens=4096,
        )

    changed_second_target = dense_reference_tokens.copy()
    changed_second_target[-1] += 1
    with pytest.raises(collection.CollectionError, match="parity"):
        collection.recorder_dense_parity(
            sample,
            {"messages": messages},
            dense_reference_tokens=changed_second_target,
            dense_reference_loss_mask=dense_reference_loss_mask,
            model=receipt["model"],
            interface=receipt["interface"],
            source_closure=source_closure(),
            max_length=16384,
            context_tokens=4096,
        )


def test_parity_requires_explicit_system_then_user_anchor():
    sample = {
        "tokens": [1, 2, 3, 4, 5, 6, 7],
        "response_length": 5,
        "loss_mask": [1, 0, 0, 1, 0],
    }
    conversation = {
        "messages": [
            {"role": "user", "content": PRIVATE},
            {"role": "assistant", "content": PRIVATE},
            {"role": "tool", "name": "bash", "content": PRIVATE},
            {"role": "assistant", "content": PRIVATE},
            {"role": "tool", "name": "submit_report", "content": PRIVATE},
        ]
    }
    with pytest.raises(collection.CollectionError, match="align|parity"):
        collection.recorder_dense_parity(
            sample,
            conversation,
            dense_reference_tokens=[1, 2, 3, 4, 5, 6, 7],
            dense_reference_loss_mask=[0, 0, 1, 0, 0, 1, 0],
            model={
                "repo": collection.MODEL_REPO,
                "revision": collection.MODEL_REVISION,
                "runtime_chat_template_sha256": collection.CHAT_TEMPLATE_SHA256,
            },
            interface={
                "required_task_tools": ["bash", "submit_report"],
                "required_task_tool_catalog_sha256": collection.TOOL_CATALOG_SHA256,
                "compaction": "disabled",
            },
            source_closure=source_closure(),
            max_length=16,
            context_tokens=4,
        )


def source_fixture(tmp_path, monkeypatch):
    _config, index_path, old_episode = corpus_fixture(tmp_path, monkeypatch)
    attempt_id = "q38-self-0123456789abcdefabcd-a1"
    episode = old_episode.rename(old_episode.with_name(attempt_id))
    binding = json.loads((episode / "binding.json").read_text())
    limits = {
        "context_tokens": 32,
        "response_tokens": 16,
        "max_tokens_per_turn": 8,
        "max_turns": 4,
        "episode_seconds": 300,
        "tool_seconds": 60,
        "tool_result_chars": 10000,
        "ttl_seconds": 3600,
    }
    binding["run_id"] = attempt_id
    binding["model"] = {
        "repo": collection.MODEL_REPO,
        "revision": collection.MODEL_REVISION,
        "root": "/synthetic-model",
        "runtime_chat_template_sha256": collection.CHAT_TEMPLATE_SHA256,
    }
    binding["execution"]["required_task_tool_catalog_sha256"] = collection.TOOL_CATALOG_SHA256
    binding["rl"] = {
        key: value for key, value in limits.items() if key not in {"response_tokens", "ttl_seconds"}
    }
    binding["environment"]["ttl_seconds"] = limits["ttl_seconds"]
    binding["native_batch"] = {
        "kind": "self_trace_recollection",
        "attempt_id": attempt_id,
        "attempt_ordinal": 0,
        "collection_request_sha256": "sha256:" + "7" * 64,
    }
    binding["sampling"] = {
        "max_generate_length": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": 0,
    }
    binding["initial_prompt_sha256"] = fleet.sha256(b"synthetic-rendered-prefix")
    binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
    write(episode / "binding.json", binding)
    write(episode / "create-intent.json", {"run_id": attempt_id})
    instance = json.loads((episode / "instance.json").read_text())
    write(
        episode / "score-intent.json",
        fleet.build_scoring_payload(
            binding,
            instance_id=instance["instance_id"],
            final_answer="",
            messages=[],
        ),
    )
    _rebind_acceptance(
        episode,
        index_path,
        _config,
        config_sha=binding["config_sha256"],
    )
    expectation = {
        "schema": collection.EXPECTATION_SCHEMA,
        "collection_request_sha256": binding["native_batch"]["collection_request_sha256"],
        "producer_plan_sha256": "sha256:" + "6" * 64,
        "attempt": {
            "attempt_id": attempt_id,
            "attempt_ordinal": 0,
            "task_key": binding["task"]["key"],
            "task_version_id": binding["task"]["version_id"],
        },
        "model": {
            **binding["model"],
            "lock_file_sha256": "sha256:" + "5" * 64,
            "initialization": "exact_fresh_base",
        },
        "interface": {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": collection.TOOL_CATALOG_SHA256,
            "runtime_chat_template_sha256": collection.CHAT_TEMPLATE_SHA256,
            "system_prompt_sha256": fleet.sha256(SYSTEM_PROMPT.encode()),
            "compaction": "disabled",
        },
        "runtime_binding": {
            "environment": {
                key: value for key, value in binding["environment"].items() if key != "ttl_seconds"
            },
            "verifier": {
                key: value for key, value in binding["verifier"].items() if key != "function_name"
            },
            "current_binding_sha256": "sha256:" + "4" * 64,
        },
        "limits": limits,
        "sampling": binding["sampling"],
        "source_closure": source_closure(),
        "qualification": {
            "max_length": 32,
            "context_tokens": 8,
            "synthetic_fixture_policy": (
                "synthetic_content_only_no_task_prompt_trace_flag_score_or_credential"
            ),
        },
    }
    expectation["sha256"] = digest_json(expectation)
    original_sha256 = fleet.sha256

    def synthetic_template_identity(value):
        if value == Tokenizer.chat_template.encode():
            return collection.CHAT_TEMPLATE_SHA256
        return original_sha256(value)

    monkeypatch.setattr(fleet, "sha256", synthetic_template_identity)
    tool_catalog = json.loads(
        (ROOT / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json").read_text()
    )
    return expectation, episode, index_path, _config, Tokenizer(), tool_catalog


def test_positive_private_episode_becomes_content_free_fail_closed_source_receipt(
    tmp_path, monkeypatch
):
    expectation, episode, _index_path, _config, tokenizer, catalog = source_fixture(
        tmp_path, monkeypatch
    )
    receipt = collection.review_source(
        expectation, episode, tokenizer=tokenizer, tool_catalog=catalog
    )
    assert receipt["schema"] == collection.SOURCE_RECEIPT_SCHEMA
    assert receipt["status"] == "qualified"
    assert receipt["outcome"] == {
        "authoritative_positive": True,
        "verifier_execution_bound": True,
        "environment_release_confirmed": True,
        "possible_environment_leak": False,
    }
    assert receipt["targets"]["assistant_responses"] == 2
    assert receipt["targets"]["contains_retained_bash_response"] is True
    assert receipt["sha256"] == digest_json(
        {key: value for key, value in receipt.items() if key != "sha256"}
    )
    rendered = json.dumps(receipt)
    assert SYSTEM_PROMPT not in rendered
    assert PRIVATE not in rendered
    assert "private task" not in rendered
    assert "input_ids" not in rendered


@pytest.mark.parametrize("defect", ["extra", "mode", "system", "sampling", "batch", "score_intent"])
def test_private_source_uncertainty_or_binding_drift_never_gets_a_receipt(
    tmp_path, monkeypatch, defect
):
    expectation, episode, index_path, config, tokenizer, catalog = source_fixture(
        tmp_path, monkeypatch
    )
    if defect == "extra":
        write(episode / "unbound.json", {"synthetic": True})
    elif defect == "mode":
        (episode / "conversation.json").chmod(0o644)
    elif defect == "system":
        conversation = json.loads((episode / "conversation.json").read_text())
        conversation["messages"][0]["content"] = "different synthetic system"
        write(episode / "conversation.json", conversation)
        _rebind_acceptance(episode, index_path, config)
    elif defect in {"sampling", "batch"}:
        binding = json.loads((episode / "binding.json").read_text())
        if defect == "sampling":
            binding["sampling"]["temperature"] = 0.0
        else:
            binding["native_batch"].pop("collection_request_sha256")
        binding["config_sha256"] = fleet.digest_without(binding, "config_sha256")
        write(episode / "binding.json", binding)
        _rebind_acceptance(
            episode,
            index_path,
            config,
            config_sha=binding["config_sha256"],
        )
    else:
        write(episode / "score-intent.json", {"instance_id": "different-synthetic-instance"})
    with pytest.raises(collection.CollectionError):
        collection.review_source(expectation, episode, tokenizer=tokenizer, tool_catalog=catalog)
