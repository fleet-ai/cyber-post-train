from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
import yaml

from evals.fleet import fixed_proxy, self_hosted

CONFIG_PATH = Path("evals/fleet/configs/qwen36-27b-qwen-code-selfhosted-smoke-v1.json")
OPENCODE_CONFIG_PATH = Path("evals/fleet/configs/qwen38-opencode-train-sweep-smoke-v2.json")
PARTIAL_OBSERVER_MANIFEST = Path(
    "evals/fleet/cluster/opencode-partial-session-observers-v1.yaml"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(self_hosted.canonical_json(value) + b"\n")


def test_opencode_settings_preserve_frozen_no_autocontinue_treatment() -> None:
    config = json.loads(
        Path(
            "evals/fleet/configs/qwen38-opencode-hosted-http500-successor49-pass4-v8.json"
        ).read_text()
    )
    settings = self_hosted.opencode_settings(config)
    model = settings["provider"]["fleet-cluster"]["models"]["qwen3.8-27b"]
    assert settings["plugin"] == [
        "file:///home/node/.config/opencode/fleet-disable-compaction-autocontinue.mjs"
    ]
    assert "compaction" not in settings
    assert model["limit"] == {"context": 262144, "output": 32768}
    assert "compaction_headroom_tokens" not in config["harness"]
    assert self_hosted.OPENCODE_NO_AUTOCONTINUE_PLUGIN == (
        "export const DisableCompactionAutocontinue = async () => ({\n"
        '  "experimental.compaction.autocontinue": async (_input, output) => '
        "{ output.enabled = false; },\n"
        "});\n"
    )
    assert self_hosted.sha256(self_hosted.canonical_json(settings)) == (
        "sha256:fa7464a2a043b1e02febbabc70b1fce4278ec31a4d0968e4318677cae1f36e24"
    )
    assert self_hosted.sha256(self_hosted.OPENCODE_NO_AUTOCONTINUE_PLUGIN.encode()) == (
        "sha256:3542f8fe30d270bec6ee8e832081da8169fd78b667647ecd119425b1961a7a28"
    )


def test_opencode_settings_render_new_autocontinue_policy_separately() -> None:
    config = json.loads(OPENCODE_CONFIG_PATH.read_text())
    config["harness"].update(
        {
            "context_management": self_hosted.OPENCODE_CONTEXT_MANAGEMENT,
            "compaction_headroom_tokens": 8192,
        }
    )
    settings = self_hosted.opencode_settings(config)
    model = settings["provider"]["fleet-cluster"]["models"][config["model"]["served_id"]]
    assert settings["compaction"] == {"auto": True, "reserved": 8192}
    assert "plugin" not in settings
    assert model["limit"] == {
        "context": config["harness"]["context_window_size"],
        "input": (
            config["harness"]["context_window_size"]
            - config["harness"]["max_output_tokens"]
        ),
        "output": config["harness"]["max_output_tokens"],
    }


@pytest.mark.parametrize(
    "policy,headroom",
    [
        ("unknown", None),
        (self_hosted.OPENCODE_CONTEXT_MANAGEMENT, None),
        (self_hosted.OPENCODE_NO_AUTOCONTINUE_CONTEXT_MANAGEMENT, 8192),
    ],
)
def test_opencode_settings_fail_closed_on_context_policy_drift(
    policy: str, headroom: int | None
) -> None:
    config = json.loads(OPENCODE_CONFIG_PATH.read_text())
    config["harness"]["context_management"] = policy
    if headroom is None:
        config["harness"].pop("compaction_headroom_tokens", None)
    else:
        config["harness"]["compaction_headroom_tokens"] = headroom
    with pytest.raises(ValueError, match="context policy|headroom"):
        self_hosted.opencode_settings(config)


def test_partial_session_observers_are_read_only_create_once_high_priority() -> None:
    documents = list(yaml.safe_load_all(PARTIAL_OBSERVER_MANIFEST.read_text()))
    assert len(documents) == 2
    expected = {
        "chris-cyber-q38-source10-a3-partial-observer-v1": (
            "qwen38-v8-source10-attempt3-partial-resume.json",
            "sha256:833af87ee84a576145240c26f305d70b0faa7c605c8d32015fdc928d77d43dcb",
        ),
        "chris-cyber-glm53-a-source14-a1-partial-observer-v1": (
            "glm53-a-v5-source14-attempt1-partial-resume.json",
            "sha256:544485eadeae80770947b58657df50d34e506541d97c81db159167dd9ba9ec2a",
        ),
    }
    for document in documents:
        name = document["metadata"]["name"]
        config_name, config_digest = expected[name]
        pod = document["spec"]["template"]["spec"]
        command = pod["containers"][0]["args"][0]
        assert document["spec"]["backoffLimit"] == 0
        assert pod["priorityClassName"] == "fleet-train-high"
        assert "observe-partial-session" in command
        assert "resume-partial-session" not in command
        assert "opencode run" not in command
        assert "docker" not in command
        config = json.loads(Path(f"evals/fleet/configs/{config_name}").read_text())
        assert config["config_sha256"] == config_digest
        assert config["config_sha256"] == self_hosted.digest_without(
            config, "config_sha256"
        )

    successor_manifests = {
        "evals/fleet/cluster/opencode-glm53-a-source14-a1-partial-observer-v2.yaml": (
            "observe-partial-session",
            "chris-cyber-glm53-a-source14-a1-partial-observer-v2",
        ),
        "evals/fleet/cluster/opencode-qwen38-source10-a3-partial-diagnostic-v2.yaml": (
            "diagnose-partial-session",
            "chris-cyber-q38-source10-a3-partial-diagnostic-v2",
        ),
    }
    for path, (command_name, expected_name) in successor_manifests.items():
        document = yaml.safe_load(Path(path).read_text())
        pod = document["spec"]["template"]["spec"]
        command = pod["containers"][0]["args"][0]
        assert document["metadata"]["name"] == expected_name
        assert document["spec"]["backoffLimit"] == 0
        assert pod["priorityClassName"] == "fleet-train-high"
        assert command_name in command
        assert "resume-partial-session" not in command
        assert "opencode run" not in command
        assert "docker" not in command

    projection_documents = list(
        yaml.safe_load_all(
            Path(
                "evals/fleet/cluster/opencode-partial-session-timestamp-projection-v3.yaml"
            ).read_text()
        )
    )
    assert len(projection_documents) == 2
    for document in projection_documents:
        pod = document["spec"]["template"]["spec"]
        command = pod["containers"][0]["args"][0]
        assert document["spec"]["backoffLimit"] == 0
        assert pod["priorityClassName"] == "fleet-train-high"
        assert "diagnose-timestamp-projection" in command
        assert "resume-partial-session" not in command
        assert "opencode run" not in command
        assert "docker" not in command


def _recovery_source_fixture(tmp_path: Path) -> tuple[dict, Path, str]:
    config = json.loads(OPENCODE_CONFIG_PATH.read_text())
    source = tmp_path / "source"
    trace = source / "agent-output" / "opencode-stream.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps(
            {
                "type": "text",
                "timestamp": 1781623587086,
                "part": {"type": "text", "id": "a1", "text": "done"},
            }
        )
        + "\n"
    )
    instance_id = "rx7ruxhfwwpz"
    evidence_run_id = "22222222-2222-4222-8222-222222222222"
    verifier_execution_id = "33333333-3333-4333-8333-333333333333"
    score = 0.5
    _write_json(
        source / "binding.json",
        {
            **{
                key: config[key]
                for key in ("run_id", "task", "environment", "verifier", "model", "harness")
            },
            "authority": config["authority"],
        },
    )
    _write_json(
        source / "result.json",
        {
            "run_id": config["run_id"],
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "session_id": None,
            "session_ingest_status": "failed",
            "score": score,
            "verifier_execution_id": verifier_execution_id,
            "agent_exit_code": 0,
            "harness": "opencode",
            "agent_termination": "completed",
        },
    )
    _write_json(
        source / "reward-result.json",
        {
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": instance_id,
            "reward": score,
            "verifier_execution_id": verifier_execution_id,
        },
    )
    _write_json(
        source / "runtime-binding.json",
        {
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "tool_names": config["execution"]["required_task_tools"],
            "tool_catalog_sha256": config["execution"]["required_task_tool_catalog_sha256"],
        },
    )
    _write_json(
        source / "session-ingest.json",
        {
            "status": "failed",
            "session_id": None,
            "message_count": 1,
            "chunks_completed": 0,
            "chunk_count": 1,
            "error_type": "RuntimeError",
        },
    )
    _write_json(
        source / "cleanup.json",
        {
            "instance_created": True,
            "instance_closed": True,
            "containers_removed": True,
        },
    )
    intent = {
        "schema_version": "fleet-selfhosted-scoring-intent-v1",
        "run_id": config["run_id"],
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
        "evidence_run_id": evidence_run_id,
        "scoring_payload_mode": None,
        "request_keys": ["conversation", "final_answer", "instance_id"],
        "request_sha256": "sha256:" + "0" * 64,
    }
    intent["scoring_intent_sha256"] = self_hosted.digest_without(
        intent, "scoring_intent_sha256"
    )
    _write_json(source / "scoring-intent.json", intent)
    _write_json(
        source / "trace-manifest.json",
        {
            "canonical_trace": "agent-output/opencode-stream.jsonl",
            "canonical_trace_sha256": self_hosted.sha256(trace.read_bytes()),
            "harness": "opencode",
            "event_count": 1,
            "raw_line_count": 1,
            "malformed_line_count": 0,
            "normalized_message_count": 1,
            "fidelity": "full_opencode_json_normalized_with_tool_calls_and_observations",
        },
    )
    return config, source, verifier_execution_id


def _partial_recovery_source_fixture(tmp_path: Path) -> tuple[dict, Path, str, str]:
    config, source, verifier_execution_id = _recovery_source_fixture(tmp_path)
    session_id = "44444444-4444-4444-8444-444444444444"
    trace = source / "agent-output" / "opencode-stream.jsonl"
    events = [
        {
            "type": "text",
            "timestamp": 1781623587086 + index,
            "part": {"type": "text", "id": f"a{index}", "text": "done"},
        }
        for index in range(65)
    ]
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    result = json.loads((source / "result.json").read_text())
    result["session_id"] = session_id
    result["agent_exit_code"] = 1
    _write_json(source / "result.json", result)
    _write_json(
        source / "session-ingest.json",
        {
            "status": "failed",
            "session_id": session_id,
            "message_count": 65,
            "chunks_completed": 1,
            "chunk_count": 3,
            "error_type": "FleetRequestError",
            "error_code": "fleet_http_error",
            "http_status": 500,
            "method": "POST",
            "route": "/v1/sessions/ingest",
        },
    )
    _write_json(
        source / "trace-manifest.json",
        {
            "canonical_trace": "agent-output/opencode-stream.jsonl",
            "canonical_trace_sha256": self_hosted.sha256(trace.read_bytes()),
            "harness": "opencode",
            "event_count": 65,
            "raw_line_count": 65,
            "malformed_line_count": 0,
            "normalized_message_count": 65,
            "fidelity": "full_opencode_json_normalized_with_tool_calls_and_observations",
        },
    )
    return config, source, verifier_execution_id, session_id


def test_smoke_config_is_one_exact_eval_only_arm() -> None:
    config = _config()
    assert config["run_id"] == "chris-cyber-qwen36-qwencode0223-fleet-smoke-v1-r1"
    assert config["model"]["revision"] == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert config["harness"]["version"] == "0.22.3"
    assert config["harness"]["source_commit"] == "09825973e7d3c3fd07e17909c396aa62f48ce51f"
    assert config["execution"] == {
        "pass_k": 1,
        "max_concurrent": 1,
        "network": "chris-qwen-fleet-smoke-v1-r1",
        "training_data_eligible": False,
    }


def test_request_retries_only_transient_idempotent_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter([503, 503, 200])
    sleeps = []

    class Client:
        def request(self, method: str, url: str, **kwargs):
            code = next(responses)
            return type(
                "Response",
                (),
                {"status_code": code, "json": lambda self: {"ok": True}},
            )()

    monkeypatch.setattr(self_hosted.time, "sleep", sleeps.append)
    assert self_hosted._request(Client(), "GET", "/v1/read") == {"ok": True}
    assert sleeps == [1, 2]


def test_request_does_not_retry_mutating_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class Client:
        def request(self, method: str, url: str, **kwargs):
            nonlocal calls
            calls += 1
            return type("Response", (), {"status_code": 503})()

    monkeypatch.setattr(self_hosted.time, "sleep", lambda _: pytest.fail("must not sleep"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        self_hosted._request(Client(), "POST", "/v1/write")
    assert calls == 1


def test_task_receipt_uses_targeted_version_and_never_source_job_roster() -> None:
    config = _config()
    del config["task"]["id"]
    expected_route = f"/v1/tasks/{config['task']['key']}"
    fixture = {
        "eval_task_version_id": config["task"]["version_id"],
        "key": config["task"]["key"],
        "environment_id": config["environment"]["id"],
        "version": config["environment"]["version"],
        "data_id": config["environment"]["data_id"],
        "data_version": config["environment"]["data_version"],
        "prompt": "",
        "env_variables": {},
        "output_json_schema": None,
        "verifier_id": config["verifier"]["id"],
        "verifier": {
            "verifier_version_id": config["verifier"]["version_id"],
            "version": config["verifier"]["version"],
            "sha256": config["verifier"]["sha256"],
            "code": "",
        },
        "metadata": {
            "runtime_seed_manifest": {
                "content_sha256": config["environment"]["runtime_seed_content_sha256"]
            }
        },
    }
    config["task"]["prompt_sha256"] = self_hosted.sha256(b"")
    config["task"]["env_variables_sha256"] = self_hosted.sha256(self_hosted.canonical_json({}))
    config["task"]["output_json_schema_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(None)
    )
    config["verifier"]["code_sha256"] = self_hosted.sha256(b"")

    class Client:
        def request(self, method: str, url: str, **kwargs):
            assert method == "GET"
            assert url.endswith(expected_route)
            assert kwargs["params"] == {"version_id": config["task"]["version_id"]}
            return type("Response", (), {"status_code": 200, "json": lambda self: fixture})()

    assert "version_id" not in fixture
    assert self_hosted.load_and_verify_task(Client(), config) == fixture


def test_task_identifiers_accept_exact_version_without_legacy_id() -> None:
    config = _config()
    del config["task"]["id"]
    assert self_hosted._validate_task_identifiers(
        {"eval_task_version_id": config["task"]["version_id"]}, config["task"]
    ) == (None, config["task"]["version_id"])


def test_task_identifiers_accept_matching_legacy_id_when_present() -> None:
    config = _config()
    assert self_hosted._validate_task_identifiers(
        {
            "id": config["task"]["id"],
            "eval_task_version_id": config["task"]["version_id"],
        },
        config["task"],
    ) == (config["task"]["id"], config["task"]["version_id"])


@pytest.mark.parametrize(
    "version_id",
    (
        None,
        "",
        "unsafe",
        "00000000-0000-0000-0000-000000000000",
        "11111111-1111-4111-8111-111111111111",
    ),
)
def test_task_identifiers_reject_missing_invalid_zero_or_drifted_version(
    version_id: str | None,
) -> None:
    config = _config()
    del config["task"]["id"]
    with pytest.raises(RuntimeError, match="task version ID"):
        self_hosted._validate_task_identifiers(
            {"eval_task_version_id": version_id}, config["task"]
        )


def test_task_identifiers_reject_configured_legacy_id_absent_live() -> None:
    config = _config()
    with pytest.raises(RuntimeError, match="configured Fleet task ID is absent"):
        self_hosted._validate_task_identifiers(
            {"eval_task_version_id": config["task"]["version_id"]}, config["task"]
        )


def test_task_identifiers_reject_invalid_live_legacy_id() -> None:
    config = _config()
    del config["task"]["id"]
    with pytest.raises(RuntimeError, match="Fleet task ID is not a UUID"):
        self_hosted._validate_task_identifiers(
            {
                "id": "unsafe",
                "eval_task_version_id": config["task"]["version_id"],
            },
            config["task"],
        )


def test_build_instance_payload_preserves_exact_runtime_binding() -> None:
    config = _config()
    task = {
        "id": config["task"]["id"],
        "env_variables": {"VISIBLE": "value"},
        "metadata": {
            "runtime_seed_manifest": {
                "files": [
                    {"target_path": "/task/a", "s3_key": "key/a", "bucket": "bucket"},
                    {"target_path": "/task/b", "s3_key": "key/b", "bucket": "bucket"},
                ]
            }
        },
    }
    payload = self_hosted.build_instance_payload(config, task)
    assert payload["env_key"] == "cysec1-2-current-gen"
    assert payload["env_version"] == "v0.0.3"
    assert payload["data_key"] == "commercial"
    assert payload["data_version"] == "v0.0.9"
    assert payload["task_id"] == "65716472-8e11-4825-b2b7-3d584986f0ff"
    assert payload["env_variables"] == {"VISIBLE": "value"}
    assert len(payload["seed_overlay_files"]) == 2


def test_build_instance_payload_fails_closed_without_runtime_seed() -> None:
    with pytest.raises(RuntimeError, match="no runtime seed overlay"):
        self_hosted.build_instance_payload(_config(), {"metadata": {}})


def test_projected_config_uses_validated_live_task_id() -> None:
    config = _config()
    del config["task"]["id"]
    task = {
        "id": "11111111-1111-4111-8111-111111111111",
        "metadata": {
            "runtime_seed_manifest": {
                "files": [
                    {"target_path": "/task/a", "s3_key": "key/a", "bucket": "bucket"}
                ]
            }
        },
    }
    payload = self_hosted.build_instance_payload(config, task)
    assert payload["task_id"] == task["id"]


def test_projected_config_rejects_invalid_live_task_id() -> None:
    config = _config()
    del config["task"]["id"]
    task = {
        "id": "not-a-uuid",
        "metadata": {
            "runtime_seed_manifest": {
                "files": [
                    {"target_path": "/task/a", "s3_key": "key/a", "bucket": "bucket"}
                ]
            }
        },
    }
    with pytest.raises(RuntimeError, match="task ID is not a UUID"):
        self_hosted.build_instance_payload(config, task)


def test_write_agent_prompt_chowns_root_controller_input_for_node_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = tmp_path / "prompt.txt"
    chowns: list[tuple[Path, int, int]] = []
    monkeypatch.setattr(self_hosted.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        self_hosted.os, "chown", lambda path, uid, gid: chowns.append((path, uid, gid))
    )
    self_hosted.write_agent_prompt(prompt, "sealed task input")
    assert chowns == [(prompt, 1000, 1000)]
    assert prompt.stat().st_mode & 0o777 == 0o400


def test_write_agent_prompt_keeps_nonroot_controller_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompt = tmp_path / "prompt.txt"
    monkeypatch.setattr(self_hosted.os, "geteuid", lambda: 501)
    monkeypatch.setattr(
        self_hosted.os,
        "chown",
        lambda *_args: pytest.fail("non-root controller must not chown prompt"),
    )
    self_hosted.write_agent_prompt(prompt, "sealed task input")
    assert prompt.stat().st_mode & 0o777 == 0o400


def test_extract_final_answer_reads_last_qwen_result(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "not-json\n"
        + json.dumps({"type": "assistant", "message": {"content": "draft"}})
        + "\n"
        + json.dumps({"type": "result", "result": "final report"})
        + "\n"
    )
    assert self_hosted.extract_final_answer(trace) == "final report"


def test_qwen_trace_loader_counts_malformed_lines_without_losing_raw_trace(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "projects" / "workspace" / "chats" / "trace.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        json.dumps({"type": "assistant", "message": {"parts": []}}) + "\n" + "not-json\n"
    )
    events, canonical_trace, malformed = self_hosted.load_qwen_chat_trace(tmp_path)
    assert len(events) == 1
    assert canonical_trace == trace
    assert malformed == 1


def test_authority_paths_are_exact_task_version_routes() -> None:
    config = _config()
    prefix = (
        "/v1/rollout-rewards/"
        "cysec1-2-current-gen_blackbox-9afe9e08da314948b573657e__blackbox_ctf_v1/"
        "versions/dd8dd22e-75c0-4b93-8f8e-ea8a292d92bb"
    )
    assert self_hosted.authoritative_route(config, "provisioning") == prefix + "/instances"
    assert self_hosted.authoritative_route(config, "scoring") == prefix


def test_authority_gate_accepts_non_mutating_route_probe_when_openapi_omits_routes() -> None:
    class Response:
        status_code = 405

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @property
        def content(self) -> bytes:
            raise AssertionError("route-probe response body must not be read")

        def json(self) -> dict:
            raise AssertionError("route-probe response body must not be decoded")

    class Client:
        probed_urls: list[str] = []

        def request(self, method: str, url: str, **kwargs):
            assert method == "GET"
            assert not kwargs
            if url.endswith("/openapi.json"):
                return type(
                    "OpenAPIResponse",
                    (),
                    {"status_code": 200, "json": lambda self: {"paths": {}}},
                )()
            raise AssertionError("behavioral route probes must be streamed")

        def stream(self, method: str, url: str, **kwargs):
            assert method == "GET"
            assert not kwargs
            self.probed_urls.append(url)
            return Response()

    client = Client()
    config = _config()
    result = self_hosted.assert_authoritative_routes_deployed(client, config)
    assert result == {
        "mode": "behavioral_method_not_allowed",
        "method": "GET",
        "statuses": {"provisioning": 405, "scoring": 405},
    }
    assert client.probed_urls == [
        self_hosted.ORCHESTRATOR
        + self_hosted.authoritative_route(config, "provisioning"),
        self_hosted.ORCHESTRATOR + self_hosted.authoritative_route(config, "scoring"),
    ]


def test_authority_gate_fails_closed_when_behavioral_route_probe_is_not_matched() -> None:
    class Response:
        status_code = 404

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class Client:
        def request(self, method: str, url: str, **kwargs):
            assert method == "GET"
            if url.endswith("/openapi.json"):
                return type(
                    "OpenAPIResponse",
                    (),
                    {"status_code": 200, "json": lambda self: {"paths": {}}},
                )()
            raise AssertionError("behavioral route probes must be streamed")

        def stream(self, method: str, url: str, **kwargs):
            assert method == "GET"
            assert not kwargs
            return Response()

    with pytest.raises(RuntimeError, match="routes are not deployed"):
        self_hosted.assert_authoritative_routes_deployed(Client(), _config())


def test_authority_gate_prefers_openapi_without_behavioral_probe() -> None:
    config = _config()
    expected_paths = {
        config["authority"]["provisioning_route_template"]: {},
        config["authority"]["scoring_route_template"]: {},
    }

    class Client:
        def request(self, method: str, url: str, **kwargs):
            assert method == "GET"
            assert url.endswith("/openapi.json")
            return type(
                "OpenAPIResponse",
                (),
                {"status_code": 200, "json": lambda self: {"paths": expected_paths}},
            )()

    assert self_hosted.assert_authoritative_routes_deployed(Client(), config) == {
        "mode": "openapi",
        "routes": sorted(expected_paths),
    }


def test_qwen_trace_normalization_preserves_calls_results_and_thinking() -> None:
    events = [
        {
            "type": "assistant",
            "timestamp": "t1",
            "message": {
                "role": "model",
                "parts": [
                    {"text": "reason", "thought": True},
                    {"text": "working"},
                    {
                        "functionCall": {
                            "id": "c1",
                            "name": "fleet__bash",
                            "args": {"cmd": "id"},
                        }
                    },
                ],
            },
        },
        {
            "type": "tool_result",
            "timestamp": "t2",
            "message": {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "c1",
                            "name": "fleet__bash",
                            "response": {"output": "uid=1000"},
                        }
                    }
                ],
            },
        },
    ]
    messages = self_hosted.normalize_qwen_conversation(events)
    assert messages[0]["thinking"] == "reason"
    assert messages[0]["tool_calls"][0]["id"] == "c1"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "c1"


def test_opencode_trace_normalization_preserves_calls_results_and_thinking() -> None:
    events = [
        {
            "type": "reasoning",
            "part": {"type": "reasoning", "id": "r1", "text": "reason"},
        },
        {
            "type": "tool",
            "part": {
                "type": "tool",
                "id": "c1",
                "tool": "fleet_bash",
                "state": {
                    "status": "completed",
                    "input": {"cmd": "id"},
                    "output": {"output": "uid=1000"},
                },
            },
        },
        {
            "type": "text",
            "part": {"type": "text", "id": "a1", "text": "done"},
        },
    ]
    messages = self_hosted.normalize_opencode_conversation(events)
    assert messages[0]["thinking"] == "reason"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "fleet_bash"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "c1"
    assert messages[2]["content"] == "done"


def test_opencode_trace_normalizes_v11827_millisecond_timestamps_for_fleet() -> None:
    messages = self_hosted.normalize_opencode_conversation(
        [
            {
                "type": "text",
                "timestamp": 1781623587086,
                "part": {"type": "text", "id": "a1", "text": "done"},
            }
        ]
    )
    assert messages[0]["timestamp"] == "2026-06-16T15:26:27.086Z"
    self_hosted.validate_session_messages(messages)


def test_opencode_trace_normalizes_part_time_start_and_rejects_invalid_values() -> None:
    messages = self_hosted.normalize_opencode_conversation(
        [
            {
                "type": "text",
                "part": {
                    "type": "text",
                    "id": "a1",
                    "text": "done",
                    "time": {"start": 1781623587086, "end": 1781623587999},
                },
            }
        ]
    )
    assert messages[0]["timestamp"] == "2026-06-16T15:26:27.086Z"
    with pytest.raises(ValueError, match="unsupported type"):
        self_hosted.normalize_opencode_timestamp([1781623587086])


def test_session_message_validation_rejects_numeric_timestamp_before_mutation() -> None:
    with pytest.raises(ValueError, match="ISO-8601 string"):
        self_hosted.validate_session_messages(
            [{"role": "assistant", "content": "done", "timestamp": 1781623587086}]
        )


def test_session_trace_ingest_is_bounded_ordered_and_scores_only_final_chunk() -> None:
    calls: list[dict] = []

    class Client:
        def request(self, method: str, url: str, **kwargs):
            assert method == "POST"
            assert url.endswith("/v1/sessions/ingest")
            calls.append(kwargs["json"])
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "json": lambda self: {
                        "success": True,
                        "session_id": "session-1",
                        "message_count": len(kwargs["json"]["messages"]),
                        "created_new_session": "session_id" not in kwargs["json"],
                    },
                },
            )()

    messages = [{"role": "tool", "content": str(index)} for index in range(65)]
    receipt = self_hosted.ingest_session_trace(
        Client(),
        messages=messages,
        config=_config(),
        instance_id="instance-1",
        score=0.25,
        verifier_execution_id="verify-1",
        metadata={"training_data_eligible": False},
    )

    assert [len(call["messages"]) for call in calls] == [32, 32, 1]
    assert "session_id" not in calls[0]
    assert calls[0]["instance_id"] == "instance-1"
    assert "score" not in calls[0]
    assert calls[1]["session_id"] == "session-1"
    assert "score" not in calls[1]
    assert calls[2]["session_id"] == "session-1"
    assert calls[2]["score"] == 0.25
    assert calls[2]["verifier_execution_id"] == "verify-1"
    assert receipt == {
        "status": "completed",
        "session_id": "session-1",
        "message_count": 65,
        "chunks_completed": 3,
        "chunk_count": 3,
    }


def test_session_trace_ingest_preserves_partial_receipt_without_mutation_retry() -> None:
    calls = 0

    class Client:
        def request(self, method: str, url: str, **kwargs):
            nonlocal calls
            calls += 1
            status = 200 if calls == 1 else 413
            return type(
                "Response",
                (),
                {
                    "status_code": status,
                    "json": lambda self: {
                        "success": True,
                        "session_id": "session-1",
                        "message_count": len(kwargs["json"]["messages"]),
                        "created_new_session": "session_id" not in kwargs["json"],
                    },
                },
            )()

    with pytest.raises(self_hosted.SessionIngestError) as caught:
        self_hosted.ingest_session_trace(
            Client(),
            messages=[{"role": "tool", "content": "x"}] * 33,
            config=_config(),
            instance_id="instance-1",
            score=0.0,
            verifier_execution_id="verify-1",
            metadata={},
        )
    assert calls == 2
    assert caught.value.receipt == {
        "status": "failed",
        "session_id": "session-1",
        "message_count": 33,
        "chunks_completed": 1,
        "chunk_count": 2,
        "error_type": "FleetRequestError",
        "error_code": "fleet_http_error",
        "http_status": 413,
        "method": "POST",
        "route": "/v1/sessions/ingest",
    }


def test_session_trace_ingest_also_bounds_serialized_payload_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict] = []

    class Client:
        def request(self, method: str, url: str, **kwargs):
            payloads.append(kwargs["json"])
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "json": lambda self: {
                        "success": True,
                        "session_id": "session-1",
                        "message_count": len(kwargs["json"]["messages"]),
                        "created_new_session": "session_id" not in kwargs["json"],
                    },
                },
            )()

    monkeypatch.setattr(self_hosted, "SESSION_INGEST_CHUNK_BYTES", 90)
    self_hosted.ingest_session_trace(
        Client(),
        messages=[{"role": "tool", "content": "x" * 20}] * 3,
        config=_config(),
        instance_id="instance-1",
        score=0.0,
        verifier_execution_id="verify-1",
        metadata={},
    )
    assert [len(payload["messages"]) for payload in payloads] == [1, 1, 1]


def test_recover_session_trace_reuses_scored_source_without_model_rerun(tmp_path: Path) -> None:
    config, source, verifier_execution_id = _recovery_source_fixture(tmp_path)
    session_id = "44444444-4444-4444-8444-444444444444"
    post_payloads: list[dict] = []
    inventory_reads = 0

    class Client:
        def request(self, method: str, url: str, **kwargs):
            nonlocal inventory_reads
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                inventory_reads += 1
                sessions = []
                if inventory_reads == 1:
                    sessions.append(
                        {
                            "session_id": "old",
                            "model": "qwen3.8-27b",
                            "status": "completed",
                            "verifier_execution": {
                                "id": "55555555-5555-4555-8555-555555555555"
                            },
                        }
                    )
                else:
                    sessions.append(
                        {
                            "session_id": session_id,
                            "model": "qwen3.8-27b",
                            "status": "completed",
                            "verifier_execution": {"id": verifier_execution_id},
                        }
                    )
                payload = {"sessions": sessions, "has_more": False}
            elif method == "POST" and url.endswith("/v1/sessions/ingest"):
                post_payloads.append(kwargs["json"])
                payload = {
                    "success": True,
                    "session_id": session_id,
                    "message_count": len(kwargs["json"]["messages"]),
                    "created_new_session": True,
                }
            else:
                raise AssertionError((method, url))
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    recovered = self_hosted.recover_session_trace(
        Client(), config=config, source_dir=source, out_dir=tmp_path / "recovery"
    )
    assert recovered["recovered"] is True
    assert recovered["session_id"] == session_id
    assert len(post_payloads) == 1
    assert post_payloads[0]["messages"][0]["timestamp"] == "2026-06-16T15:26:27.086Z"
    assert post_payloads[0]["verifier_execution_id"] == verifier_execution_id
    assert post_payloads[0]["model"] == config["model"]["session_model"]
    assert (tmp_path / "recovery" / "RECOVERY-INTENT.json").exists()
    assert (tmp_path / "recovery" / "RECOVERED.json").exists()


def test_recover_session_trace_blocks_existing_verifier_backed_duplicate(
    tmp_path: Path,
) -> None:
    config, source, verifier_execution_id = _recovery_source_fixture(tmp_path)

    class Client:
        def request(self, method: str, url: str, **kwargs):
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                payload = {
                    "sessions": [
                        {
                            "session_id": "existing",
                            "model": "qwen3.8-27b",
                            "status": "completed",
                            "verifier_execution": {"id": verifier_execution_id},
                        }
                    ],
                    "has_more": False,
                }
            else:
                pytest.fail("duplicate preflight must not mutate Fleet")
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    out = tmp_path / "recovery"
    with pytest.raises(RuntimeError, match="already exists"):
        self_hosted.recover_session_trace(
            Client(), config=config, source_dir=source, out_dir=out
        )
    assert not (out / "RECOVERY-INTENT.json").exists()
    assert (out / "failure.json").exists()


def test_recover_session_trace_requires_zero_completed_original_chunks(tmp_path: Path) -> None:
    config, source, _ = _recovery_source_fixture(tmp_path)
    ingest = json.loads((source / "session-ingest.json").read_text())
    ingest["chunks_completed"] = 1
    _write_json(source / "session-ingest.json", ingest)
    with pytest.raises(RuntimeError, match="zero-chunk"):
        self_hosted._recovery_source(config, source)


def test_resume_partial_session_trace_appends_only_missing_suffix(tmp_path: Path) -> None:
    config, source, verifier_execution_id, session_id = _partial_recovery_source_fixture(
        tmp_path
    )
    post_payloads: list[dict] = []
    inventory_reads = 0
    transcript_reads = 0
    chunks, _ = self_hosted._partial_recovery_source(config, source)
    prefix = [message for chunk in chunks[:1] for message in chunk]
    complete = [message for chunk in chunks for message in chunk]

    class Client:
        def request(self, method: str, url: str, **kwargs):
            nonlocal inventory_reads, transcript_reads
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                inventory_reads += 1
                payload = {
                    "sessions": [
                        {
                            "session_id": session_id,
                            "model": "qwen3.8-27b",
                            "status": "in_progress" if inventory_reads < 3 else "completed",
                            "verifier_execution": (
                                None
                                if inventory_reads < 3
                                else {"id": verifier_execution_id}
                            ),
                        }
                    ],
                    "has_more": False,
                }
            elif method == "GET" and url.endswith(f"/{session_id}/transcript"):
                transcript_reads += 1
                payload = {
                    "harness": {},
                    "instance": {},
                    "task": {},
                    "transcript": prefix if transcript_reads < 3 else complete,
                    "verifier_execution": None,
                }
            elif method == "POST" and url.endswith("/v1/sessions/ingest"):
                post_payloads.append(kwargs["json"])
                payload = {
                    "success": True,
                    "session_id": session_id,
                    "message_count": len(kwargs["json"]["messages"]),
                    "created_new_session": False,
                }
            else:
                raise AssertionError((method, url))
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    client = Client()
    observed = self_hosted.observe_partial_session_resume(
        client, config=config, source_dir=source, out_dir=tmp_path / "observe"
    )
    resumed = self_hosted.resume_partial_session_trace(
        client,
        config=config,
        source_dir=source,
        out_dir=tmp_path / "resume",
        observer_receipt=observed,
    )
    assert resumed["resumed"] is True
    assert resumed["session_id"] == session_id
    assert resumed["chunks_completed_before"] == 1
    assert resumed["chunks_appended"] == 2
    assert [len(payload["messages"]) for payload in post_payloads] == [32, 1]
    assert all(payload["session_id"] == session_id for payload in post_payloads)
    assert all("model" not in payload for payload in post_payloads)
    assert "score" not in post_payloads[0]
    assert post_payloads[1]["verifier_execution_id"] == verifier_execution_id
    intent = json.loads((tmp_path / "resume" / "RESUME-INTENT.json").read_text())
    assert intent["persisted_prefix_message_count"] == 32
    assert intent["model_or_verifier_replayed"] is False
    assert "transcript" not in intent
    assert "done" not in (tmp_path / "observe" / "OBSERVED.json").read_text()


def test_resume_partial_session_trace_rejects_prefix_count_drift(tmp_path: Path) -> None:
    config, source, _, session_id = _partial_recovery_source_fixture(tmp_path)
    posts = 0

    class Client:
        def request(self, method: str, url: str, **kwargs):
            nonlocal posts
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                payload = {
                    "sessions": [
                        {
                            "session_id": session_id,
                            "model": "qwen3.8-27b",
                            "status": "in_progress",
                            "verifier_execution": None,
                        }
                    ],
                    "has_more": False,
                }
            elif method == "GET" and url.endswith(f"/{session_id}/transcript"):
                payload = {
                    "harness": {},
                    "instance": {},
                    "task": {},
                    "transcript": [{}] * 31,
                    "verifier_execution": None,
                }
            elif method == "POST":
                posts += 1
                raise AssertionError("prefix drift must fail before mutation")
            else:
                raise AssertionError((method, url))
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    with pytest.raises(RuntimeError, match="prefix drifted"):
        self_hosted.observe_partial_session_resume(
            Client(), config=config, source_dir=source, out_dir=tmp_path / "observe"
        )
    assert posts == 0
    assert not (tmp_path / "observe" / "OBSERVED.json").exists()


def test_partial_prefix_mismatch_diagnostic_emits_only_shapes_and_digests(
    tmp_path: Path,
) -> None:
    config, source, _, session_id = _partial_recovery_source_fixture(tmp_path)
    chunks, _ = self_hosted._partial_recovery_source(config, source)
    prefix = [message for chunk in chunks[:1] for message in chunk]
    drifted = copy.deepcopy(prefix)
    drifted[0]["content"] = "private server-only value"

    class Client:
        def request(self, method: str, url: str, **kwargs):
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                payload = {
                    "sessions": [
                        {
                            "session_id": session_id,
                            "model": "qwen3.8-27b",
                            "status": "in_progress",
                            "verifier_execution": None,
                        }
                    ],
                    "has_more": False,
                }
            elif method == "GET" and url.endswith(f"/{session_id}/transcript"):
                payload = {
                    "harness": {},
                    "instance": {},
                    "task": {},
                    "transcript": drifted,
                    "verifier_execution": None,
                }
            else:
                pytest.fail("diagnostic must be read-only")
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    receipt = self_hosted.diagnose_partial_session_prefix_mismatch(
        Client(), config=config, source_dir=source, out_dir=tmp_path / "diagnostic"
    )
    assert receipt["resume_allowed"] is False
    assert receipt["first_mismatch_index"] == 0
    assert receipt["server_prefix_message_count"] == len(prefix)
    assert receipt["server_message_shape"]["keys"] == sorted(drifted[0])
    assert receipt["server_message_sha256"] != receipt["local_message_sha256"]
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    encoded = json.dumps(receipt)
    assert "private server-only value" not in encoded
    assert "done" not in encoded


def test_partial_timestamp_projection_requires_exact_full_prefix_equality(
    tmp_path: Path,
) -> None:
    config, source, _, session_id = _partial_recovery_source_fixture(tmp_path)
    chunks, _ = self_hosted._partial_recovery_source(config, source)
    prefix = [message for chunk in chunks[:1] for message in chunk]
    projected = [
        {key: value for key, value in message.items() if key != "timestamp"}
        for message in prefix
    ]

    class Client:
        def request(self, method: str, url: str, **kwargs):
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                payload = {
                    "sessions": [
                        {
                            "session_id": session_id,
                            "model": "qwen3.8-27b",
                            "status": "in_progress",
                            "verifier_execution": None,
                        }
                    ],
                    "has_more": False,
                }
            elif method == "GET" and url.endswith(f"/{session_id}/transcript"):
                payload = {
                    "harness": {},
                    "instance": {},
                    "task": {},
                    "transcript": projected,
                    "verifier_execution": None,
                }
            else:
                pytest.fail("timestamp projection diagnostic must be read-only")
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    receipt = self_hosted.diagnose_partial_session_timestamp_projection(
        Client(), config=config, source_dir=source, out_dir=tmp_path / "projection"
    )
    assert receipt["projection"] == "omit_top_level_timestamp_only"
    assert receipt["projection_bytes_equal"] is True
    assert receipt["structural_mismatch_count_after_projection"] == 0
    assert receipt["server_prefix_sha256"] == receipt["projected_local_prefix_sha256"]
    assert receipt["resume_allowed"] is False
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )


def test_resume_partial_session_trace_rejects_source_drift_after_observer(
    tmp_path: Path,
) -> None:
    config, source, verifier_execution_id, session_id = _partial_recovery_source_fixture(
        tmp_path
    )
    chunks, _ = self_hosted._partial_recovery_source(config, source)
    prefix = [message for chunk in chunks[:1] for message in chunk]
    posts = 0

    class Client:
        def request(self, method: str, url: str, **kwargs):
            nonlocal posts
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                payload = {
                    "sessions": [
                        {
                            "session_id": session_id,
                            "model": "qwen3.8-27b",
                            "status": "in_progress",
                            "verifier_execution": None,
                        }
                    ],
                    "has_more": False,
                }
            elif method == "GET" and url.endswith(f"/{session_id}/transcript"):
                payload = {
                    "harness": {},
                    "instance": {},
                    "task": {},
                    "transcript": prefix,
                    "verifier_execution": None,
                }
            elif method == "POST":
                posts += 1
                raise AssertionError("source drift must fail before mutation")
            else:
                raise AssertionError((method, url))
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    client = Client()
    observed = self_hosted.observe_partial_session_resume(
        client, config=config, source_dir=source, out_dir=tmp_path / "observe"
    )
    reward = source / "reward-result.json"
    reward.write_bytes(reward.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="observer receipt is not authoritative"):
        self_hosted.resume_partial_session_trace(
            client,
            config=config,
            source_dir=source,
            out_dir=tmp_path / "resume",
            observer_receipt=observed,
        )
    assert posts == 0


def test_resume_partial_session_trace_rejects_same_length_final_content_drift(
    tmp_path: Path,
) -> None:
    config, source, verifier_execution_id, session_id = _partial_recovery_source_fixture(
        tmp_path
    )
    chunks, _ = self_hosted._partial_recovery_source(config, source)
    prefix = [message for chunk in chunks[:1] for message in chunk]
    complete = [message for chunk in chunks for message in chunk]
    drifted = copy.deepcopy(complete)
    drifted[-1]["content"] = "different"
    inventory_reads = 0
    transcript_reads = 0

    class Client:
        def request(self, method: str, url: str, **kwargs):
            nonlocal inventory_reads, transcript_reads
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                inventory_reads += 1
                payload = {
                    "sessions": [
                        {
                            "session_id": session_id,
                            "model": "qwen3.8-27b",
                            "status": "in_progress" if inventory_reads < 3 else "completed",
                            "verifier_execution": (
                                None
                                if inventory_reads < 3
                                else {"id": verifier_execution_id}
                            ),
                        }
                    ],
                    "has_more": False,
                }
            elif method == "GET" and url.endswith(f"/{session_id}/transcript"):
                transcript_reads += 1
                payload = {
                    "harness": {},
                    "instance": {},
                    "task": {},
                    "transcript": prefix if transcript_reads < 3 else drifted,
                    "verifier_execution": None,
                }
            elif method == "POST" and url.endswith("/v1/sessions/ingest"):
                payload = {
                    "success": True,
                    "session_id": session_id,
                    "message_count": len(kwargs["json"]["messages"]),
                    "created_new_session": False,
                }
            else:
                raise AssertionError((method, url))
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    client = Client()
    observed = self_hosted.observe_partial_session_resume(
        client, config=config, source_dir=source, out_dir=tmp_path / "observe"
    )
    with pytest.raises(RuntimeError, match="did not reconcile authoritatively"):
        self_hosted.resume_partial_session_trace(
            client,
            config=config,
            source_dir=source,
            out_dir=tmp_path / "resume",
            observer_receipt=observed,
        )


@pytest.mark.parametrize(
    ("status", "model", "verifier"),
    [
        ("completed", "qwen3.8-27b", None),
        ("in_progress", "wrong-model", None),
        ("in_progress", "qwen3.8-27b", {"id": "already-scored"}),
    ],
)
def test_resume_partial_session_trace_rejects_non_authoritative_inventory(
    tmp_path: Path, status: str, model: str, verifier: dict | None
) -> None:
    config, source, _, session_id = _partial_recovery_source_fixture(tmp_path)

    class Client:
        def request(self, method: str, url: str, **kwargs):
            if method == "GET" and url.endswith("/v1/account"):
                payload = {"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
            elif method == "GET" and url.endswith("/v1/sessions"):
                payload = {
                    "sessions": [
                        {
                            "session_id": session_id,
                            "model": model,
                            "status": status,
                            "verifier_execution": verifier,
                        }
                    ],
                    "has_more": False,
                }
            else:
                pytest.fail("invalid inventory must fail before transcript or mutation")
            return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    with pytest.raises(RuntimeError, match="inventory is not authoritative"):
        self_hosted.observe_partial_session_resume(
            Client(), config=config, source_dir=source, out_dir=tmp_path / "observe"
        )


def test_opencode_session_recovery_job_is_create_once_cpu_only_and_no_rerun() -> None:
    manifest = Path("evals/fleet/cluster/opencode-session-recovery-job.yaml").read_text()
    runner = Path("evals/fleet/scripts/run_opencode_session_recovery.sh").read_text()
    submitter = Path("evals/fleet/scripts/submit_opencode_session_recovery.sh").read_text()
    assert "backoffLimit: 0" in manifest
    assert "workload: fleetai-training-ng-cpu" in manifest
    assert "chris-cyber-opencode-evals-v2" in manifest
    assert "docker" not in manifest.lower()
    assert "recover-session" in runner
    assert 'test ! -e "$OUT_ROOT"' in runner
    assert "opencode run" not in runner
    assert "model_rollouts:0" in submitter
    assert "refusing to replace it" in submitter

    collector = Path(
        "evals/fleet/cluster/opencode-session-recovery-collector-job.yaml"
    ).read_text()
    assert "readOnly: true" in collector
    assert "ACCEPTED.json" in collector
    assert "digest_valid" in collector
    assert "FLEET_API_KEY" not in collector


def test_opencode_v2_session_recovery_receipt_is_self_digesting_and_private() -> None:
    path = Path(
        "docs/evidence/qwen38-study/"
        "2026-09-03-opencode-qwen38-glm53-smokes-v2-session-recovered.json"
    )
    receipt = json.loads(path.read_text())
    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    assert receipt["recovery_job"]["model_rollouts"] == 0
    assert receipt["acceptance_job"]["accepted_session_count"] == 2
    assert receipt["privacy"] == {
        "scores_included": False,
        "prompts_or_traces_included": False,
        "session_ids_included": False,
        "task_identifiers_included": False,
        "credentials_included": False,
    }


def test_metadata_only_session_ingest_persists_no_model_content() -> None:
    config = _config()
    config["authority"]["scoring_payload_mode"] = "runtime_evidence_only_v3"
    session_id = "b9391407-8136-4562-b4d6-7ac57ef1efca"
    verifier_id = "fcaa240d-e625-47d9-b2d3-042c33de27af"
    observed: dict = {}

    class Client:
        def request(self, method: str, url: str, **kwargs):
            assert method == "POST"
            assert url.endswith("/v1/sessions/ingest")
            observed.update(kwargs["json"])
            return type(
                "Response",
                (),
                {
                    "status_code": 200,
                    "json": lambda self: {
                        "success": True,
                        "session_id": session_id,
                        "message_count": 0,
                        "created_new_session": True,
                        "evidence_only": True,
                        "trace_persisted": False,
                        "score": 0.25,
                        "verifier_execution_id": verifier_id,
                        "task_key": config["task"]["key"],
                        "eval_task_version_id": config["task"]["version_id"],
                        "instance_id": "instance-1",
                        "model": f"qwen/{config['model']['served_id']}",
                    },
                },
            )()

    receipt = self_hosted.ingest_metadata_only_session(
        Client(),
        config=config,
        instance_id="instance-1",
        evidence_run_id=session_id,
        score=0.25,
        verifier_execution_id=verifier_id,
    )

    assert observed == {
        "messages": [],
        "model": f"qwen/{config['model']['served_id']}",
        "task_key": config["task"]["key"],
        "eval_task_version_id": config["task"]["version_id"],
        "instance_id": "instance-1",
        "score": 0.25,
        "verifier_execution_id": verifier_id,
    }
    assert "metadata" not in observed
    assert "conversation" not in observed
    assert "final_answer" not in observed
    assert receipt == {
        "status": "completed",
        "mode": "metadata_only_runtime_evidence_v1",
        "success": True,
        "evidence_only": True,
        "trace_persisted": False,
        "created_new_session": True,
        "session_id": session_id,
        "evidence_run_id": session_id,
        "message_count": 0,
        "chunks_completed": 1,
        "chunk_count": 1,
        "score": 0.25,
        "model": f"qwen/{config['model']['served_id']}",
        "verifier_execution_id": verifier_id,
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": "instance-1",
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("success", False),
        ("success", 1),
        ("session_id", "11111111-1111-4111-8111-111111111111"),
        ("message_count", 1),
        ("created_new_session", "yes"),
        ("evidence_only", False),
        ("evidence_only", 1),
        ("trace_persisted", True),
        ("trace_persisted", 0),
        ("message_count", False),
        ("score", 0.5),
        ("model", "qwen/wrong"),
        ("verifier_execution_id", "wrong"),
        ("task_key", "wrong"),
        ("eval_task_version_id", "00000000-0000-0000-0000-000000000001"),
        ("instance_id", "wrong"),
    ],
)
def test_metadata_only_session_ingest_rejects_response_drift(
    field: str, replacement: object
) -> None:
    config = _config()
    config["authority"]["scoring_payload_mode"] = "runtime_evidence_only_v3"
    verifier_id = "fcaa240d-e625-47d9-b2d3-042c33de27af"
    response = {
        "success": True,
        "session_id": "b9391407-8136-4562-b4d6-7ac57ef1efca",
        "message_count": 0,
        "created_new_session": False,
        "evidence_only": True,
        "trace_persisted": False,
        "score": 0.25,
        "verifier_execution_id": verifier_id,
        "task_key": config["task"]["key"],
        "eval_task_version_id": config["task"]["version_id"],
        "instance_id": "instance-1",
        "model": f"qwen/{config['model']['served_id']}",
    }
    response[field] = replacement

    class Client:
        def request(self, method: str, url: str, **kwargs):
            return type("Response", (), {"status_code": 200, "json": lambda self: response})()

    with pytest.raises(RuntimeError, match="response"):
        self_hosted.ingest_metadata_only_session(
            Client(),
            config=config,
            instance_id="instance-1",
            evidence_run_id="b9391407-8136-4562-b4d6-7ac57ef1efca",
            score=0.25,
            verifier_execution_id=verifier_id,
        )


def test_runtime_evidence_only_scoring_payload_has_exact_content_free_keys() -> None:
    config = _config()
    config["authority"]["scoring_payload_mode"] = self_hosted.RUNTIME_EVIDENCE_ONLY_V3
    payload = self_hosted.build_scoring_payload(
        config,
        instance_id="instance-1",
        final_answer="private final answer",
        messages=[{"role": "tool", "content": "private tool output"}],
    )
    assert tuple(sorted(payload)) == self_hosted.RUNTIME_EVIDENCE_ONLY_V3_SCORING_KEYS
    assert "conversation" not in payload
    assert "final_answer" not in payload

    payload["conversation"] = []
    with pytest.raises(RuntimeError, match="unsupported fields"):
        self_hosted.validate_scoring_payload(config, payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("task_key", "wrong-task"),
        ("task_version_id", "00000000-0000-4000-8000-000000000001"),
        ("instance_id", "../unsafe-instance"),
        ("evidence_run_id", "00000000-0000-0000-0000-000000000000"),
    ],
)
def test_rollout_instance_response_rejects_identity_drift(field: str, replacement: str) -> None:
    config = _config()
    response = {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": "11111111-1111-4111-8111-111111111111",
        "evidence_run_id": "22222222-2222-4222-8222-222222222222",
    }
    response[field] = replacement
    with pytest.raises(RuntimeError):
        self_hosted.validate_rollout_instance_response(config, response)


def test_rollout_instance_response_accepts_opaque_dns_safe_instance_id() -> None:
    config = _config()
    response = {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": "rx7ruxhfwwpz",
        "evidence_run_id": "22222222-2222-4222-8222-222222222222",
    }
    assert self_hosted.validate_rollout_instance_response(config, response) == (
        "rx7ruxhfwwpz",
        "22222222-2222-4222-8222-222222222222",
    )


@pytest.mark.parametrize(
    "instance_id",
    ("", "-leading", "trailing-", "UPPERCASE", "unsafe/path", "a" * 64),
)
def test_rollout_instance_response_rejects_unsafe_instance_id(instance_id: str) -> None:
    config = _config()
    response = {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
        "evidence_run_id": "22222222-2222-4222-8222-222222222222",
    }
    with pytest.raises(RuntimeError, match="DNS-safe identifier"):
        self_hosted.validate_rollout_instance_response(config, response)


def _authoritative_reward_response(config: dict) -> dict:
    verifier_id = "33333333-3333-4333-8333-333333333333"
    return {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": "11111111-1111-4111-8111-111111111111",
        "reward": 0.25,
        "verifier_execution_id": verifier_id,
        "cyber_verification_result": {
            "schema_version": "cyber_verification_result_v3",
            "reward": 0.25,
            "components": {"private": "must not persist"},
            "diagnostics": {"private": "must not persist"},
            "bindings": {"task_version_id": config["task"]["version_id"]},
        },
        "cyber_evidence": {
            "mode": "authoritative",
            "status": "authoritative",
            "match": True,
            "production_execution_id": verifier_id,
            "direct_verifier": {
                "status": "authoritative",
                "match": True,
                "execution_id": verifier_id,
                "verifier_contract_version": "3.0.0",
                "context_schema_version": "cyber_verification_context_v1",
            },
            "private_evidence": "must not persist",
        },
    }


def test_authoritative_reward_response_retains_only_sanitized_attestation() -> None:
    config = _config()
    config["authority"]["scoring_payload_mode"] = self_hosted.RUNTIME_EVIDENCE_ONLY_V3
    config["authority"]["required_cyber_contract"] = {
        "submission_protocol": "2.0.0",
        "evidence_schema": "1.0.0",
        "verifier_contract": "3.0.0",
    }
    response = _authoritative_reward_response(config)
    sanitized = self_hosted.sanitize_authoritative_reward_response(
        config,
        response,
        instance_id=response["instance_id"],
        evidence_run_id="22222222-2222-4222-8222-222222222222",
    )
    encoded = json.dumps(sanitized)
    assert "must not persist" not in encoded
    assert "cyber_verification_result" not in sanitized
    assert "cyber_evidence" not in sanitized
    attestation = sanitized["direct_authority_attestation"]
    assert attestation["context"]["task_version_id"] == config["task"]["version_id"]
    assert attestation["context"]["verifier_version_id"] == config["verifier"]["version_id"]
    assert attestation["shadow"]["production_execution_id"] == response["verifier_execution_id"]
    assert all(value is False for value in attestation["data_minimization"].values())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("task_version_id", "wrong"),
        lambda value: value["cyber_verification_result"]["bindings"].__setitem__(
            "task_version_id", "wrong"
        ),
        lambda value: value["cyber_evidence"].__setitem__("match", False),
        lambda value: value["cyber_evidence"]["direct_verifier"].__setitem__(
            "execution_id", "wrong"
        ),
        lambda value: value["cyber_evidence"]["direct_verifier"].__setitem__(
            "verifier_contract_version", "3.1.0"
        ),
        lambda value: value["cyber_evidence"]["direct_verifier"].__setitem__(
            "context_schema_version", "cyber_verification_context_v2"
        ),
    ],
)
def test_authoritative_reward_response_rejects_crosslink_drift(mutate) -> None:
    config = _config()
    config["authority"]["scoring_payload_mode"] = self_hosted.RUNTIME_EVIDENCE_ONLY_V3
    config["authority"]["required_cyber_contract"] = {
        "submission_protocol": "2.0.0",
        "evidence_schema": "1.0.0",
        "verifier_contract": "3.0.0",
    }
    response = copy.deepcopy(_authoritative_reward_response(config))
    mutate(response)
    with pytest.raises(RuntimeError, match="binding|attestation"):
        self_hosted.sanitize_authoritative_reward_response(
            config,
            response,
            instance_id="11111111-1111-4111-8111-111111111111",
            evidence_run_id="22222222-2222-4222-8222-222222222222",
        )


def test_docker_secret_is_in_environment_not_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(self_hosted.subprocess, "run", fake_run)
    self_hosted._docker(
        "run",
        "-e",
        "FIXED_AUTH_VALUE",
        "proxy",
        env={"FIXED_AUTH_VALUE": "secret-value"},
    )
    assert "secret-value" not in observed["argv"]
    assert observed["env"]["FIXED_AUTH_VALUE"] == "secret-value"


def test_fixed_proxy_enforces_path_size_and_request_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    monkeypatch.setenv("FIXED_UPSTREAM", f"http://127.0.0.1:{upstream.server_port}")
    monkeypatch.setenv("FIXED_ALLOWED_PATHS", "/allowed")
    monkeypatch.setenv("FIXED_MAX_REQUESTS", "1")
    monkeypatch.setenv("FIXED_MAX_REQUEST_BYTES", "4")
    fixed_proxy.Handler.request_count = 0
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), fixed_proxy.Handler)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    base = f"http://127.0.0.1:{proxy.server_port}"
    try:
        assert httpx.get(base + "/healthz").status_code == 200
        assert httpx.get(base + "/denied").status_code == 404
        assert httpx.post(base + "/allowed", content=b"12345").status_code == 413
        assert httpx.get(base + "/allowed").status_code == 200
        assert httpx.get(base + "/allowed").status_code == 429
    finally:
        proxy.shutdown()
        upstream.shutdown()
