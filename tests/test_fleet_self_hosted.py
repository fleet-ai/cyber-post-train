from __future__ import annotations

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from evals.fleet import fixed_proxy, self_hosted

CONFIG_PATH = Path("evals/fleet/configs/qwen36-27b-qwen-code-selfhosted-smoke-v1.json")


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


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
    expected_route = f"/v1/tasks/{config['task']['key']}"
    fixture = {
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

    assert self_hosted.load_and_verify_task(Client(), config) == fixture


def test_build_instance_payload_preserves_exact_runtime_binding() -> None:
    config = _config()
    task = {
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


def test_authority_gate_accepts_exact_behavioral_guard_when_openapi_lags() -> None:
    class Response:
        status_code = 422
        content = b"yes"

        def json(self) -> dict:
            return {"detail": "Authoritative RL rollout rewards support report-only tasks"}

    class Client:
        def request(self, method: str, url: str, **kwargs):
            assert method == "GET"
            return type(
                "OpenAPIResponse",
                (),
                {"status_code": 200, "json": lambda self: {"paths": {}}},
            )()

        def post(self, url: str, **kwargs):
            return Response()

    result = self_hosted.assert_authoritative_routes_deployed(Client(), _config())
    assert result == {
        "mode": "behavioral_report_only_guard",
        "statuses": {"provisioning": 422, "scoring": 422},
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
                {"status_code": 200, "json": lambda self: {"session_id": "session-1"}},
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
                    "json": lambda self: {"session_id": "session-1"},
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
        "error_type": "RuntimeError",
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
                {"status_code": 200, "json": lambda self: {"session_id": "session-1"}},
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
                        "session_id": session_id,
                        "message_count": 0,
                        "score": 0.25,
                        "verifier_execution_id": verifier_id,
                        "task_key": config["task"]["key"],
                        "eval_task_version_id": config["task"]["version_id"],
                        "instance_id": "instance-1",
                    },
                },
            )()

    receipt = self_hosted.ingest_metadata_only_session(
        Client(),
        config=config,
        instance_id="instance-1",
        evidence_run_id="77777777-7777-4777-8777-777777777777",
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
        "session_id": session_id,
        "evidence_run_id": "77777777-7777-4777-8777-777777777777",
        "message_count": 0,
        "chunks_completed": 1,
        "chunk_count": 1,
        "score": 0.25,
        "verifier_execution_id": verifier_id,
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": "instance-1",
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("session_id", "00000000-0000-0000-0000-000000000000"),
        ("message_count", 1),
        ("score", 0.5),
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
        "session_id": "b9391407-8136-4562-b4d6-7ac57ef1efca",
        "message_count": 0,
        "score": 0.25,
        "verifier_execution_id": verifier_id,
        "task_key": config["task"]["key"],
        "eval_task_version_id": config["task"]["version_id"],
        "instance_id": "instance-1",
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
        ("instance_id", "not-a-uuid"),
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
