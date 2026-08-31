from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from evals.fleet import fixed_proxy, self_hosted

CONFIG_PATH = Path(
    "evals/fleet/configs/qwen36-27b-qwen-code-selfhosted-smoke-v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def test_smoke_config_is_one_exact_eval_only_arm() -> None:
    config = _config()
    assert config["run_id"] == "chris-cyber-qwen36-qwencode0223-fleet-smoke-v1"
    assert config["model"]["revision"] == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert config["harness"]["version"] == "0.22.3"
    assert config["harness"]["source_commit"] == "09825973e7d3c3fd07e17909c396aa62f48ce51f"
    assert config["execution"] == {
        "pass_k": 1,
        "max_concurrent": 1,
        "network": "chris-qwen-fleet-smoke-v1",
        "training_data_eligible": False,
    }


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
