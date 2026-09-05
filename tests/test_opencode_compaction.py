from __future__ import annotations

import copy
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from evals.fleet import opencode_train_sweep, self_hosted


def _config() -> dict:
    return {
        "harness": copy.deepcopy(opencode_train_sweep.HARNESS),
        "model": {"served_id": "compaction-test"},
    }


def test_compaction_budget_leaves_output_and_tool_headroom() -> None:
    settings = self_hosted.opencode_settings(_config())
    limits = settings["provider"]["fleet-cluster"]["models"]["compaction-test"]["limit"]
    # Exercise the explicit-input branch of the pinned OpenCode overflow contract.
    threshold = limits["input"] - settings["compaction"]["reserved"]
    assert threshold + 20000 + limits["output"] == limits["context"]
    # Both observed pre-overflow turns now trigger compaction before the next tool result.
    assert threshold < min(230061, 229008)
    assert settings["compaction"]["auto"] is True
    assert settings.get("plugin", []) == []


@pytest.mark.parametrize(
    "field", ["context_window_size", "max_output_tokens", "compaction_headroom_tokens"]
)
@pytest.mark.parametrize("value", [0, -1, True, "20000", None])
def test_invalid_compaction_budget_is_rejected(field: str, value: object) -> None:
    config = _config()
    config["harness"][field] = value
    with pytest.raises(ValueError, match="positive integer"):
        self_hosted.opencode_settings(config)


def test_compaction_reserve_cannot_consume_the_context() -> None:
    config = _config()
    config["harness"]["compaction_headroom_tokens"] = (
        config["harness"]["context_window_size"] - config["harness"]["max_output_tokens"]
    )
    with pytest.raises(ValueError, match="leave room for input"):
        self_hosted.opencode_settings(config)


def test_historical_treatment_renders_frozen_no_autocontinue_settings() -> None:
    config = _config()
    config["harness"]["context_management"] = "opencode_1.18.27_native_compaction_no_autocontinue"
    config["harness"].pop("compaction_headroom_tokens", None)
    settings = self_hosted.opencode_settings(config)
    assert settings.get("compaction") is None
    assert settings["plugin"] == [
        "file:///home/node/.config/opencode/fleet-disable-compaction-autocontinue.mjs"
    ]
    assert settings["provider"]["fleet-cluster"]["models"]["compaction-test"][
        "limit"
    ] == {"context": 262144, "output": 32768}


def test_unsupported_treatment_is_rejected_before_launch(tmp_path: Path) -> None:
    config = _config()
    config["harness"]["context_management"] = "unsupported"
    output = tmp_path / "attempt"
    with pytest.raises(ValueError, match="explicitly supported"):
        self_hosted.run(config, output, tmp_path / "proxy.py")
    assert not output.exists()


@pytest.mark.parametrize(
    "change",
    [
        None,
        "old_policy",
        "missing_policy",
        "session_id",
        "task_version_id",
        "verifier_execution_id",
    ],
)
def test_new_plan_only_credits_a_matching_smoke(
    monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    tasks = [
        {
            "rank": index + 1,
            "task_key": f"task-{index}",
            "task_version_id": f"version-{index}",
            "split": "train",
            "prompt_sha256": "prompt-digest",
            "env_variables_sha256": "variables-digest",
            "output_json_schema_sha256": "schema-digest",
            "cyber_contract": opencode_train_sweep.AUTHORITY["required_cyber_contract"],
            "env_key": "env",
            "env_version": 1,
            "environment_version_id": "env-version",
            "data_key": "data",
            "data_version": 1,
            "runtime_seed_content_sha256": "seed-digest",
            "verifier": {},
        }
        for index in range(100)
    ]
    split = {"tasks": tasks, "manifest_digest": "split-digest"}
    selection = {
        "schema_version": opencode_train_sweep.SELECTION_SCHEMA,
        "tasks": tasks,
        "split_manifest_digest": split["manifest_digest"],
    }
    selection["selection_sha256"] = self_hosted.digest_without(selection, "selection_sha256")
    result = {
        "harness_config": copy.deepcopy(opencode_train_sweep.HARNESS),
        "session_id": "smoke-session",
        "task_key": "task-0",
        "task_version_id": "version-0",
        "verifier_execution_id": "verifier-execution",
        "agent_exit_code": 0,
        "agent_termination": "completed",
        "session_ingest_status": "completed",
    }
    if change == "old_policy":
        result["harness_config"]["context_management"] = "native_compaction_no_autocontinue"
    elif change == "missing_policy":
        result.pop("harness_config")
    elif change:
        result[change] = "different"
    monkeypatch.setenv("FLEET_API_KEY", "inert-local-test")
    monkeypatch.setattr(opencode_train_sweep.holdout, "_task_receipt", lambda _c, row, _r: row)
    monkeypatch.setattr(opencode_train_sweep.holdout, "live_model_identity", lambda *_a: {})
    monkeypatch.setattr(self_hosted, "assert_authoritative_routes_deployed", lambda *_a: None)
    monkeypatch.setattr(
        opencode_train_sweep,
        "_session_inventory",
        lambda *_a: [
            {
                "session_id": "smoke-session",
                "task_key": "task-0",
                "model": "qwen3.8-27b",
                "status": "completed",
                "verifier_execution": {"id": "verifier-execution"},
            }
        ],
    )
    transport = httpx.MockTransport(
        lambda _r: httpx.Response(
            200, json={"team_name": "fleet", "team_id": self_hosted.FLEET_TEAM_ID}
        )
    )
    with httpx.Client(transport=transport) as client:
        if change:
            with pytest.raises((ValueError, RuntimeError), match="smoke"):
                opencode_train_sweep.build_full_plan(
                    client, selection, split, "qwen38", "smoke-session", credit_result=result
                )
        else:
            plan = opencode_train_sweep.build_full_plan(
                client, selection, split, "qwen38", "smoke-session", credit_result=result
            )
            assert len(plan["attempts"]) == 199
            assert plan["harness"] == result["harness_config"]
            assert plan["credited_smoke"]["result_sha256"] == self_hosted.sha256(
                self_hosted.canonical_json(result)
            )


@pytest.mark.parametrize("disable_continue", [False, True], ids=["fixed", "original-override"])
def test_pinned_opencode_compacts_and_resumes(tmp_path: Path, disable_continue: bool) -> None:
    """Real OpenCode process; only model generation and an inert MCP tool are fakes."""
    binary = os.environ.get("OPENCODE_TEST_BINARY")
    if not binary:
        pytest.skip("set OPENCODE_TEST_BINARY to an OpenCode 1.18.27 executable")
    assert subprocess.check_output([binary, "--version"], text=True).strip() == "1.18.27"
    requests: list[dict] = []

    class Server(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            # The local plugin has no imports. Fail optional npm discovery locally
            # instead of making this regression depend on the public registry.
            self.send_error(404)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/mcp":
                method = body.get("method")
                if "id" not in body:
                    self.send_response(202)
                    self.end_headers()
                    return
                result = {
                    "initialize": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "inert-test-tool", "version": "1"},
                    },
                    "tools/list": {
                        "tools": [
                            {
                                "name": "bash",
                                "description": "An inert test tool; executes nothing.",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                        ]
                    },
                    "tools/call": {"content": [{"type": "text", "text": "Tool completed."}]},
                    "prompts/list": {"prompts": []},
                    "resources/list": {"resources": []},
                }.get(method, {})
                payload = json.dumps(
                    {"jsonrpc": "2.0", "id": body["id"], "result": result}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            requests.append(body)
            index = len(requests)
            if index == 1:
                delta = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "test-call",
                            "type": "function",
                            "function": {"name": "fleet_bash", "arguments": "{}"},
                        }
                    ],
                }
                finish, tokens = "tool_calls", 210000
            else:
                delta = {
                    "role": "assistant",
                    "content": (
                        "The tool completed. Continue the original task."
                        if index == 2
                        else "COMPACTION_RESUMED"
                    ),
                }
                finish, tokens = "stop", 100
            chunks = [
                {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    "usage": {
                        "prompt_tokens": tokens,
                        "completion_tokens": 10,
                        "total_tokens": tokens + 10,
                    },
                },
            ]
            payload = (
                "".join(
                    "data: "
                    + json.dumps(
                        {
                            "id": f"test-{index}",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": "compaction-test",
                            **chunk,
                        }
                    )
                    + "\n\n"
                    for chunk in chunks
                )
                + "data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload.encode())))
            self.end_headers()
            self.wfile.write(payload.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    settings = self_hosted.opencode_settings(_config())
    origin = f"http://127.0.0.1:{server.server_port}"
    settings["provider"]["fleet-cluster"]["options"]["baseURL"] = origin + "/v1"
    settings["mcp"]["fleet"]["url"] = origin + "/mcp"
    settings["enabled_providers"] = ["fleet-cluster"]
    if disable_continue:
        plugin = tmp_path / "disable-continue.mjs"
        plugin.write_text(
            'export const Disable = async () => ({"experimental.compaction.autocontinue": '
            "async (_input, output) => { output.enabled = false; }});\n"
        )
        settings["plugin"] = [plugin.as_uri()]
    config_path = tmp_path / "opencode.json"
    config_path.write_text(json.dumps(settings))
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "OPENCODE_CONFIG": str(config_path),
        "OPENCODE_DISABLE_MODELS_FETCH": "true",
        "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
        "OPENCODE_DISABLE_AUTOUPDATE": "true",
        "NPM_CONFIG_REGISTRY": origin,
        "NPM_CONFIG_FETCH_RETRIES": "0",
    }
    try:
        result = subprocess.run(
            [
                binary,
                "run",
                "--format",
                "json",
                "--model",
                "fleet-cluster/compaction-test",
                "--title",
                "Compaction regression",
                "--dir",
                str(tmp_path),
                "--auto",
                "--",
                "Call the test tool, then finish.",
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert result.returncode == 0, result.stderr + result.stdout
    assert len(requests) == (2 if disable_continue else 3), result.stdout + result.stderr
    assert not requests[1].get("tools"), "The second request must be compaction, not agent work"
    if not disable_continue:
        assert requests[2].get("tools"), "Agent tools must return after compaction"
        assert "COMPACTION_RESUMED" in result.stdout
    else:
        assert "COMPACTION_RESUMED" not in result.stdout
