"""Non-scored OpenCode 1.18.27 parity against hosted model endpoints.

This probe runs the real pinned OpenCode image and provider adapter with a
local, inert MCP server exposing the frozen bash and submit_report schemas. It
never contacts Fleet task, instance, session, verifier, or scoring routes and
never persists model responses, tool arguments, or credentials.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
import threading
import uuid
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import laptop_opencode_lane_v1 as laptop
from evals.fleet import production_blackbox_tool_catalog_v1 as production_tools
from evals.fleet import self_hosted

SCHEMA = "fleet-opencode-actual-harness-hosted-parity-v1"
HOSTED_ORIGIN = "https://inference.flt.build"
IMAGE = laptop.IMAGE
IMAGE_ID = laptop.IMAGE_ID
MODEL_KEYS = ("qwen3.8-27b", "glm-5.3")
EXPECTED_CALL_ORDER = ["bash", "submit_report"]
MAX_MODEL_REQUESTS = 600
TIMEOUT_SECONDS = 900
FINAL_MARKER = "HARNESS_PARITY_COMPLETE"
REPO_ROOT = Path(__file__).resolve().parents[2]


class ActualHarnessParityError(RuntimeError):
    """Stable content-free parity failure."""


def mcp_tools() -> list[dict[str, Any]]:
    return production_tools.load(REPO_ROOT)


def expected_openai_tools() -> list[dict[str, Any]]:
    projected = []
    for row in mcp_tools():
        parameters = copy.deepcopy(row["inputSchema"])
        if parameters.get("type") != "object" or "additionalProperties" in parameters:
            raise ActualHarnessParityError("production_mcp_schema_projection_drift")
        # OpenCode 1.18.27 / AI SDK closes MCP object schemas on projection.
        parameters["additionalProperties"] = False
        projected.append(
            {
                "type": "function",
                "function": {
                    "name": f"fleet_{row['name']}",
                    "description": row["description"],
                    "parameters": parameters,
                },
            },
        )
    return projected


def classify_model_request_catalog(body: Mapping[str, Any], expected_model: str) -> dict[str, Any]:
    """Classify tool-bearing versus normal terminal requests without content."""
    expected = expected_openai_tools()
    actual_tools = body.get("tools")
    if not isinstance(actual_tools, list):
        return {"classification": "NO_TOOLS_TERMINAL_REQUEST", "catalog_sha256": "absent"}
    actual_functions = [row.get("function") for row in actual_tools if isinstance(row, Mapping)]
    names_valid = [row.get("name") for row in actual_functions] == [
        row["function"]["name"] for row in expected
    ]
    descriptions_valid = [row.get("description") for row in actual_functions] == [
        row["function"]["description"] for row in expected
    ]
    parameters_valid = [row.get("parameters") for row in actual_functions] == [
        row["function"]["parameters"] for row in expected
    ]
    return {
        "classification": "CATALOG_BEARING_REQUEST",
        "catalog_sha256": crypto.sha256(self_hosted.canonical_json(actual_tools)),
        "names_valid": names_valid,
        "descriptions_valid": descriptions_valid,
        "parameters_valid": parameters_valid,
        "catalog_valid": (
            body.get("model") == expected_model
            and actual_tools == expected
            and names_valid
            and descriptions_valid
            and parameters_valid
        ),
    }


def validate_server_binding(value: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    required = {
        "api_run_id",
        "rayjob_uid",
        "head_pod_uid",
        "service_uid",
        "served_id",
        "model_revision",
        "context_length",
    }
    if set(value) != required or not isinstance(value.get("api_run_id"), str):
        raise ActualHarnessParityError("dedicated_server_binding_invalid")
    if not value["api_run_id"].startswith("ft-run-"):
        raise ActualHarnessParityError("dedicated_server_binding_invalid")
    for field in ("rayjob_uid", "head_pod_uid", "service_uid"):
        try:
            parsed = uuid.UUID(str(value.get(field)))
        except ValueError as exc:
            raise ActualHarnessParityError("dedicated_server_binding_invalid") from exc
        if parsed.int == 0:
            raise ActualHarnessParityError("dedicated_server_binding_invalid")
    expected_model = exact.EXPECTED_MODELS[model_key]
    if (
        value.get("served_id") != expected_model["served_id"]
        or value.get("model_revision") != expected_model["revision"]
        or value.get("context_length") != 262144
    ):
        raise ActualHarnessParityError("dedicated_server_binding_invalid")
    return dict(value)


def treatment_config(model_key: str) -> dict[str, Any]:
    if model_key not in MODEL_KEYS:
        raise ActualHarnessParityError("unsupported_model")
    treatment = exact.EXPECTED_TREATMENT
    return {
        "harness": {
            "name": treatment["harness"],
            "version": treatment["harness_version"],
            "release_asset_sha256": treatment["release_asset_sha256"],
            "provider_adapter": treatment["provider_adapter"],
            "context_management": treatment["context_management"],
            "context_window_size": treatment["context_window_size"],
            "compaction_headroom_tokens": treatment["compaction_headroom_tokens"],
            "max_output_tokens": treatment["max_output_tokens"],
            "max_model_requests": treatment["max_model_requests"],
            "timeout_seconds": treatment["timeout_seconds"],
        },
        "model": exact.EXPECTED_MODELS[model_key],
    }


def render_settings(model_key: str, model_port: int, mcp_port: int) -> dict[str, Any]:
    settings = self_hosted.opencode_settings(treatment_config(model_key))
    provider = settings["provider"]["fleet-cluster"]
    provider["options"]["baseURL"] = f"http://host.docker.internal:{model_port}/v1"
    settings["mcp"]["fleet"]["url"] = f"http://host.docker.internal:{mcp_port}/mcp"
    settings["enabled_providers"] = ["fleet-cluster"]
    return settings


def _json_response(handler: BaseHTTPRequestHandler, status: int, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


class _State:
    def __init__(self, model: str, upstream_origin: str, api_key: str) -> None:
        self.model = model
        self.upstream_origin = upstream_origin.rstrip("/")
        self.api_key = api_key
        self.lock = threading.Lock()
        self.model_requests = 0
        self.model_tool_catalog_valid = True
        self.model_tool_names_valid = True
        self.model_tool_descriptions_valid = True
        self.model_tool_parameters_valid = True
        self.model_tool_catalog_sha256s: list[str] = []
        self.model_requests_with_tools = 0
        self.model_requests_without_tools = 0
        self.mcp_calls: list[str] = []
        self.mcp_arguments_valid = True


def _mcp_handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/mcp":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            except (KeyError, ValueError, json.JSONDecodeError):
                self.send_error(400)
                return
            if "id" not in body:
                self.send_response(202)
                self.end_headers()
                return
            method = body.get("method")
            if method == "initialize":
                result: Any = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fleet-inert-parity", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": mcp_tools()}
            elif method in {"prompts/list", "resources/list"}:
                result = {"prompts" if method == "prompts/list" else "resources": []}
            elif method == "tools/call":
                params = body.get("params") or {}
                name = params.get("name")
                arguments = params.get("arguments")
                valid = isinstance(arguments, Mapping)
                if name == "bash":
                    valid = (
                        valid
                        and set(arguments).issubset({"script", "timeoutMs"})
                        and isinstance(arguments.get("script"), str)
                        and bool(arguments["script"])
                        and (
                            "timeoutMs" not in arguments
                            or (
                                type(arguments["timeoutMs"]) is int
                                and 1 <= arguments["timeoutMs"] <= 300000
                            )
                        )
                    )
                elif name == "submit_report":
                    valid = (
                        valid
                        and set(arguments).issubset({"flag", "flags", "verdict", "explanation"})
                        and isinstance(arguments.get("explanation"), str)
                        and arguments.get("verdict") == "no_flag"
                        and "flag" not in arguments
                        and "flags" not in arguments
                    )
                else:
                    valid = False
                with state.lock:
                    state.mcp_calls.append(str(name))
                    state.mcp_arguments_valid = state.mcp_arguments_valid and valid
                result = {"content": [{"type": "text", "text": "Tool completed."}]}
            else:
                result = {}
            _json_response(
                self,
                200,
                {"jsonrpc": "2.0", "id": body["id"], "result": result},
            )

    return Handler


def _model_handler(state: _State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                self.send_error(400)
                return
            observation = classify_model_request_catalog(body, state.model)
            has_tools = observation["classification"] == "CATALOG_BEARING_REQUEST"
            with state.lock:
                state.model_requests += 1
                if has_tools:
                    state.model_requests_with_tools += 1
                    state.model_tool_catalog_valid = (
                        state.model_tool_catalog_valid and observation["catalog_valid"]
                    )
                    state.model_tool_names_valid = (
                        state.model_tool_names_valid and observation["names_valid"]
                    )
                    state.model_tool_descriptions_valid = (
                        state.model_tool_descriptions_valid and observation["descriptions_valid"]
                    )
                    state.model_tool_parameters_valid = (
                        state.model_tool_parameters_valid and observation["parameters_valid"]
                    )
                else:
                    state.model_requests_without_tools += 1
                state.model_tool_catalog_sha256s.append(observation["catalog_sha256"])
                count = state.model_requests
            if count > MAX_MODEL_REQUESTS:
                self.send_error(429)
                return
            try:
                with httpx.stream(
                    "POST",
                    state.upstream_origin + "/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": self.headers.get("Accept", "*/*"),
                        **({"Authorization": f"Bearer {state.api_key}"} if state.api_key else {}),
                    },
                    content=raw,
                    timeout=300,
                ) as response:
                    self.send_response(response.status_code)
                    self.send_header(
                        "Content-Type", response.headers.get("content-type", "application/json")
                    )
                    self.end_headers()
                    for chunk in response.iter_bytes():
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except (httpx.HTTPError, BrokenPipeError):
                return

    return Handler


def _serve(server: ThreadingHTTPServer) -> threading.Thread:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def run(
    model_key: str,
    api_key: str,
    *,
    upstream_origin: str = HOSTED_ORIGIN,
    server_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    is_hosted = upstream_origin.rstrip("/") == HOSTED_ORIGIN
    if is_hosted and not api_key:
        raise ActualHarnessParityError("fleet_credential_absent")
    if not is_hosted and not server_binding:
        raise ActualHarnessParityError("dedicated_server_binding_absent")
    validated_binding = (
        None if is_hosted else validate_server_binding(server_binding or {}, model_key)
    )
    config = treatment_config(model_key)
    served_id = config["model"]["served_id"]
    state = _State(served_id, upstream_origin, api_key)
    model_server = ThreadingHTTPServer(("127.0.0.1", 0), _model_handler(state))
    mcp_server = ThreadingHTTPServer(("127.0.0.1", 0), _mcp_handler(state))
    settings = render_settings(model_key, model_server.server_port, mcp_server.server_port)
    threads = [_serve(model_server), _serve(mcp_server)]
    try:
        with tempfile.TemporaryDirectory(prefix="opencode-parity-") as temp:
            root = Path(temp)
            home = root / "home"
            config_dir = home / ".config" / "opencode"
            config_dir.mkdir(parents=True)
            (config_dir / "opencode.json").write_bytes(self_hosted.canonical_json(settings) + b"\n")
            workspace = root / "workspace"
            workspace.mkdir()
            prompt = (
                "This is a non-scored availability check. Call bash exactly once with a benign "
                "script. Then call submit_report exactly once with verdict no_flag and a brief "
                f"benign explanation. After both tools complete, reply {FINAL_MARKER}."
            )
            completed = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--platform",
                    "linux/amd64",
                    "-e",
                    "OPENCODE_DISABLE_MODELS_FETCH=true",
                    "-e",
                    "OPENCODE_DISABLE_DEFAULT_PLUGINS=true",
                    "-e",
                    "OPENCODE_DISABLE_AUTOUPDATE=true",
                    "-v",
                    f"{home}:/home/node",
                    "-v",
                    f"{workspace}:/workspace",
                    IMAGE,
                    "opencode",
                    "run",
                    "--format",
                    "json",
                    "--thinking",
                    "--model",
                    f"fleet-cluster/{served_id}",
                    "--dir",
                    "/workspace",
                    "--auto",
                    "--",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
    finally:
        model_server.shutdown()
        mcp_server.shutdown()
        model_server.server_close()
        mcp_server.server_close()
        for thread in threads:
            thread.join()
    marker = FINAL_MARKER in completed.stdout
    with state.lock:
        calls = list(state.mcp_calls)
        arguments_valid = state.mcp_arguments_valid
        model_requests = state.model_requests
        catalog_valid = state.model_tool_catalog_valid
        names_valid = state.model_tool_names_valid
        descriptions_valid = state.model_tool_descriptions_valid
        parameters_valid = state.model_tool_parameters_valid
        actual_catalog_digests = sorted(set(state.model_tool_catalog_sha256s))
        requests_with_tools = state.model_requests_with_tools
        requests_without_tools = state.model_requests_without_tools
    passed = (
        completed.returncode == 0
        and calls == EXPECTED_CALL_ORDER
        and arguments_valid
        and catalog_valid
        and requests_with_tools >= 1
        and marker
        and 1 <= model_requests <= MAX_MODEL_REQUESTS
    )
    body = {
        "schema_version": SCHEMA,
        "status": "PASSED_NON_SCORED" if passed else "FAILED",
        "classification": ("ACTUAL_HARNESS_PARITY" if passed else "ACTUAL_HARNESS_PARITY_BLOCKER"),
        "model": config["model"],
        "endpoint": {
            "origin": upstream_origin.rstrip("/"),
            "kind": (
                "shared_hosted_inference"
                if upstream_origin.rstrip("/") == HOSTED_ORIGIN
                else "dedicated_uid_bound_inference"
            ),
            "server_binding": validated_binding,
            "server_binding_sha256": (
                crypto.sha256(self_hosted.canonical_json(validated_binding))
                if validated_binding is not None
                else None
            ),
        },
        "harness": {
            **config["harness"],
            "image": IMAGE,
            "image_id": IMAGE_ID,
            "settings_sha256": crypto.sha256(self_hosted.canonical_json(settings)),
        },
        "tool_contract": {
            "names": EXPECTED_CALL_ORDER,
            "mcp_catalog_sha256": crypto.sha256(self_hosted.canonical_json(mcp_tools())),
            "production_catalog_provenance": production_tools.provenance(REPO_ROOT),
            "openai_catalog_sha256": crypto.sha256(
                self_hosted.canonical_json(expected_openai_tools())
            ),
            "model_request_catalog_exact": catalog_valid,
            "model_request_tool_names_exact": names_valid,
            "model_request_tool_descriptions_exact": descriptions_valid,
            "model_request_tool_parameters_exact": parameters_valid,
            "observed_model_request_catalog_sha256s": actual_catalog_digests,
            "model_requests_with_tools": requests_with_tools,
            "model_requests_without_tools": requests_without_tools,
            "calls_observed_in_order": calls,
            "arguments_structurally_valid": arguments_valid,
        },
        "execution": {
            "harness_exit_code": completed.returncode,
            "model_requests": model_requests,
            "final_marker_observed": marker,
            "task_instance_session_verifier_scoring_calls": 0,
            "scored_launch_authorized": False,
        },
        "privacy": {
            "credentials_included": False,
            "prompt_included": False,
            "responses_or_model_outputs_included": False,
            "tool_arguments_included": False,
            "stderr_or_stdout_included": False,
            "benchmark_content_included": False,
        },
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--origin", default=HOSTED_ORIGIN)
    parser.add_argument("--server-binding", type=Path)
    args = parser.parse_args()
    key = os.environ.get("FLEET_API_KEY", "")
    binding = None
    if args.server_binding is not None:
        loaded = json.loads(args.server_binding.read_text())
        if not isinstance(loaded, dict):
            raise ActualHarnessParityError("dedicated_server_binding_invalid")
        binding = loaded
    result = run(
        args.model,
        key,
        upstream_origin=args.origin,
        server_binding=binding,
    )
    self_hosted.write_json_once(args.out, result)
    return 0 if result["status"] == "PASSED_NON_SCORED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
