"""Self-hosted official Qwen Code runner for one exactly pinned Fleet task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

ORCHESTRATOR = "https://orchestrator.fleetai.com"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
TRANSIENT_READ_STATUS_CODES = {429, 502, 503, 504}
MAX_READ_ATTEMPTS = 6
SESSION_INGEST_CHUNK_MESSAGES = 32
SESSION_INGEST_CHUNK_BYTES = 512 * 1024


class SessionIngestError(RuntimeError):
    """A bounded trace ingest failed after zero or more recorded chunks."""

    def __init__(self, receipt: dict[str, Any]) -> None:
        super().__init__("Fleet session trace ingest did not complete")
        self.receipt = receipt


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Durably claim a one-shot boundary before its external side effect."""
    payload = canonical_json(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    attempts = MAX_READ_ATTEMPTS if method == "GET" else 1
    for attempt in range(attempts):
        response = client.request(method, f"{ORCHESTRATOR}{path}", **kwargs)
        if response.status_code < 400:
            return response.json()
        if response.status_code not in TRANSIENT_READ_STATUS_CODES or attempt + 1 == attempts:
            route = path.split("?")[0]
            raise RuntimeError(f"Fleet {method} {route} failed with HTTP {response.status_code}")
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def load_and_verify_task(client: httpx.Client, config: dict[str, Any]) -> dict[str, Any]:
    expected = config["task"]
    # The legacy source-job roster is large and is provenance only. Read the
    # frozen task/version directly; the versioned provisioning and scoring
    # authorities hydrate and revalidate this exact pair again server-side.
    task = _request(
        client,
        "GET",
        f"/v1/tasks/{expected['key']}",
        params={"version_id": expected["version_id"]},
    )
    verifier = task.get("verifier") or {}
    metadata = task.get("metadata") or {}
    actual = {
        "key": task.get("key"),
        "environment_id": task.get("environment_id"),
        "environment_version": task.get("version"),
        "data_id": task.get("data_id"),
        "data_version": task.get("data_version"),
        "prompt_sha256": sha256((task.get("prompt") or "").encode()),
        "env_variables_sha256": sha256(canonical_json(task.get("env_variables") or {})),
        "output_json_schema_sha256": sha256(canonical_json(task.get("output_json_schema"))),
        "verifier_id": task.get("verifier_id"),
        "verifier_version_id": verifier.get("verifier_version_id"),
        "verifier_version": verifier.get("version"),
        "verifier_sha256": verifier.get("sha256"),
        "runtime_seed_content_sha256": (metadata.get("runtime_seed_manifest") or {}).get(
            "content_sha256"
        ),
    }
    wanted = {
        "key": expected["key"],
        "environment_id": config["environment"]["id"],
        "environment_version": config["environment"]["version"],
        "data_id": config["environment"]["data_id"],
        "data_version": config["environment"]["data_version"],
        "prompt_sha256": expected["prompt_sha256"],
        "env_variables_sha256": expected["env_variables_sha256"],
        "output_json_schema_sha256": expected["output_json_schema_sha256"],
        "verifier_id": config["verifier"]["id"],
        "verifier_version_id": config["verifier"]["version_id"],
        "verifier_version": config["verifier"]["version"],
        "verifier_sha256": config["verifier"]["sha256"],
        "runtime_seed_content_sha256": config["environment"]["runtime_seed_content_sha256"],
    }
    if "code_sha256" in config["verifier"]:
        actual["verifier_code_sha256"] = sha256((verifier.get("code") or "").encode())
        wanted["verifier_code_sha256"] = config["verifier"]["code_sha256"]
    if actual != wanted:
        raise RuntimeError("exact task prompt/verifier/runtime-seed binding drifted")
    return task


def build_instance_payload(config: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    manifest = (task.get("metadata") or {}).get("runtime_seed_manifest") or {}
    overlays = [
        {"target_path": row["target_path"], "s3_key": row["s3_key"], "bucket": row["bucket"]}
        for row in manifest.get("files", [])
    ]
    if not overlays:
        raise RuntimeError("pinned cyber task has no runtime seed overlay")
    return {
        "env_key": config["environment"]["id"],
        "env_version": config["environment"]["version"],
        "data_key": config["environment"]["data_id"],
        "data_version": config["environment"]["data_version"],
        "env_variables": task.get("env_variables") or {},
        "task_id": config["task"]["id"],
        "seed_overlay_files": overlays,
        "ttl_seconds": config["environment"]["ttl_seconds"],
        "run_id": config["run_id"],
        "created_from": "api",
        "max_wait_seconds": 1200,
    }


def _mcp_json(response: httpx.Response) -> dict[str, Any]:
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                value = json.loads(line.removeprefix("data:").strip())
                if isinstance(value, dict):
                    return value
        raise RuntimeError("MCP response contained no JSON data event")
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("MCP returned a non-object response")
    return value


def discover_tools(root_url: str, runner_header: str, runner_token: str) -> tuple[list[str], str]:
    headers = {
        runner_header: runner_token,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    endpoint = root_url.rstrip("/") + "/mcp"
    with httpx.Client(timeout=60) as client:
        init = client.post(
            endpoint,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "cyber-post-train-preflight", "version": "1"},
                },
            },
        )
        init.raise_for_status()
        _mcp_json(init)
        session_id = init.headers.get("mcp-session-id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        ready = client.post(
            endpoint,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        ready.raise_for_status()
        listed = client.post(
            endpoint,
            headers=headers,
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        listed.raise_for_status()
        payload = _mcp_json(listed)
    tools = payload.get("result", {}).get("tools", [])
    names = sorted(row.get("name") for row in tools if isinstance(row, dict) and row.get("name"))
    if not {"bash", "submit_report"}.issubset(names):
        raise RuntimeError("Fleet MCP is missing the blackbox cyber bash/submit_report tools")
    return names, sha256(canonical_json(tools))


def extract_final_answer(path: Path) -> str:
    final = ""
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            final = event["result"]
        message = event.get("message")
        if event.get("type") in {"assistant", "message"} and isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                final = content
            elif isinstance(content, list):
                texts = [
                    row.get("text")
                    for row in content
                    if isinstance(row, dict) and row.get("type") == "text"
                ]
                if texts:
                    final = "\n".join(str(text) for text in texts)
    return final


def load_qwen_chat_trace(qwen_home: Path) -> tuple[list[dict[str, Any]], Path, int]:
    paths = sorted(qwen_home.glob("projects/*/chats/*.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one Qwen Code chat trace, found {len(paths)}")
    events: list[dict[str, Any]] = []
    malformed_line_count = 0
    for line in paths[0].read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed_line_count += 1
            continue
        if isinstance(value, dict):
            events.append(value)
    if not events:
        raise RuntimeError("Qwen Code chat trace contained no JSON events")
    return events, paths[0], malformed_line_count


def normalize_qwen_conversation(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Qwen's lossless chat log to Fleet's role/tool-call trace schema."""
    messages: list[dict[str, Any]] = []
    for event in events:
        message = event.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("parts"), list):
            continue
        parts = message["parts"]
        timestamp = event.get("timestamp")
        if event.get("type") == "tool_result":
            for part in parts:
                response = part.get("functionResponse") if isinstance(part, dict) else None
                if not isinstance(response, dict):
                    continue
                content = response.get("response")
                if not isinstance(content, str):
                    content = json.dumps(content, sort_keys=True, ensure_ascii=True)
                messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_call_id": str(response.get("id") or ""),
                        "timestamp": timestamp,
                        "metadata": {"qwen_tool_name": response.get("name")},
                    }
                )
            continue
        role = message.get("role")
        if role == "model":
            text = []
            thinking = []
            tool_calls = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    target = thinking if part.get("thought") else text
                    target.append(part["text"])
                call = part.get("functionCall")
                if isinstance(call, dict):
                    tool_calls.append(
                        {
                            "id": str(call.get("id") or ""),
                            "type": "function",
                            "function": {
                                "name": str(call.get("name") or ""),
                                "arguments": json.dumps(
                                    call.get("args") or {}, sort_keys=True, ensure_ascii=True
                                ),
                            },
                        }
                    )
            normalized: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text) or None,
                "timestamp": timestamp,
            }
            if thinking:
                normalized["thinking"] = "\n".join(thinking)
            if tool_calls:
                normalized["tool_calls"] = tool_calls
            messages.append(normalized)
        elif role == "user":
            text = [
                part["text"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            if text:
                messages.append(
                    {"role": "user", "content": "\n".join(text), "timestamp": timestamp}
                )
    if not messages:
        raise RuntimeError("Qwen Code chat trace normalized to no messages")
    return messages


def final_answer_from_conversation(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str) and content:
            return content
    return ""


def ingest_session_trace(
    client: httpx.Client,
    *,
    messages: list[dict[str, Any]],
    config: dict[str, Any],
    instance_id: str,
    score: float,
    verifier_execution_id: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Ingest a full trace in bounded, ordered, non-retried mutations.

    Qwen Code traces can contain hundreds of tool messages.  Sending the whole
    trace as one request can exceed an ingress or request-processing limit even
    though authoritative scoring already succeeded.  Create one session with
    the first bounded chunk, append the remaining chunks in order, and only
    attach the score to the final chunk.  Mutating requests remain single-shot.
    """
    if not messages:
        raise ValueError("cannot ingest an empty session trace")
    chunks: list[list[dict[str, Any]]] = []
    for message in messages:
        if len(canonical_json({"messages": [message]})) > SESSION_INGEST_CHUNK_BYTES:
            raise ValueError("one session message exceeds the bounded ingest payload")
        candidate = [*(chunks[-1] if chunks else []), message]
        if chunks and (
            len(candidate) > SESSION_INGEST_CHUNK_MESSAGES
            or len(canonical_json({"messages": candidate})) > SESSION_INGEST_CHUNK_BYTES
        ):
            chunks.append([message])
        elif chunks:
            chunks[-1] = candidate
        else:
            chunks.append(candidate)
    session_id: str | None = None
    receipt: dict[str, Any] = {
        "status": "in_progress",
        "session_id": None,
        "message_count": len(messages),
        "chunks_completed": 0,
        "chunk_count": len(chunks),
    }
    for index, chunk in enumerate(chunks):
        payload: dict[str, Any] = {"messages": chunk}
        if session_id is None:
            payload.update(
                {
                    "model": f"qwen/{config['model']['served_id']}",
                    "task_key": config["task"]["key"],
                    "eval_task_version_id": config["task"]["version_id"],
                    "instance_id": instance_id,
                    "metadata": metadata,
                }
            )
        else:
            payload["session_id"] = session_id
        if index + 1 == len(chunks):
            payload["score"] = score
            payload["verifier_execution_id"] = verifier_execution_id
        try:
            response = _request(client, "POST", "/v1/sessions/ingest", json=payload)
        except Exception as exc:  # noqa: BLE001
            receipt.update(status="failed", error_type=type(exc).__name__)
            raise SessionIngestError(receipt) from exc
        returned_id = response.get("session_id")
        if not isinstance(returned_id, str) or not returned_id:
            receipt.update(status="failed", error_type="MissingSessionId")
            raise SessionIngestError(receipt)
        if session_id is not None and returned_id != session_id:
            receipt.update(status="failed", error_type="SessionIdChanged")
            raise SessionIngestError(receipt)
        session_id = returned_id
        receipt.update(session_id=session_id, chunks_completed=index + 1)
    receipt["status"] = "completed"
    return receipt


def authoritative_route(config: dict[str, Any], kind: str) -> str:
    template = config["authority"][f"{kind}_route_template"]
    return template.format(
        task_key=config["task"]["key"], task_version_id=config["task"]["version_id"]
    )


def assert_authoritative_routes_deployed(
    client: httpx.Client, config: dict[str, Any]
) -> dict[str, Any]:
    openapi = _request(client, "GET", "/openapi.json")
    paths = openapi.get("paths") or {}
    expected = {
        config["authority"]["provisioning_route_template"],
        config["authority"]["scoring_route_template"],
    }
    missing = sorted(path for path in expected if path not in paths)
    if not missing:
        return {"mode": "openapi", "routes": sorted(expected)}

    # Canonical OpenAPI can lag the newly deployed public router. Exercise the
    # report-only guard with a deliberately invalid task shape: the guard runs
    # before hydration/provisioning, so a 422 proves routing without creating an
    # instance or invoking a verifier. A 404 or any other response fails closed.
    probe_prefix = "/v1/rollout-rewards/qwen-route-probe/versions/not-a-task-version"
    results = {}
    for kind, path, body in (
        ("provisioning", probe_prefix + "/instances", {}),
        ("scoring", probe_prefix, {"instance_id": "route-probe"}),
    ):
        response = client.post(f"{ORCHESTRATOR}{path}", json=body)
        try:
            response_body = response.json() if response.content else {}
        except ValueError:
            response_body = {}
        detail = str(response_body.get("detail") or "")
        results[kind] = response.status_code
        if response.status_code != 422 or "report-only" not in detail:
            raise RuntimeError("authoritative rollout-reward routes are not deployed")
    return {"mode": "behavioral_report_only_guard", "statuses": results}


def _docker(
    *args: str,
    check: bool = True,
    capture: bool = False,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    return subprocess.run(
        ["docker", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
        env=process_env,
    )


def wait_for_proxy(container: str, port: int, attempts: int = 30) -> None:
    probe = (
        "import urllib.request; "
        f"assert urllib.request.urlopen('http://127.0.0.1:{port}/healthz', timeout=1).status == 200"
    )
    for _ in range(attempts):
        result = _docker(
            "exec", container, "python", "-c", probe, check=False, capture=True, timeout=5
        )
        if result.returncode == 0:
            return
        time.sleep(1)
    raise RuntimeError(f"fixed proxy {container} did not become ready")


def runtime_preflight(
    client: httpx.Client, config: dict[str, Any], task: dict[str, Any]
) -> dict[str, Any]:
    """Create, inspect, and terminate one exact environment without running an agent."""
    instance_id = None
    terminated_at = None
    try:
        payload = build_instance_payload(config, task)
        payload["run_id"] = config["run_id"] + "-runtime-preflight"
        payload["ttl_seconds"] = min(int(payload["ttl_seconds"]), 1800)
        response = client.post(
            f"{ORCHESTRATOR}/v1/env/instances",
            headers={"X-Request-ID": str(uuid.uuid4())},
            json=payload,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Fleet runtime preflight create failed with HTTP {response.status_code}"
            )
        instance = response.json()
        instance_id = instance["instance_id"]
        actual = {
            "env_key": instance.get("env_key"),
            "environment_version": instance.get("version"),
            "data_key": instance.get("data_key"),
            "data_version": instance.get("data_version"),
        }
        expected = {
            "env_key": config["environment"]["id"],
            "environment_version": config["environment"]["version"],
            "data_key": config["environment"]["data_id"],
            "data_version": config["environment"]["data_version"],
        }
        if actual != expected:
            raise RuntimeError("runtime preflight environment/data binding drifted")
        token_payload = _request(client, "GET", "/v1/runner-auth/token")
        tool_names, tool_digest = discover_tools(
            instance["urls"]["root"], token_payload["header"], token_payload["token"]
        )
        return {
            "instance_id": instance_id,
            "runtime_binding": actual,
            "tool_names": tool_names,
            "tool_catalog_sha256": tool_digest,
        }
    finally:
        if instance_id:
            deleted = _request(client, "DELETE", f"/v1/env/instances/{instance_id}")
            terminated_at = deleted.get("terminated_at")
            if not terminated_at:
                raise RuntimeError("runtime preflight delete did not return termination evidence")


def run(config: dict[str, Any], out_dir: Path, proxy_script: Path) -> dict[str, Any]:
    api_key = os.environ.get("FLEET_API_KEY")
    qwen_image = os.environ.get("QWEN_CODE_IMAGE")
    proxy_image = os.environ.get("FIXED_PROXY_IMAGE")
    if not api_key or not qwen_image or not proxy_image:
        raise RuntimeError("FLEET_API_KEY, QWEN_CODE_IMAGE, and FIXED_PROXY_IMAGE are required")
    out_dir.mkdir(parents=True, exist_ok=False)
    out_dir.chmod(0o700)
    client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=1800,
    )
    instance_id = None
    network = config["execution"]["network"]
    suffix = hashlib.sha256(config["run_id"].encode()).hexdigest()[:8]
    model_proxy = f"qwen-model-proxy-{suffix}"
    mcp_proxy = f"qwen-mcp-proxy-{suffix}"
    qwen_agent = f"qwen-agent-{suffix}"
    cleanup: dict[str, Any] = {
        "instance_created": False,
        "instance_closed": False,
        "containers_removed": False,
    }
    started_at = time.time()
    try:
        account = _request(client, "GET", "/v1/account")
        if account.get("team_name") != "fleet" or account.get("team_id") not in {
            None,
            FLEET_TEAM_ID,
        }:
            raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
        task = load_and_verify_task(client, config)
        authority_gate = assert_authoritative_routes_deployed(client, config)
        binding = {
            "schema_version": config["schema_version"],
            "run_id": config["run_id"],
            "source_job_id": config["source_job_id"],
            "task": config["task"],
            "environment": config["environment"],
            "verifier": config["verifier"],
            "authority": config["authority"],
            "authority_gate": authority_gate,
            "model": config["model"],
            "harness": config["harness"],
        }
        (out_dir / "binding.json").write_bytes(canonical_json(binding) + b"\n")
        (out_dir / "prompt.txt").write_text(task["prompt"])

        response = client.post(
            f"{ORCHESTRATOR}{authoritative_route(config, 'provisioning')}",
            json={},
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Fleet authoritative instance create failed with HTTP {response.status_code}"
            )
        rollout_instance = response.json()
        instance_id = rollout_instance["instance_id"]
        cleanup["instance_created"] = True
        instance = _request(client, "GET", f"/v1/env/instances/{instance_id}")
        if (
            instance.get("env_key") != config["environment"]["id"]
            or instance.get("version") != config["environment"]["version"]
            or instance.get("data_key") != config["environment"]["data_id"]
            or instance.get("data_version") != config["environment"]["data_version"]
        ):
            raise RuntimeError("created instance does not match exact environment/data pins")
        token_payload = _request(client, "GET", "/v1/runner-auth/token")
        tool_names, tool_digest = discover_tools(
            instance["urls"]["root"], token_payload["header"], token_payload["token"]
        )
        (out_dir / "runtime-binding.json").write_bytes(
            canonical_json(
                {
                    "instance_id": instance_id,
                    "evidence_run_id": rollout_instance["evidence_run_id"],
                    "env_key": instance["env_key"],
                    "environment_version": instance["version"],
                    "data_key": instance["data_key"],
                    "data_version": instance["data_version"],
                    "tool_names": tool_names,
                    "tool_catalog_sha256": tool_digest,
                }
            )
            + b"\n"
        )

        write_json_once(
            out_dir / "resource-plan.json",
            {
                "schema_version": "fleet-selfhosted-resource-plan-v1",
                "run_id": config["run_id"],
                "instance_id": instance_id,
                "evidence_run_id": rollout_instance["evidence_run_id"],
                "containers": [qwen_agent, model_proxy, mcp_proxy],
                "network": network,
            },
        )

        _docker("network", "create", "--internal", network)
        proxy_mount = f"{proxy_script.resolve()}:/proxy.py:ro"
        _docker(
            "run", "-d", "--name", model_proxy, "--network", "bridge",
            "-e", "FIXED_UPSTREAM", "-e", "FIXED_AUTH_HEADER", "-e", "FIXED_AUTH_VALUE",
            "-e", "FIXED_PROXY_PORT", "-e", "FIXED_ALLOWED_PATHS",
            "-e", "FIXED_MAX_REQUESTS", "-e", "FIXED_MAX_REQUEST_BYTES",
            "-v", proxy_mount, proxy_image, "python", "/proxy.py",
            env={
                "FIXED_UPSTREAM": config["model"]["endpoint_origin"],
                "FIXED_AUTH_HEADER": "Authorization",
                "FIXED_AUTH_VALUE": f"Bearer {api_key}",
                "FIXED_PROXY_PORT": "8877",
                "FIXED_ALLOWED_PATHS": "/v1/chat/completions,/v1/models",
                "FIXED_MAX_REQUESTS": str(config["harness"]["max_model_requests"] + 30),
                "FIXED_MAX_REQUEST_BYTES": "16777216",
            },
        )
        _docker("network", "connect", "--alias", "model-proxy", network, model_proxy)
        wait_for_proxy(model_proxy, 8877)
        _docker(
            "run", "-d", "--name", mcp_proxy, "--network", "bridge",
            "-e", "FIXED_UPSTREAM", "-e", "FIXED_AUTH_HEADER", "-e", "FIXED_AUTH_VALUE",
            "-e", "FIXED_PROXY_PORT", "-e", "FIXED_ALLOWED_PATHS",
            "-e", "FIXED_MAX_REQUESTS", "-e", "FIXED_MAX_REQUEST_BYTES",
            "-v", proxy_mount, proxy_image, "python", "/proxy.py",
            env={
                "FIXED_UPSTREAM": instance["urls"]["root"].rstrip("/"),
                "FIXED_AUTH_HEADER": token_payload["header"],
                "FIXED_AUTH_VALUE": token_payload["token"],
                "FIXED_PROXY_PORT": "8090",
                "FIXED_ALLOWED_PATHS": "/mcp",
                "FIXED_MAX_REQUESTS": "4000",
                "FIXED_MAX_REQUEST_BYTES": "16777216",
            },
        )
        _docker("network", "connect", "--alias", "fleet-mcp-proxy", network, mcp_proxy)
        wait_for_proxy(mcp_proxy, 8090)

        qwen_home = out_dir / "qwen-home"
        qwen_home.mkdir(mode=0o700)
        agent_dir = out_dir / "agent-output"
        agent_dir.mkdir(mode=0o700)
        settings = {
            "security": {"auth": {"selectedType": "openai"}},
            "telemetry": {"enabled": False},
            "general": {"disableUpdateNag": True},
            "model": {"name": config["model"]["served_id"]},
            "modelProviders": {
                "openai": [
                    {
                        "id": config["model"]["served_id"],
                        "name": config["model"]["served_id"],
                        "baseUrl": "http://model-proxy:8877/v1",
                        "envKey": "QWEN_CODE_API_KEY",
                        "generationConfig": {"contextWindowSize": 262144},
                    }
                ]
            },
        }
        settings_path = qwen_home / "settings.json"
        settings_path.write_bytes(canonical_json(settings) + b"\n")
        # The pinned Node image's non-root user is uid/gid 1000. Give the agent only
        # its isolated home and output directory, never controller receipts or secrets.
        os.chown(qwen_home, 1000, 1000)
        os.chown(settings_path, 1000, 1000)
        os.chown(agent_dir, 1000, 1000)
        trace = agent_dir / "qwen-stream.jsonl"
        command = (
            "qwen mcp add fleet http://fleet-mcp-proxy:8090/mcp --transport http --trust "
            ">/tmp/mcp-add.log 2>&1 && "
            "qwen --yolo --output-format stream-json "
            f"--max-session-turns {config['harness']['max_model_requests']} "
            "< /input/prompt.txt > /output/qwen-stream.jsonl 2> /output/qwen-stderr.log"
        )
        agent_termination = "completed"
        try:
            result = _docker(
                "run", "--rm", "--name", qwen_agent, "--network", network,
                "-e", "OPENAI_API_KEY=local-proxy-only",
                "-e", "QWEN_CODE_API_KEY=local-proxy-only",
                "-e", "OPENAI_BASE_URL=http://model-proxy:8877/v1",
                "-e", f"OPENAI_MODEL={config['model']['served_id']}",
                "-v", f"{agent_dir.resolve()}:/output",
                "-v", f"{(out_dir / 'prompt.txt').resolve()}:/input/prompt.txt:ro",
                "-v", f"{qwen_home.resolve()}:/home/node/.qwen",
                qwen_image, "bash", "-lc", command,
                check=False,
                timeout=float(config["harness"]["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired:
            agent_termination = "execution_timeout"
            _docker("stop", "--time", "5", qwen_agent, check=False, capture=True, timeout=15)
            result = subprocess.CompletedProcess(args=["docker", "run"], returncode=124)
        events, canonical_trace, malformed_line_count = load_qwen_chat_trace(qwen_home)
        messages = normalize_qwen_conversation(events)
        final_answer = final_answer_from_conversation(messages)
        if not final_answer and trace.exists():
            final_answer = extract_final_answer(trace)
        (out_dir / "final-answer.txt").write_text(final_answer)
        trace_digest = sha256(canonical_trace.read_bytes())
        trace_fidelity = (
            "full_qwen_chat_normalized_with_tool_calls_and_observations"
            if malformed_line_count == 0
            else "raw_qwen_chat_canonical_with_partial_valid_json_normalization"
        )
        trace_manifest = {
            "canonical_trace": str(canonical_trace.relative_to(out_dir)),
            "canonical_trace_sha256": trace_digest,
            "qwen_event_count": len(events),
            "qwen_raw_line_count": len(events) + malformed_line_count,
            "qwen_malformed_line_count": malformed_line_count,
            "normalized_message_count": len(messages),
            "fidelity": trace_fidelity,
        }
        (out_dir / "trace-manifest.json").write_bytes(canonical_json(trace_manifest) + b"\n")
        scoring_payload = {
            "instance_id": instance_id,
            "final_answer": final_answer,
            "conversation": messages,
            "scoring_mode": config["authority"]["scoring_mode"],
            "multi_app_aggregation_mode": config["authority"]["multi_app_aggregation_mode"],
        }
        write_json_once(
            out_dir / "scoring-intent.json",
            {
                "schema_version": "fleet-selfhosted-scoring-intent-v1",
                "run_id": config["run_id"],
                "instance_id": instance_id,
                "evidence_run_id": rollout_instance["evidence_run_id"],
                "request_sha256": sha256(canonical_json(scoring_payload)),
            },
        )
        reward_result = _request(
            client,
            "POST",
            authoritative_route(config, "scoring"),
            json=scoring_payload,
        )
        (out_dir / "reward-result.json").write_bytes(canonical_json(reward_result) + b"\n")
        score = float(reward_result["reward"])
        session_metadata = {
            "self_hosted_harness": "qwen-code-0.22.3",
            "run_id": config["run_id"],
            "tool_catalog_sha256": tool_digest,
            "qwen_exit_code": result.returncode,
            "agent_termination": agent_termination,
            "trace_fidelity": trace_manifest["fidelity"],
            "canonical_trace_sha256": trace_digest,
            "training_data_eligible": False,
        }
        try:
            session_receipt = ingest_session_trace(
                client,
                messages=messages,
                config=config,
                instance_id=instance_id,
                score=score,
                verifier_execution_id=reward_result.get("verifier_execution_id"),
                metadata=session_metadata,
            )
        except SessionIngestError as exc:
            # Authoritative scoring is the primary experiment outcome.  Preserve
            # that valid outcome even if the ancillary Fleet dashboard trace
            # copy fails; the full canonical trace remains in this run bundle.
            session_receipt = exc.receipt
        (out_dir / "session-ingest.json").write_bytes(
            canonical_json(session_receipt) + b"\n"
        )
        result_record = {
            "run_id": config["run_id"],
            "session_id": session_receipt.get("session_id"),
            "session_ingest_status": session_receipt["status"],
            "score": score,
            "verifier_execution_id": reward_result.get("verifier_execution_id"),
            "qwen_exit_code": result.returncode,
            "agent_termination": agent_termination,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        (out_dir / "result.json").write_bytes(canonical_json(result_record) + b"\n")
        return result_record
    except BaseException as exc:
        if out_dir.exists():
            (out_dir / "failure.json").write_bytes(
                canonical_json(
                    {
                        "error_type": type(exc).__name__,
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "run_id": config["run_id"],
                    }
                )
                + b"\n"
            )
        raise
    finally:
        _docker("rm", "-f", qwen_agent, check=False, capture=True)
        for container in (model_proxy, mcp_proxy):
            logs = _docker("logs", container, check=False, capture=True)
            if out_dir.exists() and logs.stdout:
                (out_dir / f"{container}.jsonl").write_text(logs.stdout)
            _docker("rm", "-f", container, check=False, capture=True)
        _docker("network", "rm", network, check=False, capture=True)
        cleanup["containers_removed"] = True
        if instance_id:
            try:
                _request(client, "DELETE", f"/v1/env/instances/{instance_id}")
                cleanup["instance_closed"] = True
            except Exception as exc:  # noqa: BLE001
                cleanup["instance_close_error"] = type(exc).__name__
        if out_dir.exists():
            (out_dir / "cleanup.json").write_bytes(canonical_json(cleanup) + b"\n")
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "runtime-preflight", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy-script", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    if args.command == "preflight":
        with httpx.Client(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=180,
        ) as client:
            task = load_and_verify_task(client, config)
            authority_gate = assert_authoritative_routes_deployed(client, config)
            payload = build_instance_payload(config, task)
        print(
            json.dumps(
                {
                    "ok": True,
                    "run_id": config["run_id"],
                    "task_key": config["task"]["key"],
                    "environment": f"{payload['env_key']}:{payload['env_version']}",
                    "data": f"{payload['data_key']}:{payload['data_version']}",
                    "runtime_seed_file_count": len(payload["seed_overlay_files"]),
                    "verifier_version_id": config["verifier"]["version_id"],
                    "model_revision": config["model"]["revision"],
                    "harness_version": config["harness"]["version"],
                    "authority_gate": authority_gate,
                    "mutations": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "runtime-preflight":
        with httpx.Client(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=1800,
        ) as client:
            task = load_and_verify_task(client, config)
            result = runtime_preflight(client, config, task)
        result["cleanup_confirmed"] = True
        print(json.dumps(result, sort_keys=True))
        return 0
    if not args.out_dir or not args.proxy_script:
        parser.error("run requires --out-dir and --proxy-script")
    result = run(config, args.out_dir, args.proxy_script)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
