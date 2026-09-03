"""Self-hosted agent runner for one exactly pinned Fleet cyber task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ORCHESTRATOR = "https://orchestrator.fleetai.com"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
RUNTIME_EVIDENCE_ONLY_V3 = "runtime_evidence_only_v3"
RUNTIME_EVIDENCE_ONLY_V3_SCORING_KEYS = (
    "instance_id",
    "multi_app_aggregation_mode",
    "scoring_mode",
)
DIRECT_AUTHORITY_ATTESTATION_SCHEMA = "fleet-direct-authority-attestation-v1"
TRANSIENT_READ_STATUS_CODES = {429, 502, 503, 504}
MAX_READ_ATTEMPTS = 6
SESSION_INGEST_CHUNK_MESSAGES = 32
SESSION_INGEST_CHUNK_BYTES = 512 * 1024


class SessionIngestError(RuntimeError):
    """A bounded trace ingest failed after zero or more recorded chunks."""

    def __init__(self, receipt: dict[str, Any]) -> None:
        super().__init__("Fleet session trace ingest did not complete")
        self.receipt = receipt


class FleetRequestError(RuntimeError):
    """A Fleet request failed, retaining only non-sensitive routing facts."""

    def __init__(self, method: str, route: str, status_code: int) -> None:
        super().__init__(f"Fleet {method} {route} failed with HTTP {status_code}")
        self.method = method
        self.route = route
        self.status_code = status_code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def session_model_identity(config: dict[str, Any]) -> str:
    explicit = (config.get("model") or {}).get("session_model")
    if isinstance(explicit, str) and explicit:
        return explicit
    served_id = config["model"]["served_id"]
    if (config.get("harness") or {}).get("name") == "qwen_code":
        return f"qwen/{served_id}"
    return f"self-hosted/{served_id}"


def persisted_session_model_identity(config: dict[str, Any]) -> str:
    """Return the model identity Fleet persists after removing a provider prefix."""
    return session_model_identity(config).split("/", 1)[-1]


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Durably claim a one-shot boundary before its external side effect."""
    payload = canonical_json(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _request(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    attempts = MAX_READ_ATTEMPTS if method == "GET" else 1
    for attempt in range(attempts):
        response = client.request(method, f"{ORCHESTRATOR}{path}", **kwargs)
        if response.status_code < 400:
            return response.json()
        if response.status_code not in TRANSIENT_READ_STATUS_CODES or attempt + 1 == attempts:
            route = path.split("?")[0]
            raise FleetRequestError(method, route, response.status_code)
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
        "cyber_contract": config["task"].get("cyber_contract"),
    }
    actual["cyber_contract"] = (task.get("metadata") or {}).get("cyber_contract")
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


def assert_required_task_tools(
    config: dict[str, Any], tool_names: list[str], tool_catalog_sha256: str
) -> None:
    """Fail closed if a calibrated task exposes a different tool surface."""
    required = (config.get("execution") or {}).get("required_task_tools")
    if required is None:
        return
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        raise RuntimeError("required task tools are malformed")
    if len(required) != len(set(required)):
        raise RuntimeError("required task tools contain duplicates")
    if tool_names != required:
        raise RuntimeError("runtime task tools do not match the exact required tool surface")
    required_digest = (config.get("execution") or {}).get("required_task_tool_catalog_sha256")
    if required_digest is not None and tool_catalog_sha256 != required_digest:
        raise RuntimeError("runtime task tool schemas do not match the exact required catalog")


def agent_container_user_args() -> list[str]:
    """Use the invoking uid/gid for Docker Desktop bind mounts."""
    if os.geteuid() == 0:
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


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


def load_opencode_trace(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read OpenCode's documented JSON event stream without accepting prose lines."""
    events: list[dict[str, Any]] = []
    malformed_line_count = 0
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed_line_count += 1
            continue
        if isinstance(value, dict):
            events.append(value)
    if not events:
        raise RuntimeError("OpenCode trace contained no JSON events")
    return events, malformed_line_count


def normalize_opencode_timestamp(value: Any) -> str | None:
    """Normalize OpenCode 1.18.27 millisecond timestamps to Fleet ISO-8601 strings."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("start") if value.get("start") is not None else value.get("end")
        if value is None:
            return None
    if isinstance(value, str):
        if not value:
            return None
        candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError("OpenCode timestamp string is not ISO-8601") from exc
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("OpenCode timestamp has an unsupported type")
    if not math.isfinite(float(value)):
        raise ValueError("OpenCode timestamp is not finite")
    try:
        timestamp = datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError("OpenCode timestamp is outside the supported range") from exc
    return timestamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def validate_session_messages(messages: list[dict[str, Any]]) -> None:
    """Fail before mutation when a normalized message violates Fleet's trace schema."""
    if not messages:
        raise ValueError("cannot ingest an empty session trace")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("session trace message requires a string role")
        timestamp = message.get("timestamp")
        if timestamp is not None and not isinstance(timestamp, str):
            raise ValueError("session trace timestamp must be an ISO-8601 string")
        if isinstance(timestamp, str):
            candidate = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
            try:
                datetime.fromisoformat(candidate)
            except ValueError as exc:
                raise ValueError("session trace timestamp must be ISO-8601") from exc


def normalize_opencode_conversation(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenCode JSON parts to Fleet's role/tool-call trace schema."""
    messages: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    seen_reasoning: set[str] = set()
    seen_tools: set[str] = set()
    pending_reasoning: list[str] = []
    for event in events:
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or event.get("type") or "").lower()
        message_id = str(part.get("messageID") or part.get("messageId") or "")
        part_id = str(part.get("id") or message_id or len(messages))
        raw_timestamp = event.get("timestamp")
        if raw_timestamp is None:
            raw_timestamp = part.get("time")
        timestamp = normalize_opencode_timestamp(raw_timestamp)
        if part_type in {"reasoning", "thinking"}:
            text = part.get("text") or part.get("content")
            if isinstance(text, str) and text and part_id not in seen_reasoning:
                pending_reasoning.append(text)
                seen_reasoning.add(part_id)
            continue
        if part_type == "text":
            text = part.get("text") or part.get("content")
            if isinstance(text, str) and text and part_id not in seen_text:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": text,
                    "timestamp": timestamp,
                }
                if pending_reasoning:
                    message["thinking"] = "\n".join(pending_reasoning)
                    pending_reasoning.clear()
                messages.append(message)
                seen_text.add(part_id)
            continue
        if part_type not in {"tool", "tool-call", "tool_call", "tool-invocation"}:
            continue
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        status = str(state.get("status") or "").lower()
        if status not in {"completed", "error", "failed"}:
            continue
        call_id = str(part.get("toolCallId") or state.get("id") or part_id)
        if call_id in seen_tools:
            continue
        name = str(
            part.get("tool")
            or part.get("name")
            or part.get("toolName")
            or state.get("name")
            or "tool"
        )
        arguments = state.get(
            "input", state.get("args", part.get("input", part.get("args", {})))
        )
        output = state.get(
            "output",
            state.get("result", state.get("error", part.get("output", part.get("result")))),
        )
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "timestamp": timestamp,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments or {}, sort_keys=True, ensure_ascii=True),
                    },
                }
            ],
        }
        if pending_reasoning:
            assistant["thinking"] = "\n".join(pending_reasoning)
            pending_reasoning.clear()
        messages.append(assistant)
        if not isinstance(output, str):
            output = json.dumps(output, sort_keys=True, ensure_ascii=True)
        messages.append(
            {
                "role": "tool",
                "content": output,
                "tool_call_id": call_id,
                "timestamp": timestamp,
                "metadata": {"opencode_tool_name": name, "is_error": status != "completed"},
            }
        )
        seen_tools.add(call_id)
    if not messages:
        raise RuntimeError("OpenCode trace normalized to no messages")
    validate_session_messages(messages)
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
    validate_session_messages(messages)
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
                    "model": session_model_identity(config),
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
            if isinstance(exc, FleetRequestError):
                receipt.update(
                    error_code="fleet_http_error",
                    http_status=exc.status_code,
                    method=exc.method,
                    route=exc.route,
                )
            raise SessionIngestError(receipt) from exc
        returned_id = response.get("session_id")
        expected_creation_state = index == 0
        if (
            response.get("success") is not True
            or response.get("message_count") != len(chunk)
            or response.get("created_new_session") is not expected_creation_state
        ):
            receipt.update(status="failed", error_type="ResponseBindingDrift")
            raise SessionIngestError(receipt)
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


def ingest_metadata_only_session(
    client: httpx.Client,
    *,
    config: dict[str, Any],
    instance_id: str,
    evidence_run_id: str,
    score: float,
    verifier_execution_id: str | None,
) -> dict[str, Any]:
    """Persist verifier-backed outcome metadata while storing zero model messages."""
    if config["authority"].get("scoring_payload_mode") != RUNTIME_EVIDENCE_ONLY_V3:
        raise ValueError("metadata-only session ingestion requires runtime-evidence-only v3")
    evidence_run_id = _nonzero_uuid(evidence_run_id, "metadata-only session evidence-run ID")
    if not isinstance(verifier_execution_id, str) or not verifier_execution_id:
        raise ValueError("metadata-only session ingestion requires a verifier execution ID")
    payload = {
        "messages": [],
        "model": session_model_identity(config),
        "task_key": config["task"]["key"],
        "eval_task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
        "score": score,
        "verifier_execution_id": verifier_execution_id,
    }
    response = _request(client, "POST", "/v1/sessions/ingest", json=payload)
    session_id = response.get("session_id")
    try:
        parsed_session_id = uuid.UUID(str(session_id))
    except ValueError as exc:
        raise RuntimeError("metadata-only session response has an invalid session ID") from exc
    if parsed_session_id.int == 0:
        raise RuntimeError("metadata-only session response has a zero session ID")
    if str(parsed_session_id) != evidence_run_id:
        raise RuntimeError("metadata-only session response is not bound to the evidence-run ID")
    response_score = response.get("score")
    if isinstance(response_score, bool):
        raise RuntimeError("metadata-only session response score is invalid")
    try:
        exact_score = float(response_score)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("metadata-only session response score is invalid") from exc
    expected = {
        "success": True,
        "evidence_only": True,
        "trace_persisted": False,
        "message_count": 0,
        "model": session_model_identity(config),
        "task_key": config["task"]["key"],
        "eval_task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
        "verifier_execution_id": verifier_execution_id,
    }
    if any(response.get(key) != value for key, value in expected.items()) or exact_score != score:
        raise RuntimeError("metadata-only session response binding drifted")
    if (
        response.get("success") is not True
        or response.get("evidence_only") is not True
        or response.get("trace_persisted") is not False
        or type(response.get("message_count")) is not int
        or not isinstance(response.get("created_new_session"), bool)
    ):
        raise RuntimeError("metadata-only session response creation state is invalid")
    return {
        "status": "completed",
        "mode": "metadata_only_runtime_evidence_v1",
        "success": True,
        "evidence_only": True,
        "trace_persisted": False,
        "created_new_session": response["created_new_session"],
        "session_id": str(parsed_session_id),
        "evidence_run_id": evidence_run_id,
        "message_count": 0,
        "chunks_completed": 1,
        "chunk_count": 1,
        "score": exact_score,
        "model": expected["model"],
        "verifier_execution_id": verifier_execution_id,
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
    }


def authoritative_route(config: dict[str, Any], kind: str) -> str:
    template = config["authority"][f"{kind}_route_template"]
    return template.format(
        task_key=config["task"]["key"], task_version_id=config["task"]["version_id"]
    )


def provisioning_request_id(config: dict[str, Any]) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fleet-qwen-provision:{config['run_id']}"))


def _nonzero_uuid(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not a UUID") from exc
    if parsed.int == 0:
        raise RuntimeError(f"{label} is a zero UUID")
    return str(parsed)


def _instance_identifier(value: Any) -> str:
    """Validate the opaque, DNS-safe identifier returned by Fleet environments."""
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", value
    ):
        raise RuntimeError("Fleet authoritative instance ID is not a DNS-safe identifier")
    return value


def validate_rollout_instance_response(config: dict[str, Any], response: Any) -> tuple[str, str]:
    """Bind provisioning output to the exact route inputs before agent execution."""
    if not isinstance(response, dict):
        raise RuntimeError("Fleet authoritative instance response is not an object")
    if (
        response.get("task_key") != config["task"]["key"]
        or response.get("task_version_id") != config["task"]["version_id"]
    ):
        raise RuntimeError("Fleet authoritative instance response task binding drifted")
    instance_id = _instance_identifier(response.get("instance_id"))
    evidence_run_id = _nonzero_uuid(
        response.get("evidence_run_id"), "Fleet authoritative evidence-run ID"
    )
    return instance_id, evidence_run_id


def validate_scoring_payload(config: dict[str, Any], payload: Any) -> None:
    """Reject hidden legacy content channels in the direct-authority v3 request."""
    if not isinstance(payload, dict):
        raise RuntimeError("Fleet scoring payload is not an object")
    mode = config["authority"].get("scoring_payload_mode")
    if mode != RUNTIME_EVIDENCE_ONLY_V3:
        return
    if tuple(sorted(payload)) != RUNTIME_EVIDENCE_ONLY_V3_SCORING_KEYS:
        raise RuntimeError("runtime-evidence-only v3 scoring payload contains unsupported fields")


def sanitize_authoritative_reward_response(
    config: dict[str, Any],
    response: Any,
    *,
    instance_id: str,
    evidence_run_id: str,
) -> dict[str, Any]:
    """Validate v3 authority cross-links and retain only non-sensitive attestations."""
    if not isinstance(response, dict):
        raise RuntimeError("Fleet authoritative reward response is not an object")
    expected = {
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "instance_id": instance_id,
    }
    if any(response.get(field) != value for field, value in expected.items()):
        raise RuntimeError("Fleet authoritative reward response binding drifted")
    reward = response.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise RuntimeError("Fleet authoritative reward is not numeric")
    reward = float(reward)
    if not (0.0 <= reward <= 1.0) or not math.isfinite(reward):
        raise RuntimeError("Fleet authoritative reward is outside [0, 1]")
    execution_id = _nonzero_uuid(
        response.get("verifier_execution_id"),
        "Fleet authoritative verifier execution ID",
    )
    sanitized = {
        **expected,
        "reward": reward,
        "verifier_execution_id": execution_id,
    }
    if config["authority"].get("scoring_payload_mode") != RUNTIME_EVIDENCE_ONLY_V3:
        return sanitized

    structured = response.get("cyber_verification_result")
    shadow = response.get("cyber_evidence")
    if not isinstance(structured, dict) or not isinstance(shadow, dict):
        raise RuntimeError("Fleet direct-authority attestation is missing")
    structured_reward = structured.get("reward")
    bindings = structured.get("bindings")
    direct = shadow.get("direct_verifier")
    required_contract = config["authority"].get("required_cyber_contract") or {}
    if (
        structured.get("schema_version") != "cyber_verification_result_v3"
        or isinstance(structured_reward, bool)
        or not isinstance(structured_reward, (int, float))
        or float(structured_reward) != reward
        or not isinstance(bindings, dict)
        or bindings.get("task_version_id") != config["task"]["version_id"]
        or shadow.get("mode") != "authoritative"
        or shadow.get("status") != "authoritative"
        or shadow.get("match") is not True
        or shadow.get("production_execution_id") != execution_id
        or not isinstance(direct, dict)
        or direct.get("status") != "authoritative"
        or direct.get("match") is not True
        or direct.get("execution_id") != execution_id
        or direct.get("verifier_contract_version") != required_contract.get("verifier_contract")
        or direct.get("context_schema_version") != "cyber_verification_context_v1"
    ):
        raise RuntimeError("Fleet direct-authority attestation binding drifted")
    sanitized["direct_authority_attestation"] = {
        "schema_version": DIRECT_AUTHORITY_ATTESTATION_SCHEMA,
        "context": {
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "verifier_version_id": config["verifier"]["version_id"],
            "scoring_payload_mode": RUNTIME_EVIDENCE_ONLY_V3,
        },
        "activity": {
            "result_schema_version": structured["schema_version"],
            "reward": reward,
            "task_version_id": bindings["task_version_id"],
            "verifier_execution_id": execution_id,
        },
        "shadow": {
            "mode": shadow["mode"],
            "status": shadow["status"],
            "match": True,
            "production_execution_id": execution_id,
            "direct_verifier": {
                "status": direct["status"],
                "match": True,
                "execution_id": execution_id,
                "verifier_contract_version": direct["verifier_contract_version"],
                "context_schema_version": direct["context_schema_version"],
            },
        },
        "data_minimization": {
            "components_included": False,
            "diagnostics_included": False,
            "evidence_payloads_included": False,
            "prompts_included": False,
            "traces_included": False,
            "flags_included": False,
        },
    }
    return sanitized


def build_scoring_payload(
    config: dict[str, Any],
    *,
    instance_id: str,
    final_answer: str,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "instance_id": instance_id,
        "scoring_mode": config["authority"]["scoring_mode"],
        "multi_app_aggregation_mode": config["authority"]["multi_app_aggregation_mode"],
    }
    if config["authority"].get("scoring_payload_mode") != RUNTIME_EVIDENCE_ONLY_V3:
        payload.update(final_answer=final_answer, conversation=messages)
    validate_scoring_payload(config, payload)
    return payload


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
        supported_shape_guard = any(
            marker in detail
            for marker in (
                "report-only",
                "exact black-box capability tasks",
            )
        )
        if response.status_code != 422 or not supported_shape_guard:
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
        assert_required_task_tools(config, tool_names, tool_digest)
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


def run(
    config: dict[str, Any],
    out_dir: Path,
    proxy_script: Path,
    *,
    safe_scoring_intent_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("FLEET_API_KEY")
    harness_name = str(config.get("harness", {}).get("name") or "")
    agent_image = os.environ.get("AGENT_HARNESS_IMAGE") or os.environ.get("QWEN_CODE_IMAGE")
    proxy_image = os.environ.get("FIXED_PROXY_IMAGE")
    if harness_name not in {"qwen_code", "opencode"}:
        raise RuntimeError("self-hosted harness must be qwen_code or opencode")
    if not api_key or not agent_image or not proxy_image:
        raise RuntimeError(
            "FLEET_API_KEY, AGENT_HARNESS_IMAGE, and FIXED_PROXY_IMAGE are required"
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    out_dir.chmod(0o700)
    client = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=1800,
    )
    instance_id = None
    network = config["execution"]["network"]
    suffix = hashlib.sha256(config["run_id"].encode()).hexdigest()[:8]
    model_proxy = f"agent-model-proxy-{suffix}"
    mcp_proxy = f"agent-mcp-proxy-{suffix}"
    agent_container = f"agent-runtime-{suffix}"
    cleanup: dict[str, Any] = {
        "instance_created": False,
        "instance_closed": False,
        "containers_removed": False,
    }
    started_at = time.time()
    try:
        account = _request(client, "GET", "/v1/account")
        if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
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
            headers={"X-Request-ID": provisioning_request_id(config)},
            json={},
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Fleet authoritative instance create failed with HTTP {response.status_code}"
            )
        rollout_instance = response.json()
        if isinstance(rollout_instance, dict):
            # Bind this before validating any other response field so that a
            # later contract failure cannot orphan an already-created instance.
            instance_id = _instance_identifier(rollout_instance.get("instance_id"))
        instance_id, evidence_run_id = validate_rollout_instance_response(config, rollout_instance)
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
        assert_required_task_tools(config, tool_names, tool_digest)
        runtime_binding = {
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "env_key": instance["env_key"],
            "environment_version": instance["version"],
            "data_key": instance["data_key"],
            "data_version": instance["data_version"],
            "tool_names": tool_names,
            "tool_catalog_sha256": tool_digest,
        }
        if config["environment"].get("version_id") is not None:
            runtime_binding["environment_version_id"] = config["environment"]["version_id"]
        (out_dir / "runtime-binding.json").write_bytes(canonical_json(runtime_binding) + b"\n")

        resource_plan = {
            "schema_version": "fleet-selfhosted-resource-plan-v1",
            "run_id": config["run_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "containers": [agent_container, model_proxy, mcp_proxy],
            "network": network,
        }
        resource_plan["resource_plan_sha256"] = sha256(canonical_json(resource_plan))
        write_json_once(out_dir / "resource-plan.json", resource_plan)

        _docker("network", "create", "--internal", network)
        proxy_mount = f"{proxy_script.resolve()}:/proxy.py:ro"
        _docker(
            "run",
            "-d",
            "--name",
            model_proxy,
            "--network",
            "bridge",
            "-e",
            "FIXED_UPSTREAM",
            "-e",
            "FIXED_AUTH_HEADER",
            "-e",
            "FIXED_AUTH_VALUE",
            "-e",
            "FIXED_PROXY_PORT",
            "-e",
            "FIXED_ALLOWED_PATHS",
            "-e",
            "FIXED_MAX_REQUESTS",
            "-e",
            "FIXED_MAX_REQUEST_BYTES",
            "-v",
            proxy_mount,
            proxy_image,
            "python",
            "/proxy.py",
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
            "run",
            "-d",
            "--name",
            mcp_proxy,
            "--network",
            "bridge",
            "-e",
            "FIXED_UPSTREAM",
            "-e",
            "FIXED_AUTH_HEADER",
            "-e",
            "FIXED_AUTH_VALUE",
            "-e",
            "FIXED_PROXY_PORT",
            "-e",
            "FIXED_ALLOWED_PATHS",
            "-e",
            "FIXED_MAX_REQUESTS",
            "-e",
            "FIXED_MAX_REQUEST_BYTES",
            "-v",
            proxy_mount,
            proxy_image,
            "python",
            "/proxy.py",
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

        agent_home = out_dir / f"{harness_name}-home"
        agent_home.mkdir(mode=0o700)
        agent_dir = out_dir / "agent-output"
        agent_dir.mkdir(mode=0o700)
        if harness_name == "opencode":
            config_dir = agent_home / ".config" / "opencode"
            config_dir.mkdir(parents=True, mode=0o700)
            plugin_path = config_dir / "fleet-disable-compaction-autocontinue.mjs"
            plugin_path.write_text(
                "export const DisableCompactionAutocontinue = async () => ({\n"
                '  "experimental.compaction.autocontinue": async (_input, output) => '
                "{ output.enabled = false; },\n"
                "});\n"
            )
            model_id = config["model"]["served_id"]
            settings = {
                "$schema": "https://opencode.ai/config.json",
                "plugin": [
                    "file:///home/node/.config/opencode/fleet-disable-compaction-autocontinue.mjs"
                ],
                "provider": {
                    "fleet-cluster": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Fleet cluster inference",
                        "options": {
                            "baseURL": "http://model-proxy:8877/v1",
                            "apiKey": "local-proxy-only",
                            "timeout": False,
                            "chunkTimeout": 300000,
                        },
                        "models": {
                            model_id: {
                                "name": model_id,
                                "reasoning": True,
                                "tool_call": True,
                                "interleaved": "reasoning_content",
                                "limit": {
                                    "context": int(config["harness"]["context_window_size"]),
                                    "output": int(config["harness"]["max_output_tokens"]),
                                },
                            }
                        },
                    }
                },
                "mcp": {
                    "fleet": {
                        "type": "remote",
                        "url": "http://fleet-mcp-proxy:8090/mcp",
                        "enabled": True,
                    }
                },
                "permission": {"*": "deny", "fleet_*": "allow"},
                "tools": {
                    "bash": False,
                    "edit": False,
                    "read": False,
                    "glob": False,
                    "grep": False,
                    "list": False,
                    "task": False,
                    "webfetch": False,
                    "websearch": False,
                    "skill": False,
                },
            }
            settings_path = config_dir / "opencode.json"
            trace = agent_dir / "opencode-stream.jsonl"
            command = (
                "opencode run --format json --thinking "
                f"--model fleet-cluster/{model_id} --dir /workspace --auto -- "
                '"$(</input/prompt.txt)" '
                "> /output/opencode-stream.jsonl 2> /output/opencode-stderr.log"
            )
            home_mount = f"{agent_home.resolve()}:/home/node"
        else:
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
                            "generationConfig": {
                                "contextWindowSize": config["harness"].get(
                                    "context_window_size", 262144
                                )
                            },
                        }
                    ]
                },
            }
            settings_path = agent_home / "settings.json"
            trace = agent_dir / "qwen-stream.jsonl"
            command = (
                "qwen mcp add fleet http://fleet-mcp-proxy:8090/mcp --transport http --trust "
                ">/tmp/mcp-add.log 2>&1 && "
                "qwen --yolo --output-format stream-json "
                f"--max-session-turns {config['harness']['max_model_requests']} "
                "< /input/prompt.txt > /output/qwen-stream.jsonl 2> /output/qwen-stderr.log"
            )
            home_mount = f"{agent_home.resolve()}:/home/node/.qwen"
        settings_path.write_bytes(canonical_json(settings) + b"\n")
        # The pinned Node image's non-root user is uid/gid 1000. Give the agent only
        # its isolated home and output directory, never controller receipts or secrets.
        agent_user_args = agent_container_user_args()
        if os.geteuid() == 0:
            for directory in [agent_home, *agent_home.rglob("*")]:
                os.chown(directory, 1000, 1000)
            os.chown(settings_path, 1000, 1000)
            os.chown(agent_dir, 1000, 1000)
        # Docker Desktop preserves host ownership on bind mounts. A non-root
        # controller therefore runs the agent as its own uid/gid rather than
        # attempting a privileged chown; the global qwen binary remains pinned.
        agent_termination = "completed"
        try:
            result = _docker(
                "run",
                "--rm",
                "--name",
                agent_container,
                "--network",
                network,
                *agent_user_args,
                "-e",
                "OPENAI_API_KEY=local-proxy-only",
                "-e",
                "QWEN_CODE_API_KEY=local-proxy-only",
                "-e",
                "OPENAI_BASE_URL=http://model-proxy:8877/v1",
                "-e",
                f"OPENAI_MODEL={config['model']['served_id']}",
                "-v",
                f"{agent_dir.resolve()}:/output",
                "-v",
                f"{(out_dir / 'prompt.txt').resolve()}:/input/prompt.txt:ro",
                "-v",
                home_mount,
                agent_image,
                "bash",
                "-lc",
                command,
                check=False,
                timeout=float(config["harness"]["timeout_seconds"]),
            )
        except subprocess.TimeoutExpired:
            agent_termination = "execution_timeout"
            _docker(
                "stop", "--time", "5", agent_container, check=False, capture=True, timeout=15
            )
            result = subprocess.CompletedProcess(args=["docker", "run"], returncode=124)
        if harness_name == "opencode":
            canonical_trace = trace
            events, malformed_line_count = load_opencode_trace(trace)
            messages = normalize_opencode_conversation(events)
            trace_fidelity = (
                "full_opencode_json_normalized_with_tool_calls_and_observations"
                if malformed_line_count == 0
                else "raw_opencode_json_with_partial_valid_json_normalization"
            )
        else:
            events, canonical_trace, malformed_line_count = load_qwen_chat_trace(agent_home)
            messages = normalize_qwen_conversation(events)
            trace_fidelity = (
                "full_qwen_chat_normalized_with_tool_calls_and_observations"
                if malformed_line_count == 0
                else "raw_qwen_chat_canonical_with_partial_valid_json_normalization"
            )
        final_answer = final_answer_from_conversation(messages)
        if not final_answer and trace.exists():
            final_answer = extract_final_answer(trace)
        (out_dir / "final-answer.txt").write_text(final_answer)
        trace_digest = sha256(canonical_trace.read_bytes())
        trace_manifest = {
            "canonical_trace": str(canonical_trace.relative_to(out_dir)),
            "canonical_trace_sha256": trace_digest,
            "harness": harness_name,
            "event_count": len(events),
            "raw_line_count": len(events) + malformed_line_count,
            "malformed_line_count": malformed_line_count,
            "normalized_message_count": len(messages),
            "fidelity": trace_fidelity,
        }
        (out_dir / "trace-manifest.json").write_bytes(canonical_json(trace_manifest) + b"\n")
        scoring_payload = build_scoring_payload(
            config,
            instance_id=instance_id,
            final_answer=final_answer,
            messages=messages,
        )
        scoring_intent = {
            "schema_version": "fleet-selfhosted-scoring-intent-v1",
            "run_id": config["run_id"],
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "scoring_payload_mode": config["authority"].get("scoring_payload_mode"),
            "request_keys": sorted(scoring_payload),
            "request_sha256": sha256(canonical_json(scoring_payload)),
        }
        scoring_intent["scoring_intent_sha256"] = sha256(canonical_json(scoring_intent))
        write_json_once(out_dir / "scoring-intent.json", scoring_intent)
        if safe_scoring_intent_sink is not None:
            safe_scoring_intent_sink(scoring_intent)
        reward_response = _request(
            client,
            "POST",
            authoritative_route(config, "scoring"),
            json=scoring_payload,
        )
        reward_result = sanitize_authoritative_reward_response(
            config,
            reward_response,
            instance_id=instance_id,
            evidence_run_id=evidence_run_id,
        )
        (out_dir / "reward-result.json").write_bytes(canonical_json(reward_result) + b"\n")
        score = float(reward_result["reward"])
        try:
            if config["authority"].get("scoring_payload_mode") == RUNTIME_EVIDENCE_ONLY_V3:
                session_receipt = ingest_metadata_only_session(
                    client,
                    config=config,
                    instance_id=instance_id,
                    evidence_run_id=evidence_run_id,
                    score=score,
                    verifier_execution_id=reward_result.get("verifier_execution_id"),
                )
            else:
                session_receipt = ingest_session_trace(
                    client,
                    messages=messages,
                    config=config,
                    instance_id=instance_id,
                    score=score,
                    verifier_execution_id=reward_result.get("verifier_execution_id"),
                    metadata={
                        "self_hosted_harness": (
                            f"{harness_name}-{config['harness']['version']}"
                        ),
                        "run_id": config["run_id"],
                        "tool_catalog_sha256": tool_digest,
                        "agent_exit_code": result.returncode,
                        "agent_termination": agent_termination,
                        "trace_fidelity": trace_manifest["fidelity"],
                        "canonical_trace_sha256": trace_digest,
                        "training_data_eligible": bool(
                            config["execution"].get("training_data_eligible", False)
                        ),
                    },
                )
        except SessionIngestError as exc:
            # Preserve the single-shot mutation receipt for terminal diagnosis;
            # callers still classify incomplete ingestion as infrastructure-invalid.
            session_receipt = exc.receipt
        (out_dir / "session-ingest.json").write_bytes(canonical_json(session_receipt) + b"\n")
        result_record = {
            "run_id": config["run_id"],
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": instance_id,
            "evidence_run_id": evidence_run_id,
            "session_id": session_receipt.get("session_id"),
            "session_ingest_status": session_receipt["status"],
            "score": score,
            "verifier_execution_id": reward_result.get("verifier_execution_id"),
            "agent_exit_code": result.returncode,
            "harness": harness_name,
            "agent_termination": agent_termination,
            "elapsed_seconds": round(time.time() - started_at, 3),
        }
        if harness_name == "qwen_code":
            result_record["qwen_exit_code"] = result.returncode
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
        _docker("rm", "-f", agent_container, check=False, capture=True)
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


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required recovery input is not a regular file: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"required recovery input is not an object: {path.name}")
    return value


def _recovery_source(
    config: dict[str, Any], source_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate immutable post-score source evidence without exposing its content."""
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise RuntimeError("recovery source must be a real directory")
    if config.get("config_sha256") != digest_without(config, "config_sha256"):
        raise RuntimeError("recovery config digest mismatch")

    binding = _read_json_object(source_dir / "binding.json")
    result = _read_json_object(source_dir / "result.json")
    reward = _read_json_object(source_dir / "reward-result.json")
    runtime = _read_json_object(source_dir / "runtime-binding.json")
    original_ingest = _read_json_object(source_dir / "session-ingest.json")
    cleanup = _read_json_object(source_dir / "cleanup.json")
    scoring_intent = _read_json_object(source_dir / "scoring-intent.json")
    trace_manifest = _read_json_object(source_dir / "trace-manifest.json")

    for field in ("run_id", "task", "environment", "verifier", "model", "harness"):
        if binding.get(field) != config.get(field):
            raise RuntimeError(f"recovery source binding drifted at {field}")
    expected_result = {
        "run_id": config["run_id"],
        "task_key": config["task"]["key"],
        "task_version_id": config["task"]["version_id"],
        "harness": "opencode",
        "agent_exit_code": 0,
        "agent_termination": "completed",
    }
    if any(result.get(field) != value for field, value in expected_result.items()):
        raise RuntimeError("recovery result identity or completion state drifted")
    instance_id = _instance_identifier(result.get("instance_id"))
    evidence_run_id = _nonzero_uuid(result.get("evidence_run_id"), "recovery evidence-run ID")
    verifier_execution_id = _nonzero_uuid(
        result.get("verifier_execution_id"), "recovery verifier execution ID"
    )
    if (
        runtime.get("instance_id") != instance_id
        or runtime.get("evidence_run_id") != evidence_run_id
        or runtime.get("tool_names") != config["execution"]["required_task_tools"]
        or runtime.get("tool_catalog_sha256")
        != config["execution"]["required_task_tool_catalog_sha256"]
    ):
        raise RuntimeError("recovery runtime binding drifted")
    if (
        reward.get("task_key") != config["task"]["key"]
        or reward.get("task_version_id") != config["task"]["version_id"]
        or reward.get("instance_id") != instance_id
        or reward.get("verifier_execution_id") != verifier_execution_id
    ):
        raise RuntimeError("recovery authoritative reward binding drifted")
    score = reward.get("reward")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
        or result.get("score") != score
    ):
        raise RuntimeError("recovery authoritative reward value drifted")
    if original_ingest.get("status") != "failed" or any(
        (
            original_ingest.get("session_id") is not None,
            original_ingest.get("chunks_completed") != 0,
            result.get("session_id") is not None,
            result.get("session_ingest_status") != "failed",
        )
    ):
        raise RuntimeError("recovery is allowed only after a zero-chunk ingest failure")
    if cleanup.get("instance_created") is not True or cleanup.get("instance_closed") is not True:
        raise RuntimeError("recovery source instance cleanup is incomplete")
    if cleanup.get("containers_removed") is not True:
        raise RuntimeError("recovery source container cleanup is incomplete")
    if scoring_intent.get("scoring_intent_sha256") != digest_without(
        scoring_intent, "scoring_intent_sha256"
    ):
        raise RuntimeError("recovery scoring intent digest mismatch")
    if any(
        (
            scoring_intent.get("run_id") != config["run_id"],
            scoring_intent.get("task_key") != config["task"]["key"],
            scoring_intent.get("task_version_id") != config["task"]["version_id"],
            scoring_intent.get("instance_id") != instance_id,
            scoring_intent.get("evidence_run_id") != evidence_run_id,
        )
    ):
        raise RuntimeError("recovery scoring intent binding drifted")

    trace_name = trace_manifest.get("canonical_trace")
    if not isinstance(trace_name, str):
        raise RuntimeError("recovery trace manifest lacks a canonical path")
    relative_trace = Path(trace_name)
    if relative_trace.is_absolute() or ".." in relative_trace.parts:
        raise RuntimeError("recovery trace path escapes its source root")
    trace_path = source_dir / relative_trace
    if trace_path.is_symlink() or not trace_path.is_file():
        raise RuntimeError("recovery canonical trace is not a regular file")
    trace_digest = sha256(trace_path.read_bytes())
    if trace_manifest.get("canonical_trace_sha256") != trace_digest:
        raise RuntimeError("recovery canonical trace digest mismatch")
    events, malformed_line_count = load_opencode_trace(trace_path)
    messages = normalize_opencode_conversation(events)
    if any(
        (
            trace_manifest.get("harness") != "opencode",
            trace_manifest.get("event_count") != len(events),
            trace_manifest.get("raw_line_count") != len(events) + malformed_line_count,
            trace_manifest.get("malformed_line_count") != malformed_line_count,
            trace_manifest.get("normalized_message_count") != len(messages),
        )
    ):
        raise RuntimeError("recovery trace manifest counts drifted")

    source = {
        "instance_id": instance_id,
        "evidence_run_id": evidence_run_id,
        "verifier_execution_id": verifier_execution_id,
        "score": float(score),
        "trace_sha256": trace_digest,
        "trace_fidelity": trace_manifest.get("fidelity"),
        "tool_catalog_sha256": runtime["tool_catalog_sha256"],
        "source_result_sha256": sha256((source_dir / "result.json").read_bytes()),
        "source_reward_sha256": sha256((source_dir / "reward-result.json").read_bytes()),
        "original_ingest_sha256": sha256((source_dir / "session-ingest.json").read_bytes()),
    }
    return messages, source


def _task_sessions(client: httpx.Client, task_key: str) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = _request(
            client,
            "GET",
            "/v1/sessions",
            params={"task_key": task_key, "limit": 500, "offset": offset},
        )
        page = response.get("sessions") or []
        if not isinstance(page, list):
            raise RuntimeError("Fleet session inventory returned an invalid page")
        sessions.extend(row for row in page if isinstance(row, dict))
        if response.get("has_more") is False:
            return sessions
        if not page:
            raise RuntimeError("Fleet session inventory pagination made no progress")
        offset += len(page)


def _matching_recovery_sessions(
    sessions: list[dict[str, Any]], config: dict[str, Any], verifier_execution_id: str
) -> list[dict[str, Any]]:
    persisted_model = persisted_session_model_identity(config)
    return [
        session
        for session in sessions
        if session.get("model") == persisted_model
        and isinstance(session.get("verifier_execution"), dict)
        and session["verifier_execution"].get("id") == verifier_execution_id
    ]


def recover_session_trace(
    client: httpx.Client,
    *,
    config: dict[str, Any],
    source_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Create one Fleet session from an immutable, post-score failed trace source."""
    out_dir.mkdir(parents=True, exist_ok=False)
    out_dir.chmod(0o700)
    try:
        account = _request(client, "GET", "/v1/account")
        if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
            raise RuntimeError("FLEET_API_KEY is not scoped to the Fleet team")
        messages, source = _recovery_source(config, source_dir)
        before = _task_sessions(client, config["task"]["key"])
        if _matching_recovery_sessions(before, config, source["verifier_execution_id"]):
            raise RuntimeError("equivalent verifier-backed Fleet session already exists")

        intent = {
            "schema_version": "fleet-opencode-session-recovery-intent-v1",
            "run_id": config["run_id"],
            "config_sha256": config["config_sha256"],
            "session_model": session_model_identity(config),
            "persisted_session_model": persisted_session_model_identity(config),
            "task_key": config["task"]["key"],
            "task_version_id": config["task"]["version_id"],
            "instance_id": source["instance_id"],
            "evidence_run_id": source["evidence_run_id"],
            "verifier_execution_id": source["verifier_execution_id"],
            "message_count": len(messages),
            "canonical_trace_sha256": source["trace_sha256"],
            "source_result_sha256": source["source_result_sha256"],
            "source_reward_sha256": source["source_reward_sha256"],
            "original_ingest_sha256": source["original_ingest_sha256"],
            "preflight_equivalent_sessions": 0,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        intent["intent_sha256"] = digest_without(intent, "intent_sha256")
        write_json_once(out_dir / "RECOVERY-INTENT.json", intent)

        try:
            ingest = ingest_session_trace(
                client,
                messages=messages,
                config=config,
                instance_id=source["instance_id"],
                score=source["score"],
                verifier_execution_id=source["verifier_execution_id"],
                metadata={
                    "self_hosted_harness": f"opencode-{config['harness']['version']}",
                    "run_id": config["run_id"],
                    "tool_catalog_sha256": source["tool_catalog_sha256"],
                    "agent_exit_code": 0,
                    "agent_termination": "completed",
                    "trace_fidelity": source["trace_fidelity"],
                    "canonical_trace_sha256": source["trace_sha256"],
                    "training_data_eligible": bool(
                        config["execution"].get("training_data_eligible", False)
                    ),
                    "session_recovery": "post_score_timestamp_schema_v1",
                },
            )
        except SessionIngestError as exc:
            failed_ingest = {**exc.receipt, "scores_included": False}
            write_json_once(out_dir / "session-ingest.json", failed_ingest)
            raise
        write_json_once(out_dir / "session-ingest.json", ingest)

        after = _task_sessions(client, config["task"]["key"])
        matches = _matching_recovery_sessions(after, config, source["verifier_execution_id"])
        if len(matches) != 1 or matches[0].get("session_id") != ingest["session_id"]:
            raise RuntimeError(
                "recovered Fleet session did not reconcile by exact verifier identity"
            )
        if matches[0].get("status") != "completed":
            raise RuntimeError("recovered Fleet session is not completed")
        recovered = {
            "schema_version": "fleet-opencode-session-recovered-v1",
            "recovered": True,
            "run_id": config["run_id"],
            "session_id": ingest["session_id"],
            "verifier_execution_id": source["verifier_execution_id"],
            "message_count": len(messages),
            "chunk_count": ingest["chunk_count"],
            "canonical_trace_sha256": source["trace_sha256"],
            "intent_sha256": intent["intent_sha256"],
            "authoritative_score_attached": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        recovered["receipt_sha256"] = digest_without(recovered, "receipt_sha256")
        write_json_once(out_dir / "RECOVERED.json", recovered)
        return recovered
    except BaseException as exc:
        failure_path = out_dir / "failure.json"
        if not failure_path.exists():
            failure: dict[str, Any] = {
                "schema_version": "fleet-opencode-session-recovery-failure-v1",
                "error_type": type(exc).__name__,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
            if isinstance(exc, FleetRequestError):
                failure.update(
                    http_status=exc.status_code, method=exc.method, route=exc.route
                )
            failure["receipt_sha256"] = digest_without(failure, "receipt_sha256")
            write_json_once(failure_path, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("preflight", "runtime-preflight", "run", "recover-session")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--proxy-script", type=Path)
    parser.add_argument("--source-dir", type=Path)
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
    if args.command == "recover-session":
        if not args.source_dir or not args.out_dir:
            parser.error("recover-session requires --source-dir and --out-dir")
        with httpx.Client(
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=1800,
        ) as client:
            recovered = recover_session_trace(
                client, config=config, source_dir=args.source_dir, out_dir=args.out_dir
            )
        print(
            json.dumps(
                {
                    "recovered": recovered["recovered"],
                    "run_id": recovered["run_id"],
                    "session_id": recovered["session_id"],
                    "message_count": recovered["message_count"],
                    "scores_included": False,
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.out_dir or not args.proxy_script:
        parser.error("run requires --out-dir and --proxy-script")
    result = run(config, args.out_dir, args.proxy_script)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
