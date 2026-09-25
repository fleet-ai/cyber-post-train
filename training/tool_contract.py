"""Sanitize the *outbound* OpenCode model request at the model-proxy boundary.

Call ``capture_request`` on the bytes sent to ``/v1/chat/completions`` after
the fixed proxy applies its policy, not on an MCP catalog or a guessed schema.
The caller must separately attest the exact OpenCode image/release and proxy
run identity. A unit-test fixture is never evidence of a real request.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCHEMA = "cyber_opencode_model_request_tool_capture_v1"
RELEASE = "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702"
TOOLS = "sha256:585574ec1a459141a2e79f4945d140864876224ebef1260be65f06c6d237610f"
HARNESS = {"name": "opencode", "version": "1.18.27",
           "release_asset_sha256": RELEASE, "mcp_server": "fleet"}


def _digest(value: Any, *, ascii: bool = False) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii)
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate outbound JSON key")
        result[key] = value
    return result


def capture_request(raw: bytes, *, model_id: str, revision: str,
                    path: str = "/v1/chat/completions") -> dict:
    """Return only digest-bound tools and metadata; never retain messages.

    ``raw`` must be the final request body observed inside the fixed model
    proxy. The release hash is also the bound provider-transform binary hash;
    it is *not* a claim that source-catalog derivation observed the wire.
    """
    if path != "/v1/chat/completions" or not isinstance(raw, bytes) or len(raw) > 16_777_216:
        raise ValueError("not a bounded outbound chat request")
    if not isinstance(model_id, str) or not model_id or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("exact served model and revision required")
    try:
        request = json.loads(raw, object_pairs_hook=_unique,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeDecodeError, ValueError):
        raise ValueError("outbound request is not strict JSON") from None
    if not isinstance(request, dict) or request.get("model") != model_id:
        raise ValueError("outbound model differs")
    tools = request.get("tools")
    if (not isinstance(tools, list) or len(tools) != 2 or
            any(not isinstance(item, dict) or not isinstance(item.get("function"), dict)
                for item in tools) or
            [item["function"].get("name") for item in tools]
            != ["fleet_bash", "fleet_submit_report"] or _digest(tools, ascii=True) != TOOLS):
        raise ValueError("outbound Fleet tool definitions differ")
    result = {"schema": SCHEMA, "target_harness": HARNESS,
              "target_model": {"repo": "Qwen/Qwen3.8-27B", "revision": revision},
              "provider_schema_transform_sha256": RELEASE,
              "request_envelope_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
              "captured_tools": tools}
    result["sha256"] = _digest(result)
    return result


def _anchors(raw: bytes) -> list[tuple[str, str]]:
    request = json.loads(raw, object_pairs_hook=_unique)
    messages = request.get("messages")
    if not isinstance(messages, list):
        raise ValueError("outbound messages missing")

    def text_content(value: Any) -> str:
        if isinstance(value, str):
            return value
        if (isinstance(value, list) and value and
                all(isinstance(part, dict) and part.get("type") == "text" and
                    isinstance(part.get("text"), str) for part in value)):
            return "".join(part["text"] for part in value)
        raise ValueError("outbound anchor content is not text-only")

    anchors = [(message.get("role"), text_content(message.get("content")))
               for message in messages if isinstance(message, dict) and
               message.get("role") in {"system", "developer", "user"}]
    if not anchors or anchors[-1][0] != "user" or any(role == "user" for role, _ in anchors[:-1]):
        raise ValueError("first-request anchor roles differ")
    return anchors


def anchor_digests(raw: bytes, *, task_version_id: str, source_system_sha256: str,
                   source_task_sha256: str, model_id: str, revision: str) -> dict:
    """Compare private source anchors with the first real model request by hash.

    A mismatch is not silently rewritten: OpenCode may intentionally add its
    own system text or wrap the task prompt. The receipt records that fact
    without storing the text or guessing an equivalent training anchor.
    """
    tool_capture = capture_request(raw, model_id=model_id, revision=revision)
    if (not isinstance(task_version_id, str) or not task_version_id or
            any(not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in (source_system_sha256, source_task_sha256))):
        raise ValueError("exact source anchor binding required")
    anchors = _anchors(raw)
    system_digests = ["sha256:" + hashlib.sha256(content.encode()).hexdigest()
                      for role, content in anchors[:-1] if role in {"system", "developer"}]
    target_task = "sha256:" + hashlib.sha256(anchors[-1][1].encode()).hexdigest()
    result = {"schema": "cyber_opencode_anchor_digest_comparison_v1",
              "task_version_id": task_version_id,
              "request_envelope_sha256": tool_capture["request_envelope_sha256"],
              "source_system_sha256": source_system_sha256,
              "source_task_sha256": source_task_sha256,
              "target_system_message_sha256": system_digests,
              "target_task_sha256": target_task,
              "system_equal": len(system_digests) == 1 and source_system_sha256 == system_digests[0],
              "task_equal": source_task_sha256 == target_task}
    result["sha256"] = _digest(result)
    return result


def write_private_target_anchor(raw: bytes, *, model_id: str, revision: str,
                                task_version_id: str, prompt_path: Path,
                                output: Path) -> dict:
    """Save a real wire-observed target system/user pair outside Git only.

    OpenCode 1.18.27 ``run.ts`` quotes each CLI argument containing a literal
    space and escapes embedded double quotes. Bash command substitution first
    removes trailing newlines. Verify that transform against the wire, not a
    synthetic approximation.
    """
    capture = capture_request(raw, model_id=model_id, revision=revision)
    if (not task_version_id or not prompt_path.is_file() or not output.is_absolute() or
            output.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("private target-anchor path or identity invalid")
    prompt = prompt_path.read_text(encoding="utf-8")
    anchors = _anchors(raw)
    argument = prompt.rstrip("\n")
    expected = '"' + argument.replace('"', '\\"') + '"' if " " in argument else argument
    if anchors[-1] != ("user", expected):
        raise ValueError("OpenCode task-prompt wrapper differs")
    artifact = {"schema": "cyber_opencode_private_target_anchor_v1",
                "task_version_id": task_version_id,
                "request_envelope_sha256": capture["request_envelope_sha256"],
                "messages": [{"role": role, "content": content} for role, content in anchors]}
    artifact["sha256"] = _digest(artifact)
    data = (json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return {"schema": artifact["schema"], "task_version_id": task_version_id,
            "request_envelope_sha256": capture["request_envelope_sha256"],
            "target_anchor_sha256": artifact["sha256"],
            "source_task_sha256": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
            "target_system_message_sha256": ["sha256:" + hashlib.sha256(text.encode()).hexdigest()
                                             for role, text in anchors if role in {"system", "developer"}],
            "target_task_sha256": "sha256:" + hashlib.sha256(anchors[-1][1].encode()).hexdigest(),
            "verified_cli_argument_transform": True}


def write_once(path: Path, capture: dict) -> None:
    """Write only the sanitized receipt, private and create-once."""
    if capture.get("schema") != SCHEMA or capture.get("sha256") != _digest(
        {key: value for key, value in capture.items() if key != "sha256"}
    ):
        raise ValueError("unsealed request capture")
    data = (json.dumps(capture, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def fetch_exact_task_prompt(task_key: str, version_id: str, path: Path) -> str:
    """Privately fetch one exact Fleet task; return a digest, never its text.

    This is read-only and does not provision an environment or run a model.
    The output must be outside Git; the caller owns its prompt-file lifecycle.
    """
    key = os.environ.get("FLEET_API_KEY")
    if (not key or not re.fullmatch(r"[A-Za-z0-9_.-]+", task_key) or
            not re.fullmatch(r"[0-9a-f-]{36}", version_id) or not path.is_absolute() or
            path.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("private exact task fetch preflight failed")

    def get(route: str) -> dict:
        request = urllib.request.Request("https://orchestrator.fleetai.com" + route,
                                         headers={"Authorization": "Bearer " + key,
                                                  "Accept": "application/json"})
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_: Any) -> None:
                return None
        try:
            with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
                value = json.load(response)
        except Exception:
            raise ValueError("Fleet read-only task fetch failed") from None
        if not isinstance(value, dict):
            raise ValueError("Fleet read-only response malformed")
        return value

    account = get("/v1/account")
    if account.get("team_id") != "a1025f0b-ad67-49fc-a023-51800ab43e84":
        raise ValueError("Fleet team identity differs")
    route = ("/v1/tasks/" + urllib.parse.quote(task_key, safe="") + "?version_id=" +
             urllib.parse.quote(version_id, safe=""))
    task = get(route)
    prompt = task.get("prompt")
    if (task.get("key") != task_key or task.get("eval_task_version_id") != version_id or
            not isinstance(prompt, str) or not prompt):
        raise ValueError("Fleet exact task identity or prompt differs")
    data = prompt.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return "sha256:" + hashlib.sha256(data).hexdigest()
