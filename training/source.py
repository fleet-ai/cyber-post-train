"""Fetch exact Fleet teacher successes into a private, create-once corpus source.

The caller supplies digests for the selected-session file, reviewed family
roles, and an *actual* OpenCode model-request tool capture. This module never
prints or commits session content. It cannot manufacture the tool capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


API = "https://orchestrator.fleetai.com"
TEAM = "a1025f0b-ad67-49fc-a023-51800ab43e84"
TOOL_DIGEST = "sha256:585574ec1a459141a2e79f4945d140864876224ebef1260be65f06c6d237610f"
TARGET_SYSTEM_DIGEST = "sha256:28a674ed99cf78ded8ee2d4773316d2e89f6b459253dadb01b275c3d47796d91"
ANCHOR_METHOD = "opencode_1_18_27_target_anchor_substitution_v1"
MIN_ANCHOR_PROBES = 2
HARNESS = {"name": "opencode", "version": "1.18.27", "release_asset_sha256":
           "sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702",
           "mcp_server": "fleet"}
ALIASES = {"bash": "fleet_bash", "submit_report": "fleet_submit_report",
           "fleet_bash": "fleet_bash", "fleet_submit_report": "fleet_submit_report",
           "fleet_environment__bash": "fleet_bash",
           "fleet_environment__fleet_environment__bash": "fleet_bash",
           "mcp__fleet_environment__bash": "fleet_bash",
           "fleet_environment__submit_report": "fleet_submit_report",
           "mcp__fleet_environment__submit_report": "fleet_submit_report",
           "mcp__fleet_environment__mcp__fleet_environment__submit_report": "fleet_submit_report"}
OLD_ANCHOR = re.compile(
    r"`(?:bash|submit_report|use_tool|search_tool)`|"
    r"\b(?:call|invoke|use)\s+(?:the\s+)?(?:bash|submit_report|use_tool|search_tool)\s+tool\b|"
    r'"(?:name|tool_name)"\s*:\s*"(?:bash|submit_report|use_tool|search_tool)"',
    re.IGNORECASE,
)


class SourceError(ValueError):
    """Safe, content-free source-contract failure."""


def digest(value: Any, *, ascii: bool = False) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii)
    return "sha256:" + hashlib.sha256(body.encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def text_digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _bound_file(binding: Any, *, jsonl: bool = False) -> Any:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"} or not _sha(binding["sha256"]):
        raise SourceError("input needs an exact path and SHA-256")
    path = Path(binding["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or file_digest(path) != binding["sha256"]:
        raise SourceError("bound input is absent or changed")
    try:
        with path.open(encoding="utf-8") as stream:
            value = [json.loads(line) for line in stream if line.strip()] if jsonl else json.load(stream)
    except (OSError, ValueError):
        raise SourceError("bound input is not valid JSON") from None
    return value


def _sealed(value: Any, schema: str) -> None:
    if (not isinstance(value, dict) or value.get("schema") != schema
            or value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"})):
        raise SourceError("sealed input identity differs")


def _request(path: str) -> dict:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise SourceError("FLEET_API_KEY is unavailable")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_: Any) -> None:
            return None

    opener = urllib.request.build_opener(NoRedirect)
    request = urllib.request.Request(API + path, headers={"Accept": "application/json",
                                                        "Authorization": "Bearer " + key})
    for attempt in range(3):
        try:
            with opener.open(request, timeout=120) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise SourceError("Fleet GET returned non-object JSON")
            return value
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504}:
                raise SourceError(f"Fleet GET failed (HTTP {error.code})") from None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        if attempt < 2:
            time.sleep(0.5 * (2 ** attempt))
    raise SourceError("Fleet GET failed after three attempts")


def _summary(get: Callable[[str], dict], task_key: str, session_id: str,
             cache: dict[str, dict[str, dict]]) -> dict:
    if task_key not in cache:
        found: dict[str, dict] = {}
        offset = 0
        while True:
            path = "/v1/sessions?" + urllib.parse.urlencode({"task_key": task_key, "limit": 100, "offset": offset})
            page = get(path)
            rows = page.get("sessions")
            if not isinstance(rows, list) or not isinstance(page.get("has_more"), bool):
                raise SourceError("Fleet session listing is incomplete")
            for item in rows:
                sid = item.get("session_id") if isinstance(item, dict) else None
                if not isinstance(sid, str) or not sid or sid in found:
                    raise SourceError("Fleet session listing has duplicate or invalid identity")
                found[sid] = item
            if not page["has_more"]:
                break
            if not rows:
                raise SourceError("Fleet session listing did not advance")
            offset += len(rows)
        cache[task_key] = found
    if session_id not in cache[task_key]:
        raise SourceError("selected session missing from Fleet task listing")
    return cache[task_key][session_id]


def _tool_kind(call: Any) -> str | None:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return None
    name, arguments = function.get("name"), function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return None
    if name == "use_tool":
        return (ALIASES.get(arguments.get("tool_name"))
                if isinstance(arguments, dict) and isinstance(arguments.get("tool_input"), dict)
                else None)
    if name == "shell" and isinstance(arguments, dict) and set(arguments) == {"command"}:
        return "fleet_bash"
    return ALIASES.get(name) if isinstance(name, str) else None


def _report_call(messages: list[dict]) -> str | None:
    calls, results = [], set()
    for message in messages:
        if not isinstance(message, dict):
            return None
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if _tool_kind(call) == "fleet_submit_report":
                    calls.append(call.get("id"))
        elif message.get("role") == "tool" and isinstance(message.get("tool_call_id"), str):
            results.add(message["tool_call_id"])
    return calls[0] if len(calls) == 1 and isinstance(calls[0], str) and calls[0] in results else None


def _tool_operations(messages: list[dict], end: int) -> tuple[list[dict], list[dict]] | None:
    """Rewrite only proven source calls into exact target calls, preserving results."""
    operations, target_messages, pending, seen = [], [], {}, set()
    for index, message in enumerate(messages[:end + 1]):
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if (set(message) != {"role", "tool_call_id", "content"}
                    or not isinstance(message["content"], str)
                    or not isinstance(call_id, str) or call_id not in pending):
                return None
            pending.pop(call_id)["result_sha256"] = digest(message)
            target_messages.append(dict(message))
            continue
        if pending or role != "assistant":
            if pending:
                return None
            target_messages.append(dict(message))
            continue
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            return None
        target_calls = []
        for call in calls:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                return None
            call_id, function = call.get("id"), call["function"]
            if (not isinstance(call_id, str) or not call_id or call_id in seen
                    or set(call) - {"id", "type", "index", "function"}
                    or call.get("type", "function") != "function"
                    or "index" in call and type(call["index"]) is not int
                    or set(function) != {"name", "arguments"}):
                return None
            seen.add(call_id)
            name, args = function.get("name"), function.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    return None
            if name == "use_tool":
                # Its result is a source-harness wrapper, not OpenCode MCP text.
                return None
            elif name == "shell":
                if not isinstance(args, dict) or set(args) != {"command"}:
                    return None
                name, args = "bash", {"script": args["command"]}
            target = ALIASES.get(name) if isinstance(name, str) else None
            if target == "fleet_bash":
                valid = (isinstance(args, dict) and set(args) <= {"script", "timeoutMs"}
                         and isinstance(args.get("script"), str) and bool(args["script"])
                         and ("timeoutMs" not in args or type(args["timeoutMs"]) is int
                              and 1 <= args["timeoutMs"] <= 300000))
            elif target == "fleet_submit_report":
                valid = (isinstance(args, dict) and set(args) <= {"explanation", "flag", "flags", "verdict"}
                         and isinstance(args.get("explanation"), str)
                         and ("flag" not in args or isinstance(args["flag"], str))
                         and ("flags" not in args or isinstance(args["flags"], list)
                              and all(isinstance(flag, str) for flag in args["flags"]))
                         and ("verdict" not in args or args["verdict"] == "no_flag"
                              and "flag" not in args and "flags" not in args))
            else:
                return None
            if not valid:
                return None
            operation = {"call_id_sha256": digest(call_id),
                         "source_message_index": index,
                               "source_function_sha256": digest(function),
                               "target_function_sha256": digest({"name": target, "arguments": args})}
            operations.append(operation)
            pending[call_id] = operation
            target_calls.append({"id": call_id, "type": "function",
                                 "function": {"name": target, "arguments": args}})
        target_message = dict(message)
        if calls:
            target_message["tool_calls"] = target_calls
        target_messages.append(target_message)
    return (operations, target_messages) if not pending else None


def _target_user(prompt: str) -> str:
    # Exact OpenCode 1.18.27 `run` argv construction after shell substitution.
    argument = prompt.rstrip("\n")
    return '"' + argument.replace('"', '\\"') + '"' if " " in argument else argument


def _exact_discovery_metadata(content: Any, tools: list[dict]) -> bool:
    try:
        value = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        return False
    if (not isinstance(value, dict) or set(value) != {"note", "results", "status", "total_hidden_tools"}
            or value["note"] is not None or value["status"] != "ready"
            or type(value["total_hidden_tools"]) is not int or not isinstance(value["results"], list)):
        return False
    expected = {"fleet_environment__" + tool["function"]["name"].removeprefix("fleet_"):
                tool["function"] for tool in tools}
    for group in value["results"]:
        if (not isinstance(group, dict) or set(group) != {"server", "tools"}
                or group["server"] != "fleet_environment" or not isinstance(group["tools"], list)):
            return False
        for tool in group["tools"]:
            if (not isinstance(tool, dict) or set(tool) != {"description", "input_schema", "score", "tool_name"}
                    or not isinstance(tool["tool_name"], str) or tool["tool_name"] not in expected
                    or type(tool["score"]) not in {int, float}
                    or not math.isfinite(tool["score"])):
                return False
            target = expected[tool["tool_name"]]
            if (tool["description"] != target["description"] or
                    tool["input_schema"] != {key: val for key, val in target["parameters"].items()
                                             if key != "additionalProperties"}):
                return False
    return True


def _elide_exact_discovery(messages: list[dict], end: int, tools: list[dict]) -> tuple[list[dict], dict] | None:
    """Drop only up-front, state-free discovery of tools already in the target anchor."""
    out, removed = [], []
    index, executable = 0, False
    helper_reference = re.compile(r"\b(?:search_tool|ToolSearch|submit_final_answer)\b", re.IGNORECASE)
    while index <= end:
        message = messages[index]
        calls = message.get("tool_calls") or []
        if message.get("role") == "assistant" and any(
            _tool_kind(call) in {"fleet_bash", "fleet_submit_report"} for call in calls
        ):
            executable = True
        search = [call for call in calls if isinstance(call, dict)
                  and isinstance(call.get("function"), dict)
                  and call["function"].get("name") == "search_tool"]
        if search:
            ids = [call.get("id") for call in search]
            results = messages[index + 1:index + 1 + len(search)]
            content = message.get("content")
            if (executable or len(search) != len(calls)
                    or not all(isinstance(x, str) and x for x in ids)
                    or len(set(ids)) != len(ids)
                    or content is not None and not isinstance(content, str)
                    or helper_reference.search(content or "")
                    or len(results) != len(search)
                    or not all(isinstance(result, dict)
                               and isinstance(result.get("tool_call_id"), str) for result in results)
                    or {result["tool_call_id"] for result in results} != set(ids)
                    or any(result.get("role") != "tool" or
                           not _exact_discovery_metadata(result.get("content"), tools)
                           for result in results)):
                return None
            removed.extend({"call_sha256": digest(call),
                            "result_sha256": digest(next(result for result in results
                                                          if result.get("tool_call_id") == call["id"]))}
                           for call in search)
            if message.get("content"):
                out.append({key: val for key, val in message.items() if key != "tool_calls"})
            index += 1 + len(search)
            continue
        out.append(message)
        index += 1
    if not removed:
        return None
    out.extend(messages[end + 1:])
    if any(helper_reference.search(message.get("content") or "") for message in out[2:]
           if message.get("role") == "assistant" and isinstance(message.get("content"), str)):
        return None
    transform = {"schema": "fleet_exact_target_tool_discovery_elision_v1",
                 "method": "exact_target_tool_discovery_elision_v1",
                 "source_messages_sha256": digest(messages),
                 "output_messages_sha256": digest(out),
                 "removed_calls": removed}
    transform["sha256"] = digest(transform)
    return out, transform


def _text_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(
        isinstance(part, dict) and set(part) == {"type", "text"}
        and part["type"] == "text" and isinstance(part["text"], str) for part in value
    ):
        return "".join(part["text"] for part in value)
    return None


def _training_messages(messages: list[dict]) -> list[dict] | None:
    """Use visible content only; keep hidden reasoning solely in the raw envelope."""
    out = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            return None
        converted = {key: value for key, value in message.items()
                     if key not in {"thinking", "reasoning", "reasoning_content", "analysis"}}
        if index < 2:
            text = _text_content(message.get("content"))
            if not text:
                return None
            converted["content"] = text
        out.append(converted)
    return out


def _write(path: Path, value: Any, *, lines: bool = False) -> str:
    data = ("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
                    for row in value) if lines else
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return file_digest(path)


def _source_rows(binding: dict) -> list[dict]:
    rows = _bound_file(binding, jsonl=True)
    seen = set()
    if not isinstance(rows, list) or not rows:
        raise SourceError("source selection is empty")
    for row in rows:
        if not isinstance(row, dict):
            raise SourceError("source selection row is malformed")
        sid, key, version = (row.get(name) for name in ("session_id", "task_key", "task_version_id"))
        if (not all(isinstance(x, str) and x for x in (sid, key, version, row.get("model_id")))
                or not all(_sha(row.get(name)) for name in ("trace_sha256", "harness_sha256",
                                                             "acceptance_sha256"))
                or sid in seen):
            raise SourceError("source selection lacks exact identity")
        seen.add(sid)
    return sorted(rows, key=lambda row: row["session_id"])


def _validate_envelope(row: dict, summary: dict, envelope: dict) -> None:
    if not isinstance(envelope, dict):
        raise SourceError("selected transcript envelope is malformed")
    sid, key, version = (row[name] for name in ("session_id", "task_key", "task_version_id"))
    verifier = envelope.get("verifier_execution")
    score = verifier.get("score") if isinstance(verifier, dict) else None
    harness, task, instance = (envelope.get(name) for name in ("harness", "task", "instance"))
    if (not all(isinstance(x, dict) for x in (verifier, harness, task, instance, summary))
            or digest(envelope) != row["trace_sha256"]
            or summary.get("session_id") != sid or summary.get("model") != row["model_id"]
            or summary.get("task_key") != key or task.get("key") != key
            or task.get("eval_task_version_id") != version or instance.get("team_id") != TEAM
            or harness.get("mode") != row.get("harness_mode")
            or digest(harness) != row["harness_sha256"]
            or summary.get("verifier_execution") != verifier
            or summary.get("status") not in {"completed", "succeeded", "success"}
            or not isinstance(verifier.get("id"), str) or not verifier["id"]
            or verifier.get("success") is not True or type(score) not in {int, float}
            or not math.isfinite(score) or score < 1):
        raise SourceError("selected source changed or lacks authoritative success")


def _validate_raw(row: dict, envelope: dict) -> None:
    if not isinstance(envelope, dict):
        raise SourceError("selected transcript envelope is malformed")
    verifier, harness, task, instance = (envelope.get(name) for name in
                                         ("verifier_execution", "harness", "task", "instance"))
    score = verifier.get("score") if isinstance(verifier, dict) else None
    if (not all(isinstance(x, dict) for x in (verifier, harness, task, instance))
            or digest(envelope) != row["trace_sha256"]
            or task.get("key") != row["task_key"]
            or task.get("eval_task_version_id") != row["task_version_id"]
            or instance.get("team_id") != TEAM
            or harness.get("mode") != row.get("harness_mode")
            or digest(harness) != row["harness_sha256"]
            or not isinstance(verifier.get("id"), str) or not verifier["id"]
            or verifier.get("success") is not True or type(score) not in {int, float}
            or not math.isfinite(score) or score < 1):
        raise SourceError("selected transcript changed or lacks successful verifier")


def hydrate(request: dict, *, get: Callable[[str], dict] = _request,
            max_new: int | None = None) -> dict:
    """Resume exact private source download; no trainer-readiness decision here."""
    _sealed(request, "fleet_teacher_source_hydration_v1")
    if set(request) != {"schema", "selection", "output", "sha256"}:
        raise SourceError("hydration request fields differ")
    rows = _source_rows(request["selection"])
    if max_new is not None and (type(max_new) is not int or max_new < 1):
        raise SourceError("max_new must be a positive integer")
    output = Path(request["output"])
    if not output.is_absolute() or output.is_symlink():
        raise SourceError("hydration output must be an absolute private directory")
    if output.exists():
        if (not output.is_dir() or output.stat().st_mode & 0o077
                or not (output / "HYDRATE_REQUEST.json").is_file()
                or json.loads((output / "HYDRATE_REQUEST.json").read_text()) != request):
            raise SourceError("existing hydration directory has another identity or unsafe mode")
    else:
        output.mkdir(mode=0o700, parents=False, exist_ok=False)
        os.chmod(output, 0o700)
        _write(output / "HYDRATE_REQUEST.json", request)
        (output / "sessions").mkdir(mode=0o700)
    sessions = output / "sessions"
    if sessions.is_symlink() or not sessions.is_dir() or sessions.stat().st_mode & 0o077:
        raise SourceError("hydration session directory is unsafe")
    account = get("/v1/account")
    if account.get("team_id") != TEAM or account.get("team_name") != "fleet":
        raise SourceError("Fleet account is not the authorized team")
    existing = 0
    missing = []
    for row in rows:
        sid = row["session_id"]
        path = sessions / (hashlib.sha256(sid.encode()).hexdigest() + ".json")
        if path.exists():
            if path.is_symlink() or path.stat().st_mode & 0o077:
                raise SourceError("existing hydrated source file is unsafe")
            try:
                saved = json.loads(path.read_text())
            except (OSError, ValueError):
                raise SourceError("existing hydrated source file is unreadable") from None
            if (not isinstance(saved, dict) or saved.get("schema") != "fleet_teacher_hydrated_session_v1"
                    or saved.get("selection") != row):
                raise SourceError("existing hydrated source identity differs")
            _validate_raw(row, saved.get("transcript_envelope"))
            if saved.get("summary") is not None:
                _validate_envelope(row, saved["summary"], saved["transcript_envelope"])
            existing += 1
            continue
        missing.append((row, path))

    def download(row: dict, path: Path) -> None:
        envelope = get("/v1/sessions/" + urllib.parse.quote(row["session_id"], safe="") + "/transcript")
        _validate_raw(row, envelope)
        with tempfile.NamedTemporaryFile(dir=sessions, prefix=".partial-", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            temporary_path.unlink()
            _write(temporary_path, {"schema": "fleet_teacher_hydrated_session_v1", "selection": row,
                                    "transcript_envelope": envelope})
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    selected_missing = missing[:max_new]
    added = 0
    with ThreadPoolExecutor(max_workers=min(4, len(selected_missing) or 1)) as pool:
        futures = [pool.submit(download, row, path) for row, path in selected_missing]
        for future in as_completed(futures):
            future.result()
            added += 1
    complete = existing + added == len(rows)
    if file_digest(Path(request["selection"]["path"])) != request["selection"]["sha256"]:
        raise SourceError("source selection changed during hydration")
    result = {"selected_sessions": len(rows), "hydrated_sessions": existing + added,
              "new_sessions": added, "complete": complete}
    if complete:
        seal = {"schema": "fleet_teacher_hydration_receipt_v1",
                "request_sha256": request["sha256"],
                "selection_sha256": request["selection"]["sha256"],
                "sessions_sha256": digest({path.name: file_digest(path) for path in sessions.glob("*.json")}),
                "session_count": len(rows)}
        seal["sha256"] = digest(seal)
        receipt = output / "HYDRATED.json"
        if receipt.exists():
            if json.loads(receipt.read_text()) != seal:
                raise SourceError("hydration receipt changed")
        else:
            _write(receipt, seal)
        result["receipt_sha256"] = seal["sha256"]
    return result


def verify_hydration_cache(root: Path, selection_path: Path,
                           expected_receipt_file_sha: str) -> dict:
    """Read-only exact cache attestation, including every private envelope."""
    root, selection_path = Path(root), Path(selection_path)
    receipt_path, request_path, sessions = (root / name for name in
                                            ("HYDRATED.json", "HYDRATE_REQUEST.json", "sessions"))
    if (not _sha(expected_receipt_file_sha) or not root.is_absolute()
            or root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o077
            or sessions.is_symlink() or not sessions.is_dir() or sessions.stat().st_mode & 0o077
            or any(path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077
                   for path in (receipt_path, request_path))
            or file_digest(receipt_path) != expected_receipt_file_sha):
        raise SourceError("private hydration cache is absent, changed, or unsafe")
    try:
        receipt, request = json.loads(receipt_path.read_text()), json.loads(request_path.read_text())
    except (OSError, ValueError):
        raise SourceError("private hydration cache has invalid JSON") from None
    _sealed(receipt, "fleet_teacher_hydration_receipt_v1")
    _sealed(request, "fleet_teacher_source_hydration_v1")
    if (receipt.get("request_sha256") != request["sha256"]
            or receipt.get("selection_sha256") != request.get("selection", {}).get("sha256")):
        raise SourceError("private hydration request and receipt disagree")
    rows = _source_rows({"path": str(selection_path), "sha256": receipt["selection_sha256"]})
    names = {hashlib.sha256(row["session_id"].encode()).hexdigest() + ".json" for row in rows}
    files = list(sessions.iterdir())
    if (receipt.get("session_count") != len(rows) or {path.name for path in files} != names
            or any(path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077
                   for path in files)
            or digest({path.name: file_digest(path) for path in files}) != receipt.get("sessions_sha256")):
        raise SourceError("private hydration session set is incomplete or changed")
    for row in rows:
        path = sessions / (hashlib.sha256(row["session_id"].encode()).hexdigest() + ".json")
        try:
            saved = json.loads(path.read_text())
        except (OSError, ValueError):
            raise SourceError("private hydrated session has invalid JSON") from None
        if (not isinstance(saved, dict) or saved.get("schema") != "fleet_teacher_hydrated_session_v1"
                or saved.get("selection") != row):
            raise SourceError("private hydrated session identity differs")
        _validate_raw(row, saved.get("transcript_envelope"))
    return {"selected_sessions": len(rows), "selection_sha256": receipt["selection_sha256"],
            "hydration_receipt_file_sha256": expected_receipt_file_sha,
            "sessions_sha256": receipt["sessions_sha256"]}


def fetch(request: dict, *, get: Callable[[str], dict] = _request) -> dict:
    """Download only digest-bound, verified successes; receipt contains no content."""
    _sealed(request, "fleet_teacher_source_fetch_v1")
    required = {"schema", "selection", "hydration", "roles", "tool_capture", "target_anchors",
                "model_revision", "output", "sha256"}
    if (set(request) != required or not isinstance(request["model_revision"], str)
            or not re.fullmatch(r"[0-9a-f]{40}", request["model_revision"])):
        raise SourceError("source request fields or model revision differ")
    selection = _source_rows(request["selection"])
    hydration = _bound_file(request["hydration"])
    _sealed(hydration, "fleet_teacher_hydration_receipt_v1")
    hydrated_root = Path(request["hydration"]["path"]).parent
    hydrated_files = hydrated_root / "sessions"
    verified = verify_hydration_cache(hydrated_root, Path(request["selection"]["path"]),
                                      request["hydration"]["sha256"])
    if (verified["selected_sessions"] != len(selection)
            or verified["selection_sha256"] != request["selection"]["sha256"]):
        raise SourceError("private hydration and selection disagree")
    roles = _bound_file(request["roles"])
    capture = _bound_file(request["tool_capture"])
    _sealed(roles, "fleet_teacher_family_roles_v1")
    _sealed(capture, "cyber_opencode_model_request_tool_capture_v1")
    bindings = request["target_anchors"]
    if not isinstance(bindings, list) or len(bindings) < MIN_ANCHOR_PROBES:
        raise SourceError("two exact private target-anchor probes are required")
    anchors = [_bound_file(binding) for binding in bindings]
    probes = {}
    for anchor in anchors:
        _sealed(anchor, "cyber_opencode_private_target_anchor_v1")
        version, messages = anchor.get("task_version_id"), anchor.get("messages")
        if (not isinstance(version, str) or not version or version in probes
                or not isinstance(messages, list) or len(messages) != 2
                or [m.get("role") for m in messages if isinstance(m, dict)] != ["system", "user"]
                or any(not isinstance(m.get("content"), str) for m in messages)
                or text_digest(messages[0]["content"]) != TARGET_SYSTEM_DIGEST):
            raise SourceError("private target-anchor probe is malformed or differs")
        probes[version] = messages[1]["content"]
    target_system = anchors[0]["messages"][0]["content"]
    tools = capture.get("captured_tools")
    if (capture.get("target_harness") != HARNESS or
            capture.get("target_model") != {"repo": "Qwen/Qwen3.8-27B", "revision": request["model_revision"]} or
            not _sha(capture.get("request_envelope_sha256")) or
            not _sha(capture.get("provider_schema_transform_sha256")) or
            not isinstance(tools, list) or
            [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)] !=
            ["fleet_bash", "fleet_submit_report"] or digest(tools, ascii=True) != TOOL_DIGEST):
        raise SourceError("actual OpenCode model-request tool capture is absent or incompatible")
    identities = roles.get("identities")
    if not isinstance(identities, list):
        raise SourceError("reviewed role roster is malformed")
    roster, families = {}, {}
    for item in identities:
        if not isinstance(item, dict) or set(item) != {"task_key", "task_version_id", "family_id", "split"}:
            raise SourceError("reviewed role identity is malformed")
        version, family, split = item["task_version_id"], item["family_id"], item["split"]
        if (not all(isinstance(x, str) and x for x in (item["task_key"], version, family))
                or split not in {"train", "dev", "test", "final_test"}
                or version in roster or families.setdefault(family, split) != split):
            raise SourceError("reviewed family roles conflict")
        roster[version] = {"task_key": item["task_key"], "family_id": family,
                           "split": "test" if split == "final_test" else split}
    selected, seen = [], set()
    for item in selection:
        if not isinstance(item, dict):
            raise SourceError("source selection row is malformed")
        sid, key, version = (item.get(name) for name in ("session_id", "task_key", "task_version_id"))
        if (not all(isinstance(x, str) and x for x in (sid, key, version, item.get("model_id")))
                or not _sha(item.get("trace_sha256")) or not _sha(item.get("acceptance_sha256"))
                or not isinstance(item.get("group_id"), str) or not item["group_id"]
                or sid in seen or
                roster.get(version, {}).get("task_key") != key):
            raise SourceError("source selection lacks exact reviewed identity")
        seen.add(sid)
        selected.append(item)
    account = get("/v1/account")
    if account.get("team_id") != TEAM or account.get("team_name") != "fleet":
        raise SourceError("Fleet account is not the authorized team")
    output = Path(request["output"])
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise SourceError("output must be a new absolute private directory")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    os.chmod(output, 0o700)
    request_file_sha256 = _write(output / "REQUEST.json", request)
    records, proofs, target_records, target_proofs, raw_sources = [], {}, [], [], []
    quarantine, exclusions, summaries = [], Counter(), {}
    probe_versions, old_anchor_mentions = set(), 0
    try:
        for item in sorted(selected, key=lambda row: row["session_id"]):
            sid, key, version = item["session_id"], item["task_key"], item["task_version_id"]
            if roster[version]["split"] == "test":
                exclusions["final_test"] += 1
                continue
            private = hydrated_files / (hashlib.sha256(sid.encode()).hexdigest() + ".json")
            if private.is_symlink() or private.stat().st_mode & 0o077:
                raise SourceError("hydrated session file is unsafe")
            saved = json.loads(private.read_text())
            if saved.get("schema") != "fleet_teacher_hydrated_session_v1" or saved.get("selection") != item:
                raise SourceError("hydrated session identity differs")
            envelope = saved.get("transcript_envelope")
            _validate_raw(item, envelope)
            summary = saved.get("summary") or _summary(get, key, sid, summaries)
            verifier = envelope.get("verifier_execution")
            score = verifier.get("score") if isinstance(verifier, dict) else None
            messages = envelope.get("transcript")
            if (not isinstance(verifier, dict) or not isinstance(envelope.get("task"), dict)
                    or not isinstance(envelope.get("instance"), dict)
                    or digest(envelope) != item["trace_sha256"] or summary.get("session_id") != sid
                    or summary.get("model") != item["model_id"] or summary.get("task_key") != key
                    or envelope.get("task", {}).get("key") != key
                    or envelope.get("task", {}).get("eval_task_version_id") != version
                    or envelope.get("instance", {}).get("team_id") != TEAM
                    or envelope.get("harness", {}).get("mode") != item.get("harness_mode")
                    or digest(envelope.get("harness")) != item.get("harness_sha256")
                    or summary.get("verifier_execution") != verifier
                    or summary.get("status") not in {"completed", "succeeded", "success"}
                    or not isinstance(verifier.get("id"), str) or not verifier["id"]
                    or verifier.get("success") is not True or type(score) not in {int, float}
                    or not math.isfinite(score) or score < 1):
                raise SourceError("selected source changed or lacks authoritative success")
            raw_sources.append({"selection": item, "summary": summary, "transcript_envelope": envelope})
            if (not isinstance(messages, list) or len(messages) < 4 or
                    [m.get("role") for m in messages[:2] if isinstance(m, dict)] != ["system", "user"] or
                    not all(_text_content(m.get("content")) for m in messages[:2])):
                exclusions["missing_original_anchor"] += 1
                quarantine.append({"selection": item, "summary": summary,
                                   "transcript_envelope": envelope, "reason": "missing_original_anchor"})
                continue
            source_system, source_user = (_text_content(m["content"]) for m in messages[:2])
            prompt = envelope["task"].get("prompt")
            if not isinstance(prompt, str) or source_user != prompt:
                exclusions["source_task_prompt_mismatch"] += 1
                quarantine.append({"selection": item, "summary": summary,
                                   "transcript_envelope": envelope, "reason": "source_task_prompt_mismatch"})
                continue
            target_user = _target_user(prompt)
            if version in probes:
                if target_user != probes[version]:
                    raise SourceError("exact-version OpenCode target-user probe differs")
                probe_versions.add(version)
            old_anchor_mentions += bool(OLD_ANCHOR.search(source_system) or OLD_ANCHOR.search(source_user))
            training_messages = _training_messages(messages)
            if training_messages is None:
                exclusions["unsupported_message_content_or_private_reasoning"] += 1
                quarantine.append({"selection": item, "summary": summary,
                                   "transcript_envelope": envelope,
                                   "reason": "unsupported_message_content_or_private_reasoning"})
                continue
            visible_messages = training_messages
            report_id = _report_call(training_messages)
            if report_id is None:
                exclusions["unbound_successful_report"] += 1
                quarantine.append({"selection": item, "summary": summary,
                                   "transcript_envelope": envelope, "reason": "unbound_successful_report"})
                continue
            report_end = next(index for index, message in enumerate(training_messages)
                              if message.get("role") == "tool" and message.get("tool_call_id") == report_id)
            tool_checked = _tool_operations(training_messages, report_end)
            discovery = None
            if tool_checked is None:
                candidate = _elide_exact_discovery(training_messages, report_end, tools)
                if candidate is not None:
                    training_messages, discovery = candidate
                    report_end = next(index for index, message in enumerate(training_messages)
                                      if message.get("role") == "tool" and message.get("tool_call_id") == report_id)
                    tool_checked = _tool_operations(training_messages, report_end)
            if tool_checked is None:
                exclusions["tool_argument_or_result_contract_mismatch"] += 1
                quarantine.append({"selection": item, "summary": summary,
                                   "transcript_envelope": envelope,
                                   "reason": "tool_argument_or_result_contract_mismatch"})
                continue
            operations, target_messages = tool_checked
            transform = {"schema": "fleet_opencode_anchor_substitution_v1",
                         "method": ANCHOR_METHOD, "source_trace_sha256": item["trace_sha256"],
                         "source_system_sha256": text_digest(source_system),
                         "source_user_sha256": text_digest(source_user),
                         "source_task_prompt_sha256": text_digest(prompt),
                         "target_system_sha256": text_digest(target_system),
                         "target_user_sha256": text_digest(target_user),
                         "probe_file_sha256": [binding["sha256"] for binding in bindings]}
            transform["sha256"] = digest(transform)
            tool_transform = {"schema": "fleet_source_tool_contract_check_v1",
                              "source_messages_sha256": digest(training_messages),
                              "accepted_prefix_result_index": report_end,
                              "operations": operations, "target_tool_schema_sha256": TOOL_DIGEST}
            visibility = {"schema": "fleet_visible_only_assistant_transform_v1",
                          "original_messages_sha256": digest(messages),
                          "visible_messages_sha256": digest(visible_messages),
                          "hidden_reasoning_fields_removed": sum(
                              field in message for message in messages if isinstance(message, dict)
                              for field in ("thinking", "reasoning", "reasoning_content", "analysis"))}
            visibility["sha256"] = digest(visibility)
            target_messages[0] = {**target_messages[0], "content": target_system}
            target_messages[1] = {**target_messages[1], "content": target_user}
            tool_transform["target_messages_sha256"] = digest(target_messages)
            tool_transform["sha256"] = digest(tool_transform)
            record = {"session_id": sid, "task_key": key, "task_version_id": version,
                      "model_id": item["model_id"], "family_id": roster[version]["family_id"],
                      "source_group_id": item["group_id"],
                      "split": roster[version]["split"], "trace_sha256": item["trace_sha256"],
                      "summary_sha256": digest(summary), "messages": target_messages,
                      "anchor_transform": transform, "tool_transform": tool_transform,
                      "visibility_transform": visibility, "discovery_transform": discovery}
            records.append(record)
            proofs[sid] = {"verified_success": True, "source_sha256": digest(record, ascii=True),
                           "report_call_id": report_id, "verifier_execution_id": verifier["id"],
                           "trace_sha256": item["trace_sha256"],
                           "legacy_selection_acceptance_sha256_unverified": item["acceptance_sha256"]}
            # New target-anchored method; the historical original-anchor algorithm must not claim it.
            old = {"schema": "fleet_cyber_trajectory_v1", "record_id": sid,
                   "source": {"session_id": sid, "model": item["model_id"],
                              "harness_mode": item["harness_mode"],
                              "harness_sha256": item["harness_sha256"]},
                   "lineage": {"task_key": key, "eval_task_version_id": version},
                   "eligibility": {"sft": True},
                   "outcome": {"infra_valid": True, "success": True, "score": score},
                   "messages": target_messages, "anchor_transform": transform,
                   "tool_transform": tool_transform, "visibility_transform": visibility,
                   "discovery_transform": discovery}
            old["content_digest"] = digest(old)
            target_records.append(old)
            old_proof = {"schema": "fleet_broad_success_evidence_v2", "session_id": sid,
                         "model_id": item["model_id"], "task_key": key,
                         "task_version_id": version,
                         "normalized_record_sha256": digest(old),
                         "transcript_sha256": item["trace_sha256"],
                         "verifier_execution_id": verifier["id"],
                         "routes": {"summary": "GET /v1/sessions?task_key=<exact-train-key>",
                                    "transcript": "GET /v1/sessions/{session_id}/transcript"},
                         "successful_report_call_id": report_id,
                         "outcome": {"status": "completed", "verifier_process_success": True,
                                     "score_at_least_one": True}}
            old_proof["sha256"] = digest(old_proof)
            target_proofs.append(old_proof)
        if len(probe_versions) < MIN_ANCHOR_PROBES:
            raise SourceError("exact-version target-anchor probes were not bound to source")
        files = {"request": request_file_sha256,
                 "records": _write(output / "records.jsonl", records, lines=True),
                 "quarantine": _write(output / "quarantine.private.jsonl", quarantine, lines=True),
                 "raw_sources": _write(output / "raw-sources.private.jsonl", raw_sources, lines=True),
                 "evidence": _write(output / "success-evidence.json", proofs),
                 "dense_target_normalized": _write(output / "dense-target-anchored.jsonl", target_records, lines=True),
                 "dense_success_evidence": _write(output / "dense-success-evidence.jsonl", target_proofs, lines=True),
                 "roster": _write(output / "family-roster.json", roster),
                 "tools": _write(output / "model-facing-tools.json", tools),
                 "aliases": _write(output / "tool-aliases.json", ALIASES),
                 "capture": _write(output / "model-request-capture.json", capture)}
        if any(file_digest(Path(binding["path"])) != binding["sha256"]
               for binding in [request[name] for name in
                               ("selection", "hydration", "roles", "tool_capture")] + bindings):
            raise SourceError("source binding changed during download")
        receipt = {"schema": "fleet_teacher_source_receipt_v1", "request_sha256": request["sha256"],
                   "input_sha256": {name: request[name]["sha256"] for name in
                                    ("selection", "hydration", "roles", "tool_capture")},
                   "target_anchor_probe_file_sha256": [binding["sha256"] for binding in bindings],
                   "files": files, "selected_sessions": len(selected), "retained_sessions": len(records),
                   "retained_by_split": dict(sorted(Counter(row["split"] for row in records).items())),
                   "excluded_sessions": dict(sorted(exclusions.items())),
                   "original_anchor_legacy_tool_name_sessions": old_anchor_mentions,
                   "exact_discovery_elided_sessions": sum(
                       record["discovery_transform"] is not None for record in records),
                   "anchor_method": ANCHOR_METHOD,
                   "visibility_method": "visible_only_assistant_content_v1",
                   "model_facing_tools_sha256": TOOL_DIGEST,
                   "download_complete": len(records) + len(quarantine) + exclusions["final_test"] == len(selected),
                   "training_ready": False,
                   "training_blocker": "live_fleet_mcp_and_served_model_request_attestation_required"}
        receipt["sha256"] = digest(receipt)
        _write(output / "RECEIPT.json", receipt)  # Written last: absence means partial output.
        return receipt
    except BaseException:
        # Preserve the private, incomplete destination for diagnosis; never reuse it.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch digest-bound Fleet teacher sources privately")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--hydrate-selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--hydrate-output", type=Path)
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    try:
        if args.request and not args.hydrate_selection:
            receipt = fetch(json.loads(args.request.read_text()))
            summary = {"receipt_sha256": receipt["sha256"],
                       "retained_sessions": receipt["retained_sessions"],
                       "excluded_sessions": receipt["excluded_sessions"]}
        elif args.hydrate_selection and args.selection_sha256 and args.hydrate_output and not args.request:
            binding = {"path": str(args.hydrate_selection), "sha256": args.selection_sha256}
            request = {"schema": "fleet_teacher_source_hydration_v1", "selection": binding,
                       "output": str(args.hydrate_output)}
            summary = hydrate({**request, "sha256": digest(request)}, max_new=args.max_new)
        else:
            raise SourceError("choose one exact source-fetch or hydration request")
    except Exception as error:
        raise SystemExit("source fetch refused: " + type(error).__name__) from None
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
