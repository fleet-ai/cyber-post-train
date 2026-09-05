"""Score-blind concurrency qualification for the shared hosted model routes.

This probe deliberately never contacts task, instance, session, verifier, or
scoring APIs.  It sends only small synthetic structured-tool requests to the
hosted inference endpoint, first at concurrency two and then at concurrency
four for each exact served model id.  Only aggregate latency and protocol
counts are persisted.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SCHEMA = "fleet-hosted-concurrency4-qualification-v1"
ORIGIN = "https://inference.flt.build"
ACCOUNT_URL = "https://orchestrator.fleetai.com/v1/account"
MODELS_URL = f"{ORIGIN}/v1/models"
COMPLETIONS_URL = f"{ORIGIN}/v1/chat/completions"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
EXPECTED_CONTEXT_LENGTH = 262_144
EXPECTED_SECRET_NAME = "chris-cyber-opencode-evals-v2"
EXPECTED_SECRET_UID = "e0febd8e-94a2-46b0-a0bf-dd6b3154187b"
EXPECTED_MODELS = {
    "qwen3.8-27b": {
        "repository": "Qwen/Qwen3.8-27B",
        "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    },
    "glm-5.3": {
        "repository": "zai-org/GLM-5.3",
        "revision": "30333038ada1f1dacb294a93270305a890b50c14",
    },
}
WAVES = (2, 4)
REQUESTS_PER_STREAM = 2
REQUEST_TIMEOUT_SECONDS = 300
MAX_REQUEST_LATENCY_SECONDS = 300.0
MAX_WAVE_ELAPSED_SECONDS = 600.0
MIN_C4_THROUGHPUT_GAIN = 1.25
MAX_C4_TO_C2_P95_LATENCY_RATIO = 2.5
LEASE_NAMESPACE = "hosted-concurrency-qualification-v1"
DEFAULT_LEASE_ROOT = Path("/mnt/sfs/endpoint-leases") / LEASE_NAMESPACE

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in an isolated challenge sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "Submit recovered flags and an explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flags": {"type": "array", "items": {"type": "string"}},
                    "explanation": {"type": "string"},
                },
                "required": ["flags", "explanation"],
                "additionalProperties": False,
            },
        },
    },
]


class QualificationError(RuntimeError):
    """A stable, response-content-free qualification failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _safe_uuid(value: str, label: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise QualificationError(f"{label}_invalid") from exc
    if parsed.int == 0:
        raise QualificationError(f"{label}_invalid")
    return str(parsed)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationError("response_duplicate_json_key")
        result[key] = value
    return result


def _load_json(raw: bytes, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError(code) from exc
    if not isinstance(value, dict):
        raise QualificationError(code)
    return value


def get_json(url: str, api_key: str) -> dict[str, Any]:
    request = Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise QualificationError("identity_get_http_status")
            raw = response.read(4 * 1024 * 1024 + 1)
    except QualificationError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise QualificationError("identity_get_failed") from exc
    if len(raw) > 4 * 1024 * 1024:
        raise QualificationError("identity_get_too_large")
    return _load_json(raw, code="identity_get_invalid_json")


def _request_payload(model: str, tool_name: str) -> dict[str, Any]:
    if tool_name == "bash":
        instruction = "Call bash exactly once with a benign command. Do not call submit_report."
    elif tool_name == "submit_report":
        instruction = (
            "Call submit_report exactly once with an empty flags list and a brief benign "
            "availability statement. Do not call bash."
        )
    else:
        raise ValueError("unsupported synthetic tool")
    return {
        "model": model,
        "messages": [{"role": "user", "content": instruction}],
        "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
    }


def _validate_completion(value: dict[str, Any], *, model: str, tool_name: str) -> None:
    if value.get("model") != model:
        raise QualificationError("completion_model_identity_mismatch")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise QualificationError("completion_choice_shape_mismatch")
    choice = choices[0]
    if choice.get("finish_reason") != "tool_calls":
        raise QualificationError("completion_finish_reason_mismatch")
    message = choice.get("message")
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise QualificationError("completion_tool_call_count_mismatch")
    function = calls[0].get("function")
    if not isinstance(function, dict) or function.get("name") != tool_name:
        raise QualificationError("completion_tool_name_mismatch")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise QualificationError("completion_tool_arguments_invalid")
    parsed = _load_json(arguments.encode(), code="completion_tool_arguments_invalid")
    if tool_name == "bash":
        if set(parsed) != {"command"} or not isinstance(parsed["command"], str):
            raise QualificationError("completion_bash_arguments_mismatch")
    elif (
        set(parsed) != {"flags", "explanation"}
        or not isinstance(parsed["flags"], list)
        or any(not isinstance(item, str) for item in parsed["flags"])
        or not isinstance(parsed["explanation"], str)
    ):
        raise QualificationError("completion_submit_report_arguments_mismatch")


def post_completion(model: str, tool_name: str, api_key: str) -> float:
    body = canonical_json(_request_payload(model, tool_name))
    request = Request(
        COMPLETIONS_URL,
        method="POST",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise QualificationError("completion_http_status")
            raw = response.read(4 * 1024 * 1024 + 1)
    except QualificationError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise QualificationError("completion_request_failed") from exc
    elapsed = time.monotonic() - started
    if len(raw) > 4 * 1024 * 1024:
        raise QualificationError("completion_response_too_large")
    _validate_completion(
        _load_json(raw, code="completion_invalid_json"), model=model, tool_name=tool_name
    )
    if elapsed > MAX_REQUEST_LATENCY_SECONDS:
        raise QualificationError("completion_request_latency_exceeded")
    return elapsed


CompletionCaller = Callable[[str, str, str], float]


def _stream(model: str, api_key: str, caller: CompletionCaller) -> dict[str, Any]:
    latencies = [caller(model, tool_name, api_key) for tool_name in ("bash", "submit_report")]
    return {"request_latencies": latencies, "elapsed_seconds": sum(latencies)}


def _percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "min": round(min(values), 6),
        "median": round(statistics.median(values), 6),
        "p95": round(_percentile95(values), 6),
        "max": round(max(values), 6),
    }


def run_wave(
    model: str, concurrency: int, api_key: str, *, caller: CompletionCaller = post_completion
) -> dict[str, Any]:
    if concurrency not in WAVES:
        raise ValueError("unsupported qualification wave")
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_stream, model, api_key, caller) for _ in range(concurrency)]
        for future in as_completed(futures):
            results.append(future.result())
    wave_elapsed = time.monotonic() - started
    if wave_elapsed > MAX_WAVE_ELAPSED_SECONDS:
        raise QualificationError("wave_elapsed_limit_exceeded")
    request_latencies = [latency for result in results for latency in result["request_latencies"]]
    stream_latencies = [result["elapsed_seconds"] for result in results]
    return {
        "concurrency": concurrency,
        "streams_started": concurrency,
        "streams_succeeded": len(results),
        "requests_expected": concurrency * REQUESTS_PER_STREAM,
        "requests_succeeded": len(request_latencies),
        "protocol_valid_requests": len(request_latencies),
        "errors": 0,
        "wave_elapsed_seconds": round(wave_elapsed, 6),
        "streams_per_minute": round(60.0 * len(results) / wave_elapsed, 6),
        "request_latency_seconds": _stats(request_latencies),
        "stream_latency_seconds": _stats(stream_latencies),
    }


def evaluate_waves(waves: list[dict[str, Any]]) -> dict[str, Any]:
    if [wave.get("concurrency") for wave in waves] != list(WAVES):
        raise QualificationError("wave_order_mismatch")
    for wave in waves:
        concurrency = wave["concurrency"]
        if (
            wave.get("streams_started") != concurrency
            or wave.get("streams_succeeded") != concurrency
            or wave.get("requests_expected") != concurrency * REQUESTS_PER_STREAM
            or wave.get("requests_succeeded") != concurrency * REQUESTS_PER_STREAM
            or wave.get("protocol_valid_requests") != concurrency * REQUESTS_PER_STREAM
            or wave.get("errors") != 0
            or wave.get("wave_elapsed_seconds", MAX_WAVE_ELAPSED_SECONDS + 1)
            > MAX_WAVE_ELAPSED_SECONDS
            or wave.get("request_latency_seconds", {}).get("max", MAX_REQUEST_LATENCY_SECONDS + 1)
            > MAX_REQUEST_LATENCY_SECONDS
        ):
            raise QualificationError("wave_acceptance_threshold_failed")
    c2, c4 = waves
    throughput_gain = c4["streams_per_minute"] / c2["streams_per_minute"]
    latency_ratio = c4["stream_latency_seconds"]["p95"] / max(
        c2["stream_latency_seconds"]["p95"], 0.000001
    )
    return {
        "concurrency4_over_concurrency2_throughput_ratio": round(throughput_gain, 6),
        "concurrency4_over_concurrency2_p95_stream_latency_ratio": round(latency_ratio, 6),
        "minimum_throughput_ratio": MIN_C4_THROUGHPUT_GAIN,
        "maximum_p95_stream_latency_ratio": MAX_C4_TO_C2_P95_LATENCY_RATIO,
        "zero_error_required": True,
        "accepted": (
            throughput_gain >= MIN_C4_THROUGHPUT_GAIN
            and latency_ratio <= MAX_C4_TO_C2_P95_LATENCY_RATIO
        ),
    }


def _context_values(value: Any) -> list[int]:
    fields = {"context_length", "context_window", "max_context_length", "max_model_len"}
    found: list[int] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in fields and type(item) is int:
                found.append(item)
            else:
                found.extend(_context_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_context_values(item))
    return found


def validate_identity(api_key: str) -> dict[str, Any]:
    account = get_json(ACCOUNT_URL, api_key)
    if account.get("team_name") != "fleet" or account.get("team_id") != FLEET_TEAM_ID:
        raise QualificationError("fleet_team_identity_mismatch")
    roster = get_json(MODELS_URL, api_key)
    rows = roster.get("data")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise QualificationError("model_roster_shape_mismatch")
    context_observations: dict[str, dict[str, Any]] = {}
    for model in EXPECTED_MODELS:
        matches = [row for row in rows if row.get("id") == model]
        if len(matches) != 1:
            raise QualificationError("model_roster_identity_mismatch")
        values = sorted(set(_context_values(matches[0])))
        if values and values != [EXPECTED_CONTEXT_LENGTH]:
            raise QualificationError("model_roster_context_mismatch")
        context_observations[model] = {
            "context_length_observable": bool(values),
            "observed_context_length": values[0] if values else None,
        }
    return context_observations


@contextmanager
def qualification_lease(root: Path) -> Iterator[dict[str, Any]]:
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise QualificationError("qualification_lease_root_unsafe")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / "exclusive.lock"
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    handle = os.fdopen(fd, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QualificationError("qualification_lease_already_held") from exc
        yield {"namespace": LEASE_NAMESPACE, "exclusive": True}
    finally:
        if not handle.closed:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def run(out_dir: Path, *, caller: CompletionCaller = post_completion) -> dict[str, Any]:
    out_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    api_key = os.environ.get("FLEET_API_KEY", "")
    if not api_key:
        raise QualificationError("fleet_api_key_absent")
    job_uid = _safe_uuid(os.environ.get("JOB_UID", ""), "job_uid")
    pod_uid = _safe_uuid(os.environ.get("POD_UID", ""), "pod_uid")
    if os.environ.get("EXPECTED_SECRET_UID") != EXPECTED_SECRET_UID:
        raise QualificationError("secret_uid_mismatch")
    package_commit = os.environ.get("PACKAGE_COMMIT", "")
    package_sha = os.environ.get("PACKAGE_SHA256", "")
    if len(package_commit) != 40 or any(ch not in "0123456789abcdef" for ch in package_commit):
        raise QualificationError("package_commit_invalid")
    if not package_sha.startswith("sha256:") or len(package_sha) != 71:
        raise QualificationError("package_sha256_invalid")

    contexts = validate_identity(api_key)
    models: list[dict[str, Any]] = []
    lease_root = Path(os.environ.get("QUALIFICATION_LEASE_ROOT", str(DEFAULT_LEASE_ROOT)))
    with qualification_lease(lease_root) as lease:
        for model, source in EXPECTED_MODELS.items():
            waves = [run_wave(model, concurrency, api_key, caller=caller) for concurrency in WAVES]
            decision = evaluate_waves(waves)
            model_receipt = {
                "schema_version": SCHEMA,
                "status": "PASSED" if decision["accepted"] else "REJECTED",
                "model": {
                    "served_id": model,
                    "repository": source["repository"],
                    "revision": source["revision"],
                    "endpoint_origin": ORIGIN,
                    "response_model_exact": True,
                },
                "context_contract": {
                    "expected_context_length": EXPECTED_CONTEXT_LENGTH,
                    **contexts[model],
                    "unobservable_roster_context_requires_existing_exact_harness_gate": True,
                },
                "tool_contract": {
                    "names": ["bash", "submit_report"],
                    "openai_tool_schema_sha256": sha256(canonical_json(TOOLS)),
                    "forced_selection_order_per_stream": ["bash", "submit_report"],
                    "argument_shapes_valid": True,
                    "tool_execution_performed": False,
                },
                "waves": waves,
                "decision": decision,
                "lease": lease,
                "request_counts": {
                    "chat_completions": sum(wave["requests_succeeded"] for wave in waves),
                    "task_instance": 0,
                    "session": 0,
                    "scoring": 0,
                    "verifier": 0,
                },
                "privacy": {
                    "request_bodies_included": False,
                    "response_bodies_included": False,
                    "tool_arguments_included": False,
                    "prompts_traces_flags_or_scores_included": False,
                    "credentials_included": False,
                },
            }
            model_receipt["receipt_sha256"] = digest_without(model_receipt, "receipt_sha256")
            write_json_once(out_dir / f"{model}.json", model_receipt)
            models.append(model_receipt)

    all_accepted = all(model["decision"]["accepted"] for model in models)
    terminal = {
        "schema_version": SCHEMA,
        "status": "PASSED" if all_accepted else "REJECTED",
        "classification": "operational_gate_no_capability_claim",
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runtime": {
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "secret_name": EXPECTED_SECRET_NAME,
            "secret_uid": EXPECTED_SECRET_UID,
            "package_commit": package_commit,
            "package_sha256": package_sha,
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
            "cpu_only": True,
        },
        "models": [
            {
                "served_id": model["model"]["served_id"],
                "status": model["status"],
                "receipt_sha256": model["receipt_sha256"],
            }
            for model in models
        ],
        "request_counts": {
            "fleet_account_get": 1,
            "hosted_models_get": 1,
            "chat_completions": len(EXPECTED_MODELS) * sum(WAVES) * REQUESTS_PER_STREAM,
            "task_instance": 0,
            "session": 0,
            "scoring": 0,
            "verifier": 0,
        },
        "endpoint_lease": {
            "namespace": LEASE_NAMESPACE,
            "separate_from_scored_endpoint_leases": True,
            "released": True,
        },
        "scored_bulk_launch_authorized": False,
        "privacy": {
            "request_bodies_included": False,
            "response_bodies_included": False,
            "tool_arguments_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    terminal["receipt_sha256"] = digest_without(terminal, "receipt_sha256")
    write_json_once(out_dir / "TERMINAL.json", terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.out_dir)
    except Exception as exc:
        code = exc.code if isinstance(exc, QualificationError) else "unexpected_failure"
        if args.out_dir.is_dir() and not (args.out_dir / "TERMINAL.json").exists():
            failure = {
                "schema_version": SCHEMA,
                "status": "FAILED",
                "classification": "infrastructure_invalid_no_capability_claim",
                "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "failure_code": code,
                "runtime": {
                    "job_uid": os.environ.get("JOB_UID"),
                    "pod_uid": os.environ.get("POD_UID"),
                    "priority_class": "fleet-serve-low",
                    "preemption_policy": "Never",
                    "cpu_only": True,
                },
                "task_instance_session_scoring_or_verifier_calls": 0,
                "response_bodies_included": False,
                "credentials_included": False,
            }
            failure["receipt_sha256"] = digest_without(failure, "receipt_sha256")
            write_json_once(args.out_dir / "TERMINAL.json", failure)
        return 1
    # A threshold rejection is a valid terminal capacity observation, not an
    # infrastructure failure.  The receipt status carries the decision.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
