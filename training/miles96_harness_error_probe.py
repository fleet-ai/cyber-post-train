"""Classify preserved Miles96 OpenCode errors without retaining private text."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SOURCE = Path("/mnt/sfs/jobs/chris-q38-m96-signal-a2")
OUTPUT = SOURCE / "SIGNAL_HARNESS_ERROR_KIND.json"
SOURCE_JOB_UID = "d3388003-61ce-4b5d-95cf-5b85d7e6ac8b"
SOURCE_RECEIPT_SHA256 = "ba6c8e650050ae9580e131bc44a1a50d4bd90c2218f79b06c031080cee9cce57"
JOB_NAME = "chris-q38-m96-herr-kind-a2"
CONFIG_MAP_NAME = JOB_NAME + "-code"
NAMESPACE = "fleet-train-jobs"
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)

ERROR_NAMES = {
    "APIError",
    "ContentFilterError",
    "ContextOverflowError",
    "MessageAbortedError",
    "MessageOutputLengthError",
    "ProviderAuthError",
    "StructuredOutputError",
    "UnknownError",
}
RESPONSE_CODES = {
    "context_length_exceeded",
    "insufficient_quota",
    "invalid_prompt",
    "invalid_request_error",
    "model_not_found",
    "not_found",
    "rate_limit_exceeded",
    "server_error",
    "server_is_overloaded",
    "unsupported_parameter",
    "unsupported_value",
    "usage_not_included",
}
MESSAGE_CATEGORIES = (
    ("context_limit", r"context.{0,24}(?:length|window|limit)|too many tokens"),
    ("model_not_found", r"model.{0,32}(?:not found|does not exist|unknown)"),
    ("authentication", r"unauthori[sz]ed|authentication|invalid api key"),
    ("forbidden", r"forbidden|permission denied"),
    ("rate_limit", r"rate.?limit|too many requests"),
    ("connection_refused", r"econnrefused|connection refused"),
    ("connection_error", r"unable to connect|connection (?:error|failed)"),
    ("name_resolution", r"enotfound|name or service not known|dns"),
    ("timeout", r"timed? ?out|timeout"),
    ("tool_schema", r"tool.{0,40}(?:schema|invalid|required|unsupported)"),
    ("unsupported_parameter", r"unsupported.{0,32}(?:parameter|value)"),
    ("invalid_request", r"invalid.{0,24}(?:request|prompt)"),
    ("server_failure", r"internal server|bad gateway|service unavailable"),
)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required input is absent or unsafe: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"required input is not an object: {path.name}")
    return value


def _source_receipt(source: Path) -> None:
    value = read_object(source / "SIGNAL_PROCESS_CLASSIFICATION.json")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if str(value.get("sha256", "")).removeprefix("sha256:") != SOURCE_RECEIPT_SHA256:
        raise ValueError("source classifier identity changed")
    if digest(body) != SOURCE_RECEIPT_SHA256:
        raise ValueError("source classifier digest changed")


def _error_category(error: object) -> tuple[str, str, str, str, str]:
    if not isinstance(error, dict):
        return "invalid", "absent", "absent", "absent", "unclassified"
    name = error.get("name")
    name_category = name if name in ERROR_NAMES else "unrecognized"
    data = error.get("data")
    if not isinstance(data, dict):
        return name_category, "absent", "absent", "absent", "unclassified"
    status = data.get("statusCode")
    status_category = str(status) if type(status) is int and 100 <= status <= 599 else "absent"
    retryable = data.get("isRetryable")
    retryable_category = str(retryable).lower() if type(retryable) is bool else "absent"
    response_code = "absent"
    response_body = data.get("responseBody")
    if isinstance(response_body, str):
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            candidate = parsed.get("code")
            if isinstance(parsed.get("error"), dict):
                candidate = parsed["error"].get("code", candidate)
            if isinstance(candidate, str):
                response_code = candidate if candidate in RESPONSE_CODES else "unrecognized"
    message = data.get("message")
    category = "unclassified"
    if isinstance(message, str):
        lowered = message.lower()
        category = next(
            (label for label, pattern in MESSAGE_CATEGORIES if re.search(pattern, lowered)),
            "unclassified",
        )
    return name_category, status_category, retryable_category, response_code, category


def classify(source: Path = SOURCE, output: Path = OUTPUT) -> dict[str, Any]:
    if source.is_symlink() or not source.is_dir() or output.exists():
        raise ValueError("source is absent, unsafe, or already classified")
    _source_receipt(source)
    attempts = sorted((source / "attempts").iterdir())
    if len(attempts) != 8 or any(path.is_symlink() or not path.is_dir() for path in attempts):
        raise ValueError("bounded attempt set changed")

    names: collections.Counter[str] = collections.Counter()
    statuses: collections.Counter[str] = collections.Counter()
    retryable: collections.Counter[str] = collections.Counter()
    codes: collections.Counter[str] = collections.Counter()
    messages: collections.Counter[str] = collections.Counter()
    message_hashes: set[str] = set()
    error_events_per_attempt: collections.Counter[str] = collections.Counter()
    total = 0
    for attempt in attempts:
        manifest = read_object(attempt / "trace-manifest.json")
        relative = Path(str(manifest.get("canonical_trace") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("canonical trace path is unsafe")
        trace = (attempt / relative).resolve()
        trace.relative_to(attempt.resolve())
        if trace.is_symlink() or not trace.is_file():
            raise ValueError("canonical trace is absent or unsafe")
        expected = manifest.get("canonical_trace_sha256")
        actual = "sha256:" + hashlib.sha256(trace.read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError("canonical trace digest changed")
        attempt_errors = 0
        with trace.open() as stream:
            for line in stream:
                event = json.loads(line)
                if not isinstance(event, dict) or event.get("type") != "error":
                    continue
                attempt_errors += 1
                total += 1
                error = event.get("error")
                name, status, can_retry, code, message = _error_category(error)
                names[name] += 1
                statuses[status] += 1
                retryable[can_retry] += 1
                codes[code] += 1
                messages[message] += 1
                if isinstance(error, dict) and isinstance(error.get("data"), dict):
                    raw_message = error["data"].get("message")
                    if isinstance(raw_message, str):
                        message_hashes.add(hashlib.sha256(raw_message.encode()).hexdigest())
        if attempt_errors < 1:
            raise ValueError("one harness-error trace has no structured error event")
        error_events_per_attempt[str(attempt_errors)] += 1

    body = {
        "schema": "cyber_qwen38_miles96_harness_error_kind_v1",
        "source_job_uid": SOURCE_JOB_UID,
        "source_classifier_sha256": "sha256:" + SOURCE_RECEIPT_SHA256,
        "attempt_count": 8,
        "error_event_count": total,
        "error_events_per_attempt": dict(sorted(error_events_per_attempt.items())),
        "error_name_counts": dict(sorted(names.items())),
        "api_status_counts": dict(sorted(statuses.items())),
        "retryable_counts": dict(sorted(retryable.items())),
        "response_code_counts": dict(sorted(codes.items())),
        "message_category_counts": dict(sorted(messages.items())),
        "distinct_message_sha256_count": len(message_hashes),
        "stderr_or_log_text_read": False,
        "private_text_retained": False,
        "identifiers_rewards_prompts_traces_flags_answers_or_scores_included": False,
    }
    receipt = {**body, "sha256": "sha256:" + digest(body)}
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def packet() -> dict[str, Any]:
    labels = {
        "cyber-post-train.fleet.ai/owner": "chris",
        "cyber-post-train.fleet.ai/role": "miles96-harness-error-probe",
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    annotations = {
        "fleet.ai/failure-alerts": "off",
        "cyber-post-train.fleet.ai/create-once": "true",
    }
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "immutable": True,
        "metadata": {"name": CONFIG_MAP_NAME, "namespace": NAMESPACE},
        "data": {"probe.py": Path(__file__).read_text()},
    }
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": JOB_NAME,
            "namespace": NAMESPACE,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": {
            "suspend": False,
            "backoffLimit": 0,
            "activeDeadlineSeconds": 300,
            "template": {
                "metadata": {"labels": labels, "annotations": annotations},
                "spec": {
                    "automountServiceAccountToken": False,
                    "restartPolicy": "Never",
                    "priorityClassName": "c1",
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "workload": "fleetai-training-ng-cpu",
                    },
                    "tolerations": [
                        {
                            "key": "workload",
                            "operator": "Equal",
                            "value": "fleetai-training-ng-cpu",
                            "effect": "NoSchedule",
                        }
                    ],
                    "containers": [
                        {
                            "name": "probe",
                            "image": IMAGE,
                            "command": ["python", "/bootstrap/probe.py"],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "1Gi"},
                                "limits": {"cpu": "2", "memory": "2Gi"},
                            },
                            "volumeMounts": [
                                {"name": "bootstrap", "mountPath": "/bootstrap", "readOnly": True},
                                {"name": "sfs", "mountPath": "/mnt/sfs"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "bootstrap", "configMap": {"name": CONFIG_MAP_NAME}},
                        {"name": "sfs", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
                    ],
                },
            },
        },
    }
    body = {
        "schema": "cyber_qwen38_miles96_harness_error_probe_packet_v1",
        "source_job_uid": SOURCE_JOB_UID,
        "source_classifier_sha256": "sha256:" + SOURCE_RECEIPT_SHA256,
        "job_name": JOB_NAME,
        "config_map_name": CONFIG_MAP_NAME,
        "bundle": {"apiVersion": "v1", "kind": "List", "items": [config_map, job]},
        "expected": {
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "maximum_seconds": 300,
        },
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", action="store_true")
    args = parser.parse_args()
    print(json.dumps(packet() if args.packet else classify(), sort_keys=True))


if __name__ == "__main__":
    main()
