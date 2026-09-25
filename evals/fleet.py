"""Offline, paired Fleet pass@4 accounting. No network or rollout execution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

SCHEMA = "fleet_paired_pass4_v1"
SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
TASK_FIELDS = {
    "task_key", "task_version_id", "application", "family_id",
    "environment_version_id", "data_version", "verifier_sha256",
}
COMMON_FIELDS = {
    "model_repository", "tokenizer_sha256", "chat_template_sha256",
    "serving_image", "serving_config_sha256", "harness_image", "harness_version",
    "system_prompt_sha256", "tools", "tool_schema_sha256", "context_policy",
    "context_window_tokens", "max_output_tokens", "max_model_requests",
    "timeout_seconds", "temperature", "top_p", "seeds", "retry_limit",
}
EVENT_FIELDS = {
    "protocol_sha256", "arm", "task_version_id", "attempt", "seed",
    "model_revision", "weights_sha256", "checkpoint_sha256", "process_exit_code", "termination",
    "verifier_sha256", "verifier_status", "success",
}


def _digest(value: dict) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha(value: object) -> bool:
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _fields(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} has missing or unknown fields")
    return value


def _structure(protocol: dict) -> None:
    _fields(protocol, {"schema", "study_id", "role", "tasks", "common", "arms",
                       "final_selection_sha256"}, "protocol")
    if protocol["schema"] != SCHEMA or not _text(protocol["study_id"]):
        raise ValueError("invalid protocol identity")
    if protocol["role"] not in {"dev", "final"}:
        raise ValueError("role must be dev or final")
    if (protocol["role"] == "final" and not _sha(protocol["final_selection_sha256"])) or (
        protocol["role"] == "dev" and protocol["final_selection_sha256"] is not None
    ):
        raise ValueError("final evaluation requires a frozen development selection receipt")
    if not isinstance(protocol["tasks"], list) or not protocol["tasks"]:
        raise ValueError("at least one exact task is required")
    versions, families = set(), set()
    for task in protocol["tasks"]:
        _fields(task, TASK_FIELDS, "task")
        if not all(_text(task[key]) for key in TASK_FIELDS - {"verifier_sha256"}) or not _sha(task["verifier_sha256"]):
            raise ValueError("task identity is incomplete")
        family = (task["application"], task["family_id"])
        if task["task_version_id"] in versions or family in families:
            raise ValueError("one exact task version per reviewed family is required")
        versions.add(task["task_version_id"])
        families.add(family)
    common = _fields(protocol["common"], COMMON_FIELDS, "common treatment")
    for key in ("model_repository", "harness_version", "context_policy"):
        if not _text(common[key]):
            raise ValueError(f"{key} is missing")
    if any(not isinstance(common[key], str) or IMAGE.fullmatch(common[key]) is None
           for key in ("serving_image", "harness_image")):
        raise ValueError("serving and harness images must be pinned by digest")
    for key in ("tokenizer_sha256", "chat_template_sha256", "serving_config_sha256",
                "system_prompt_sha256", "tool_schema_sha256"):
        if not _sha(common[key]):
            raise ValueError(f"{key} must be an immutable digest")
    if not isinstance(common["tools"], list) or not common["tools"] or not all(map(_text, common["tools"])) or len(set(common["tools"])) != len(common["tools"]):
        raise ValueError("ordered tool names must be unique and nonempty")
    for key in ("context_window_tokens", "max_output_tokens", "max_model_requests", "timeout_seconds"):
        if type(common[key]) is not int or common[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if common["max_output_tokens"] > common["context_window_tokens"]:
        raise ValueError("output budget exceeds context window")
    for key in ("temperature", "top_p"):
        if type(common[key]) not in {int, float} or not math.isfinite(common[key]):
            raise ValueError(f"{key} must be finite")
    if not 0 <= common["temperature"] <= 2 or not 0 < common["top_p"] <= 1:
        raise ValueError("invalid sampling")
    seeds = common["seeds"]
    if not isinstance(seeds, list) or len(seeds) != 4 or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != 4:
        raise ValueError("pass@4 requires four distinct fixed seeds")
    if type(common["retry_limit"]) is not int or common["retry_limit"] not in (0, 1):
        raise ValueError("reviewed infrastructure retry limit must be zero or one")
    arms = _fields(protocol["arms"], {"base", "candidate"}, "arms")
    _fields(arms["base"], {"model_revision", "weights_sha256"}, "base arm")
    _fields(arms["candidate"], {"model_revision", "weights_sha256", "checkpoint_sha256"}, "candidate arm")
    for arm in arms.values():
        if not _text(arm["model_revision"]) or not _sha(arm["weights_sha256"]):
            raise ValueError("arm lacks exact model identity")
    if not _sha(arms["candidate"]["checkpoint_sha256"]):
        raise ValueError("candidate lacks immutable checkpoint digest")
    if arms["base"]["weights_sha256"] == arms["candidate"]["weights_sha256"]:
        raise ValueError("base and candidate weights must differ")


def seal_protocol(value: dict) -> dict:
    """Freeze one task roster and one common treatment; only weight identities differ."""
    protocol = json.loads(json.dumps(value, allow_nan=False))
    _structure(protocol)
    return {**protocol, "sha256": _digest(protocol)}


def validate_protocol(value: dict) -> dict:
    protocol = _fields(value, {"schema", "study_id", "role", "tasks", "common", "arms",
                               "final_selection_sha256", "sha256"}, "sealed protocol")
    if not _sha(protocol["sha256"]) or _digest({k: v for k, v in protocol.items() if k != "sha256"}) != protocol["sha256"]:
        raise ValueError("protocol digest mismatch")
    _structure({k: v for k, v in protocol.items() if k != "sha256"})
    return protocol


def _outcome(event: dict, task: dict) -> str:
    """An unsuccessful *completed* attempt is valid; runtime failures are not zeros."""
    if type(event["process_exit_code"]) is not int or event["process_exit_code"] != 0:
        reason = "process_error"
    elif event["termination"] != "completed":
        reason = "output_limit" if event["termination"] == "output_limit" else "abnormal_termination"
    elif event["verifier_status"] != "completed":
        reason = "verifier_incomplete"
    elif event["verifier_sha256"] != task["verifier_sha256"]:
        reason = "verifier_identity_mismatch"
    elif type(event["success"]) is not bool:
        reason = "missing_valid_result"
    else:
        return "valid"
    if event["success"] is not None:
        raise ValueError(f"{reason} cannot carry a capability result")
    return reason


def _binomial_tail(n: int, k: int, p: float) -> float:
    return math.fsum(math.comb(n, j) * p**j * (1 - p)**(n - j) for j in range(k, n + 1))


def _tail_inverse(n: int, k: int, target: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        if _binomial_tail(n, k, mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def summarize(protocol: dict, events: list[dict]) -> dict:
    """Return a paired family result only when every planned attempt is valid."""
    validate_protocol(protocol)
    tasks = {task["task_version_id"]: task for task in protocol["tasks"]}
    cells = {}
    invalid = {}
    for event in events:
        _fields(event, EVENT_FIELDS, "attempt")
        arm, version, attempt = event["arm"], event["task_version_id"], event["attempt"]
        if arm not in ("base", "candidate") or version not in tasks or type(attempt) is not int or attempt not in (1, 2, 3, 4):
            raise ValueError("attempt is outside the frozen arm/task roster")
        if event["protocol_sha256"] != protocol["sha256"] or event["seed"] != protocol["common"]["seeds"][attempt - 1]:
            raise ValueError("attempt protocol or seed mismatch")
        artifact = protocol["arms"][arm]
        if (event["model_revision"] != artifact["model_revision"]
                or event["weights_sha256"] != artifact["weights_sha256"]
                or event["checkpoint_sha256"] != artifact.get("checkpoint_sha256")):
            raise ValueError("attempt weight or checkpoint binding mismatch")
        key = (arm, version, attempt)
        if key in cells:
            raise ValueError("duplicate attempt; reviewed retries need a new frozen protocol")
        outcome = _outcome(event, tasks[version])
        cells[key] = event
        if outcome != "valid":
            invalid[key] = outcome
    expected = {(arm, version, attempt) for arm in ("base", "candidate")
                for version in tasks for attempt in range(1, 5)}
    missing = expected - cells.keys()
    result = {
        "protocol_sha256": protocol["sha256"], "role": protocol["role"],
        "families": len(tasks), "valid_attempts": len(cells) - len(invalid),
        "missing_attempts": len(missing), "infrastructure_invalid_attempts": len(invalid),
        "missing_cells": [list(key) for key in sorted(missing)],
        "invalid_cells": [{"cell": list(key), "reason": invalid[key]} for key in sorted(invalid)],
        "invalid_reasons": {reason: sum(x == reason for x in invalid.values()) for reason in sorted(set(invalid.values()))},
    }
    if missing or invalid:
        return {**result, "status": "incomplete", "candidate_minus_base_pass4": None,
                "conditional_95pct_interval": None}
    solved = {arm: {version: any(cells[arm, version, i]["success"] for i in range(1, 5))
                    for version in tasks} for arm in ("base", "candidate")}
    wins = sum(solved["candidate"][v] and not solved["base"][v] for v in tasks)
    losses = sum(solved["base"][v] and not solved["candidate"][v] for v in tasks)
    discordant = wins + losses
    theta_lo = 0.0 if wins == 0 else _tail_inverse(discordant, wins, 0.025)
    theta_hi = 1.0 if wins == discordant else _tail_inverse(discordant, wins + 1, 0.975)
    scale = discordant / len(tasks)
    family_results = [{
        "application": task["application"], "family_id": task["family_id"],
        "task_version_id": version, "base_pass4": solved["base"][version],
        "candidate_pass4": solved["candidate"][version],
    } for version, task in tasks.items()]
    return {
        **result, "status": "complete", "base_pass4": sum(solved["base"].values()),
        "candidate_pass4": sum(solved["candidate"].values()), "candidate_wins": wins,
        "base_wins": losses, "ties": len(tasks) - discordant, "family_results": family_results,
        "candidate_minus_base_pass4": (wins - losses) / len(tasks),
        "one_sided_exact_discordance_p": _binomial_tail(discordant, wins, 0.5),
        "conditional_95pct_interval": [scale * (2 * theta_lo - 1), scale * (2 * theta_hi - 1)],
        "interval_condition": "Clopper-Pearson on candidate wins given observed discordant families",
    }


def dev_decision(summary: dict) -> tuple[float, float]:
    """Final-test results are reportable, never a checkpoint-selection signal."""
    if summary.get("role") != "dev" or summary.get("status") != "complete":
        raise ValueError("checkpoint selection requires a complete development comparison")
    return summary["candidate_minus_base_pass4"], summary["one_sided_exact_discordance_p"]


def claim_checkpoint_eval(protocol: dict, queue_dir: Path, kind: str) -> Path | None:
    """Atomically claim a local dispatch intent; this does not launch an eval."""
    validate_protocol(protocol)
    if kind != "fleet_pass4":
        raise ValueError("only protocol-bound Fleet pass@4 can be claimed")
    identity = {
        "kind": kind, "role": protocol["role"],
        "checkpoint_sha256": protocol["arms"]["candidate"]["checkpoint_sha256"],
        "base_weights_sha256": protocol["arms"]["base"]["weights_sha256"],
        "candidate_weights_sha256": protocol["arms"]["candidate"]["weights_sha256"],
        "tasks_sha256": _digest({"tasks": protocol["tasks"]}),
        "treatment_sha256": _digest(protocol["common"]),
    }
    path = Path(queue_dir) / (_digest(identity).split(":", 1)[1] + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w") as stream:
        json.dump({**identity, "protocol_sha256": protocol["sha256"]}, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path
