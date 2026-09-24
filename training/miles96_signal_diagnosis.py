"""Seal a score-blind diagnosis of the failed Miles96 reward qualification."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SOURCE = Path("/mnt/sfs/jobs/chris-q38-m96-signal-a2")
RECEIPT = SOURCE / "SIGNAL_DIAGNOSIS.json"
HEADER_RECEIPT = SOURCE / "SIGNAL_HEADER_PROBE.json"
PREDICATE_RECEIPT = SOURCE / "SIGNAL_QUALIFICATION_PROBE.json"
PROCESS_RECEIPT = SOURCE / "SIGNAL_PROCESS_CLASSIFICATION.json"
SOURCE_JOB_UID = "d3388003-61ce-4b5d-95cf-5b85d7e6ac8b"
PLAN_SHA256 = "a4cca0bc9af5f8cd47d572c02bf7d513759a8b7c7ee4e912aada087c439261c0"
TASK_VERSION_ID = "0920e798-c7e7-4da6-9d5e-ebeba45ec05a"
VERIFIER_VERSION_ID = "9356b7ca-43b4-4926-a871-d9a95b41f6e5"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SERVED_MODEL = "qwen/chris-q38-base-pass4-v1"
RELEASE_SHA256 = "sha256:1c019cd9fa9906dc4453f1e80766c0b10fc984f18d3c9e94bcb8b5a9b3814583"
JOB_NAME = "chris-q38-m96-signal-a2-diagnosis-a2"
CM_NAME = JOB_NAME + "-code"
HEADER_JOB_NAME = "chris-q38-m96-signal-a2-header-a1"
HEADER_CM_NAME = HEADER_JOB_NAME + "-code"
PREDICATE_JOB_NAME = "chris-q38-m96-signal-a2-diagnosis-a4"
PREDICATE_CM_NAME = PREDICATE_JOB_NAME + "-code"
PROCESS_JOB_NAME = "chris-q38-m96-signal-a2-diagnosis-a5"
PROCESS_CM_NAME = PROCESS_JOB_NAME + "-code"
NAMESPACE = "fleet-train-jobs"
IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:"
    "9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)


def digest(value: Any, *, prefix: bool = True) -> str:
    value_digest = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return "sha256:" + value_digest if prefix else value_digest


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("required diagnosis input is absent")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("required diagnosis input is invalid")
    return value


def _write_once(path: Path, value: dict[str, Any]) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _terminal_digests_match(terminal: dict[str, Any]) -> bool:
    plain_body = {
        key: value for key, value in terminal.items() if key not in {"sha256", "receipt_sha256"}
    }
    receipt_body = {key: value for key, value in terminal.items() if key != "receipt_sha256"}
    return terminal.get("sha256") == digest(plain_body, prefix=False) and terminal.get(
        "receipt_sha256"
    ) == digest(receipt_body)


def header_probe(source: Path = SOURCE) -> dict[str, Any]:
    """Report only identity booleans for the preserved campaign terminal."""
    terminal_path = source / "EVAL_TERMINAL.json"
    receipt_path = source / HEADER_RECEIPT.name
    if source.is_symlink() or not source.is_dir() or receipt_path.exists():
        raise ValueError("source output is absent, unsafe, or already probed")
    terminal = _read(terminal_path)
    body = {
        "schema": "cyber_qwen38_miles96_signal_terminal_header_probe_v1",
        "source_job_uid": SOURCE_JOB_UID,
        "terminal_file_sha256": "sha256:" + hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "schema_matches": terminal.get("schema") == "fleet_eval_campaign_terminal_v1",
        "self_digest_matches": _terminal_digests_match(terminal),
        "plan_matches": terminal.get("plan_sha256") == PLAN_SHA256,
        "terminal_fields_included": False,
        "reward_or_trace_content_read": False,
    }
    receipt = {**body, "sha256": digest(body)}
    _write_once(receipt_path, receipt)
    return receipt


def diagnose(source: Path = SOURCE) -> dict[str, Any]:
    receipt_path = source / RECEIPT.name
    if source.is_symlink() or not source.is_dir() or receipt_path.exists():
        raise ValueError("source output is absent, unsafe, or already diagnosed")
    release = _read(source / "LEAK_RECONCILED.json")
    if (
        release.get("sha256") != RELEASE_SHA256
        or release.get("sha256") != digest({k: v for k, v in release.items() if k != "sha256"})
        or release.get("source_job_uid") != SOURCE_JOB_UID
        or release.get("all_instances_released_after") is not True
        or release.get("live_instance_count_after") != 0
    ):
        raise ValueError("instance release receipt differs")

    terminal = _read(source / "EVAL_TERMINAL.json")
    if (
        terminal.get("schema") != "fleet_eval_campaign_terminal_v1"
        or not _terminal_digests_match(terminal)
        or terminal.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("evaluation terminal differs")
    summary = terminal.get("summary")
    routes = terminal.get("routes")
    if not isinstance(summary, dict) or not isinstance(routes, dict) or set(routes) != {"base"}:
        raise ValueError("evaluation terminal summary differs")
    by_state = summary.get("by_state")
    if summary.get("total") != 8 or not isinstance(by_state, dict):
        raise ValueError("evaluation terminal cell count differs")

    route = routes["base"]
    results = route.get("results") if isinstance(route, dict) else None
    allowed_result_keys = {
        "accepted",
        "claimed",
        "controller_failure_code",
        "failure_code",
        "ledger_cell_id",
        "receipt_sha256",
        "serving_block",
    }
    if not isinstance(results, list) or len(results) != 8:
        raise ValueError("evaluation route result count differs")
    failure_codes = []
    for result in results:
        if not isinstance(result, dict) or not set(result) <= allowed_result_keys:
            raise ValueError("evaluation route result contains private or unknown fields")
        code = result.get("failure_code") or result.get("controller_failure_code")
        if result.get("accepted") is not True:
            if not isinstance(code, str) or not code:
                raise ValueError("failed evaluation result lacks a sanitized failure code")
            failure_codes.append(code)

    attempts = sorted((source / "attempts").iterdir())
    if len(attempts) != 8 or any(path.is_symlink() or not path.is_dir() for path in attempts):
        raise ValueError("bounded campaign attempt count differs")
    rewards = []
    execution_ids = set()
    for attempt in attempts:
        result = _read(attempt / "result.json")
        reward = _read(attempt / "reward-result.json")
        binding = _read(attempt / "binding.json")
        cleanup = _read(attempt / "cleanup.json")
        score = reward.get("reward")
        attestation = reward.get("direct_authority_attestation") or {}
        context = attestation.get("context") or {}
        execution_id = reward.get("verifier_execution_id")
        if (
            result.get("task_version_id") != TASK_VERSION_ID
            or binding.get("task", {}).get("version_id") != TASK_VERSION_ID
            or binding.get("verifier", {}).get("version_id") != VERIFIER_VERSION_ID
            or binding.get("model", {}).get("revision") != MODEL_REVISION
            or binding.get("model", {}).get("session_model") != SERVED_MODEL
            or context.get("task_version_id") != TASK_VERSION_ID
            or context.get("verifier_version_id") != VERIFIER_VERSION_ID
            or result.get("verifier_execution_id") != execution_id
            or result.get("score") != score
            or result.get("agent_termination") != "completed"
            or not isinstance(execution_id, str)
            or not execution_id
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or cleanup
            != {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            }
        ):
            raise ValueError("one reward result is not exact, finite, completed, and released")
        rewards.append(float(score))
        execution_ids.add(execution_id)
    if len(execution_ids) != 8:
        raise ValueError("verifier execution identities are not unique")

    reward_counts = collections.Counter(rewards)
    body = {
        "schema": "cyber_qwen38_miles96_signal_diagnosis_v1",
        "source_job_uid": SOURCE_JOB_UID,
        "episode_count": 8,
        "finite_reward_count": len(rewards),
        "distinct_reward_count": len(reward_counts),
        "reward_multiplicities": sorted(reward_counts.values()),
        "reward_variation": len(reward_counts) >= 2,
        "accepted_result_count": sum(result.get("accepted") is True for result in results),
        "failure_code_counts": dict(sorted(collections.Counter(failure_codes).items())),
        "ledger_state_counts": dict(sorted(by_state.items())),
        "all_instances_released": True,
        "release_receipt_sha256": RELEASE_SHA256,
        "raw_reward_values_included": False,
        "prompts_traces_flags_answers_or_scores_included": False,
    }
    receipt = {**body, "sha256": digest(body)}
    _write_once(receipt_path, receipt)
    return receipt


def predicate_probe(source: Path = SOURCE) -> dict[str, Any]:
    """Report content-free classic-authority, termination, and reward aggregates."""
    receipt_path = source / PREDICATE_RECEIPT.name
    if source.is_symlink() or not source.is_dir() or receipt_path.exists():
        raise ValueError("source output is absent, unsafe, or already probed")
    release = _read(source / "LEAK_RECONCILED.json")
    if (
        release.get("sha256") != RELEASE_SHA256
        or release.get("sha256") != digest({k: v for k, v in release.items() if k != "sha256"})
        or release.get("source_job_uid") != SOURCE_JOB_UID
        or release.get("all_instances_released_after") is not True
        or release.get("live_instance_count_after") != 0
    ):
        raise ValueError("instance release receipt differs")
    terminal = _read(source / "EVAL_TERMINAL.json")
    if (
        terminal.get("schema") != "fleet_eval_campaign_terminal_v1"
        or not _terminal_digests_match(terminal)
        or terminal.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("evaluation terminal differs")

    attempts = sorted((source / "attempts").iterdir())
    if len(attempts) != 8 or any(path.is_symlink() or not path.is_dir() for path in attempts):
        raise ValueError("bounded campaign attempt count differs")
    failures: collections.Counter[str] = collections.Counter()
    termination_counts: collections.Counter[str] = collections.Counter()
    rewards: list[float] = []
    complete = 0
    termination_vocabulary = {
        "completed",
        "execution_timeout",
        "process_error",
        "malformed_trace",
        "harness_error",
        "missing_terminal_step",
        "incomplete_terminal_step",
        "output_limit",
    }
    for attempt in attempts:
        result = _read(attempt / "result.json")
        reward = _read(attempt / "reward-result.json")
        binding = _read(attempt / "binding.json")
        cleanup = _read(attempt / "cleanup.json")
        score = reward.get("reward")
        execution_id = reward.get("verifier_execution_id")
        termination = result.get("agent_termination")
        termination_counts[termination if termination in termination_vocabulary else "other"] += 1
        finite_reward = (
            not isinstance(score, bool)
            and isinstance(score, (int, float))
            and math.isfinite(float(score))
            and 0.0 <= float(score) <= 1.0
        )
        checks = {
            "binding_model_revision": binding.get("model", {}).get("revision") == MODEL_REVISION,
            "binding_served_model": binding.get("model", {}).get("session_model") == SERVED_MODEL,
            "binding_task_identity": binding.get("task", {}).get("version_id") == TASK_VERSION_ID,
            "binding_verifier_identity": binding.get("verifier", {}).get("version_id")
            == VERIFIER_VERSION_ID,
            "completed_termination": result.get("agent_termination") == "completed",
            "exact_cleanup": cleanup
            == {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
            "execution_identity_equal": result.get("verifier_execution_id") == execution_id,
            "execution_identity_present": isinstance(execution_id, str) and bool(execution_id),
            "finite_unit_reward": finite_reward,
            "result_reward_equal": result.get("score") == score,
            "result_task_identity": result.get("task_version_id") == TASK_VERSION_ID,
        }
        failures.update(key for key, passed in checks.items() if not passed)
        complete += all(checks.values())
        if finite_reward:
            rewards.append(float(score))

    reward_counts = collections.Counter(rewards)

    body = {
        "schema": "cyber_qwen38_miles96_signal_qualification_probe_v1",
        "source_job_uid": SOURCE_JOB_UID,
        "episode_count": 8,
        "complete_contract_count": complete,
        "failed_predicate_counts": {
            key: failures[key]
            for key in (
                "binding_model_revision",
                "binding_served_model",
                "binding_task_identity",
                "binding_verifier_identity",
                "completed_termination",
                "exact_cleanup",
                "execution_identity_equal",
                "execution_identity_present",
                "finite_unit_reward",
                "result_reward_equal",
                "result_task_identity",
            )
        },
        "termination_category_counts": {
            key: termination_counts[key]
            for key in (
                "completed",
                "execution_timeout",
                "process_error",
                "malformed_trace",
                "harness_error",
                "missing_terminal_step",
                "incomplete_terminal_step",
                "output_limit",
                "other",
            )
        },
        "finite_reward_count": len(rewards),
        "distinct_reward_count": len(reward_counts),
        "reward_multiplicities": sorted(reward_counts.values()),
        "reward_variation": len(reward_counts) >= 2,
        "all_instances_released": True,
        "release_receipt_sha256": RELEASE_SHA256,
        "identities_or_values_included": False,
        "prompts_traces_flags_answers_rewards_or_scores_included": False,
    }
    receipt = {**body, "sha256": digest(body)}
    _write_once(receipt_path, receipt)
    return receipt


def _exit_category(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "invalid"
    return {
        0: "zero",
        1: "generic_failure",
        2: "usage_or_configuration",
        124: "timeout_wrapper",
        126: "cannot_execute",
        127: "command_not_found",
        137: "sigkill_or_oom",
        143: "sigterm_or_external_stop",
    }.get(value, "other_signal_style" if value < 0 or 128 <= value <= 255 else "other_nonzero")


def _structural_trace(path: Path) -> tuple[str, int, int]:
    """Classify only documented event types/reasons; never retain event content."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("canonical trace is absent or unsafe")
    event_count = 0
    malformed = 0
    has_error = False
    last_finish: tuple[int, str] | None = None
    step_after_finish = False
    with path.open(errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(event, dict):
                malformed += 1
                continue
            event_count += 1
            event_type = event.get("type")
            if event_type == "error":
                has_error = True
            if event_type == "step_finish" and isinstance(event.get("part"), dict):
                reason = event["part"].get("reason")
                last_finish = (
                    event_count,
                    reason if reason in {"stop", "length"} else "other",
                )
                step_after_finish = False
            elif event_type == "step_start" and last_finish is not None:
                step_after_finish = True
    if event_count == 0:
        return "empty_trace", event_count, malformed
    if malformed:
        return "malformed_trace", event_count, malformed
    if has_error:
        return "harness_error", event_count, malformed
    if last_finish is None:
        return "missing_terminal_step", event_count, malformed
    if step_after_finish:
        return "incomplete_terminal_step", event_count, malformed
    if last_finish[1] == "stop":
        return "completed", event_count, malformed
    if last_finish[1] == "length":
        return "output_limit", event_count, malformed
    return "incomplete_terminal_step", event_count, malformed


def process_probe(source: Path = SOURCE) -> dict[str, Any]:
    """Explain process_error using aggregate process and trace-shape categories only."""
    receipt_path = source / PROCESS_RECEIPT.name
    if source.is_symlink() or not source.is_dir() or receipt_path.exists():
        raise ValueError("source output is absent, unsafe, or already classified")
    release = _read(source / "LEAK_RECONCILED.json")
    if (
        release.get("sha256") != RELEASE_SHA256
        or release.get("sha256") != digest({k: v for k, v in release.items() if k != "sha256"})
        or release.get("source_job_uid") != SOURCE_JOB_UID
        or release.get("all_instances_released_after") is not True
        or release.get("live_instance_count_after") != 0
    ):
        raise ValueError("instance release receipt differs")
    terminal_path = source / "EVAL_TERMINAL.json"
    terminal = _read(terminal_path)
    if (
        terminal.get("schema") != "fleet_eval_campaign_terminal_v1"
        or not _terminal_digests_match(terminal)
        or terminal.get("plan_sha256") != PLAN_SHA256
    ):
        raise ValueError("evaluation terminal differs")

    attempts = sorted((source / "attempts").iterdir())
    if len(attempts) != 8 or any(path.is_symlink() or not path.is_dir() for path in attempts):
        raise ValueError("bounded campaign attempt count differs")
    exit_categories: collections.Counter[str] = collections.Counter()
    trace_categories: collections.Counter[str] = collections.Counter()
    repair_categories: collections.Counter[str] = collections.Counter()
    ingest_categories: collections.Counter[str] = collections.Counter()
    process_error_count = 0
    exit_binding_count = 0
    manifest_binding_count = 0
    scoring_result_count = 0
    cleanup_exact_count = 0
    for attempt in attempts:
        result = _read(attempt / "result.json")
        process = _read(attempt / "agent-process.json")
        manifest = _read(attempt / "trace-manifest.json")
        cleanup = _read(attempt / "cleanup.json")
        ingest = _read(attempt / "session-ingest.json")
        exit_code = process.get("exit_code")
        exit_category = _exit_category(exit_code)
        exit_categories[exit_category] += 1
        process_error_count += result.get("agent_termination") == "process_error"
        exit_binding_count += result.get("agent_exit_code") == exit_code

        relative = Path(str(manifest.get("canonical_trace") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("canonical trace manifest path is unsafe")
        trace_path = (attempt / relative).resolve()
        trace_path.relative_to(attempt.resolve())
        trace_category, event_count, malformed_count = _structural_trace(trace_path)
        trace_categories[trace_category] += 1
        manifest_binding_count += (
            manifest.get("event_count") == event_count
            and manifest.get("malformed_line_count") == malformed_count
            and manifest.get("agent_termination") == result.get("agent_termination")
        )
        ingest_status = ingest.get("status")
        ingest_categories[
            ingest_status if ingest_status in {"completed", "failed"} else "other"
        ] += 1
        scoring_result_count += (attempt / "reward-result.json").is_file()
        cleanup_exact_count += cleanup == {
            "instance_created": True,
            "instance_closed": True,
            "containers_removed": True,
        }

        if exit_category in {"sigkill_or_oom", "sigterm_or_external_stop"}:
            repair_categories["signal_or_resource_exit"] += 1
        elif trace_category in {"completed", "output_limit"}:
            repair_categories["nonzero_exit_after_gradeable_trace"] += 1
        elif trace_category == "harness_error":
            repair_categories["structured_harness_error"] += 1
        elif trace_category == "malformed_trace":
            repair_categories["malformed_trace"] += 1
        elif trace_category == "empty_trace":
            repair_categories["startup_or_cli_failure"] += 1
        else:
            repair_categories["unfinished_trace"] += 1

    body = {
        "schema": "cyber_qwen38_miles96_process_error_classification_v1",
        "source_identity_sha256": digest(
            {"source_job_uid": SOURCE_JOB_UID, "plan_sha256": PLAN_SHA256}
        ),
        "terminal_file_sha256": "sha256:" + hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "episode_count": 8,
        "process_error_count": process_error_count,
        "exit_category_counts": dict(sorted(exit_categories.items())),
        "structural_trace_category_counts": dict(sorted(trace_categories.items())),
        "repair_category_counts": dict(sorted(repair_categories.items())),
        "session_ingest_category_counts": dict(sorted(ingest_categories.items())),
        "result_process_exit_binding_count": exit_binding_count,
        "trace_manifest_binding_count": manifest_binding_count,
        "scoring_result_present_count": scoring_result_count,
        "exact_cleanup_count": cleanup_exact_count,
        "all_instances_released": True,
        "release_receipt_sha256": RELEASE_SHA256,
        "stderr_or_log_text_read": False,
        "identifiers_reward_values_or_private_content_included": False,
        "prompts_traces_flags_answers_or_scores_included": False,
    }
    receipt = {**body, "sha256": digest(body)}
    _write_once(receipt_path, receipt)
    return receipt


def packet() -> dict[str, Any]:
    source = Path(__file__).read_text()
    bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "immutable": True,
                "metadata": {"name": CM_NAME, "namespace": NAMESPACE},
                "data": {"diagnosis.py": source},
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": JOB_NAME,
                    "namespace": NAMESPACE,
                    "annotations": {"fleet.ai/failure-alerts": "off"},
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/role": "miles96-signal-diagnosis",
                        "kueue.x-k8s.io/queue-name": "training-lq",
                        "kueue.x-k8s.io/priority-class": "q1",
                    },
                },
                "spec": {
                    "suspend": True,
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": 300,
                    "template": {
                        "metadata": {"annotations": {"fleet.ai/failure-alerts": "off"}},
                        "spec": {
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
                                    "name": "diagnose",
                                    "image": IMAGE,
                                    "command": ["python", "/bootstrap/diagnosis.py"],
                                    "resources": {
                                        "requests": {"cpu": "1", "memory": "2Gi"},
                                        "limits": {"cpu": "2", "memory": "4Gi"},
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "bootstrap",
                                            "mountPath": "/bootstrap",
                                            "readOnly": True,
                                        },
                                        {"name": "sfs", "mountPath": "/mnt/sfs"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "bootstrap", "configMap": {"name": CM_NAME}},
                                {
                                    "name": "sfs",
                                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                                },
                            ],
                        },
                    },
                },
            },
        ],
    }
    body = {
        "schema": "cyber_qwen38_miles96_signal_diagnosis_packet_v2",
        "job_name": JOB_NAME,
        "config_map_name": CM_NAME,
        "source_job_uid": SOURCE_JOB_UID,
        "bundle": bundle,
        "create_counts": {"config_map": 1, "job": 1, "retry": 0, "patch": 0},
        "expected": {
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
        },
    }
    return {**body, "sha256": digest(body)}


def header_packet() -> dict[str, Any]:
    source = Path(__file__).read_text()
    bundle = {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "immutable": True,
                "metadata": {"name": HEADER_CM_NAME, "namespace": NAMESPACE},
                "data": {"diagnosis.py": source},
            },
            {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": HEADER_JOB_NAME,
                    "namespace": NAMESPACE,
                    "annotations": {"fleet.ai/failure-alerts": "off"},
                    "labels": {
                        "cyber-post-train.fleet.ai/owner": "chris",
                        "cyber-post-train.fleet.ai/role": "miles96-signal-header-probe",
                        "kueue.x-k8s.io/queue-name": "training-lq",
                        "kueue.x-k8s.io/priority-class": "q1",
                    },
                },
                "spec": {
                    "suspend": True,
                    "backoffLimit": 0,
                    "activeDeadlineSeconds": 300,
                    "template": {
                        "metadata": {"annotations": {"fleet.ai/failure-alerts": "off"}},
                        "spec": {
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
                                    "command": [
                                        "python",
                                        "/bootstrap/diagnosis.py",
                                        "--header-probe",
                                    ],
                                    "resources": {
                                        "requests": {"cpu": "1", "memory": "1Gi"},
                                        "limits": {"cpu": "2", "memory": "2Gi"},
                                    },
                                    "volumeMounts": [
                                        {
                                            "name": "bootstrap",
                                            "mountPath": "/bootstrap",
                                            "readOnly": True,
                                        },
                                        {"name": "sfs", "mountPath": "/mnt/sfs"},
                                    ],
                                }
                            ],
                            "volumes": [
                                {"name": "bootstrap", "configMap": {"name": HEADER_CM_NAME}},
                                {
                                    "name": "sfs",
                                    "persistentVolumeClaim": {"claimName": "sfs-shared"},
                                },
                            ],
                        },
                    },
                },
            },
        ],
    }
    body = {
        "schema": "cyber_qwen38_miles96_signal_header_probe_packet_v1",
        "job_name": HEADER_JOB_NAME,
        "config_map_name": HEADER_CM_NAME,
        "source_job_uid": SOURCE_JOB_UID,
        "bundle": bundle,
        "create_counts": {"config_map": 1, "job": 1, "retry": 0, "patch": 0},
        "expected": {
            "gpus": 0,
            "priority_class": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
        },
    }
    return {**body, "sha256": digest(body)}


def predicate_packet() -> dict[str, Any]:
    base = packet()
    config_map, job = base["bundle"]["items"]
    config_map["metadata"]["name"] = PREDICATE_CM_NAME
    job["metadata"]["name"] = PREDICATE_JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/role"] = "miles96-signal-predicate-probe"
    job["spec"]["template"]["spec"]["containers"][0]["command"].append("--predicate-probe")
    job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] = PREDICATE_CM_NAME
    body = {
        "schema": "cyber_qwen38_miles96_signal_qualification_packet_v2",
        "job_name": PREDICATE_JOB_NAME,
        "config_map_name": PREDICATE_CM_NAME,
        "source_job_uid": SOURCE_JOB_UID,
        "bundle": base["bundle"],
        "create_counts": {"config_map": 1, "job": 1, "retry": 0, "patch": 0},
        "expected": base["expected"],
    }
    return {**body, "sha256": digest(body)}


def process_packet() -> dict[str, Any]:
    base = packet()
    config_map, job = base["bundle"]["items"]
    config_map["metadata"]["name"] = PROCESS_CM_NAME
    job["metadata"]["name"] = PROCESS_JOB_NAME
    job["metadata"]["labels"]["cyber-post-train.fleet.ai/role"] = "miles96-process-error-classifier"
    job["spec"]["template"]["spec"]["containers"][0]["command"].append("--process-probe")
    job["spec"]["template"]["spec"]["volumes"][0]["configMap"]["name"] = PROCESS_CM_NAME
    body = {
        "schema": "cyber_qwen38_miles96_process_error_packet_v1",
        "job_name": PROCESS_JOB_NAME,
        "config_map_name": PROCESS_CM_NAME,
        "source_identity_sha256": digest(
            {"source_job_uid": SOURCE_JOB_UID, "plan_sha256": PLAN_SHA256}
        ),
        "bundle": base["bundle"],
        "create_counts": {"config_map": 1, "job": 1, "retry": 0, "patch": 0},
        "expected": base["expected"],
    }
    return {**body, "sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", action="store_true")
    parser.add_argument("--header-packet", action="store_true")
    parser.add_argument("--header-probe", action="store_true")
    parser.add_argument("--predicate-packet", action="store_true")
    parser.add_argument("--predicate-probe", action="store_true")
    parser.add_argument("--process-packet", action="store_true")
    parser.add_argument("--process-probe", action="store_true")
    args = parser.parse_args()
    selected = sum(
        (
            args.packet,
            args.header_packet,
            args.header_probe,
            args.predicate_packet,
            args.predicate_probe,
            args.process_packet,
            args.process_probe,
        )
    )
    if selected > 1:
        parser.error("select at most one operation")
    if args.process_packet:
        value = process_packet()
    elif args.process_probe:
        value = process_probe()
    elif args.predicate_packet:
        value = predicate_packet()
    elif args.predicate_probe:
        value = predicate_probe()
    elif args.packet:
        value = packet()
    elif args.header_packet:
        value = header_packet()
    elif args.header_probe:
        value = header_probe()
    else:
        value = diagnose()
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
