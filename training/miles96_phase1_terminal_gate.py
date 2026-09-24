"""Caller-side terminal authority required before a Miles96 learning POST."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import JobsError, digest
from training import miles96_mechanics_canary as mechanics
from training import miles96_signal_qualification as signal

SCHEMA = "cyber_qwen38_miles96_phase1_terminal_authority_v1"
TERMINAL_SCHEMA = "cyber_qwen38_miles96_phase1_terminal_observation_v1"
CENSUS_SCHEMA = "cyber_qwen38_miles96_phase1_empty_resource_census_v1"
MAX_CENSUS_AGE_SECONDS = 300
MIN_CENSUS_SEPARATION_SECONDS = 2


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "sha256": "sha256:" + digest(body)}


def _verify_seal(value: dict[str, Any], schema: str) -> None:
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise JobsError(f"expected {schema}")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != "sha256:" + digest(body):
        raise JobsError(f"invalid {schema} seal")


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise JobsError("terminal authority timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JobsError("terminal authority timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise JobsError("terminal authority timestamp lacks timezone")
    return parsed.astimezone(UTC)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise JobsError(f"{label} is not a SHA-256 digest")
    return value


def _uuid(value: object, label: str) -> str:
    try:
        parsed = str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise JobsError(f"{label} is not a UUID") from exc
    if parsed != value:
        raise JobsError(f"{label} is not canonical")
    return parsed


def _phase1_plan(phase2_plan: dict[str, Any]) -> dict[str, Any]:
    phase2 = mechanics.validate_plan(phase2_plan)
    evidence = phase2["task_signal_evidence"]
    return signal.build_plan(
        name=evidence["phase1_run_name"],
        model_root=signal.HF_MODEL_ROOT,
        model_binding_sha256=evidence["model_binding_sha256"],
        task_binding=phase2["task_binding"],
        authority_config_sha256=evidence["authority_config_sha256"],
        current_binding_sha256=evidence["current_binding_sha256"],
        production_split_sha256=evidence["production_split_sha256"],
    )


def _phase1_hashes(phase1_plan: dict[str, Any]) -> dict[str, str]:
    request = signal.job_request(phase1_plan)
    return {
        "phase1_plan_sha256": "sha256:" + mechanics.digest(phase1_plan),
        "phase1_request_sha256": "sha256:" + digest(request),
        "runtime_bundle_sha256": signal._runtime_bundle_sha256(request),
        "request_binding_sha256": signal._request_binding_sha256(request),
        "runtime_source_manifest_sha256": "sha256:"
        + mechanics.digest(phase1_plan["runtime_sources"]),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise JobsError(f"actual phase-1 receipt is absent or unsafe: {path.name}")
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise JobsError(f"actual phase-1 receipt is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise JobsError(f"actual phase-1 receipt is not an object: {path.name}")
    return value


def _raw_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def collect_actual_source(phase2_plan: dict[str, Any]) -> dict[str, Any]:
    """Revalidate the actual SFS receipt tree; never accept supplied evidence."""
    phase2 = mechanics.validate_plan(phase2_plan)
    phase1 = _phase1_plan(phase2)
    root = Path(phase1["identity"]["run_dir"])
    private_root = root / mechanics.PRIVATE_EVIDENCE_DIR
    paths = {
        signal.EVIDENCE_FILE: root / signal.EVIDENCE_FILE,
        signal.NATIVE_TERMINAL_FILE: root / signal.NATIVE_TERMINAL_FILE,
        signal.RUNTIME_PREFLIGHT_FILE: root / signal.RUNTIME_PREFLIGHT_FILE,
        f"{mechanics.PRIVATE_EVIDENCE_DIR}/{signal.PRIVATE_GROUP_FILE}": (
            private_root / signal.PRIVATE_GROUP_FILE
        ),
    }
    if root.is_symlink() or private_root.is_symlink():
        raise JobsError("actual phase-1 receipt root is unsafe")
    values = {name: _read_json(path) for name, path in paths.items()}
    try:
        public = signal.aggregate(phase1)
    except (OSError, ValueError) as exc:
        raise JobsError("actual phase-1 receipt tree failed native validation") from exc
    if public != values[signal.EVIDENCE_FILE] or public != phase2["task_signal_evidence"]:
        raise JobsError("phase-2 plan does not embed the actual phase-1 evidence")

    episodes = signal._private_episode_receipts(phase1)
    claims = signal._private_claim_receipts(phase1)
    attempts = signal._private_attempt_receipts(phase1)
    slots = signal._private_slot_receipts(phase1)
    private = values[f"{mechanics.PRIVATE_EVIDENCE_DIR}/{signal.PRIVATE_GROUP_FILE}"]
    known_private = {private_root / signal.PRIVATE_GROUP_FILE}
    known_private.update(private_root / (item["sha256"][7:] + ".json") for item in episodes)
    known_private.update(private_root / f"claim-{item['slot_id']}.json" for item in claims)
    known_private.update(
        private_root / f"signal-{item['attempt_id']}-{item['stage']}.json" for item in attempts
    )
    known_private.update(
        private_root / f"slot-{item['slot_id']}-{item['attempt_number']}.json" for item in slots
    )
    actual_private = set(private_root.glob("*.json"))
    if actual_private != known_private or any(path.is_symlink() for path in known_private):
        raise JobsError("actual phase-1 private receipt inventory changed")
    paths.update(
        {
            str(path.relative_to(root)): path
            for path in sorted(known_private - {private_root / signal.PRIVATE_GROUP_FILE})
        }
    )
    rewards = [float(item["reward"]) for item in episodes]
    if not rewards or not all(math.isfinite(item) for item in rewards):
        raise JobsError("actual phase-1 source has non-finite rewards")
    reward_count = len(set(rewards))
    model_output = root / "model-output"
    checkpoint_dir = model_output / "checkpoints"
    if model_output.is_symlink() or checkpoint_dir.is_symlink():
        raise JobsError("actual phase-1 checkpoint path is unsafe")
    checkpoint_absent = not (
        checkpoint_dir.exists() and any(path.is_file() for path in checkpoint_dir.rglob("*"))
    )
    hashes = _phase1_hashes(phase1)
    if (
        reward_count < 2
        or not checkpoint_absent
        or public["optimizer_steps"] != 0
        or public["checkpoint_artifacts_absent"] is not True
        or public["runtime_bundle_sha256"] != hashes["runtime_bundle_sha256"]
        or public["request_binding_sha256"] != hashes["request_binding_sha256"]
        or public["runtime_source_manifest_sha256"] != hashes["runtime_source_manifest_sha256"]
    ):
        raise JobsError("actual phase-1 source is not a zero-update mixed-signal terminal")
    receipt_files = {name: _raw_sha(path) for name, path in sorted(paths.items())}
    return {
        **hashes,
        "public_evidence_sha256": public["sha256"],
        "private_group_sha256": private["sha256"],
        "native_terminal_sha256": public["native_terminal_sha256"],
        "runtime_preflight_sha256": public["runtime_preflight_sha256"],
        "receipt_files": receipt_files,
        "receipt_file_manifest_sha256": "sha256:" + digest(receipt_files),
        "receipt_file_count": len(receipt_files),
        "completed_episode_count": len(episodes),
        "distinct_finite_reward_count": reward_count,
        "unique_verifier_execution_count": len(
            {item["verifier_execution_id"] for item in episodes}
        ),
        "terminal_slot_count": len(slots),
        "optimizer_steps": 0,
        "checkpoint_artifacts_absent": True,
        "actual_sfs_revalidation": True,
    }


def validate_terminal_observation(
    value: dict[str, Any],
    phase1_plan: dict[str, Any],
) -> dict[str, Any]:
    _verify_seal(value, TERMINAL_SCHEMA)
    expected = {
        "schema",
        "observed_at",
        "phase1_plan_sha256",
        "phase1_request_sha256",
        "runtime_bundle_sha256",
        "runtime_source_manifest_sha256",
        "request_binding_sha256",
        "jobs_api",
        "rayjob",
        "raycluster_uid",
        "workload_uid",
        "pod",
        "sha256",
    }
    if set(value) != expected:
        raise JobsError("phase-1 terminal observation fields changed")
    _timestamp(value["observed_at"])
    hashes = _phase1_hashes(phase1_plan)
    if any(value[key] != item for key, item in hashes.items()):
        raise JobsError("phase-1 terminal observation source binding changed")
    request = signal.job_request(phase1_plan)
    jobs = value["jobs_api"]
    rayjob = value["rayjob"]
    pod = value["pod"]
    if set(jobs) != {"run_id", "name", "status", "run_dir", "requested_image"}:
        raise JobsError("Jobs API terminal observation fields changed")
    _uuid(jobs.get("run_id"), "Jobs API run id")
    expected_name = re.escape(request["name"]) + r"-[0-9a-f]{8}"
    if (
        not re.fullmatch(expected_name, str(jobs.get("name")))
        or jobs.get("status") != "SUCCEEDED"
        or jobs.get("run_dir") != request["run_dir"]
        or jobs.get("requested_image") != request["image"]
    ):
        raise JobsError("Jobs API did not prove exact successful phase-1 execution")
    if set(rayjob) != {
        "name",
        "uid",
        "namespace",
        "job_status",
        "deployment_status",
        "requested_image",
        "failure_alerts",
    }:
        raise JobsError("RayJob terminal observation fields changed")
    rayjob_uid = _uuid(rayjob.get("uid"), "RayJob uid")
    if (
        rayjob.get("name") != jobs["name"]
        or rayjob.get("namespace") != mechanics.NAMESPACE
        or rayjob.get("job_status") != "SUCCEEDED"
        or rayjob.get("deployment_status") != "Complete"
        or rayjob.get("requested_image") != request["image"]
        or rayjob.get("failure_alerts") != "off"
    ):
        raise JobsError("RayJob did not prove exact successful phase-1 execution")
    raycluster_uid = _uuid(value.get("raycluster_uid"), "RayCluster uid")
    _uuid(value.get("workload_uid"), "Workload uid")
    if set(pod) != {
        "name",
        "uid",
        "namespace",
        "phase",
        "restart_count",
        "exit_code",
        "termination_reason",
        "requested_image",
        "runtime_image_id",
        "owner_raycluster_uid",
        "root_rayjob_uid",
    }:
        raise JobsError("Pod terminal observation fields changed")
    _uuid(pod.get("uid"), "Pod uid")
    runtime_image = str(pod.get("runtime_image_id", "")).removeprefix("docker-pullable://")
    if (
        pod.get("namespace") != mechanics.NAMESPACE
        or not str(pod.get("name", "")).startswith(jobs["name"] + "-")
        or pod.get("phase") != "Succeeded"
        or pod.get("restart_count") != 0
        or pod.get("exit_code") != 0
        or pod.get("termination_reason") != "Completed"
        or pod.get("requested_image") != request["image"]
        or runtime_image != request["image"]
        or pod.get("owner_raycluster_uid") != raycluster_uid
        or pod.get("root_rayjob_uid") != rayjob_uid
    ):
        raise JobsError("Pod did not prove exact zero-restart phase-1 execution")
    return value


def validate_empty_census(
    value: dict[str, Any],
    terminal: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    _verify_seal(value, CENSUS_SCHEMA)
    if set(value) != {
        "schema",
        "observation_id",
        "observed_at",
        "context",
        "namespace",
        "jobs_api_run_id",
        "run_name",
        "rayjob_uid",
        "raycluster_uid",
        "workload_uid",
        "pod_uid",
        "root_query",
        "workload_query",
        "owner_ref_query",
        "sha256",
    }:
        raise JobsError("phase-1 resource census fields changed")
    _uuid(value.get("observation_id"), "census observation id")
    observed = _timestamp(value["observed_at"])
    age = (now - observed).total_seconds()
    if not 0 <= age <= MAX_CENSUS_AGE_SECONDS:
        raise JobsError("phase-1 resource census is stale or from the future")
    jobs = terminal["jobs_api"]
    rayjob = terminal["rayjob"]
    pod = terminal["pod"]
    if (
        value.get("context") != mechanics.PROD_CONTEXT
        or value.get("namespace") != mechanics.NAMESPACE
        or value.get("jobs_api_run_id") != jobs["run_id"]
        or value.get("run_name") != jobs["name"]
        or value.get("rayjob_uid") != rayjob["uid"]
        or value.get("raycluster_uid") != terminal["raycluster_uid"]
        or value.get("workload_uid") != terminal["workload_uid"]
        or value.get("pod_uid") != pod["uid"]
        or observed < _timestamp(terminal["observed_at"])
    ):
        raise JobsError("phase-1 resource census identity changed")
    expected_queries = {
        "root_query": {
            "selector": f"exact-root-name={jobs['name']}",
            "resource_version": value["root_query"].get("resource_version"),
            "items": [],
        },
        "workload_query": {
            "selector": f"owner-uid={rayjob['uid']}",
            "resource_version": value["workload_query"].get("resource_version"),
            "items": [],
        },
        "owner_ref_query": {
            "selector": "owner-uids="
            + ",".join(
                sorted(
                    {
                        rayjob["uid"],
                        terminal["raycluster_uid"],
                        terminal["workload_uid"],
                        pod["uid"],
                    }
                )
            ),
            "resource_version": value["owner_ref_query"].get("resource_version"),
            "items": [],
        },
    }
    for key, expected in expected_queries.items():
        query = value.get(key)
        if (
            not isinstance(query, dict)
            or set(query) != {"selector", "resource_version", "items"}
            or not isinstance(query.get("resource_version"), str)
            or not query["resource_version"]
            or query != expected
        ):
            raise JobsError("phase-1 resource census did not prove empty resources")
    return value


def build_authority(
    phase2_plan: dict[str, Any],
    phase2_request: dict[str, Any],
    terminal_observation: dict[str, Any],
    resource_censuses: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build only from actual SFS receipts plus exact terminal observations."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    phase2 = mechanics.validate_plan(phase2_plan)
    if phase2_request != mechanics.job_request(phase2):
        raise JobsError("phase-2 request differs from its exact plan")
    phase1 = _phase1_plan(phase2)
    terminal = validate_terminal_observation(terminal_observation, phase1)
    if len(resource_censuses) != 2:
        raise JobsError("phase-1 authority requires two empty resource censuses")
    censuses = [validate_empty_census(item, terminal, now=now) for item in resource_censuses]
    observed = [_timestamp(item["observed_at"]) for item in censuses]
    if (
        censuses[0]["observation_id"] == censuses[1]["observation_id"]
        or not MIN_CENSUS_SEPARATION_SECONDS
        <= (observed[1] - observed[0]).total_seconds()
        <= MAX_CENSUS_AGE_SECONDS
    ):
        raise JobsError("phase-1 empty censuses are not two ordered independent reads")
    source = collect_actual_source(phase2)
    body = {
        "schema": SCHEMA,
        "status": "review_ready_not_launch_authority_without_external_digest_pin",
        "validated_at": now.isoformat().replace("+00:00", "Z"),
        "phase2_plan_sha256": "sha256:" + mechanics.digest(phase2),
        "phase2_request_sha256": "sha256:" + digest(phase2_request),
        "phase1_source": source,
        "terminal_observation": terminal,
        "resource_censuses": censuses,
        "conditions": {
            "actual_sfs_receipts_revalidated": True,
            "jobs_api_rayjob_pod_terminal_success": True,
            "requested_image_equals_runtime_image_id": True,
            "zero_restarts_and_exit_zero": True,
            "zero_optimizer_updates": True,
            "checkpoint_artifacts_absent": True,
            "two_fresh_empty_resource_censuses": True,
        },
        "privacy": {
            "prompt_text_included": False,
            "reward_values_included": False,
            "verifier_code_included": False,
            "credentials_included": False,
        },
    }
    return _seal(body)


def validate_authority(
    phase2_plan: dict[str, Any],
    phase2_request: dict[str, Any],
    authority: dict[str, Any],
    *,
    expected_sha256: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate a root-reviewed authority pinned independently at invocation."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    phase2 = mechanics.validate_plan(phase2_plan)
    if phase2_request != mechanics.job_request(phase2):
        raise JobsError("phase-2 request differs from its exact plan")
    _verify_seal(authority, SCHEMA)
    if authority.get("sha256") != _sha(expected_sha256, "independent authority pin"):
        raise JobsError("phase-1 terminal authority lacks the independent digest pin")
    if set(authority) != {
        "schema",
        "status",
        "validated_at",
        "phase2_plan_sha256",
        "phase2_request_sha256",
        "phase1_source",
        "terminal_observation",
        "resource_censuses",
        "conditions",
        "privacy",
        "sha256",
    }:
        raise JobsError("phase-1 terminal authority fields changed")
    if (
        authority.get("status") != "review_ready_not_launch_authority_without_external_digest_pin"
        or authority.get("phase2_plan_sha256") != "sha256:" + mechanics.digest(phase2)
        or authority.get("phase2_request_sha256") != "sha256:" + digest(phase2_request)
    ):
        raise JobsError("phase-1 authority belongs to a different phase-2 POST")
    phase1 = _phase1_plan(phase2)
    source = authority["phase1_source"]
    expected_hashes = _phase1_hashes(phase1)
    if not isinstance(source, dict) or any(
        source.get(key) != item for key, item in expected_hashes.items()
    ):
        raise JobsError("phase-1 source hashes changed")
    if set(source) != {
        *expected_hashes,
        "public_evidence_sha256",
        "private_group_sha256",
        "native_terminal_sha256",
        "runtime_preflight_sha256",
        "receipt_files",
        "receipt_file_manifest_sha256",
        "receipt_file_count",
        "completed_episode_count",
        "distinct_finite_reward_count",
        "unique_verifier_execution_count",
        "terminal_slot_count",
        "optimizer_steps",
        "checkpoint_artifacts_absent",
        "actual_sfs_revalidation",
    }:
        raise JobsError("phase-1 source receipt fields changed")
    files = source.get("receipt_files")
    signal_evidence = phase2["task_signal_evidence"]
    if (
        source.get("public_evidence_sha256") != signal_evidence["sha256"]
        or source.get("private_group_sha256") != signal_evidence["source_receipt_sha256"]
        or source.get("native_terminal_sha256") != signal_evidence["native_terminal_sha256"]
        or source.get("runtime_preflight_sha256") != signal_evidence["runtime_preflight_sha256"]
        or not isinstance(files, dict)
        or source.get("receipt_file_count") != len(files)
        or source.get("receipt_file_manifest_sha256") != "sha256:" + digest(files)
        or any(_sha(item, "actual receipt file") != item for item in files.values())
        or source.get("completed_episode_count") != signal_evidence["completed_episode_count"]
        or source.get("distinct_finite_reward_count", 0) < 2
        or source.get("unique_verifier_execution_count")
        != signal_evidence["completed_episode_count"]
        or source.get("terminal_slot_count") != 8
        or source.get("optimizer_steps") != 0
        or source.get("checkpoint_artifacts_absent") is not True
        or source.get("actual_sfs_revalidation") is not True
    ):
        raise JobsError("phase-1 source receipt is incomplete or synthetic")
    # The reviewed digest is necessary but not sufficient: immediately before
    # POST, independently re-read and fully validate the exact SFS receipt
    # tree.  This prevents a self-sealed synthetic public evidence object (or a
    # copied review artifact whose source files later changed) from authorizing
    # learning.
    if collect_actual_source(phase2) != source:
        raise JobsError("phase-1 source no longer matches the actual SFS receipts")
    terminal = validate_terminal_observation(authority["terminal_observation"], phase1)
    censuses = authority["resource_censuses"]
    if not isinstance(censuses, list) or len(censuses) != 2:
        raise JobsError("phase-1 authority lacks two resource censuses")
    validated = [validate_empty_census(item, terminal, now=now) for item in censuses]
    observed = [_timestamp(item["observed_at"]) for item in validated]
    if (
        validated[0]["observation_id"] == validated[1]["observation_id"]
        or not MIN_CENSUS_SEPARATION_SECONDS
        <= (observed[1] - observed[0]).total_seconds()
        <= MAX_CENSUS_AGE_SECONDS
    ):
        raise JobsError("phase-1 empty censuses are not independent")
    if authority["conditions"] != {
        "actual_sfs_receipts_revalidated": True,
        "jobs_api_rayjob_pod_terminal_success": True,
        "requested_image_equals_runtime_image_id": True,
        "zero_restarts_and_exit_zero": True,
        "zero_optimizer_updates": True,
        "checkpoint_artifacts_absent": True,
        "two_fresh_empty_resource_censuses": True,
    } or authority["privacy"] != {
        "prompt_text_included": False,
        "reward_values_included": False,
        "verifier_code_included": False,
        "credentials_included": False,
    }:
        raise JobsError("phase-1 terminal authority claims changed")
    if _timestamp(authority["validated_at"]) > now:
        raise JobsError("phase-1 terminal authority is from the future")
    return authority
