"""Build and validate a prompt-free, easiest-first Qwen3.8 Fleet slate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import UUID

from evals.credential_rotation import (
    build_credential_rotation_receipt,
    validate_credential_rotation,
)

SELECTION_SCHEMA = "qwen38-fleet-ranked50-selection-v1"
EXCLUSIONS_SCHEMA = "qwen38-fleet-prior-attempt-exclusions-v1"
EXPECTED_COUNT = 50
EXPECTED_MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EXPECTED_HARNESS_VERSION = "0.22.3"
ALLOWED_SPLITS = {"train", "dev"}
RANKING = [
    "historical_pass_rate_desc",
    "historical_passes_desc",
    "historical_sessions_desc",
    "task_key_asc",
    "task_version_id_asc",
]
RANKING_INTERPRETATION = (
    "historical_frontier_capability_sweep_not_qwen_specific_difficulty_estimate"
)
TASK_BINDING_FIELDS = (
    "task_key",
    "task_version_id",
    "task_version",
    "environment_version_id",
    "env_key",
    "env_version",
    "data_key",
    "data_version",
    "split",
)
EXPECTED_ATTEMPTS = (
    (
        "chris-cyber-q38-qcode-reward-cal-p1-v2",
        1,
        "2b9ba166-6f43-44aa-894d-75761314c150",
        "valid_model_outcome",
    ),
    (
        "chris-cyber-q38-qcode-reward-cal-p1-v2",
        2,
        "25b185c5-6aee-479d-97ba-d94ed9df1f42",
        "unresolved_excluded",
    ),
    (
        "chris-cyber-q38-qcode-reward-cal-p1-v2",
        3,
        "4a4aba06-38d4-4fc7-83ee-b625fc934d00",
        "unresolved_excluded",
    ),
    (
        "chris-cyber-q38-qcode-reward-cal-p1-v2",
        4,
        "53669bab-1389-48cf-9d08-68e411f9cd78",
        "unresolved_excluded",
    ),
    (
        "chris-cyber-q38-qcode-positive-gate-p1-v3",
        1,
        "2c47ef51-29f9-4736-9916-18b748059cc6",
        "valid_model_outcome",
    ),
)
V2_EVIDENCE_PATH = (
    "docs/evidence/qwen38-study/2026-09-01-fleet-calibration-v2-infrastructure-incident.json"
)
V3_EVIDENCE_PATH = (
    "docs/evidence/qwen38-study/2026-09-01-fleet-calibration-v3-terminal-valid-zero.json"
)
V2_PLAN_PATH = "evals/fleet/configs/qwen38-27b-qwen-code-reward-calibration-pass1-v2.json"
V3_PLAN_PATH = "evals/fleet/configs/qwen38-27b-qwen-code-reward-calibration-pass1-v3.json"
RUN_ACCEPTANCE_SCHEMA = "fleet-qwen-code-ranked50-run-acceptance-v1"
FINAL_ACCEPTANCE_SCHEMA = "fleet-qwen-code-ranked50-final-acceptance-v1"
EXPECTED_SOURCE_JOB = "chris-cyber-qwen38-qcode-fleet-ranked50-base-v1"
EXPECTED_SOURCE_EXPERIMENT = "qwen38-qwen-code-fleet-ranked50-base-v1"
EXPECTED_EVALUATOR_IMAGE = (
    "ghcr.io/astral-sh/uv:python3.12-bookworm@"
    "sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7"
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Persist a receipt without following links or replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = canonical_json(value) + b"\n"
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_durable_run_tree(root: Path, acceptance: dict[str, Any]) -> None:
    manifest = acceptance.get("artifact_manifest") or {}
    if manifest.get("manifest_sha256") != digest_without(manifest, "manifest_sha256"):
        raise ValueError("ranked-50 durable artifact manifest digest is invalid")
    rows = manifest.get("files") or []
    expected: dict[str, str] = {}
    for row in rows:
        relative = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest))
            or relative in expected
        ):
            raise ValueError("ranked-50 durable artifact manifest row is invalid")
        expected[relative] = digest
    actual = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("ranked-50 durable run tree contains a symbolic link")
        if path.is_file():
            actual[str(path.relative_to(root))] = file_sha256(path)
    accepted_path = root / "ACCEPTED.json"
    if not accepted_path.is_file():
        raise ValueError("ranked-50 source ACCEPTED.json is missing")
    accepted_digest = file_sha256(accepted_path)
    if actual != {**expected, "ACCEPTED.json": accepted_digest}:
        raise ValueError("ranked-50 durable run tree contains unaccepted artifacts")


def treatment_identity(plan: dict[str, Any]) -> dict[str, Any]:
    model = plan.get("model") or {}
    execution = plan.get("execution") or {}
    return {
        "model": {
            field: model.get(field)
            for field in ("repository", "revision", "served_id", "endpoint_origin")
        },
        "harness": plan.get("harness"),
        "execution": {
            field: execution.get(field)
            for field in (
                "pass_k",
                "training_data_eligible",
                "required_task_tools",
                "required_task_tool_catalog_sha256",
            )
        },
    }


def _nonzero_uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is not a UUID")
    try:
        parsed = UUID(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a UUID") from exc
    if parsed.int == 0:
        raise ValueError(f"{label} is a zero UUID")
    return value


def _validate_exclusion_evidence(evidence: Any) -> None:
    expected = {
        "v2_incident_binding": V2_EVIDENCE_PATH,
        "v2_incident_raw_file_sha256": (
            "sha256:7a725154a95779c159beb90e82f4199fbbaea8e224241f1d8b29ec99d34e9821"
        ),
        "v2_binding_receipt_sha256": (
            "sha256:6f5b609d139c87a015263feac5f099df1a83e002d3146e038a98087ec8f9a1ad"
        ),
        "v3_terminal_binding": V3_EVIDENCE_PATH,
        "v3_terminal_binding_raw_file_sha256": (
            "sha256:372ed161068c2a98f5f22cf0dfc9deef34f595fb0e792a61dcbe765c8c3bdcc6"
        ),
        "v3_frozen_preflight_canonical_sha256": (
            "sha256:2dabbd5eb0677c3e5b62f4a00f1979db07ab0e57c0b3ddeb6ff1222f021f8f0d"
        ),
        "v3_terminal_summary_raw_file_sha256": (
            "sha256:8f8c42aea41355d151f75363d827c18a703068362ced10822ef8a3de2e447fa3"
        ),
        "v2_treatment_plan": V2_PLAN_PATH,
        "v2_treatment_plan_raw_file_sha256": (
            "sha256:2e5e075287f24162ca1381549790b3647c578e1e9e0f310bb3c33525428c13a8"
        ),
        "v3_treatment_plan": V3_PLAN_PATH,
        "v3_treatment_plan_raw_file_sha256": (
            "sha256:71dc09116ca376129000df78cc6bbd2c5f577b9ddfa5013f12886f1942adcb77"
        ),
    }
    if evidence != expected:
        raise ValueError("prior-attempt exclusion evidence binding drifted")
    repository = Path(__file__).resolve().parents[2]
    v2_path = repository / V2_EVIDENCE_PATH
    v3_path = repository / V3_EVIDENCE_PATH
    v2_plan_path = repository / V2_PLAN_PATH
    v3_plan_path = repository / V3_PLAN_PATH
    if file_sha256(v2_path) != expected["v2_incident_raw_file_sha256"]:
        raise ValueError("v2 incident evidence file digest drifted")
    if file_sha256(v3_path) != expected["v3_terminal_binding_raw_file_sha256"]:
        raise ValueError("v3 terminal evidence file digest drifted")
    if file_sha256(v2_plan_path) != expected["v2_treatment_plan_raw_file_sha256"]:
        raise ValueError("v2 treatment plan digest drifted")
    if file_sha256(v3_plan_path) != expected["v3_treatment_plan_raw_file_sha256"]:
        raise ValueError("v3 treatment plan digest drifted")
    v2 = load_json(v2_path)
    if v2.get("binding_receipt_sha256") != expected["v2_binding_receipt_sha256"]:
        raise ValueError("v2 incident evidence receipt digest drifted")
    v3 = load_json(v3_path)
    sources = v3.get("source_receipts") or {}
    if (
        sources.get("frozen_preflight_canonical_sha256")
        != expected["v3_frozen_preflight_canonical_sha256"]
        or sources.get("terminal_summary_raw_file_sha256")
        != expected["v3_terminal_summary_raw_file_sha256"]
    ):
        raise ValueError("v3 terminal source receipt binding drifted")
    if treatment_identity(load_json(v2_plan_path)) != treatment_identity(load_json(v3_plan_path)):
        raise ValueError("v2/v3 prior-attempt treatment identity disagrees")


def _validate_v3_terminal_outcome(row: dict[str, Any]) -> None:
    expected = {
        "run_id": "chris-cyber-q38-qcode-positive-gate-p1-v3-t01-8ad4f2ce",
        "status": "model_outcome",
        "evidence_run_id": "f42ab8f1-4d85-4040-88cc-22a677dd7e52",
        "session_id": "d993539e-ccc3-47f2-87b8-84d88cfe8dbe",
        "verifier_execution_id": "3e6a6376-3c5c-4911-8e54-3120cdaabea5",
        "qwen_exit_code": 0,
        "ingested_chunks": 10,
        "planned_chunks": 10,
        "cleanup_instance_closed": True,
        "cleanup_containers_removed": True,
        "authoritative_readback_confirmed": True,
    }
    if row.get("terminal_evidence") != expected:
        raise ValueError("v3 terminal outcome evidence drifted")
    for field in ("evidence_run_id", "session_id", "verifier_execution_id"):
        _nonzero_uuid(expected[field], f"v3 terminal {field}")


def validate_exclusions(receipt: dict[str, Any]) -> set[str]:
    if receipt.get("schema_version") != EXCLUSIONS_SCHEMA:
        raise ValueError("unsupported prior-attempt exclusion schema")
    if receipt.get("receipt_sha256") != digest_without(receipt, "receipt_sha256"):
        raise ValueError("prior-attempt exclusion receipt digest mismatch")
    if receipt.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise ValueError("prior-attempt exclusion model revision drifted")
    if receipt.get("harness_version") != EXPECTED_HARNESS_VERSION:
        raise ValueError("prior-attempt exclusion harness version drifted")
    treatment = receipt.get("treatment_identity")
    v3_plan = load_json(Path(__file__).resolve().parents[2] / V3_PLAN_PATH)
    if treatment != treatment_identity(v3_plan):
        raise ValueError("prior-attempt exclusion treatment identity drifted")
    if receipt.get("treatment_sha256") != sha256(canonical_json(treatment)):
        raise ValueError("prior-attempt exclusion treatment digest drifted")
    rows = receipt.get("attempts") or []
    if len(rows) != 5:
        raise ValueError("prior-attempt exclusion receipt must contain five known attempts")
    versions: set[str] = set()
    for row, (campaign_id, task_index, version_id, disposition) in zip(
        rows, EXPECTED_ATTEMPTS, strict=True
    ):
        if (
            row.get("campaign_id"),
            row.get("task_index"),
            row.get("task_version_id"),
            row.get("disposition"),
        ) != (campaign_id, task_index, version_id, disposition):
            raise ValueError("prior-attempt exclusion identity or order drifted")
        _nonzero_uuid(row.get("task_version_id"), "excluded task version")
        versions.add(row["task_version_id"])
        if disposition == "unresolved_excluded":
            if set(row) != {"campaign_id", "task_index", "task_version_id", "disposition"}:
                raise ValueError(
                    "unresolved prior attempt must not claim score or terminal evidence"
                )
        elif row.get("score") != 0:
            raise ValueError("known valid Qwen3.8 outcome must preserve score zero")
    if len(versions) != len(rows):
        raise ValueError("prior-attempt exclusion task versions are not unique")
    if set(rows[0]) != {
        "campaign_id",
        "task_index",
        "task_version_id",
        "disposition",
        "score",
    }:
        raise ValueError("v2 valid outcome contains unsupported fields")
    if set(rows[4]) != {
        "campaign_id",
        "task_index",
        "task_version_id",
        "disposition",
        "score",
        "terminal_evidence",
    }:
        raise ValueError("v3 valid outcome evidence is incomplete")
    _validate_v3_terminal_outcome(rows[4])
    _validate_exclusion_evidence(receipt.get("evidence"))
    if receipt.get("policy") != {
        "valid_model_outcomes_are_never_rerun": True,
        "unresolved_attempts_are_excluded_conservatively": True,
        "sealed_test_tasks_are_not_candidates": True,
    }:
        raise ValueError("prior-attempt exclusion policy drifted")
    return versions


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    historical = row["historical_ease"]
    return (
        -float(historical["pass_rate"]),
        -int(historical["passes"]),
        -int(historical["sessions"]),
        row["task_key"],
        row["task_version_id"],
    )


def authoritative_task_identity(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("export_schema") != "fleet_session_export_v1":
        raise ValueError("historical export record schema is unsupported")
    source = record.get("source") or {}
    roster = source.get("roster_task_binding") or {}
    task_key = roster.get("key")
    version_id = roster.get("eval_task_version_id")
    if not isinstance(task_key, str) or not task_key:
        raise ValueError("historical roster task key is missing")
    _nonzero_uuid(version_id, "historical roster task version")
    if source.get("task_key_from_roster") != task_key:
        raise ValueError("historical roster task-key projections disagree")
    transcript_task = (record.get("transcript_envelope") or {}).get("task") or {}
    if transcript_task.get("key") != task_key:
        raise ValueError("historical transcript and roster task keys disagree")
    # Historical transcript envelopes can contain a stale task-version projection.
    # The source roster binding is the immutable launch authority.
    return task_key, version_id


def build_final_acceptance(
    run_acceptance: dict[str, Any],
    job: dict[str, Any],
    pods: dict[str, Any],
    config_map: dict[str, Any],
    source_binding: dict[str, Any],
) -> dict[str, Any]:
    if run_acceptance.get("schema_version") != RUN_ACCEPTANCE_SCHEMA:
        raise ValueError("ranked-50 run acceptance schema is unsupported")
    if run_acceptance.get("acceptance_sha256") != digest_without(
        run_acceptance, "acceptance_sha256"
    ):
        raise ValueError("ranked-50 run acceptance digest is invalid")
    if run_acceptance.get("accepted") is not True:
        raise ValueError("ranked-50 source run is not accepted")
    if run_acceptance.get("campaign_id") != EXPECTED_SOURCE_JOB:
        raise ValueError("ranked-50 source campaign identity drifted")
    sha_fields = (
        "plan_sha256",
        "frozen_receipt_sha256",
        "selection_sha256",
        "prior_attempt_exclusions_sha256",
        "duplicate_preflight_sha256",
        "summary_sha256",
    )
    if any(
        not isinstance(run_acceptance.get(field), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", run_acceptance[field])
        for field in sha_fields
    ):
        raise ValueError("ranked-50 source immutable run binding is incomplete")
    if len(run_acceptance.get("outcomes") or []) != EXPECTED_COUNT:
        raise ValueError("ranked-50 source accepted outcome set is incomplete")
    if run_acceptance.get("fleet_account") != {
        "team_name": "fleet",
        "team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
    }:
        raise ValueError("ranked-50 source Fleet-team identity drifted")
    artifact_manifest = run_acceptance.get("artifact_manifest") or {}
    if artifact_manifest.get("manifest_sha256") != digest_without(
        artifact_manifest, "manifest_sha256"
    ):
        raise ValueError("ranked-50 source artifact manifest digest is invalid")
    runtime_images = run_acceptance.get("runtime_images") or {}
    if runtime_images.get("receipt_sha256") != digest_without(
        runtime_images, "receipt_sha256"
    ):
        raise ValueError("ranked-50 source runtime-image receipt digest is invalid")
    if run_acceptance.get("data_minimization") != {
        "prompts_included": False,
        "transcripts_included": False,
        "tool_content_included": False,
        "verifier_content_included": False,
        "flags_included": False,
        "credentials_included": False,
    }:
        raise ValueError("ranked-50 source data-minimization receipt drifted")
    metadata = job.get("metadata") or {}
    status = job.get("status") or {}
    spec = job.get("spec") or {}
    conditions = status.get("conditions") or []
    complete = [
        row for row in conditions if row.get("type") == "Complete" and row.get("status") == "True"
    ]
    failed = [
        row for row in conditions if row.get("type") == "Failed" and row.get("status") == "True"
    ]
    if len(complete) != 1 or failed or int(status.get("active") or 0) != 0:
        raise ValueError("ranked-50 source Job is not exclusively Complete")
    if int(status.get("succeeded") or 0) != 1:
        raise ValueError("ranked-50 source Job did not record exactly one success")
    pod_identity = run_acceptance.get("pod_identity") or {}
    if metadata.get("name") != EXPECTED_SOURCE_JOB or metadata.get("name") != pod_identity.get(
        "job_name"
    ):
        raise ValueError("ranked-50 source Job name drifted")
    labels = metadata.get("labels") or {}
    template = (spec.get("template") or {})
    template_metadata = template.get("metadata") or {}
    template_spec = template.get("spec") or {}
    if (
        labels.get("cyber-post-train.fleet.ai/experiment") != EXPECTED_SOURCE_EXPERIMENT
        or labels.get("cyber-post-train.fleet.ai/owner") != "chris"
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or (template_metadata.get("labels") or {}).get(
            "cyber-post-train.fleet.ai/experiment"
        )
        != EXPECTED_SOURCE_EXPERIMENT
        or spec.get("backoffLimit") != 0
        or spec.get("activeDeadlineSeconds") != 604800
        or spec.get("suspend") is not False
    ):
        raise ValueError("ranked-50 source Job reviewed configuration drifted")
    containers = template_spec.get("containers") or []
    evaluator = [row for row in containers if row.get("name") == "evaluator"]
    init_containers = template_spec.get("initContainers") or []
    dind = [row for row in init_containers if row.get("name") == "dind"]
    volumes = template_spec.get("volumes") or []
    config_volumes = [
        row
        for row in volumes
        if row.get("name") == "bootstrap"
        and (row.get("configMap") or {}).get("name") == EXPECTED_SOURCE_JOB
    ]
    sfs_volumes = [
        row
        for row in volumes
        if row.get("name") == "sfs"
        and (row.get("persistentVolumeClaim") or {}).get("claimName") == "sfs-shared"
    ]
    scratch_volumes = [
        row
        for row in volumes
        if row.get("name") == "scratch" and isinstance(row.get("emptyDir"), dict)
    ]
    if (
        len(evaluator) != 1
        or evaluator[0].get("image") != EXPECTED_EVALUATOR_IMAGE
        or evaluator[0].get("command") != ["/bin/bash", "-ceu", "--"]
        or len(evaluator[0].get("args") or []) != 1
        or "run_selfhosted_qwen38_ranked50.sh" not in evaluator[0]["args"][0]
        or "read_secret_snapshot_in_memory" not in evaluator[0]["args"][0]
        or "credential-rotation-receipt.json" not in evaluator[0]["args"][0]
        or template_spec.get("restartPolicy") != "Never"
        or template_spec.get("serviceAccountName") != EXPECTED_SOURCE_JOB
        or len(config_volumes) != 1
        or len(sfs_volumes) != 1
        or len(scratch_volumes) != 1
        or len(dind) != 1
        or dind[0].get("restartPolicy") != "Always"
        or (dind[0].get("securityContext") or {}).get("privileged") is not True
    ):
        raise ValueError("ranked-50 source Job image or ConfigMap binding drifted")
    env = {row.get("name"): row for row in evaluator[0].get("env") or []}
    pod_name_field_ref = (
        ((env.get("FLEET_EVAL_POD_NAME") or {}).get("valueFrom") or {}).get("fieldRef") or {}
    ).get("fieldPath")
    pod_uid_field_ref = (
        ((env.get("FLEET_EVAL_POD_UID") or {}).get("valueFrom") or {}).get("fieldRef") or {}
    ).get("fieldPath")
    if (
        "FLEET_API_KEY" in env
        or (env.get("FLEET_EVAL_JOB_NAME") or {}).get("value") != EXPECTED_SOURCE_JOB
        or (env.get("FLEET_EVAL_SCRATCH_ROOT") or {}).get("value") != "/scratch"
        or pod_name_field_ref != "metadata.name"
        or pod_uid_field_ref != "metadata.uid"
    ):
        raise ValueError("ranked-50 source Job runtime identity injection drifted")
    job_uid = _nonzero_uuid(metadata.get("uid"), "ranked-50 source Job UID")
    pod_rows = pods.get("items")
    if not isinstance(pod_rows, list) or len(pod_rows) != 1:
        raise ValueError("ranked-50 source Job must have exactly one Pod")
    pod_metadata = pod_rows[0].get("metadata") or {}
    pod_uid = _nonzero_uuid(pod_metadata.get("uid"), "ranked-50 source Pod UID")
    config_metadata = config_map.get("metadata") or {}
    config_uid = _nonzero_uuid(
        config_metadata.get("uid"), "ranked-50 source ConfigMap UID"
    )
    if (
        config_metadata.get("name") != EXPECTED_SOURCE_JOB
        or config_map.get("immutable") is not True
        or source_binding
        != {
            "schema_version": "fleet-qwen38-ranked50-acceptance-source-binding-v1",
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "configmap_uid": config_uid,
        }
    ):
        raise ValueError("ranked-50 immutable source binding drifted")
    config_data = config_map.get("data")
    if not isinstance(config_data, dict) or not config_data:
        raise ValueError("ranked-50 source ConfigMap data is missing")
    config_data_sha256 = sha256(canonical_json(config_data))
    if pod_metadata.get("name") != pod_identity.get("pod_name") or pod_uid != pod_identity.get(
        "pod_uid"
    ):
        raise ValueError("ranked-50 source Pod identity drifted")
    owner_references = pod_metadata.get("ownerReferences") or []
    expected_owner = [
        row
        for row in owner_references
        if row.get("apiVersion") == "batch/v1"
        and row.get("kind") == "Job"
        and row.get("name") == EXPECTED_SOURCE_JOB
        and row.get("uid") == job_uid
        and row.get("controller") is True
    ]
    pod_status = pods["items"][0].get("status") or {}
    container_statuses = pod_status.get("containerStatuses") or []
    evaluator_status = [row for row in container_statuses if row.get("name") == "evaluator"]
    terminated = ((evaluator_status[0].get("state") or {}).get("terminated") or {}) if len(
        evaluator_status
    ) == 1 else {}
    if (
        len(expected_owner) != 1
        or pod_status.get("phase") != "Succeeded"
        or len(evaluator_status) != 1
        or evaluator_status[0].get("restartCount") != 0
        or terminated.get("exitCode") != 0
    ):
        raise ValueError("ranked-50 source Pod terminal runtime evidence drifted")
    completion_time = status.get("completionTime")
    if not isinstance(completion_time, str) or not completion_time:
        raise ValueError("ranked-50 source Job completion time is missing")
    final = {
        "schema_version": FINAL_ACCEPTANCE_SCHEMA,
        "campaign_id": run_acceptance["campaign_id"],
        "accepted": True,
        "source_job": {
            "name": metadata["name"],
            "uid": job_uid,
            "completion_time": completion_time,
        },
        "source_pod": {
            "name": pod_metadata["name"],
            "uid": pod_uid,
            "phase": "Succeeded",
            "evaluator_exit_code": 0,
            "evaluator_restart_count": 0,
        },
        "source_config_map": {
            "name": config_metadata["name"],
            "uid": config_uid,
            "immutable": True,
            "data_sha256": config_data_sha256,
        },
        "source_run_acceptance_sha256": run_acceptance["acceptance_sha256"],
        "sealed_results_included": False,
    }
    final["acceptance_sha256"] = sha256(canonical_json(final))
    return final


def build_selection(
    split: dict[str, Any],
    export_path: Path,
    export_manifest: dict[str, Any],
    exclusions: dict[str, Any],
) -> dict[str, Any]:
    if split.get("schema") != "fleet_rl_task_split_v1":
        raise ValueError("unsupported Fleet split schema")
    if file_sha256(export_path) != export_manifest.get("sha256"):
        raise ValueError("private historical export digest mismatch")
    excluded = validate_exclusions(exclusions)
    source_job_id = (split.get("source") or {}).get("job_id")
    if export_manifest.get("jobs") != [source_job_id]:
        raise ValueError("historical export source job does not match the split")

    candidates = {
        row["task_version_id"]: row
        for row in split.get("tasks", [])
        if row.get("split") in ALLOWED_SPLITS and row.get("task_version_id") not in excluded
    }
    outcomes: dict[str, list[int]] = defaultdict(list)
    records = 0
    with export_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            if (record.get("source") or {}).get("job_id") != source_job_id:
                raise ValueError("historical export contains an unexpected source job")
            session = record.get("session") or {}
            task_key, version_id = authoritative_task_identity(record)
            if version_id not in candidates:
                continue
            if candidates[version_id]["task_key"] != task_key:
                raise ValueError("historical roster binding disagrees with the frozen split")
            verifier = session.get("verifier_execution") or {}
            if session.get("status") != "completed" or verifier.get("success") is not True:
                continue
            _nonzero_uuid(verifier.get("id"), "historical verifier execution")
            score = verifier.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("historical verifier score is not numeric")
            numeric = float(score)
            if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
                raise ValueError("historical verifier score is not binary")
            outcomes[version_id].append(int(numeric))
    if records != export_manifest.get("sessions"):
        raise ValueError("historical export record count does not match its manifest")

    ranked: list[dict[str, Any]] = []
    for version_id, scores in outcomes.items():
        source = candidates[version_id]
        row = {field: source[field] for field in TASK_BINDING_FIELDS}
        row["historical_ease"] = {
            "sessions": len(scores),
            "passes": sum(scores),
            "pass_rate": sum(scores) / len(scores),
        }
        ranked.append(row)
    ranked.sort(key=_rank_key)
    selected = ranked[:EXPECTED_COUNT]
    if len(selected) != EXPECTED_COUNT:
        raise ValueError("fewer than fifty eligible task versions have historical evidence")
    for index, row in enumerate(selected, 1):
        row["rank"] = index

    receipt: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "source_job_id": source_job_id,
        "split_manifest_digest": split["manifest_digest"],
        "historical_export": {
            "sessions": export_manifest["sessions"],
            "sha256": export_manifest["sha256"],
        },
        "prior_attempt_exclusions_sha256": exclusions["receipt_sha256"],
        "allowed_splits": sorted(ALLOWED_SPLITS),
        "ranking": RANKING,
        "ranking_interpretation": RANKING_INTERPRETATION,
        "selected_count": EXPECTED_COUNT,
        "tasks": selected,
        "privacy": {
            "prompts_included": False,
            "transcripts_included": False,
            "tool_content_included": False,
            "verifier_content_included": False,
        },
    }
    receipt["selection_sha256"] = sha256(canonical_json(receipt))
    return receipt


def validate_selection(
    selection: dict[str, Any],
    split: dict[str, Any],
    exclusions: dict[str, Any],
) -> list[dict[str, Any]]:
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unsupported ranked-50 selection schema")
    if selection.get("selection_sha256") != digest_without(selection, "selection_sha256"):
        raise ValueError("ranked-50 selection digest mismatch")
    if selection.get("split_manifest_digest") != split.get("manifest_digest"):
        raise ValueError("ranked-50 selection does not pin this split")
    if selection.get("source_job_id") != (split.get("source") or {}).get("job_id"):
        raise ValueError("ranked-50 source job does not match the split")
    if selection.get("allowed_splits") != sorted(ALLOWED_SPLITS):
        raise ValueError("ranked-50 selection must be train/dev only")
    if selection.get("ranking") != RANKING:
        raise ValueError("ranked-50 selection ordering policy drifted")
    if selection.get("ranking_interpretation") != RANKING_INTERPRETATION:
        raise ValueError("ranked-50 selection interpretation drifted")
    if selection.get("selected_count") != EXPECTED_COUNT:
        raise ValueError("ranked-50 selection must declare exactly fifty tasks")
    if selection.get("prior_attempt_exclusions_sha256") != exclusions.get("receipt_sha256"):
        raise ValueError("ranked-50 selection exclusion receipt drifted")
    excluded = validate_exclusions(exclusions)
    tasks = selection.get("tasks") or []
    if len(tasks) != EXPECTED_COUNT:
        raise ValueError("ranked-50 selection must contain exactly fifty tasks")
    if len({row.get("task_version_id") for row in tasks}) != EXPECTED_COUNT:
        raise ValueError("ranked-50 task versions are not unique")
    if {row["task_version_id"] for row in tasks} & excluded:
        raise ValueError("ranked-50 selection repeats a prior Qwen3.8 attempt")
    split_by_version = {row["task_version_id"]: row for row in split.get("tasks", [])}
    for index, row in enumerate(tasks, 1):
        if row.get("rank") != index:
            raise ValueError("ranked-50 task ranks are not contiguous")
        source = split_by_version.get(row.get("task_version_id"))
        if source is None or source.get("split") not in ALLOWED_SPLITS:
            raise ValueError("ranked-50 task is absent from train/dev")
        if {field: row.get(field) for field in TASK_BINDING_FIELDS} != {
            field: source.get(field) for field in TASK_BINDING_FIELDS
        }:
            raise ValueError("ranked-50 task binding drifted from the frozen split")
        historical = row.get("historical_ease") or {}
        sessions = historical.get("sessions")
        passes = historical.get("passes")
        pass_rate = historical.get("pass_rate")
        if not isinstance(sessions, int) or sessions <= 0:
            raise ValueError("ranked-50 historical session count is invalid")
        if not isinstance(passes, int) or not 0 <= passes <= sessions:
            raise ValueError("ranked-50 historical pass count is invalid")
        if pass_rate != passes / sessions:
            raise ValueError("ranked-50 historical pass rate is inconsistent")
    if tasks != sorted(tasks, key=_rank_key):
        raise ValueError("ranked-50 tasks are not easiest-first")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("build", "validate", "credential-receipt", "credential-gate", "finalize"),
    )
    parser.add_argument("--split", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--export-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--credential-receipt", type=Path)
    parser.add_argument("--secret-metadata", type=Path)
    parser.add_argument("--rotation-not-before")
    parser.add_argument("--before-metadata", type=Path)
    parser.add_argument("--after-metadata", type=Path)
    parser.add_argument("--confirmed-by")
    parser.add_argument("--rotation-completed-at")
    parser.add_argument("--run-acceptance", type=Path)
    parser.add_argument("--job-json", type=Path)
    parser.add_argument("--pods-json", type=Path)
    parser.add_argument("--configmap-json", type=Path)
    parser.add_argument("--source-binding", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if args.command == "credential-receipt":
        if (
            not args.before_metadata
            or not args.after_metadata
            or not args.confirmed_by
            or not args.rotation_completed_at
            or not args.output
        ):
            parser.error(
                "credential-receipt requires --before-metadata, --after-metadata, "
                "--confirmed-by, --rotation-completed-at, and --output"
            )
        receipt = build_credential_rotation_receipt(
            load_json(args.before_metadata),
            load_json(args.after_metadata),
            confirmed_by=args.confirmed_by,
            rotation_completed_at=args.rotation_completed_at,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(receipt) + b"\n")
        return 0
    if args.command == "credential-gate":
        if not args.credential_receipt or not args.secret_metadata or not args.rotation_not_before:
            parser.error(
                "credential-gate requires --credential-receipt, --secret-metadata, "
                "and --rotation-not-before"
            )
        validate_credential_rotation(
            load_json(args.credential_receipt),
            load_json(args.secret_metadata),
            rotation_not_before=args.rotation_not_before,
        )
        return 0
    if args.command == "finalize":
        if (
            not args.run_acceptance
            or not args.job_json
            or not args.pods_json
            or not args.configmap_json
            or not args.source_binding
            or not args.run_root
            or not args.output
        ):
            parser.error(
                "finalize requires --run-acceptance, --job-json, --pods-json, "
                "--configmap-json, --source-binding, --run-root, and --output"
            )
        run_acceptance = load_json(args.run_acceptance)
        validate_durable_run_tree(args.run_root, run_acceptance)
        final = build_final_acceptance(
            run_acceptance,
            load_json(args.job_json),
            load_json(args.pods_json),
            load_json(args.configmap_json),
            load_json(args.source_binding),
        )
        write_json_once(args.output, final)
        return 0
    if not args.split or not args.exclusions:
        parser.error("build and validate require --split and --exclusions")
    split = load_json(args.split)
    exclusions = load_json(args.exclusions)
    if args.command == "build":
        if not args.export or not args.export_manifest or not args.output:
            parser.error("build requires --export, --export-manifest, and --output")
        selection = build_selection(
            split,
            args.export,
            load_json(args.export_manifest),
            exclusions,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(selection) + b"\n")
    else:
        if not args.selection:
            parser.error("validate requires --selection")
        selection = load_json(args.selection)
        validate_selection(selection, split, exclusions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
