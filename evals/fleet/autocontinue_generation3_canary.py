"""Fail-closed generation-3 successor for pre-Pod generation-2 admission failures."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import autocontinue_canary_controller as legacy
from evals.fleet import autocontinue_generation2_authority_v5 as generation2_release
from evals.fleet import autocontinue_generation2_canary_v3 as generation2
from evals.fleet import exact_pass4_universe as exact
from evals.fleet import self_hosted

SPEC_SCHEMA = "fleet-opencode-autocontinue-generation3-canary-spec-v1"
HELD_SCHEMA = "fleet-opencode-autocontinue-generation3-canary-held-release-v1"
CLAIM_SCHEMA = "fleet-statistical-cell-execution-claim-v3"
TERMINAL_SCHEMA = "fleet-opencode-autocontinue-generation3-terminal-v1"
INCIDENT_SCHEMA = "fleet-opencode-autocontinue-generation2-admission-failure-v1"
TOMBSTONE_SCHEMA = "fleet-opencode-autocontinue-generation2-admission-tombstones-v1"

INCIDENT_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-admission-failure-v1.json"
)
INCIDENT_FILE_SHA = "sha256:c0c726cd72ee061e924639e36db260c46e3283d6265ae1935d6a4250ffc417cc"
INCIDENT_SHA = "sha256:99bbde2c57290cad719a436c054deab12675257ac84f8dfb5471787aa2e5ee0a"
TOMBSTONE_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation2-admission-tombstones-v1.json"
)
TOMBSTONE_FILE_SHA = "sha256:38939ff1884b66b72caa266da09cbdeb67f525dc075abbc3fc8e268f900cab63"
TOMBSTONE_SHA = "sha256:348d8ebe2a80eed139e9f57a6c796f95c3f34b28040aea4aaaf8fb0b5b9a156a"
HELD_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-generation3-canaries-held-v1.json"
)
MODULE_PATH = "evals/fleet/autocontinue_generation3_canary.py"
MANIFEST_PATH = "evals/fleet/cluster/opencode-autocontinue-generation3-canaries-v1.yaml"
RUN_PATH = "evals/fleet/scripts/run_opencode_autocontinue_generation3_canary_v1.sh"
SUBMIT_PATH = "evals/fleet/scripts/submit_opencode_autocontinue_generation3_canaries_v1.sh"
CLAIM_ROOT = "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1"
INTENT_NAME = "chris-ac-g3-canary-submit-v1"

G2_SPEC_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation2-v2.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation2-v2.json",
}
G3_SPEC_PATHS = {
    "qwen3.8-27b": "evals/fleet/configs/qwen38-opencode-autocontinue-canary-generation3-v1.json",
    "glm-5.3": "evals/fleet/configs/glm53-opencode-autocontinue-canary-generation3-v1.json",
}
G2_RELEASE_PATHS = {
    "qwen3.8-27b": (
        "docs/evidence/qwen38-study/"
        "2026-09-04-qwen38-autocontinue-generation2-scoring-release-v5.json"
    ),
    "glm-5.3": (
        "docs/evidence/qwen38-study/"
        "2026-09-04-glm53-autocontinue-generation2-scoring-release-v5.json"
    ),
}
EXPECTED = {
    "qwen3.8-27b": {
        "job_name": "chris-q38-ac-r004-a1-g3-v1",
        "configmap_name": "chris-q38-ac-r004-a1-g3-run-v1",
        "run_id": "chris-q38-ac-g3-r004-a1-02dd4e3f",
        "network": "q38-ac-g3-r004-a1-02dd4e3f",
        "generation2_job_uid": "2c2e6346-4d8f-49ba-8890-e5cf59090976",
    },
    "glm-5.3": {
        "job_name": "chris-glm53-ac-r013-a1-g3-v1",
        "configmap_name": "chris-glm53-ac-r013-a1-g3-run-v1",
        "run_id": "chris-glm53-ac-g3-r013-a1-9375a9b9",
        "network": "glm53-ac-g3-r013-a1-9375a9b9",
        "generation2_job_uid": "4af2be6d-668d-4968-a8d7-439118ab0375",
    },
}


def load(path: Path) -> dict[str, Any]:
    return generation2.load(path)


def digest(value: dict[str, Any], field: str) -> str:
    return generation2.digest(value, field)


def file_sha256(path: Path) -> str:
    return generation2.file_sha256(path)


def _canonical_utc(value: Any) -> bool:
    return generation2._canonical_utc(value)


def _incident_expected() -> dict[str, Any]:
    return {
        "schema_version": INCIDENT_SCHEMA,
        "append_only": True,
        "status": "INFRASTRUCTURE_INVALID_PRE_POD",
        "observed_at_utc": "2026-09-05T05:00:43Z",
        "classification": "priority_class_preemption_policy_conflict",
        "namespace": "fleet-train-jobs",
        "submission": {
            "intent_configmap": {
                "name": "chris-ac-g2-canary-submit-v1",
                "uid": "b1779942-7f29-49fa-9b16-4de557852c54",
                "immutable": True,
                "created_at_utc": "2026-09-05T04:57:58Z",
            },
            "runtime_configmaps": [
                {
                    "name": "chris-q38-ac-r004-a1-g2-run-v1",
                    "uid": "af3a9671-fd41-4fe7-b434-f39d0bf192ef",
                    "immutable": True,
                    "created_at_utc": "2026-09-05T04:58:01Z",
                },
                {
                    "name": "chris-glm53-ac-r013-a1-g2-run-v1",
                    "uid": "c07f5793-dc76-472d-892e-4e5b3d21d002",
                    "immutable": True,
                    "created_at_utc": "2026-09-05T04:58:04Z",
                },
            ],
        },
        "priority_classes": {
            "requested": {
                "name": "fleet-train-high",
                "value": 10000,
                "preemption_policy": "PreemptLowerPriority",
            },
            "highest_installed_never": {
                "name": "fleet-serve-low",
                "value": 100,
                "preemption_policy": "Never",
            },
        },
        "jobs": [],
        "side_effects": {
            "model_calls": 0,
            "sessions": 0,
            "verifier_executions": 0,
            "generation2_claims": 0,
            "output_roots": 0,
        },
        "retry_disposition": {
            "same_jobs_must_not_be_recreated": True,
            "same_execution_generation_must_not_be_reused": True,
            "fresh_execution_generation_required": 3,
            "same_statistical_cells_preserved": True,
        },
        "privacy": {
            "logs_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }


def validate_incident(receipt: dict[str, Any], root: Path) -> None:
    expected = _incident_expected()
    for model in ("qwen3.8-27b", "glm-5.3"):
        g2_spec = load(root / G2_SPEC_PATHS[model])
        generation2.validate_spec(g2_spec, root)
        release_path = root / G2_RELEASE_PATHS[model]
        release = load(release_path)
        generation2_release.validate_release(
            release,
            g2_spec,
            load(root / generation2_release.AUTH_PATH),
            load(root / generation2_release.HELD_PATH),
            root,
            release["package_commit"],
        )
        event = {
            "reason": "FailedCreate",
            "first_at_utc": "2026-09-05T04:58:06Z",
            "last_at_utc": "2026-09-05T04:59:09Z",
            "event_count": 7,
            "error_class": "explicit_Never_conflicts_with_priority_class_PreemptLowerPriority",
        }
        expected["jobs"].append(
            {
                "model": model,
                "name": generation2_release.held_v1.EXPECTED[model]["job_name"],
                "uid": EXPECTED[model]["generation2_job_uid"],
                "release_file_sha256": file_sha256(release_path),
                "release_receipt_sha256": release["receipt_sha256"],
                "kueue_admitted": True,
                "cluster_queue": "training-cq",
                "failed_create": event,
                "pods_created": 0,
                "current_job_objects": 0,
            }
        )
    if (
        file_sha256(root / INCIDENT_PATH) != INCIDENT_FILE_SHA
        or receipt.get("receipt_sha256") != INCIDENT_SHA
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {k: v for k, v in receipt.items() if k != "receipt_sha256"} != expected
    ):
        raise ValueError("generation-2 admission incident drifted")


def validate_tombstones(receipt: dict[str, Any], root: Path) -> None:
    incident = load(root / INCIDENT_PATH)
    validate_incident(incident, root)
    rows = []
    for model in ("qwen3.8-27b", "glm-5.3"):
        spec = load(root / G2_SPEC_PATHS[model])
        plan = generation2.validate_spec(spec, root)
        expected = EXPECTED[model]
        execution_id = spec["execution"]["execution_id"]
        job_name = generation2_release.held_v1.EXPECTED[model]["job_name"]
        rows.append(
            {
                "model": model,
                "cell_id": spec["statistical_cell"]["cell_id"],
                "generation2_spec_sha256": spec["generation2_spec_sha256"],
                "generation2_plan_sha256": plan["plan_sha256"],
                "generation2_execution_id": execution_id,
                "generation2_job": {
                    "name": job_name,
                    "uid": expected["generation2_job_uid"],
                    "current_objects": 0,
                    "pods_created": 0,
                },
                "generation2_output_root": {
                    "path": f"/mnt/sfs/jobs/{job_name}",
                    "absent": True,
                },
                "generation2_execution_claim": {
                    "path": f"{CLAIM_ROOT}/{execution_id.removeprefix('sha256:')}.json",
                    "absent": True,
                },
                "authoritative_api": {
                    "exact_treatment_sessions": 0,
                    "planned_run_id_collisions": 0,
                },
                "replacement": {
                    "permitted_execution_generation": 3,
                    "same_cell": True,
                    "generation2_nonrepeatable": True,
                },
            }
        )
    expected_receipt = {
        "schema_version": TOMBSTONE_SCHEMA,
        "append_only": True,
        "status": "SEALED_PRE_MODEL_INFRASTRUCTURE_TOMBSTONES",
        "observed_at_utc": "2026-09-05T05:00:43Z",
        "incident": {
            "path": INCIDENT_PATH,
            "file_sha256": INCIDENT_FILE_SHA,
            "receipt_sha256": INCIDENT_SHA,
        },
        "models": rows,
        "privacy": {
            "logs_included": False,
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        file_sha256(root / TOMBSTONE_PATH) != TOMBSTONE_FILE_SHA
        or receipt.get("receipt_sha256") != TOMBSTONE_SHA
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        != expected_receipt
    ):
        raise ValueError("generation-2 admission tombstones drifted")


def _render_plan(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-3 model is unsupported")
    old_spec = load(root / G2_SPEC_PATHS[model])
    old_plan = generation2.validate_spec(old_spec, root)
    expected = EXPECTED[model]
    plan = copy.deepcopy(old_plan)
    plan["campaign_id"] = expected["job_name"]
    plan["source_job_id"] = expected["job_name"]
    plan["preflight_job_name"] = expected["job_name"] + "-preflight"
    plan["scored_job_name"] = expected["job_name"]
    plan["sfs_root"] = expected["job_name"]
    plan["attempts"][0].update(
        {
            "run_id": expected["run_id"],
            "network": expected["network"],
            "execution_generation": 3,
        }
    )
    plan["execution"].update(
        {
            "retry_policy": "generation3_only_after_exact_generation2_pre_pod_tombstone",
            "required_priority_class": "fleet-serve-low",
            "execution_generation": 3,
            "generation_claim_root": CLAIM_ROOT,
            "old_claims_preserved": True,
            "launch_authorized": False,
        }
    )
    plan["source"] = {
        "exact_pass4_campaign_path": old_plan["source"]["exact_pass4_campaign_path"],
        "statistical_cell_id": old_spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "generation2_spec_sha256": old_spec["generation2_spec_sha256"],
        "generation2_plan_sha256": old_plan["plan_sha256"],
        "generation2_incident_receipt_sha256": INCIDENT_SHA,
        "generation2_tombstone_receipt_sha256": TOMBSTONE_SHA,
    }
    plan["plan_sha256"] = digest(plan, "plan_sha256")
    return plan


def validate_spec(spec: dict[str, Any], root: Path) -> dict[str, Any]:
    model = spec.get("model")
    if model not in EXPECTED:
        raise ValueError("generation-3 model is unsupported")
    incident = load(root / INCIDENT_PATH)
    tombstones = load(root / TOMBSTONE_PATH)
    validate_incident(incident, root)
    validate_tombstones(tombstones, root)
    old_spec = load(root / G2_SPEC_PATHS[model])
    old_plan = generation2.validate_spec(old_spec, root)
    expected = EXPECTED[model]
    execution = exact.execution_for(old_spec["statistical_cell"]["cell_id"], 3)
    expected_spec = {
        "schema_version": SPEC_SCHEMA,
        "model": model,
        "statistical_cell": old_spec["statistical_cell"],
        "execution": execution,
        "predecessor_generation2": {
            "spec_path": G2_SPEC_PATHS[model],
            "spec_sha256": old_spec["generation2_spec_sha256"],
            "plan_sha256": old_plan["plan_sha256"],
            "job_uid": expected["generation2_job_uid"],
        },
        "supersession": {
            "incident_path": INCIDENT_PATH,
            "incident_file_sha256": INCIDENT_FILE_SHA,
            "incident_receipt_sha256": INCIDENT_SHA,
            "tombstone_path": TOMBSTONE_PATH,
            "tombstone_file_sha256": TOMBSTONE_FILE_SHA,
            "tombstone_receipt_sha256": TOMBSTONE_SHA,
            "prior_generation": 2,
            "new_generation": 3,
            "same_statistical_cell": True,
            "generation2_nonrepeatable": True,
        },
        "identities": {
            "job_name": expected["job_name"],
            "configmap_name": expected["configmap_name"],
            "intent_configmap_name": INTENT_NAME,
            "run_id": expected["run_id"],
            "network": expected["network"],
            "sfs_root": f"/mnt/sfs/jobs/{expected['job_name']}",
            "generation_claim_root": CLAIM_ROOT,
        },
        "rendered_plan_sha256": spec.get("rendered_plan_sha256"),
        "launch_authorized": False,
    }
    if (
        spec.get("generation3_spec_sha256") != digest(spec, "generation3_spec_sha256")
        or {k: v for k, v in spec.items() if k != "generation3_spec_sha256"}
        != expected_spec
    ):
        raise ValueError("generation-3 specification drifted")
    plan = _render_plan(spec, root)
    if plan["plan_sha256"] != spec["rendered_plan_sha256"]:
        raise ValueError("generation-3 rendered plan digest drifted")
    settings = self_hosted.opencode_settings(plan)
    canonical = self_hosted.canonical_json(settings)
    if (
        plan["harness"]["settings_canonical_sha256"] != self_hosted.sha256(canonical)
        or plan["harness"]["settings_file_sha256"]
        != self_hosted.sha256(canonical + b"\n")
        or settings.get("compaction") != {"auto": True, "reserved": 20000}
        or "plugin" in settings
        or plan["execution"]["required_task_tools"] != ["bash", "submit_report"]
    ):
        raise ValueError("generation-3 executable treatment renderer drifted")
    return plan


def validate_held(receipt: dict[str, Any], root: Path) -> None:
    specs = [load(root / G3_SPEC_PATHS[m]) for m in ("qwen3.8-27b", "glm-5.3")]
    plans = [validate_spec(spec, root) for spec in specs]
    expected = {
        "schema_version": HELD_SCHEMA,
        "append_only": True,
        "status": "HELD",
        "observed_at_utc": receipt.get("observed_at_utc"),
        "incident": {
            "path": INCIDENT_PATH,
            "file_sha256": INCIDENT_FILE_SHA,
            "receipt_sha256": INCIDENT_SHA,
        },
        "tombstones": {
            "path": TOMBSTONE_PATH,
            "file_sha256": TOMBSTONE_FILE_SHA,
            "receipt_sha256": TOMBSTONE_SHA,
        },
        "generation3_spec_sha256s": [s["generation3_spec_sha256"] for s in specs],
        "rendered_plan_sha256s": [p["plan_sha256"] for p in plans],
        "jobs": [EXPECTED[s["model"]]["job_name"] for s in specs],
        "execution_generation": 3,
        "statistical_cells": 2,
        "priority": {
            "class": "fleet-serve-low",
            "value": 100,
            "preemption_policy": "Never",
            "highest_installed_never_class": True,
        },
        "implementation": {
            "module_path": MODULE_PATH,
            "module_sha256": file_sha256(root / MODULE_PATH),
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": file_sha256(root / MANIFEST_PATH),
            "run_path": RUN_PATH,
            "run_sha256": file_sha256(root / RUN_PATH),
            "submit_path": SUBMIT_PATH,
            "submit_sha256": file_sha256(root / SUBMIT_PATH),
        },
        "launch_authorized": False,
        "bulk_release_authorized": False,
        "dedicated_serving_authorized": False,
        "remaining_gates": [
            "independent_generation3_package_audit",
            "fresh_hosted_route_and_duplicate_inventory",
            "append_only_model_specific_generation3_scoring_releases",
            "explicit_root_launch_authorization",
        ],
        "privacy": {
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        },
    }
    if (
        not _canonical_utc(receipt.get("observed_at_utc"))
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {k: v for k, v in receipt.items() if k != "receipt_sha256"} != expected
    ):
        raise ValueError("generation-3 held release drifted")


def validate_preclaim_tombstone(spec: dict[str, Any], root: Path) -> None:
    validate_tombstones(load(root / TOMBSTONE_PATH), root)
    old = next(
        row for row in load(root / TOMBSTONE_PATH)["models"] if row["model"] == spec["model"]
    )
    if Path(old["generation2_output_root"]["path"]).exists() or Path(
        old["generation2_execution_claim"]["path"]
    ).exists():
        raise RuntimeError("generation-2 output or execution claim appeared after tombstone")


def claim_execution_generation(
    spec: dict[str, Any], plan: dict[str, Any], *, root: Path, claim_root: Path | None = None
) -> dict[str, Any]:
    expected_plan = validate_spec(spec, root)
    if plan != expected_plan:
        raise ValueError("generation-3 spec-plan chain drifted")
    if not legacy._is_uuid(os.environ.get("JOB_UID")) or not legacy._is_uuid(
        os.environ.get("POD_UID")
    ):
        raise RuntimeError("generation-3 claim requires downward API UIDs")
    validate_preclaim_tombstone(spec, root)
    claim_dir = claim_root or Path(CLAIM_ROOT)
    root_fd = legacy._open_directory_nofollow(claim_dir)
    try:
        lock_fd = legacy._open_claim_lock(root_fd)
        with os.fdopen(lock_fd, "a+b") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise RuntimeError("generation-3 claim lock is not regular")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            name = spec["execution"]["execution_id"].removeprefix("sha256:") + ".json"
            try:
                os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise RuntimeError("generation-3 execution is already claimed")
            receipt = {
                "schema_version": CLAIM_SCHEMA,
                "generation3_spec_sha256": spec["generation3_spec_sha256"],
                "plan_sha256": plan["plan_sha256"],
                "cell_id": spec["statistical_cell"]["cell_id"],
                "execution_id": spec["execution"]["execution_id"],
                "execution_generation": 3,
                "run_id": plan["attempts"][0]["run_id"],
                "job_uid": os.environ["JOB_UID"],
                "pod_uid": os.environ["POD_UID"],
                "claimed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "generation2_tombstone_receipt_sha256": TOMBSTONE_SHA,
                "prior_claims_preserved": True,
                "immutable": True,
                "automatic_retry": False,
                "scores_included": False,
                "prompts_or_traces_included": False,
            }
            receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(name, flags, 0o600, dir_fd=root_fd)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise RuntimeError("generation-3 claim path is unsafe")
                with os.fdopen(fd, "wb") as stored:
                    fd = -1
                    stored.write(self_hosted.canonical_json(receipt) + b"\n")
                    stored.flush()
                    os.fsync(stored.fileno())
            finally:
                if fd >= 0:
                    os.close(fd)
            os.fsync(root_fd)
            validate_claim(receipt, spec, plan, root=root)
            return receipt
    finally:
        os.close(root_fd)


def validate_claim(
    claim: dict[str, Any], spec: dict[str, Any], plan: dict[str, Any], *, root: Path
) -> None:
    if plan != validate_spec(spec, root):
        raise ValueError("generation-3 claim spec-plan chain drifted")
    expected = {
        "schema_version": CLAIM_SCHEMA,
        "generation3_spec_sha256": spec["generation3_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 3,
        "run_id": plan["attempts"][0]["run_id"],
        "job_uid": claim.get("job_uid"),
        "pod_uid": claim.get("pod_uid"),
        "claimed_at_utc": claim.get("claimed_at_utc"),
        "generation2_tombstone_receipt_sha256": TOMBSTONE_SHA,
        "prior_claims_preserved": True,
        "immutable": True,
        "automatic_retry": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if (
        not legacy._is_uuid(claim.get("job_uid"))
        or not legacy._is_uuid(claim.get("pod_uid"))
        or not _canonical_utc(claim.get("claimed_at_utc"))
        or claim.get("receipt_sha256") != digest(claim, "receipt_sha256")
        or {k: v for k, v in claim.items() if k != "receipt_sha256"} != expected
    ):
        raise ValueError("generation-3 claim is not authoritative")


def terminal_receipt(
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim: dict[str, Any],
    result: dict[str, Any],
    *,
    root: Path,
    terminal_at_utc: str | None = None,
) -> dict[str, Any]:
    if plan != validate_spec(spec, root):
        raise ValueError("generation-3 terminal spec-plan chain drifted")
    validate_claim(claim, spec, plan, root=root)
    validated_result = generation2._validated_result(result, plan)
    receipt = {
        "schema_version": TERMINAL_SCHEMA,
        "generation3_spec_sha256": spec["generation3_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 3,
        "generation_claim_receipt_sha256": claim["receipt_sha256"],
        "job_uid": claim["job_uid"],
        "pod_uid": claim["pod_uid"],
        "terminal_at_utc": terminal_at_utc
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "result": validated_result,
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    receipt["receipt_sha256"] = digest(receipt, "receipt_sha256")
    validate_terminal(receipt, spec, plan, claim, root=root)
    return receipt


def validate_terminal(
    receipt: dict[str, Any],
    spec: dict[str, Any],
    plan: dict[str, Any],
    claim: dict[str, Any],
    *,
    root: Path,
) -> None:
    validate_claim(claim, spec, plan, root=root)
    result = generation2._validated_result(receipt.get("result", {}), plan)
    expected = {
        "schema_version": TERMINAL_SCHEMA,
        "generation3_spec_sha256": spec["generation3_spec_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "cell_id": spec["statistical_cell"]["cell_id"],
        "execution_id": spec["execution"]["execution_id"],
        "execution_generation": 3,
        "generation_claim_receipt_sha256": claim["receipt_sha256"],
        "job_uid": claim["job_uid"],
        "pod_uid": claim["pod_uid"],
        "terminal_at_utc": receipt.get("terminal_at_utc"),
        "result": result,
        "retry_allowed": False,
        "bulk_release_authorized": False,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    if (
        not _canonical_utc(receipt.get("terminal_at_utc"))
        or receipt.get("receipt_sha256") != digest(receipt, "receipt_sha256")
        or {k: v for k, v in receipt.items() if k != "receipt_sha256"} != expected
    ):
        raise ValueError("generation-3 terminal is not authoritative")


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("validate-evidence", "validate-spec", "validate-held", "preview"),
    )
    parser.add_argument("--spec", action="append", type=Path, default=[])
    parser.add_argument("--held", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.command == "validate-evidence":
        validate_incident(load(args.repo / INCIDENT_PATH), args.repo)
        validate_tombstones(load(args.repo / TOMBSTONE_PATH), args.repo)
        return 0
    if args.command == "validate-spec":
        if len(args.spec) != 1:
            parser.error("validate-spec requires exactly one spec")
        validate_spec(load(args.spec[0]), args.repo)
        return 0
    if not args.held:
        parser.error(f"{args.command} requires --held")
    validate_held(load(args.held), args.repo)
    if args.command == "preview":
        print(
            json.dumps(
                {
                    "ok": True,
                    "status": "HELD",
                    "execution_generation": 3,
                    "statistical_cells": 2,
                    "priority_class": "fleet-serve-low",
                    "preemption_policy": "Never",
                    "launch_authorized": False,
                    "objects_created": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
