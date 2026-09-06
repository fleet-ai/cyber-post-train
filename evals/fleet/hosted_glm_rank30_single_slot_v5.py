"""Rank-30 scored authority gated by the successful v6 runtime diagnostic."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank30_single_slot_v4 as prior
from evals.fleet import self_hosted

CONTROLLER = prior.CONTROLLER
CONTROLLERS = prior.CONTROLLERS
RELEASE_SCHEMA = "fleet-hosted-glm-rank30-single-slot-release-v4"
RELEASE_PATH = prior.RELEASE_PATH
RELEASE_MAX_AGE_SECONDS = prior.prior.RELEASE_MAX_AGE_SECONDS
PEER_STREAM_MAX_AGE_SECONDS = prior.prior.PEER_STREAM_MAX_AGE_SECONDS
SHA256_RE = prior.SHA256_RE
COMMIT_RE = prior.COMMIT_RE
CLAIM_ROOT = prior.CLAIM_ROOT
JOBS_ROOT = prior.JOBS_ROOT
LEASE_ROOT = prior.LEASE_ROOT
LEASE_ENDPOINT_KEY = prior.LEASE_ENDPOINT_KEY
build_plan = prior.build_plan
build_runtime_plan = prior.build_runtime_plan
validate_all = prior.validate_all
release_projection = prior.release_projection
validate_current_peer = prior.prior.validate_current_peer
load = prior.prior.load
AtomicWholeTaskClaims = prior.prior.AtomicWholeTaskClaims

DIAGNOSTIC_PATH = Path(
    "/mnt/sfs/jobs/chris-glm53-r030-release-diagnostic-v6/DIAGNOSTIC.json"
)
DIAGNOSTIC_SELF_SHA256 = (
    "sha256:28b56482621bc165d9379b419de17e93eefb521bea140a46bd1d2c9bb005d875"
)
DIAGNOSTIC_FILE_SHA256 = (
    "sha256:b3c55eb136648b3956320754a3f4a1f9cf23b6475f2865d1dcad5a56cca7fbd1"
)
DIAGNOSTIC_JOB_UID = "d5b0fd0d-f99c-426f-875f-946d84dd43e4"
DIAGNOSTIC_POD_UID = "04d64479-d5e3-4ce2-b6bd-69201487411b"
DIAGNOSTIC_PLAN_SHA256 = (
    "sha256:e0bef037fe7853d3e4d8a0762841ceb6467bdde846a386d6dc5b15d5aecb42c6"
)
DIAGNOSTIC_PHASES = [
    "01-materialized-sources",
    "02-failed-v3-terminal",
    "03-observer-bindings",
    "04-inventory-load",
    "05-runtime-plan",
    "06-current-a4-peer",
    "07-endpoint-slot",
    "08-local-collisions",
    "09-release-projection",
]


def validate_runtime_gate_diagnostic(path: Path = DIAGNOSTIC_PATH) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 262_144:
        raise RuntimeError("rank-30 v6 diagnostic is absent or unsafe")
    payload = path.read_bytes()
    value = json.loads(payload)
    safe_counts = value.get("safe_counts") if isinstance(value, dict) else {}
    if any(
        (
            not isinstance(value, dict),
            "sha256:" + hashlib.sha256(payload).hexdigest()
            != DIAGNOSTIC_FILE_SHA256,
            value.get("receipt_sha256") != DIAGNOSTIC_SELF_SHA256,
            value.get("receipt_sha256")
            != self_hosted.digest_without(value, "receipt_sha256"),
            value.get("schema_version")
            != "fleet-hosted-glm-rank30-release-phase-diagnostic-v6",
            value.get("status") != "PASSED_TO_SESSION_BOUNDARY",
            value.get("observer_job_uid") != DIAGNOSTIC_JOB_UID,
            value.get("observer_pod_uid") != DIAGNOSTIC_POD_UID,
            value.get("plan_sha256") != DIAGNOSTIC_PLAN_SHA256,
            value.get("completed_phases") != DIAGNOSTIC_PHASES,
            value.get("phase_order") != DIAGNOSTIC_PHASES,
            value.get("last_completed_phase") != DIAGNOSTIC_PHASES[-1],
            value.get("failed_phase") is not None,
            value.get("error_sha256") is not None,
            value.get("error_type_class") is not None,
            value.get("projected_attempts") != [1, 2, 3, 4],
            value.get("session_boundary_executed") is not False,
            any(
                value.get(field) != 0
                for field in (
                    "model_calls",
                    "task_calls",
                    "session_calls",
                    "verifier_calls",
                    "scoring_calls",
                    "api_mutation_calls",
                )
            ),
            value.get("scores_read") is not False,
            value.get("prompts_traces_flags_read") is not False,
            not isinstance(safe_counts, dict),
            safe_counts.get("claim_collisions") != 0,
            safe_counts.get("accepted_collisions") != 0,
            safe_counts.get("output_collisions") != 0,
            safe_counts.get("target_object_collisions") != 0,
        )
    ):
        raise RuntimeError("rank-30 v6 diagnostic drifted")
    return {
        "path": str(path),
        "receipt_sha256": DIAGNOSTIC_SELF_SHA256,
        "file_sha256": DIAGNOSTIC_FILE_SHA256,
        "job_uid": DIAGNOSTIC_JOB_UID,
        "pod_uid": DIAGNOSTIC_POD_UID,
        "status": "PASSED_TO_SESSION_BOUNDARY",
        "last_completed_phase": DIAGNOSTIC_PHASES[-1],
        "plan_sha256": DIAGNOSTIC_PLAN_SHA256,
        "projected_attempts": [1, 2, 3, 4],
        "model_task_session_verifier_scoring_calls": 0,
        "api_mutation_calls": 0,
    }


def validate_release(
    receipt: dict[str, Any], plan: dict[str, Any], source_package_sha256: str
) -> None:
    diagnostic = receipt.get("runtime_gate_diagnostic")
    expected = validate_runtime_gate_diagnostic()
    if diagnostic != expected or set(receipt) != {
        "schema_version",
        "status",
        "checked_at_utc",
        "launch_authorized",
        "scoring_authorized",
        "controller_cap",
        "controller",
        "source_package_sha256",
        "ledger_authority",
        "selection_authority",
        "live_rank29_peer",
        "endpoint_lease_observer",
        "fresh_collision_reconciliation",
        "runtime_gate_diagnostic",
        "privacy",
        "receipt_sha256",
    }:
        raise RuntimeError("rank-30 v6-gated release drifted")
    if (
        receipt.get("schema_version") != RELEASE_SCHEMA
        or receipt.get("receipt_sha256")
        != self_hosted.digest_without(receipt, "receipt_sha256")
    ):
        raise RuntimeError("rank-30 v6-gated release digest drifted")
    predecessor = copy.deepcopy(receipt)
    predecessor.pop("runtime_gate_diagnostic")
    predecessor["schema_version"] = prior.RELEASE_SCHEMA
    predecessor["receipt_sha256"] = self_hosted.digest_without(
        predecessor, "receipt_sha256"
    )
    prior.validate_release(predecessor, plan, source_package_sha256)
