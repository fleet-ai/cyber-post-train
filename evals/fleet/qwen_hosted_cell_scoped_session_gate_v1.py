"""Held score-blind rules for cell-scoped hosted-session reconciliation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from evals.fleet import self_hosted

SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
SESSION_IDENTITY_FIELDS = {
    "session_id",
    "eval_task_id",
    "eval_task_version_id",
    "task_key",
    "model_identity",
    "model_identity_status",
    "cell_id",
    "execution_id",
    "run_id",
    "status",
}


class IdentityAmbiguous(ValueError):
    """The metadata-only identity is insufficient to classify safely."""


def _identity_tuple(value: Mapping[str, Any]) -> tuple[str, str, str] | None:
    cell_id = value.get("cell_id")
    execution_id = value.get("execution_id")
    run_id = value.get("run_id")
    if cell_id is execution_id is run_id is None:
        return None
    if (
        not isinstance(cell_id, str)
        or SHA256_RE.fullmatch(cell_id) is None
        or not isinstance(execution_id, str)
        or SHA256_RE.fullmatch(execution_id) is None
        or not isinstance(run_id, str)
        or not run_id
    ):
        raise IdentityAmbiguous("session statistical identity is malformed")
    return cell_id, execution_id, run_id


def classify(
    value: Mapping[str, Any],
    *,
    task_key: str,
    task_version_id: str,
    session_model: str,
    planned_cell_ids: set[str],
    known_non_target_executions: set[tuple[str, str, str]],
) -> str:
    """Classify one strict metadata-only row without looking at task content."""
    row = dict(value)
    if set(row) != SESSION_IDENTITY_FIELDS:
        raise IdentityAmbiguous("session identity projection shape is invalid")
    if (
        not isinstance(row["session_id"], str)
        or not row["session_id"]
        or not isinstance(row["eval_task_id"], str)
        or not row["eval_task_id"]
        or row["task_key"] != task_key
        or not isinstance(row["status"], str)
        or not row["status"]
    ):
        raise IdentityAmbiguous("session identity projection binding is invalid")
    version_id = row["eval_task_version_id"]
    if not isinstance(version_id, str) or not version_id:
        raise IdentityAmbiguous("session task version is missing")

    identity = _identity_tuple(row)
    if identity is not None and identity[0] in planned_cell_ids:
        return "TARGET_CELL_COLLISION"
    if version_id != task_version_id:
        return "NON_TARGET_VERSION"

    model_status = row["model_identity_status"]
    model = row["model_identity"]
    if model_status == "resolved":
        if not isinstance(model, str) or not model:
            raise IdentityAmbiguous("resolved model identity is missing")
        return "TARGET_TREATMENT_COLLISION" if model == session_model else "NON_TARGET_MODEL"
    if model_status != "ambiguous" or model is not None:
        raise IdentityAmbiguous("model identity status is invalid")
    if identity is not None and identity in known_non_target_executions:
        return "KNOWN_NON_TARGET_EXECUTION"
    raise IdentityAmbiguous("exact target-version session identity is ambiguous")


def contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "fleet-qwen38-cell-scoped-session-gate-v1",
        "status": "HELD_UNTIL_METADATA_IDENTITY_API_DEPLOYED",
        "launch_authorized": False,
        "scoring_authorized": False,
        "scientific_cell": ["model", "task_version_id", "attempt"],
        "evidence_layers": [
            "authoritative_ledger_tally",
            "global_canonical_claim_cell_scan",
            "global_accepted_receipt_cell_scan",
            "fresh_job_configmap_pod_and_sfs_absence",
            "metadata_only_session_identity_projection",
        ],
        "session_rule_order": [
            "planned_cell_id_collision",
            "different_exact_task_version_is_non_target",
            "exact_version_and_exact_model_is_collision",
            "exact_version_and_different_resolved_model_is_non_target",
            "exact_known_non_target_execution_tuple_is_non_target",
            "otherwise_fail_ambiguous",
        ],
        "null_model_policy": {
            "blanket_task_level_block_forbidden": True,
            "different_exact_task_version_may_be_ignored": True,
            "known_non_target_execution_tuple_may_be_ignored": True,
            "exact_target_version_without_known_cell_identity_still_blocks": True,
        },
        "required_projection_fields": sorted(SESSION_IDENTITY_FIELDS),
        "forbidden_projection_fields": [
            "prompt",
            "transcript",
            "reference_traces",
            "tool_use_workflow",
            "verifier",
            "verifier_execution",
            "score",
            "flag",
            "metadata",
            "workflow_input_json",
        ],
        "authoritative_tally": {
            "accepted": 51,
            "active": 0,
            "blocked_nonrepeatable": 9,
            "unstarted": 340,
        },
        "deployment_gate": {
            "metadata_identity_api_deployed": False,
            "behavioral_probe_passed": False,
            "fresh_observer_required": True,
            "rank_walk_forbidden": True,
        },
        "privacy": {
            "methods": ["GET"],
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "model_calls": 0,
            "session_mutations": 0,
            "scoring_calls": 0,
        },
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate_contract(value: Mapping[str, Any]) -> None:
    if dict(value) != contract():
        raise ValueError("cell-scoped session gate contract drifted")
