"""Classify sealed hosted-controller aborts without inferring a score or retry."""

from __future__ import annotations

from typing import Any, Mapping

from evals.fleet import self_hosted


def classify(abort: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version": "fleet-exact-pass4-bulk-controller-abort-v3",
        "error_type": "FleetRequestError",
        "automatic_tail_continuation": False,
        "systemic_failure_requires_review": True,
        "prompts_or_traces_included": False,
        "scores_included": False,
    }
    if any(abort.get(key) != value for key, value in required.items()):
        raise ValueError("hosted abort is not the reviewed request-failure shape")
    if abort.get("receipt_sha256") != self_hosted.digest_without(
        dict(abort), "receipt_sha256"
    ):
        raise ValueError("hosted abort self digest drifted")
    for field in ("cell_id", "execution_id", "claim_sha256", "plan_sha256"):
        value = abort.get(field)
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
            raise ValueError(f"hosted abort {field} drifted")
    return {
        "status": "INFRASTRUCTURE_BLOCKED_UNRESOLVED",
        "scoring_interpretation": "NOT_A_SCORE",
        "retry_authorized": False,
        "cell_id": abort["cell_id"],
        "execution_id": abort["execution_id"],
        "receipt_sha256": abort["receipt_sha256"],
    }

