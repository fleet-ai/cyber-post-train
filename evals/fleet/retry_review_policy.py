"""Arm-blind policy for reviewed Fleet recovery; this module never mutates state."""

from __future__ import annotations

from typing import Any

ROLLOUT_RETRY_FAILURE_CODES = {
    "post_claim.connecterror",
    "post_claim.operationalerror",
}
MAX_ROLLOUT_RETRIES = 1
MAX_SCORING_RECOVERIES = 1

FACT_FIELDS = {
    "cell_state",
    "failure_code",
    "termination_class",
    "retry_count",
    "max_retries",
    "scoring_recovery_count",
    "has_local_result",
    "has_session",
    "has_trace_manifest",
    "has_scoring_intent",
    "authoritative_absence_proven",
    "authoritative_session_count",
    "authoritative_session_completed",
    "authoritative_model_matches",
    "authoritative_task_version_matches",
    "authoritative_verifier_matches",
    "authoritative_score_finite",
    "immutable_attempt_evidence_complete",
}


class RetryPolicyError(ValueError):
    """The score-blind fact envelope is incomplete or contradictory."""


def _validate(facts: dict[str, Any]) -> None:
    if set(facts) != FACT_FIELDS:
        raise RetryPolicyError("retry-review facts have missing or unknown fields")
    if facts["cell_state"] != "retry_review":
        raise RetryPolicyError("only retry_review cells may be classified")
    if not isinstance(facts["failure_code"], str) or not facts["failure_code"]:
        raise RetryPolicyError("failure code is missing")
    if facts["termination_class"] not in {
        "output_limit",
        "process_error",
        "post_claim_transport",
        "other",
    }:
        raise RetryPolicyError("termination class is not allowlisted")
    for field in ("retry_count", "max_retries", "scoring_recovery_count"):
        value = facts[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RetryPolicyError(f"{field} must be a non-negative integer")
    sessions = facts["authoritative_session_count"]
    if isinstance(sessions, bool) or not isinstance(sessions, int) or sessions < 0:
        raise RetryPolicyError("authoritative session count must be a non-negative integer")
    for field in FACT_FIELDS - {
        "cell_state",
        "failure_code",
        "termination_class",
        "retry_count",
        "max_retries",
        "scoring_recovery_count",
        "authoritative_session_count",
    }:
        if type(facts[field]) is not bool:
            raise RetryPolicyError(f"{field} must be boolean")
    if facts["authoritative_absence_proven"] and sessions != 0:
        raise RetryPolicyError("authoritative absence conflicts with observed sessions")
    if sessions == 0 and any(
        facts[field]
        for field in (
            "authoritative_session_completed",
            "authoritative_model_matches",
            "authoritative_task_version_matches",
            "authoritative_verifier_matches",
            "authoritative_score_finite",
        )
    ):
        raise RetryPolicyError("authoritative session facts exist without a session")


def classify(facts: dict[str, Any]) -> dict[str, Any]:
    """Choose one bounded recovery lane without reading or branching on a score."""
    _validate(facts)
    generation_evidence = (
        any(
            facts[field]
            for field in (
                "has_local_result",
                "has_session",
                "has_trace_manifest",
                "has_scoring_intent",
            )
        )
        or facts["authoritative_session_count"] > 0
    )
    exact_scored_session = all(
        (
            facts["authoritative_session_count"] == 1,
            facts["authoritative_session_completed"],
            facts["authoritative_model_matches"],
            facts["authoritative_task_version_matches"],
            facts["authoritative_verifier_matches"],
            facts["authoritative_score_finite"],
            facts["immutable_attempt_evidence_complete"],
        )
    )
    if generation_evidence:
        if exact_scored_session:
            return {
                "action": "accept_existing_scored_session",
                "model_generation_allowed": False,
                "scoring_only": False,
                "reason": "one_exact_authoritative_scored_session",
            }
        score_only_eligible = all(
            (
                facts["authoritative_session_count"] == 1,
                facts["authoritative_session_completed"],
                facts["authoritative_model_matches"],
                facts["authoritative_task_version_matches"],
                facts["immutable_attempt_evidence_complete"],
                facts["scoring_recovery_count"] < MAX_SCORING_RECOVERIES,
            )
        )
        if score_only_eligible:
            return {
                "action": "scoring_only_recovery",
                "model_generation_allowed": False,
                "scoring_only": True,
                "reason": "immutable_session_requires_authoritative_scoring",
            }
        return {
            "action": "manual_terminal_review",
            "model_generation_allowed": False,
            "scoring_only": False,
            "reason": "generation_evidence_cannot_be_replayed",
        }

    retry_eligible = all(
        (
            facts["termination_class"] == "post_claim_transport",
            facts["failure_code"] in ROLLOUT_RETRY_FAILURE_CODES,
            facts["authoritative_absence_proven"],
            facts["immutable_attempt_evidence_complete"],
            facts["retry_count"] < min(facts["max_retries"], MAX_ROLLOUT_RETRIES),
        )
    )
    if retry_eligible:
        return {
            "action": "rollout_retry",
            "model_generation_allowed": True,
            "scoring_only": False,
            "reason": "transport_failure_before_any_outcome_evidence",
        }
    return {
        "action": "manual_terminal_review",
        "model_generation_allowed": False,
        "scoring_only": False,
        "reason": "no_arm_blind_recovery_rule_matches",
    }
