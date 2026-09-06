import copy

import pytest

from evals.fleet import hosted_abort_classification_v1 as classification
from evals.fleet import self_hosted


def _abort():
    value = {
        "schema_version": "fleet-exact-pass4-bulk-controller-abort-v3",
        "error_type": "FleetRequestError",
        "automatic_tail_continuation": False,
        "systemic_failure_requires_review": True,
        "prompts_or_traces_included": False,
        "scores_included": False,
        "cell_id": "sha256:" + "1" * 64,
        "execution_id": "sha256:" + "2" * 64,
        "claim_sha256": "sha256:" + "3" * 64,
        "plan_sha256": "sha256:" + "4" * 64,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_request_abort_is_blocked_and_never_a_score_or_automatic_retry():
    result = classification.classify(_abort())
    assert result["status"] == "INFRASTRUCTURE_BLOCKED_UNRESOLVED"
    assert result["scoring_interpretation"] == "NOT_A_SCORE"
    assert result["retry_authorized"] is False


def test_any_retry_or_digest_drift_fails_closed():
    value = copy.deepcopy(_abort())
    value["automatic_tail_continuation"] = True
    with pytest.raises(ValueError):
        classification.classify(value)

