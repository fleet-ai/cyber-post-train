import pytest

from evals.fleet import retry_review_policy as policy


def facts(**changes):
    value = {
        "cell_state": "retry_review",
        "failure_code": "authoritative_scoring_started.runtimeerror",
        "termination_class": "output_limit",
        "retry_count": 0,
        "max_retries": 1,
        "scoring_recovery_count": 0,
        "has_local_result": True,
        "has_session": True,
        "has_trace_manifest": True,
        "has_scoring_intent": True,
        "authoritative_absence_proven": False,
        "authoritative_session_count": 1,
        "authoritative_session_completed": True,
        "authoritative_model_matches": True,
        "authoritative_task_version_matches": True,
        "authoritative_verifier_matches": True,
        "authoritative_score_finite": True,
        "immutable_attempt_evidence_complete": True,
    }
    return {**value, **changes}


@pytest.mark.parametrize("termination_class", ["output_limit", "process_error"])
def test_existing_authoritative_outcome_is_accepted_without_generation(termination_class):
    result = policy.classify(facts(termination_class=termination_class))
    assert result == {
        "action": "accept_existing_scored_session",
        "model_generation_allowed": False,
        "scoring_only": False,
        "reason": "one_exact_authoritative_scored_session",
    }


def test_immutable_session_uses_scoring_only_recovery():
    result = policy.classify(
        facts(
            termination_class="process_error",
            authoritative_verifier_matches=False,
            authoritative_score_finite=False,
        )
    )
    assert result["action"] == "scoring_only_recovery"
    assert result["model_generation_allowed"] is False
    assert result["scoring_only"] is True


@pytest.mark.parametrize("failure_code", ["post_claim.connecterror", "post_claim.operationalerror"])
def test_transport_failure_without_any_outcome_can_retry_once(failure_code):
    result = policy.classify(
        facts(
            failure_code=failure_code,
            termination_class="post_claim_transport",
            has_local_result=False,
            has_session=False,
            has_trace_manifest=False,
            has_scoring_intent=False,
            authoritative_absence_proven=True,
            authoritative_session_count=0,
            authoritative_session_completed=False,
            authoritative_model_matches=False,
            authoritative_task_version_matches=False,
            authoritative_verifier_matches=False,
            authoritative_score_finite=False,
        )
    )
    assert result["action"] == "rollout_retry"
    assert result["model_generation_allowed"] is True


def test_transport_failure_never_replays_when_any_session_exists():
    result = policy.classify(
        facts(
            failure_code="post_claim.connecterror",
            termination_class="post_claim_transport",
        )
    )
    assert result["action"] == "accept_existing_scored_session"
    assert result["model_generation_allowed"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"retry_count": 1},
        {"authoritative_absence_proven": False},
        {"immutable_attempt_evidence_complete": False},
        {"failure_code": "post_claim.valueerror"},
    ],
)
def test_rollout_retry_fails_closed(changes):
    base = facts(
        has_local_result=False,
        has_session=False,
        has_trace_manifest=False,
        has_scoring_intent=False,
        authoritative_absence_proven=True,
        authoritative_session_count=0,
        authoritative_session_completed=False,
        authoritative_model_matches=False,
        authoritative_task_version_matches=False,
        authoritative_verifier_matches=False,
        authoritative_score_finite=False,
        failure_code="post_claim.connecterror",
        termination_class="post_claim_transport",
    )
    result = policy.classify({**base, **changes})
    assert result["action"] == "manual_terminal_review"
    assert result["model_generation_allowed"] is False


def test_policy_rejects_score_values_and_unknown_fields():
    with pytest.raises(policy.RetryPolicyError, match="unknown"):
        policy.classify({**facts(), "score": 0.0})


def test_authoritative_absence_must_not_contradict_session_count():
    with pytest.raises(policy.RetryPolicyError, match="absence conflicts"):
        policy.classify(facts(authoritative_absence_proven=True))


def test_observer_policy_is_outside_sealed_evaluator_runtime():
    from evals.fleet import evaluate

    assert "retry_review_policy.py" not in evaluate.RUNTIME_FILES


@pytest.mark.parametrize("termination_class", ["output_limit", "process_error"])
def test_output_or_process_failure_without_a_session_never_regenerates(
    termination_class,
):
    result = policy.classify(
        facts(
            termination_class=termination_class,
            has_local_result=False,
            has_session=False,
            has_trace_manifest=False,
            has_scoring_intent=False,
            authoritative_session_count=0,
            authoritative_session_completed=False,
            authoritative_model_matches=False,
            authoritative_task_version_matches=False,
            authoritative_verifier_matches=False,
            authoritative_score_finite=False,
        )
    )
    assert result["action"] == "manual_terminal_review"
    assert result["model_generation_allowed"] is False


def test_only_retry_review_state_is_accepted():
    with pytest.raises(policy.RetryPolicyError, match="only retry_review"):
        policy.classify(facts(cell_state="accepted"))
