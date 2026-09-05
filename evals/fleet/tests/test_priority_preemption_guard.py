from __future__ import annotations

import pytest

from evals.fleet.priority_preemption_guard import validate_pod_priority_policy


def test_accepts_highest_nonpreempting_evaluator_class() -> None:
    validate_pod_priority_policy(
        {"priorityClassName": "fleet-serve-low", "preemptionPolicy": "Never"}
    )


def test_rejects_never_with_preempting_priority_class() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        validate_pod_priority_policy(
            {"priorityClassName": "fleet-train-high", "preemptionPolicy": "Never"}
        )


def test_live_policy_overrides_stale_static_assumption() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        validate_pod_priority_policy(
            {"priorityClassName": "fleet-serve-low", "preemptionPolicy": "Never"},
            {"fleet-serve-low": "PreemptLowerPriority"},
        )
