from __future__ import annotations

import pytest

from evals.fleet.priority_preemption_guard import (
    select_highest_nonpreempting,
    validate_pod_priority_policy,
)


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


def test_selects_highest_live_nonpreempting_class() -> None:
    assert select_highest_nonpreempting(
        [
            {
                "metadata": {"name": "fleet-infra-quiet", "uid": "quiet-uid"},
                "value": -1000,
                "preemptionPolicy": "Never",
            },
            {
                "metadata": {"name": "fleet-serve-low", "uid": "serve-uid"},
                "value": 100,
                "preemptionPolicy": "Never",
            },
            {
                "metadata": {"name": "fleet-train-high", "uid": "train-uid"},
                "value": 10000,
                "preemptionPolicy": "PreemptLowerPriority",
            },
        ]
    ) == {
        "name": "fleet-serve-low",
        "uid": "serve-uid",
        "value": 100,
        "preemption_policy": "Never",
    }


def test_highest_nonpreempting_class_must_be_unique() -> None:
    rows = [
        {"metadata": {"name": name}, "value": 100, "preemptionPolicy": "Never"}
        for name in ("one", "two")
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        select_highest_nonpreempting(rows)
