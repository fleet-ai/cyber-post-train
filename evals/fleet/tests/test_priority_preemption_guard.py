from __future__ import annotations

import pytest

from evals.fleet.priority_preemption_guard import (
    require_no_kueue_preemption,
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
        "selection_scope": "live_cluster",
    }


def test_selects_highest_nonpreempting_class_supported_by_jobs_api() -> None:
    rows = [
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
    assert select_highest_nonpreempting(
        rows, allowed_names={"fleet-train-high", "fleet-infra-quiet"}
    ) == {
        "name": "fleet-infra-quiet",
        "uid": "quiet-uid",
        "value": -1000,
        "preemption_policy": "Never",
        "selection_scope": "live_cluster_intersect_jobs_api",
    }


def test_highest_nonpreempting_class_must_be_unique() -> None:
    rows = [
        {"metadata": {"name": name}, "value": 100, "preemptionPolicy": "Never"}
        for name in ("one", "two")
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        select_highest_nonpreempting(rows)


def test_kueue_continuity_accepts_same_uid_without_preemption() -> None:
    require_no_kueue_preemption(
        {
            "metadata": {"uid": "workload-uid"},
            "status": {
                "conditions": [
                    {
                        "type": "Admitted",
                        "status": "True",
                        "reason": "Admitted",
                        "message": "The workload is admitted",
                    }
                ]
            },
        },
        expected_uid="workload-uid",
    )


def test_kueue_continuity_rejects_readmitted_preempted_workload() -> None:
    workload = {
        "metadata": {"uid": "workload-uid"},
        "status": {
            "conditions": [
                {
                    "type": "Admitted",
                    "status": "True",
                    "reason": "Admitted",
                    "message": "The workload is admitted",
                },
                {
                    "type": "Evicted",
                    "status": "False",
                    "reason": "QuotaReserved",
                    "message": "Previously: Preempted to accommodate a workload",
                },
            ]
        },
    }
    with pytest.raises(RuntimeError, match="preempted or evicted"):
        require_no_kueue_preemption(workload, expected_uid="workload-uid")


def test_kueue_continuity_rejects_uid_rotation() -> None:
    with pytest.raises(ValueError, match="UID changed"):
        require_no_kueue_preemption(
            {"metadata": {"uid": "new"}, "status": {}}, expected_uid="old"
        )
