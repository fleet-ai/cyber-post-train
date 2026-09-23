"""Offline regression contract for legacy V17, prod8, and five broad arms."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cyber_post_train.jobs import digest
from scripts import audit_qwen38_skyrl_launch_readiness as audit

ROOT = Path(__file__).resolve().parents[1]


def test_audit_is_current_self_sealed_and_performs_no_external_action() -> None:
    value = audit.build()
    assert audit.AUDIT.read_bytes() == audit.raw(value)
    assert value["sha256"] == "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert value["external_activity"] == {
        "Jobs_API_requests": 0,
        "Kubernetes_requests": 0,
        "WandB_requests": 0,
        "cluster_mutations": 0,
        "private_logs_read": False,
    }
    assert value["failure_policy"] == {
        "numeric_failure_budget": None,
        "historical_source_receipt_preserved": True,
        "historical_source_receipt_is_not_a_current_submission_gate": True,
        "failures_require_evidence_repair_and_resource_release": True,
    }
    assert all("failure_budget" not in item for item in value["blockers"])
    assert value["launch_authorized"] is False

    result = subprocess.run(
        [sys.executable, str(audit.__file__), "--check"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["sha256"] == value["sha256"]
    assert output["launch_authorized"] is False


def test_legacy_v17_and_prod8_recompile_to_the_frozen_digests_and_resource_shape() -> None:
    value = audit.build()
    topology = value["legacy_v17_development_topology_evidence"]
    prod8 = value["prod8_one_step_reward_gate"]
    assert topology["gating"] is False
    assert topology["compiled_digests"] == {
        "plan_sha256": ("sha256:c7dba701b3b8805a8efbf5cda67f346a5aec0fddd2c8356756131830beca2c40"),
        "request_sha256": (
            "sha256:6e6826a35a45028b8ef0aad05ef476a9c941f22fcb58dbbde830712891f95d95"
        ),
        "fleetjob_manifest_sha256": (
            "sha256:0a8ba4e08b8a1e3149b9a19f1bba339834f0369894e9d8f42c854099fd0dae05"
        ),
        "preflight_manifest_sha256": (
            "sha256:4f97ad8c26a6a3529813218424249e5d2894f48657ac5306ae6cccce011646b4"
        ),
        "receipt_verifier_manifest_sha256": (
            "sha256:4eb41b7109baae0b6d95b993dcb293a55fdaf6651ca9f517dd74f9ef87b8aa27"
        ),
    }
    assert topology["resource_shape"] == {
        "priority": "c1",
        "physical_nodes": 1,
        "gpus": 8,
        "gpu_head_pods": 1,
        "gpu_worker_replicas": 0,
        "engines": 2,
        "tensor_parallel_size": 4,
    }
    assert topology["science"] == {
        "task_rows": 0,
        "rollout_episodes": 0,
        "verifier_calls": 0,
        "optimizer_steps": 0,
        "checkpoints": 0,
    }
    assert topology["bounds"]["process_total_seconds"] == 1500
    assert topology["bounds"]["fleetjob_active_deadline_seconds"] == 1800
    assert topology["submission_gate"]["submission_authorized"] is False

    assert prod8["compiled_digests"] == {
        "plan_sha256": ("sha256:e53e407a68a5ca56752381d76e48e1459cfa038595f381197c9271e6c7cc9b1e"),
        "request_sha256": (
            "sha256:1ec530f63e3ef1c68e77b469d9c6a6e950d9829175c0b624913d6e1d9f05fa00"
        ),
        "sanitized_manifest_self_sha256": (
            "sha256:5561f1a349abbe1a580dd5763368c1e6c1524861c39d744f7dad4f25d9950dd3"
        ),
        "staged_manifest_observed": False,
    }
    assert prod8["resource_shape"] == {
        "priority": "c1",
        "nodes": 1,
        "gpus_per_node": 8,
        "gpus": 8,
        "requeue_if_preempted": False,
    }
    assert prod8["wandb"]["run_id"] == "chris-q38-rlreward-prod8"
    assert prod8["wandb"]["resume"] == "never"
    assert prod8["wandb"]["fresh_run_ID_absence_checked_before_submit"] is False
    assert prod8["watchdog"]["hard_seconds"] == 14 * 60 * 60
    assert prod8["watchdog"]["episode_budget"]["fits_watchdog_hard_bound"] is True
    assert (
        prod8["watchdog"]["episode_budget"]["minimum_hard_seconds_without_shortening_episode"]
        == 45300
    )
    encoded = prod8["encoded_absence_checks"]
    assert encoded["fresh_Kubernetes_name_absence_encoded"] is True
    assert encoded["fresh_SFS_manifest_payload_and_output_absence_encoded"] is True
    assert encoded["fresh_WandB_run_ID_absence_encoded"] is True
    assert encoded["fresh_guard_executed"] is False
    assert prod8["submission_gate"]["submission_authorized"] is False


def test_full_arms_have_exact_offline_plans_and_plan_bound_watchdogs() -> None:
    value = audit.build()
    queue = value["full_c1_arms"]
    assert queue["all_c1_one_node_eight_GPU"] is True
    assert queue["all_legal_episode_ceilings_fit_plan_watchdogs"] is True
    assert [arm["id"] for arm in queue["arms"]] == [
        "a1",
        "lr3e7",
        "lr3e6",
        "seed43",
        "dose50",
    ]
    assert len({arm["wandb"]["run_id"] for arm in queue["arms"]}) == 5
    assert all(arm["plan_sha256"].startswith("sha256:") for arm in queue["arms"])
    assert all(arm["request_sha256"].startswith("sha256:") for arm in queue["arms"])
    assert all(
        arm["plan_state"] == "exact_offline_plan_compiled_not_live_preflighted"
        for arm in queue["arms"]
    )
    assert all(arm["resource_shape"]["priority"] == "c1" for arm in queue["arms"])
    assert all(arm["resource_shape"]["gpus"] == 8 for arm in queue["arms"])
    assert all(arm["watchdog"]["fits_watchdog_hard_bound"] for arm in queue["arms"])
    assert all(arm["fresh_absence_guard"]["encoded"] for arm in queue["arms"])
    assert all(not arm["fresh_absence_guard"]["executed"] for arm in queue["arms"])
    assert all(not arm["release_observer"]["armed"] for arm in queue["arms"])
    ceilings = {
        arm["id"]: arm["watchdog"][
            "legal_episode_ceiling_seconds_excluding_startup_and_optimization"
        ]
        for arm in queue["arms"]
    }
    assert ceilings == {
        "a1": 38400,
        "lr3e7": 38400,
        "lr3e6": 38400,
        "seed43": 38400,
        "dose50": 163200,
    }
    hard_bounds = {arm["id"]: arm["watchdog"]["watchdog"]["hard_seconds"] for arm in queue["arms"]}
    assert hard_bounds == {
        "a1": 43200,
        "lr3e7": 43200,
        "lr3e6": 43200,
        "seed43": 43200,
        "dose50": 172800,
    }
    assert (
        "full_arm_legal_episode_ceilings_exceed_the_shared_eight_hour_watchdog_bound"
        not in value["blockers"]
    )
