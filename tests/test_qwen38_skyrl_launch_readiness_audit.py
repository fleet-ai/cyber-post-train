"""Offline regression contract for the V17, prod4, and five-arm launch audit."""

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
    assert value["failure_budget"] == {
        "used": 10,
        "limit": 10,
        "reset_recorded": False,
        "external_cluster_post_stop": True,
    }
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


def test_v17_and_prod4_recompile_to_the_frozen_digests_and_resource_shape() -> None:
    value = audit.build()
    topology = value["v17_development_topology_gate"]
    prod4 = value["prod4_one_step_reward_gate"]
    assert topology["compiled_digests"] == {
        "plan_sha256": ("sha256:51accb1d9857254fb4bee013001c361785698c5c5b7b825a7b328ee785b717cd"),
        "request_sha256": (
            "sha256:287f9a1a6cb58ae5b03a002be8f14109c53abcf9e097cb410e9d9fa1a3fa146b"
        ),
        "fleetjob_manifest_sha256": (
            "sha256:047485386a653f3c8e6cb7b16b853b0bd12accf027b862c9fab58f0fbf5e0b99"
        ),
        "preflight_manifest_sha256": (
            "sha256:b36a70da18448291771bddbeee9f17c007249bddfe402c5b0aef0bc283d60ddb"
        ),
        "receipt_verifier_manifest_sha256": (
            "sha256:59c3c1f1fd22c523fd8860e4d61155ee884a1c1002dff7f23d57ffc893b33ee7"
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

    assert prod4["compiled_digests"] == {
        "plan_sha256": ("sha256:513e39ff78605f74ab08af22187cb7e2bb3ab2f9290da03a1402d7d282f03a65"),
        "request_sha256": (
            "sha256:07fc87fda635341352a1726cb061de874be5b4a1eabc82985e8b28ef85104473"
        ),
        "sanitized_manifest_self_sha256": (
            "sha256:b03825787255a72b5dd5bc79fde43c3d0a59e3b75561ca8b7d8d3bef683cb326"
        ),
        "staged_manifest_observed": False,
    }
    assert prod4["resource_shape"] == {
        "priority": "c1",
        "nodes": 1,
        "gpus_per_node": 8,
        "gpus": 8,
        "requeue_if_preempted": False,
    }
    assert prod4["wandb"]["run_id"] == "chris-q38-rlreward-prod4"
    assert prod4["wandb"]["resume"] == "never"
    assert prod4["wandb"]["fresh_run_ID_absence_checked_before_submit"] is False
    encoded = prod4["encoded_absence_checks"]
    assert encoded["fresh_Kubernetes_name_absence_encoded"] is True
    assert encoded["fresh_SFS_manifest_payload_and_output_absence_encoded"] is True
    assert encoded["fresh_WandB_run_ID_absence_encoded"] is True
    assert encoded["fresh_guard_executed"] is False
    assert prod4["submission_gate"]["submission_authorized"] is False


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
    hard_bounds = {
        arm["id"]: arm["watchdog"]["watchdog"]["hard_seconds"] for arm in queue["arms"]
    }
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
