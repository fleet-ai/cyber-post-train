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
        "plan_sha256": ("sha256:b5c8b625f9c42f4d77c5f2c5bfc6825ba7c7eadee685fa76b0acf2e258f311ee"),
        "request_sha256": (
            "sha256:202174da94bcd30dfdd235a9c50218d3aa4f46b8dc9b6f44e0a582acfc869085"
        ),
        "fleetjob_manifest_sha256": (
            "sha256:3d5f1b508e6999a8853d67ae1fdfb4c396ed05be92922c4c9b419ce325ab9c6f"
        ),
        "preflight_manifest_sha256": (
            "sha256:fa008b04313ad576d46adfd0815b0836c8405ea62a47a6b569c13ad0277811fe"
        ),
        "receipt_verifier_manifest_sha256": (
            "sha256:d13bcda162a66989a40d207828cd79dd3ce174b2a47b3b2480c4795349ff3529"
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
        "plan_sha256": ("sha256:d43600f5bb9d8e4c4948f2b1c185ce891c2e416384543ef4d14305e129a4f336"),
        "request_sha256": (
            "sha256:dece471161351ebe22b3cd7794115c3b0d19b5b09a8344aead6ea6ed0bfc096f"
        ),
        "sanitized_manifest_self_sha256": (
            "sha256:7e5c2e293a69b4d3bad6402a9945520182853a1caae1ae8a8c74f8f8bbe99471"
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
    assert prod4["wandb"]["run_id"] == "chris-q38-rlreward-prod7"
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
