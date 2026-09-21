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


def test_v17_and_prod4_recompile_to_the_frozen_digests_and_resource_shape() -> None:
    value = audit.build()
    topology = value["v17_development_topology_gate"]
    prod4 = value["prod4_one_step_reward_gate"]
    assert topology["compiled_digests"] == {
        "plan_sha256": ("sha256:b4122e3b07b0df82ed5a037c47cf268fb74ef20ec1e78f0f5d8c61d3f1712a53"),
        "request_sha256": (
            "sha256:72c08c60b5e8f299e49977a1caa037c6b6f0524eba56806c2718c464fe6cb3a3"
        ),
        "fleetjob_manifest_sha256": (
            "sha256:de9ed0b8eed9a13d8c1fc180d7e3d09ff67c39e587f7bd7665f62f71dbe85803"
        ),
        "preflight_manifest_sha256": (
            "sha256:5eb59f63df0a436f88859e7586442507789304a08426374c255be4a353144a76"
        ),
        "receipt_verifier_manifest_sha256": (
            "sha256:e1d17fa2a3f941eb529b69a3f1eb0ca2a1f740e275ea48bb8d2f5308cad0b736"
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
        "plan_sha256": ("sha256:09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de"),
        "request_sha256": (
            "sha256:7c2df31feceb5c741cb16463b554b91d00b11c6bf50ec85b3203706823c2fb44"
        ),
        "sanitized_manifest_self_sha256": (
            "sha256:5561f1a349abbe1a580dd5763368c1e6c1524861c39d744f7dad4f25d9950dd3"
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
    assert prod4["wandb"]["run_id"] == "chris-q38-rlreward-prod8"
    assert prod4["wandb"]["resume"] == "never"
    assert prod4["wandb"]["fresh_run_ID_absence_checked_before_submit"] is False
    assert prod4["watchdog"]["hard_seconds"] == 14 * 60 * 60
    assert prod4["watchdog"]["episode_budget"]["fits_watchdog_hard_bound"] is True
    assert (
        prod4["watchdog"]["episode_budget"]["minimum_hard_seconds_without_shortening_episode"]
        == 45300
    )
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
