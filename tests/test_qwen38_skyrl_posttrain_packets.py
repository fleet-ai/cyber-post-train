"""Offline downstream gates for every broad Qwen3.8 SkyRL arm."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts import prepare_qwen38_skyrl_posttrain_packets as packets
from scripts import prepare_qwen38_skyrl_production_queue as queue

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_packets_are_current_self_sealed_and_offline() -> None:
    expected = packets.build()
    assert len(expected) == 6
    for path, value in expected.items():
        assert path.read_bytes() == packets.raw(value)
        assert value["sha256"] == packets.object_digest(
            {key: item for key, item in value.items() if key != "sha256"}
        )
    result = subprocess.run(
        [sys.executable, str(packets.__file__), "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"artifacts": 6, "external_mutations": 0}
    evidence = expected[packets.EVIDENCE]
    assert evidence["external_mutations"] == 0
    assert evidence["scope"] == "offline_no_submit_no_stage_no_serve_no_eval"


def test_all_five_arms_bind_unique_terminal_outputs_and_native_cadence() -> None:
    expected = packets.build()
    values = [expected[packets.packet_path(arm)] for arm in queue.ARMS]
    assert len(values) == 5
    assert len({value["arm"]["output_root"] for value in values}) == 5
    assert len({value["serving"]["model_id"] for value in values}) == 5
    assert len({value["export"]["output_root"] for value in values}) == 5
    for arm, value in zip(queue.ARMS, values, strict=True):
        terminal = arm["steps"]
        expected_saves = list(range(10, terminal + 1, 10))
        assert value["checkpoint"]["expected_save_steps"] == expected_saves
        assert value["checkpoint"]["expected_retained_steps_at_terminal"] == expected_saves[-2:]
        assert value["checkpoint"]["terminal_step"] == terminal
        assert value["checkpoint"]["terminal_checkpoint"].endswith(
            f"/checkpoints/global_step_{terminal}"
        )
        assert value["training_acceptance"]["minimum_real_train_rollouts"] == terminal * 8
        assert value["training_acceptance"]["expected_optimizer_updates"] == terminal
        if arm["id"] == "dose50":
            assert value["arm"]["additional_eligibility_gates"]
        else:
            assert value["arm"]["additional_eligibility_gates"] == []


def test_reward_update_checkpoint_export_and_reload_gates_fail_closed() -> None:
    for arm in queue.ARMS:
        value = read(packets.packet_path(arm))
        acceptance = value["training_acceptance"]
        assert acceptance["accepted"] is False
        assert (
            "reward_max_is_strictly_greater_than_reward_min_over_counted_train_rollouts"
            in (acceptance["requirements"])
        )
        assert (
            "at_least_one_optimizer_update_has_a_finite_nonzero_parameter_delta"
            in (acceptance["requirements"])
        )
        assert (
            "terminal_checkpoint_and_sampler_state_are_complete_and_reloadable"
            in (acceptance["requirements"])
        )
        assert value["export"]["launchable"] is False
        assert value["export"]["tooling_ready"] is True
        assert value["export"]["adapter"]["path"] == "training/skyrl_posttrain.py"
        assert (
            value["export"]["checkpoint_manifest_schema"]
            == "cyber_native_skyrl_rl_checkpoint_manifest_v1"
        )
        assert value["export"]["gpus"] == value["export"]["optimizer_updates"] == 0
        assert value["export"]["create_once"] is True
        assert (
            "source_checkpoint_sizes_mtimes_and_digests_unchanged"
            in value["export"]["requirements"]
        )
        reload = value["reload"]
        assert reload["launchable"] is False
        assert reload["cpu_check"]["gpus"] == 0
        assert reload["one_gpu_check"]["gpus_per_worker"] == 1
        assert "zero_optimizer_updates" in reload["one_gpu_check"]["requires"]
        assert "UID_bound_terminal_success_and_GPU_release" in reload["one_gpu_check"]["requires"]


def test_serving_is_create_once_paused_new_uid_and_live_parity_gated() -> None:
    template = None
    for arm in queue.ARMS:
        value = read(packets.packet_path(arm))
        serving = value["serving"]
        assert serving["launchable"] is False and serving["create_once"] is True
        registration = serving["registration"]
        assert registration["must_be_absent_before_create"] is True
        assert registration["must_receive_a_new_server_UID"] is True
        assert registration["must_not_resume_or_replace_an_existing_registration"] is True
        assert registration["initial_desired_state"] == "paused"
        assert registration["routing_must_not_serve_traffic_before_live_parity"] is True
        assert serving["temporary_dev_qualification"]["maximum_runtime_seconds"] == 1800
        assert serving["live_parity"]["must_pass_before_candidate_route_activation"] is True
        assert serving["live_parity"]["only_permitted_difference"] == "exact_weight_manifest"
        template = template or serving["fixed_template"]
        assert serving["fixed_template"] == template


def test_matched_opencode_evaluations_bind_only_dev_and_external_report() -> None:
    split = read(queue.SPLIT)
    dev_ids = {
        (row["task_key"], row["task_version_id"]) for row in split["tasks"] if row["split"] == "dev"
    }
    final_ids = {
        (row["task_key"], row["task_version_id"])
        for row in split["tasks"]
        if row["split"] == "test"
    }
    assert len(dev_ids) == 20 and len(final_ids) == 10
    for arm in queue.ARMS:
        value = read(packets.packet_path(arm))
        evaluation = value["evaluation"]
        assert evaluation["launchable"] is False
        fleet = evaluation["fleet_development"]
        planned = {(row["task_key"], row["task_version_id"]) for row in fleet["tasks"]}
        assert planned == dev_ids and not planned & final_ids
        assert fleet["task_count"] == fleet["attempts_per_arm"] == 20
        assert fleet["harness"]["name"] == "opencode"
        assert fleet["harness"]["version"] == "1.18.27"
        assert fleet["harness"]["native_compaction"] is True
        assert fleet["final_test_tasks"] == "closed_and_forbidden"
        web = evaluation["webexploitbench"]
        assert web["mode"] == "full" and web["task_count"] == 15 and web["pass_k"] == 4
        assert web["attempts_per_arm"] == 60 and web["joint_attempts"] == 120
        assert web["collection_and_scoring_are_separate"] is True
        assert web["defer_scoring"] is True
        assert web["may_not_influence_training_recipe_or_checkpoint_selection"] is True
        assert value["firewall"]["final_test_tasks"] == "closed"
        assert value["firewall"]["final_test_identity_or_outcomes_in_this_packet"] is False
        assert value["firewall"]["external_benchmark_content_or_outcomes_read"] == 0


def test_evidence_rehashes_every_packet() -> None:
    evidence = read(packets.EVIDENCE)
    assert len(evidence["arms"]) == 5
    for row in evidence["arms"]:
        path = ROOT / row["packet"]
        value = read(path)
        assert (
            row["packet_file_sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        )
        assert row["packet_self_sha256"] == value["sha256"]
        assert row["launchable"] is False
    assert not any("adapter_not_implemented" in blocker for blocker in evidence["common_blockers"])
    assert (
        "RL_specific_plan_bound_checkpoint_seal_and_BF16_export_adapter_implemented"
        in evidence["ready_now"]
    )
