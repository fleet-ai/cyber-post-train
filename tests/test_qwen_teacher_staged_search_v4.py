"""The accepted world-size-eight reload opens only a bounded cadence gate."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-production-layout-dev-v1.json"
CADENCE = ROOT / "configs/qualification/qwen38-teacher-checkpoint-cadence-dev-v1.json"
V3 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v3.json"
V4 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v4.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-12-production-layout-reload-dev-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_self_digested(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_layout_evidence_binds_the_exact_zero_update_fragmented_reload() -> None:
    evidence = assert_self_digested(EVIDENCE)
    reload_config = read(
        ROOT
        / "configs/qualification/qwen38-teacher-production-layout-reload-fragmented-dev-v2.json"
    )
    assert evidence["run"]["config"]["file_sha256"] == file_sha256(
        ROOT / evidence["run"]["config"]["path"]
    )
    assert evidence["run"]["config_name"] == reload_config["name"]
    assert evidence["source_checkpoint"] == {
        "manifest_path": reload_config["recovery"]["manifest"],
        "manifest_file_sha256": "sha256:" + reload_config["recovery"]["sha256"],
        "manifest_receipt_sha256": (
            "sha256:8cbc5ca42dafe6e1137e4c09a8620c207cbc0a60e330226f7dcb43de10ca7af3"
        ),
        "optimizer_step": 6,
        "world_size": 8,
    }
    terminal = evidence["terminal_evidence"]
    assert terminal["recovered_file_sha256"] == (
        "sha256:59924e458c057e827345840659c430f91f0fd0d49a80b9065db3e3cf5fdb8ccb"
    )
    assert terminal["reload_validated_file_sha256"] == (
        "sha256:08194c425c24a6d4c5175e083495511dedbd7f31f21b9f237f38eb26896ac385"
    )
    assert terminal["rank_count"] == 8
    assert terminal["optimizer_states_per_rank"] == 851
    assert terminal["scheduler_state_restored"] is True
    assert terminal["sampler_state_restored"] is True
    assert terminal["optimizer_updates_executed"] == terminal["container_restarts"] == 0
    assert terminal["resources_released"] is True


def test_cadence_canary_changes_only_identities_layout_and_checkpoint_gate() -> None:
    source, cadence = read(SOURCE), read(CADENCE)
    restored = copy.deepcopy(cadence)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["wandb"] = source["wandb"]
    restored["recipe"]["nodes"] = source["recipe"]["nodes"]
    restored["recipe"]["gpus_per_node"] = source["recipe"]["gpus_per_node"]
    restored["recipe"]["checkpoint_interval"] = source["recipe"]["checkpoint_interval"]
    restored["pause_after_step"] = source["pause_after_step"]
    assert restored == source


def test_cadence_canary_is_bounded_world_size_eight_and_scalar_safe() -> None:
    cadence, v4 = read(CADENCE), assert_self_digested(V4)
    recipe = cadence["recipe"]
    expected = v4["next_dev_gate"]["expected"]
    assert cadence["name"] == cadence["wandb"]["run_id"] == cadence["wandb"]["name"]
    assert len(cadence["name"]) <= 31
    assert recipe["nodes"] * recipe["gpus_per_node"] == expected["world_size"] == 8
    assert recipe["batch_size"] == expected["global_batch"] == 8
    assert recipe["microbatch_per_gpu"] == expected["microbatch_per_gpu"] == 1
    assert recipe["checkpoint_interval"] == expected["checkpoint_interval"] == 20
    assert recipe["keep_checkpoints"] == expected["keep_checkpoints"] == 3
    assert cadence["pause_after_step"] == expected["pause_after_optimizer_step"] == 21
    assert cadence["pause_after_step"] == recipe["checkpoint_interval"] + 1
    assert expected["full_horizon_steps"] == math.ceil(602 / recipe["batch_size"])
    assert expected["warmup_steps"] == math.ceil(
        expected["full_horizon_steps"] * recipe["warmup_ratio"]
    )
    assert cadence["cluster"]["priority"] == "c1"
    assert "queue_priority_class" not in cadence["cluster"]
    assert "requeueIfPreempted" not in cadence
    assert cadence["validation_mode"] == "task_outcomes_only"
    assert "outcome-only" in cadence["wandb"]["tags"]
    assert "checkpoint-cadence-20" in cadence["wandb"]["tags"]


def test_v4_binds_v3_evidence_and_cadence_without_opening_production() -> None:
    v3, v4 = assert_self_digested(V3), assert_self_digested(V4)
    assert v4["supersedes"]["file_sha256"] == file_sha256(V3)
    assert v4["supersedes"]["embedded_sha256"] == v3["sha256"]
    evidence_binding = v4["accepted_dev_evidence"]["fragmented_layout_reload"]
    assert evidence_binding["file_sha256"] == file_sha256(EVIDENCE)
    assert evidence_binding["embedded_sha256"] == assert_self_digested(EVIDENCE)["sha256"]
    assert v4["next_dev_gate"]["config"]["file_sha256"] == file_sha256(CADENCE)
    assert v4["next_dev_gate"]["accepted_receipt_sha256"] is None
    assert v4["next_dev_gate"]["launchable_by_this_file"] is False
    assert v4["production_gate"]["launchable"] is False
    assert v4["execution"] == {
        "kind": "sealed_metadata_only_not_a_job_request",
        "new_jobs_authorized": False,
        "cluster_or_api_mutations_performed": False,
        "all_arms_currently_launchable": False,
        "promotion": (
            "accepted_dev_cadence_and_matched_base_control_then_reviewed_prod_preview_"
            "and_explicit_submit"
        ),
    }


def test_v4_fails_closed_on_the_exact_missing_split_a_base_control() -> None:
    v3, v4 = assert_self_digested(V3), assert_self_digested(V4)
    binding = v4["production_gate"]["base_control"]
    control = assert_self_digested(ROOT / binding["path"])
    assert binding["file_sha256"] == file_sha256(ROOT / binding["path"])
    assert binding["embedded_sha256"] == control["sha256"]
    assert binding["accepted_result_sha256"] is None
    assert v3["evidence"]["fleet_dev_v2"]["base_controls"]["a"]["accepted_result_sha256"] is None
    assert (
        "production_gate.base_control.accepted_result_sha256"
        in v4["production_gate"]["unresolved_dependencies"]
    )
    assert control["serving_binding"]["launchable"] is False
    assert control["launch_gate"]["paid_or_scored_work_authorized_by_this_file"] is False


def test_first_arm_is_frozen_but_not_executable_and_inherits_scalar_safety() -> None:
    v3, v4 = assert_self_digested(V3), assert_self_digested(V4)
    arm = v4["production_gate"]["first_arm"]
    assert arm["state"] == "frozen_candidate_not_a_job_request"
    assert arm["name"] == arm["wandb"]["run_id"] == arm["wandb"]["name"]
    assert arm["recipe"]["max_steps"] == math.ceil(
        arm["data"]["rows"] / arm["recipe"]["batch_size"]
    )
    assert arm["recipe"]["checkpoint_interval"] == 20
    assert arm["recipe"]["keep_checkpoints"] == 3
    assert arm["recipe"]["nodes"] == 1
    assert arm["recipe"]["gpus_per_node"] == 8
    assert arm["cluster"]["priority"] == "c1"
    assert arm["cluster"]["expected_priority_value"] == 10_000
    assert arm["cluster"]["requeue_if_preempted"] is False
    assert arm["validation_mode"] == "task_outcomes_only"
    assert arm["wandb"]["automatic_system_telemetry"] is False
    assert arm["wandb"]["payload_policy"] == "scalar_allowlist_inherited_from_v3"
    scalar_contract = v3["evidence"]["wandb_scalar_contract"]
    assert scalar_contract["teacher_reference_cross_entropy"] == "not_computed_or_uploaded"
    assert {"traces", "flags", "answers", "task_scores", "evaluation_outcomes"} <= set(
        scalar_contract["forbidden_payloads"]
    )
