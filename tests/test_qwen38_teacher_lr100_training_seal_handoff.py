"""Fail-closed terminal/seal handoff for the LR1e-4 dev segment."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-14-lr100-training-seal-handoff-v1.json"
CONFIG = ROOT / "configs/qualification/qwen38-teacher-lr100-cosine-layout-dev-v2.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence() -> dict:
    value = read(EVIDENCE)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_evidence_is_self_digesting_and_config_bound() -> None:
    value = evidence()
    assert value["source"]["config_file_sha256"] == file_sha256(CONFIG)
    assert value["source"]["output_root"] == read(CONFIG)["output_root"]
    assert value["run"]["jobs_api_posts"] == 1
    assert value["run"]["priority_class"] == "c1"
    assert value["run"]["queue_priority_class"] == "q1"
    assert value["run"]["effective_priority"] == 10_000


def test_missing_terminal_identity_and_pod_fields_are_preserved_explicitly() -> None:
    value = evidence()
    run = value["run"]
    provenance = value["pod_provenance"]
    assert run["api_run_id"] is None
    assert run["api_run_id_not_captured_before_terminal_cleanup"] is True
    assert run["resolved_runtime_image_digest"] == run["requested_image_digest"]
    assert provenance["exact_runtime_image_observed"] is True
    assert provenance["zero_restarts_observed_while_running"] is True
    assert provenance["pod_absent_at_terminal_reconciliation"] is True
    assert provenance["final_restart_count"] is provenance["final_exit_code"] is None
    assert provenance["full_lifecycle_zero_restart_claimed"] is False


def test_training_paused_cleanly_after_twenty_one_finite_updates() -> None:
    value = evidence()
    terminal = value["terminal_controller_reconciliation"]
    training = value["bounded_training"]
    receipt = value["terminal_receipt"]
    assert terminal["jobs_api_status"] == terminal["rayjob_status"] == "SUCCEEDED"
    assert terminal["workload_finished"] is True
    assert terminal["workload_terminal_reason"] == "Succeeded"
    assert receipt["status"] == "training_paused"
    assert receipt["optimizer_step"] == receipt["optimizer_steps_executed"] == 21
    assert training["planned_pause_after_optimizer_step"] == 21
    assert training["observed_optimizer_steps"] == training["local_metric_records"] == 21
    assert training["optimizer_step_sequence_contiguous_from_one"] is True
    assert training["all_observed_losses_finite"] is True
    assert training["all_observed_gradient_norms_finite"] is True
    assert training["all_gradient_finite_markers_true"] is True
    assert training["reference_cross_entropy_enabled"] is False
    assert training["teacher_cross_entropy_holdout_used"] is False


def test_tracking_checkpoint_creation_and_gpu_release_are_exact() -> None:
    value = evidence()
    wandb = value["wandb"]
    terminal = value["terminal_controller_reconciliation"]
    assert wandb["tracking_status"] == "synced"
    assert wandb["finish_acknowledged"] is True
    assert wandb["local_metric_events"] == 21
    assert wandb["attempted_metric_events"] == wandb["accepted_metric_events"] == 21
    assert wandb["scalar_coverage_complete"] is True
    assert wandb["automatic_system_telemetry"] is False
    assert terminal["raycluster_absent"] is terminal["head_pod_absent"] is True
    assert terminal["released_gpus"] == 8
    assert terminal["gpus_held"] == terminal["active_dev_gpu_requests_after_release"] == 0
    assert terminal["jobs_api_delete_required"] is terminal["jobs_api_delete_sent"] is False
    for step in (20, 21):
        checkpoint = value["checkpoints"][f"step_{step}"]
        assert checkpoint["files"] == 33
        assert checkpoint["bytes"] == 324_627_486_731
        assert checkpoint["producer_receipt_digest_valid"] is True


def test_seals_exist_but_independent_pair_acceptance_stays_closed() -> None:
    value = evidence()
    seals = value["checkpoint_seals"]
    verifier = value["independent_checkpoint_pair_verification"]
    assert seals["sealer"]["exit_code"] == seals["sealer"]["restarts"] == 0
    assert seals["sealer"]["gpus"] == 0
    assert seals["sealer"]["deleted"] is seals["sealer"]["absence_verified"] is True
    for step in (20, 21):
        seal = seals[f"step_{step}"]
        assert seal["files"] == 33
        assert seal["bytes"] == 324_627_486_731
        assert seal["sealer_returned_digest_valid_receipt"] is True
        assert seal["gpu_reload_verified_in_manifest"] is False
        assert seal["manifest_file_sha256"] is None
    assert verifier["state"] == "incomplete"
    assert verifier["accepted"] is False
    assert verifier["manifest_file_digests_independently_recorded"] is False
    assert verifier["both_source_checkpoints_fully_rehashed_with_one_terminal_proof"] is False


def test_verifier_failures_are_preserved_without_upgrading_evidence() -> None:
    verifier = evidence()["independent_checkpoint_pair_verification"]
    helper = verifier["helper"]
    first = verifier["first_exec_attempt"]
    corrected = verifier["corrected_exec_attempt"]
    assert helper["main_container_exit_code"] == helper["restarts"] == helper["gpus"] == 0
    assert helper["read_only_sfs"] is True
    assert helper["deleted"] is helper["absence_verified"] is True
    assert first["error_class"] == "verifier_only_contract_key_typo"
    assert first["checkpoint_20_check_files_verify_returned_before_error"] is True
    assert first["terminal_proof_emitted"] is False
    assert corrected["local_python_syntax_check_passed"] is True
    assert corrected["contract_key_changed_only_from_learning_rate_to_lr"] is True
    assert corrected["terminal_proof_emitted"] is False
    assert corrected["exec_exit_code"] == 137
    assert corrected["checkpoint_20_full_rehash_proof_preserved"] is False
    assert corrected["checkpoint_21_full_rehash_proof_preserved"] is False
    assert first["checkpoint_or_training_mutation"] is False
    assert corrected["checkpoint_or_training_mutation"] is False


def test_acceptance_and_next_gate_fail_closed() -> None:
    value = evidence()
    acceptance = value["acceptance"]
    assert acceptance["lr100_twenty_one_step_operational_dev_segment_accepted"] is True
    assert acceptance["training_tracking_and_checkpoint_creation_accepted"] is True
    assert acceptance["all_training_gpus_released"] is True
    assert acceptance["cpu_sealer_receipts_present_and_digest_valid"] is True
    for key in (
        "independent_checkpoint_pair_verification_accepted",
        "checkpoint_pair_accepted_for_reload",
        "gpu_reload_submitted",
        "zero_optimizer_reload_accepted",
        "hyperparameter_selected",
        "model_quality_claimed",
        "hf_export_accepted",
        "serving_reload_accepted",
        "live_eval_parity_accepted",
        "production_training_authorized",
    ):
        assert acceptance[key] is False
    assert value["blocker"]["kind"] == "fresh_nebius_authentication_required"
    assert value["blocker"]["gpu_reload_remains_closed"] is True


def test_evidence_contains_no_private_training_material() -> None:
    privacy = evidence()["privacy"]
    assert privacy == {
        "private_trainer_logs_read": False,
        "raw_tensor_values_read": False,
        "prompts_traces_flags_answers_credentials_or_scores_read": False,
    }
