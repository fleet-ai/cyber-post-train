"""Terminal evidence for the LR3e-5 two-by-four zero-update reload."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-14-lr30-2x4-reload-terminal-v1.json"
CONFIG = ROOT / "configs/qualification/qwen38-teacher-lr30-cosine-2x4-reload-dev-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_evidence_is_self_digesting_and_config_bound() -> None:
    evidence = read(EVIDENCE)
    assert evidence["sha256"] == digest_json(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    assert evidence["reload"]["config_file_sha256"] == file_sha256(CONFIG)


def test_controller_succeeded_and_released_all_gpus() -> None:
    terminal = read(EVIDENCE)["terminal_controller_reconciliation"]
    assert terminal["jobs_api_status"] == terminal["rayjob_status"] == "SUCCEEDED"
    assert terminal["rayjob_failed_count"] == 0
    assert terminal["rayjob_succeeded_count"] == 1
    assert terminal["workload_finished"] is True
    assert terminal["raycluster_absent"] is True
    assert terminal["head_pod_absent"] is terminal["worker_pod_absent"] is True
    assert terminal["released_gpus"] == 8
    assert terminal["gpus_held"] == 0
    assert terminal["jobs_api_delete_sent"] is False


def test_all_ranks_restored_without_an_optimizer_update() -> None:
    receipts = read(EVIDENCE)["receipts"]
    names = ("started", "recovered", "reload_validated")
    assert all(receipts[name]["digest_valid"] for name in names)
    recovered = receipts["recovered"]
    assert recovered["rank_count"] == 8
    assert recovered["rank_ids"] == list(range(8))
    assert recovered["rank_optimizer_steps"] == [6]
    assert recovered["optimizer_states_per_rank"] == [851]
    assert recovered["scheduler_restored_all_ranks"] is True
    assert recovered["sampler_restored"] is True
    terminal = receipts["reload_validated"]
    assert terminal["optimizer_step"] == 6
    assert terminal["optimizer_steps_executed"] == 0
    assert terminal["validation_scope"] == "checkpoint_and_sampler_reload_only_no_ce"
    assert receipts["reload_rejected_present"] is receipts["failed_present"] is False


def test_independent_post_reload_rehash_is_exact() -> None:
    audit = read(EVIDENCE)["independent_post_reload_verification"]
    assert audit["gpus"] == audit["restarts"] == audit["mismatch_count"] == 0
    assert audit["verified_files"] == 33
    assert audit["verified_bytes"] == 324_627_486_731
    assert audit["source_checkpoint_unchanged_after_reload"] is True
    assert audit["reload_output_checkpoint_present"] is False
    assert audit["failure_or_rejection_receipt_present"] is False
    assert audit["training_terminal_receipt_present"] is False
    assert audit["helper_deleted"] is audit["helper_absence_verified"] is True
    retired = audit["retired_packaging_attempt"]
    assert retired["gpus"] == retired["restarts"] == retired["jobs_api_posts"] == 0
    assert retired["checkpoint_or_training_mutation"] is False
    assert retired["deleted"] is retired["absence_verified"] is True


def test_acceptance_is_operational_only() -> None:
    acceptance = read(EVIDENCE)["acceptance"]
    assert acceptance["lr30_two_by_four_operational_dev_checkpoint_reload_accepted"] is True
    assert acceptance["zero_update_reload_accepted"] is True
    assert acceptance["source_checkpoint_integrity_accepted"] is True
    assert acceptance["hyperparameter_selected"] is False
    assert acceptance["model_quality_claimed"] is False
    assert acceptance["hf_export_accepted"] is False
    assert acceptance["serving_reload_accepted"] is False
    assert acceptance["live_eval_parity_accepted"] is False
    assert acceptance["production_training_authorized"] is False
