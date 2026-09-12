"""Terminal acceptance for the exact LR1 two-by-four zero-update reload."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-reload-terminal-v1.json"
CONFIG = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-reload-dev-v1.json"
SEAL = ROOT / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-checkpoint-seal-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reload_evidence_is_self_digesting_and_source_bound() -> None:
    evidence, seal = read(EVIDENCE), read(SEAL)
    assert evidence["sha256"] == digest_json(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    assert evidence["reload"]["config_file_sha256"] == file_sha256(CONFIG)
    bound = evidence["source"]["checkpoint_seal_evidence"]
    assert bound["file_sha256"] == file_sha256(SEAL)
    assert bound["embedded_sha256"] == seal["sha256"]
    assert (
        evidence["source"]["checkpoint_manifest"]["receipt_sha256"]
        == seal["seal"]["manifest_receipt_sha256"]
    )


def test_exact_dev_controller_succeeded_and_released_all_gpus() -> None:
    evidence = read(EVIDENCE)
    reload = evidence["reload"]
    assert reload["cluster"] == "dev"
    assert reload["kube_context"] == ("nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb")
    assert reload["api_run_id"] == "5f3ced34-9b14-4523-95d0-04eea1b7c3a1"
    assert reload["rayjob_uid"] == "075cef21-3b51-4110-8c48-79521d3791c0"
    assert reload["workload_uid"] == "ca26d9b5-3202-456a-9990-0a3949dfedd1"
    assert reload["topology"] == {"workers": 2, "gpus_per_worker": 4, "world_size": 8}
    assert (reload["priority_class"], reload["queue_priority_class"]) == ("c1", "q1")
    assert reload["effective_priority"] == 10_000
    assert reload["automatic_requeue"] is False
    terminal = evidence["terminal_controller_reconciliation"]
    assert terminal["jobs_api_status"] == terminal["rayjob_status"] == "SUCCEEDED"
    assert terminal["rayjob_deployment_status"] == "Complete"
    assert terminal["rayjob_failed_count"] == 0
    assert terminal["rayjob_succeeded_count"] == 1
    assert terminal["workload_finished"] is True
    assert terminal["workload_terminal_reason"] == "Succeeded"
    assert terminal["workload_quota_reserved"] is terminal["workload_admitted"] is False
    assert terminal["raycluster_absent"] is True
    assert terminal["head_pod_absent"] is terminal["worker_pod_absent"] is True
    assert terminal["released_gpus"] == 8
    assert terminal["gpus_held"] == 0


def test_all_rank_reload_was_digest_valid_and_executed_no_update() -> None:
    evidence = read(EVIDENCE)
    receipts = evidence["receipts"]
    assert all(value["digest_valid"] for value in receipts.values())
    recovered = receipts["recovered"]
    assert recovered["optimizer_step"] == 6
    assert recovered["rank_count"] == 8
    assert recovered["rank_ids"] == list(range(8))
    assert recovered["rank_optimizer_steps"] == [6]
    assert recovered["optimizer_states_per_rank"] == [851]
    assert recovered["optimizer_restored_all_ranks"] is True
    assert recovered["scheduler_restored_rank_ids"] == list(range(8))
    assert recovered["scheduler_restored_all_ranks"] is True
    assert recovered["rng_restored_all_ranks"] is True
    assert recovered["rng_evidence_kind"] == (
        "success_through_exact_hash_bound_runtime_guard_before_rank_receipt"
    )
    assert recovered["sampler_restored"] is True
    terminal = receipts["reload_validated"]
    assert terminal["status"] == "reload_validated"
    assert terminal["optimizer_step"] == 6
    assert terminal["optimizer_steps_executed"] == 0
    assert terminal["validation_scope"] == "checkpoint_and_sampler_reload_only_no_ce"
    assert terminal["source_manifest_sha256"] == recovered["source_manifest_sha256"]


def test_exact_recovery_runtime_enforced_optimizer_scheduler_and_rng() -> None:
    evidence = read(EVIDENCE)
    runtime = evidence["recovery_runtime_contract"]
    expected = "sha256:d728b9e6dca0973b3d42e4abe940f0ac4f7f844998e21bc0df974e9510a81533"
    assert runtime["repository_commit"] == evidence["reload"]["config_commit"]
    assert runtime["plan_recovery_runtime_sha256"] == expected
    assert runtime["staged_file_sha256"] == expected
    assert runtime["repository_file_sha256_at_commit"] == expected
    assert runtime["runtime_digest_match"] is True
    assert runtime["runtime_source_digest_validated_before_reload"] is True
    assert runtime["guards_before_rank_receipt"] == {
        "native_load_requires_optimizer_and_scheduler_states": True,
        "every_optimizer_counter_must_equal_manifest_step": True,
        "scheduler_last_epoch_must_equal_manifest_step": True,
        "scheduler_state_must_exactly_equal_checkpoint_state": True,
        "rng_state_must_exactly_equal_checkpoint_state": True,
        "rank_receipt_emitted_only_after_all_guards_pass": True,
    }


def test_independent_dev_rehash_and_scope_are_exact() -> None:
    evidence = read(EVIDENCE)
    audit = evidence["independent_dev_verification"]
    assert audit["helper_uid"] == "27bfa544-dd48-4ebd-9e23-0eea61c1f440"
    assert audit["gpus"] == audit["restarts"] == audit["mismatch_count"] == 0
    assert audit["verified_files"] == evidence["source"]["checkpoint_manifest"]["files"] == 33
    assert (
        audit["verified_bytes"]
        == evidence["source"]["checkpoint_manifest"]["bytes"]
        == 324_627_486_731
    )
    assert audit["source_manifest_digest_valid"] is True
    assert audit["source_checkpoint_unchanged_after_reload"] is True
    assert audit["reload_output_checkpoint_present"] is False
    assert audit["failure_or_rejection_receipt_present"] is False
    assert audit["training_terminal_receipt_present"] is False
    assert audit["helper_deleted"] is audit["helper_absence_verified"] is True
    helpers = evidence["contract_audit_helpers"]
    assert [item["uid"] for item in helpers] == [
        "d11df499-c600-4dbf-9fa8-ef469ecc9372",
        "22a219f2-06b8-446f-89a3-35022a13e07c",
        "df18dae2-10a9-4629-985e-127af0b8d29f",
    ]
    assert all(item["gpus"] == item["restarts"] == 0 for item in helpers)
    assert all(item["deleted"] is item["absence_verified"] is True for item in helpers)
    failed = helpers[1]
    assert failed["exit_code"] == 1
    assert failed["checkpoint_or_reload_mutation"] is False
    assert failed["jobs_api_posts"] == 0
    acceptance = evidence["acceptance"]
    assert acceptance["lr1_two_by_four_operational_dev_checkpoint_reload_accepted"] is True
    assert acceptance["next_lr30_dev_cell_may_enter_fresh_live_gates"] is True
    assert acceptance["hyperparameter_selected"] is False
    assert acceptance["capability_claim_allowed"] is False
    assert acceptance["literal_one_by_eight_layout_qualified"] is False
    assert acceptance["production_training_authorized"] is False


def test_wrong_default_context_is_explicitly_excluded_from_authority() -> None:
    correction = read(EVIDENCE)["context_correction"]
    assert correction == {
        "desktop_default_context_was_production": True,
        "first_read_only_audit_used_default_context": True,
        "first_audit_used_as_dev_authority": False,
        "first_helper_deleted_and_confirmed_absent": True,
        "authoritative_controller_and_receipt_audit_repeated_with_explicit_dev_context": True,
        "future_kubernetes_calls_must_pass_explicit_context": True,
    }
