"""LR1 acceptance advances only the exact dev study cell to fresh LR30 gates."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v6.json"
V7 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v7.json"
V8 = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v8.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def evidence_ref(binding: dict) -> dict:
    path = ROOT / binding["path"]
    value = sealed(path)
    assert binding == {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": file_sha256(path),
        "embedded_sha256": value["sha256"],
    }
    return value


def test_v8_is_an_immutable_metadata_successor() -> None:
    old, value = sealed(V7), sealed(V8)
    assert value["supersedes"]["path"] == str(V7.relative_to(ROOT))
    assert value["supersedes"]["file_sha256"] == file_sha256(V7)
    assert value["supersedes"]["embedded_sha256"] == old["sha256"]
    assert value["launchable"] is False
    assert value["execution"] == {
        "kind": "metadata_only_not_a_job_request",
        "cluster_or_api_mutations_performed": False,
        "new_training_producer_created": False,
        "production_submission_authorized": False,
    }


def test_v8_cross_checks_the_exact_lr1_producer_seal_and_reload() -> None:
    value, v7 = sealed(V8), sealed(V7)
    accepted = value["accepted_lr1"]
    terminal = evidence_ref(accepted["evidence"]["training_terminal"])
    seal = evidence_ref(accepted["evidence"]["checkpoint_seal"])
    reload = evidence_ref(accepted["evidence"]["reload_terminal"])

    assert accepted["cluster"] == terminal["run"]["cluster"] == reload["reload"]["cluster"] == "dev"
    assert accepted["config"]["file_sha256"] == file_sha256(ROOT / accepted["config"]["path"])
    assert accepted["reload_config"]["file_sha256"] == file_sha256(
        ROOT / accepted["reload_config"]["path"]
    )
    successor = v7["successor_two_by_four"]
    assert accepted["config"]["path"] == successor["config_path"]
    assert accepted["config"]["file_sha256"] == successor["config_file_sha256"]
    assert (
        accepted["topology"]
        == successor["topology"]
        == {
            "nodes": 2,
            "gpus_per_node": 4,
            "world_size": 8,
        }
    )
    ids = accepted["identities"]
    assert ids["producer_api_run_id"] == terminal["run"]["api_run_id"]
    assert ids["producer_rayjob_uid"] == terminal["run"]["rayjob_uid"] == successor["rayjob_uid"]
    assert (
        ids["producer_workload_uid"] == terminal["run"]["workload_uid"] == successor["workload_uid"]
    )
    assert ids["reload_api_run_id"] == reload["reload"]["api_run_id"]
    assert ids["reload_rayjob_uid"] == reload["reload"]["rayjob_uid"]
    assert ids["reload_workload_uid"] == reload["reload"]["workload_uid"]
    assert accepted["common_image"] == terminal["run"]["requested_image"]
    assert accepted["common_image"].endswith("@" + reload["reload"]["requested_image_digest"])
    assert (
        seal["seal"]["manifest_receipt_sha256"]
        == accepted["gate_facts"]["checkpoint_manifest_receipt_sha256"]
    )
    assert (
        reload["source"]["checkpoint_manifest"]["receipt_sha256"]
        == seal["seal"]["manifest_receipt_sha256"]
    )


def test_v8_accepts_all_rank_state_recovery_but_not_a_completed_lr_arm() -> None:
    value = sealed(V8)
    accepted = value["accepted_lr1"]
    training = evidence_ref(accepted["evidence"]["training_terminal"])
    seal = evidence_ref(accepted["evidence"]["checkpoint_seal"])
    reload = evidence_ref(accepted["evidence"]["reload_terminal"])
    recovered = reload["receipts"]["recovered"]
    facts = accepted["gate_facts"]
    paused = training["receipts"]["training_paused"]
    assert facts["finite_optimizer_steps_completed"] == paused["optimizer_steps_executed"] == 6
    assert facts["planned_optimizer_steps"] == paused["planned_optimizer_steps"] == 76
    assert facts["checkpoint_step"] == seal["seal"]["optimizer_step"] == 6
    assert recovered["rank_ids"] == recovered["scheduler_restored_rank_ids"] == list(range(8))
    assert recovered["rank_optimizer_steps"] == [6]
    assert recovered["optimizer_states_per_rank"] == [851]
    assert recovered["optimizer_restored_all_ranks"] is True
    assert recovered["scheduler_restored_all_ranks"] is True
    assert recovered["rng_restored_all_ranks"] is True
    assert recovered["sampler_restored"] is True
    assert reload["recovery_runtime_contract"]["runtime_digest_match"] is True
    assert all(reload["recovery_runtime_contract"]["guards_before_rank_receipt"].values())
    assert facts["all_eight_ranks_restored"] is True
    assert facts["optimizer_restored_all_ranks"] is True
    assert facts["scheduler_restored_all_ranks"] is True
    assert facts["rng_restored_all_ranks"] is True
    assert facts["sampler_restored"] is True
    assert (
        facts["reload_optimizer_steps_executed"]
        == reload["receipts"]["reload_validated"]["optimizer_steps_executed"]
        == 0
    )
    audit = reload["independent_dev_verification"]
    assert (
        facts["source_checkpoint_unchanged"]
        is audit["source_checkpoint_unchanged_after_reload"]
        is True
    )
    assert (
        facts["reload_output_checkpoint_present"]
        is audit["reload_output_checkpoint_present"]
        is False
    )
    terminal = reload["terminal_controller_reconciliation"]
    assert terminal["released_gpus"] == 8
    assert terminal["gpus_held"] == 0
    assert facts["all_allocated_gpus_released"] is True
    assert accepted["accepted"] is True
    assert accepted["completed_lr_arm"] is False
    assert accepted["full_lifecycle_zero_restart_claimed"] is False


def test_v8_inherits_the_complete_v6_pre_post_gate_contract() -> None:
    value = sealed(V8)
    contract = value["fresh_gate_contract"]
    source = evidence_ref(contract["source"])
    assert contract["json_pointer"] == "/pre_post_gates"
    assert contract["required_gate_count"] == len(source["pre_post_gates"]) == 11
    assert contract["preserve_source_gates_verbatim"] is True
    joined = " ".join(source["pre_post_gates"])
    for required in (
        "Create-once",
        "authenticated dev Jobs API",
        "UIDs plus imageID/restarts",
        "single durable pre-POST intent",
        "ambiguous responses are a hold",
    ):
        assert required in joined


def test_v8_orders_lr30_then_lr100_and_preserves_wave_release_gates() -> None:
    value, v6, v7 = sealed(V8), sealed(V6), sealed(V7)
    waves = value["remaining_dev_waves"]
    assert [wave["order"] for wave in waves] == [1, 2]
    old = {cell["id"]: cell for cell in v7["unchanged_cells"]}
    for wave, label in zip(waves, ("lr30", "lr100"), strict=True):
        cell = wave["cell"]
        assert cell["id"] == label
        assert cell["config_path"] == old[label]["config_path"]
        assert cell["config_file_sha256"] == old[label]["config_file_sha256"]
        assert cell["config_file_sha256"] == file_sha256(ROOT / cell["config_path"])
        assert cell["topology"] == old[label]["topology"]
        assert cell["fresh_gates_verified"] is False
        assert cell["jobs_api_post_authorized_by_this_document"] is False
    assert waves[0]["inherited_wave_max_nodes"] == v6["dev_waves"][0]["max_nodes"] == 2
    assert waves[0]["cell"]["status"] == "eligible_for_fresh_live_gates_only"
    assert waves[1]["inherited_wave_max_nodes"] == v6["dev_waves"][1]["max_nodes"] == 1
    assert waves[1]["requires"] == [
        "lr30_terminal_reconciliation",
        "first_wave_allocation_release",
        "all_fresh_pre_post_gates_rechecked",
    ]
    assert waves[1]["cell"]["status"] == (
        "blocked_until_first_wave_terminal_release_and_fresh_gates"
    )


def test_v8_preserves_every_scientific_and_production_boundary() -> None:
    value = sealed(V8)
    assert set(value["claim_boundaries"].values()) == {False}
    assert value["production"] == {"state": "blocked", "authorization_added": False}
    invariants = value["invariants"]
    assert invariants["one_canonical_cell_one_accepted_or_active_producer"] is True
    assert invariants["lr1_must_never_be_replayed"] is True
    assert invariants["c1_q1_effective_priority"] == 10_000
    assert invariants["automatic_requeue"] is False
    assert invariants["maximum_active_experiment_nodes_excluding_inference"] == 8
    assert invariants["wandb_scalar_only"] is True
    assert invariants["validation_mode"] == "task_outcomes_only"
    assert invariants["reference_cross_entropy_enabled"] is False
