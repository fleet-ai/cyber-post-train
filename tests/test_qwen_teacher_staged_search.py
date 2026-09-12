"""Regression checks for the nonlaunchable staged Qwen teacher-SFT search."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
PLAN_PATH = (
    ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v2.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_staged_search_is_self_digested_evidence_bound_and_nonlaunchable():
    plan = read(PLAN_PATH)
    unsigned = {key: value for key, value in plan.items() if key != "sha256"}
    assert plan["sha256"] == digest_json(unsigned)
    assert plan["execution"] == {
        "kind": "sealed_metadata_only_not_a_job_request",
        "new_jobs_authorized": False,
        "cluster_or_api_mutations_performed": False,
        "all_arms_currently_launchable": False,
        "promotion": "dev_qualification_then_reviewed_prod_preview_and_explicit_submit",
    }

    for evidence_name in ("balanced_exposure_treatment", "base_route_audit"):
        evidence = plan["evidence"][evidence_name]
        value = read(ROOT / evidence["path"])
        assert value["sha256"] == evidence["sha256"]
        assert value["sha256"] == digest_json(
            {key: item for key, item in value.items() if key != "sha256"}
        )

    # This plan is frozen, nonlaunchable historical metadata. Its evidence pins
    # intentionally remain at the implementation commits that sealed it rather
    # than following later runtime edits in place.
    for evidence_name in ("outcome_sealing", "wandb_scalar_contract"):
        evidence = plan["evidence"][evidence_name]
        assert evidence["file_sha256"].startswith("sha256:")
        assert evidence["file_sha256"] != file_sha256(ROOT / evidence["path"])

    numeric_template_evidence = plan["evidence"]["numeric_lr_canary_template"]
    numeric_template = read(ROOT / numeric_template_evidence["path"])
    assert numeric_template["sha256"] == numeric_template_evidence["sha256"]
    assert numeric_template["sha256"] == digest_json(
        {key: value for key, value in numeric_template.items() if key != "sha256"}
    )
    assert numeric_template["launchable"] is False
    assert (
        numeric_template["runtime"]["sha256"]
        == numeric_template_evidence["runtime_sha256"]
    )
    assert (
        numeric_template["runtime"]["implementation_commit"]
        == numeric_template_evidence["runtime_implementation_commit"]
        == plan["evidence"]["wandb_scalar_contract"]["implementation_commit"]
    )

    assert plan["evidence"]["base_route_audit"][
        "accepted_as_serving_or_parity_evidence"
    ] is False
    assert plan["evidence"]["wandb_scalar_contract"][
        "dev_gpu_qualification_receipt_sha256"
    ] is None
    assert all(not stage["launchable"] for stage in plan["stages"])


def test_only_fixed_safe_scalars_are_sent_to_wandb_and_hpo_is_outcome_only():
    plan = read(PLAN_PATH)
    contract = plan["evidence"]["wandb_scalar_contract"]
    assert contract["history_keys"] == [
        "train/global_step",
        "train/loss",
        "train/total_supervised_tokens",
        "train/supervised_tokens",
        "train/lr",
        "train/grad_norm",
        "train/grad_norm_finite",
        "train/supervised_tokens_per_second",
    ]
    assert contract["series_step_axis"] == "train/total_supervised_tokens"
    assert contract["automatic_system_telemetry"] is False
    assert {"task_scores", "evaluation_outcomes", "traces", "token_ids"} <= set(
        contract["forbidden_payloads"]
    )
    assert plan["study_invariants"]["selection_signal"] == (
        "fresh_fleet_dev_pass_at_1_task_outcomes_only"
    )
    assert plan["study_invariants"]["teacher_reference_cross_entropy"] == (
        "not_computed_or_used"
    )
    assert plan["study_invariants"]["training_loss"] == "diagnostic_only"


def test_stages_are_serial_bounded_and_preserve_the_requested_search_order():
    plan = read(PLAN_PATH)
    stages = plan["stages"]
    assert [stage["order"] for stage in stages] == list(range(1, 7))
    assert [stage["id"] for stage in stages] == [
        "dev-lr-extreme-numeric-canaries",
        "dev-production-layout-canary",
        "split-a-broad-lr",
        "split-a-balanced-exposure",
        "split-b-and-seed-confirmation",
        "batch-and-horizon-refinement",
    ]
    assert max(stage["wave_nodes"] for stage in stages) <= 8
    assert plan["study_invariants"]["max_active_study_nodes"] == 8
    assert plan["study_invariants"]["stage_concurrency"] == (
        "one_stage_wave_at_a_time_after_prior_allocations_release"
    )

    numeric, layout, broad, balanced, confirmation, refinement = stages
    assert [arm["lr"] for arm in numeric["arms"]] == [1e-6, 3e-5]
    assert all(arm["pause_after_optimizer_step"] == 2 for arm in numeric["arms"])
    assert "exact_8_gpu" in layout["arms"][0]["acceptance"]
    assert [arm["lr"] for arm in broad["arms"]] == [1e-6, 3e-6, 1e-5, 3e-5]
    assert {(arm["split"], arm["seed"]) for arm in confirmation["arms"]} == {
        ("a", 20260911),
        ("b", 42),
        ("b", 20260911),
    }
    assert {(arm.get("global_batch", 8), arm["epochs"]) for arm in refinement["arms"]} == {
        (16, 1),
        (32, 1),
        (8, 2),
        (8, 4),
    }
    assert balanced["arms"][0]["lr"] is None
    assert all(arm["lr"] is None for arm in confirmation["arms"])
    assert all(arm["lr"] is None for arm in refinement["arms"])


def test_uplift_and_later_treatments_remain_blocked_on_fresh_outcomes():
    plan = read(PLAN_PATH)
    assert all(
        control["accepted_result_sha256"] is None
        and control["state"] == "blocked_missing_exact_serving_and_live_pair_parity"
        for control in plan["fresh_base_controls"].values()
    )
    stages = {stage["id"]: stage for stage in plan["stages"]}
    assert "fresh_base_controls.a.accepted_result_sha256" in stages[
        "split-a-broad-lr"
    ]["unresolved_dependencies"]
    assert stages["split-a-broad-lr"]["outcome_barrier"][
        "selection_decision_sha256"
    ] is None
    assert stages["split-a-balanced-exposure"]["arms"][0]["corpus"] == (
        "balanced_a"
    )
    assert stages["split-a-balanced-exposure"]["outcome_barrier"][
        "treatment_decision_sha256"
    ] is None
    assert "fresh_base_controls.b.accepted_result_sha256" in stages[
        "split-b-and-seed-confirmation"
    ]["unresolved_dependencies"]
    assert stages["split-b-and-seed-confirmation"]["outcome_barrier"][
        "confirmation_decision_sha256"
    ] is None
    assert stages["batch-and-horizon-refinement"]["outcome_barrier"][
        "refinement_decision_sha256"
    ] is None


def test_balanced_corpus_bindings_match_the_teacher_availability_treatment():
    plan = read(PLAN_PATH)
    treatment = read(
        ROOT / plan["evidence"]["balanced_exposure_treatment"]["path"]
    )
    for split in ("a", "b"):
        variant = treatment["variants"][split]
        available = plan["teacher_corpora"][f"available_{split}"]
        balanced = plan["teacher_corpora"][f"balanced_{split}"]
        assert available["manifest_sha256"] == variant["available_corpus_sha256"]
        assert available["source_selection_sha256"] == variant[
            "available_source_selection_sha256"
        ]
        assert balanced["manifest_sha256"] == variant["balanced_corpus_sha256"]
        assert balanced["source_selection_sha256"] == variant[
            "balanced_source_selection_sha256"
        ]
        assert balanced["sessions"] == variant["balanced"]["episodes"]
        assert balanced["supervised_tokens"] < available["supervised_tokens"]
