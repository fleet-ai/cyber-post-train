"""The corrected teacher-SFT search is exact, inert, and outcome-driven."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/studies/qwen-blackbox-teacher-staged-search-v3.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plan() -> dict:
    value = read(PLAN_PATH)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def assert_bound_json(binding: dict) -> dict:
    value = read(ROOT / binding["path"])
    assert binding["file_sha256"] == file_sha256(ROOT / binding["path"])
    assert binding["embedded_sha256"] == value["sha256"]
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_v3_is_a_nonlaunchable_successor_without_rewriting_v2() -> None:
    value = plan()
    assert value["study_id"] == "qwen38-blackbox-teacher-staged-search-v3"
    assert value["status"] == "frozen_nonlaunchable_successor"
    assert_bound_json(value["supersedes"])
    assert value["execution"] == {
        "kind": "sealed_metadata_only_not_a_job_request",
        "new_jobs_authorized": False,
        "cluster_or_api_mutations_performed": False,
        "all_arms_currently_launchable": False,
        "promotion": "dev_qualification_then_reviewed_prod_preview_and_explicit_submit",
    }
    assert all(not stage["launchable"] for stage in value["stages"])


def test_v3_binds_committed_four_attempt_protocols_and_base_controls() -> None:
    fleet = plan()["evidence"]["fleet_dev_v2"]
    assert fleet["implementation_commit"] == ("8c6e52120e7f702eadd8981dd05acf791b194e52")
    assert fleet["estimator"]["file_sha256"] == file_sha256(ROOT / fleet["estimator"]["path"])

    for split in ("a", "b"):
        protocol = assert_bound_json(fleet["protocols"][split])
        control = assert_bound_json(fleet["base_controls"][split])
        primary = protocol["metrics"]["primary"]
        uncertainty = protocol["metrics"]["uncertainty"]
        assert primary["name"] == "fleet_dev_paired_mean_success_delta_4fixed"
        assert primary["attempt_seeds"] == [42, 43, 44, 45]
        assert "task-version and attempt-seed pair" in primary["paired_comparison"]
        assert uncertainty["method"] == "paired task-clustered percentile bootstrap"
        assert uncertainty["within_cluster"] == (
            "retain all four matched base/candidate seed pairs"
        )
        assert control["parent_protocol"]["sha256"] == protocol["sha256"]
        assert fleet["base_controls"][split]["accepted_result_sha256"] is None


def test_v3_binds_scheduler_gate_and_classifies_numeric_canaries_only() -> None:
    evidence = plan()["evidence"]
    scheduler = evidence["scheduler_qualification"]
    template = read(ROOT / scheduler["template_path"])
    assert scheduler["template_file_sha256"] == file_sha256(ROOT / scheduler["template_path"])
    assert scheduler["template_embedded_sha256"] == template["sha256"]
    assert template["sha256"] == digest_json(
        {key: item for key, item in template.items() if key != "sha256"}
    )
    assert scheduler["implementation_commits"] == {
        "runtime_and_compiler": "593115cdda6316c29eda0c518a82970fb9e3fd48",
        "qualification_template": "16fbfdc56247eea19492aeb402739876eb099b22",
        "frozen_evidence_pin_tests": "71ac12300b2ecad668d1016b1b871bfc6b2b993b",
    }
    for component in ("runtime", "compiler"):
        assert scheduler[component]["file_sha256"] == file_sha256(
            ROOT / scheduler[component]["path"]
        )
    assert scheduler["candidate_pair_for_scientific_arms"] == {
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "state": "blocked_until_exact_dev_qualification_acceptance",
    }
    assert scheduler["accepted_receipt_sha256"] is None

    numeric = evidence["completed_numeric_canaries"]
    assert numeric["file_sha256"] == file_sha256(ROOT / numeric["path"])
    assert numeric["learning_rates"] == [1e-6, 3e-5]
    assert numeric["schedule_used"] == "historical_constant_no_warmup"
    assert numeric["classification"] == (
        "accepted_numeric_runtime_stability_only_not_scheduler_qualification_hpo_or_model_quality"
    )


def test_v3_lr_bracket_is_broad_literature_grounded_and_safety_gated() -> None:
    value = plan()
    broad = next(stage for stage in value["stages"] if stage["id"] == "split-a-broad-lr")
    assert broad["lr_bracket"] == [1e-6, 1e-5, 3e-5, 1e-4]
    assert [arm["lr"] for arm in broad["arms"]] == broad["lr_bracket"]
    assert broad["common"]["scheduler"] == "cosine"
    assert broad["common"]["warmup_ratio"] == 0.05
    assert set(broad["lr_rationale"]) == {"1e-6", "1e-5", "3e-5", "1e-4"}
    assert (
        "lr-1e-4-extended-dev-stability-and-reload_receipt_sha256"
        in broad["unresolved_dependencies"]
    )

    literature = value["evidence"]["literature_snapshot"]
    assert literature["file_sha256"] == file_sha256(ROOT / literature["path"])
    sources = {item["url"] for item in literature["primary_sources"]}
    assert "https://arxiv.org/html/2609.01244v1#S2.SS1" in sources
    assert "https://arxiv.org/html/2503.04625v1#S4.SS3" in sources
    assert "https://arxiv.org/html/2412.21139v1#S3.SS2" in sources


def test_v3_preserves_capacity_priority_tracking_and_sealing_boundaries() -> None:
    value = plan()
    invariants = value["study_invariants"]
    assert invariants["priority"] == {
        "cli_class": "c1",
        "expected_queue_class": "q1",
        "expected_value": 10_000,
    }
    assert invariants["requeue_if_preempted"] is False
    assert invariants["max_active_study_nodes"] == 8
    assert invariants["excluded_existing_inference_endpoint_nodes"] == 4
    assert max(stage["wave_nodes"] for stage in value["stages"]) <= 8
    assert [stage["order"] for stage in value["stages"]] == list(range(1, 8))

    wandb = value["evidence"]["wandb_scalar_contract"]
    assert wandb["history_keys"] == [
        "train/global_step",
        "train/loss",
        "train/total_supervised_tokens",
        "train/supervised_tokens",
        "train/lr",
        "train/grad_norm",
        "train/grad_norm_finite",
        "train/supervised_tokens_per_second",
    ]
    assert wandb["automatic_system_telemetry"] is False
    assert wandb["teacher_reference_cross_entropy"] == "not_computed_or_uploaded"
    assert {"traces", "flags", "answers", "task_scores", "evaluation_outcomes"} <= set(
        wandb["forbidden_payloads"]
    )

    web = assert_bound_json(value["evidence"]["webexploitbench_parent"])
    assert web["results"]["sealed_during_hpo_and_checkpoint_selection"] is True
    assert web["results"]["may_select_or_retime_training"] is False
    assert value["study_invariants"]["webexploitbench"] == (
        "sealed_and_forbidden_for_hpo_checkpoint_selection_or_tiebreaking"
    )


def test_v3_binds_the_frozen_exposure_matched_control() -> None:
    value = plan()
    exposure = value["evidence"]["exposure_matched_control"]
    control = assert_bound_json(exposure)
    assert exposure["state"] == "frozen_nonlaunchable_control"
    assert exposure["implementation_commit"] == ("b8471c4c911e2ed676e062518dba794325bada73")
    bound_fields = (
        "outcome_protocol_sha256",
        "balanced_corpus_sha256",
        "balanced_train_parquet_sha256",
        "matched_source_selection_sha256",
        "match_receipt_sha256",
        "selected_episode_ids_sha256",
        "matched_corpus_sha256",
        "matched_train_parquet_sha256",
    )
    for split in ("a", "b"):
        source = control["variants"][split]
        binding = exposure["variants"][split]
        for field in bound_fields:
            assert binding[field] == source[field]
        assert binding["matched_counts"] == {
            field: source["matched_control"][field]
            for field in ("episodes", "segments", "supervised_tokens")
        }
        assert binding["matched_counts"] == {
            field: source["balanced"][field]
            for field in ("episodes", "segments", "supervised_tokens")
        }

    stage = next(
        item for item in value["stages"] if item["id"] == "split-a-exposure-matched-control"
    )
    assert (
        "evidence.exposure_matched_control.embedded_sha256" not in stage["unresolved_dependencies"]
    )
    assert stage["unresolved_dependencies"] == [
        "split-a-broad-lr.outcome_barrier.selection_decision_sha256",
        "exposure_control_dev_qualification_receipt_sha256",
    ]
    assert stage["launchable"] is False
    assert "Input/context exposure is not matched" in stage["interpretation"]
