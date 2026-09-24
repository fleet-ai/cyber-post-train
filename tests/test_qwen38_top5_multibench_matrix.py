"""Prospective, no-launch contract for the Qwen3.8 baseline-plus-top-five study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/evaluation/qwen38-top5-multibench-pass4-matrix-20260923-v1.json"
EXPOSURE_AUDIT = ROOT / "docs/evidence/qwen38-fleet-final8-exposure-audit-20260923.json"


def load() -> dict:
    return json.loads(MATRIX.read_text())


def test_matrix_is_self_bound_and_authorizes_no_work() -> None:
    matrix = load()
    assert matrix["sha256"] == digest(
        {key: value for key, value in matrix.items() if key != "sha256"}
    )
    assert matrix["status"] == "prepared_no_launch"
    assert matrix["launchable"] is False
    assert all(value == 0 for value in matrix["operation"].values())


def test_selection_is_prospective_and_fixes_exact_six_arms() -> None:
    matrix = load()
    assert matrix["scientific_boundary"]["fleet_confirmation_outcomes_used_for_selection"] is False
    assert matrix["selection"]["ranked_candidate_ids"] == [
        "q38-teacher3k-32k-step1000",
        "q38-d32-b16-lr5e6-step300",
        "q38-t3k64-b8-step285",
        "q38-t3k32-lr1-step700",
        "q38-t3k96-b8-step300",
    ]
    assert [arm["arm_id"] for arm in matrix["arms"]] == [
        "base",
        "step1000",
        "b16_step300",
        "context64_step285",
        "lr1_step700",
        "context96_step300",
    ]
    context96 = next(arm for arm in matrix["arms"] if arm["arm_id"] == "context96_step300")
    assert context96["recipe"]["optimizer_step"] == 300
    assert context96["recipe"]["supervised_tokens_seen"] == 20_234_966
    assert context96["checkpoint"]["receipt_file_sha256"] == (
        "sha256:c0c146b342b13da64943e6ae8a46b4ca4b4d14b1355e8950aedeed0d58ba5dfd"
    )
    assert context96["fallback"]["artifact_id"] == "q38-t3k96-b8-step100"


def test_pass4_and_locked_fleet_confirmation_set_are_frozen() -> None:
    matrix = load()
    protocol = matrix["evaluation_protocol"]
    assert protocol["pass_k"] == 4
    assert protocol["independent_pass1_attempt_indices"] == [0, 1, 2, 3]
    assert protocol["attempt_labels"] == ["r00", "r01", "r02", "r03"]
    fleet = next(row for row in matrix["benchmarks"] if row["benchmark_id"] == "fleet_final_test8")
    assert fleet["task_count_execution_roster"] == 8
    assert fleet["selection_sha256"] == (
        "sha256:9623149c4a021bc13ed2cf94ca26e107b30c18816cf3c02d76b5020cab5066f4"
    )
    assert fleet["seeds"] == [46, 47, 48, 49]
    assert fleet["selection_role"] == "historically_exposed_locked_confirmation_set"
    assert fleet["outcomes_state"] == "outcomes_not_used_for_checkpoint_selection"
    audit = matrix["fleet_confirmation_exposure_audit"]
    assert audit["path"] == str(EXPOSURE_AUDIT.relative_to(ROOT))
    assert (
        audit["file_sha256"] == "sha256:" + hashlib.sha256(EXPOSURE_AUDIT.read_bytes()).hexdigest()
    )
    assert audit["receipt_sha256"] == json.loads(EXPOSURE_AUDIT.read_text())["sha256"]
    assert audit["disposition"] == fleet["selection_role"]
    assert audit["selection_outcomes_used"] is False
    assert audit["historical_cells_may_substitute_for_new_matched_cells"] is False
    assert fleet["exposure_audit_file_sha256"] == audit["file_sha256"]
    assert fleet["exposure_audit_receipt_sha256"] == audit["receipt_sha256"]
    assert fleet["historical_execution_boundary"] == {
        "exact_versions_with_prior_quality_certification_metadata": 8,
        "exact_versions_with_accepted_pre_split_model_comparison_execution": 2,
        "exact_versions_with_only_unstarted_pre_split_model_comparison_cells": 6,
        "historical_cells_reused": 0,
        "new_matched_outcomes_required": 192,
    }


def test_sampling_is_bound_per_benchmark_without_global_override() -> None:
    matrix = load()
    assert "temperature" not in matrix["evaluation_protocol"]["cross_arm_controls"]
    benchmarks = {row["benchmark_id"]: row for row in matrix["benchmarks"]}
    assert benchmarks["webexploitbench"]["sampling"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "seed": None,
        "max_output_tokens": 32768,
    }
    assert benchmarks["fleet_final_test8"]["sampling"]["temperature"] == 0.6
    for benchmark_id in ("cvebench_zero_day", "nyu_ctf_web", "cybench_web"):
        assert benchmarks[benchmark_id]["sampling"] == {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": None,
            "seed": None,
        }


def test_matrix_shape_and_alert_gate_are_explicit() -> None:
    matrix = load()
    assert matrix["matrix_shape"] == {
        "arm_count": 6,
        "attempts_per_task": 4,
        "official_task_rows": 88,
        "execution_roster_task_rows": 84,
        "official_cells": 2112,
        "execution_roster_cells_if_all_benchmark_gates_pass": 2016,
        "formula": "arms times attempts times tasks",
    }
    assert any("fleet_ai_failure_alerts_off" in gate for gate in matrix["global_launch_gates"])
