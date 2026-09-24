"""Prospective, no-launch contract for the Qwen3.8 baseline-plus-top-five study."""

from __future__ import annotations

import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/evaluation/qwen38-top5-multibench-pass4-matrix-20260923-v1.json"


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
    assert matrix["scientific_boundary"]["fleet_final_test_outcomes_read"] is False
    assert matrix["selection"]["ranked_candidate_ids"] == [
        "q38-teacher3k-32k-step1000",
        "q38-d32-b16-lr5e6-step300",
        "q38-t3k64-b8-step285",
        "q38-t3k32-lr1-step700",
        "q38-t3k96-b8-step100",
    ]
    assert [arm["arm_id"] for arm in matrix["arms"]] == [
        "base",
        "step1000",
        "b16_step300",
        "context64_step285",
        "lr1_step700",
        "context96_step100",
    ]


def test_pass4_and_untouched_fleet_final_test_are_frozen() -> None:
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
    assert fleet["outcomes_state"] == "sealed_not_read"


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
