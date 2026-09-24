"""Availability-only amendment for the frozen top-five multibench study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/evaluation/qwen38-top5-multibench-pass4-matrix-20260923-v1.json"
MATRIX = (
    ROOT / "configs/evaluation/qwen38-top5-multibench-pass4-operational-matrix-20260924-v2.json"
)
EVIDENCE = ROOT / "docs/evidence/qwen38-top5-retention-fallbacks-20260924.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_amendment_is_self_bound_and_authorizes_no_work() -> None:
    matrix = load(MATRIX)
    assert matrix["sha256"] == digest(
        {key: value for key, value in matrix.items() if key != "sha256"}
    )
    assert matrix["evaluation_protocol"]["campaign_matrix_input_digest_field"] == "sha256"
    assert matrix["status"] == "prepared_no_launch"
    assert matrix["launchable"] is False
    assert all(value == 0 for value in matrix["operation"].values())


def test_base_matrix_and_retention_evidence_are_exactly_bound() -> None:
    matrix = load(MATRIX)
    evidence = load(EVIDENCE)
    assert matrix["base_matrix"]["file_sha256"] == (
        "sha256:" + hashlib.sha256(BASE.read_bytes()).hexdigest()
    )
    assert matrix["base_matrix"]["matrix_sha256"] == "sha256:" + load(BASE)["sha256"]
    assert matrix["amendment_evidence"]["file_sha256"] == (
        "sha256:" + hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()
    )
    assert matrix["amendment_evidence"]["receipt_sha256"] == "sha256:" + evidence["sha256"]
    assert evidence["sha256"] == digest(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )


def test_only_three_availability_substitutions_are_made() -> None:
    matrix = load(MATRIX)
    assert matrix["scientific_boundary"] == {
        "prospective_selection_order_changed": False,
        "availability_substitutions_only": True,
        "benchmark_outcomes_used": False,
        "fleet_confirmation_outcomes_used": False,
        "webexploitbench_outcomes_used": False,
        "external_benchmark_outcomes_used": False,
        "benchmark_protocol_changed": False,
        "pass_k_changed": False,
    }
    assert matrix["operational_arm_order"] == [
        "base",
        "step1000",
        "b16_step200",
        "context64_step225",
        "lr1_step500",
        "context96_step300",
    ]
    substitutions = [arm for arm in matrix["arm_bindings"] if arm["substitution"]]
    assert [(arm["frozen_arm_id"], arm["operational_arm_id"]) for arm in substitutions] == [
        ("b16_step300", "b16_step200"),
        ("context64_step285", "context64_step225"),
        ("lr1_step700", "lr1_step500"),
    ]
    assert all(arm["promotion_state"] == "accepted_through_cpu_validation" for arm in substitutions)


def test_matched_pass4_benchmark_protocol_is_inherited_unchanged() -> None:
    matrix = load(MATRIX)
    protocol = matrix["evaluation_protocol"]
    assert protocol["pass_k"] == 4
    assert protocol["attempt_indices"] == [0, 1, 2, 3]
    assert protocol["attempt_labels"] == ["r00", "r01", "r02", "r03"]
    assert protocol["benchmark_ids"] == [
        "webexploitbench",
        "fleet_final_test8",
        "cvebench_zero_day",
        "nyu_ctf_web",
        "cybench_web",
    ]
    assert any("fleet_ai_failure_alerts_off" in gate for gate in matrix["global_launch_gates"])
