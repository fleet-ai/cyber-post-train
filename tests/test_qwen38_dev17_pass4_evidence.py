from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
VALIDITY = EVIDENCE / "qwen38-dev17-pass4-validity-20260923.json"
AGGREGATE = EVIDENCE / "qwen38-dev17-pass4-aggregate-20260923.json"
CHRONOLOGY = EVIDENCE / "qwen38-dev17-pass4-chronology-correction-20260923.json"
REVIEW = EVIDENCE / "qwen38-dev17-pass4-independent-review-20260923.json"
RECEIPTS = (VALIDITY, AGGREGATE, CHRONOLOGY, REVIEW)


def _canonical_sha256(value: dict[str, Any]) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    payload = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    assert path.is_file() and not path.is_symlink()
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert value["sha256"] == _canonical_sha256(value)
    return value


def test_receipts_are_self_digesting_and_public_safe() -> None:
    forbidden_key_fragments = (
        "cell_id",
        "database",
        "job_uid",
        "session_id",
        "task_id",
        "verifier_id",
    )
    for path in RECEIPTS:
        value = _read(path)
        serialized = json.dumps(value, sort_keys=True)
        assert "/private/tmp" not in serialized
        assert "/mnt/sfs" not in serialized
        assert "q38_dev17_" not in serialized
        assert (
            re.search(
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
                serialized,
                re.IGNORECASE,
            )
            is None
        )
        for key in _walk_keys(value):
            assert not any(fragment in key.lower() for fragment in forbidden_key_fragments)


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [key for key, nested in value.items() for key in (key, *_walk_keys(nested))]
    if isinstance(value, list):
        return [key for nested in value for key in _walk_keys(nested)]
    return []


def test_score_blind_validity_gate_is_complete_and_matched() -> None:
    receipt = _read(VALIDITY)
    assert receipt["status"] == "valid_for_descriptive_aggregate"
    assert receipt["scope"] == {
        "benchmark": "fleet_development_dev17",
        "arms": ["base", "step1000"],
        "retained_seeds": [48, 54, 61, 62],
        "task_count": 17,
        "attempts_per_task_arm": 4,
        "attempts_per_arm": 68,
    }
    assert receipt["shared_protocol"]["harness"] == "opencode"
    assert receipt["shared_protocol"]["harness_version"] == "1.18.27"
    gate = receipt["completion_gate"]
    assert gate["arm_count"] == 8
    assert gate["all_arms_match"] is True
    assert gate["expected_records_per_arm"] == gate["accepted_records_per_arm"] == 17
    assert gate["locally_preserved_results_per_arm"] == 17
    assert gate["completed_runs_per_arm"] == 17
    assert gate["records_needing_review_per_arm"] == 0
    assert gate["stale_active_records_per_arm"] == 0
    assert gate["duplicate_within_arm_detected"] is False
    assert receipt["source_evidence"]["score_data_opened_during_validity_check"] is False


def test_aggregate_counts_rates_and_limitations_are_exact() -> None:
    receipt = _read(AGGREGATE)
    assert receipt["status"] == "final_descriptive"
    assert receipt["scope"]["descriptive_only"] is True
    assert receipt["scope"]["final_heldout_result"] is False
    assert receipt["scope"]["task_count"] == 17
    assert receipt["scope"]["attempts_per_arm"] == 68

    base = receipt["results"]["base"]
    candidate = receipt["results"]["step1000"]
    delta = receipt["results"]["step1000_minus_base"]
    assert base["task_level_pass_at_4_percent"] == pytest.approx(100 * 7 / 17)
    assert base["attempt_level_pass_at_1_percent"] == pytest.approx(100 * 15 / 68)
    assert candidate["task_level_pass_at_4_percent"] == 0.0
    assert candidate["attempt_level_pass_at_1_percent"] == 0.0
    assert delta["task_level_pass_at_4_percentage_points"] == pytest.approx(-100 * 7 / 17)
    assert delta["attempt_level_pass_at_1_percentage_points"] == pytest.approx(-100 * 15 / 68)
    assert delta["paired_task_bootstrap_95_percent_interval_percentage_points"] == pytest.approx(
        [-64.70588235294117, -17.647058823529413]
    )
    assert receipt["uncertainty"]["replicates"] == 10000
    assert receipt["uncertainty"]["resampling_unit"] == "task"


def test_prospective_selection_and_corrected_chronology_precede_score_opening() -> None:
    aggregate = _read(AGGREGATE)
    selection = aggregate["selection_boundary"]
    assert selection["prospective_plan_pull_request"] == 560
    assert selection["prospective_plan_head"] == "731ef08d261e84a5e248d5964d38a2b839703a0f"
    assert selection["frozen_six_arm_roster"] == [
        "base",
        "step1000",
        "b16_step300",
        "context64_step285",
        "lr1_step700",
        "context96_step100",
    ]
    assert selection["plan_predated_score_opening"] is True
    assert selection["evaluation_outcomes_used_for_selection"] is False

    correction = _read(CHRONOLOGY)
    events = correction["authoritative_order"]
    assert [row["event"] for row in events] == [
        "six_arm_checkpoint_plan_committed",
        "selection_boundary_final_bytes_written",
        "independently_reviewed_aggregation_program_written",
        "aggregate_receipt_written_after_authorized_score_opening",
    ]
    times = [datetime.fromisoformat(row["time_utc"].replace("Z", "+00:00")) for row in events]
    assert times == sorted(times)
    assert len(set(times)) == len(times)
    assert (
        correction["scientific_effect"]["evaluation_outcomes_used_for_checkpoint_selection"]
        is False
    )
    assert correction["scientific_effect"]["aggregate_metrics_changed"] is False


def test_independent_review_binds_the_exact_aggregate_claim() -> None:
    aggregate = _read(AGGREGATE)
    review = _read(REVIEW)
    assert review["status"] == "review_complete"
    assert review["review"]["preexecution_program_review"] == "GO"
    assert review["review"]["postexecution_integrity_review"] == "GO"
    assert review["review"]["aggregate_metrics_recomputed"] is True
    assert review["review"]["blocker_found"] is False
    claim = review["reviewed_claim"]
    assert (
        claim["base_task_level_pass_at_4_percent"]
        == aggregate["results"]["base"]["task_level_pass_at_4_percent"]
    )
    assert (
        claim["base_attempt_level_pass_at_1_percent"]
        == aggregate["results"]["base"]["attempt_level_pass_at_1_percent"]
    )
    assert (
        claim["step1000_task_level_pass_at_4_percent"]
        == aggregate["results"]["step1000"]["task_level_pass_at_4_percent"]
    )
    assert (
        claim["paired_delta_percentage_points"]
        == aggregate["results"]["step1000_minus_base"]["task_level_pass_at_4_percentage_points"]
    )
