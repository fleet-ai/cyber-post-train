from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
VALIDITY = EVIDENCE / "qwen38-dev17-pass4-validity-20260923.json"
AGGREGATE = EVIDENCE / "qwen38-dev17-pass4-aggregate-20260923.json"
CHRONOLOGY = EVIDENCE / "qwen38-dev17-pass4-chronology-correction-20260923.json"
REVIEW = EVIDENCE / "qwen38-dev17-pass4-independent-review-20260923.json"
SUPERSESSION = EVIDENCE / "qwen38-dev17-pass4-outcome-validity-supersession-20260924.json"
RECEIPTS = (VALIDITY, AGGREGATE, CHRONOLOGY, REVIEW, SUPERSESSION)


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


def test_supersession_preserves_scope_but_rejects_the_old_completion_claim() -> None:
    old = _read(VALIDITY)
    receipt = _read(SUPERSESSION)
    assert old["status"] == "valid_for_descriptive_aggregate"
    assert receipt["status"] == "incomplete_pending_predeclared_replacement_attempts"
    assert receipt["scope"] == {
        "benchmark": "fleet_development_dev17",
        "arms": ["base", "step1000"],
        "retained_seeds": [48, 54, 61, 62],
        "task_count": 17,
        "planned_attempts_per_arm": 68,
        "planned_pass_k": 4,
    }
    assert receipt["corrected_interpretation"]["complete_pass_at_4_available"] is False
    assert receipt["corrected_interpretation"]["paired_task_delta_available"] is False
    assert receipt["corrected_interpretation"]["paired_confidence_interval_available"] is False


def test_corrected_score_blind_denominators_exclude_held_attempts() -> None:
    receipt = _read(SUPERSESSION)
    census = receipt["score_blind_lifecycle_census"]
    assert census["base"] == {
        "normal_completed": 52,
        "held_output_limit": 15,
        "infrastructure_invalid_process_error": 1,
        "normal_completion_eligible_denominator": 52,
    }
    assert census["step1000"] == {
        "normal_completed": 18,
        "held_output_limit": 49,
        "infrastructure_invalid_process_error": 1,
        "normal_completion_eligible_denominator": 18,
    }
    for arm in census.values():
        assert (
            arm["normal_completed"]
            + arm["held_output_limit"]
            + arm["infrastructure_invalid_process_error"]
            == 68
        )
    assert receipt["source_evidence"] == {
        "kind": "score_blind_behavioral_lifecycle_audit",
        "records_classified": 136,
        "private_source_digest_included": False,
        "limitation": (
            "The public correction records only the sanitized lifecycle census. "
            "The private per-task-and-seed roster must be independently bound before "
            "it can authorize paired replacement cells."
        ),
    }


def test_supersession_binds_the_exact_original_receipts() -> None:
    superseded = {row["file"]: row for row in _read(SUPERSESSION)["superseded_evidence"]}
    assert superseded == {
        VALIDITY.name: {
            "file": VALIDITY.name,
            "file_sha256": "sha256:" + hashlib.sha256(VALIDITY.read_bytes()).hexdigest(),
            "self_sha256": _read(VALIDITY)["sha256"],
        },
        AGGREGATE.name: {
            "file": AGGREGATE.name,
            "file_sha256": "sha256:" + hashlib.sha256(AGGREGATE.read_bytes()).hexdigest(),
            "self_sha256": _read(AGGREGATE)["sha256"],
        },
        REVIEW.name: {
            "file": REVIEW.name,
            "file_sha256": "sha256:" + hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
            "self_sha256": _read(REVIEW)["sha256"],
        },
    }
    replacement = _read(SUPERSESSION)["replacement_rule"]
    assert replacement["replace_the_paired_task_seed_cell_when_either_arm_is_invalid"] is True
    assert replacement["silent_zero_imputation_forbidden"] is True
    assert replacement["scores_may_open_only_after_four_valid_paired_attempts_per_task"] is True


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


def test_independent_review_is_preserved_but_superseded() -> None:
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
    superseded = {row["file"] for row in _read(SUPERSESSION)["superseded_evidence"]}
    assert {VALIDITY.name, AGGREGATE.name, REVIEW.name} == superseded
