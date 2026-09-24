import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/evidence/qwen38-study/2026-09-24-q38-step1000-causal-synthesis-v1.json"
REPORT = ROOT / "docs/QWEN38_TEACHER3K_STEP1000_CAUSAL_SYNTHESIS_2026-09-24.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value: dict, digest_field: str = "sha256") -> str:
    unsigned = {key: item for key, item in value.items() if key != digest_field}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in all_keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in all_keys(item)}
    return set()


def test_causal_synthesis_is_self_digesting_and_source_bound() -> None:
    receipt = load(RECEIPT)
    assert receipt["sha256"] == "sha256:" + canonical_digest(receipt)

    for binding in receipt["source_bindings"].values():
        path = ROOT / binding["path"]
        assert file_digest(path) == binding["file_sha256"]
        source = load(path)
        if "logical_sha256" in binding:
            assert source["sha256"] == binding["logical_sha256"]
            assert source["sha256"].removeprefix("sha256:") == canonical_digest(source)
        if "receipt_sha256" in binding:
            assert source["receipt_sha256"] == binding["receipt_sha256"]
            assert source["receipt_sha256"] == canonical_digest(
                source, digest_field="receipt_sha256"
            )


def test_historical_signal_is_exact_but_not_treated_as_a_valid_effect() -> None:
    receipt = load(RECEIPT)
    observed = receipt["historical_regression_signal"]
    assert observed["task_families"] == 17
    assert observed["attempts_per_arm"] == 68
    assert observed["base"] == {
        "pass_at_4_solved_families": 7,
        "successful_attempts": 15,
    }
    assert observed["step1000"] == {
        "pass_at_4_solved_families": 0,
        "successful_attempts": 0,
    }
    assert observed["step1000_minus_base"]["solved_families"] == -7
    assert observed["interpretation"] == (
        "superseded_descriptive_aggregate_not_a_valid_capability_effect"
    )
    assert observed["original_receipt_preserved"] is True
    disagreement = receipt["evidence_disagreement"]
    assert "valid_for_descriptive_aggregate" in disagreement["older_immutable_receipts"]
    assert "not present on current merged main" in disagreement["missing_from_current_main"]
    assert "hold" in disagreement["resolution"]

    gate = receipt["outcome_validity_precondition"]
    assert gate["roster"] == "lineage_clean_dev13"
    assert gate["required_pairs"] == 13
    assert "-3/13" in gate["confirm_historical_harm"]
    assert "p-value is <= 0.05" in gate["confirm_historical_harm"]
    assert gate["current_state"] == (
        "evidence_conflict_held_pending_corrected_classifier_and_complete_paired_replacements"
    )


def test_every_hypothesis_has_a_mechanism_confidence_and_falsifier() -> None:
    receipt = load(RECEIPT)
    hypotheses = {row["id"]: row for row in receipt["hypotheses"]}
    assert set(hypotheses) == {
        "model_facing_tool_contract_mismatch",
        "raw_token_clipping_and_anchor_loss",
        "source_and_lineage_concentration",
        "family_alias_duplication_and_heldout_leakage",
        "export_serving_or_checkpoint_stage_difference",
    }
    for row in hypotheses.values():
        assert row["mechanism"]
        assert row["confidence"]
        test = row["matched_heldout_test"]
        assert test["design"]
        assert any(key.startswith("confirm") for key in test)
        assert any(key.startswith("refute") for key in test)

    tool = hypotheses["model_facing_tool_contract_mismatch"]["evidence"]
    assert tool["synthetic_initial_context_token_delta"] == 874
    assert tool["candidate_invalid_bare_bash_events"] == 32
    assert tool["candidate_attempts_with_invalid_bare_bash"] == 24
    assert tool["candidate_valid_fleet_bash_events"] == 5023
    assert (
        tool["candidate_valid_fleet_bash_completed"] + tool["candidate_valid_fleet_bash_errors"]
        == 5023
    )
    assert tool["candidate_valid_fleet_submit_report_completed"] == 0
    assert tool["base_valid_fleet_submit_report_completed"] == 20

    clipped = hypotheses["raw_token_clipping_and_anchor_loss"]["evidence"]
    assert clipped["step1000_clipped_rows"] == 6451
    assert clipped["step1000_rows_consumed"] == 8000
    assert clipped["sampled_clipped_rows_without_expected_chat_delimiter"] == 254
    assert clipped["sampled_clipped_rows"] == 256

    concentration = hypotheses["source_and_lineage_concentration"]["evidence"]
    assert concentration["largest_teacher_token_percent"] == 49.141954
    assert concentration["largest_transitive_lineage_component_token_percent"] == 13.271579
    assert concentration["step1000_exact_source_mix_available"] is False

    assert math.comb(5, 5) / (2**5) == 0.03125
    assert "0.03125" in receipt["common_diagnostic_rule"]["strong_material_support"]


def test_synthesis_is_aggregate_only_and_performs_no_external_operation() -> None:
    receipt = load(RECEIPT)
    assert not ({"task_key", "task_version_id", "component_id", "session_id"} & all_keys(receipt))
    encoded = json.dumps(receipt, sort_keys=True)
    assert "cysec" not in encoded
    assert (
        re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            encoded,
        )
        is None
    )
    assert receipt["privacy"] == {
        "task_keys_or_version_ids_included": False,
        "prompt_response_trace_answer_flag_or_per_task_score_content_included": False,
        "private_source_rows_read": False,
        "external_benchmark_content_included": False,
    }
    assert receipt["effects"] == {
        "external_jobs_created": 0,
        "model_calls": 0,
        "scoring_calls": 0,
        "cluster_or_serving_mutations": 0,
        "private_data_materializations": 0,
    }
    assert receipt["scope"]["external_jobs_authorized"] is False
    assert receipt["scope"]["causal_claim_authorized"] is False
    assert receipt["scope"]["historical_dev17_capability_claim_authorized"] is False

    report = REPORT.read_text()
    assert "outcome-valid regression" in report
    assert "regression and causation not yet" in report
    assert "no launch" in report
