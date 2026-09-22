import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs"
    / "evidence"
    / "webexploitbench"
    / "2026-09-22-qwen38-opencode-full15-partial-baseline.json"
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_partial_baseline_receipt_is_self_consistent_and_excludes_invalid_cells() -> None:
    receipt = json.loads(EVIDENCE.read_text())
    result = receipt["aggregate_result"]
    conclusion = receipt["scientific_conclusion"]

    unsigned = dict(receipt)
    expected_receipt_sha256 = unsigned.pop("receipt_sha256")
    assert expected_receipt_sha256 == _canonical_sha256(unsigned)

    assert receipt["label"] == "PARTIAL baseline result"
    assert receipt["status"] == "partial"
    assert result["planned_websites"] == 15
    assert result["accepted_collections"] == 14
    assert result["collection_infrastructure_invalid"] == 1
    assert result["accepted_score_cells"] == 14
    assert result["scoring_infrastructure_invalid"] == 0
    assert result["accepted_collections"] + result["collection_infrastructure_invalid"] == 15
    assert (
        result["accepted_score_cells"]
        + result["scoring_infrastructure_invalid"]
        + result["collection_infrastructure_invalid"]
        == 15
    )
    assert result["scored_weakness_checks_with_credit"] == 11
    assert result["scored_weakness_checks_total"] == 98
    assert result["scored_weakness_check_rate"] == 11 / 98
    assert result["replayed_model_attempts"] == 0
    assert result["live_temporary_environments"] == 0

    assert {item["classification"] for item in receipt["exclusions"]} == {"infrastructure-invalid"}
    assert all("never counted as zero" in item["treatment"] for item in receipt["exclusions"])
    assert conclusion["complete_baseline_claim"] is False
    assert conclusion["training_improvement_claim"] is False
    assert conclusion["training_regression_claim"] is False
    assert conclusion["infrastructure_invalid_cells_counted_as_zero"] is False


def test_partial_baseline_receipt_is_aggregate_only() -> None:
    receipt = json.loads(EVIDENCE.read_text())
    serialized = json.dumps(receipt, sort_keys=True).lower()

    assert "/private/" not in serialized
    assert "private/tmp" not in serialized
    assert "task_id" not in serialized
    assert "target_name" not in serialized
    assert "session_id" not in serialized
    assert "sandbox_id" not in serialized
    assert "raw_trace" not in serialized
    assert "final_answer" not in serialized
    assert "flag{" not in serialized
    assert receipt["source_terminal_evidence"] == {
        "schema_version": "webexploitbench_full15_uniform_scoring_terminal_evidence_v1",
        "semantic_sha256": "f1f1a259d44a49253e6f806963a143c0ce314db51af1da3a96e569cc23a2f1cf",
        "self_sha256": "5c7f87fc5f2b431daf5e8c797a965e03ce806c8c7573ee505e357a7c65292372",
        "file_sha256": "524e5645cb51c594ddcace54480fc3469a5f305c787875474600cf9630d76378",
    }


def test_public_page_labels_the_new_result_partial_and_makes_no_lift_claim() -> None:
    html = (ROOT / "site" / "index.html").read_text()
    data = (ROOT / "site" / "report-data.js").read_text()
    app = (ROOT / "site" / "app.js").read_text()

    assert 'id: "opencode-fresh-base-partial"' in data
    assert 'status: "PARTIAL baseline"' in data
    assert 'warning: "PARTIAL — not a complete baseline"' in data
    assert 'attempts: "14 / 15"' in data
    assert 'scored: "14 / 15"' in data
    assert 'weaknessesFound: "11 / 98"' in data
    assert "excluded rather than counted as zero" in data
    assert "This partial result cannot show whether training helped or hurt the model" in data
    assert "A cleaner base-model test is still partial" in html
    assert "14 of 15 accepted score cells" in html
    assert "not a complete baseline" in app
    assert "does not show whether training helped" in app
    assert EVIDENCE.name in html
    assert EVIDENCE.name in data
