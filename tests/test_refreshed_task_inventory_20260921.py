"""Integrity checks for the 2026-09-21 metadata-only catalog refresh."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "data"
EVIDENCE = ROOT / "docs" / "evidence" / "fleet-task-inventory-20260921"
PREVIOUS_ALL = DATA / "fleet-blackbox-current-production-20260915-v1.json"
ALL = DATA / "fleet-blackbox-current-production-20260921-v1.json"
FILTERED = DATA / "fleet-blackbox-current-high-quality-20260914-v1.json"
PENDING = DATA / "fleet-blackbox-qa-candidates-20260914-v1.json"
COVERAGE = DATA / "fleet-blackbox-training-coverage-20260921-v1.json"
CENSUS = EVIDENCE / "catalog-census-v1.json"


def _self_digest(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert value["sha256"] == _self_digest(value)
    return value


def _rows(value: dict) -> list[dict]:
    fields = [field for field in ("tasks", "task_versions") if field in value]
    assert len(fields) == 1
    return value[fields[0]]


def _identities(value: dict) -> set[tuple[str, str]]:
    identities = {(row["task_id"], row["task_version_id"]) for row in _rows(value)}
    assert len(identities) == value["task_count"]
    return identities


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()


def test_current_blackbox_supply_grew_without_rewriting_the_accepted_set() -> None:
    previous = _load(PREVIOUS_ALL)
    current = _load(ALL)
    census = _load(CENSUS)
    filtered = _load(FILTERED)
    pending = _load(PENDING)

    previous_ids = _identities(previous)
    current_ids = _identities(current)
    added = [
        row
        for row in _rows(current)
        if (row["task_id"], row["task_version_id"]) not in previous_ids
    ]

    assert previous["task_count"] == 1055
    assert current["task_count"] == 1093
    assert not (previous_ids - current_ids)
    assert len(current_ids - previous_ids) == len(added) == 38
    assert all(row["task_shape"] == "blackbox" for row in added)
    assert {row["qa_status"] for row in added} == {"not_analyzed"}

    assert census["catalog"]["task_count"] == 1676
    assert census["catalog"]["lifecycle_status"]["production"] == 1288
    assert census["current_production_quality"]["task_shape"]["blackbox"] == 1093
    assert census["refreshed_selections"]["conservative_current_receipt_proven"] == 75
    assert census["refreshed_selections"]["new_qa_cleared_pending_receipts"] == 17
    # The census binds the selected rows; each dated file self-digests its
    # enclosing receipt.  Both identities are useful, but they are not the
    # same JSON object and must not be compared as if they were.
    assert census["refreshed_selections"]["conservative_current_receipt_proven_sha256"] == _digest(
        _rows(filtered)
    )
    assert census["refreshed_selections"]["new_qa_cleared_pending_receipts_sha256"] == _digest(
        _rows(pending)
    )


def test_receipt_coverage_partitions_current_supply_without_scores_or_sessions() -> None:
    current = _load(ALL)
    coverage = _load(COVERAGE)
    counts = coverage["counts"]

    assert counts == {
        "current_blackbox_task_keys": 1093,
        "current_exact_receipt_task_keys": 80,
        "current_exact_success_proven_task_keys": 44,
        "current_exact_canonical_failure_task_keys": 36,
        "high_quality_exact_success_proven_task_keys": 42,
        "high_quality_exact_canonical_failure_task_keys": 33,
        "excluded_exact_success_proven_task_keys": 2,
        "excluded_exact_canonical_failure_task_keys": 3,
        "no_exact_receipt_in_this_refresh_task_keys": 1013,
        "no_exact_success_proven_task_keys": 1049,
        "proven_failure_only_task_keys": None,
    }
    successes = set(coverage["current_exact_success_proven_task_keys"])
    failures = set(coverage["current_exact_canonical_failure_task_keys"])
    missing = set(coverage["no_exact_receipt_in_this_refresh_task_keys"])
    current_keys = {row["task_key"] for row in _rows(current)}
    assert not (successes & failures)
    assert successes | failures | missing == current_keys
    assert coverage["numeric_scores_persisted"] is False
    assert coverage["session_ids_persisted"] is False


def test_new_refresh_artifacts_have_no_private_payload_fields() -> None:
    forbidden = {
        "prompt",
        "trace",
        "answer",
        "flag",
        "credential",
        "score",
        "session_id",
        "verifier_execution_id",
    }
    for path in (ALL, COVERAGE, CENSUS):
        value = _load(path)
        assert value["private_content_persisted"] is False
        assert not (_keys(value) & forbidden)
