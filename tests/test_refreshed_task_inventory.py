import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "data"
ALL = DATA / "fleet-blackbox-current-production-20260914-v1.json"
FILTERED = DATA / "fleet-blackbox-current-high-quality-20260914-v1.json"
PENDING = DATA / "fleet-blackbox-qa-candidates-20260914-v1.json"
COVERAGE = DATA / "fleet-blackbox-training-coverage-20260914-v1.json"


def rows(value: dict) -> list[dict]:
    fields = [field for field in ("tasks", "task_versions") if field in value]
    assert len(fields) == 1
    return value[fields[0]]


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {key: item for key, item in value.items() if key != "sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert value["sha256"] == expected
    assert value["task_count"] == len(rows(value))
    return value


def identities(value: dict) -> set[tuple[str, str]]:
    values = {(row["task_id"], row["task_version_id"]) for row in rows(value)}
    assert len(values) == value["task_count"]
    return values


def test_current_blackbox_inventory_and_filters_are_exact_subsets() -> None:
    all_tasks = load(ALL)
    filtered = load(FILTERED)
    pending = load(PENDING)

    assert all_tasks["task_count"] == 1054
    assert filtered["task_count"] == 75
    assert pending["task_count"] == 17
    all_ids = identities(all_tasks)
    assert identities(filtered) < all_ids
    assert identities(pending) < all_ids
    assert "task_versions" in filtered
    assert all(row["task_shape"] == "blackbox" for row in rows(all_tasks))
    assert all(row["qa_status"] != "broken_task" for row in rows(filtered))
    assert all(row["qa_status"] in {"clean", "agent_failure"} for row in rows(pending))


def test_filtered_tasks_have_reviewed_split_metadata_and_provenance() -> None:
    filtered = load(FILTERED)
    required = {
        "lineage_key",
        "application",
        "environment",
        "task_family",
        "vulnerability_family",
        "difficulty",
    }
    for row in rows(filtered):
        assert set(row["lineage"]) == required
        assert row["lineage"]["lineage_key"]
        assert row["lineage"]["application"]
        assert row["lineage"]["environment"]
        assert row["lineage"]["task_family"]
        assert row["lineage"]["vulnerability_family"]
        assert row["lineage"]["difficulty"] in {"easy", "medium", "hard"}
        assert (
            row["provenance"]["evidence_class"]
            == "prior_exact_execution_receipt_current_not_known_broken"
        )


def test_public_manifests_contain_no_private_payload_fields() -> None:
    forbidden = {"prompt", "trace", "answer", "flag", "credential", "score"}
    for manifest in (load(ALL), load(FILTERED), load(PENDING)):
        for row in rows(manifest):
            assert forbidden.isdisjoint(row)


def test_training_coverage_is_task_key_only_and_partitions_current_inventory() -> None:
    coverage = json.loads(COVERAGE.read_text())
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                {key: item for key, item in coverage.items() if key != "sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert coverage["sha256"] == expected
    successes = coverage["current_exact_success_proven_task_keys"]
    failures = coverage["current_exact_canonical_failure_task_keys"]
    high_quality_successes = coverage["high_quality_exact_success_proven_task_keys"]
    high_quality_failures = coverage["high_quality_exact_canonical_failure_task_keys"]
    excluded_successes = coverage["excluded_exact_success_proven_task_keys"]
    excluded_failures = coverage["excluded_exact_canonical_failure_task_keys"]
    missing = coverage["no_exact_receipt_in_this_refresh_task_keys"]
    groups = [
        successes,
        failures,
        high_quality_successes,
        high_quality_failures,
        excluded_successes,
        excluded_failures,
        missing,
    ]
    assert all(isinstance(key, str) and key for group in groups for key in group)
    assert len(set(successes)) == len(successes) == 44
    assert len(set(failures)) == len(failures) == 36
    assert len(set(high_quality_successes)) == len(high_quality_successes) == 42
    assert len(set(high_quality_failures)) == len(high_quality_failures) == 33
    assert len(set(excluded_successes)) == len(excluded_successes) == 2
    assert len(set(excluded_failures)) == len(excluded_failures) == 3
    assert len(set(missing)) == len(missing) == 974
    assert not (set(successes) & set(failures))
    assert set(high_quality_successes) | set(excluded_successes) == set(successes)
    assert set(high_quality_failures) | set(excluded_failures) == set(failures)
    assert coverage["counts"]["proven_failure_only_task_keys"] is None
    assert coverage["numeric_scores_persisted"] is False
    assert coverage["session_ids_persisted"] is False
