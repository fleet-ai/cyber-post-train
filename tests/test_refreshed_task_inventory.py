import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs" / "data"
EVIDENCE = ROOT / "docs" / "evidence" / "fleet-task-inventory-20260915"
PREVIOUS_ALL = DATA / "fleet-blackbox-current-production-20260914-v1.json"
ALL = DATA / "fleet-blackbox-current-production-20260915-v1.json"
FILTERED = DATA / "fleet-blackbox-current-high-quality-20260914-v1.json"
PENDING = DATA / "fleet-blackbox-qa-candidates-20260914-v1.json"
COVERAGE = DATA / "fleet-blackbox-training-coverage-20260915-v1.json"
CENSUS = EVIDENCE / "catalog-census-v1.json"
ADMIN = EVIDENCE / "admin-workflow-count-v1.json"
RECHECK = EVIDENCE / "catalog-recheck-1616-v1.json"


def assert_self_digest(value: dict) -> None:
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


def digest_json(value: dict) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )


def rows(value: dict) -> list[dict]:
    fields = [field for field in ("tasks", "task_versions") if field in value]
    assert len(fields) == 1
    return value[fields[0]]


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert_self_digest(value)
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

    assert all_tasks["task_count"] == 1055
    assert filtered["task_count"] == 75
    assert pending["task_count"] == 17
    all_ids = identities(all_tasks)
    assert identities(filtered) < all_ids
    assert identities(pending) < all_ids
    assert "task_versions" in filtered
    assert all(row["task_shape"] == "blackbox" for row in rows(all_tasks))
    assert all(row["qa_status"] != "broken_task" for row in rows(filtered))
    assert all(row["qa_status"] in {"clean", "agent_failure"} for row in rows(pending))


def test_current_inventory_is_one_exact_addition_from_the_previous_refresh() -> None:
    previous = load(PREVIOUS_ALL)
    current = load(ALL)
    previous_ids = identities(previous)
    current_ids = identities(current)

    assert not (previous_ids - current_ids)
    assert len(current_ids - previous_ids) == 1
    added = [
        row
        for row in rows(current)
        if (row["task_id"], row["task_version_id"]) in current_ids - previous_ids
    ]
    assert len(added) == 1
    assert added[0]["task_shape"] == "blackbox"
    assert added[0]["qa_status"] == "not_analyzed"


def test_live_census_and_admin_workflow_count_keep_evidence_tiers_separate() -> None:
    census = json.loads(CENSUS.read_text())
    admin = json.loads(ADMIN.read_text())
    assert_self_digest(census)
    assert_self_digest(admin)

    assert census["catalog"]["task_count"] == 1633
    assert census["catalog"]["lifecycle_status"]["production"] == 1250
    assert census["current_production_quality"]["task_shape"]["blackbox"] == 1055
    assert census["refreshed_selections"]["conservative_current_receipt_proven"] == 75
    assert census["private_content_persisted"] is False

    assert admin["matching_top_level_workflows"] == 176
    assert admin["accepted_entry_inventory_available"] is False
    assert "not an accepted-task" in admin["meaning"]
    assert admin["private_content_persisted"] is False


def test_later_recheck_is_self_digesting_and_proves_no_selection_change() -> None:
    recheck = json.loads(RECHECK.read_text())
    receipt = recheck.pop("receipt_sha256")
    assert receipt == digest_json(recheck)

    census = json.loads(CENSUS.read_text())
    all_tasks = load(ALL)
    filtered = load(FILTERED)
    pending = load(PENDING)
    coverage = json.loads(COVERAGE.read_text())
    semantic_coverage = {
        key: value for key, value in coverage.items() if key not in {"observed_through", "sha256"}
    }
    program = ROOT / recheck["program"]["path"]

    assert recheck["program"]["file_sha256"] == (
        "sha256:" + hashlib.sha256(program.read_bytes()).hexdigest()
    )
    assert recheck["prior_census"]["receipt_sha256"] == census["sha256"]
    assert recheck["prior_census"]["observed_through"] == census["observed_through"]
    assert recheck["recheck"]["all_blackbox_receipt_sha256"] == all_tasks["sha256"]
    assert recheck["recheck"]["high_quality_receipt_sha256"] == filtered["sha256"]
    assert recheck["recheck"]["pending_receipt_sha256"] == pending["sha256"]
    assert recheck["recheck"]["training_coverage_semantic_sha256"] == digest_json(semantic_coverage)
    assert recheck["recheck"]["production_blackbox_versions"] == all_tasks["task_count"]
    assert recheck["recheck"]["high_quality_versions"] == filtered["task_count"]
    assert recheck["recheck"]["pending_quality_candidates"] == pending["task_count"]
    assert (
        recheck["recheck"]["current_exact_execution_receipts"]
        == coverage["counts"]["current_exact_receipt_task_keys"]
    )
    assert recheck["change_since_prior_census"] == {
        "catalog_selected_versions": 0,
        "production_versions": 0,
        "production_blackbox_versions": 0,
        "high_quality_versions": 0,
        "pending_quality_candidates": 0,
        "published_task_source_keys": 2,
    }
    assert recheck["private_content_persisted"] is False


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
    assert_self_digest(coverage)
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
    assert len(set(missing)) == len(missing) == 975
    assert not (set(successes) & set(failures))
    assert set(high_quality_successes) | set(excluded_successes) == set(successes)
    assert set(high_quality_failures) | set(excluded_failures) == set(failures)
    assert coverage["counts"]["proven_failure_only_task_keys"] is None
    assert coverage["numeric_scores_persisted"] is False
    assert coverage["session_ids_persisted"] is False
