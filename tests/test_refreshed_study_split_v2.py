import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from training.study_split_v2 import (
    TARGET_COUNTS,
    assign,
    canonical_digest,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "configs/data/fleet-blackbox-current-high-quality-20260914-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"


def load() -> tuple[dict, list[dict]]:
    value = json.loads(SPLIT.read_text())
    inventory = json.loads(INVENTORY.read_text())["task_versions"]
    validate(value, inventory_path=INVENTORY.relative_to(ROOT))
    return value, inventory


def test_split_assigns_every_exact_version_once_without_family_leakage() -> None:
    value, inventory = load()
    expected = {(row["task_key"], row["task_version_id"]) for row in inventory}
    actual = [(row["task_key"], row["task_version_id"]) for row in value["tasks"]]
    assert len(actual) == len(set(actual)) == 75
    assert set(actual) == expected
    assert Counter(row["split"] for row in value["tasks"]) == Counter(TARGET_COUNTS)

    family_splits = defaultdict(set)
    by_identity = {(row["task_key"], row["task_version_id"]): row for row in inventory}
    for task in value["tasks"]:
        row = by_identity[(task["task_key"], task["task_version_id"])]
        family_splits[(row["lineage"]["application"], row["lineage"]["task_family"])].add(
            task["split"]
        )
    assert all(len(splits) == 1 for splits in family_splits.values())
    assert value["leakage_checks"] == {
        "all_inventory_versions_assigned_once": True,
        "exact_identity_overlap": 0,
        "reviewed_family_overlap": 0,
    }


def test_train_and_evaluation_selections_are_disjoint_and_digest_bound() -> None:
    value, _ = load()
    train = value["training_split"]
    dev = value["evaluation"]["dev"]
    final = value["evaluation"]["final_test"]
    groups = [
        {(row["task_key"], row["task_version_id"]) for row in part["tasks"]}
        for part in (train, dev, final)
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert [len(group) for group in groups] == [50, 17, 8]
    for part in (train, dev, final):
        assert part["sha256"] == canonical_digest(
            {key: item for key, item in part.items() if key != "sha256"}
        )


def test_representable_strata_occur_in_every_split_and_tiny_policy_is_explicit() -> None:
    value, _ = load()
    for dimension in ("application", "environment", "difficulty", "vulnerability_family"):
        summary = value["representation"][dimension]
        for label, count in summary["population_counts"].items():
            if count >= 5:
                assert all(
                    summary["split_counts"][split].get(label, 0) > 0 for split in TARGET_COUNTS
                )
    assert value["policy"]["tiny_strata"] == {
        "count_1": "record placement; do not claim cross-split representation",
        "count_2_to_4": "balance proportionally when compatible with stronger strata",
        "count_5_or_more": "require at least one example in every split",
    }


def test_assignment_is_deterministic_and_order_independent() -> None:
    _, inventory = load()
    assert assign(inventory) == assign(list(reversed(inventory)))


def test_split_contains_only_safe_task_identity_fields() -> None:
    value, _ = load()
    assert all(
        set(row) == {"task_key", "task_version_id", "group_id", "split"} for row in value["tasks"]
    )
    forbidden = {"prompt", "trace", "answer", "flag", "credential", "score", "session_id"}
    assert forbidden.isdisjoint(json.dumps(value).lower().split('"'))


def test_validator_rejects_assignment_tampering() -> None:
    value, _ = load()
    value["tasks"][0]["split"] = "dev"
    value["sha256"] = canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="target drift"):
        validate(value, inventory_path=INVENTORY.relative_to(ROOT))
