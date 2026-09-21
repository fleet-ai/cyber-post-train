import copy
import json
from pathlib import Path

import pytest

from training import task_family_split as splits

ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict]:
    rows = []
    for index in range(15):
        lineage = {
            "application": f"app-{index % 3}",
            "environment": f"environment-{index % 3}",
            "task_family": f"family-{index:02d}",
            "difficulty": f"difficulty-{index % 3}",
            "vulnerability_family": [f"vulnerability-{index % 3}"],
        }
        rows.append(
            {
                "task_key": f"task-{index:02d}",
                "task_version_id": f"version-{index:02d}-a",
                "lineage": lineage,
            }
        )
    # A second exact version must remain with the same reviewed task family.
    rows.append(
        {
            "task_key": "task-00",
            "task_version_id": "version-00-b",
            "lineage": {**rows[0]["lineage"]},
        }
    )
    return rows


def _build(rows: list[dict] | None = None, **overrides) -> dict:
    options = {
        "inventory_sha256": "sha256:" + "1" * 64,
        "seed": "scalable-split-fixture-v1",
        "ratios": {"train": 0.6, "dev": 0.2, "test": 0.2},
        "max_group_task_version_fraction": 0.2,
    }
    options.update(overrides)
    return splits.build(
        _rows() if rows is None else rows,
        **options,
    )


def test_grouped_split_is_deterministic_safe_and_sealed() -> None:
    value = _build()
    assert value == _build(list(reversed(_rows())))
    assert value["sha256"] == splits.canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    assert {split: value["counts"][split]["groups"] for split in value["counts"]} == {
        "dev": 3,
        "test": 3,
        "train": 9,
    }
    assert sum(value["counts"][split]["task_versions"] for split in value["counts"]) == 16
    assert value["leakage_checks"] == {
        "exact_identity_overlap": 0,
        "reviewed_family_overlap": 0,
        "all_inventory_versions_assigned_once": True,
    }
    family = {row["task_version_id"]: row["split"] for row in value["tasks"]}
    assert family["version-00-a"] == family["version-00-b"]
    assert all(
        set(row) == {"task_key", "task_version_id", "group_id", "split"} for row in value["tasks"]
    )
    assert value["concentration"]["groups_with_multiple_versions"] == 1
    assert value["concentration"]["largest_family_task_versions"] == 2
    splits.validate(value, _rows())


def test_representable_metadata_is_present_across_each_partition() -> None:
    value = _build()
    for dimension, representation in value["representation"].items():
        for label, count in representation["population_group_counts"].items():
            if count >= 3:
                assert all(
                    representation["split_group_counts"][split].get(label, 0) > 0
                    for split in ("train", "dev", "test")
                ), (dimension, label)


def test_missing_or_unknown_reviewed_metadata_fails_closed() -> None:
    rows = _rows()
    del rows[0]["lineage"]["difficulty"]
    with pytest.raises(ValueError, match="difficulty"):
        _build(rows)

    rows = _rows()
    rows[0]["lineage"]["task_family"] = "unknown"
    with pytest.raises(ValueError, match="task_family"):
        _build(rows)


def test_explicit_concentration_limit_is_a_real_gate() -> None:
    with pytest.raises(ValueError, match="concentration"):
        _build(max_group_task_version_fraction=0.1)


def test_validator_rejects_any_assignment_or_policy_drift() -> None:
    value = _build()
    tampered = copy.deepcopy(value)
    tampered["tasks"][0]["split"] = "dev"
    tampered["sha256"] = splits.canonical_digest(
        {key: item for key, item in tampered.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="drift"):
        splits.validate(tampered, _rows())

    tampered = copy.deepcopy(value)
    tampered["policy"]["ratios"]["train"] = 0.0
    tampered["sha256"] = splits.canonical_digest(
        {key: item for key, item in tampered.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="ratio"):
        splits.validate(tampered, _rows())


def test_current_reviewed_inventory_can_use_the_parameterized_contract() -> None:
    inventory = json.loads(
        (ROOT / "configs/data/fleet-blackbox-current-high-quality-20260914-v1.json").read_text()
    )
    value = splits.build(
        inventory["task_versions"],
        inventory_sha256=inventory["sha256"],
        seed="fleet-blackbox-current-high-quality-20260914-scalable-v1",
        ratios={"train": 50 / 75, "dev": 17 / 75, "final_test": 8 / 75},
        max_group_task_version_fraction=0.1,
    )
    splits.validate(value, inventory["task_versions"])
    assert {split: row["groups"] for split, row in value["counts"].items()} == {
        "train": 50,
        "dev": 17,
        "final_test": 8,
    }
