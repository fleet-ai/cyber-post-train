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
        # The fixture has only three families in each smaller partition.  A
        # feasible split-local limit therefore must allow one family to occupy
        # one third of that small partition.
        "max_group_task_version_fraction": 0.7,
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


def test_concentration_limit_applies_to_each_split_not_only_the_inventory() -> None:
    rows = []
    for index in range(6):
        lineage = {
            "application": "app",
            "environment": "environment",
            "task_family": f"family-{index}",
            "difficulty": "difficulty",
            "vulnerability_family": ["vulnerability"],
        }
        rows.append(
            {
                "task_key": f"task-{index}",
                "task_version_id": f"version-{index}-a",
                "lineage": lineage,
            }
        )
        if index == 0:
            for version in range(1, 4):
                rows.append(
                    {
                        "task_key": "task-0",
                        "task_version_id": f"version-0-{version}",
                        "lineage": {**lineage},
                    }
                )

    # The multi-version family is only 4/9 of the whole inventory, but it is
    # 4/5 of the two-family split that receives it.  A global-only check would
    # incorrectly accept this at 0.5.
    with pytest.raises(ValueError, match="per-split concentration"):
        _build(
            rows,
            ratios={"train": 1 / 3, "dev": 1 / 3, "test": 1 / 3},
            max_group_task_version_fraction=0.5,
        )


def test_task_key_cannot_change_its_reviewed_family_across_versions() -> None:
    rows = _rows()
    rows[-1]["lineage"]["task_family"] = "different-family"
    with pytest.raises(ValueError, match="inconsistent reviewed application/task_family lineage"):
        _build(rows)


def test_representable_strata_coverage_is_mandatory_and_output_is_create_once(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        _build(require_representable_labels=False)

    output = tmp_path / "sealed" / "split.json"
    value = _build()
    splits.write_once(output, value)
    assert json.loads(output.read_text()) == value
    with pytest.raises(FileExistsError, match="create-once split output"):
        splits.write_once(output, value)


def test_modest_scale_split_is_deterministic_and_keeps_all_versions_together() -> None:
    rows = []
    for index in range(90):
        lineage = {
            "application": f"app-{index % 6}",
            "environment": f"environment-{index % 5}",
            "task_family": f"family-{index:03d}",
            "difficulty": f"difficulty-{index % 4}",
            "vulnerability_family": [f"vulnerability-{index % 7}"],
        }
        rows.append(
            {
                "task_key": f"task-{index:03d}",
                "task_version_id": f"version-{index:03d}-a",
                "lineage": lineage,
            }
        )
        if index % 10 == 0:
            rows.append(
                {
                    "task_key": f"task-{index:03d}",
                    "task_version_id": f"version-{index:03d}-b",
                    "lineage": {**lineage},
                }
            )
    options = {
        "inventory_sha256": "sha256:" + "2" * 64,
        "seed": "modest-scale-fixture-v1",
        "ratios": {"train": 0.7, "dev": 0.15, "test": 0.15},
        "max_group_task_version_fraction": 0.25,
    }
    value = splits.build(rows, **options)
    assert value == splits.build(list(reversed(rows)), **options)
    assert sum(row["task_versions"] for row in value["counts"].values()) == len(rows)
    assert all(
        value["concentration"]["per_split"][split]["largest_family_task_version_fraction"]
        <= options["max_group_task_version_fraction"]
        for split in value["counts"]
    )
    splits.validate(value, rows)


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

    tampered = copy.deepcopy(value)
    tampered["policy"]["require_representable_labels"] = False
    tampered["sha256"] = splits.canonical_digest(
        {key: item for key, item in tampered.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="coverage is mandatory"):
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
        max_group_task_version_fraction=0.2,
    )
    splits.validate(value, inventory["task_versions"])
    assert {split: row["groups"] for split, row in value["counts"].items()} == {
        "train": 50,
        "dev": 17,
        "final_test": 8,
    }
