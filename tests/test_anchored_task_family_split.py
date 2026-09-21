"""Regression contracts for expanding a catalog without moving old families."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from training import task_family_split as splits

ROOT = Path(__file__).resolve().parents[1]


def _row(index: int, *, version: str = "a") -> dict:
    return {
        "task_key": f"task-{index:03d}",
        "task_version_id": f"version-{index:03d}-{version}",
        "lineage": {
            "application": "app",
            "environment": "environment",
            "task_family": f"family-{index:03d}",
            "difficulty": "medium",
            "vulnerability_family": ["web"],
        },
    }


def _rows(count: int) -> list[dict]:
    return [_row(index) for index in range(count)]


def _base_and_anchor() -> tuple[list[dict], dict, dict]:
    base = _rows(12)
    split = splits.build(
        base,
        inventory_sha256="sha256:" + "1" * 64,
        seed="anchored-base-v1",
        ratios={"train": 0.5, "dev": 0.25, "final_test": 0.25},
        max_group_task_version_fraction=0.6,
    )
    return base, split, splits.freeze_role_anchor(split, base)


def _assignment(value: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in value["tasks"]:
        prior = result.setdefault(row["group_id"], row["split"])
        assert prior == row["split"]
    return result


def _expanded_rows(base: list[dict], base_split: dict) -> list[dict]:
    roles = {row["task_key"]: row["split"] for row in base_split["tasks"]}
    train_key = next(key for key, role in roles.items() if role == "train")
    heldout_key = next(key for key, role in roles.items() if role == "dev")
    train_index = int(train_key.rsplit("-", 1)[1])
    heldout_index = int(heldout_key.rsplit("-", 1)[1])
    # New exact versions exercise inheritance for both training and held-out
    # families.  The remaining rows are truly new task families.
    return [
        *base,
        _row(train_index, version="b"),
        _row(heldout_index, version="b"),
        *[_row(index) for index in range(12, 24)],
    ]


def test_anchored_split_preserves_roles_and_assigns_only_new_families() -> None:
    base, base_split, anchor = _base_and_anchor()
    rows = _expanded_rows(base, base_split)
    # `_expanded_rows` deliberately adds 12 new families and two new exact
    # versions; all identities must remain unique.
    assert len({(row["task_key"], row["task_version_id"]) for row in rows}) == len(rows)
    value = splits.build_anchored(
        rows,
        inventory_sha256="sha256:" + "2" * 64,
        role_anchor=anchor,
        seed="anchored-extension-v1",
        ratios={"train": 0.5, "dev": 0.25, "final_test": 0.25},
        max_group_task_version_fraction=0.6,
    )
    assert value == splits.build_anchored(
        list(reversed(rows)),
        inventory_sha256="sha256:" + "2" * 64,
        role_anchor=anchor,
        seed="anchored-extension-v1",
        ratios={"train": 0.5, "dev": 0.25, "final_test": 0.25},
        max_group_task_version_fraction=0.6,
    )
    original = _assignment(base_split)
    actual = _assignment(value)
    assert {group: actual[group] for group in original} == original
    by_identity = {(row["task_key"], row["task_version_id"]): row for row in value["tasks"]}
    for original_row in base_split["tasks"]:
        for version in ("a", "b"):
            identity = (
                original_row["task_key"],
                original_row["task_version_id"].replace("-a", f"-{version}"),
            )
            if identity in by_identity:
                assert by_identity[identity]["split"] == original_row["split"]
    assert value["anchor_audit"]["inherited_role_count"] == 12
    assert value["anchor_audit"]["new_group_count"] == 12
    assert value["leakage_checks"]["immutable_inherited_role_drift"] == 0
    splits.validate(value, rows)


def test_anchor_rejects_historical_task_key_family_drift() -> None:
    base, base_split, anchor = _base_and_anchor()
    rows = _expanded_rows(base, base_split)
    changed = copy.deepcopy(rows)
    changed_key = changed[0]["task_key"]
    for row in changed:
        if row["task_key"] == changed_key:
            row["lineage"]["task_family"] = "renamed-historical-family"
    with pytest.raises(ValueError, match="historical task_key"):
        splits.build_anchored(
            changed,
            inventory_sha256="sha256:" + "3" * 64,
            role_anchor=anchor,
            seed="anchored-extension-v1",
            ratios={"train": 0.5, "dev": 0.25, "final_test": 0.25},
            max_group_task_version_fraction=0.6,
        )


def test_anchor_cannot_be_used_with_infeasible_target_ratios() -> None:
    base, _base_split, anchor = _base_and_anchor()
    # The frozen base has three development families, whereas a 12-family
    # 80/10/10 target permits only one.  Fail before silently moving one.
    with pytest.raises(ValueError, match="immutable anchored roles exceed"):
        splits.build_anchored(
            base,
            inventory_sha256="sha256:" + "4" * 64,
            role_anchor=anchor,
            seed="anchored-infeasible-v1",
            ratios={"train": 0.8, "dev": 0.1, "final_test": 0.1},
            max_group_task_version_fraction=0.6,
        )


def test_freezing_an_anchored_split_retains_absent_heldout_roles() -> None:
    base, base_split, anchor = _base_and_anchor()
    rows = _expanded_rows(base, base_split)
    first = splits.build_anchored(
        rows,
        inventory_sha256="sha256:" + "5" * 64,
        role_anchor=anchor,
        seed="anchored-extension-v1",
        ratios={"train": 0.5, "dev": 0.25, "final_test": 0.25},
        max_group_task_version_fraction=0.6,
    )
    next_anchor = splits.freeze_role_anchor(first, rows)
    dev_task = next(row for row in first["tasks"] if row["split"] == "dev")
    dev_group = dev_task["group_id"]
    without_family = [row for row in rows if row["task_key"] != dev_task["task_key"]]
    remaining_roles = {
        role: sum(
            row["split"] == role and row["group_id"] != dev_group for row in next_anchor["roles"]
        )
        for role in ("train", "dev", "final_test")
    }
    remaining_total = sum(remaining_roles.values())
    middle = splits.build_anchored(
        without_family,
        inventory_sha256="sha256:" + "6" * 64,
        role_anchor=next_anchor,
        seed="anchored-extension-v2",
        ratios={role: count / remaining_total for role, count in remaining_roles.items()},
        max_group_task_version_fraction=0.6,
    )
    final_anchor = splits.freeze_role_anchor(middle, without_family)
    restored = splits.build_anchored(
        rows,
        inventory_sha256="sha256:" + "7" * 64,
        role_anchor=final_anchor,
        seed="anchored-extension-v3",
        ratios={"train": 0.5, "dev": 0.25, "final_test": 0.25},
        max_group_task_version_fraction=0.6,
    )
    restored_role = next(
        row["split"] for row in restored["tasks"] if row["task_key"] == dev_task["task_key"]
    )
    assert restored_role == "dev"


def test_real_locked_study_v2_converts_without_reassigning_roles() -> None:
    split_path = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
    legacy = json.loads(split_path.read_text())
    anchor = splits.freeze_study_v2_role_anchor(
        legacy,
        legacy_inventory_path=Path(legacy["inventory"]["path"]),
    )
    assert anchor["source"]["split_sha256"] == legacy["sha256"]
    assert len(anchor["roles"]) == 75
    assert len(anchor["heldout_group_ids"]) == 25
    assert {
        role: sum(row["split"] == role for row in anchor["roles"])
        for role in ("train", "dev", "final_test")
    } == {"train": 50, "dev": 17, "final_test": 8}
