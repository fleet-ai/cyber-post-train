from copy import deepcopy

import pytest

from training.splits import apply_splits, assign_split, split_key


def example(version=1, *, app="fira", family="demo", registry="cyber/task-graphs/demo"):
    return {
        "lineage": {
            "application": app,
            "task_family": family,
            "lineage_key": f"registry:{registry}@{version}",
        }
    }


def test_versions_sessions_and_prompt_changes_cannot_cross_splits():
    records = [example(v) for v in range(1, 101)]
    for i, row in enumerate(records):
        row.update(record_id=f"session-{i}", prompt_sha256=str(i))
    apply_splits(records)
    assert len({r["split_unit"] for r in records}) == 1
    assert len({r["split"] for r in records}) == 1


def test_fleet_versions_group_without_registry_metadata():
    records = [example() for _ in range(100)]
    for i, row in enumerate(records):
        row["lineage"]["lineage_key"] = f"fleet-task-version:{i}"
    apply_splits(records)
    assert len({r["split_unit"] for r in records}) == 1


def test_deterministic_independent_of_order_and_subset():
    records = [example(family=f"task-{i}", registry=f"cyber/task-graphs/{i}") for i in range(100)]
    reverse = deepcopy(records[::-1])
    subset = deepcopy(records[::3])
    apply_splits(iter(records))
    apply_splits(reverse)
    apply_splits(subset)
    mapping = {r["split_unit"]: r["split"] for r in records}
    assert all(mapping[r["split_unit"]] == r["split"] for r in reverse + subset)
    assert set(mapping.values()) == {"train", "dev", "test"}


def test_same_registry_family_rename_fails_before_any_mutation():
    records = [example(1), example(2, family="renamed")]
    before = deepcopy(records)
    with pytest.raises(ValueError, match="conflicting"):
        apply_splits(records)
    assert records == before


@pytest.mark.parametrize("field", ["application", "task_family", "lineage_key"])
@pytest.mark.parametrize("value", [None, "", 42])
def test_missing_identity_is_rejected(field, value):
    row = example()
    row["lineage"][field] = value
    with pytest.raises(ValueError):
        apply_splits([row])
    assert "split" not in row


def test_split_key_has_no_delimiter_collision():
    assert split_key(example(app="a|b", family="c")) != split_key(example(app="a", family="b|c"))


@pytest.mark.parametrize(
    "train,dev",
    [(0, 0.1), (-1, 0), (0.8, 0.2), (0.8, -0.1), (float("nan"), 0), (0.8, float("inf"))],
)
def test_invalid_ratios(train, dev):
    with pytest.raises(ValueError, match="ratios"):
        assign_split("unit", train=train, dev=dev)
