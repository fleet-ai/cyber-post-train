"""Build and validate the refreshed, representative Fleet study split.

The split is deterministic, exact-version bound, and task-family safe.  It
balances reviewed metadata without pretending that one-off taxonomy labels can
be represented in every partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "cyber_representative_study_split_v2"
SELECTION_SCHEMA = "cyber_eval_task_selection_v2"
TRAINING_SCHEMA = "cyber_task_split_v3"
SEED = "fleet-blackbox-current-high-quality-20260914-representative-v2"
SPLITS = ("train", "dev", "final_test")
TARGET_COUNTS = {"train": 50, "dev": 17, "final_test": 8}
DIMENSIONS = ("application", "environment", "difficulty", "vulnerability_family")
DIMENSION_WEIGHTS = {
    "application": 3.0,
    "environment": 3.0,
    "difficulty": 2.0,
    "vulnerability_family": 1.0,
}
SPLIT_UNIT = "reviewed application and task family; one selected exact version per family"
GENERALIZATION_SCOPE = "task-family-held-out; applications are represented in every split"
TINY_STRATA = {
    "count_1": "record placement; do not claim cross-split representation",
    "count_2_to_4": "balance proportionally when compatible with stronger strata",
    "count_5_or_more": "require at least one example in every split",
}


def canonical_digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, (SEED, *parts))).encode()).hexdigest()


def _group_id(row: dict[str, Any]) -> str:
    lineage = row["lineage"]
    values = [lineage["application"], lineage["task_family"]]
    if any(not isinstance(value, str) or not value or value == "unknown" for value in values):
        raise ValueError("representative split requires reviewed application and task family")
    return canonical_digest(values)


def _features(row: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    lineage = row["lineage"]
    result = {name: (lineage[name],) for name in ("application", "environment", "difficulty")}
    labels = lineage["vulnerability_family"]
    if (
        not isinstance(labels, list)
        or not labels
        or any(not isinstance(x, str) or not x for x in labels)
    ):
        raise ValueError("representative split requires reviewed vulnerability-family labels")
    result["vulnerability_family"] = tuple(sorted(set(labels)))
    return result


def _population(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    counts = {dimension: Counter() for dimension in DIMENSIONS}
    for row in rows:
        for dimension, labels in _features(row).items():
            counts[dimension].update(labels)
    return counts


def _actual(
    rows: list[dict[str, Any]], assignments: dict[str, str]
) -> dict[str, dict[str, Counter[str]]]:
    counts = {dimension: {split: Counter() for split in SPLITS} for dimension in DIMENSIONS}
    for row in rows:
        split = assignments.get(row["task_version_id"])
        if split is None:
            continue
        for dimension, labels in _features(row).items():
            counts[dimension][split].update(labels)
    return counts


def _score(rows: list[dict[str, Any]], assignments: dict[str, str]) -> float:
    population = _population(rows)
    actual = _actual(rows, assignments)
    total = 0.0
    for dimension in DIMENSIONS:
        dimension_error = 0.0
        for label, count in population[dimension].items():
            for split in SPLITS:
                target = count * TARGET_COUNTS[split] / len(rows)
                observed = actual[dimension][split][label]
                dimension_error += (observed - target) ** 2 / max(target, 0.5)
                if count >= 5 and observed == 0:
                    dimension_error += 1_000.0
        total += DIMENSION_WEIGHTS[dimension] * dimension_error / len(population[dimension])
    return total


def _initial_assignment(rows: list[dict[str, Any]], trial: int) -> dict[str, str]:
    population = _population(rows)

    def rarity(row: dict[str, Any]) -> float:
        return sum(
            1 / population[dimension][label]
            for dimension, labels in _features(row).items()
            for label in labels
        )

    order = sorted(
        rows,
        key=lambda row: (-rarity(row), _rank("order", trial, row["task_version_id"])),
    )
    assignment: dict[str, str] = {}
    used = Counter()
    for row in order:
        candidates = [split for split in SPLITS if used[split] < TARGET_COUNTS[split]]
        split = min(
            candidates,
            key=lambda candidate: (
                _score(rows, assignment | {row["task_version_id"]: candidate}),
                _rank("choice", trial, row["task_version_id"], candidate),
            ),
        )
        assignment[row["task_version_id"]] = split
        used[split] += 1
    return assignment


def _improve(rows: list[dict[str, Any]], assignment: dict[str, str]) -> dict[str, str]:
    assignment = dict(assignment)
    current = _score(rows, assignment)
    ordered = sorted(rows, key=lambda row: row["task_version_id"])
    while True:
        best: tuple[float, str, str] | None = None
        for index, left in enumerate(ordered):
            left_id = left["task_version_id"]
            for right in ordered[index + 1 :]:
                right_id = right["task_version_id"]
                if assignment[left_id] == assignment[right_id]:
                    continue
                candidate = dict(assignment)
                candidate[left_id], candidate[right_id] = candidate[right_id], candidate[left_id]
                value = _score(rows, candidate)
                choice = (value, left_id, right_id)
                if value + 1e-12 < current and (best is None or choice < best):
                    best = choice
        if best is None:
            return assignment
        current, left_id, right_id = best
        assignment[left_id], assignment[right_id] = assignment[right_id], assignment[left_id]


def assign(rows: list[dict[str, Any]]) -> dict[str, str]:
    rows = sorted(rows, key=lambda row: (row["task_key"], row["task_version_id"]))
    if len(rows) != sum(TARGET_COUNTS.values()):
        raise ValueError("v2 split is bound to the reviewed 75-task inventory")
    identities = [(row["task_key"], row["task_version_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("task/version identities must be unique")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[_group_id(row)].append(row["task_version_id"])
    if any(len(members) != 1 for members in groups.values()):
        raise ValueError(
            "v2 exact-size optimizer requires one selected version per reviewed family"
        )

    candidates = [_initial_assignment(rows, trial) for trial in range(96)]
    best = min(
        candidates, key=lambda candidate: (_score(rows, candidate), canonical_digest(candidate))
    )
    return _improve(rows, best)


def _selection(
    rows: list[dict[str, Any]], assignments: dict[str, str], split: str
) -> dict[str, Any]:
    tasks = [
        {
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
            "group_id": _group_id(row),
            "split": split,
        }
        for row in rows
        if assignments[row["task_version_id"]] == split
    ]
    tasks.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    result = {"schema": SELECTION_SCHEMA, "selection_role": split, "tasks": tasks}
    result["sha256"] = canonical_digest(result)
    return result


def _training_selection(rows: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    tasks = _selection(rows, assignments, "train")["tasks"]
    result = {"schema": TRAINING_SCHEMA, "selection_role": "train", "tasks": tasks}
    result["sha256"] = canonical_digest(result)
    return result


def _representation(rows: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    population = _population(rows)
    actual = _actual(rows, assignments)
    result: dict[str, Any] = {}
    for dimension in DIMENSIONS:
        result[dimension] = {
            "population_counts": dict(sorted(population[dimension].items())),
            "split_counts": {
                split: dict(sorted(actual[dimension][split].items())) for split in SPLITS
            },
            "max_absolute_share_gap": {
                split: max(
                    abs(actual[dimension][split][label] / TARGET_COUNTS[split] - count / len(rows))
                    for label, count in population[dimension].items()
                )
                for split in SPLITS
            },
        }
    return result


def build(inventory_path: Path) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text())
    rows = sorted(
        inventory["task_versions"], key=lambda row: (row["task_key"], row["task_version_id"])
    )
    assignments = assign(rows)
    selections = {split: _selection(rows, assignments, split) for split in SPLITS}
    tasks = [task for split in SPLITS for task in selections[split]["tasks"]]
    value = {
        "schema": SCHEMA,
        "inventory": {
            "path": inventory_path.as_posix(),
            "file_sha256": file_sha256(inventory_path),
            "logical_sha256": inventory["sha256"],
            "task_versions": len(rows),
        },
        "seed": SEED,
        "split_unit": SPLIT_UNIT,
        "generalization_scope": GENERALIZATION_SCOPE,
        "policy": {
            "target_counts": TARGET_COUNTS,
            "balanced_dimensions": list(DIMENSIONS),
            "tiny_strata": TINY_STRATA,
            "optimizer": "deterministic rarity-first assignment plus improving pair swaps",
        },
        "tasks": sorted(tasks, key=lambda row: (row["task_key"], row["task_version_id"])),
        "training_split": _training_selection(rows, assignments),
        "evaluation": {"dev": selections["dev"], "final_test": selections["final_test"]},
        "counts": {
            split: {
                "groups": len(selections[split]["tasks"]),
                "task_versions": len(selections[split]["tasks"]),
            }
            for split in SPLITS
        },
        "representation": _representation(rows, assignments),
        "leakage_checks": {
            "exact_identity_overlap": 0,
            "reviewed_family_overlap": 0,
            "all_inventory_versions_assigned_once": True,
        },
        "limitations": [
            "This is task-family-held-out, not application-held-out.",
            "One-off and very small vulnerability strata cannot appear in every split.",
            "Reviewed taxonomy labels overlap and are not independent vulnerability classes.",
        ],
    }
    value["sha256"] = canonical_digest(value)
    return validate(value, inventory_path=inventory_path)


def validate(
    value: dict[str, Any],
    *,
    inventory_path: Path,
    inventory_display_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a v2 split against bytes at ``inventory_path``.

    ``inventory_display_path`` is needed only by source-only tools that stage
    the frozen inventory outside the repository.  The historic split binds the
    original reviewed path string as well as the file bytes, so callers must
    supply that exact declared path explicitly rather than silently rewriting
    historical provenance.
    """
    inventory_display_path = (
        inventory_path if inventory_display_path is None else inventory_display_path
    )
    if value.get("schema") != SCHEMA or value.get("seed") != SEED:
        raise ValueError("unsupported representative split plan")
    if value["inventory"] != {
        "path": inventory_display_path.as_posix(),
        "file_sha256": file_sha256(inventory_path),
        "logical_sha256": json.loads(inventory_path.read_text())["sha256"],
        "task_versions": 75,
    }:
        raise ValueError("split inventory binding drift")
    if (
        value.get("split_unit") != SPLIT_UNIT
        or value.get("generalization_scope") != GENERALIZATION_SCOPE
        or value.get("policy")
        != {
            "target_counts": TARGET_COUNTS,
            "balanced_dimensions": list(DIMENSIONS),
            "tiny_strata": TINY_STRATA,
            "optimizer": "deterministic rarity-first assignment plus improving pair swaps",
        }
    ):
        raise ValueError("representative split policy drift")
    inventory_rows = json.loads(inventory_path.read_text())["task_versions"]
    inventory_ids = {(row["task_key"], row["task_version_id"]) for row in inventory_rows}
    task_ids = [(row["task_key"], row["task_version_id"]) for row in value["tasks"]]
    if len(task_ids) != 75 or set(task_ids) != inventory_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("split does not assign the exact inventory once")
    assignments = {row["task_version_id"]: row["split"] for row in value["tasks"]}
    if set(assignments.values()) != set(SPLITS):
        raise ValueError("unexpected partition name")
    if Counter(assignments.values()) != Counter(TARGET_COUNTS):
        raise ValueError("v2 50/17/8 target drift")
    groups: dict[str, set[str]] = defaultdict(set)
    by_identity = {(row["task_key"], row["task_version_id"]): row for row in inventory_rows}
    for task in value["tasks"]:
        source = by_identity[(task["task_key"], task["task_version_id"])]
        if task["group_id"] != _group_id(source):
            raise ValueError("reviewed family binding drift")
        groups[task["group_id"]].add(task["split"])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("reviewed task family crosses partitions")
    population = _population(inventory_rows)
    actual = _actual(inventory_rows, assignments)
    for dimension in DIMENSIONS:
        for label, count in population[dimension].items():
            if count >= 5 and any(actual[dimension][split][label] == 0 for split in SPLITS):
                raise ValueError(f"representable stratum missing from a split: {dimension}/{label}")
    expected = build_sections(inventory_rows, assignments)
    for name in ("training_split", "evaluation", "counts", "representation", "leakage_checks"):
        if value[name] != expected[name]:
            raise ValueError(f"derived split section drift: {name}")
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != canonical_digest(unsigned):
        raise ValueError("representative split digest mismatch")
    return value


def build_sections(rows: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    selections = {split: _selection(rows, assignments, split) for split in SPLITS}
    return {
        "training_split": _training_selection(rows, assignments),
        "evaluation": {"dev": selections["dev"], "final_test": selections["final_test"]},
        "counts": {
            split: {
                "groups": len(selections[split]["tasks"]),
                "task_versions": len(selections[split]["tasks"]),
            }
            for split in SPLITS
        },
        "representation": _representation(rows, assignments),
        "leakage_checks": {
            "exact_identity_overlap": 0,
            "reviewed_family_overlap": 0,
            "all_inventory_versions_assigned_once": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        validate(json.loads(args.output.read_text()), inventory_path=args.inventory)
        return
    value = build(args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
