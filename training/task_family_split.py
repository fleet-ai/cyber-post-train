"""Build a deterministic, metadata-only, task-family-safe study split.

This is deliberately separate from :mod:`study_split_v2`, whose published
75-task artifact has fixed counts and a historical contract.  The helper here
accepts any reviewed inventory, keeps all versions of a task family together,
and produces a sealed split plus a concentration audit.  It never reads task
prompts, session transcripts, scores, or model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "cyber_parameterized_task_family_split_v1"
ROLE_ANCHOR_SCHEMA = "cyber_task_family_role_anchor_v1"
ANCHORED_SCHEMA = "cyber_anchored_task_family_split_v1"
SUPPORTED_SCHEMAS = frozenset({SCHEMA, ANCHORED_SCHEMA})
DEFAULT_DIMENSIONS = ("application", "environment", "difficulty", "vulnerability_family")
SPLIT_UNIT = "reviewed application and task family; all exact versions stay together"
FORBIDDEN_OUTPUT_TERMS = {
    "prompt",
    "trace",
    "answer",
    "flag",
    "credential",
    "score",
    "session_id",
}


def canonical_digest(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _rank(seed: str, *parts: object) -> str:
    return hashlib.sha256(":".join((seed, *(str(part) for part in parts))).encode()).hexdigest()


def _sealed(value: dict[str, Any]) -> bool:
    return value.get("sha256") == canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    )


def is_supported_schema(value: object) -> bool:
    """Return whether a split uses one of this module's sealed contracts."""
    return isinstance(value, dict) and value.get("schema") in SUPPORTED_SCHEMAS


def requires_role_anchor(value: object) -> bool:
    """Return whether validation needs the independently sealed parent anchor."""
    return isinstance(value, dict) and value.get("schema") == ANCHORED_SCHEMA


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != len("sha256:") + 64
    ):
        raise ValueError(f"exact {label} digest is required")
    try:
        int(value.removeprefix("sha256:"), 16)
    except ValueError as error:
        raise ValueError(f"exact {label} digest is required") from error
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value == "unknown":
        raise ValueError(f"reviewed {label} metadata is required")
    return value


def _labels(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"reviewed {label} metadata is required")
    labels = tuple(sorted({_string(item, label) for item in value}))
    if not labels:
        raise ValueError(f"reviewed {label} metadata is required")
    return labels


def _task_identity(row: dict[str, Any]) -> tuple[str, str]:
    return _string(row.get("task_key"), "task_key"), _string(
        row.get("task_version_id"), "task_version_id"
    )


def _metadata(row: dict[str, Any], dimensions: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    lineage = row.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("reviewed lineage metadata is required")
    values: dict[str, tuple[str, ...]] = {}
    for dimension in dimensions:
        if dimension == "vulnerability_family":
            values[dimension] = _labels(lineage.get(dimension), dimension)
        else:
            values[dimension] = (_string(lineage.get(dimension), dimension),)
    return values


def _group_id(application: str, task_family: str) -> str:
    return canonical_digest([application, task_family])


def _group_rows(rows: list[dict[str, Any]], dimensions: tuple[str, ...]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("reviewed inventory must not be empty")
    seen: set[tuple[str, str]] = set()
    # A task key denotes one logical task across exact versions.  Letting it
    # silently change application or family would make the held-out boundary
    # ambiguous, so reject that inventory rather than guessing a grouping.
    task_key_lineages: dict[str, tuple[str, str]] = {}
    grouped: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("reviewed inventory contains a non-object row")
        identity = _task_identity(raw)
        if identity in seen:
            raise ValueError("reviewed inventory duplicates a task/version identity")
        seen.add(identity)
        lineage = raw.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("reviewed lineage metadata is required")
        application = _string(lineage.get("application"), "application")
        task_family = _string(lineage.get("task_family"), "task_family")
        task_key_lineage = (application, task_family)
        prior_lineage = task_key_lineages.setdefault(identity[0], task_key_lineage)
        if prior_lineage != task_key_lineage:
            raise ValueError(
                "one task_key has inconsistent reviewed application/task_family "
                "lineage across versions"
            )
        group_id = _group_id(application, task_family)
        features = _metadata(raw, dimensions)
        group = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "application": application,
                "task_family": task_family,
                "task_rows": [],
                "features": {dimension: set() for dimension in dimensions},
            },
        )
        if group["application"] != application or group["task_family"] != task_family:
            raise ValueError("task-family grouping collision")
        group["task_rows"].append({"task_key": identity[0], "task_version_id": identity[1]})
        for dimension, labels in features.items():
            group["features"][dimension].update(labels)
    result = []
    for group in grouped.values():
        group["task_rows"].sort(key=lambda row: (row["task_key"], row["task_version_id"]))
        group["features"] = {
            dimension: tuple(sorted(labels)) for dimension, labels in group["features"].items()
        }
        result.append(group)
    return sorted(result, key=lambda group: group["group_id"])


def _normalise_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if not isinstance(ratios, dict) or len(ratios) < 2:
        raise ValueError("at least two named split ratios are required")
    result: dict[str, float] = {}
    for split, ratio in ratios.items():
        if not isinstance(split, str) or not split:
            raise ValueError("split names must be nonempty strings")
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(ratio)
        ):
            raise ValueError("split ratios must be finite numbers")
        if ratio <= 0:
            raise ValueError("every declared split ratio must be positive")
        result[split] = float(ratio)
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to one")
    return dict(sorted(result.items()))


def _target_group_counts(group_count: int, ratios: dict[str, float]) -> dict[str, int]:
    splits = tuple(ratios)
    if group_count < len(splits):
        raise ValueError("fewer task families than declared splits")
    raw = {split: group_count * ratio for split, ratio in ratios.items()}
    targets = {split: math.floor(value) for split, value in raw.items()}
    for split in splits:
        if targets[split] == 0:
            targets[split] = 1
    while sum(targets.values()) > group_count:
        donor = max(splits, key=lambda split: (targets[split], raw[split], split))
        if targets[donor] <= 1:
            raise ValueError("cannot allocate at least one family to every split")
        targets[donor] -= 1
    while sum(targets.values()) < group_count:
        recipient = max(
            splits,
            key=lambda split: (raw[split] - targets[split], raw[split], split),
        )
        targets[recipient] += 1
    return targets


def _population(
    groups: list[dict[str, Any]], dimensions: tuple[str, ...]
) -> dict[str, Counter[str]]:
    result = {dimension: Counter() for dimension in dimensions}
    for group in groups:
        for dimension in dimensions:
            result[dimension].update(group["features"][dimension])
    return result


def _assignment_score(
    groups: list[dict[str, Any]],
    assignment: dict[str, str],
    *,
    targets: dict[str, int],
    dimensions: tuple[str, ...],
) -> float:
    population = _population(groups, dimensions)
    observed = {dimension: {split: Counter() for split in targets} for dimension in dimensions}
    for group in groups:
        split = assignment.get(group["group_id"])
        if split is None:
            continue
        for dimension in dimensions:
            observed[dimension][split].update(group["features"][dimension])
    total_groups = len(groups)
    total = 0.0
    for dimension in dimensions:
        for label, count in population[dimension].items():
            for split, split_groups in targets.items():
                target = count * split_groups / total_groups
                total += (observed[dimension][split][label] - target) ** 2 / max(target, 0.5)
                if count >= len(targets) and observed[dimension][split][label] == 0:
                    # Coverage is a hard scientific requirement when a label
                    # occurs in enough distinct task families to plausibly
                    # span every partition.  The later repair is a guard,
                    # not the primary way we obtain this property.
                    total += 1_000.0
    return total


def _assign(
    groups: list[dict[str, Any]],
    *,
    seed: str,
    targets: dict[str, int],
    dimensions: tuple[str, ...],
) -> dict[str, str]:
    population = _population(groups, dimensions)

    def rarity(group: dict[str, Any]) -> float:
        return sum(
            1 / population[dimension][label]
            for dimension in dimensions
            for label in group["features"][dimension]
        )

    order = sorted(
        groups,
        key=lambda group: (-rarity(group), _rank(seed, "order", group["group_id"])),
    )
    assignment: dict[str, str] = {}
    used = Counter()
    for group in order:
        candidates = [split for split in targets if used[split] < targets[split]]
        if not candidates:
            raise ValueError("task-family allocation exhausted unexpectedly")
        split = min(
            candidates,
            key=lambda candidate: (
                _assignment_score(
                    groups,
                    assignment | {group["group_id"]: candidate},
                    targets=targets,
                    dimensions=dimensions,
                ),
                _rank(seed, "choice", group["group_id"], candidate),
            ),
        )
        assignment[group["group_id"]] = split
        used[split] += 1
    return assignment


def _representation(
    groups: list[dict[str, Any]],
    assignment: dict[str, str],
    *,
    targets: dict[str, int],
    dimensions: tuple[str, ...],
) -> dict[str, Any]:
    population = _population(groups, dimensions)
    observed = {dimension: {split: Counter() for split in targets} for dimension in dimensions}
    for group in groups:
        for dimension in dimensions:
            observed[dimension][assignment[group["group_id"]]].update(group["features"][dimension])
    total_groups = len(groups)
    result: dict[str, Any] = {}
    for dimension in dimensions:
        result[dimension] = {
            "population_group_counts": dict(sorted(population[dimension].items())),
            "split_group_counts": {
                split: dict(sorted(observed[dimension][split].items())) for split in targets
            },
            "max_absolute_share_gap": {
                split: max(
                    abs(observed[dimension][split][label] / targets[split] - count / total_groups)
                    for label, count in population[dimension].items()
                )
                for split in targets
            },
            "labels_not_representable_in_every_split": sorted(
                label for label, count in population[dimension].items() if count < len(targets)
            ),
        }
    return result


def _require_representable_labels(
    representation: dict[str, Any], *, splits: tuple[str, ...]
) -> None:
    for dimension, value in representation.items():
        for label, count in value["population_group_counts"].items():
            if count >= len(splits) and any(
                value["split_group_counts"][split].get(label, 0) == 0 for split in splits
            ):
                raise ValueError(
                    f"representable reviewed stratum missing from a split: {dimension}/{label}"
                )


def _representation_deficit(
    representation: dict[str, Any], *, splits: tuple[str, ...]
) -> tuple[tuple[str, str, str], ...]:
    """Return only labels that have enough *families* to span every split."""
    missing = []
    for dimension, value in representation.items():
        for label, count in value["population_group_counts"].items():
            if count < len(splits):
                continue
            for split in splits:
                if value["split_group_counts"][split].get(label, 0) == 0:
                    missing.append((dimension, label, split))
    return tuple(sorted(missing))


def _repair_representable_labels(
    groups: list[dict[str, Any]],
    assignment: dict[str, str],
    *,
    seed: str,
    targets: dict[str, int],
    dimensions: tuple[str, ...],
    locked_group_ids: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Use bounded, deterministic swaps to repair a greedy balance allocation.

    The primary assignment is linear-ish and works for large inventories.  A
    rare-stratum repair only runs when the declared policy requires every
    representable label in every split.  It never silently relaxes that
    contract: an unrepaired label rejects the split rather than leaking a
    family or claiming false coverage.
    """
    assignment = dict(assignment)
    splits = tuple(targets)
    current = _representation(groups, assignment, targets=targets, dimensions=dimensions)
    deficits = _representation_deficit(current, splits=splits)
    attempts = 0
    # This bound is intentionally explicit.  A much larger inventory that
    # cannot repair within it must be reviewed, not assigned optimistically.
    limit = max(256, len(groups) * len(splits) * len(dimensions) * 8)
    while deficits:
        best: tuple[int, float, str, str, dict[str, str]] | None = None
        for dimension, label, needed_split in deficits:
            donors = [
                group
                for group in groups
                if group["group_id"] not in locked_group_ids
                and assignment[group["group_id"]] != needed_split
                and label in group["features"][dimension]
            ]
            receivers = [
                group
                for group in groups
                if group["group_id"] not in locked_group_ids
                and assignment[group["group_id"]] == needed_split
            ]
            for donor in sorted(donors, key=lambda group: _rank(seed, "donor", group["group_id"])):
                donor_split = assignment[donor["group_id"]]
                for receiver in sorted(
                    receivers, key=lambda group: _rank(seed, "receiver", group["group_id"])
                ):
                    candidate = dict(assignment)
                    candidate[donor["group_id"]] = needed_split
                    candidate[receiver["group_id"]] = donor_split
                    representation = _representation(
                        groups, candidate, targets=targets, dimensions=dimensions
                    )
                    remaining = _representation_deficit(representation, splits=splits)
                    attempts += 1
                    if attempts > limit:
                        raise ValueError(
                            "representative split repair exceeded its explicit review bound"
                        )
                    if len(remaining) >= len(deficits):
                        continue
                    candidate_score = _assignment_score(
                        groups, candidate, targets=targets, dimensions=dimensions
                    )
                    choice = (
                        len(remaining),
                        candidate_score,
                        donor["group_id"],
                        receiver["group_id"],
                        candidate,
                    )
                    if best is None or choice[:4] < best[:4]:
                        best = choice
        if best is None:
            dimension, label, _ = deficits[0]
            raise ValueError(
                f"representable reviewed stratum cannot be placed without breaking another: "
                f"{dimension}/{label}"
            )
        assignment = best[4]
        current = _representation(groups, assignment, targets=targets, dimensions=dimensions)
        deficits = _representation_deficit(current, splits=splits)
    return assignment


def _concentration(
    groups: list[dict[str, Any]], assignment: dict[str, str], targets: dict[str, int]
) -> dict[str, Any]:
    task_total = sum(len(group["task_rows"]) for group in groups)
    largest = max(groups, key=lambda group: (len(group["task_rows"]), group["group_id"]))
    per_split = {}
    for split in targets:
        selected = [group for group in groups if assignment[group["group_id"]] == split]
        task_versions = sum(len(group["task_rows"]) for group in selected)
        largest_count = max(len(group["task_rows"]) for group in selected)
        per_split[split] = {
            "groups": len(selected),
            "task_versions": task_versions,
            "largest_family_task_versions": largest_count,
            "largest_family_task_version_fraction": largest_count / task_versions,
        }
    return {
        "groups": len(groups),
        "task_versions": task_total,
        "groups_with_multiple_versions": sum(len(group["task_rows"]) > 1 for group in groups),
        "largest_family_task_versions": len(largest["task_rows"]),
        "largest_family_task_version_fraction": len(largest["task_rows"]) / task_total,
        "per_split": per_split,
    }


def _require_split_concentration(
    concentration: dict[str, Any], *, max_group_task_version_fraction: float
) -> None:
    """Reject any partition dominated by one reviewed task family.

    A global ratio can look safe while a smaller development or final-test
    partition is mostly one family.  The declared limit therefore applies to
    every split independently, which is the only ratio a later evaluator will
    actually observe.
    """
    for split, value in concentration["per_split"].items():
        if value["largest_family_task_version_fraction"] > max_group_task_version_fraction:
            raise ValueError(
                "one reviewed task family exceeds the explicit per-split concentration "
                f"limit in {split}"
            )


def build(
    rows: list[dict[str, Any]],
    *,
    inventory_sha256: str,
    seed: str,
    ratios: dict[str, float],
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
    max_group_task_version_fraction: float,
) -> dict[str, Any]:
    """Return a sealed split from reviewed metadata, or fail before a launch.

    ``inventory_sha256`` must bind the exact sanitized inventory supplied by a
    separate census.  It intentionally does not make this helper load private
    task content.
    """
    inventory_sha256 = _sha256(inventory_sha256, "sanitized inventory")
    if not isinstance(seed, str) or not seed:
        raise ValueError("nonempty split seed is required")
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("unique balanced dimensions are required")
    if set(dimensions) - set(DEFAULT_DIMENSIONS):
        raise ValueError("unsupported balanced dimension")
    if (
        isinstance(max_group_task_version_fraction, bool)
        or not isinstance(max_group_task_version_fraction, (int, float))
        or not 0 < max_group_task_version_fraction <= 1
    ):
        raise ValueError("an explicit 0 < maximum family fraction <= 1 is required")
    ratios = _normalise_ratios(ratios)
    groups = _group_rows(rows, dimensions)
    targets = _target_group_counts(len(groups), ratios)
    assignment = _assign(groups, seed=seed, targets=targets, dimensions=dimensions)
    assignment = _repair_representable_labels(
        groups,
        assignment,
        seed=seed,
        targets=targets,
        dimensions=dimensions,
    )
    representation = _representation(groups, assignment, targets=targets, dimensions=dimensions)
    _require_representable_labels(representation, splits=tuple(ratios))
    concentration = _concentration(groups, assignment, targets)
    _require_split_concentration(
        concentration,
        max_group_task_version_fraction=max_group_task_version_fraction,
    )
    tasks = []
    for group in groups:
        split = assignment[group["group_id"]]
        tasks.extend(
            {**row, "group_id": group["group_id"], "split": split} for row in group["task_rows"]
        )
    value = {
        "schema": SCHEMA,
        "inventory_sha256": inventory_sha256,
        "seed": seed,
        "split_unit": SPLIT_UNIT,
        "policy": {
            "ratios": ratios,
            "target_group_counts": targets,
            "balanced_dimensions": list(dimensions),
            "algorithm": "deterministic rarity-first grouped assignment",
            "require_representable_labels": True,
            "max_group_task_version_fraction": max_group_task_version_fraction,
        },
        "tasks": sorted(tasks, key=lambda row: (row["task_key"], row["task_version_id"])),
        "counts": {
            split: {
                "groups": sum(assignment[group["group_id"]] == split for group in groups),
                "task_versions": sum(
                    len(group["task_rows"])
                    for group in groups
                    if assignment[group["group_id"]] == split
                ),
            }
            for split in ratios
        },
        "representation": representation,
        "concentration": concentration,
        "leakage_checks": {
            "exact_identity_overlap": 0,
            "reviewed_family_overlap": 0,
            "all_inventory_versions_assigned_once": True,
        },
    }
    value["sha256"] = canonical_digest(value)
    return value


def _validate_v1(value: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the split from its sealed public policy and reject any drift."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or not _sealed(value):
        raise ValueError("invalid parameterized task-family split")
    policy = value.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("split policy is missing")
    if policy.get("require_representable_labels") is not True:
        raise ValueError("representable-strata coverage is mandatory")
    rebuilt = build(
        rows,
        inventory_sha256=value.get("inventory_sha256"),
        seed=value.get("seed"),
        ratios=policy.get("ratios"),
        dimensions=tuple(policy.get("balanced_dimensions") or ()),
        max_group_task_version_fraction=policy.get("max_group_task_version_fraction"),
    )
    if rebuilt != value:
        raise ValueError("parameterized task-family split drift")
    forbidden = json.dumps(value).lower()
    if any(f'"{term}"' in forbidden for term in FORBIDDEN_OUTPUT_TERMS):
        raise ValueError("split emitted a private task-content field")
    return value


def _role_entries(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must contain at least one role")
    result: dict[str, str] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"group_id", "split"}:
            raise ValueError(f"{label} role is malformed")
        group_id = _sha256(raw.get("group_id"), f"{label} group")
        split = _string(raw.get("split"), f"{label} split")
        if group_id in result:
            raise ValueError(f"{label} duplicates a group")
        result[group_id] = split
    if value != [{"group_id": group_id, "split": result[group_id]} for group_id in sorted(result)]:
        raise ValueError(f"{label} roles must be sorted canonically")
    return result


def _task_key_group_entries(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    result: dict[str, str] = {}
    for raw in value:
        if not isinstance(raw, dict) or set(raw) != {"task_key", "group_id"}:
            raise ValueError(f"{label} entry is malformed")
        task_key = _string(raw.get("task_key"), f"{label} task_key")
        group_id = _sha256(raw.get("group_id"), f"{label} group")
        if task_key in result:
            raise ValueError(f"{label} duplicates a task_key")
        result[task_key] = group_id
    if value != [
        {"task_key": task_key, "group_id": result[task_key]} for task_key in sorted(result)
    ]:
        raise ValueError(f"{label} entries must be sorted canonically")
    return result


def _role_anchor(value: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    required = {
        "schema",
        "source",
        "roles",
        "task_key_groups",
        "heldout_group_ids",
        "sha256",
    }
    if (
        not isinstance(value, dict)
        or value.get("schema") != ROLE_ANCHOR_SCHEMA
        or set(value) != required
        or not _sealed(value)
    ):
        raise ValueError("invalid task-family role anchor")
    source = value.get("source")
    if not isinstance(source, dict) or set(source) != {
        "split_schema",
        "split_sha256",
        "inventory_sha256",
    }:
        raise ValueError("task-family role-anchor source is malformed")
    if source.get("split_schema") not in {
        SCHEMA,
        ANCHORED_SCHEMA,
        "cyber_representative_study_split_v2",
    }:
        raise ValueError("task-family role-anchor source schema is unsupported")
    _sha256(source.get("split_sha256"), "role-anchor source split")
    _sha256(source.get("inventory_sha256"), "role-anchor source inventory")
    roles = _role_entries(value.get("roles"), "role-anchor")
    task_key_groups = _task_key_group_entries(value.get("task_key_groups"), "role-anchor")
    heldout = value.get("heldout_group_ids")
    if (
        not isinstance(heldout, list)
        or any(not isinstance(group, str) for group in heldout)
        or heldout != sorted(set(heldout))
        or any(_sha256(group, "role-anchor held-out group") != group for group in heldout)
    ):
        raise ValueError("role-anchor held-out groups are malformed")
    if heldout != sorted(group for group, split in roles.items() if split != "train"):
        raise ValueError("role-anchor held-out groups do not match its immutable roles")
    for task_key, group_id in task_key_groups.items():
        if group_id not in roles:
            raise ValueError(f"role-anchor task_key {task_key!r} names an unknown group")
    return roles, task_key_groups


def _role_rows(roles: dict[str, str]) -> list[dict[str, str]]:
    return [{"group_id": group_id, "split": roles[group_id]} for group_id in sorted(roles)]


def _task_key_group_rows(groups: dict[str, str]) -> list[dict[str, str]]:
    return [{"task_key": key, "group_id": groups[key]} for key in sorted(groups)]


def _make_role_anchor(
    *,
    split_schema: str,
    split_sha256: str,
    inventory_sha256: str,
    roles: dict[str, str],
    task_key_groups: dict[str, str],
) -> dict[str, Any]:
    """Seal a role anchor after callers have validated its source split."""
    value = {
        "schema": ROLE_ANCHOR_SCHEMA,
        "source": {
            "split_schema": split_schema,
            "split_sha256": _sha256(split_sha256, "role-anchor source split"),
            "inventory_sha256": _sha256(inventory_sha256, "role-anchor source inventory"),
        },
        "roles": _role_rows(roles),
        "task_key_groups": _task_key_group_rows(task_key_groups),
        "heldout_group_ids": sorted(
            group_id for group_id, split in roles.items() if split != "train"
        ),
    }
    value["sha256"] = canonical_digest(value)
    _role_anchor(value)
    return value


def _task_maps_from_split(
    value: dict[str, Any], rows: list[dict[str, Any]], dimensions: tuple[str, ...]
) -> tuple[dict[str, str], dict[str, str]]:
    """Return all currently assigned group roles and task-key lineage bindings."""
    groups = _group_rows(rows, dimensions)
    expected: dict[tuple[str, str], str] = {}
    task_key_groups: dict[str, str] = {}
    for group in groups:
        for row in group["task_rows"]:
            identity = (row["task_key"], row["task_version_id"])
            expected[identity] = group["group_id"]
            prior = task_key_groups.setdefault(row["task_key"], group["group_id"])
            if prior != group["group_id"]:
                raise ValueError("one task_key has inconsistent current family identity")
    assigned: dict[str, str] = {}
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(expected):
        raise ValueError("split task assignments are incomplete")
    seen: set[tuple[str, str]] = set()
    for raw in tasks:
        if not isinstance(raw, dict) or set(raw) != {
            "task_key",
            "task_version_id",
            "group_id",
            "split",
        }:
            raise ValueError("split task assignment is malformed")
        identity = _task_identity(raw)
        if identity in seen or identity not in expected:
            raise ValueError("split task assignment is ambiguous")
        seen.add(identity)
        group_id = _sha256(raw.get("group_id"), "split task group")
        if group_id != expected[identity]:
            raise ValueError("split task assignment changes reviewed family identity")
        split = _string(raw.get("split"), "split task role")
        prior = assigned.setdefault(group_id, split)
        if prior != split:
            raise ValueError("one reviewed task family crosses split roles")
    if seen != set(expected):
        raise ValueError("split task assignments do not cover the reviewed inventory")
    return assigned, task_key_groups


def freeze_role_anchor(
    value: dict[str, Any], rows: list[dict[str, Any]], *, role_anchor: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Freeze a validated parameterized split for a later catalog expansion.

    The output intentionally records all roles, including current training
    families.  Freezing only held-out groups would allow an old training family
    to migrate later and make comparisons with an earlier study ambiguous.
    """
    if not is_supported_schema(value):
        raise ValueError("only supported parameterized family splits can be frozen")
    validate(value, rows, role_anchor=role_anchor)
    policy = value["policy"]
    roles, task_key_groups = _task_maps_from_split(
        value, rows, tuple(policy["balanced_dimensions"])
    )
    if value["schema"] == ANCHORED_SCHEMA:
        inherited_roles = _role_entries(value["inherited_roles"], "anchored inherited")
        inherited_keys = _task_key_group_entries(
            value["inherited_task_key_groups"], "anchored inherited task-key"
        )
        for group_id, split in inherited_roles.items():
            prior = roles.setdefault(group_id, split)
            if prior != split:
                raise ValueError("anchored split changed an inherited role before freezing")
        for task_key, group_id in inherited_keys.items():
            prior = task_key_groups.setdefault(task_key, group_id)
            if prior != group_id:
                raise ValueError("anchored split changed an inherited task-key family")
    return _make_role_anchor(
        split_schema=value["schema"],
        split_sha256=value["sha256"],
        inventory_sha256=value["inventory_sha256"],
        roles=roles,
        task_key_groups=task_key_groups,
    )


def freeze_study_v2_role_anchor(
    legacy_split: dict[str, Any],
    *,
    legacy_inventory_path: Path,
    legacy_inventory_display_path: Path | None = None,
) -> dict[str, Any]:
    """Convert the locked 75-task study split once, without recomputing roles."""
    from . import study_split_v2

    study_split_v2.validate(
        legacy_split,
        inventory_path=legacy_inventory_path,
        inventory_display_path=legacy_inventory_display_path,
    )
    inventory = json.loads(legacy_inventory_path.read_text())
    rows = inventory.get("task_versions")
    if not isinstance(rows, list):
        raise ValueError("legacy study inventory omits task_versions")
    groups = _group_rows(rows, DEFAULT_DIMENSIONS)
    by_identity = {
        (row["task_key"], row["task_version_id"]): group["group_id"]
        for group in groups
        for row in group["task_rows"]
    }
    roles: dict[str, str] = {}
    task_key_groups: dict[str, str] = {}
    tasks = legacy_split.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("legacy study split omits task assignments")
    for raw in tasks:
        if not isinstance(raw, dict) or set(raw) != {
            "task_key",
            "task_version_id",
            "group_id",
            "split",
        }:
            raise ValueError("legacy study split task assignment is malformed")
        identity = _task_identity(raw)
        expected_group = by_identity.get(identity)
        group_id = _sha256(raw.get("group_id"), "legacy study group")
        if expected_group is None or group_id != expected_group:
            raise ValueError("legacy study split task assignment drifts from its inventory")
        split = _string(raw.get("split"), "legacy study role")
        prior = roles.setdefault(group_id, split)
        if prior != split:
            raise ValueError("legacy study assigns one family to multiple roles")
        prior_group = task_key_groups.setdefault(identity[0], group_id)
        if prior_group != group_id:
            raise ValueError("legacy study task_key changes family identity")
    if legacy_split["inventory"]["logical_sha256"] != inventory.get("sha256"):
        raise ValueError("legacy study inventory logical digest drift")
    return _make_role_anchor(
        split_schema=legacy_split["schema"],
        split_sha256=legacy_split["sha256"],
        inventory_sha256=legacy_split["inventory"]["logical_sha256"],
        roles=roles,
        task_key_groups=task_key_groups,
    )


def _assign_anchored(
    groups: list[dict[str, Any]],
    *,
    seed: str,
    targets: dict[str, int],
    dimensions: tuple[str, ...],
    locked: dict[str, str],
) -> dict[str, str]:
    """Assign only novel groups while preserving every anchored role exactly."""
    population = _population(groups, dimensions)

    def rarity(group: dict[str, Any]) -> float:
        return sum(
            1 / population[dimension][label]
            for dimension in dimensions
            for label in group["features"][dimension]
        )

    assignment = dict(locked)
    used = Counter(locked.values())
    order = sorted(
        (group for group in groups if group["group_id"] not in locked),
        key=lambda group: (-rarity(group), _rank(seed, "anchored-order", group["group_id"])),
    )
    for group in order:
        candidates = [split for split in targets if used[split] < targets[split]]
        if not candidates:
            raise ValueError("anchored task-family allocation exhausted unexpectedly")
        split = min(
            candidates,
            key=lambda candidate: (
                _assignment_score(
                    groups,
                    assignment | {group["group_id"]: candidate},
                    targets=targets,
                    dimensions=dimensions,
                ),
                _rank(seed, "anchored-choice", group["group_id"], candidate),
            ),
        )
        assignment[group["group_id"]] = split
        used[split] += 1
    return assignment


def _validate_anchor_roles_for_build(roles: dict[str, str], ratios: dict[str, float]) -> None:
    if set(roles.values()) != set(ratios):
        raise ValueError("role-anchor roles must exactly match the declared split roles")
    if "train" not in ratios:
        raise ValueError("anchored task-family splits require a train role")


def _build_anchored_from_parts(
    rows: list[dict[str, Any]],
    *,
    inventory_sha256: str,
    parent_role_anchor_sha256: str,
    inherited_roles: dict[str, str],
    inherited_task_key_groups: dict[str, str],
    seed: str,
    ratios: dict[str, float],
    dimensions: tuple[str, ...],
    max_group_task_version_fraction: float,
) -> dict[str, Any]:
    inventory_sha256 = _sha256(inventory_sha256, "sanitized inventory")
    parent_role_anchor_sha256 = _sha256(parent_role_anchor_sha256, "parent role anchor")
    if not isinstance(seed, str) or not seed:
        raise ValueError("nonempty split seed is required")
    if not dimensions or len(set(dimensions)) != len(dimensions):
        raise ValueError("unique balanced dimensions are required")
    if set(dimensions) - set(DEFAULT_DIMENSIONS):
        raise ValueError("unsupported balanced dimension")
    if (
        isinstance(max_group_task_version_fraction, bool)
        or not isinstance(max_group_task_version_fraction, (int, float))
        or not 0 < max_group_task_version_fraction <= 1
    ):
        raise ValueError("an explicit 0 < maximum family fraction <= 1 is required")
    ratios = _normalise_ratios(ratios)
    _validate_anchor_roles_for_build(inherited_roles, ratios)
    groups = _group_rows(rows, dimensions)
    current_group_ids = {group["group_id"] for group in groups}
    current_task_key_groups = {
        row["task_key"]: group["group_id"] for group in groups for row in group["task_rows"]
    }
    for task_key, prior_group in inherited_task_key_groups.items():
        if task_key in current_task_key_groups and current_task_key_groups[task_key] != prior_group:
            raise ValueError("historical task_key changes its immutable family identity")
    locked = {
        group_id: split
        for group_id, split in inherited_roles.items()
        if group_id in current_group_ids
    }
    targets = _target_group_counts(len(groups), ratios)
    locked_counts = Counter(locked.values())
    if any(locked_counts[split] > targets[split] for split in targets):
        raise ValueError("immutable anchored roles exceed the declared target split counts")
    assignment = _assign_anchored(
        groups,
        seed=seed,
        targets=targets,
        dimensions=dimensions,
        locked=locked,
    )
    assignment = _repair_representable_labels(
        groups,
        assignment,
        seed=seed,
        targets=targets,
        dimensions=dimensions,
        locked_group_ids=frozenset(locked),
    )
    if any(assignment[group_id] != split for group_id, split in locked.items()):
        raise ValueError("representative repair changed an immutable anchored role")
    representation = _representation(groups, assignment, targets=targets, dimensions=dimensions)
    _require_representable_labels(representation, splits=tuple(ratios))
    concentration = _concentration(groups, assignment, targets)
    _require_split_concentration(
        concentration,
        max_group_task_version_fraction=max_group_task_version_fraction,
    )
    tasks = []
    for group in groups:
        split = assignment[group["group_id"]]
        tasks.extend(
            {**row, "group_id": group["group_id"], "split": split} for row in group["task_rows"]
        )
    new_group_ids = sorted(current_group_ids - set(inherited_roles))
    value = {
        "schema": ANCHORED_SCHEMA,
        "inventory_sha256": inventory_sha256,
        "parent_role_anchor_sha256": parent_role_anchor_sha256,
        "inherited_roles": _role_rows(inherited_roles),
        "inherited_task_key_groups": _task_key_group_rows(inherited_task_key_groups),
        "new_group_ids": new_group_ids,
        "seed": seed,
        "split_unit": SPLIT_UNIT,
        "policy": {
            "ratios": ratios,
            "target_group_counts": targets,
            "balanced_dimensions": list(dimensions),
            "algorithm": (
                "deterministic rarity-first assignment of new groups with immutable inherited roles"
            ),
            "require_representable_labels": True,
            "max_group_task_version_fraction": max_group_task_version_fraction,
            "inherited_roles_immutable": True,
        },
        "tasks": sorted(tasks, key=lambda row: (row["task_key"], row["task_version_id"])),
        "counts": {
            split: {
                "groups": sum(assignment[group["group_id"]] == split for group in groups),
                "task_versions": sum(
                    len(group["task_rows"])
                    for group in groups
                    if assignment[group["group_id"]] == split
                ),
            }
            for split in ratios
        },
        "representation": representation,
        "concentration": concentration,
        "leakage_checks": {
            "exact_identity_overlap": 0,
            "reviewed_family_overlap": 0,
            "all_inventory_versions_assigned_once": True,
            "immutable_inherited_role_drift": 0,
        },
        "anchor_audit": {
            "inherited_role_count": len(inherited_roles),
            "present_inherited_group_count": len(locked),
            "absent_inherited_group_count": len(inherited_roles) - len(locked),
            "new_group_count": len(new_group_ids),
            "inherited_heldout_group_count": sum(
                split != "train" for split in inherited_roles.values()
            ),
            "historical_task_key_group_drift": 0,
        },
    }
    value["sha256"] = canonical_digest(value)
    return value


def build_anchored(
    rows: list[dict[str, Any]],
    *,
    inventory_sha256: str,
    role_anchor: dict[str, Any],
    seed: str,
    ratios: dict[str, float],
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS,
    max_group_task_version_fraction: float,
) -> dict[str, Any]:
    """Extend a catalog without changing any role already frozen in ``role_anchor``."""
    roles, task_key_groups = _role_anchor(role_anchor)
    return _build_anchored_from_parts(
        rows,
        inventory_sha256=inventory_sha256,
        parent_role_anchor_sha256=role_anchor["sha256"],
        inherited_roles=roles,
        inherited_task_key_groups=task_key_groups,
        seed=seed,
        ratios=ratios,
        dimensions=dimensions,
        max_group_task_version_fraction=max_group_task_version_fraction,
    )


def _validate_anchored(
    value: dict[str, Any], rows: list[dict[str, Any]], *, role_anchor: dict[str, Any] | None
) -> dict[str, Any]:
    required = {
        "schema",
        "inventory_sha256",
        "parent_role_anchor_sha256",
        "inherited_roles",
        "inherited_task_key_groups",
        "new_group_ids",
        "seed",
        "split_unit",
        "policy",
        "tasks",
        "counts",
        "representation",
        "concentration",
        "leakage_checks",
        "anchor_audit",
        "sha256",
    }
    if (
        not isinstance(value, dict)
        or value.get("schema") != ANCHORED_SCHEMA
        or set(value) != required
        or not _sealed(value)
    ):
        raise ValueError("invalid anchored task-family split")
    if value.get("split_unit") != SPLIT_UNIT:
        raise ValueError("anchored split unit drift")
    if role_anchor is None:
        raise ValueError("anchored split requires its independently sealed role anchor")
    anchored_roles, anchored_task_key_groups = _role_anchor(role_anchor)
    if value.get("parent_role_anchor_sha256") != role_anchor.get("sha256"):
        raise ValueError("anchored split parent role anchor digest mismatch")
    inherited_roles = _role_entries(value.get("inherited_roles"), "anchored inherited")
    inherited_task_key_groups = _task_key_group_entries(
        value.get("inherited_task_key_groups"), "anchored inherited task-key"
    )
    if inherited_roles != anchored_roles or inherited_task_key_groups != anchored_task_key_groups:
        raise ValueError("anchored split does not exactly inherit its sealed role anchor")
    for group_id in inherited_task_key_groups.values():
        if group_id not in inherited_roles:
            raise ValueError("anchored inherited task_key names an unknown role")
    policy = value.get("policy")
    if not isinstance(policy, dict) or set(policy) != {
        "ratios",
        "target_group_counts",
        "balanced_dimensions",
        "algorithm",
        "require_representable_labels",
        "max_group_task_version_fraction",
        "inherited_roles_immutable",
    }:
        raise ValueError("anchored split policy is missing")
    if (
        policy.get("algorithm")
        != "deterministic rarity-first assignment of new groups with immutable inherited roles"
        or policy.get("require_representable_labels") is not True
        or policy.get("inherited_roles_immutable") is not True
    ):
        raise ValueError("anchored split policy drift")
    rebuilt = _build_anchored_from_parts(
        rows,
        inventory_sha256=value.get("inventory_sha256"),
        parent_role_anchor_sha256=value.get("parent_role_anchor_sha256"),
        inherited_roles=inherited_roles,
        inherited_task_key_groups=inherited_task_key_groups,
        seed=value.get("seed"),
        ratios=policy.get("ratios"),
        dimensions=tuple(policy.get("balanced_dimensions") or ()),
        max_group_task_version_fraction=policy.get("max_group_task_version_fraction"),
    )
    if rebuilt != value:
        raise ValueError("anchored task-family split drift")
    forbidden = json.dumps(value).lower()
    if any(f'"{term}"' in forbidden for term in FORBIDDEN_OUTPUT_TERMS):
        raise ValueError("anchored split emitted a private task-content field")
    return value


def validate(
    value: dict[str, Any], rows: list[dict[str, Any]], *, role_anchor: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate either the original split or an additive anchored extension."""
    if not isinstance(value, dict):
        raise ValueError("invalid parameterized task-family split")
    if value.get("schema") == SCHEMA:
        if role_anchor is not None:
            raise ValueError("parameterized split must not carry an unrelated role anchor")
        return _validate_v1(value, rows)
    if value.get("schema") == ANCHORED_SCHEMA:
        return _validate_anchored(value, rows, role_anchor=role_anchor)
    raise ValueError("unsupported parameterized task-family split schema")


def _read_inventory(path: Path) -> tuple[list[dict[str, Any]], str]:
    value = json.loads(path.read_text())
    rows = value.get("task_versions")
    if not isinstance(rows, list):
        raise ValueError("inventory omitted task_versions[]")
    if not _sealed(value):
        raise ValueError("inventory must carry a valid sanitized digest")
    return rows, value["sha256"]


def write_once(path: Path, value: dict[str, Any]) -> None:
    """Write the sealed public split exactly once; never replace prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError("create-once split output already exists") from error
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument(
        "--ratios-json",
        required=True,
        help='e.g. {"train":0.8,"dev":0.1,"test":0.1}',
    )
    parser.add_argument("--max-group-task-version-fraction", type=float, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows, inventory_sha256 = _read_inventory(args.inventory)
    if args.check:
        validate(json.loads(args.output.read_text()), rows)
        return
    ratios = json.loads(args.ratios_json)
    if not isinstance(ratios, dict):
        raise ValueError("ratios JSON must be an object")
    value = build(
        rows,
        inventory_sha256=inventory_sha256,
        seed=args.seed,
        ratios=ratios,
        max_group_task_version_fraction=args.max_group_task_version_fraction,
    )
    write_once(args.output, value)


if __name__ == "__main__":
    main()
