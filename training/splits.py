"""Deterministic, lineage-safe dataset assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

SPLIT_SEED = "fleet-cyber-split-v1"


def split_key(record: Mapping[str, Any]) -> str:
    """Bind app, vulnerability family and immutable lineage into one split unit."""
    lineage = record["lineage"]
    return "|".join(
        [
            str(lineage.get("application")),
            str(lineage.get("task_family")),
            str(lineage.get("lineage_key")),
        ]
    )


def assign_split(key: str, *, train: float = 0.8, dev: float = 0.1) -> str:
    if train <= 0 or dev < 0 or train + dev >= 1:
        raise ValueError("split ratios must leave positive train and test partitions")
    digest = hashlib.sha256(f"{SPLIT_SEED}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8]) / 2**64
    if value < train:
        return "train"
    if value < train + dev:
        return "dev"
    return "test"


def apply_splits(records: Iterable[dict[str, Any]]) -> None:
    observed: dict[str, set[str]] = defaultdict(set)
    for record in records:
        unit = split_key(record)
        split = assign_split(unit)
        record["split"] = split
        record["split_unit"] = unit
        observed[str(record["lineage"]["lineage_key"])].add(split)
    leaking = [lineage for lineage, splits in observed.items() if len(splits) != 1]
    if leaking:
        raise ValueError(f"lineage leakage across splits: {leaking[:5]}")
