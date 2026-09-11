"""Deterministic, lineage-safe dataset assignment."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

SPLIT_SEED = "fleet-cyber-family-split-v2"


def split_key(record: Mapping[str, Any]) -> str:
    """Keep all versions/attempts of an application/task family together.

    Immutable version IDs identify examples, not independent split units. This
    is family-held-out, not application-held-out: different families of one app
    may be in different partitions. Existing frozen v1 corpora are not rewritten.
    """
    lineage = record["lineage"]
    values = [lineage.get("application"), lineage.get("task_family")]
    if any(not isinstance(v, str) or not v.strip() or v == "unknown" for v in values):
        raise ValueError("split requires a known application and task family")
    return json.dumps(values, separators=(",", ":"))


def assign_split(key: str, *, train: float = 0.8, dev: float = 0.1) -> str:
    if (
        not math.isfinite(train)
        or not math.isfinite(dev)
        or train <= 0
        or dev < 0
        or train + dev >= 1
    ):
        raise ValueError("split ratios must leave positive train and test partitions")
    digest = hashlib.sha256(f"{SPLIT_SEED}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8]) / 2**64
    if value < train:
        return "train"
    if value < train + dev:
        return "dev"
    return "test"


def apply_splits(records: Iterable[dict[str, Any]]) -> None:
    assignments = []
    observed: dict[str, str] = {}
    for record in records:
        unit = split_key(record)
        identity = record["lineage"].get("lineage_key")
        if not isinstance(identity, str) or not identity:
            raise ValueError("split requires task lineage identity")
        # A renamed registry task must not evade the family boundary. Conflicting
        # metadata needs review, not an order-dependent choice of split.
        if identity.startswith("registry:"):
            identity = identity.rsplit("@", 1)[0]
        if observed.setdefault(identity, unit) != unit:
            raise ValueError("conflicting application/family for one task lineage")
        assignments.append((record, unit))
    # Validate the whole input before mutating any record.
    for record, unit in assignments:
        record.update(split=assign_split(unit), split_unit=unit)
