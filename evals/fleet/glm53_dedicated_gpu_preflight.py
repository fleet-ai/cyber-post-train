"""Fail-closed hardware inventory gate for GLM5.3 TP8 serving."""

from __future__ import annotations

import csv
import io
from typing import Any

EXPECTED_GPU_COUNT = 8
EXPECTED_GPU_NAME = "NVIDIA B300 SXM6 PC"
EXPECTED_MEMORY_MIB = 275040
EXPECTED_COMPUTE_CAPABILITY = "10.3"


def parse_nvidia_smi_csv(value: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in csv.reader(io.StringIO(value)):
        if len(raw) != 4:
            raise ValueError("GPU inventory row shape drifted")
        index, name, memory_mib, compute_capability = (item.strip() for item in raw)
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_mib": int(memory_mib),
                "compute_capability": compute_capability,
            }
        )
    return rows


def validate_inventory(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_GPU_COUNT:
        raise ValueError("TP8 requires exactly eight visible GPUs")
    if [row["index"] for row in rows] != list(range(EXPECTED_GPU_COUNT)):
        raise ValueError("GPU indices are not the exact contiguous TP8 inventory")
    if {row["name"] for row in rows} != {EXPECTED_GPU_NAME}:
        raise ValueError("GPU product inventory is heterogeneous")
    if {row["compute_capability"] for row in rows} != {EXPECTED_COMPUTE_CAPABILITY}:
        raise ValueError("GPU compute-capability inventory is heterogeneous")
    if {row["memory_mib"] for row in rows} != {EXPECTED_MEMORY_MIB}:
        raise ValueError("GPU memory inventory is inconsistent with the B300 node contract")


def validate_nvidia_smi_csv(value: str) -> list[dict[str, Any]]:
    rows = parse_nvidia_smi_csv(value)
    validate_inventory(rows)
    return rows
