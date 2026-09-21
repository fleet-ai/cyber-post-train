"""Freeze the existing immutable study split into a reusable role anchor.

This is a source-only migration helper.  It validates the historical split
against its exact sanitized inventory and preserves every old family role; it
does not rebalance, inspect task content, or contact Fleet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import file_sha256
from .sft import _known
from .task_family_split import freeze_study_v2_role_anchor, write_once

REQUEST_SCHEMA = "cyber_fleet_collection_role_anchor_request_v1"


def _reference(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a file reference")
    _known(value, {"path", "sha256"}, label)
    path_value = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"{label} path is required")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        raise ValueError(f"{label} SHA-256 is required")
    path = Path(path_value)
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file digest mismatch")
    return path


def _output(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("role-anchor output path is required")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.exists() or path.is_symlink():
        raise FileExistsError("role-anchor output already exists")
    return path


def build(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Validate and freeze the v2 study roles into one create-once JSON file."""
    _known(
        config,
        {"schema", "legacy_split", "legacy_inventory", "output"},
        "Fleet role-anchor request",
    )
    if config.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported Fleet role-anchor request schema")
    split_path = _reference(relative_to, config.get("legacy_split"), "legacy split")
    inventory_path = _reference(relative_to, config.get("legacy_inventory"), "legacy inventory")
    try:
        split = json.loads(split_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid legacy split JSON") from error
    if not isinstance(split, dict):
        raise ValueError("legacy split must be a JSON object")
    declared_inventory_path = (split.get("inventory") or {}).get("path")
    if not isinstance(declared_inventory_path, str) or not declared_inventory_path:
        raise ValueError("legacy split does not declare its immutable inventory path")
    display_path = Path(declared_inventory_path)
    expected_path = display_path if display_path.is_absolute() else relative_to / display_path
    if expected_path.resolve() != inventory_path.resolve():
        raise ValueError(
            "legacy inventory reference differs from the split's immutable inventory path"
        )
    anchor = freeze_study_v2_role_anchor(
        split,
        legacy_inventory_path=inventory_path,
        legacy_inventory_display_path=display_path,
    )
    output = _output(relative_to, config.get("output"))
    write_once(output, anchor)
    return {
        "submitted": False,
        "artifact_kind": "immutable_family_role_anchor",
        "external_actions_submitted": False,
        "output": str(output),
        "role_anchor_sha256": anchor["sha256"],
        "inherited_family_count": len(anchor["roles"]),
        "heldout_family_count": len(anchor["heldout_group_ids"]),
    }
