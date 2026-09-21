"""Build a sealed, growing Fleet collection roster without opening task content.

The module deliberately works only with a sanitized supply catalog and a
separately qualified subset.  It does not read prompts, trajectories, model
outputs, rollout outcomes, or credentials.  Qualification is an upstream
evidence-producing process; this module verifies its digest-bound metadata,
preserves every inherited task-family role, and renders the generic inputs for
``training.collection_campaign``.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from . import collection_campaign, task_family_split
from . import fleet_collection_admission as admission
from .io import file_sha256
from .sft import _known

REQUEST_SCHEMA = "cyber_fleet_collection_roster_request_v1"
SUPPLY_SCHEMA = "cyber_fleet_blackbox_task_supply_catalog_v1"
QUALIFIED_SCHEMA = "cyber_fleet_task_qualification_catalog_v1"
RECEIPT_SCHEMA = "cyber_fleet_collection_roster_receipt_v1"

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_FORBIDDEN_KEYS = collection_campaign.FORBIDDEN_CONTENT_KEYS | {
    "attempt",
    "failure",
    "grade",
    "outcome",
    "pass",
    "result",
    "reward",
    "rollout",
    "success",
    "verdict",
}


def _sealed(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != collection_campaign.canonical_digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid sealed {schema}")


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"exact {label} SHA-256 is required")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"nonempty {label} is required")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _input(root: Path, value: object, label: str) -> tuple[Path, dict[str, Any]]:
    reference = _mapping(value, label)
    _known(reference, {"path", "sha256"}, label)
    path = _path(root, _string(reference.get("path"), f"{label} path"))
    expected = _sha(reference.get("sha256"), f"{label} file")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} file digest mismatch")
    try:
        parsed = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} JSON") from error
    return path, _mapping(parsed, label)


def _no_content_or_outcome(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("roster metadata keys must be strings")
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError("roster metadata must not contain content or rollout outcomes")
            _no_content_or_outcome(item)
    elif isinstance(value, list):
        for item in value:
            _no_content_or_outcome(item)


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"canonical {label} UUID is required")
    try:
        if str(uuid.UUID(value)) != value:
            raise ValueError
    except ValueError as error:
        raise ValueError(f"canonical {label} UUID is required") from error
    return value


def _identity(row: dict[str, Any], label: str) -> tuple[str, str]:
    return _string(row.get("task_key"), f"{label} task_key"), _uuid(
        row.get("task_version_id"), f"{label} task version"
    )


def _supply(value: dict[str, Any]) -> set[tuple[str, str]]:
    _sealed(value, SUPPLY_SCHEMA)
    if set(value) != {"schema", "catalog_snapshot_sha256", "task_versions", "sha256"}:
        raise ValueError("supply catalog has unknown or missing fields")
    _no_content_or_outcome(value)
    _sha(value.get("catalog_snapshot_sha256"), "supply source catalog snapshot")
    rows = value.get("task_versions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("supply catalog has no task versions")
    result = set()
    for raw in rows:
        row = _mapping(raw, "supply task")
        if set(row) != {"task_key", "task_version_id"}:
            raise ValueError("supply catalog task has unknown or missing fields")
        identity = _identity(row, "supply task")
        if identity in result:
            raise ValueError("supply catalog duplicates a task version")
        result.add(identity)
    return result


def _lineage(value: object) -> dict[str, Any]:
    lineage = _mapping(value, "qualified task lineage")
    if set(lineage) != collection_campaign.LINEAGE_FIELDS:
        raise ValueError("qualified task lineage is incomplete")
    for field in collection_campaign.LINEAGE_FIELDS - {"vulnerability_family"}:
        if (
            not isinstance(lineage.get(field), str)
            or not lineage[field]
            or lineage[field] == "unknown"
        ):
            raise ValueError("qualified task lineage is not reviewed")
    families = lineage.get("vulnerability_family")
    if (
        not isinstance(families, list)
        or not families
        or any(not isinstance(item, str) or not item or item == "unknown" for item in families)
    ):
        raise ValueError("qualified vulnerability-family metadata is not reviewed")
    return {
        **lineage,
        "vulnerability_family": sorted(set(families)),
    }


def _runtime(value: object) -> dict[str, str]:
    runtime = _mapping(value, "qualified task runtime")
    if set(runtime) != set(collection_campaign.EXACT_TASK_FIELDS):
        raise ValueError("qualified task runtime is incomplete")
    identity = _identity(runtime, "qualified runtime")
    _uuid(runtime.get("environment_version_id"), "environment version")
    if any(not isinstance(runtime[field], str) or not runtime[field] for field in runtime):
        raise ValueError("qualified task runtime has an invalid exact binding")
    return {**runtime, "task_key": identity[0], "task_version_id": identity[1]}


def _qualified(
    value: dict[str, Any], supply: set[tuple[str, str]], supply_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    _sealed(value, QUALIFIED_SCHEMA)
    if set(value) != {
        "schema",
        "supply_catalog_sha256",
        "catalog_snapshot_sha256",
        "task_validity_receipt_sha256",
        "task_versions",
        "sha256",
    }:
        raise ValueError("qualified catalog has unknown or missing fields")
    _no_content_or_outcome(value)
    if value.get("supply_catalog_sha256") != supply_sha256:
        raise ValueError("qualified catalog is bound to a different supply catalog")
    if value.get("catalog_snapshot_sha256") is None:
        raise ValueError("qualified catalog omits its source catalog snapshot")
    _sha(value["catalog_snapshot_sha256"], "qualified source catalog snapshot")
    validity = _sha(value.get("task_validity_receipt_sha256"), "task-validity receipt")
    rows = value.get("task_versions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("qualified catalog has no task versions")
    metadata: list[dict[str, Any]] = []
    bindings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in rows:
        row = _mapping(raw, "qualified task")
        if set(row) != {
            "task_key",
            "task_version_id",
            "lineage",
            "runtime",
            "qualification_receipt_sha256",
        }:
            raise ValueError("qualified task has unknown or missing fields")
        identity = _identity(row, "qualified task")
        if identity in seen or identity not in supply:
            raise ValueError("qualified task is duplicated or absent from the supply catalog")
        seen.add(identity)
        runtime = _runtime(row.get("runtime"))
        if (runtime["task_key"], runtime["task_version_id"]) != identity:
            raise ValueError("qualified task identity differs from its runtime binding")
        metadata.append(
            {
                "task_key": identity[0],
                "task_version_id": identity[1],
                "lineage": _lineage(row.get("lineage")),
            }
        )
        bindings.append(runtime)
        _sha(row.get("qualification_receipt_sha256"), "task qualification receipt")
    metadata.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    bindings.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    inventory = collection_campaign.sealed(
        {
            "schema": collection_campaign.INVENTORY_SCHEMA,
            "task_validity_receipt_sha256": validity,
            "task_versions": metadata,
        }
    )
    runtime_bindings = collection_campaign.sealed(
        {
            "schema": collection_campaign.RUNTIME_BINDINGS_SCHEMA,
            "metadata_inventory_sha256": inventory["sha256"],
            "task_validity_receipt_sha256": validity,
            "task_versions": bindings,
        }
    )
    # Reuse the generic validators so this adapter cannot emit an object the
    # packet renderer would later interpret differently.
    collection_campaign._runtime_bindings(runtime_bindings, inventory, metadata)
    return metadata, inventory, runtime_bindings


def render(config: dict[str, Any], *, relative_to: Path) -> dict[str, dict[str, Any]]:
    """Render a public-safe roster and generic collection inputs without I/O writes."""
    _known(
        config,
        {
            "schema",
            "supply_catalog",
            "qualified_catalog",
            "role_anchor",
            "seed",
            "ratios",
            "max_group_task_version_fraction",
            "output",
        },
        "Fleet collection roster request",
    )
    if config.get("schema") != REQUEST_SCHEMA:
        raise ValueError("unsupported Fleet collection roster request schema")
    _, supply = _input(relative_to, config.get("supply_catalog"), "supply catalog")
    _, qualified = _input(relative_to, config.get("qualified_catalog"), "qualified catalog")
    _, anchor = _input(relative_to, config.get("role_anchor"), "role anchor")
    supply_ids = _supply(supply)
    metadata, inventory, runtime_bindings = _qualified(qualified, supply_ids, supply["sha256"])
    if qualified["catalog_snapshot_sha256"] != supply["catalog_snapshot_sha256"]:
        raise ValueError("qualified catalog is bound to a different source catalog snapshot")
    # ``build_anchored`` validates the immutable parent evidence before
    # allocating any newly qualified family.  No historical split is re-run or
    # rebalanced here.
    split = task_family_split.build_anchored(
        metadata,
        inventory_sha256=inventory["sha256"],
        role_anchor=anchor,
        seed=_string(config.get("seed"), "roster seed"),
        ratios=_mapping(config.get("ratios"), "roster ratios"),
        max_group_task_version_fraction=config.get("max_group_task_version_fraction"),
    )
    task_family_split.validate(split, metadata, role_anchor=anchor)
    protected = collection_campaign.sealed(
        {
            "schema": admission.PROTECTED_FAMILY_LOCK_SCHEMA,
            "source_split_sha256": split["sha256"],
            "heldout_group_ids": sorted(
                {row["group_id"] for row in split["tasks"] if row["split"] in {"dev", "final_test"}}
            ),
        }
    )
    receipt = collection_campaign.sealed(
        {
            "schema": RECEIPT_SCHEMA,
            "supply_catalog_sha256": supply["sha256"],
            "catalog_snapshot_sha256": supply["catalog_snapshot_sha256"],
            "qualified_catalog_sha256": qualified["sha256"],
            "role_anchor_sha256": anchor["sha256"],
            "metadata_inventory_sha256": inventory["sha256"],
            "runtime_bindings_sha256": runtime_bindings["sha256"],
            "family_split_sha256": split["sha256"],
            "protected_family_lock_sha256": protected["sha256"],
            "task_versions": len(metadata),
            "counts": split["counts"],
            "outcome_blind": True,
            "external_actions_submitted": False,
        }
    )
    return {
        "metadata-inventory.json": inventory,
        "runtime-bindings.json": runtime_bindings,
        "role-anchor.json": anchor,
        "family-split.json": split,
        "protected-family-lock.json": protected,
        "ROSTER.json": receipt,
    }


def _output(root: Path, value: object) -> Path:
    path = _path(root, _string(value, "roster output"))
    if path.exists() or path.is_symlink():
        raise FileExistsError("collection roster destination already exists")
    return path


def _write_once(output: Path, rendered: dict[str, dict[str, Any]]) -> None:
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError("collection roster destination already exists") from error
    try:
        for name, value in rendered.items():
            path = output / name
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        # A partial create-once evidence directory must never look like a
        # complete roster.  Preserve no stale files for an operator to reuse.
        for child in output.glob("*"):
            child.unlink(missing_ok=True)
        output.rmdir()
        raise


def build(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Write one sealed roster directory and return aggregate-only handoff facts."""
    output = _output(relative_to, config.get("output"))
    rendered = render(config, relative_to=relative_to)
    _write_once(output, rendered)
    receipt = rendered["ROSTER.json"]
    return {
        "submitted": False,
        "artifact_kind": "metadata_roster_handoff_only",
        "external_actions_submitted": False,
        "output": str(output),
        "task_versions": receipt["task_versions"],
        "counts": receipt["counts"],
        "roster_receipt_sha256": receipt["sha256"],
        "family_split_sha256": receipt["family_split_sha256"],
    }
