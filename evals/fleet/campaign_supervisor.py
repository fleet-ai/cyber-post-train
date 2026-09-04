"""Cluster-native, score-blind campaign ledger and exact-worker monitor.

The ledger identifies evaluation cells by model, serving block, immutable task
version and pass@k attempt.  It never reads prompts, traces, flags or scores.
All state transitions are append-only and use O_EXCL so two controllers cannot
claim or terminalize the same cell concurrently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAMPAIGN_SCHEMA = "fleet-score-blind-campaign-v1"
UNIVERSE_SCHEMA = "fleet-score-blind-cell-universe-v1"
LEDGER_EVENT_SCHEMA = "fleet-score-blind-cell-event-v1"
WORKER_SCHEMA = "fleet-score-blind-worker-registration-v1"
HEARTBEAT_SCHEMA = "fleet-score-blind-campaign-heartbeat-v1"
LEGACY_IMPORT_SCHEMA = "fleet-score-blind-legacy-import-v1"
SCIENTIFIC_MAPPING_SCHEMA = "fleet-score-blind-scientific-mapping-v2"
HIGH_PRIORITY_CLASS = "fleet-train-high"


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def digest_without(value: dict[str, Any], field: str) -> str:
    return sha256(canonical_json({key: item for key, item in value.items() if key != field}))


def read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required input is not a regular file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"required input is not an object: {path}")
    return value


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = canonical_json(value) + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _uuidish(value: Any) -> str:
    text = str(value or "")
    try:
        parsed = uuid.UUID(text)
    except ValueError as exc:
        raise ValueError("worker UID must be a nonzero UUID string") from exc
    if parsed.int == 0 or str(parsed) != text:
        raise ValueError("worker UID must be a nonzero UUID string")
    return text


def validate_campaign(campaign: dict[str, Any]) -> None:
    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise ValueError("unsupported campaign schema")
    if campaign.get("campaign_sha256") != digest_without(campaign, "campaign_sha256"):
        raise ValueError("campaign digest mismatch")
    if (
        campaign.get("release_status") != "released"
        or campaign.get("launch_authorized") is not True
    ):
        raise ValueError("campaign universe is blocked pending an append-only replacement release")
    expected = campaign.get("expected") or {}
    if expected != {
        "models": {"qwen3.8-27b": 200, "glm-5.3": 400},
        "total_cells": 600,
        "dedicated_nodes": 2,
        "dedicated_gpus": 16,
    }:
        raise ValueError("campaign denominator or dedicated capacity drifted")
    components = campaign.get("components") or []
    if not isinstance(components, list) or len(components) != 5:
        raise ValueError("campaign requires the five frozen serving components")
    if len({row.get("id") for row in components}) != len(components):
        raise ValueError("campaign component IDs duplicated")
    dedicated_nodes = sum(int(row.get("dedicated_nodes") or 0) for row in components)
    dedicated_gpus = sum(int(row.get("dedicated_gpus") or 0) for row in components)
    if dedicated_nodes != 2 or dedicated_gpus != 16:
        raise ValueError("campaign exceeds or understates the two-node/16-GPU cap")
    for row in components:
        if row.get("priority_class") != HIGH_PRIORITY_CLASS:
            raise ValueError("campaign component is not fleet-train-high")
        if not isinstance(row.get("source_ranks"), list) or not row["source_ranks"]:
            raise ValueError("campaign component lacks an explicit source-rank partition")
        fragments = row.get("execution_fragments")
        if fragments is None:
            fragments = [row]
        if not isinstance(fragments, list) or not fragments:
            raise ValueError("campaign component lacks execution fragments")
        fragment_cells: list[tuple[int, int]] = []
        for fragment in fragments:
            digest = fragment.get("plan_sha256")
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise ValueError("campaign component lacks an immutable plan digest")
            fragment_cells.extend(_fragment_cells(fragment, row))
        expected_cells = {
            (int(rank), attempt)
            for rank in row["source_ranks"]
            for attempt in range(1, 5)
        }
        if set(fragment_cells) != expected_cells or len(fragment_cells) != len(
            set(fragment_cells)
        ):
            raise ValueError("campaign execution fragments do not exactly partition component")


def _mapping_selector_cells(selector: dict[str, Any]) -> set[tuple[int, int]]:
    ranks = selector.get("source_ranks")
    cells = selector.get("cells")
    if (ranks is None) == (cells is None):
        raise ValueError("mapping selector must declare source_ranks xor cells")
    if ranks is not None:
        if not isinstance(ranks, list) or not ranks or len(ranks) != len(set(ranks)):
            raise ValueError("mapping source-rank selector is invalid")
        return {(int(rank), attempt) for rank in ranks for attempt in range(1, 5)}
    if not isinstance(cells, list) or not cells:
        raise ValueError("mapping cell selector is invalid")
    result: set[tuple[int, int]] = set()
    for row in cells:
        attempts = row.get("attempts")
        rank = int(row.get("source_rank") or 0)
        if (
            rank <= 0
            or not isinstance(attempts, list)
            or not attempts
            or len(attempts) != len(set(attempts))
            or any(int(attempt) not in range(1, 5) for attempt in attempts)
        ):
            raise ValueError("mapping cell selector is invalid")
        result.update((rank, int(attempt)) for attempt in attempts)
    return result


def validate_scientific_mapping(
    mapping: dict[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    """Validate a blocked, score-blind denominator before executable release.

    This contract deliberately permits unresolved execution bindings, but it
    still resolves every scientific task identity and all four attempt slots.
    It therefore cannot initialize a ledger or authorize a workload.
    """

    if mapping.get("schema_version") != SCIENTIFIC_MAPPING_SCHEMA:
        raise ValueError("unsupported scientific mapping schema")
    if mapping.get("mapping_sha256") != digest_without(mapping, "mapping_sha256"):
        raise ValueError("scientific mapping digest mismatch")
    if (
        mapping.get("release_status") != "blocked_pending_execution_fragment_binding"
        or mapping.get("launch_authorized") is not False
        or mapping.get("ledger_initialization_authorized") is not False
    ):
        raise ValueError("scientific mapping must remain blocked")
    if mapping.get("expected") != {
        "models": {"qwen3.8-27b": 200, "glm-5.3": 400},
        "total_tasks": 150,
        "total_cells": 600,
        "dedicated_nodes": 2,
        "dedicated_gpus": 16,
    }:
        raise ValueError("scientific mapping denominator drifted")
    components = mapping.get("components") or []
    expected_components = {
        "qwen-hosted-retained-source4": ("qwen3.8-27b", 1, 0, 0),
        "qwen-hosted-primary49": ("qwen3.8-27b", 49, 0, 0),
        "glm-hosted-primary46": ("glm-5.3", 46, 0, 0),
        "glm-dedicated-a-v5-primary27": ("glm-5.3", 27, 1, 8),
        "glm-dedicated-b-v5-primary27": ("glm-5.3", 27, 1, 8),
    }
    if {row.get("id") for row in components} != set(expected_components):
        raise ValueError("scientific mapping component set drifted")
    scientific: list[dict[str, Any]] = []
    unresolved: set[tuple[str, int, int]] = set()
    for component in components:
        component_id = component["id"]
        model, expected_tasks, nodes, gpus = expected_components[component_id]
        if (
            component.get("model") != model
            or component.get("priority_class") != HIGH_PRIORITY_CLASS
            or component.get("dedicated_nodes") != nodes
            or component.get("dedicated_gpus") != gpus
            or component.get("owned_attempts_per_task") != [1, 2, 3, 4]
        ):
            raise ValueError("scientific mapping component treatment drifted")
        task_rows: dict[int, dict[str, Any]] = {}
        for source in component.get("scientific_task_sources") or []:
            kind = source.get("kind")
            if kind == "plan":
                path = root / str(source.get("repo_plan_path") or "")
                plan = read_object(path)
                if (
                    plan.get("plan_sha256") != source.get("plan_sha256")
                    or plan.get("plan_sha256") != digest_without(plan, "plan_sha256")
                ):
                    raise ValueError("scientific task-source plan drifted")
                plan_tasks = {int(row["source_rank"]): row for row in plan.get("tasks") or []}
                selected = {int(rank) for rank in source.get("source_ranks") or []}
                if not selected or not selected <= set(plan_tasks):
                    raise ValueError("scientific task-source ranks drifted")
                rows = [plan_tasks[rank] for rank in selected]
            elif kind == "inline":
                receipt = source.get("authority_receipt_sha256")
                if not isinstance(receipt, str) or not receipt.startswith("sha256:"):
                    raise ValueError("inline scientific task lacks immutable authority")
                rows = source.get("tasks") or []
            else:
                raise ValueError("scientific task-source kind is invalid")
            for row in rows:
                rank = int(row.get("source_rank") or 0)
                task = row.get("task") or {}
                if (
                    rank <= 0
                    or rank in task_rows
                    or not isinstance(task.get("key"), str)
                    or not task["key"]
                    or not isinstance(task.get("version_id"), str)
                    or not task["version_id"]
                ):
                    raise ValueError("scientific task identity is invalid or duplicated")
                task_rows[rank] = row
        if len(task_rows) != expected_tasks:
            raise ValueError("scientific mapping component task count drifted")
        expected_cells = {(rank, attempt) for rank in task_rows for attempt in range(1, 5)}
        owned_cells: set[tuple[int, int]] = set()
        for binding in component.get("attempt_ownership") or []:
            cells = _mapping_selector_cells(binding)
            if owned_cells & cells:
                raise ValueError("scientific mapping attempt ownership overlaps")
            state = binding.get("state")
            if state == "resolved":
                if not isinstance(binding.get("plan_sha256"), str) or not binding[
                    "plan_sha256"
                ].startswith("sha256:"):
                    raise ValueError("resolved attempt ownership lacks plan binding")
            elif state == "unresolved":
                if binding.get("plan_sha256") is not None or not binding.get("blocked_reason"):
                    raise ValueError("unresolved attempt ownership is not fail closed")
                unresolved.update((component_id, rank, attempt) for rank, attempt in cells)
            else:
                raise ValueError("attempt ownership state is invalid")
            owned_cells |= cells
        if owned_cells != expected_cells:
            raise ValueError("scientific mapping attempt ownership has gaps")
        scientific.extend(
            {
                "model": model,
                "component_id": component_id,
                "source_rank": rank,
                "task_key": row["task"]["key"],
                "task_version_id": row["task"]["version_id"],
            }
            for rank, row in task_rows.items()
        )
    identities = [(row["model"], row["task_version_id"]) for row in scientific]
    if len(identities) != len(set(identities)):
        raise ValueError("scientific mapping repeats a model/task identity")
    counts = Counter(row["model"] for row in scientific)
    if {key: value * 4 for key, value in counts.items()} != mapping["expected"]["models"]:
        raise ValueError("scientific mapping model count drifted")
    replacements = mapping.get("replacement_mappings") or []
    replacement_targets = [
        (row.get("model"), row.get("component_id"), int(row.get("replacement_source_rank") or 0))
        for row in replacements
    ]
    if len(replacement_targets) != len(set(replacement_targets)):
        raise ValueError("scientific replacement target duplicated")
    primary = {(row["model"], row["component_id"], row["source_rank"]) for row in scientific}
    for row, target in zip(replacements, replacement_targets, strict=True):
        if (
            target not in primary
            or not isinstance(row.get("selection_receipt_sha256"), str)
            or not row["selection_receipt_sha256"].startswith("sha256:")
            or int(row.get("excluded_source_rank") or 0) <= 0
            or (row["model"], row["component_id"], int(row["excluded_source_rank"])) in primary
        ):
            raise ValueError("scientific replacement mapping is invalid")
    policy = mapping.get("legacy_import_semantics") or {}
    if policy != {
        "accepted_or_reconciled": "import_terminal_accepted_never_repeat",
        "active_claim": "import_claimed_same_run_may_terminalize_without_reclaim",
        "excluded_attrition": "append_only_nonprimary_never_repeat",
        "unclaimed": "admit_only_after_exhaustive_api_sfs_kubernetes_zero_inventory",
        "scores_or_content_read": False,
    }:
        raise ValueError("legacy import semantics drifted")
    return {
        "tasks": len(scientific),
        "cells": len(scientific) * 4,
        "model_cells": dict(sorted((key, value * 4) for key, value in counts.items())),
        "unresolved_cells": len(unresolved),
        "unresolved_components": sorted({row[0] for row in unresolved}),
    }


def _plan_path(component: dict[str, Any], cluster: bool) -> Path:
    field = "cluster_plan_path" if cluster else "repo_plan_path"
    path = component.get(field)
    if not isinstance(path, str) or not path:
        raise ValueError(f"component lacks {field}")
    return Path(path)


def _fragment_cells(
    fragment: dict[str, Any], component: dict[str, Any]
) -> list[tuple[int, int]]:
    selected = fragment.get("cells")
    if selected is None:
        ranks = fragment.get("source_ranks", component.get("source_ranks"))
        if not isinstance(ranks, list) or not ranks:
            raise ValueError("campaign execution fragment lacks source ranks or cells")
        return [(int(rank), attempt) for rank in ranks for attempt in range(1, 5)]
    if not isinstance(selected, list) or not selected:
        raise ValueError("campaign execution fragment cells are invalid")
    cells: list[tuple[int, int]] = []
    for row in selected:
        rank = int(row.get("source_rank") or 0)
        attempts = row.get("attempts")
        if (
            rank <= 0
            or not isinstance(attempts, list)
            or not attempts
            or any(int(attempt) not in range(1, 5) for attempt in attempts)
        ):
            raise ValueError("campaign execution fragment cell selector is invalid")
        cells.extend((rank, int(attempt)) for attempt in attempts)
    return cells


def build_universe(campaign: dict[str, Any], *, cluster: bool) -> dict[str, Any]:
    validate_campaign(campaign)
    cells: list[dict[str, Any]] = []
    component_counts: dict[str, int] = {}
    for component in campaign["components"]:
        component_cell_count = 0
        fragments = component.get("execution_fragments") or [component]
        for fragment in fragments:
            plan = read_object(_plan_path(fragment, cluster))
            if plan.get("plan_sha256") != fragment["plan_sha256"]:
                raise ValueError(f"plan digest binding drifted for {component['id']}")
            if plan.get("plan_sha256") != digest_without(plan, "plan_sha256"):
                raise ValueError(f"plan self-digest drifted for {component['id']}")
            selected_cells = set(_fragment_cells(fragment, component))
            selected = {rank for rank, _ in selected_cells}
            tasks = {
                int(task["source_rank"]): task
                for task in plan.get("tasks") or []
                if int(task["source_rank"]) in selected
            }
            if set(tasks) != selected:
                raise ValueError(
                    f"component source-rank selection drifted for {component['id']}"
                )
            attempts = [
                attempt
                for attempt in plan.get("attempts") or []
                if int(attempt["source_rank"]) in selected
            ]
            actual_cells = {
                (int(attempt["source_rank"]), int(attempt["attempt"]))
                for attempt in attempts
                if (int(attempt["source_rank"]), int(attempt["attempt"]))
                in selected_cells
            }
            attempts = [
                attempt
                for attempt in attempts
                if (int(attempt["source_rank"]), int(attempt["attempt"]))
                in selected_cells
            ]
            if actual_cells != selected_cells or len(attempts) != len(selected_cells):
                raise ValueError(
                    f"component is not an exact pass@4 Cartesian block: {component['id']}"
                )
            for attempt in attempts:
                source_rank = int(attempt["source_rank"])
                task = tasks[source_rank]
                material = {
                    "model": component["model"],
                    "serving_block": component["serving_block"],
                    "task_version_id": task["task"]["version_id"],
                    "attempt": int(attempt["attempt"]),
                }
                cell = {
                    "cell_id": sha256(canonical_json(material)),
                    **material,
                    "task_key": task["task"]["key"],
                    "source_rank": source_rank,
                    "component_id": component["id"],
                    "source_plan_sha256": fragment["plan_sha256"],
                    "source_run_id": attempt["run_id"],
                }
                cells.append(cell)
            component_cell_count += len(attempts)
        component_counts[component["id"]] = component_cell_count
    cell_ids = [cell["cell_id"] for cell in cells]
    scientific_keys = [(cell["model"], cell["task_version_id"], cell["attempt"]) for cell in cells]
    if len(cell_ids) != len(set(cell_ids)) or len(scientific_keys) != len(set(scientific_keys)):
        raise ValueError("campaign contains duplicate experimental cells")
    model_counts = Counter(cell["model"] for cell in cells)
    expected = campaign["expected"]
    if len(cells) != expected["total_cells"] or dict(model_counts) != expected["models"]:
        raise ValueError("campaign cell universe does not match the frozen denominator")
    universe = {
        "schema_version": UNIVERSE_SCHEMA,
        "campaign_sha256": campaign["campaign_sha256"],
        "cell_count": len(cells),
        "model_counts": dict(sorted(model_counts.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "cells": sorted(
            cells,
            key=lambda row: (
                row["model"],
                row["serving_block"],
                row["source_rank"],
                row["attempt"],
            ),
        ),
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    universe["universe_sha256"] = digest_without(universe, "universe_sha256")
    return universe


def initialize_ledger(
    campaign: dict[str, Any], ledger_root: Path, *, cluster: bool
) -> dict[str, Any]:
    universe = build_universe(campaign, cluster=cluster)
    ledger_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in (
        "claims",
        "accepted",
        "quarantine",
        "workers",
        "heartbeats",
        "restart-events",
    ):
        (ledger_root / name).mkdir(mode=0o700)
    write_once(ledger_root / "CAMPAIGN.json", campaign)
    write_once(ledger_root / "UNIVERSE.json", universe)
    return universe


def _universe_index(ledger_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    universe = read_object(ledger_root / "UNIVERSE.json")
    if universe.get("schema_version") != UNIVERSE_SCHEMA:
        raise RuntimeError("ledger universe schema drifted")
    if universe.get("universe_sha256") != digest_without(universe, "universe_sha256"):
        raise RuntimeError("ledger universe digest drifted")
    cells = universe.get("cells") or []
    index = {cell["cell_id"]: cell for cell in cells}
    if len(index) != universe.get("cell_count"):
        raise RuntimeError("ledger universe duplicated")
    return universe, index


def _evidence_digest(evidence: dict[str, Any]) -> str:
    field = next(
        (
            name
            for name in ("receipt_sha256", "event_sha256", "claim_sha256")
            if name in evidence
        ),
        None,
    )
    if field is None or evidence[field] != digest_without(evidence, field):
        raise ValueError("legacy evidence is not self-digested")
    return str(evidence[field])


def _validate_import_entry(
    cell: dict[str, Any], entry: dict[str, Any]
) -> dict[str, Any]:
    state = entry.get("state")
    if state not in {"claimed", "accepted", "quarantined"}:
        raise ValueError("legacy import state is invalid")
    exact = {
        "task_version_id": cell["task_version_id"],
        "attempt": cell["attempt"],
        "source_rank": cell["source_rank"],
        "component_id": cell["component_id"],
        "source_plan_sha256": cell["source_plan_sha256"],
        "run_id": cell["source_run_id"],
    }
    if any(entry.get(field) != value for field, value in exact.items()):
        raise ValueError("legacy import cell binding drifted")
    _uuidish(entry.get("worker_uid"))
    if not isinstance(entry.get("worker_name"), str) or not entry["worker_name"]:
        raise ValueError("legacy import worker identity is invalid")
    evidence_path = Path(str(entry.get("evidence_path") or ""))
    evidence = read_object(evidence_path)
    if (
        evidence.get("scores_included") is True
        or evidence.get("prompts_or_traces_included") is True
        or "score" in evidence
        or "reward" in evidence
        or (
            state != "claimed"
            and (
                evidence.get("scores_included") is not False
                or evidence.get("prompts_or_traces_included") is not False
            )
        )
    ):
        raise ValueError("legacy evidence is not score/content blind")
    evidence_digest = _evidence_digest(evidence)
    if entry.get("evidence_sha256") != evidence_digest:
        raise ValueError("legacy import evidence digest drifted")
    for field, value in (
        ("plan_sha256", cell["source_plan_sha256"]),
        ("run_id", cell["source_run_id"]),
        ("task_version_id", cell["task_version_id"]),
        ("attempt", cell["attempt"]),
        ("source_rank", cell["source_rank"]),
    ):
        if evidence.get(field) not in {None, value}:
            raise ValueError(f"legacy evidence contradicts {field}")
    if state == "accepted" and not (
        evidence.get("accepted") is True
        or evidence.get("credited") is True
        or evidence.get("classification") == "RECONCILED_ACCEPTED"
    ):
        raise ValueError("legacy accepted evidence lacks authoritative credit")
    if state == "quarantined" and evidence.get("retry_allowed") is not False:
        raise ValueError("legacy quarantine does not forbid retry")
    if state == "claimed" and evidence.get("run_id") != cell["source_run_id"]:
        raise ValueError("legacy claim evidence lacks the exact run identity")
    return evidence


def import_legacy_evidence(ledger_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Seal the exhaustive pre-ledger inventory without replaying any cell.

    The manifest is the only imported-state file.  All entries and their source
    receipts are validated before the O_EXCL seal is written, so a crash cannot
    leave a partially imported set that looks complete.
    """

    universe, index = _universe_index(ledger_root)
    if (
        manifest.get("schema_version") != LEGACY_IMPORT_SCHEMA
        or manifest.get("import_sha256") != digest_without(manifest, "import_sha256")
        or manifest.get("campaign_sha256") != universe["campaign_sha256"]
        or manifest.get("universe_sha256") != universe["universe_sha256"]
        or manifest.get("complete_inventory") is not True
        or manifest.get("scores_included") is not False
        or manifest.get("prompts_or_traces_included") is not False
    ):
        raise ValueError("legacy import manifest is not an exact complete inventory")
    entries = manifest.get("entries") or []
    if not isinstance(entries, list):
        raise ValueError("legacy import entries must be a list")
    cell_ids = [entry.get("cell_id") for entry in entries]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("legacy import cells duplicated")
    if manifest.get("observed_cell_count") != len(entries):
        raise ValueError("legacy import observed count drifted")
    for entry in entries:
        cell = index.get(entry.get("cell_id"))
        if cell is None:
            raise ValueError("legacy import cell is outside the frozen universe")
        _validate_import_entry(cell, entry)
    write_once(ledger_root / "IMPORT_COMPLETE.json", manifest)
    return manifest


def _imported_entries(ledger_root: Path) -> dict[str, dict[str, Any]]:
    path = ledger_root / "IMPORT_COMPLETE.json"
    if not path.exists():
        raise RuntimeError("authoritative legacy inventory is not sealed")
    manifest = read_object(path)
    universe, index = _universe_index(ledger_root)
    if (
        manifest.get("schema_version") != LEGACY_IMPORT_SCHEMA
        or manifest.get("import_sha256") != digest_without(manifest, "import_sha256")
        or manifest.get("campaign_sha256") != universe["campaign_sha256"]
        or manifest.get("universe_sha256") != universe["universe_sha256"]
        or manifest.get("complete_inventory") is not True
        or manifest.get("scores_included") is not False
        or manifest.get("prompts_or_traces_included") is not False
    ):
        raise RuntimeError("authoritative legacy inventory seal drifted")
    entries = manifest.get("entries") or []
    result = {str(entry["cell_id"]): entry for entry in entries}
    if len(result) != len(entries):
        raise RuntimeError("authoritative legacy inventory duplicated")
    if set(result) - set(index):
        raise RuntimeError("authoritative legacy inventory contains foreign cells")
    return result


def claim_cell(
    ledger_root: Path,
    *,
    cell_id: str,
    worker_name: str,
    worker_uid: str,
    run_id: str,
) -> dict[str, Any]:
    universe, index = _universe_index(ledger_root)
    imported = _imported_entries(ledger_root)
    cell = index.get(cell_id)
    if cell is None:
        raise ValueError("cell is outside the frozen 600-cell universe")
    if cell_id in imported:
        raise RuntimeError("cell already exists in the authoritative legacy inventory")
    worker = read_object(ledger_root / "workers" / f"{worker_name}.json")
    if worker.get("worker_uid") != _uuidish(worker_uid):
        raise ValueError("cell claimant does not match the registered worker UID")
    owned = any(
        partition.get("component_id") == cell["component_id"]
        and cell["source_rank"] in partition.get("source_ranks", [])
        for partition in worker.get("owned_partitions") or []
    )
    if not owned:
        raise ValueError("worker partition does not own this cell")
    prefix = worker.get("run_id_prefix")
    if not isinstance(prefix, str) or not run_id.startswith(prefix):
        raise ValueError("cell run ID is outside the registered worker namespace")
    event = {
        "schema_version": LEDGER_EVENT_SCHEMA,
        "event": "claimed",
        "campaign_sha256": universe["campaign_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "cell_id": cell_id,
        "worker_name": worker_name,
        "worker_uid": worker_uid,
        "run_id": run_id,
        "task_version_id": cell["task_version_id"],
        "attempt": cell["attempt"],
        "scores_included": False,
    }
    event["event_sha256"] = digest_without(event, "event_sha256")
    write_once(ledger_root / "claims" / f"{cell_id}.json", event)
    return event


def record_outcome(
    ledger_root: Path,
    *,
    cell_id: str,
    kind: str,
    evidence_path: Path,
) -> dict[str, Any]:
    if kind not in {"accepted", "quarantine"}:
        raise ValueError("outcome must be accepted or quarantine")
    universe, index = _universe_index(ledger_root)
    cell = index.get(cell_id)
    if cell is None:
        raise ValueError("outcome cell is outside the universe")
    claim_path = ledger_root / "claims" / f"{cell_id}.json"
    if claim_path.exists():
        claim = read_object(claim_path)
    else:
        imported = _imported_entries(ledger_root).get(cell_id)
        if imported is None or imported.get("state") != "claimed":
            raise RuntimeError("outcome lacks an authoritative claim")
        claim = {
            "event_sha256": sha256(
                canonical_json(
                    {
                        "import_sha256": read_object(
                            ledger_root / "IMPORT_COMPLETE.json"
                        )["import_sha256"],
                        "cell_id": cell_id,
                        "run_id": imported["run_id"],
                    }
                )
            ),
            "worker_name": imported["worker_name"],
            "worker_uid": imported["worker_uid"],
        }
    evidence = read_object(evidence_path)
    if (
        evidence.get("scores_included") is not False
        or evidence.get("prompts_or_traces_included") is not False
        or "score" in evidence
        or "reward" in evidence
    ):
        raise ValueError("outcome evidence is not score/content blind")
    receipt_field = next(
        (field for field in ("receipt_sha256", "event_sha256") if field in evidence), None
    )
    if receipt_field is None or evidence[receipt_field] != digest_without(evidence, receipt_field):
        raise ValueError("outcome evidence is not self-digested")
    if evidence.get("task_version_id") not in {None, cell["task_version_id"]}:
        raise ValueError("outcome evidence contradicts the cell task version")
    if evidence.get("attempt") not in {None, cell["attempt"]}:
        raise ValueError("outcome evidence contradicts the cell attempt")
    if kind == "accepted" and not (
        evidence.get("accepted") is True
        or evidence.get("credited") is True
        or evidence.get("classification") == "RECONCILED_ACCEPTED"
    ):
        raise ValueError("accepted outcome lacks authoritative credit evidence")
    if kind == "quarantine" and evidence.get("retry_allowed") is not False:
        raise ValueError("quarantine must explicitly forbid an automatic retry")
    other = "quarantine" if kind == "accepted" else "accepted"
    if (ledger_root / other / f"{cell_id}.json").exists():
        raise RuntimeError("cell already has a contradictory terminal outcome")
    event = {
        "schema_version": LEDGER_EVENT_SCHEMA,
        "event": kind,
        "campaign_sha256": universe["campaign_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "cell_id": cell_id,
        "claim_event_sha256": claim["event_sha256"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence[receipt_field],
        "worker_name": claim["worker_name"],
        "worker_uid": claim["worker_uid"],
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    event["event_sha256"] = digest_without(event, "event_sha256")
    write_once(ledger_root / kind / f"{cell_id}.json", event)
    return event


def register_worker(ledger_root: Path, registration: dict[str, Any]) -> dict[str, Any]:
    universe, _ = _universe_index(ledger_root)
    if registration.get("schema_version") != WORKER_SCHEMA:
        raise ValueError("worker registration schema drifted")
    if registration.get("registration_sha256") != digest_without(
        registration, "registration_sha256"
    ):
        raise ValueError("worker registration digest mismatch")
    if registration.get("campaign_sha256") != universe["campaign_sha256"]:
        raise ValueError("worker registration belongs to another campaign")
    _uuidish(registration.get("worker_uid"))
    if registration.get("namespace") != "fleet-train-jobs":
        raise ValueError("worker must remain in the Fleet training namespace")
    if registration.get("priority_class") != HIGH_PRIORITY_CLASS:
        raise ValueError("worker is not fleet-train-high")
    if registration.get("owned") is not True:
        raise ValueError("supervisor may monitor only explicitly owned workers")
    partitions = registration.get("owned_partitions") or []
    if not isinstance(partitions, list) or not partitions:
        raise ValueError("worker lacks a predeclared task partition")
    component_sources: dict[str, set[int]] = defaultdict(set)
    for cell in read_object(ledger_root / "UNIVERSE.json").get("cells") or []:
        component_sources[cell["component_id"]].add(int(cell["source_rank"]))
    for partition in partitions:
        component = partition.get("component_id")
        ranks = partition.get("source_ranks")
        if (
            component not in component_sources
            or not isinstance(ranks, list)
            or not ranks
            or len(ranks) != len(set(ranks))
            or not {int(rank) for rank in ranks} <= component_sources[component]
        ):
            raise ValueError("worker partition is outside the frozen universe")
    prefix = registration.get("run_id_prefix")
    if not isinstance(prefix, str) or len(prefix) < 20:
        raise ValueError("worker lacks an exact run-ID namespace")
    write_once(ledger_root / "workers" / f"{registration['worker_name']}.json", registration)
    return registration


def status(ledger_root: Path) -> dict[str, Any]:
    universe, index = _universe_index(ledger_root)
    imported = _imported_entries(ledger_root)
    claims = {path.stem for path in (ledger_root / "claims").glob("*.json")}
    accepted = {path.stem for path in (ledger_root / "accepted").glob("*.json")}
    quarantined = {path.stem for path in (ledger_root / "quarantine").glob("*.json")}
    imported_claims = {
        cell_id for cell_id, entry in imported.items() if entry["state"] == "claimed"
    }
    imported_accepted = {
        cell_id for cell_id, entry in imported.items() if entry["state"] == "accepted"
    }
    imported_quarantined = {
        cell_id for cell_id, entry in imported.items() if entry["state"] == "quarantined"
    }
    claims |= imported_claims
    accepted |= imported_accepted
    quarantined |= imported_quarantined
    if (accepted & quarantined) or not (claims | accepted | quarantined) <= set(index):
        raise RuntimeError("ledger contains contradictory or foreign cells")
    rows: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for cell_id, cell in index.items():
        state = (
            "accepted"
            if cell_id in accepted
            else "quarantined"
            if cell_id in quarantined
            else "claimed"
            if cell_id in claims
            else "unclaimed"
        )
        rows[(cell["model"], cell["serving_block"])][state] += 1
    blocks = [
        {
            "model": model,
            "serving_block": block,
            **{
                state: counts[state]
                for state in ("accepted", "quarantined", "claimed", "unclaimed")
            },
            "total": sum(counts.values()),
        }
        for (model, block), counts in sorted(rows.items())
    ]
    return {
        "schema_version": "fleet-score-blind-campaign-status-v1",
        "campaign_sha256": universe["campaign_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "cell_count": universe["cell_count"],
        "blocks": blocks,
        "scores_read": False,
        "prompts_or_traces_read": False,
    }


class KubeReader:
    """Minimal in-cluster Kubernetes reader; exact object names only."""

    def __init__(self) -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST")
        port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443")
        if not host:
            raise RuntimeError("cluster service-account environment is unavailable")
        token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
        ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        self.base = f"https://{host}:{port}"
        self.token = token_path.read_text().strip()
        self.context = ssl.create_default_context(cafile=str(ca_path))

    def get(self, route: str) -> dict[str, Any] | None:
        request = urllib.request.Request(
            self.base + route, headers={"Authorization": f"Bearer {self.token}"}
        )
        try:
            with urllib.request.urlopen(request, context=self.context, timeout=30) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise RuntimeError(f"Kubernetes GET failed with HTTP {exc.code}") from None
        if not isinstance(value, dict):
            raise RuntimeError("Kubernetes object response is not an object")
        return value


def heartbeat_once(ledger_root: Path, kube: KubeReader) -> dict[str, Any]:
    universe, _ = _universe_index(ledger_root)
    workers: list[dict[str, Any]] = []
    for path in sorted((ledger_root / "workers").glob("*.json")):
        registered = read_object(path)
        namespace = registered["namespace"]
        name = registered["worker_name"]
        job = kube.get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}")
        if job is None:
            state = "absent"
            observed_uid = None
            active = succeeded = failed = 0
        else:
            observed_uid = (job.get("metadata") or {}).get("uid")
            if observed_uid != registered["worker_uid"]:
                raise RuntimeError(f"worker UID drifted for {name}")
            spec = ((job.get("spec") or {}).get("template") or {}).get("spec") or {}
            if spec.get("priorityClassName") != HIGH_PRIORITY_CLASS:
                raise RuntimeError(f"worker priority drifted for {name}")
            status_value = job.get("status") or {}
            active = int(status_value.get("active") or 0)
            succeeded = int(status_value.get("succeeded") or 0)
            failed = int(status_value.get("failed") or 0)
            if active:
                state = "active"
            elif succeeded:
                state = "succeeded"
            elif failed:
                state = "failed"
            else:
                state = "pending"
        workers.append(
            {
                "worker_name": name,
                "expected_uid": registered["worker_uid"],
                "observed_uid": observed_uid,
                "state": state,
                "active": active,
                "succeeded": succeeded,
                "failed": failed,
            }
        )
    now = datetime.now(UTC)
    receipt = {
        "schema_version": HEARTBEAT_SCHEMA,
        "campaign_sha256": universe["campaign_sha256"],
        "universe_sha256": universe["universe_sha256"],
        "observed_at": now.isoformat().replace("+00:00", "Z"),
        "workers": workers,
        "ledger_status": status(ledger_root),
        "exact_named_gets_only": True,
        "peer_objects_mutated": False,
        "scores_read": False,
        "prompts_or_traces_read": False,
    }
    receipt["heartbeat_sha256"] = digest_without(receipt, "heartbeat_sha256")
    stamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    write_once(ledger_root / "heartbeats" / f"{stamp}.json", receipt)
    return receipt


def _print_table(snapshot: dict[str, Any]) -> None:
    print("model\tserving_block\taccepted\tquarantined\tclaimed\tunclaimed\ttotal")
    for row in snapshot["blocks"]:
        print(
            "\t".join(
                str(row[key])
                for key in (
                    "model",
                    "serving_block",
                    "accepted",
                    "quarantined",
                    "claimed",
                    "unclaimed",
                    "total",
                )
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--campaign", type=Path, required=True)
    init.add_argument("--ledger-root", type=Path, required=True)
    init.add_argument("--cluster-plans", action="store_true")
    mapping_status = sub.add_parser("mapping-status")
    mapping_status.add_argument("--mapping", type=Path, required=True)
    show = sub.add_parser("status")
    show.add_argument("--ledger-root", type=Path, required=True)
    show.add_argument("--format", choices=("json", "table"), default="table")
    register = sub.add_parser("register-worker")
    register.add_argument("--ledger-root", type=Path, required=True)
    register.add_argument("--registration", type=Path, required=True)
    import_legacy = sub.add_parser("import-legacy")
    import_legacy.add_argument("--ledger-root", type=Path, required=True)
    import_legacy.add_argument("--manifest", type=Path, required=True)
    claim = sub.add_parser("claim")
    claim.add_argument("--ledger-root", type=Path, required=True)
    claim.add_argument("--cell-id", required=True)
    claim.add_argument("--worker-name", required=True)
    claim.add_argument("--worker-uid", required=True)
    claim.add_argument("--run-id", required=True)
    outcome = sub.add_parser("record")
    outcome.add_argument("--ledger-root", type=Path, required=True)
    outcome.add_argument("--cell-id", required=True)
    outcome.add_argument("--kind", choices=("accepted", "quarantine"), required=True)
    outcome.add_argument("--evidence", type=Path, required=True)
    watch = sub.add_parser("watch")
    watch.add_argument("--ledger-root", type=Path, required=True)
    watch.add_argument("--interval-seconds", type=int, default=30)
    watch.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.command == "init":
        campaign = read_object(args.campaign)
        universe = initialize_ledger(campaign, args.ledger_root, cluster=args.cluster_plans)
        result = {
            "initialized": True,
            "campaign_sha256": universe["campaign_sha256"],
            "universe_sha256": universe["universe_sha256"],
            "cell_count": universe["cell_count"],
            "model_counts": universe["model_counts"],
            "component_counts": universe["component_counts"],
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
    elif args.command == "mapping-status":
        mapping = read_object(args.mapping)
        result = {
            "mapping_sha256": mapping["mapping_sha256"],
            "release_status": mapping["release_status"],
            **validate_scientific_mapping(mapping),
            "launch_authorized": False,
            "ledger_initialization_authorized": False,
            "scores_read": False,
            "prompts_or_traces_read": False,
        }
    elif args.command == "status":
        result = status(args.ledger_root)
        if args.format == "table":
            _print_table(result)
            return 0
    elif args.command == "register-worker":
        result = register_worker(args.ledger_root, read_object(args.registration))
    elif args.command == "import-legacy":
        result = import_legacy_evidence(
            args.ledger_root, read_object(args.manifest)
        )
    elif args.command == "claim":
        result = claim_cell(
            args.ledger_root,
            cell_id=args.cell_id,
            worker_name=args.worker_name,
            worker_uid=args.worker_uid,
            run_id=args.run_id,
        )
    elif args.command == "record":
        result = record_outcome(
            args.ledger_root,
            cell_id=args.cell_id,
            kind=args.kind,
            evidence_path=args.evidence,
        )
    else:
        if args.interval_seconds < 10:
            raise ValueError("cluster watch interval must be at least 10 seconds")
        kube = KubeReader()
        while True:
            result = heartbeat_once(args.ledger_root, kube)
            print(canonical_json({"heartbeat_sha256": result["heartbeat_sha256"]}).decode())
            sys.stdout.flush()
            if args.once:
                break
            time.sleep(args.interval_seconds)
        return 0
    print(canonical_json(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
