#!/usr/bin/env python3
"""Create a metadata-only census of Fleet cyber task supply.

This intentionally uses endpoints that do not return task prompts, traces,
answers, flags, scores, or grader source.  Catalog production status and a
verifier attachment are candidate-selection facts, not proof that a task has
run and scored correctly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PUBLIC_API = "https://orchestrator.fleetai.com"
PRIVATE_API = "https://api.internal.fleet-platform.fleetai.com"
FLEET_TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
OTS_CYBER_PROJECT_ID = "63d6fda8-48c4-4726-9ec3-d1028f2c47f5"
OLD_ROSTER = "configs/data/fleet-a62-task-split-v1.json"
STRICT_ROSTER = "configs/data/qwen-blackbox-eligible-v1.json"
STUDY_INVENTORY = "configs/data/qwen-blackbox-study-inventory-v1.json"


class ReadClient:
    def __init__(self, api_key: str, *, timeout: int = 90):
        self.api_key = api_key
        self.timeout = timeout

    def get(self, base: str, path: str) -> dict[str, Any]:
        url = base.rstrip("/") + path
        for attempt in range(5):
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.load(response)
                if not isinstance(value, dict):
                    raise RuntimeError("metadata endpoint returned non-object JSON")
                return value
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"metadata GET failed with HTTP {error.code}") from None
            except urllib.error.URLError:
                pass
            if attempt < 4:
                time.sleep(min(8, 0.5 * (2**attempt)) + random.random() * 0.2)
        raise RuntimeError("metadata GET failed after five attempts")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def source_shape(key: str) -> tuple[str, str]:
    suffixes = {
        "--blackbox_ctf_v1": "blackbox",
        "--whitebox_patch_v1": "whitebox",
        "--combined_ctf_patch_v1": "combined",
    }
    for suffix, shape in suffixes.items():
        if key.endswith(suffix):
            return shape, key[: -len(suffix)]
    return "unsuffixed", key


def task_shape(key: str) -> str:
    lowered = key.lower()
    if "whitebox" in lowered:
        return "whitebox"
    if "combined_ctf_patch" in lowered:
        return "combined"
    if "blackbox" in lowered:
        return "blackbox"
    return "unrecognized"


def registry_artifacts(
    client: ReadClient, *, kind: str, labels: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    declared_totals: set[int] = set()
    while True:
        query: list[tuple[str, str | int]] = [
            ("kind", kind),
            ("limit", 200),
            ("offset", offset),
        ]
        query.extend(("label", f"{key}:{value}") for key, value in (labels or {}).items())
        page = client.get(PRIVATE_API, "/v1/registry/artifacts?" + urllib.parse.urlencode(query))
        batch = page.get("items")
        if not isinstance(batch, list):
            raise RuntimeError("Registry page omitted items[]")
        declared_totals.add(int(page.get("total") or 0))
        rows.extend(row for row in batch if isinstance(row, dict))
        offset += len(batch)
        if not batch or offset >= int(page.get("total") or 0):
            break
    keys = [row.get("key") for row in rows]
    if any(not isinstance(key, str) or not key for key in keys):
        raise RuntimeError("Registry page contained an invalid key")
    if len(keys) != len(set(keys)) or len(declared_totals) != 1:
        raise RuntimeError("Registry changed during pagination; rerun the census")
    if len(rows) != next(iter(declared_totals)):
        raise RuntimeError("Registry pagination was incomplete")
    return rows


def current_version_labels(client: ReadClient, key: str) -> tuple[dict[str, Any], int]:
    quoted = urllib.parse.quote(key, safe="/")
    page = client.get(PRIVATE_API, f"/v1/registry/artifacts/{quoted}/versions?limit=1&summary=true")
    versions = page.get("versions") or page.get("items") or []
    if not isinstance(versions, list) or not versions:
        raise RuntimeError("Registry artifact has no selected version summary")
    labels = versions[0].get("labels") or {}
    if not isinstance(labels, dict):
        raise RuntimeError("Registry version labels are invalid")
    return labels, int(page.get("total") or 0)


def catalog_status(client: ReadClient, task_id: str) -> dict[str, Any]:
    quoted = urllib.parse.quote(task_id, safe="")
    value = client.get(PRIVATE_API, f"/v1/pipeline/tasks/{quoted}/status")
    safe = {
        "task_id": value.get("eval_task_id"),
        "current_version_id": value.get("current_version_id"),
        "lifecycle_status": value.get("lifecycle_status"),
        "instance_status": value.get("instance_status"),
        "verifier_attached": value.get("verifier_attached"),
    }
    if safe["task_id"] != task_id or not isinstance(safe["current_version_id"], str):
        raise RuntimeError("catalog status identity was incomplete")
    if not isinstance(safe["verifier_attached"], bool):
        raise RuntimeError("catalog status verifier field was invalid")
    return safe


def task_quality(client: ReadClient) -> list[dict[str, Any]]:
    page = client.get(
        PRIVATE_API,
        f"/v1/qa/projects/{OTS_CYBER_PROJECT_ID}/task-quality",
    )
    rows = page.get("tasks")
    if not isinstance(rows, list):
        raise RuntimeError("project task-quality response omitted tasks[]")
    projected = []
    allowed_statuses = {"broken_task", "agent_failure", "clean", "not_analyzed"}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("project task-quality response contained an invalid row")
        value = {
            "task_id": row.get("eval_task_id"),
            "task_version_id": row.get("eval_task_version_id"),
            "task_key": row.get("task_key"),
            "qa_status": row.get("status"),
        }
        if (
            not all(isinstance(value[key], str) and value[key] for key in value)
            or value["qa_status"] not in allowed_statuses
        ):
            raise RuntimeError("project task-quality identity or status was invalid")
        value["task_shape"] = task_shape(value["task_key"])
        projected.append(value)
    projected.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    if len(projected) != len({row["task_id"] for row in projected}):
        raise RuntimeError("project task-quality response duplicated a task")
    return projected


def parallel_map(function: Any, values: list[Any], workers: int) -> list[Any]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, values))


def load_version_ids(path: Path, field: str) -> set[str]:
    value = json.loads(path.read_text())
    rows = value.get(field)
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} omitted {field}[]")
    ids = {row.get("task_version_id") for row in rows if isinstance(row, dict)}
    if None in ids or len(ids) != len(rows):
        raise RuntimeError(f"{path} has missing or duplicate task-version identities")
    return ids


def load_rows_by_version(path: Path, field: str) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    rows = value.get(field)
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} omitted {field}[]")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_version_id"), str):
            raise RuntimeError(f"{path} has an invalid task identity")
        if row["task_version_id"] in indexed:
            raise RuntimeError(f"{path} duplicated a task version")
        indexed[row["task_version_id"]] = row
    return indexed


def load_study_metadata(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    rows = value.get("tasks")
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} omitted tasks[]")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_version_id"), str):
            raise RuntimeError(f"{path} has an invalid task identity")
        version_id = row["task_version_id"]
        if version_id in indexed:
            raise RuntimeError(f"{path} duplicated a task version")
        taxonomy = row.get("taxonomy")
        if not isinstance(taxonomy, dict):
            raise RuntimeError(f"{path} omitted reviewed taxonomy")
        indexed[version_id] = row
    return indexed


def enrich_receipt_proven_row(
    row: dict[str, Any],
    *,
    reviewed: dict[str, dict[str, Any]],
    strict_sha256: str,
    inventory_sha256: str,
) -> dict[str, Any]:
    source = reviewed.get(row["task_version_id"])
    if source is None:
        raise RuntimeError("receipt-proven task omitted from reviewed study inventory")

    def taxonomy_value(field: str) -> Any:
        evidence = source["taxonomy"].get(field)
        if not isinstance(evidence, dict) or evidence.get("status") != "verified":
            raise RuntimeError(f"receipt-proven task lacks verified {field} metadata")
        return evidence.get("value")

    return {
        **row,
        "lineage": {
            "lineage_key": source.get("lineage_id"),
            "application": taxonomy_value("application"),
            "environment": taxonomy_value("environment"),
            "task_family": taxonomy_value("task_family"),
            "vulnerability_family": taxonomy_value("vulnerability_family"),
            "difficulty": taxonomy_value("difficulty"),
        },
        "provenance": {
            "evidence_class": "prior_exact_execution_receipt_current_not_known_broken",
            "strict_filter_path": STRICT_ROSTER,
            "strict_filter_sha256": strict_sha256,
            "reviewed_metadata_path": STUDY_INVENTORY,
            "reviewed_metadata_sha256": inventory_sha256,
            "certification": source.get("certification"),
            "exposure": source.get("exposure"),
        },
    }


def canonical_execution_outcome(
    client: ReadClient,
    *,
    task_key: str,
    session_id: str,
    verifier_execution_id: str,
) -> str:
    """Classify one receipt-bound session without retaining its score or content."""
    offset = 0
    while True:
        query = urllib.parse.urlencode({"task_key": task_key, "limit": 500, "offset": offset})
        page = client.get(PUBLIC_API, f"/v1/sessions?{query}")
        sessions = page.get("sessions")
        if not isinstance(sessions, list):
            raise RuntimeError("session-summary response omitted sessions[]")
        for session in sessions:
            if not isinstance(session, dict) or session.get("session_id") != session_id:
                continue
            verifier = session.get("verifier_execution")
            score = verifier.get("score") if isinstance(verifier, dict) else None
            if (
                session.get("task_key") != task_key
                or session.get("status") != "completed"
                or not session.get("ended_at")
                or not isinstance(verifier, dict)
                or verifier.get("id") != verifier_execution_id
                or verifier.get("success") is not True
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RuntimeError("receipt-bound session summary is not a healthy finite outcome")
            return "proven_success" if float(score) >= 1.0 else "proven_model_failure"
        if not page.get("has_more"):
            raise RuntimeError("receipt-bound session is absent from the safe summary route")
        if not sessions:
            raise RuntimeError("session-summary pagination stopped making progress")
        offset += len(sessions)


def training_coverage(
    client: ReadClient,
    *,
    all_rows: list[dict[str, Any]],
    filtered_rows: list[dict[str, Any]],
    strict_rows: dict[str, dict[str, Any]],
    workers: int,
    all_manifest_sha256: str,
    filtered_manifest_sha256: str,
) -> dict[str, Any]:
    def classify(row: dict[str, Any]) -> tuple[str, str]:
        strict = strict_rows.get(row["task_version_id"])
        evidence = strict.get("execution_evidence") if isinstance(strict, dict) else None
        if not isinstance(evidence, dict):
            raise RuntimeError("filtered task omitted strict execution evidence")
        outcome = canonical_execution_outcome(
            client,
            task_key=row["task_key"],
            session_id=evidence["session_id"],
            verifier_execution_id=evidence["verifier_execution_id"],
        )
        return row["task_key"], outcome

    receipt_rows = [row for row in all_rows if row["task_version_id"] in strict_rows]
    outcomes = dict(parallel_map(classify, receipt_rows, workers))
    successes = sorted(key for key, outcome in outcomes.items() if outcome == "proven_success")
    failures = sorted(key for key, outcome in outcomes.items() if outcome == "proven_model_failure")
    all_keys = {row["task_key"] for row in all_rows}
    filtered_keys = {row["task_key"] for row in filtered_rows}
    receipt_keys = {row["task_key"] for row in receipt_rows}
    if (
        len(all_keys) != len(all_rows)
        or len(filtered_keys) != len(filtered_rows)
        or len(receipt_keys) != len(receipt_rows)
        or set(successes) | set(failures) != receipt_keys
        or set(successes) & set(failures)
    ):
        raise RuntimeError("training-coverage task-key partition is inconsistent")
    high_quality_successes = sorted(filtered_keys & set(successes))
    high_quality_failures = sorted(filtered_keys & set(failures))
    excluded_successes = sorted(set(successes) - filtered_keys)
    excluded_failures = sorted(set(failures) - filtered_keys)
    without_receipt = sorted(all_keys - receipt_keys)
    value = {
        "schema": "fleet_current_blackbox_training_coverage_v1",
        "observed_through": utc_now(),
        "purpose": "task-key-only successful-demonstration coverage for training planning",
        "source_manifests": {
            "all_current_blackbox_sha256": all_manifest_sha256,
            "filtered_current_high_quality_sha256": filtered_manifest_sha256,
        },
        "counts": {
            "current_blackbox_task_keys": len(all_keys),
            "current_exact_receipt_task_keys": len(receipt_keys),
            "current_exact_success_proven_task_keys": len(successes),
            "current_exact_canonical_failure_task_keys": len(failures),
            "high_quality_exact_success_proven_task_keys": len(high_quality_successes),
            "high_quality_exact_canonical_failure_task_keys": len(high_quality_failures),
            "excluded_exact_success_proven_task_keys": len(excluded_successes),
            "excluded_exact_canonical_failure_task_keys": len(excluded_failures),
            "no_exact_receipt_in_this_refresh_task_keys": len(without_receipt),
            "no_exact_success_proven_task_keys": len(failures) + len(without_receipt),
            "proven_failure_only_task_keys": None,
        },
        "current_exact_success_proven_task_keys": successes,
        "current_exact_canonical_failure_task_keys": failures,
        "high_quality_exact_success_proven_task_keys": high_quality_successes,
        "high_quality_exact_canonical_failure_task_keys": high_quality_failures,
        "excluded_exact_success_proven_task_keys": excluded_successes,
        "excluded_exact_canonical_failure_task_keys": excluded_failures,
        "no_exact_receipt_in_this_refresh_task_keys": without_receipt,
        "interpretation": {
            "success": (
                "the exact current version's receipt-bound session is completed, has a "
                "healthy verifier execution, and meets the repository's full-success threshold"
            ),
            "canonical_failure": (
                "the exact current version's receipt-bound session is a genuine model failure; "
                "this does not prove that every other attempt on the task failed"
            ),
            "high_quality": "passes the conservative 75-task current quality filter",
            "excluded": (
                "has an exact historical receipt but is outside the 75-task set because the "
                "current quality review marks it broken"
            ),
            "missing_receipt": (
                "this refresh has no exact current-version execution receipt; it does not prove "
                "that the task was never executed"
            ),
            "failure_only": (
                "not computed because the safe session-summary route does not expose exact task "
                "versions, so it cannot exclude a different successful attempt on the same key"
            ),
        },
        "numeric_scores_persisted": False,
        "session_ids_persisted": False,
        "private_content_persisted": False,
    }
    value["sha256"] = digest(value)
    return value


def count(values: list[Any]) -> dict[str, int]:
    projected = ("<missing>" if value is None else str(value) for value in values)
    return dict(sorted(Counter(projected).items()))


def build_snapshot(
    client: ReadClient,
    *,
    repo_root: Path,
    registry_cutoff: str,
    workers: int,
) -> dict[str, Any]:
    started_at = utc_now()
    account = client.get(PUBLIC_API, "/v1/account")
    if account.get("team_id") != FLEET_TEAM_ID or account.get("team_name") != "fleet":
        raise RuntimeError("FLEET_API_KEY does not resolve to the Fleet team")

    membership_path = f"/v1/projects/{OTS_CYBER_PROJECT_ID}/tasks"
    membership = client.get(PRIVATE_API, membership_path)
    task_ids = membership.get("taskIds")
    if not isinstance(task_ids, list) or len(task_ids) != len(set(task_ids)):
        raise RuntimeError("OTS Cyber membership was missing or duplicated")
    statuses = parallel_map(lambda task_id: catalog_status(client, task_id), task_ids, workers)
    membership_after = client.get(PRIVATE_API, membership_path).get("taskIds")
    if task_ids != membership_after:
        raise RuntimeError("OTS Cyber membership changed during the census; rerun it")
    statuses.sort(key=lambda row: (row["task_id"], row["current_version_id"]))

    quality_rows = task_quality(client)
    production_rows = {
        row["task_id"]: row for row in statuses if row["lifecycle_status"] == "production"
    }
    if set(production_rows) != {row["task_id"] for row in quality_rows}:
        raise RuntimeError("task-quality rows do not equal current production membership")
    if any(
        production_rows[row["task_id"]]["current_version_id"] != row["task_version_id"]
        for row in quality_rows
    ):
        raise RuntimeError("task-quality version differs from catalog current version")

    old_path = repo_root / OLD_ROSTER
    strict_path = repo_root / STRICT_ROSTER
    inventory_path = repo_root / STUDY_INVENTORY
    old_versions = load_version_ids(old_path, "tasks")
    strict_versions = load_version_ids(strict_path, "task_versions")
    reviewed_metadata = load_study_metadata(inventory_path)
    strict_sha256 = file_digest(strict_path)
    inventory_sha256 = file_digest(inventory_path)
    current_versions = {row["current_version_id"] for row in statuses}
    production_versions = {
        row["current_version_id"] for row in statuses if row["lifecycle_status"] == "production"
    }
    blackbox_rows = [row for row in quality_rows if row["task_shape"] == "blackbox"]
    filtered_rows = [
        enrich_receipt_proven_row(
            row,
            reviewed=reviewed_metadata,
            strict_sha256=strict_sha256,
            inventory_sha256=inventory_sha256,
        )
        for row in blackbox_rows
        if row["task_version_id"] in strict_versions and row["qa_status"] != "broken_task"
    ]
    pending_rows = [row for row in blackbox_rows if row["qa_status"] in {"clean", "agent_failure"}]

    source_rows = registry_artifacts(client, kind="cyber_task_graph_source")
    source_tuples = sorted(
        (
            row["id"],
            row["key"],
            row.get("current_version_id"),
            row.get("created_at"),
        )
        for row in source_rows
    )
    recent_sources = [
        row for row in source_rows if str(row.get("created_at") or "") >= registry_cutoff
    ]
    recent_metadata = parallel_map(
        lambda row: current_version_labels(client, row["key"]), recent_sources, workers
    )
    recent_shapes = [source_shape(row["key"])[0] for row in recent_sources]
    naming_roots: dict[str, set[str]] = {}
    for row in recent_sources:
        shape, root = source_shape(row["key"])
        naming_roots.setdefault(root, set()).add(shape)

    eval_rows = registry_artifacts(client, kind="cyber_run", labels={"stage": "eval"})
    eval_metadata = parallel_map(
        lambda row: current_version_labels(client, row["key"]), eval_rows, workers
    )

    finished_at = utc_now()
    snapshot = {
        "schema": "fleet_blackbox_task_inventory_census_v1",
        "observed_from": started_at,
        "observed_through": finished_at,
        "scope": "allowlisted metadata only",
        "account": {"team_id": account["team_id"], "team_name": account["team_name"]},
        "catalog": {
            "project_id": OTS_CYBER_PROJECT_ID,
            "task_count": len(statuses),
            "current_version_count": len(current_versions),
            "lifecycle_status": count([row["lifecycle_status"] for row in statuses]),
            "latest_instance_status": count([row["instance_status"] for row in statuses]),
            "verifier_attached": sum(row["verifier_attached"] for row in statuses),
            "production_current_versions_not_in_old_160": len(production_versions - old_versions),
            "old_160_current_overlap": len(current_versions & old_versions),
            "old_160_production_overlap": len(production_versions & old_versions),
            "strict_89_current_overlap": len(current_versions & strict_versions),
            "strict_89_production_overlap": len(production_versions & strict_versions),
            "task_versions_sha256": digest(statuses),
            "task_versions": statuses,
        },
        "current_production_quality": {
            "source": f"/v1/qa/projects/{OTS_CYBER_PROJECT_ID}/task-quality",
            "task_count": len(quality_rows),
            "task_shape": count([row["task_shape"] for row in quality_rows]),
            "qa_status": count([row["qa_status"] for row in quality_rows]),
            "blackbox_qa_status": count([row["qa_status"] for row in blackbox_rows]),
            "task_versions_sha256": digest(quality_rows),
            "task_versions": quality_rows,
            "meaning": (
                "broken_task is an exclusion signal; clean and agent_failure are "
                "promising candidates but still require exact runtime/scoring/cleanup receipts"
            ),
        },
        "registry_sources": {
            "artifact_kind": "cyber_task_graph_source",
            "source_key_count": len(source_rows),
            "key_shape": count([source_shape(row["key"])[0] for row in source_rows]),
            "safe_identity_tuple_sha256": digest(source_tuples),
            "recent_cutoff": registry_cutoff,
            "recent_source_key_count": len(recent_sources),
            "recent_naming_root_count": len(naming_roots),
            "recent_key_shape": count(recent_shapes),
            "recent_shape_sets": count(
                ["+".join(sorted(shapes)) for shapes in naming_roots.values()]
            ),
            "recent_app_label": count([labels.get("app") for labels, _ in recent_metadata]),
            "recent_status_label": count([labels.get("status") for labels, _ in recent_metadata]),
            "published_versions_across_recent_keys": sum(total for _, total in recent_metadata),
        },
        "registry_eval_evidence": {
            "artifact_kind": "cyber_run",
            "stage": "eval",
            "run_key_count": len(eval_rows),
            "published_version_count": sum(total for _, total in eval_metadata),
            "lifecycle_event": count(
                [labels.get("lifecycle_event") for labels, _ in eval_metadata]
            ),
            "status": count([labels.get("status") for labels, _ in eval_metadata]),
        },
        "frozen_quality_set": {
            "path": STRICT_ROSTER,
            "file_sha256": strict_sha256,
            "exact_task_versions": len(strict_versions),
            "current_catalog_overlap": len(current_versions & strict_versions),
            "production_catalog_overlap": len(production_versions & strict_versions),
            "enlarged_by_this_census": False,
        },
        "refreshed_selections": {
            "all_current_production_blackbox": len(blackbox_rows),
            "conservative_current_receipt_proven": len(filtered_rows),
            "new_qa_cleared_pending_receipts": len(pending_rows),
            "conservative_current_receipt_proven_sha256": digest(filtered_rows),
            "new_qa_cleared_pending_receipts_sha256": digest(pending_rows),
        },
        "old_roster": {
            "path": OLD_ROSTER,
            "file_sha256": file_digest(old_path),
            "exact_task_versions": len(old_versions),
        },
        "interpretation": {
            "candidate_ceiling": (
                "production catalog rows with verifier attachments are candidates, not "
                "proof of a runnable environment and correctly completed grading"
            ),
            "registry_boundary": (
                "source publications and published versions are not accepted platform tasks"
            ),
            "family_boundary": (
                "naming roots are source-product groupings, not independently normalized "
                "vulnerability families"
            ),
            "remaining_join": (
                "accepted mATG entry -> exact catalog task/version -> runtime/environment/"
                "verifier binding -> terminal scored execution and cleanup evidence"
            ),
        },
        "private_content_persisted": False,
    }
    snapshot["sha256"] = digest(snapshot)
    return snapshot


def selection(
    *,
    schema: str,
    purpose: str,
    tasks: list[dict[str, Any]],
    evidence: list[str],
    rows_field: str = "tasks",
) -> dict[str, Any]:
    if rows_field not in {"tasks", "task_versions"}:
        raise ValueError("selection rows must be tasks or task_versions")
    value = {
        "schema": schema,
        "purpose": purpose,
        "task_count": len(tasks),
        "evidence": evidence,
        rows_field: tasks,
        "private_content_persisted": False,
    }
    value["sha256"] = digest(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-blackbox-output", type=Path)
    parser.add_argument("--filtered-output", type=Path)
    parser.add_argument("--pending-output", type=Path)
    parser.add_argument("--training-coverage-output", type=Path)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--registry-cutoff", default="2026-09-11T21:57:00Z")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise SystemExit("--workers must be in 1..32")
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise SystemExit("FLEET_API_KEY is required")
    repo_root = Path(__file__).resolve().parents[1]
    result = build_snapshot(
        ReadClient(api_key),
        repo_root=repo_root,
        registry_cutoff=args.registry_cutoff,
        workers=args.workers,
    )
    atomic_json(args.output, result)
    quality_rows = result["current_production_quality"]["task_versions"]
    blackbox_rows = [row for row in quality_rows if row["task_shape"] == "blackbox"]
    strict_path = repo_root / STRICT_ROSTER
    inventory_path = repo_root / STUDY_INVENTORY
    strict_versions = load_version_ids(strict_path, "task_versions")
    strict_rows = load_rows_by_version(strict_path, "task_versions")
    reviewed_metadata = load_study_metadata(inventory_path)
    filtered_rows = [
        enrich_receipt_proven_row(
            row,
            reviewed=reviewed_metadata,
            strict_sha256=file_digest(strict_path),
            inventory_sha256=file_digest(inventory_path),
        )
        for row in blackbox_rows
        if row["task_version_id"] in strict_versions and row["qa_status"] != "broken_task"
    ]
    pending_rows = [row for row in blackbox_rows if row["qa_status"] in {"clean", "agent_failure"}]
    if args.all_blackbox_output:
        atomic_json(
            args.all_blackbox_output,
            selection(
                schema="fleet_current_production_blackbox_inventory_v1",
                purpose="all current production task keys explicitly identified as blackbox",
                tasks=blackbox_rows,
                evidence=[
                    "current OTS Cyber project membership",
                    "current production lifecycle status",
                    "current exact version and attached verifier",
                    "cross-session QA bucket; not a validity certificate",
                ],
            ),
        )
    if args.filtered_output:
        atomic_json(
            args.filtered_output,
            selection(
                schema="fleet_current_blackbox_receipt_proven_filter_v1",
                purpose="conservative currently selected receipt-proven blackbox task versions",
                tasks=filtered_rows,
                rows_field="task_versions",
                evidence=[
                    "member of the frozen 2026-09-12 strict execution-receipt set",
                    "still the current production catalog version",
                    "current verifier attached",
                    "not currently classified as broken_task by cross-session QA",
                ],
            ),
        )
    if args.pending_output:
        atomic_json(
            args.pending_output,
            selection(
                schema="fleet_current_blackbox_qa_candidates_v1",
                purpose="new current blackbox candidates awaiting exact receipt validation",
                tasks=pending_rows,
                evidence=[
                    "current production catalog version with attached verifier",
                    "cross-session QA classified clean or agent_failure rather than broken_task",
                    "not accepted until runtime, grading, finite-result, and cleanup receipts join",
                ],
            ),
        )
    if args.training_coverage_output:
        if not args.all_blackbox_output or not args.filtered_output:
            raise SystemExit(
                "--training-coverage-output requires --all-blackbox-output and --filtered-output"
            )
        atomic_json(
            args.training_coverage_output,
            training_coverage(
                ReadClient(api_key),
                all_rows=blackbox_rows,
                filtered_rows=filtered_rows,
                strict_rows=strict_rows,
                workers=min(args.workers, 8),
                all_manifest_sha256=file_digest(args.all_blackbox_output),
                filtered_manifest_sha256=file_digest(args.filtered_output),
            ),
        )
    print(
        f"wrote {result['catalog']['task_count']} catalog tasks, "
        f"{result['registry_sources']['source_key_count']} source keys; "
        f"sha256={result['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
