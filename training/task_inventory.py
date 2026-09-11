"""Audit exact Fleet task bindings without persisting task payloads.

The Fleet task response contains private prompt material.  This module keeps it
in memory only and emits an allowlisted, digest-bound projection suitable for a
later execution-receipt join.  A valid current binding is necessary but is not
by itself proof that the environment and grader completed successfully.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet.opencode_self_hosted import bind_task

from .fleet import FleetClient
from .io import atomic_write_json, digest_json, file_sha256

SCHEMA = "cyber_task_current_binding_audit_v1"
ROW_SCHEMA = "cyber_task_current_binding_v1"
EXPECTED_FIELDS = {
    "task_key",
    "task_version_id",
    "task_version",
    "env_key",
    "env_version",
    "environment_version_id",
    "data_key",
    "data_version",
    "split",
    "resolution_authority",
}
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
SAFE_FAILURES = {
    "exact task version has no recorded starting-data binding": "starting_data_absent",
    "live task lacks an immutable runtime seed or verifier version": (
        "runtime_or_verifier_pin_absent"
    ),
    "live task runtime differs from the frozen selection": "runtime_binding_drift",
    "live task key differs from the frozen selection": "task_key_drift",
    "live verifier binding is incomplete": "verifier_binding_incomplete",
    "task data identity is only partially specified": "partial_data_binding",
    "task data identity is invalid": "invalid_data_binding",
    "single-environment task seed binding is ambiguous": "ambiguous_seed_binding",
    "versioned task seed binding is incomplete": "incomplete_seed_binding",
    "legacy and versioned task data bindings disagree": "data_binding_drift",
}


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "sha256": digest_json(value)}


def _source_projection(metadata: Mapping[str, Any]) -> dict[str, Any]:
    subject = metadata.get("cyber_subject")
    if not isinstance(subject, Mapping):
        raise ValueError("cyber_subject_absent")
    raw = subject.get("atom_sources")
    if not isinstance(raw, list) or not raw:
        raise ValueError("atom_source_absent")
    sources = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("atom_source_invalid")
        projected = {key: item.get(key) for key in ("artifact_key", "version_index", "atom_id")}
        if (
            not isinstance(projected["artifact_key"], str)
            or not projected["artifact_key"].startswith("cyber/atoms/")
            or type(projected["version_index"]) is not int
            or projected["version_index"] < 0
            or not isinstance(projected["atom_id"], str)
            or not projected["atom_id"]
        ):
            raise ValueError("atom_source_invalid")
        sources.append(projected)
    if len({(r["artifact_key"], r["version_index"]) for r in sources}) != len(sources):
        raise ValueError("atom_source_duplicate")
    return {
        "source_digest": subject.get("source_digest"),
        "source_locator": subject.get("source_locator"),
        "task_graph_id": subject.get("task_graph_id"),
        "atom_sources": sorted(sources, key=lambda r: (r["artifact_key"], r["version_index"])),
    }


def _taxonomy(
    selected: Mapping[str, Any], metadata: Mapping[str, Any], source: Mapping[str, Any]
) -> dict:
    atoms = source["atom_sources"]
    apps = sorted({row["artifact_key"].split("/", 3)[2] for row in atoms})
    families = sorted(f"{row['artifact_key']}@{row['version_index']}" for row in atoms)
    atom_ids = sorted({row["atom_id"] for row in atoms})
    band = metadata.get("task_graph_band")
    difficulty = band if isinstance(band, str) and band else None
    evidence = digest_json(
        {
            "task_version_id": selected["task_version_id"],
            "environment_version_id": selected["environment_version_id"],
            "source": source,
            "task_graph_band": difficulty,
        }
    )

    def verified(value: str | list[str]) -> dict:
        return {"status": "verified", "value": value, "evidence_sha256": evidence}

    return {
        "application": verified(apps),
        "environment": verified(selected["env_key"]),
        # Exact atom locators are the strongest reviewed family identity in the
        # task response.  They are not silently relabelled as CWE classes.
        "task_family": verified("+".join(families)),
        "vulnerability_family": verified(atom_ids),
        "difficulty": verified(difficulty) if difficulty else {"status": "missing"},
    }


def _validate_selection(rows: list[dict[str, Any]]) -> None:
    seen = set()
    if not rows:
        raise ValueError("nonempty exact task selection required")
    for row in rows:
        if set(row) != EXPECTED_FIELDS:
            raise ValueError("task selection fields drifted")
        identity = (row["task_key"], row["task_version_id"])
        if (
            identity in seen
            or not isinstance(row["task_key"], str)
            or not row["task_key"]
            or not isinstance(row["task_version_id"], str)
            or UUID.fullmatch(row["task_version_id"]) is None
            or not isinstance(row["environment_version_id"], str)
            or UUID.fullmatch(row["environment_version_id"]) is None
        ):
            raise ValueError("missing, duplicate or invalid exact task identity")
        seen.add(identity)


def audit_current_bindings(
    rows: list[dict[str, Any]],
    fetch: Callable[[str, str], dict[str, Any]],
    *,
    observed_at: str,
    workers: int = 16,
) -> dict[str, Any]:
    """Fetch and sanitize exact task versions in deterministic input order."""
    _validate_selection(rows)
    if not isinstance(observed_at, str) or not observed_at.endswith("Z"):
        raise ValueError("UTC observed_at ending in Z required")
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be in 1..32")

    def one(selected: dict[str, Any]) -> dict[str, Any]:
        identity = {key: selected[key] for key in ("task_key", "task_version_id", "split")}
        try:
            response = fetch(selected["task_key"], selected["task_version_id"])
            task, environment, verifier = bind_task(response, selected)
            metadata = response.get("metadata") or {}
            source = _source_projection(metadata)
            value = {
                "schema": ROW_SCHEMA,
                **identity,
                "status": "valid",
                "reason": None,
                "task": task,
                "environment": environment,
                "verifier": verifier,
                "lifecycle": response.get("task_lifecycle_status"),
                "source": source,
                "taxonomy": _taxonomy(selected, metadata, source),
                "private_fields_persisted": False,
            }
        except (RuntimeError, ValueError) as error:
            reason = SAFE_FAILURES.get(str(error), str(error) if str(error) in {
                "cyber_subject_absent",
                "atom_source_absent",
                "atom_source_invalid",
                "atom_source_duplicate",
            } else "unclassified_binding_failure")
            value = {
                "schema": ROW_SCHEMA,
                **identity,
                "status": "invalid",
                "reason": reason,
                "task": None,
                "environment": None,
                "verifier": None,
                "lifecycle": None,
                "source": None,
                "taxonomy": None,
                "private_fields_persisted": False,
            }
        return _seal(value)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        audited = list(pool.map(one, rows))
    audited.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    counts = Counter((row["status"], row["reason"] or "valid") for row in audited)
    return _seal(
        {
            "schema": SCHEMA,
            "observed_at": observed_at,
            "selection_sha256": digest_json(rows),
            "task_versions": audited,
            "counts": {
                "total": len(audited),
                "valid": sum(row["status"] == "valid" for row in audited),
                "invalid": sum(row["status"] == "invalid" for row in audited),
                "by_reason": {
                    f"{status}:{reason}": count
                    for (status, reason), count in sorted(counts.items())
                },
            },
            "evidence_scope": (
                "current exact task/runtime/verifier/source binding only; successful execution, "
                "grader health, cleanup and training eligibility require separate receipts"
            ),
            "persisted_fields": "allowlisted metadata and digests only",
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--base-url", default="https://orchestrator.fleetai.com")
    args = parser.parse_args(argv)
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise SystemExit("FLEET_API_KEY is required")
    import json

    selection = json.loads(args.selection.read_text())
    rows = selection.get("tasks")
    if not isinstance(rows, list):
        raise SystemExit("selection must contain tasks[]")
    client = FleetClient(api_key, base_url=args.base_url)
    observed_at = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    result = audit_current_bindings(
        rows, client.task, observed_at=observed_at, workers=args.workers
    )
    result["selection_file_sha256"] = file_sha256(args.selection)
    # Re-seal after binding the source file bytes.
    result["sha256"] = digest_json({key: value for key, value in result.items() if key != "sha256"})
    atomic_write_json(args.output, result, private=True)
    print(f"wrote {result['counts']['valid']}/{result['counts']['total']} valid task bindings")
    print(f"sha256={result['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
