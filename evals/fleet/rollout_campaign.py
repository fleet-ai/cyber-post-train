"""Build and reconcile the frozen Qwen/GLM pass@4 rollout ledger.

This module deliberately handles only score-blind scientific identity and historical
disposition. It never reads prompts, traces, flags, scores, or model responses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import rollout_ledger

CAMPAIGN_SCHEMA = "fleet-exact-easiest100-pass4-campaign-v1"
SNAPSHOT_SCHEMA = "fleet-exact-pass4-ledger-audit-v50"
HARNESS_ID = "opencode-1.18.27-autocontinue-v1"
BLOCKS = {
    ("qwen3.8-27b", "shared"): ("qwen-shared", "qwen3.8-27b"),
    ("qwen3.8-27b", "dedicated"): (
        "qwen-dedicated",
        "chris-cyber-qwen38-27b-dedicated-v1",
    ),
    ("glm-5.3", "shared"): ("glm-shared", "glm-5.3"),
    ("glm-5.3", "dedicated"): (
        "glm-dedicated",
        "chris-cyber-glm53-dedicated-v1",
    ),
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise rollout_ledger.LedgerError(f"{path} must contain a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_digest(value: dict[str, Any], field: str = "receipt_sha256") -> str:
    body = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    return rollout_ledger._require_digest(str(value or ""), name)  # noqa: SLF001


def _evidence_block(path: str) -> str | None:
    lowered = path.lower()
    if "dedicated" in lowered or "-ded-" in lowered or "ded-tp" in lowered:
        return "dedicated"
    if "hosted" in lowered:
        return "shared"
    return None


def _historical_assignments(snapshot: dict[str, Any]) -> dict[tuple[str, int], str]:
    observed: dict[tuple[str, int], set[str]] = defaultdict(set)
    for cell in snapshot.get("cells", []):
        key = (str(cell.get("model")), int(cell.get("selection_rank", 0)))
        for evidence in cell.get("v48_evidence", []):
            block = _evidence_block(str(evidence.get("path", "")))
            if block:
                observed[key].add(block)
    conflicts = {key: blocks for key, blocks in observed.items() if len(blocks) != 1}
    if conflicts:
        raise rollout_ledger.LedgerError(f"historical serving-block conflict: {conflicts}")
    return {key: next(iter(blocks)) for key, blocks in observed.items()}


def build_plan_rows(
    campaign_path: Path,
    selection_path: Path,
    snapshot_path: Path,
) -> list[dict[str, Any]]:
    campaign = _load_json(campaign_path)
    selection = _load_json(selection_path)
    snapshot = _load_json(snapshot_path)

    if campaign.get("schema_version") != CAMPAIGN_SCHEMA:
        raise rollout_ledger.LedgerError("campaign schema is not the frozen pass@4 schema")
    expected_selection_sha = _require_sha256(
        campaign.get("selection", {}).get("file_sha256"), "selection file_sha256"
    )
    if _file_sha256(selection_path) != expected_selection_sha:
        raise rollout_ledger.LedgerError("selection file digest does not match the campaign")
    snapshot_digest = _require_sha256(snapshot.get("receipt_sha256"), "snapshot receipt_sha256")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise rollout_ledger.LedgerError("historical snapshot schema is not v50")
    if _self_digest(snapshot) != snapshot_digest:
        raise rollout_ledger.LedgerError("historical snapshot self-digest is invalid")
    if snapshot.get("campaign_id") != campaign.get("campaign_id"):
        raise rollout_ledger.LedgerError("snapshot and campaign identities differ")

    models = campaign.get("models")
    tasks = selection.get("tasks")
    attempts = campaign.get("attempts")
    if not isinstance(models, dict) or set(models) != {"qwen3.8-27b", "glm-5.3"}:
        raise rollout_ledger.LedgerError("campaign must bind exactly Qwen3.8-27B and GLM-5.3")
    if not isinstance(tasks, list) or len(tasks) != 100:
        raise rollout_ledger.LedgerError("selection must contain exactly 100 tasks")
    if attempts != [1, 2, 3, 4] or campaign.get("pass_k") != 4:
        raise rollout_ledger.LedgerError("campaign must contain attempts 1..4 and pass@4")
    if campaign.get("cell_count") != 800:
        raise rollout_ledger.LedgerError("campaign cell_count must be 800")

    task_versions = [str(task.get("task_version_id", "")) for task in tasks]
    ranks = [int(task.get("rank", 0)) for task in tasks]
    if len(set(task_versions)) != 100 or sorted(ranks) != list(range(1, 101)):
        raise rollout_ledger.LedgerError("selection task versions or ranks are not unique")

    historical = _historical_assignments(snapshot)
    rows: list[dict[str, Any]] = []
    for model_id, model in sorted(models.items()):
        revision = str(model.get("revision", ""))
        if not revision:
            raise rollout_ledger.LedgerError(f"{model_id} has no immutable revision")
        for task in tasks:
            rank = int(task["rank"])
            # Historical evidence wins. Untouched tasks alternate as complete task units.
            location = historical.get((model_id, rank), "shared" if rank % 2 else "dedicated")
            serving_block, endpoint_model_id = BLOCKS[(model_id, location)]
            for attempt in attempts:
                rows.append(
                    {
                        "experiment_id": campaign["campaign_id"],
                        "task_key": task["task_key"],
                        "task_version_id": task["task_version_id"],
                        "model_id": model_id,
                        "model_revision": revision,
                        "serving_block": serving_block,
                        "endpoint_model_id": endpoint_model_id,
                        "harness_id": HARNESS_ID,
                        "attempt": attempt,
                        "max_retries": 1,
                    }
                )
    return rows


def write_plan(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    columns = (
        "experiment_id",
        "task_key",
        "task_version_id",
        "model_id",
        "model_revision",
        "serving_block",
        "endpoint_model_id",
        "harness_id",
        "attempt",
        "max_retries",
    )
    lines: list[str] = []
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    lines.append(stream.getvalue())
    rollout_ledger._atomic_write(path, lines)  # noqa: SLF001
    return {"cells": len(rows), "plan": str(path), "sha256": _file_sha256(path)}


def _cell_evidence_digest(cell: dict[str, Any], *, accepted: bool) -> str:
    evidence = cell.get("v48_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise rollout_ledger.LedgerError("historical non-pending cell has no evidence")
    candidates = [item for item in evidence if item.get("kind") == "accepted"] if accepted else []
    chosen = candidates[-1] if candidates else evidence[-1]
    return _require_sha256(chosen.get("receipt_sha256"), "historical evidence digest")


def import_snapshot(database: Path, snapshot_path: Path) -> dict[str, Any]:
    snapshot = _load_json(snapshot_path)
    snapshot_digest = _require_sha256(snapshot.get("receipt_sha256"), "snapshot receipt_sha256")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA:
        raise rollout_ledger.LedgerError("historical snapshot schema is not v50")
    if _self_digest(snapshot) != snapshot_digest:
        raise rollout_ledger.LedgerError("historical snapshot self-digest is invalid")
    cells = snapshot.get("cells")
    if not isinstance(cells, list) or len(cells) != 800:
        raise rollout_ledger.LedgerError("historical snapshot must classify exactly 800 cells")

    imported = {"accepted": 0, "terminal": 0, "pending": 0}
    now = datetime.now(UTC).isoformat()
    with closing(rollout_ledger._connect(database)) as connection:  # noqa: SLF001
        rollout_ledger._schema(connection)  # noqa: SLF001
        connection.execute("BEGIN IMMEDIATE")
        try:
            prior = connection.execute(
                "SELECT value FROM ledger_metadata WHERE key = 'historical_snapshot_sha256'"
            ).fetchone()
            if prior:
                if prior[0] != snapshot_digest:
                    raise rollout_ledger.LedgerError("a different historical snapshot was imported")
                connection.execute("COMMIT")
                return {"created": False, "snapshot_sha256": snapshot_digest}
            nonpending = connection.execute(
                "SELECT COUNT(*) FROM rollout_cells WHERE state != 'pending'"
            ).fetchone()[0]
            if nonpending:
                raise rollout_ledger.LedgerError("historical import requires a pristine ledger")

            seen: set[tuple[str, str, int]] = set()
            for source in cells:
                identity = (
                    str(source.get("model")),
                    str(source.get("task_version_id")),
                    int(source.get("attempt", 0)),
                )
                if identity in seen:
                    raise rollout_ledger.LedgerError("historical snapshot duplicates a cell")
                seen.add(identity)
                target = connection.execute(
                    """
                    SELECT * FROM rollout_cells
                    WHERE model_id = ? AND task_version_id = ? AND attempt = ?
                    """,
                    identity,
                ).fetchone()
                if target is None:
                    raise rollout_ledger.LedgerError(
                        f"historical cell is outside the plan: {identity}"
                    )
                state = source.get("state")
                if state == "unstarted":
                    imported["pending"] += 1
                    continue
                if state == "accepted":
                    evidence_digest = _cell_evidence_digest(source, accepted=True)
                    destination = "accepted"
                    result_class = "valid_historical"
                    failure_code = None
                    imported["accepted"] += 1
                elif state == "blocked_nonrepeatable":
                    evidence_digest = _cell_evidence_digest(source, accepted=False)
                    destination = "terminal"
                    result_class = "infrastructure_invalid_historical"
                    failure_code = "historical_nonrepeatable"
                    imported["terminal"] += 1
                else:
                    raise rollout_ledger.LedgerError(f"unsupported historical state: {state}")
                connection.execute(
                    """
                    UPDATE rollout_cells
                    SET state = ?, completed_at = ?, result_class = ?, receipt_digest = ?,
                        failure_code = ?, reconciliation_digest = ?, updated_at = ?
                    WHERE cell_id = ? AND state = 'pending'
                    """,
                    (
                        destination,
                        now,
                        result_class,
                        evidence_digest if destination == "accepted" else None,
                        failure_code,
                        snapshot_digest,
                        now,
                        target["cell_id"],
                    ),
                )
                rollout_ledger._event(  # noqa: SLF001
                    connection,
                    cell_id=target["cell_id"],
                    name="historical_import",
                    from_state="pending",
                    to_state=destination,
                    detail={
                        "snapshot_sha256": snapshot_digest,
                        "evidence_sha256": evidence_digest,
                    },
                    recorded_at=now,
                )
            if len(seen) != 800:
                raise rollout_ledger.LedgerError("historical snapshot did not cover the plan")
            connection.execute(
                "INSERT INTO ledger_metadata (key, value) VALUES (?, ?)",
                ("historical_snapshot_sha256", snapshot_digest),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return {"created": True, "snapshot_sha256": snapshot_digest, **imported}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan")
    build.add_argument("--campaign", type=Path, required=True)
    build.add_argument("--selection", type=Path, required=True)
    build.add_argument("--snapshot", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    reconcile = commands.add_parser("import-snapshot")
    reconcile.add_argument("--db", type=Path, required=True)
    reconcile.add_argument("--snapshot", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "build-plan":
            result = write_plan(
                arguments.output,
                build_plan_rows(arguments.campaign, arguments.selection, arguments.snapshot),
            )
        else:
            result = import_snapshot(arguments.db, arguments.snapshot)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (rollout_ledger.LedgerError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"rollout campaign failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
