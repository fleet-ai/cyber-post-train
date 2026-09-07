"""Atomic rollout-cell ledger with a human-readable CSV projection.

The SQLite database is authoritative. CSV and JSONL files are exports only and must
never be used to claim work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from contextlib import closing, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

STATES = (
    "pending",
    "claimed",
    "running",
    "grading",
    "accepted",
    "retry_review",
    "terminal",
)
ACTIVE_STATES = ("claimed", "running", "grading")
PLAN_REQUIRED_COLUMNS = (
    "experiment_id",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
)
PLAN_STORED_COLUMNS = (
    "cell_id",
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
EXPORT_COLUMNS = (
    "cell_id",
    "experiment_id",
    "task_key",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
    "state",
    "worker_id",
    "claim_id",
    "session_id",
    "started_at",
    "heartbeat_at",
    "lease_expires_at",
    "completed_at",
    "retry_count",
    "max_retries",
    "result_class",
    "receipt_digest",
    "failure_code",
    "reconciliation_digest",
    "created_at",
    "updated_at",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FAILURE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


class LedgerError(RuntimeError):
    """Safe operational error from a ledger invariant."""


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _require_text(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise LedgerError(f"{name} must be non-empty")
    if any(character in text for character in "\r\n\0"):
        raise LedgerError(f"{name} contains a control character")
    return text


def _require_digest(value: str, name: str) -> str:
    digest = value.removeprefix("sha256:").lower()
    if not SHA256_RE.fullmatch(digest):
        raise LedgerError(f"{name} must be a SHA-256 digest")
    return digest


def _require_failure_code(value: object) -> str:
    code = _require_text(value, "failure_code")
    if not FAILURE_CODE_RE.fullmatch(code):
        raise LedgerError("failure_code must be a short sanitized machine code")
    return code


def _cell_id(row: dict[str, Any]) -> str:
    identity = "\0".join(
        str(row[field]) for field in ("experiment_id", "task_version_id", "model_id", "attempt")
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"fleet-cyber-rollout-cell-v1:{identity}"))


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_umask = os.umask(0o077)
    try:
        connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
    finally:
        os.umask(previous_umask)
    return connection


def _schema(connection: sqlite3.Connection) -> None:
    states = ",".join(f"'{state}'" for state in STATES)
    connection.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS ledger_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rollout_cells (
            cell_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            task_key TEXT,
            task_version_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            serving_block TEXT NOT NULL,
            endpoint_model_id TEXT NOT NULL,
            harness_id TEXT NOT NULL,
            attempt INTEGER NOT NULL CHECK (attempt > 0),
            state TEXT NOT NULL CHECK (state IN ({states})),
            worker_id TEXT,
            claim_id TEXT,
            session_id TEXT,
            started_at TEXT,
            heartbeat_at TEXT,
            lease_expires_at TEXT,
            completed_at TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
            max_retries INTEGER NOT NULL DEFAULT 1 CHECK (max_retries >= 0),
            result_class TEXT,
            receipt_digest TEXT,
            failure_code TEXT,
            reconciliation_digest TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (experiment_id, task_version_id, model_id, attempt)
        );

        CREATE INDEX IF NOT EXISTS rollout_cells_claim_queue
            ON rollout_cells (state, serving_block, task_version_id, model_id, attempt);
        CREATE INDEX IF NOT EXISTS rollout_cells_lease
            ON rollout_cells (state, lease_expires_at);

        CREATE TABLE IF NOT EXISTS rollout_events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id TEXT NOT NULL REFERENCES rollout_cells(cell_id),
            recorded_at TEXT NOT NULL,
            event TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT,
            worker_id TEXT,
            claim_id TEXT,
            detail_json TEXT NOT NULL,
            CHECK (json_valid(detail_json))
        );
        """
    )


def _plan_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = set(PLAN_REQUIRED_COLUMNS) - columns
        if missing:
            raise LedgerError(f"plan CSV is missing columns: {sorted(missing)}")
        raw_rows = list(reader)
    if not raw_rows:
        raise LedgerError("plan CSV has no rollout cells")

    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str, str, int]] = set()
    for index, raw in enumerate(raw_rows, start=2):
        try:
            attempt = int(raw["attempt"])
            max_retries = int(raw.get("max_retries") or 1)
        except ValueError as exc:
            raise LedgerError(f"plan row {index} has a non-integer attempt or retry limit") from exc
        if attempt < 1:
            raise LedgerError(f"plan row {index} attempt must be positive")
        if max_retries < 0:
            raise LedgerError(f"plan row {index} max_retries must be non-negative")
        row = {
            field: _require_text(raw[field], f"plan row {index} {field}")
            for field in PLAN_REQUIRED_COLUMNS
            if field != "attempt"
        }
        row.update(
            {
                "attempt": attempt,
                "task_key": (raw.get("task_key") or "").strip() or None,
                "max_retries": max_retries,
            }
        )
        identity = (
            row["experiment_id"],
            row["task_version_id"],
            row["model_id"],
            row["attempt"],
        )
        if identity in identities:
            raise LedgerError(f"plan row {index} duplicates rollout identity {identity}")
        identities.add(identity)
        row["cell_id"] = _cell_id(row)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["experiment_id"],
            row["task_version_id"],
            row["model_id"],
            row["attempt"],
        )
    )
    experiment_ids = {row["experiment_id"] for row in rows}
    if len(experiment_ids) != 1:
        raise LedgerError("one ledger plan must contain exactly one experiment_id")
    scientific_cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    model_bindings: dict[tuple[str, str], tuple[str, str]] = {}
    route_bindings: dict[tuple[str, str], tuple[str, str, str, str]] = {}
    for row in rows:
        scientific_cells.setdefault(
            (row["experiment_id"], row["task_version_id"], row["model_id"]), []
        ).append(row)
        model_key = (row["experiment_id"], row["model_id"])
        model_binding = (row["model_revision"], row["harness_id"])
        if model_key in model_bindings and model_bindings[model_key] != model_binding:
            raise LedgerError(f"model binding changes within experiment: {model_key}")
        model_bindings[model_key] = model_binding
        route_key = (row["experiment_id"], row["serving_block"])
        route_binding = (
            row["model_id"],
            row["model_revision"],
            row["endpoint_model_id"],
            row["harness_id"],
        )
        if route_key in route_bindings and route_bindings[route_key] != route_binding:
            raise LedgerError(f"serving block changes identity within experiment: {route_key}")
        route_bindings[route_key] = route_binding
    for identity, cells in scientific_cells.items():
        attempts = sorted(cell["attempt"] for cell in cells)
        if attempts != list(range(1, attempts[-1] + 1)):
            raise LedgerError(f"attempts are not contiguous for scientific cell {identity}")
        assignments = {
            (
                cell["serving_block"],
                cell["endpoint_model_id"],
                cell["task_key"],
                cell["max_retries"],
            )
            for cell in cells
        }
        if len(assignments) != 1:
            raise LedgerError(
                f"attempts split identity or serving block for scientific cell {identity}"
            )
    return rows


def _plan_digest(rows: Sequence[dict[str, Any]]) -> str:
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _event(
    connection: sqlite3.Connection,
    *,
    cell_id: str,
    name: str,
    from_state: str | None,
    to_state: str | None,
    worker_id: str | None = None,
    claim_id: str | None = None,
    detail: dict[str, Any] | None = None,
    recorded_at: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO rollout_events (
            cell_id, recorded_at, event, from_state, to_state,
            worker_id, claim_id, detail_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cell_id,
            recorded_at or _timestamp(),
            name,
            from_state,
            to_state,
            worker_id,
            claim_id,
            json.dumps(detail or {}, sort_keys=True, separators=(",", ":")),
        ),
    )


def initialize(database: Path, plan: Path) -> dict[str, Any]:
    rows = _plan_rows(plan)
    digest = _plan_digest(rows)
    with closing(_connect(database)) as connection:
        _schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            existing_digest_row = connection.execute(
                "SELECT value FROM ledger_metadata WHERE key = 'plan_sha256'"
            ).fetchone()
            existing_count = connection.execute("SELECT COUNT(*) FROM rollout_cells").fetchone()[0]
            if existing_count:
                if existing_digest_row is None or existing_digest_row[0] != digest:
                    raise LedgerError("database already contains a different rollout plan")
                if existing_count != len(rows):
                    raise LedgerError("database cell count differs from its recorded plan")
                stored_rows = [
                    {field: stored[field] for field in PLAN_STORED_COLUMNS}
                    for stored in connection.execute(
                        f"""
                        SELECT {", ".join(PLAN_STORED_COLUMNS)}
                        FROM rollout_cells
                        ORDER BY experiment_id, task_version_id, model_id, attempt
                        """  # noqa: S608
                    )
                ]
                if _plan_digest(stored_rows) != digest:
                    raise LedgerError("database immutable cell bindings differ from its plan")
                connection.execute("COMMIT")
                return {"created": False, "cells": existing_count, "plan_sha256": digest}

            now = _timestamp()
            for row in rows:
                connection.execute(
                    """
                    INSERT INTO rollout_cells (
                        cell_id, experiment_id, task_key, task_version_id,
                        model_id, model_revision, serving_block, endpoint_model_id,
                        harness_id, attempt, state, max_retries, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (
                        row["cell_id"],
                        row["experiment_id"],
                        row["task_key"],
                        row["task_version_id"],
                        row["model_id"],
                        row["model_revision"],
                        row["serving_block"],
                        row["endpoint_model_id"],
                        row["harness_id"],
                        row["attempt"],
                        row["max_retries"],
                        now,
                        now,
                    ),
                )
                _event(
                    connection,
                    cell_id=row["cell_id"],
                    name="initialized",
                    from_state=None,
                    to_state="pending",
                    recorded_at=now,
                )
            connection.executemany(
                "INSERT INTO ledger_metadata (key, value) VALUES (?, ?)",
                (
                    ("schema_version", "fleet_cyber_rollout_ledger_v1"),
                    ("plan_sha256", digest),
                    ("created_at", now),
                ),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return {"created": True, "cells": len(rows), "plan_sha256": digest}


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def claim(
    database: Path,
    *,
    worker_id: str,
    serving_block: str,
    lease_seconds: int = 900,
) -> dict[str, Any] | None:
    worker_id = _require_text(worker_id, "worker_id")
    serving_block = _require_text(serving_block, "serving_block")
    if lease_seconds < 30:
        raise LedgerError("lease_seconds must be at least 30")
    with closing(_connect(database)) as connection:
        _schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT * FROM rollout_cells
                WHERE state = 'pending' AND serving_block = ?
                ORDER BY task_version_id, model_id, attempt
                LIMIT 1
                """,
                (serving_block,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            now = _now()
            timestamp = _timestamp(now)
            claim_id = str(uuid.uuid4())
            lease_expires_at = _timestamp(now + timedelta(seconds=lease_seconds))
            result = connection.execute(
                """
                UPDATE rollout_cells
                SET state = 'claimed', worker_id = ?, claim_id = ?, heartbeat_at = ?,
                    lease_expires_at = ?, updated_at = ?
                WHERE cell_id = ? AND state = 'pending'
                """,
                (worker_id, claim_id, timestamp, lease_expires_at, timestamp, row["cell_id"]),
            )
            if result.rowcount != 1:
                raise LedgerError("atomic claim lost unexpectedly")
            _event(
                connection,
                cell_id=row["cell_id"],
                name="claimed",
                from_state="pending",
                to_state="claimed",
                worker_id=worker_id,
                claim_id=claim_id,
                detail={"lease_seconds": lease_seconds},
                recorded_at=timestamp,
            )
            claimed = connection.execute(
                "SELECT * FROM rollout_cells WHERE cell_id = ?", (row["cell_id"],)
            ).fetchone()
            connection.execute("COMMIT")
            return _row_dict(claimed)
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _owned_active_row(
    connection: sqlite3.Connection,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM rollout_cells WHERE cell_id = ?", (cell_id,)).fetchone()
    if row is None:
        raise LedgerError("rollout cell does not exist")
    if row["state"] not in ACTIVE_STATES:
        raise LedgerError(f"rollout cell is {row['state']}, not active")
    if row["worker_id"] != worker_id or row["claim_id"] != claim_id:
        raise LedgerError("worker or claim token does not own this rollout cell")
    return row


def heartbeat(
    database: Path,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    lease_seconds: int = 900,
) -> dict[str, Any]:
    if lease_seconds < 30:
        raise LedgerError("lease_seconds must be at least 30")
    with closing(_connect(database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = _owned_active_row(
                connection, cell_id=cell_id, worker_id=worker_id, claim_id=claim_id
            )
            now = _now()
            timestamp = _timestamp(now)
            connection.execute(
                """
                UPDATE rollout_cells
                SET heartbeat_at = ?, lease_expires_at = ?, updated_at = ?
                WHERE cell_id = ?
                """,
                (
                    timestamp,
                    _timestamp(now + timedelta(seconds=lease_seconds)),
                    timestamp,
                    cell_id,
                ),
            )
            _event(
                connection,
                cell_id=cell_id,
                name="heartbeat",
                from_state=row["state"],
                to_state=row["state"],
                worker_id=worker_id,
                claim_id=claim_id,
                detail={"lease_seconds": lease_seconds},
                recorded_at=timestamp,
            )
            updated = connection.execute(
                "SELECT * FROM rollout_cells WHERE cell_id = ?", (cell_id,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(updated)
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _owner_transition(
    database: Path,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    allowed_from: Iterable[str],
    to_state: str,
    event_name: str,
    updates: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = tuple(allowed_from)
    with closing(_connect(database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = _owned_active_row(
                connection, cell_id=cell_id, worker_id=worker_id, claim_id=claim_id
            )
            if row["state"] not in allowed:
                raise LedgerError(
                    f"cannot transition {row['state']} to {to_state}; expected one of {allowed}"
                )
            timestamp = _timestamp()
            values = {"state": to_state, "updated_at": timestamp, **(updates or {})}
            assignments = ", ".join(f"{field} = ?" for field in values)
            connection.execute(
                f"UPDATE rollout_cells SET {assignments} WHERE cell_id = ?",  # noqa: S608
                (*values.values(), cell_id),
            )
            _event(
                connection,
                cell_id=cell_id,
                name=event_name,
                from_state=row["state"],
                to_state=to_state,
                worker_id=worker_id,
                claim_id=claim_id,
                detail=detail,
                recorded_at=timestamp,
            )
            updated = connection.execute(
                "SELECT * FROM rollout_cells WHERE cell_id = ?", (cell_id,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(updated)
        except Exception:
            connection.execute("ROLLBACK")
            raise


def start(
    database: Path,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    session_id: str,
    lease_seconds: int = 900,
) -> dict[str, Any]:
    session_id = _require_text(session_id, "session_id")
    if lease_seconds < 30:
        raise LedgerError("lease_seconds must be at least 30")
    now = _now()
    timestamp = _timestamp(now)
    return _owner_transition(
        database,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=("claimed",),
        to_state="running",
        event_name="started",
        updates={
            "session_id": session_id,
            "started_at": timestamp,
            "heartbeat_at": timestamp,
            "lease_expires_at": _timestamp(now + timedelta(seconds=lease_seconds)),
        },
        detail={"session_id": session_id, "lease_seconds": lease_seconds},
    )


def mark_grading(database: Path, *, cell_id: str, worker_id: str, claim_id: str) -> dict[str, Any]:
    return _owner_transition(
        database,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=("running",),
        to_state="grading",
        event_name="grading_started",
    )


def accept(
    database: Path,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    receipt_digest: str,
) -> dict[str, Any]:
    digest = _require_digest(receipt_digest, "receipt_digest")
    timestamp = _timestamp()
    return _owner_transition(
        database,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=("running", "grading"),
        to_state="accepted",
        event_name="accepted",
        updates={
            "completed_at": timestamp,
            "heartbeat_at": timestamp,
            "lease_expires_at": None,
            "result_class": "valid",
            "receipt_digest": digest,
        },
        detail={"receipt_digest": digest},
    )


def request_retry_review(
    database: Path,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    failure_code: str,
) -> dict[str, Any]:
    failure_code = _require_failure_code(failure_code)
    timestamp = _timestamp()
    return _owner_transition(
        database,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=ACTIVE_STATES,
        to_state="retry_review",
        event_name="retry_review_requested",
        updates={
            "completed_at": timestamp,
            "heartbeat_at": timestamp,
            "lease_expires_at": None,
            "result_class": "infrastructure_invalid",
            "failure_code": failure_code,
        },
        detail={"failure_code": failure_code},
    )


def approve_retry(database: Path, *, cell_id: str, reconciliation_digest: str) -> dict[str, Any]:
    digest = _require_digest(reconciliation_digest, "reconciliation_digest")
    with closing(_connect(database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM rollout_cells WHERE cell_id = ?", (cell_id,)
            ).fetchone()
            if row is None:
                raise LedgerError("rollout cell does not exist")
            if row["state"] != "retry_review":
                raise LedgerError("only a retry_review cell may return to pending")
            if row["retry_count"] >= row["max_retries"]:
                raise LedgerError("rollout cell has exhausted its retry allowance")
            timestamp = _timestamp()
            connection.execute(
                """
                UPDATE rollout_cells
                SET state = 'pending', worker_id = NULL, claim_id = NULL,
                    session_id = NULL, started_at = NULL, heartbeat_at = NULL,
                    lease_expires_at = NULL, completed_at = NULL,
                    retry_count = retry_count + 1, result_class = NULL,
                    receipt_digest = NULL, reconciliation_digest = ?, updated_at = ?
                WHERE cell_id = ?
                """,
                (digest, timestamp, cell_id),
            )
            _event(
                connection,
                cell_id=cell_id,
                name="retry_approved",
                from_state="retry_review",
                to_state="pending",
                detail={
                    "reconciliation_digest": digest,
                    "prior_session_id": row["session_id"],
                    "failure_code": row["failure_code"],
                },
                recorded_at=timestamp,
            )
            updated = connection.execute(
                "SELECT * FROM rollout_cells WHERE cell_id = ?", (cell_id,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(updated)
        except Exception:
            connection.execute("ROLLBACK")
            raise


def mark_terminal(
    database: Path,
    *,
    cell_id: str,
    reconciliation_digest: str,
    failure_code: str | None = None,
) -> dict[str, Any]:
    digest = _require_digest(reconciliation_digest, "reconciliation_digest")
    with closing(_connect(database)) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                "SELECT * FROM rollout_cells WHERE cell_id = ?", (cell_id,)
            ).fetchone()
            if row is None:
                raise LedgerError("rollout cell does not exist")
            if row["state"] != "retry_review":
                raise LedgerError("only a retry_review cell may be terminally closed")
            code = _require_failure_code(failure_code or row["failure_code"])
            timestamp = _timestamp()
            connection.execute(
                """
                UPDATE rollout_cells
                SET state = 'terminal', completed_at = ?, lease_expires_at = NULL,
                    failure_code = ?, reconciliation_digest = ?, updated_at = ?
                WHERE cell_id = ?
                """,
                (timestamp, code, digest, timestamp, cell_id),
            )
            _event(
                connection,
                cell_id=cell_id,
                name="terminally_closed",
                from_state="retry_review",
                to_state="terminal",
                detail={"reconciliation_digest": digest, "failure_code": code},
                recorded_at=timestamp,
            )
            updated = connection.execute(
                "SELECT * FROM rollout_cells WHERE cell_id = ?", (cell_id,)
            ).fetchone()
            connection.execute("COMMIT")
            return dict(updated)
        except Exception:
            connection.execute("ROLLBACK")
            raise


def stale_cells(database: Path, *, now: datetime | None = None) -> list[dict[str, Any]]:
    timestamp = _timestamp(now)
    with closing(_connect(database)) as connection:
        rows = connection.execute(
            """
            SELECT * FROM rollout_cells
            WHERE state IN ('claimed', 'running', 'grading')
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < ?
            ORDER BY lease_expires_at, cell_id
            """,
            (timestamp,),
        ).fetchall()
    return [dict(row) for row in rows]


def summary(database: Path) -> dict[str, Any]:
    with closing(_connect(database)) as connection:
        total = connection.execute("SELECT COUNT(*) FROM rollout_cells").fetchone()[0]
        by_state = {
            row["state"]: row["count"]
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM rollout_cells GROUP BY state"
            )
        }
        by_block = [
            dict(row)
            for row in connection.execute(
                """
                SELECT serving_block, state, COUNT(*) AS count
                FROM rollout_cells
                GROUP BY serving_block, state
                ORDER BY serving_block, state
                """
            )
        ]
        plan_row = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'plan_sha256'"
        ).fetchone()
    return {
        "total": total,
        "by_state": {state: by_state.get(state, 0) for state in STATES},
        "by_serving_block": by_block,
        "stale_active": len(stale_cells(database)),
        "plan_sha256": plan_row[0] if plan_row else None,
    }


def _atomic_write(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        with suppress(OSError):
            os.close(descriptor)
        Path(temporary_name).unlink(missing_ok=True)
        raise


def export(database: Path, *, csv_path: Path, events_path: Path | None = None) -> dict[str, Any]:
    with closing(_connect(database)) as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT * FROM rollout_cells
                ORDER BY experiment_id, task_version_id, model_id, attempt
                """
            )
        ]
        events = (
            [
                dict(row)
                for row in connection.execute("SELECT * FROM rollout_events ORDER BY sequence")
            ]
            if events_path
            else []
        )

    def csv_lines() -> Iterable[str]:
        temporary = io.StringIO(newline="")
        writer = csv.DictWriter(temporary, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in EXPORT_COLUMNS} for row in rows)
        return (temporary.getvalue(),)

    _atomic_write(csv_path, csv_lines())
    if events_path:
        _atomic_write(
            events_path,
            (json.dumps(event, sort_keys=True) + "\n" for event in events),
        )
    return {
        "cells": len(rows),
        "csv": str(csv_path),
        "events": str(events_path) if events_path else None,
    }


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _add_owner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--claim-id", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    initialize_parser = commands.add_parser("init", help="Create or verify a ledger from CSV")
    initialize_parser.add_argument("--plan", type=Path, required=True)

    claim_parser = commands.add_parser("claim", help="Atomically claim one pending cell")
    claim_parser.add_argument("--worker-id", required=True)
    claim_parser.add_argument("--serving-block", required=True)
    claim_parser.add_argument("--lease-seconds", type=int, default=900)

    start_parser = commands.add_parser("start", help="Bind a claimed cell to a Fleet session")
    _add_owner_arguments(start_parser)
    start_parser.add_argument("--session-id", required=True)
    start_parser.add_argument("--lease-seconds", type=int, default=900)

    heartbeat_parser = commands.add_parser("heartbeat", help="Extend an active claim lease")
    _add_owner_arguments(heartbeat_parser)
    heartbeat_parser.add_argument("--lease-seconds", type=int, default=900)

    grading_parser = commands.add_parser("grading", help="Mark authoritative grading in progress")
    _add_owner_arguments(grading_parser)

    accept_parser = commands.add_parser("accept", help="Accept one complete valid rollout")
    _add_owner_arguments(accept_parser)
    accept_parser.add_argument("--receipt-digest", required=True)

    review_parser = commands.add_parser(
        "retry-review", help="Quarantine an infrastructure-invalid active cell"
    )
    _add_owner_arguments(review_parser)
    review_parser.add_argument("--failure-code", required=True)

    retry_parser = commands.add_parser(
        "approve-retry", help="Return a reviewed infrastructure failure to pending"
    )
    retry_parser.add_argument("--cell-id", required=True)
    retry_parser.add_argument("--reconciliation-digest", required=True)

    terminal_parser = commands.add_parser(
        "terminal", help="Close a retry-review cell without another attempt"
    )
    terminal_parser.add_argument("--cell-id", required=True)
    terminal_parser.add_argument("--reconciliation-digest", required=True)
    terminal_parser.add_argument("--failure-code")

    commands.add_parser("stale", help="List expired active leases without changing them")
    commands.add_parser("status", help="Summarize ledger progress")

    export_parser = commands.add_parser("export", help="Atomically refresh human-readable exports")
    export_parser.add_argument("--csv", type=Path, required=True)
    export_parser.add_argument("--events", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "init":
            result = initialize(arguments.db, arguments.plan)
        elif arguments.command == "claim":
            cell = claim(
                arguments.db,
                worker_id=arguments.worker_id,
                serving_block=arguments.serving_block,
                lease_seconds=arguments.lease_seconds,
            )
            result = {"claimed": cell is not None, "cell": cell}
        elif arguments.command == "start":
            result = start(
                arguments.db,
                cell_id=arguments.cell_id,
                worker_id=arguments.worker_id,
                claim_id=arguments.claim_id,
                session_id=arguments.session_id,
                lease_seconds=arguments.lease_seconds,
            )
        elif arguments.command == "heartbeat":
            result = heartbeat(
                arguments.db,
                cell_id=arguments.cell_id,
                worker_id=arguments.worker_id,
                claim_id=arguments.claim_id,
                lease_seconds=arguments.lease_seconds,
            )
        elif arguments.command == "grading":
            result = mark_grading(
                arguments.db,
                cell_id=arguments.cell_id,
                worker_id=arguments.worker_id,
                claim_id=arguments.claim_id,
            )
        elif arguments.command == "accept":
            result = accept(
                arguments.db,
                cell_id=arguments.cell_id,
                worker_id=arguments.worker_id,
                claim_id=arguments.claim_id,
                receipt_digest=arguments.receipt_digest,
            )
        elif arguments.command == "retry-review":
            result = request_retry_review(
                arguments.db,
                cell_id=arguments.cell_id,
                worker_id=arguments.worker_id,
                claim_id=arguments.claim_id,
                failure_code=arguments.failure_code,
            )
        elif arguments.command == "approve-retry":
            result = approve_retry(
                arguments.db,
                cell_id=arguments.cell_id,
                reconciliation_digest=arguments.reconciliation_digest,
            )
        elif arguments.command == "terminal":
            result = mark_terminal(
                arguments.db,
                cell_id=arguments.cell_id,
                reconciliation_digest=arguments.reconciliation_digest,
                failure_code=arguments.failure_code,
            )
        elif arguments.command == "stale":
            result = {"cells": stale_cells(arguments.db)}
        elif arguments.command == "status":
            result = summary(arguments.db)
        elif arguments.command == "export":
            result = export(arguments.db, csv_path=arguments.csv, events_path=arguments.events)
        else:  # pragma: no cover
            raise AssertionError(arguments.command)
        _print(result)
        return 0
    except (LedgerError, sqlite3.Error, OSError) as exc:
        print(f"rollout ledger failed: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
