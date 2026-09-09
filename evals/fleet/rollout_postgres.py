"""PostgreSQL coordination backend for the exact pass@4 rollout campaign.

The database stores score-blind scheduling state plus the existing private local-result
index.  Workers may run on different hosts because PostgreSQL, rather than a shared
filesystem, owns transaction and lock coordination.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

from evals.fleet import rollout_ledger


def _connect(dsn: str) -> psycopg.Connection[dict[str, Any]]:
    if not dsn.startswith(("postgresql://", "postgres://")):
        raise rollout_ledger.LedgerError("PostgreSQL DSN has an unsupported scheme")
    return psycopg.connect(
        dsn,
        autocommit=False,
        row_factory=dict_row,
        connect_timeout=30,
        application_name="fleet-cyber-rollout-ledger-v3",
    )


@contextmanager
def _transaction(dsn: str) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    with _connect(dsn) as connection, connection.transaction():
        yield connection


@contextmanager
def _read_transaction(dsn: str) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    """Consistent observation only: never initialize schema or repair rows."""
    with _connect(dsn) as connection, connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        yield connection


def ensure_schema(connection: psycopg.Connection[dict[str, Any]]) -> None:
    states = ",".join(f"'{state}'" for state in rollout_ledger.STATES)
    connection.execute(
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
            started_at TIMESTAMPTZ,
            heartbeat_at TIMESTAMPTZ,
            lease_expires_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
            max_retries INTEGER NOT NULL DEFAULT 1 CHECK (max_retries >= 0),
            result_class TEXT,
            receipt_digest TEXT,
            failure_code TEXT,
            reconciliation_digest TEXT,
            created_at TIMESTAMPTZ NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL,
            UNIQUE (experiment_id, task_version_id, model_id, attempt)
        );

        CREATE INDEX IF NOT EXISTS rollout_cells_claim_queue
            ON rollout_cells (state, serving_block, task_version_id, model_id, attempt);
        CREATE INDEX IF NOT EXISTS rollout_cells_lease
            ON rollout_cells (state, lease_expires_at);
        CREATE INDEX IF NOT EXISTS rollout_cells_worker
            ON rollout_cells (state, worker_id);

        CREATE TABLE IF NOT EXISTS rollout_events (
            sequence BIGSERIAL PRIMARY KEY,
            cell_id TEXT NOT NULL REFERENCES rollout_cells(cell_id),
            recorded_at TIMESTAMPTZ NOT NULL,
            event TEXT NOT NULL,
            from_state TEXT,
            to_state TEXT,
            worker_id TEXT,
            claim_id TEXT,
            detail_json TEXT NOT NULL,
            CHECK (detail_json::jsonb IS NOT NULL)
        );

        CREATE TABLE IF NOT EXISTS rollout_local_results (
            execution_id TEXT PRIMARY KEY,
            cell_id TEXT NOT NULL REFERENCES rollout_cells(cell_id),
            execution_generation INTEGER NOT NULL CHECK (execution_generation > 0),
            run_id TEXT NOT NULL,
            session_id TEXT,
            verifier_execution_id TEXT NOT NULL,
            score DOUBLE PRECISION NOT NULL CHECK (score >= 0.0 AND score <= 1.0),
            config_sha256 TEXT NOT NULL,
            artifact_directory TEXT NOT NULL,
            trace_path TEXT NOT NULL,
            trace_sha256 TEXT NOT NULL,
            result_path TEXT NOT NULL,
            result_sha256 TEXT NOT NULL,
            reward_path TEXT NOT NULL,
            reward_sha256 TEXT NOT NULL,
            session_ingest_path TEXT NOT NULL,
            session_ingest_sha256 TEXT NOT NULL,
            cleanup_path TEXT NOT NULL,
            cleanup_sha256 TEXT NOT NULL,
            session_ingest_status TEXT NOT NULL,
            agent_exit_code INTEGER NOT NULL,
            agent_termination TEXT NOT NULL,
            elapsed_seconds DOUBLE PRECISION NOT NULL CHECK (elapsed_seconds >= 0.0),
            record_sha256 TEXT NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL,
            UNIQUE (cell_id, execution_generation)
        );

        CREATE INDEX IF NOT EXISTS rollout_local_results_cell
            ON rollout_local_results (cell_id);

        CREATE TABLE IF NOT EXISTS ledger_migrations (
            source_kind TEXT NOT NULL,
            source_sha256 TEXT PRIMARY KEY,
            source_row_digests_json TEXT NOT NULL,
            target_row_digests_json TEXT NOT NULL,
            receipt_sha256 TEXT NOT NULL,
            completed_at TIMESTAMPTZ NOT NULL,
            CHECK (source_row_digests_json::jsonb IS NOT NULL),
            CHECK (target_row_digests_json::jsonb IS NOT NULL)
        );

        CREATE TABLE IF NOT EXISTS ledger_reconciliations (
            receipt_sha256 TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL,
            CHECK (receipt_json::jsonb IS NOT NULL)
        );
        """
    )


def _event(
    connection: psycopg.Connection[dict[str, Any]],
    *,
    cell_id: str,
    name: str,
    from_state: str | None,
    to_state: str | None,
    worker_id: str | None = None,
    claim_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO rollout_events (
            cell_id, recorded_at, event, from_state, to_state, worker_id, claim_id, detail_json
        ) VALUES (%s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s)
        """,
        (
            cell_id,
            name,
            from_state,
            to_state,
            worker_id,
            claim_id,
            json.dumps(detail or {}, sort_keys=True, separators=(",", ":")),
        ),
    )


def verify_plan(dsn: str, plan: Path) -> dict[str, Any]:
    rows = rollout_ledger._plan_rows(plan)  # noqa: SLF001
    digest = rollout_ledger._plan_digest(rows)  # noqa: SLF001
    with _transaction(dsn) as connection:
        ensure_schema(connection)
        metadata = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'plan_sha256'"
        ).fetchone()
        count = connection.execute("SELECT COUNT(*) AS count FROM rollout_cells").fetchone()[
            "count"
        ]
        if metadata is None or metadata["value"] != digest:
            raise rollout_ledger.LedgerError("PostgreSQL ledger plan digest differs")
        if count != len(rows):
            raise rollout_ledger.LedgerError("PostgreSQL ledger cell count differs from plan")
        stored = [
            {field: row[field] for field in rollout_ledger.PLAN_STORED_COLUMNS}
            for row in connection.execute(
                f"""
                SELECT {", ".join(rollout_ledger.PLAN_STORED_COLUMNS)}
                FROM rollout_cells
                ORDER BY experiment_id, task_version_id, model_id, attempt
                """  # noqa: S608
            ).fetchall()
        ]
        if rollout_ledger._plan_digest(stored) != digest:  # noqa: SLF001
            raise rollout_ledger.LedgerError("PostgreSQL immutable cell bindings differ")
    return {"created": False, "cells": count, "plan_sha256": digest}


def claim(
    dsn: str, *, worker_id: str, serving_block: str, lease_seconds: int = 900
) -> dict[str, Any] | None:
    worker_id = rollout_ledger._require_text(worker_id, "worker_id")  # noqa: SLF001
    serving_block = rollout_ledger._require_text(serving_block, "serving_block")  # noqa: SLF001
    if lease_seconds < 30:
        raise rollout_ledger.LedgerError("lease_seconds must be at least 30")
    with _transaction(dsn) as connection:
        row = connection.execute(
            """
            SELECT * FROM rollout_cells
            WHERE state = 'pending' AND serving_block = %s
            ORDER BY task_version_id, model_id, attempt
            FOR UPDATE SKIP LOCKED
            LIMIT 1
            """,
            (serving_block,),
        ).fetchone()
        if row is None:
            return None
        claim_id = str(uuid.uuid4())
        claimed = connection.execute(
            """
            UPDATE rollout_cells
            SET state = 'claimed', worker_id = %s, claim_id = %s,
                heartbeat_at = CURRENT_TIMESTAMP,
                lease_expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                updated_at = CURRENT_TIMESTAMP
            WHERE cell_id = %s AND state = 'pending'
            RETURNING *
            """,
            (worker_id, claim_id, lease_seconds, row["cell_id"]),
        ).fetchone()
        if claimed is None:
            raise rollout_ledger.LedgerError("atomic PostgreSQL claim lost unexpectedly")
        _event(
            connection,
            cell_id=row["cell_id"],
            name="claimed",
            from_state="pending",
            to_state="claimed",
            worker_id=worker_id,
            claim_id=claim_id,
            detail={"lease_seconds": lease_seconds},
        )
        return claimed


def _owned_active_row(
    connection: psycopg.Connection[dict[str, Any]],
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
) -> dict[str, Any]:
    row = connection.execute(
        "SELECT * FROM rollout_cells WHERE cell_id = %s FOR UPDATE", (cell_id,)
    ).fetchone()
    if row is None:
        raise rollout_ledger.LedgerError("rollout cell does not exist")
    if row["state"] not in rollout_ledger.ACTIVE_STATES:
        raise rollout_ledger.LedgerError("rollout cell is not active")
    if row["worker_id"] != worker_id or row["claim_id"] != claim_id:
        raise rollout_ledger.LedgerError("worker does not own the active rollout cell")
    return row


def heartbeat(
    dsn: str,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    lease_seconds: int = 900,
) -> dict[str, Any]:
    if lease_seconds < 30:
        raise rollout_ledger.LedgerError("lease_seconds must be at least 30")
    with _transaction(dsn) as connection:
        row = _owned_active_row(connection, cell_id=cell_id, worker_id=worker_id, claim_id=claim_id)
        updated = connection.execute(
            """
            UPDATE rollout_cells
            SET heartbeat_at = CURRENT_TIMESTAMP,
                lease_expires_at = CURRENT_TIMESTAMP + (%s * INTERVAL '1 second'),
                updated_at = CURRENT_TIMESTAMP
            WHERE cell_id = %s
            RETURNING *
            """,
            (lease_seconds, cell_id),
        ).fetchone()
        _event(
            connection,
            cell_id=cell_id,
            name="heartbeat",
            from_state=row["state"],
            to_state=row["state"],
            worker_id=worker_id,
            claim_id=claim_id,
            detail={"lease_seconds": lease_seconds},
        )
        return updated


def _owner_transition(
    dsn: str,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    allowed_from: Sequence[str],
    to_state: str,
    event_name: str,
    updates: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _transaction(dsn) as connection:
        row = _owned_active_row(connection, cell_id=cell_id, worker_id=worker_id, claim_id=claim_id)
        if row["state"] not in allowed_from:
            raise rollout_ledger.LedgerError(
                f"cannot transition {row['state']} to {to_state}; expected {tuple(allowed_from)}"
            )
        fields = {"state": to_state, "updated_at": datetime.now(UTC), **(updates or {})}
        assignments = ", ".join(f"{field} = %s" for field in fields)
        updated = connection.execute(
            f"UPDATE rollout_cells SET {assignments} WHERE cell_id = %s RETURNING *",  # noqa: S608
            (*fields.values(), cell_id),
        ).fetchone()
        _event(
            connection,
            cell_id=cell_id,
            name=event_name,
            from_state=row["state"],
            to_state=to_state,
            worker_id=worker_id,
            claim_id=claim_id,
            detail=detail,
        )
        return updated


def start(
    dsn: str,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    session_id: str,
    lease_seconds: int = 900,
) -> dict[str, Any]:
    session_id = rollout_ledger._require_text(session_id, "session_id")  # noqa: SLF001
    if lease_seconds < 30:
        raise rollout_ledger.LedgerError("lease_seconds must be at least 30")
    now = datetime.now(UTC)
    return _owner_transition(
        dsn,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=("claimed",),
        to_state="running",
        event_name="started",
        updates={
            "session_id": session_id,
            "started_at": now,
            "heartbeat_at": now,
            "lease_expires_at": now + timedelta(seconds=lease_seconds),
        },
        detail={"session_id": session_id, "lease_seconds": lease_seconds},
    )


def mark_grading(dsn: str, *, cell_id: str, worker_id: str, claim_id: str) -> dict[str, Any]:
    return _owner_transition(
        dsn,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=("running",),
        to_state="grading",
        event_name="grading_started",
    )


def accept(
    dsn: str, *, cell_id: str, worker_id: str, claim_id: str, receipt_digest: str
) -> dict[str, Any]:
    digest = rollout_ledger._require_digest(receipt_digest, "receipt_digest")  # noqa: SLF001
    now = datetime.now(UTC)
    return _owner_transition(
        dsn,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=("running", "grading"),
        to_state="accepted",
        event_name="accepted",
        updates={
            "completed_at": now,
            "heartbeat_at": now,
            "lease_expires_at": None,
            "result_class": "valid",
            "receipt_digest": digest,
        },
        detail={"receipt_digest": digest},
    )


def request_retry_review(
    dsn: str, *, cell_id: str, worker_id: str, claim_id: str, failure_code: str
) -> dict[str, Any]:
    code = rollout_ledger._require_failure_code(failure_code)  # noqa: SLF001
    now = datetime.now(UTC)
    return _owner_transition(
        dsn,
        cell_id=cell_id,
        worker_id=worker_id,
        claim_id=claim_id,
        allowed_from=rollout_ledger.ACTIVE_STATES,
        to_state="retry_review",
        event_name="retry_review_requested",
        updates={
            "completed_at": now,
            "heartbeat_at": now,
            "lease_expires_at": None,
            "result_class": "infrastructure_invalid",
            "failure_code": code,
        },
        detail={"failure_code": code},
    )


def _normalized_local_result(record: dict[str, Any], *, cell_id: str) -> dict[str, Any]:
    generation = record.get("execution_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        raise rollout_ledger.LedgerError("execution_generation must be a positive integer")
    exit_code = record.get("agent_exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        raise rollout_ledger.LedgerError("agent_exit_code must be an integer")
    elapsed = record.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, int | float):
        raise rollout_ledger.LedgerError("elapsed_seconds must be numeric")
    elapsed = float(elapsed)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise rollout_ledger.LedgerError("elapsed_seconds must be finite and non-negative")
    require_text = rollout_ledger._require_text  # noqa: SLF001
    require_digest = rollout_ledger._require_digest  # noqa: SLF001
    require_path = rollout_ledger._require_relative_path  # noqa: SLF001
    normalized = {
        "execution_id": require_text(record.get("execution_id"), "execution_id"),
        "cell_id": require_text(cell_id, "cell_id"),
        "execution_generation": generation,
        "run_id": require_text(record.get("run_id"), "run_id"),
        "session_id": (
            require_text(record["session_id"], "session_id")
            if record.get("session_id") is not None
            else None
        ),
        "verifier_execution_id": require_text(
            record.get("verifier_execution_id"), "verifier_execution_id"
        ),
        "score": rollout_ledger._require_score(record.get("score")),  # noqa: SLF001
        "config_sha256": require_digest(record.get("config_sha256", ""), "config_sha256"),
        "artifact_directory": require_path(record.get("artifact_directory"), "artifact_directory"),
        "trace_path": require_path(record.get("trace_path"), "trace_path"),
        "trace_sha256": require_digest(record.get("trace_sha256", ""), "trace_sha256"),
        "result_path": require_path(record.get("result_path"), "result_path"),
        "result_sha256": require_digest(record.get("result_sha256", ""), "result_sha256"),
        "reward_path": require_path(record.get("reward_path"), "reward_path"),
        "reward_sha256": require_digest(record.get("reward_sha256", ""), "reward_sha256"),
        "session_ingest_path": require_path(
            record.get("session_ingest_path"), "session_ingest_path"
        ),
        "session_ingest_sha256": require_digest(
            record.get("session_ingest_sha256", ""), "session_ingest_sha256"
        ),
        "cleanup_path": require_path(record.get("cleanup_path"), "cleanup_path"),
        "cleanup_sha256": require_digest(record.get("cleanup_sha256", ""), "cleanup_sha256"),
        "session_ingest_status": require_text(
            record.get("session_ingest_status"), "session_ingest_status"
        ),
        "agent_exit_code": exit_code,
        "agent_termination": require_text(record.get("agent_termination"), "agent_termination"),
        "elapsed_seconds": elapsed,
    }
    if rollout_ledger.EXECUTION_ID_RE.fullmatch(normalized["execution_id"]) is None:
        raise rollout_ledger.LedgerError("execution_id must be a sha256 identity")
    return normalized


def record_local_result(
    dsn: str,
    *,
    cell_id: str,
    worker_id: str,
    claim_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    normalized = _normalized_local_result(record, cell_id=cell_id)
    record_sha256 = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    columns = [*normalized, "record_sha256"]
    with _transaction(dsn) as connection:
        cell = _owned_active_row(
            connection, cell_id=cell_id, worker_id=worker_id, claim_id=claim_id
        )
        existing = connection.execute(
            "SELECT * FROM rollout_local_results WHERE execution_id = %s",
            (normalized["execution_id"],),
        ).fetchone()
        if existing is not None:
            if existing["record_sha256"] != record_sha256:
                raise rollout_ledger.LedgerError(
                    "local result identity already has different evidence"
                )
            return {"created": False, **existing}
        placeholders = ", ".join("%s" for _ in columns)
        created = connection.execute(
            f"""
            INSERT INTO rollout_local_results ({", ".join(columns)}, recorded_at)
            VALUES ({placeholders}, CURRENT_TIMESTAMP)
            RETURNING *
            """,  # noqa: S608
            [*(normalized[column] for column in normalized), record_sha256],
        ).fetchone()
        _event(
            connection,
            cell_id=cell_id,
            name="local_result_recorded",
            from_state=cell["state"],
            to_state=cell["state"],
            worker_id=worker_id,
            claim_id=claim_id,
            detail={
                "execution_id": normalized["execution_id"],
                "record_sha256": record_sha256,
            },
        )
        return {"created": True, **created}


def active_claims(dsn: str) -> dict[str, Any]:
    with _read_transaction(dsn) as connection:
        rows = connection.execute(
            """
            SELECT serving_block, worker_id, claim_id, cell_id, lease_expires_at,
                   lease_expires_at < CURRENT_TIMESTAMP AS stale
            FROM rollout_cells
            WHERE state IN ('claimed', 'running', 'grading')
            ORDER BY serving_block, worker_id, cell_id
            """
        ).fetchall()
    by_route: dict[str, int] = {}
    for row in rows:
        by_route[row["serving_block"]] = by_route.get(row["serving_block"], 0) + 1
    return {
        "active_claims": by_route,
        "stale_active": sum(bool(row["stale"]) for row in rows),
        "owners": [
            {
                "serving_block": row["serving_block"],
                "worker_id": row["worker_id"],
                "claim_id": row["claim_id"],
                "cell_id": row["cell_id"],
                "stale": bool(row["stale"]),
            }
            for row in rows
        ],
    }


def summary(dsn: str) -> dict[str, Any]:
    with _read_transaction(dsn) as connection:
        total = connection.execute("SELECT COUNT(*) AS count FROM rollout_cells").fetchone()[
            "count"
        ]
        local_results = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_local_results"
        ).fetchone()["count"]
        by_state_rows = connection.execute(
            "SELECT state, COUNT(*) AS count FROM rollout_cells GROUP BY state"
        ).fetchall()
        by_block = connection.execute(
            """
            SELECT serving_block, state, COUNT(*) AS count
            FROM rollout_cells
            GROUP BY serving_block, state
            ORDER BY serving_block, state
            """
        ).fetchall()
        plan = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'plan_sha256'"
        ).fetchone()
        stale = connection.execute(
            """
            SELECT COUNT(*) AS count FROM rollout_cells
            WHERE state IN ('claimed', 'running', 'grading')
              AND lease_expires_at IS NOT NULL
              AND lease_expires_at < CURRENT_TIMESTAMP
            """
        ).fetchone()["count"]
    by_state = {row["state"]: row["count"] for row in by_state_rows}
    return {
        "total": total,
        "local_results": local_results,
        "by_state": {state: by_state.get(state, 0) for state in rollout_ledger.STATES},
        "by_serving_block": by_block,
        "stale_active": stale,
        "plan_sha256": plan["value"] if plan else None,
    }
