"""Losslessly migrate an archived rollout ledger from SQLite to PostgreSQL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from psycopg import sql

from evals.fleet import rollout_postgres

TABLES = (
    ("ledger_metadata", "key"),
    ("rollout_cells", "cell_id"),
    ("rollout_events", "sequence"),
    ("rollout_local_results", "execution_id"),
)
TIMESTAMP_COLUMNS = {
    "started_at",
    "heartbeat_at",
    "lease_expires_at",
    "completed_at",
    "created_at",
    "updated_at",
    "recorded_at",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _self_digest(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json_once(path: Path, value: object) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_self_digest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"receipt is not an object: {path.name}")
    expected = str(value.get("receipt_sha256") or "").removeprefix("sha256:")
    if not expected or _self_digest(value) != expected:
        raise RuntimeError(f"receipt digest is invalid: {path.name}")
    return value


def _validate_archive(root: Path, sqlite_path: Path) -> dict[str, Any]:
    root = root.resolve()
    if sqlite_path.resolve() != root / "ledger-consolidated.sqlite3":
        raise RuntimeError("migration source must be the archive's consolidated ledger")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("archive must not contain symbolic links")
    receipt_path = root / "archive-receipt.json"
    manifest_path = root / "archive-manifest.json"
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("archive proof files are missing")
    receipt = _read_self_digest(receipt_path)
    if _sha256(manifest_path) != receipt.get("manifest_sha256"):
        raise RuntimeError("archive manifest digest differs from its receipt")
    if _sha256(sqlite_path) != receipt.get("consolidated_ledger_sha256"):
        raise RuntimeError("archived SQLite digest differs from its receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or len(entries) != receipt.get("file_count"):
        raise RuntimeError("archive file count differs from its receipt")
    total_bytes = 0
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("archive manifest entry is invalid")
        relative = Path(str(entry.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise RuntimeError("archive manifest contains an unsafe path")
        if relative in seen:
            raise RuntimeError("archive manifest contains a duplicate path")
        seen.add(relative)
        archived = root / relative
        if not archived.is_file():
            raise RuntimeError("archive manifest file is missing")
        size = archived.stat().st_size
        if size != entry.get("bytes") or _sha256(archived) != entry.get("sha256"):
            raise RuntimeError("archive file differs from its manifest")
        total_bytes += size
    if total_bytes != receipt.get("total_bytes"):
        raise RuntimeError("archive byte count differs from its receipt")
    return receipt


def _normalize(value: Any, *, timestamp: bool) -> Any:
    if value is None:
        return None
    if timestamp:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return parsed.astimezone(UTC).isoformat()
    return value


def _normalized_rows(rows: list[dict[str, Any]], columns: list[str]) -> list[list[Any]]:
    return [
        [_normalize(row[column], timestamp=column in TIMESTAMP_COLUMNS) for column in columns]
        for row in rows
    ]


def _rows_digest(rows: list[dict[str, Any]], columns: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            _normalized_rows(rows, columns),
            sort_keys=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _sqlite_rows(
    connection: sqlite3.Connection, table: str, order_column: str
) -> tuple[list[str], list[dict[str, Any]]]:
    columns = [str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")]
    rows = [
        dict(row)
        for row in connection.execute(f"SELECT * FROM {table} ORDER BY {order_column}")  # noqa: S608
    ]
    return columns, rows


def _postgres_rows(
    connection: Any, table: str, order_column: str, columns: list[str]
) -> list[dict[str, Any]]:
    return connection.execute(
        sql.SQL("SELECT {} FROM {} ORDER BY {}").format(
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.Identifier(table),
            sql.Identifier(order_column),
        )
    ).fetchall()


def migrate(
    sqlite_path: Path, dsn: str, receipt_path: Path, archive_root: Path | None = None
) -> dict[str, Any]:
    if not sqlite_path.is_file():
        raise RuntimeError("consolidated SQLite archive is missing")
    if receipt_path.exists():
        raise RuntimeError("migration receipt already exists")

    archive_receipt = _validate_archive(archive_root, sqlite_path) if archive_root else None
    sqlite_sha256 = _sha256(sqlite_path)

    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as source:
        source.row_factory = sqlite3.Row
        source.execute("PRAGMA query_only = ON")
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("archived SQLite integrity check failed")
        source_tables = {
            table: _sqlite_rows(source, table, order_column) for table, order_column in TABLES
        }

    source_counts = {table: len(rows) for table, (_columns, rows) in source_tables.items()}
    source_digests = {
        table: _rows_digest(rows, columns) for table, (columns, rows) in source_tables.items()
    }

    with rollout_postgres._connect(dsn) as target, target.transaction():  # noqa: SLF001
        rollout_postgres.ensure_schema(target)
        existing = {
            table: target.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()[  # noqa: S608
                "count"
            ]
            for table, _order in TABLES
        }
        if any(existing.values()):
            raise RuntimeError("PostgreSQL migration target is not empty")

        for table, _order_column in TABLES:
            columns, rows = source_tables[table]
            statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(table),
                sql.SQL(", ").join(map(sql.Identifier, columns)),
                sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            )
            for row in rows:
                target.execute(statement, [row[column] for column in columns])

        target.execute(
            """
            SELECT setval(
                pg_get_serial_sequence('rollout_events', 'sequence'),
                COALESCE((SELECT MAX(sequence) FROM rollout_events), 1),
                EXISTS(SELECT 1 FROM rollout_events)
            )
            """
        )

        target_counts: dict[str, int] = {}
        target_digests: dict[str, str] = {}
        for table, order_column in TABLES:
            columns, _rows = source_tables[table]
            migrated = _postgres_rows(target, table, order_column, columns)
            target_counts[table] = len(migrated)
            target_digests[table] = _rows_digest(migrated, columns)
        if target_counts != source_counts or target_digests != source_digests:
            raise RuntimeError("PostgreSQL row-level parity check failed")

        state_rows = target.execute(
            "SELECT state, COUNT(*) AS count FROM rollout_cells GROUP BY state ORDER BY state"
        ).fetchall()
        state_counts = {row["state"]: row["count"] for row in state_rows}
        receipt = {
            "schema_version": "fleet-rollout-sqlite-postgresql-migration-receipt-v1",
            "sqlite_sha256": sqlite_sha256,
            "archive_receipt_sha256": (
                archive_receipt["receipt_sha256"] if archive_receipt else None
            ),
            "archive_manifest_sha256": (
                archive_receipt["manifest_sha256"] if archive_receipt else None
            ),
            "terminal_receipts": (archive_receipt["terminal_receipts"] if archive_receipt else []),
            "source_counts": source_counts,
            "target_counts": target_counts,
            "source_row_digests": source_digests,
            "target_row_digests": target_digests,
            "ledger_state_counts": state_counts,
            "source_preserved": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        receipt["receipt_sha256"] = _self_digest(receipt)
        target.execute(
            """
            INSERT INTO ledger_migrations (
                source_kind, source_sha256, source_row_digests_json,
                target_row_digests_json, receipt_sha256, completed_at
            ) VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (
                "sqlite",
                sqlite_sha256,
                json.dumps(source_digests, sort_keys=True, separators=(",", ":")),
                json.dumps(target_digests, sort_keys=True, separators=(",", ":")),
                receipt["receipt_sha256"],
            ),
        )

    # The four migrated source tables remain byte-for-byte equivalent at the
    # normalized row level. Migration provenance lives in its own table so the
    # source rows are not rewritten after parity is proven.
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_once(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--dsn-env", default="ROLLOUT_DATABASE_URL")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path)
    arguments = parser.parse_args()
    dsn = os.environ.get(arguments.dsn_env)
    if not dsn:
        raise RuntimeError(f"{arguments.dsn_env} is required")
    try:
        receipt = migrate(arguments.sqlite, dsn, arguments.receipt, arguments.archive_root)
    except Exception:
        # Driver diagnostics can contain connection strings; do not print them.
        print(
            json.dumps({"migrated": False, "error_code": "migration_failed_reconcile_before_retry"})
        )
        return 1
    status = json.dumps(
        {
            "migrated": True,
            "source_counts": receipt["source_counts"],
            "target_counts": receipt["target_counts"],
            "ledger_state_counts": receipt["ledger_state_counts"],
            "terminal_receipts": receipt["terminal_receipts"],
            "receipt_sha256": receipt["receipt_sha256"],
        },
        sort_keys=True,
    )
    if arguments.status_output:
        arguments.status_output.write_text(status + "\n", encoding="utf-8")
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
