"""Create a lossless, create-once archive of a quiescent rollout campaign.

The archive copies every campaign artifact without interpreting prompt, trace,
answer, flag, or score content.  It additionally creates a consolidated SQLite
snapshot and a private file manifest so later migration can prove byte-level
preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


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


def _terminal_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("scores_included") is not False
        or value.get("prompts_or_traces_included") is not False
    ):
        raise RuntimeError("terminal receipt is not explicitly sanitized")
    expected = str(value.get("receipt_sha256") or "").removeprefix("sha256:")
    if not expected or _self_digest(value) != expected:
        raise RuntimeError(f"terminal receipt digest is invalid: {path.name}")
    results = value.get("results") if isinstance(value.get("results"), list) else []
    return {
        "worker_id": value.get("worker_id"),
        "accepted": value.get("accepted"),
        "requested_cells": value.get("requested_cells"),
        "accepted_cells": value.get("accepted_cells"),
        "controller_failure_code": value.get("controller_failure_code"),
        "result_count": len(results),
        "ledger_cell_ids": sorted(
            str(row["ledger_cell_id"])
            for row in results
            if isinstance(row, dict) and row.get("ledger_cell_id")
        ),
        "failure_codes": sorted(
            str(row["failure_code"])
            for row in results
            if isinstance(row, dict) and row.get("failure_code")
        ),
        "receipt_sha256": expected,
    }


def _manifest(root: Path, *, excluded: set[Path]) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if relative in excluded:
            continue
        size = path.stat().st_size
        entries.append({"path": relative.as_posix(), "bytes": size, "sha256": _sha256(path)})
        total_bytes += size
    return entries, total_bytes


def archive(source: Path, destination: Path, terminal_workers: list[str]) -> dict[str, Any]:
    if not source.is_dir():
        raise RuntimeError("campaign source directory is missing")
    database = source / "ledger.sqlite3"
    if not database.is_file():
        raise RuntimeError("campaign SQLite ledger is missing")
    if destination.exists():
        raise RuntimeError("archive destination already exists")
    if destination.resolve().is_relative_to(source.resolve()):
        raise RuntimeError("archive destination must be outside the source tree")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise RuntimeError("archive source must not contain symbolic links")
    if any(not worker or Path(worker).name != worker for worker in terminal_workers):
        raise RuntimeError("terminal worker identity must be a filename component")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        copied = temporary / "campaign"
        shutil.copytree(source, copied, copy_function=shutil.copy2)

        # Preserve ``campaign`` as an exact raw copy. SQLite recovery may write
        # WAL bookkeeping even for a logical read, so perform recovery only on
        # a second scratch copy located on the archive's node-local CSI disk.
        recovery = temporary / "sqlite-recovery"
        recovery.mkdir()
        for item in copied.glob("ledger.sqlite3*"):
            if item.is_file():
                shutil.copy2(item, recovery / item.name)
        recovery_database = recovery / "ledger.sqlite3"
        if not recovery_database.is_file():
            raise RuntimeError("copied SQLite ledger is missing")

        consolidated = temporary / "ledger-consolidated.sqlite3"
        source_connection = sqlite3.connect(recovery_database, timeout=60)
        try:
            source_connection.execute("PRAGMA query_only = ON")
            integrity = source_connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError("source SQLite integrity check failed")
            target_connection = sqlite3.connect(consolidated)
            try:
                source_connection.backup(target_connection)
                target_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                target_connection.commit()
            finally:
                target_connection.close()
        finally:
            source_connection.close()
        os.chmod(consolidated, 0o600)

        with sqlite3.connect(f"file:{consolidated}?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            check = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise RuntimeError("consolidated SQLite integrity check failed")
            state_counts = {
                str(row["state"]): int(row["count"])
                for row in connection.execute(
                    "SELECT state, COUNT(*) AS count FROM rollout_cells GROUP BY state"
                )
            }
            counts = {
                "cells": connection.execute("SELECT COUNT(*) FROM rollout_cells").fetchone()[0],
                "events": connection.execute("SELECT COUNT(*) FROM rollout_events").fetchone()[0],
                "local_results": connection.execute(
                    "SELECT COUNT(*) FROM rollout_local_results"
                ).fetchone()[0],
                "by_state": state_counts,
            }

        terminal_summaries = []
        for worker in terminal_workers:
            receipt = source / f"TERMINAL-{worker}.json"
            if not receipt.is_file():
                raise RuntimeError(f"required terminal receipt is missing: {worker}")
            terminal_summaries.append(_terminal_summary(receipt))

        excluded = {Path("archive-manifest.json"), Path("archive-receipt.json")}
        entries, total_bytes = _manifest(temporary, excluded=excluded)
        manifest = {
            "schema_version": "fleet-rollout-lossless-archive-manifest-v1",
            "files": entries,
        }
        _write_json_once(temporary / "archive-manifest.json", manifest)
        manifest_digest = _sha256(temporary / "archive-manifest.json")
        receipt = {
            "schema_version": "fleet-rollout-lossless-archive-receipt-v1",
            "source_directory": str(source),
            "file_count": len(entries),
            "total_bytes": total_bytes,
            "manifest_sha256": manifest_digest,
            "consolidated_ledger_sha256": _sha256(consolidated),
            "ledger_counts": counts,
            "terminal_receipts": terminal_summaries,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        receipt["receipt_sha256"] = _self_digest(receipt)
        _write_json_once(temporary / "archive-receipt.json", receipt)
        os.sync()
        os.replace(temporary, destination)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--terminal-worker", action="append", default=[])
    parser.add_argument("--status-output", type=Path)
    arguments = parser.parse_args()
    try:
        receipt = archive(arguments.source, arguments.destination, arguments.terminal_worker)
        status = {
            "archived": True,
            "file_count": receipt["file_count"],
            "total_bytes": receipt["total_bytes"],
            "ledger_counts": receipt["ledger_counts"],
            "receipt_sha256": receipt["receipt_sha256"],
        }
    except Exception as error:
        status = {
            "archived": False,
            "error_type": type(error).__name__,
            "error_code": (
                "sqlite_recovery_failed"
                if isinstance(error, sqlite3.Error)
                else (
                    "source_tree_copy_failed"
                    if isinstance(error, (OSError, shutil.Error))
                    else "archive_validation_failed"
                )
            ),
        }
        if arguments.status_output:
            arguments.status_output.write_text(
                json.dumps(status, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(status, sort_keys=True))
        return 1
    encoded = json.dumps(status, sort_keys=True)
    if arguments.status_output:
        arguments.status_output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
