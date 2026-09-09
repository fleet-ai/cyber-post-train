from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import rollout_archive, rollout_ledger, rollout_postgres_migrate


def _digest(value: dict[str, object]) -> str:
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_archive_preserves_tree_and_consolidates_ledger(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    plan = source / "plan.csv"
    plan.write_text(
        "experiment_id,task_key,task_version_id,model_id,model_revision,"
        "serving_block,endpoint_model_id,harness_id,attempt,max_retries\n"
        "campaign,task,version,model,revision,route,endpoint,harness,1,1\n",
        encoding="utf-8",
    )
    rollout_ledger.initialize(source / "ledger.sqlite3", plan)
    artifact = source / "attempts" / "opaque" / "trace.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"opaque-private-evidence")
    for worker in ("worker-a", "worker-b"):
        terminal = {
            "schema_version": "fleet-rollout-ledger-controller-terminal-v1",
            "accepted": False,
            "worker_id": worker,
            "requested_cells": 0,
            "accepted_cells": 0,
            "results": [],
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        terminal["receipt_sha256"] = _digest(terminal)
        (source / f"TERMINAL-{worker}.json").write_text(json.dumps(terminal), encoding="utf-8")

    destination = tmp_path / "archive"
    receipt = rollout_archive.archive(source, destination, ["worker-a", "worker-b"])

    archived_artifact = destination / "campaign" / artifact.relative_to(source)
    assert archived_artifact.read_bytes() == artifact.read_bytes()
    assert receipt["ledger_counts"]["cells"] == 1
    assert receipt["ledger_counts"]["by_state"] == {"pending": 1}
    assert len(receipt["terminal_receipts"]) == 2
    persisted = json.loads((destination / "archive-receipt.json").read_text(encoding="utf-8"))
    assert persisted["receipt_sha256"] == _digest(persisted)
    validated = rollout_postgres_migrate._validate_archive(  # noqa: SLF001
        destination, destination / "ledger-consolidated.sqlite3"
    )
    assert validated["receipt_sha256"] == persisted["receipt_sha256"]


def test_archive_rejects_nested_destination_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "ledger.sqlite3").touch()
    with pytest.raises(RuntimeError, match="outside"):
        rollout_archive.archive(source, source / "archive", [])
    assert not (source / "archive").exists()


def test_archive_rejects_symlink_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "ledger.sqlite3").touch()
    (source / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(RuntimeError, match="symbolic links"):
        rollout_archive.archive(source, tmp_path / "archive", [])


def test_terminal_summary_requires_explicit_content_exclusion(tmp_path: Path) -> None:
    receipt = {"worker_id": "synthetic", "accepted": False}
    receipt["receipt_sha256"] = _digest(receipt)
    path = tmp_path / "TERMINAL.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="sanitized"):
        rollout_archive._terminal_summary(path)


def test_migration_cli_does_not_expose_driver_error(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("SYNTHETIC_DATABASE", "postgresql://synthetic-secret@localhost/test")
    monkeypatch.setattr(
        "sys.argv",
        [
            "migration",
            "--sqlite",
            str(tmp_path / "source.sqlite3"),
            "--archive-root",
            str(tmp_path),
            "--receipt",
            str(tmp_path / "receipt.json"),
            "--dsn-env",
            "SYNTHETIC_DATABASE",
        ],
    )

    def fail(_source, dsn, *_args):
        raise RuntimeError(dsn)

    monkeypatch.setattr(rollout_postgres_migrate, "migrate", fail)
    assert rollout_postgres_migrate.main() == 1
    output = capsys.readouterr()
    assert "migration_failed_reconcile_before_retry" in output.out
    assert "synthetic-secret" not in output.out + output.err
