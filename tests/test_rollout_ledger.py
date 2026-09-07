from __future__ import annotations

import csv
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evals.fleet.rollout_ledger import (
    LedgerError,
    accept,
    approve_retry,
    claim,
    export,
    heartbeat,
    initialize,
    mark_grading,
    mark_terminal,
    request_retry_review,
    stale_cells,
    start,
    summary,
)

PLAN_FIELDS = (
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
RECEIPT_DIGEST = "a" * 64
RECONCILIATION_DIGEST = "b" * 64


def _row(
    index: int, *, serving_block: str = "shared-qwen", max_retries: int = 1
) -> dict[str, object]:
    return {
        "experiment_id": "qwen-glm-pass4-v1",
        "task_key": f"task-{index}",
        "task_version_id": f"task-version-{index:03d}",
        "model_id": "qwen3.8-27b",
        "model_revision": "1d4bf0f2",
        "serving_block": serving_block,
        "endpoint_model_id": "qwen3.8-27b",
        "harness_id": "opencode-pinned-v1",
        "attempt": 1,
        "max_retries": max_retries,
    }


def _plan(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _claim_one(database: Path, worker_id: str = "worker-1") -> dict[str, object]:
    cell = claim(database, worker_id=worker_id, serving_block="shared-qwen")
    assert cell is not None
    return cell


def test_initialize_is_create_once_and_order_independent(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    rows = [_row(2), _row(1)]
    first = initialize(database, _plan(tmp_path / "plan.csv", rows))
    second = initialize(database, _plan(tmp_path / "reordered.csv", list(reversed(rows))))

    assert first["created"] is True
    assert second == {**first, "created": False}
    assert oct(database.stat().st_mode & 0o777) == "0o600"

    changed = [dict(row) for row in rows]
    for row in changed:
        row["endpoint_model_id"] = "different-endpoint"
    with pytest.raises(LedgerError, match="different rollout plan"):
        initialize(database, _plan(tmp_path / "changed.csv", changed))

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE rollout_cells SET endpoint_model_id = 'tampered' WHERE task_version_id = ?",
            (rows[0]["task_version_id"],),
        )
    with pytest.raises(LedgerError, match="immutable cell bindings"):
        initialize(database, _plan(tmp_path / "original.csv", rows))


def test_concurrent_workers_never_claim_the_same_cell(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    initialize(database, _plan(tmp_path / "plan.csv", [_row(index) for index in range(24)]))

    def take(index: int) -> dict[str, object] | None:
        return claim(database, worker_id=f"worker-{index}", serving_block="shared-qwen")

    with ThreadPoolExecutor(max_workers=12) as executor:
        claimed = list(executor.map(take, range(24)))

    cell_ids = [cell["cell_id"] for cell in claimed if cell is not None]
    assert len(cell_ids) == 24
    assert len(set(cell_ids)) == 24
    assert claim(database, worker_id="extra", serving_block="shared-qwen") is None
    assert summary(database)["by_state"]["claimed"] == 24


def test_valid_rollout_follows_owned_state_machine(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    initialize(database, _plan(tmp_path / "plan.csv", [_row(1)]))
    cell = _claim_one(database)
    owner = {
        "cell_id": str(cell["cell_id"]),
        "worker_id": "worker-1",
        "claim_id": str(cell["claim_id"]),
    }

    with pytest.raises(LedgerError, match="does not own"):
        heartbeat(database, **{**owner, "worker_id": "other-worker"})

    running = start(database, **owner, session_id="fleet-session-1")
    assert running["state"] == "running"
    assert heartbeat(database, **owner)["state"] == "running"
    assert mark_grading(database, **owner)["state"] == "grading"
    accepted = accept(database, **owner, receipt_digest=f"sha256:{RECEIPT_DIGEST}")

    assert accepted["state"] == "accepted"
    assert accepted["result_class"] == "valid"
    assert accepted["receipt_digest"] == RECEIPT_DIGEST
    assert accepted["lease_expires_at"] is None
    with pytest.raises(LedgerError, match="not active"):
        heartbeat(database, **owner)


def test_expired_lease_is_reported_but_never_reclaimed_automatically(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    initialize(database, _plan(tmp_path / "plan.csv", [_row(1)]))
    cell = _claim_one(database)
    expired = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE rollout_cells SET lease_expires_at = ? WHERE cell_id = ?",
            (expired, cell["cell_id"]),
        )

    stale = stale_cells(database)
    assert [row["cell_id"] for row in stale] == [cell["cell_id"]]
    assert claim(database, worker_id="replacement", serving_block="shared-qwen") is None


def test_retry_requires_review_receipt_and_obeys_allowance(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    initialize(database, _plan(tmp_path / "plan.csv", [_row(1, max_retries=1)]))
    first = _claim_one(database)
    first_owner = {
        "cell_id": str(first["cell_id"]),
        "worker_id": "worker-1",
        "claim_id": str(first["claim_id"]),
    }
    start(database, **first_owner, session_id="failed-session-1")
    reviewed = request_retry_review(
        database, **first_owner, failure_code="runner_exit_before_acceptance"
    )
    assert reviewed["state"] == "retry_review"

    with pytest.raises(LedgerError, match="SHA-256"):
        approve_retry(
            database, cell_id=first_owner["cell_id"], reconciliation_digest="not-a-digest"
        )

    pending = approve_retry(
        database,
        cell_id=first_owner["cell_id"],
        reconciliation_digest=RECONCILIATION_DIGEST,
    )
    assert pending["state"] == "pending"
    assert pending["retry_count"] == 1
    assert pending["session_id"] is None

    second = _claim_one(database, "worker-2")
    second_owner = {
        "cell_id": str(second["cell_id"]),
        "worker_id": "worker-2",
        "claim_id": str(second["claim_id"]),
    }
    request_retry_review(database, **second_owner, failure_code="second_infrastructure_failure")
    with pytest.raises(LedgerError, match="exhausted"):
        approve_retry(
            database,
            cell_id=second_owner["cell_id"],
            reconciliation_digest=RECONCILIATION_DIGEST,
        )
    terminal = mark_terminal(
        database,
        cell_id=second_owner["cell_id"],
        reconciliation_digest=RECONCILIATION_DIGEST,
    )
    assert terminal["state"] == "terminal"


def test_csv_and_event_exports_are_atomic_human_views(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    initialize(database, _plan(tmp_path / "plan.csv", [_row(1), _row(2)]))
    cell = _claim_one(database)
    owner = {
        "cell_id": str(cell["cell_id"]),
        "worker_id": "worker-1",
        "claim_id": str(cell["claim_id"]),
    }
    start(database, **owner, session_id="fleet-session-1")
    accept(database, **owner, receipt_digest=RECEIPT_DIGEST)

    csv_path = tmp_path / "progress.csv"
    events_path = tmp_path / "events.jsonl"
    result = export(database, csv_path=csv_path, events_path=events_path)

    assert result["cells"] == 2
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["state"] for row in rows} == {"accepted", "pending"}
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert events[-1]["event"] == "accepted"
    assert oct(csv_path.stat().st_mode & 0o777) == "0o600"
    assert oct(events_path.stat().st_mode & 0o777) == "0o600"


def test_plan_rejects_duplicate_scientific_cell(tmp_path: Path) -> None:
    row = _row(1)
    with pytest.raises(LedgerError, match="duplicates rollout identity"):
        initialize(tmp_path / "ledger.sqlite3", _plan(tmp_path / "plan.csv", [row, dict(row)]))


def test_plan_rejects_multiple_experiments_in_one_database(tmp_path: Path) -> None:
    first = _row(1)
    second = {**_row(2), "experiment_id": "another-experiment"}
    with pytest.raises(LedgerError, match="exactly one experiment_id"):
        initialize(
            tmp_path / "ledger.sqlite3",
            _plan(tmp_path / "plan.csv", [first, second]),
        )


def test_plan_keeps_all_attempts_on_one_serving_block_and_contiguous(tmp_path: Path) -> None:
    first = _row(1)
    second = {**first, "attempt": 2}
    split = {
        **second,
        "serving_block": "dedicated-qwen",
        "endpoint_model_id": "chris-cyber-qwen38-27b-dedicated-v1",
    }
    with pytest.raises(LedgerError, match="attempts split identity or serving block"):
        initialize(
            tmp_path / "split.sqlite3",
            _plan(tmp_path / "split.csv", [first, split]),
        )

    third = {**first, "attempt": 3}
    with pytest.raises(LedgerError, match="attempts are not contiguous"):
        initialize(
            tmp_path / "gap.sqlite3",
            _plan(tmp_path / "gap.csv", [first, third]),
        )


def test_failure_code_rejects_free_form_or_multiline_content(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    initialize(database, _plan(tmp_path / "plan.csv", [_row(1)]))
    cell = _claim_one(database)
    owner = {
        "cell_id": str(cell["cell_id"]),
        "worker_id": "worker-1",
        "claim_id": str(cell["claim_id"]),
    }
    with pytest.raises(LedgerError, match="sanitized machine code"):
        request_retry_review(database, **owner, failure_code="raw error details are forbidden")


def test_database_contains_no_score_or_trace_columns(tmp_path: Path) -> None:
    database = tmp_path / "ledger.sqlite3"
    initialize(database, _plan(tmp_path / "plan.csv", [_row(1)]))
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(rollout_cells)")}
    assert "score" not in columns
    assert "trace" not in columns
    assert os.path.exists(database)
