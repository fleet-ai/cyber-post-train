"""Integration tests only against an explicitly opted-in, local disposable database."""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from evals.fleet import rollout_ledger, rollout_postgres, rollout_postgres_migrate


def _isolated_dsn(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
        or parsed.path != "/cyber_post_train_test"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "tests require a local cyber_post_train_test database without query options"
        )
    query = parse_qsl(parsed.query) + [("options", f"-csearch_path={schema}")]
    return urlunsplit(parsed._replace(query=urlencode(query)))


@pytest.fixture
def pg_dsn():
    dsn = os.environ.get("TEST_POSTGRES_DSN")
    if not dsn or os.environ.get("TEST_POSTGRES_DISPOSABLE") != "1":
        pytest.skip("explicit local disposable PostgreSQL opt-in is not configured")
    schema = "test_rollout_" + uuid.uuid4().hex
    isolated = _isolated_dsn(dsn, schema)  # Validate before opening any connection.
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield isolated
    finally:
        # Only the exact fresh schema created above; never drop shared ledger tables.
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _plan(path: Path, count: int = 24) -> Path:
    rows = [
        "experiment_id,task_key,task_version_id,model_id,model_revision,"
        "serving_block,endpoint_model_id,harness_id,attempt,max_retries"
    ]
    rows.extend(
        f"synthetic,task-{i},version-{i},model,revision,route,endpoint,harness,1,1"
        for i in range(count)
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://production/rollout",
        "postgresql://localhost/rollout",
        "postgresql://localhost/cyber_post_train_test?host=production",
        "postgresql://localhost/cyber_post_train_test?options=-csearch_path=public",
    ],
)
def test_integration_fixture_rejects_unsafe_database(dsn):
    with pytest.raises(ValueError, match="local"):
        _isolated_dsn(dsn, "test_rollout_synthetic")


def test_lossless_migration_and_concurrent_claims(tmp_path: Path, pg_dsn: str) -> None:
    plan = _plan(tmp_path / "plan.csv")
    source = tmp_path / "ledger.sqlite3"
    rollout_ledger.initialize(source, plan)
    receipt = rollout_postgres_migrate.migrate(source, pg_dsn, tmp_path / "migration.json")
    assert receipt["source_counts"] == receipt["target_counts"]
    assert receipt["source_row_digests"] == receipt["target_row_digests"]
    assert rollout_postgres.verify_plan(pg_dsn, plan)["cells"] == 24

    def take(index):
        return rollout_postgres.claim(pg_dsn, worker_id=f"worker-{index}", serving_block="route")

    with ThreadPoolExecutor(max_workers=12) as executor:
        claims = list(executor.map(take, range(24)))
    assert len({row["cell_id"] for row in claims}) == 24
    assert rollout_postgres.claim(pg_dsn, worker_id="extra", serving_block="route") is None
    assert rollout_postgres.claim(pg_dsn, worker_id="wrong-route", serving_block="other") is None

    first = claims[0]
    with pytest.raises(rollout_ledger.LedgerError):
        rollout_postgres.heartbeat(
            pg_dsn,
            cell_id=first["cell_id"],
            worker_id="wrong-owner",
            claim_id=first["claim_id"],
        )
    with psycopg.connect(pg_dsn) as connection:
        connection.execute(
            "UPDATE rollout_cells SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' "
            "WHERE cell_id = %s",
            (first["cell_id"],),
        )
    assert rollout_postgres.active_claims(pg_dsn)["stale_active"] == 1
    assert rollout_postgres.summary(pg_dsn)["by_state"]["claimed"] == 24
    assert (
        rollout_postgres.claim(pg_dsn, worker_id="no-automatic-retry", serving_block="route")
        is None
    )
    with pytest.raises(RuntimeError, match="not empty"):
        rollout_postgres_migrate.migrate(source, pg_dsn, tmp_path / "second.json")


def test_observations_do_not_create_schema(pg_dsn: str) -> None:
    for operation in (rollout_postgres.summary, rollout_postgres.active_claims):
        with pytest.raises(psycopg.errors.UndefinedTable):
            operation(pg_dsn)
    with psycopg.connect(pg_dsn) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=current_schema()"
            ).fetchone()[0]
            == 0
        )


def test_observation_transaction_cannot_write(pg_dsn: str) -> None:
    with rollout_postgres._read_transaction(pg_dsn) as connection:
        assert (
            connection.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"]
            == "on"
        )
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            connection.execute("CREATE TABLE forbidden_test_write (id int)")


def test_fresh_initialization_is_exact_and_never_replaces(tmp_path, pg_dsn):
    plan = _plan(tmp_path / "plan.csv", count=3)
    created = rollout_postgres.initialize(pg_dsn, plan)
    assert created["created"] and created["cells"] == 3
    assert rollout_postgres.verify_plan(pg_dsn, plan) == {**created, "created": False}
    assert rollout_postgres.summary(pg_dsn)["by_state"]["pending"] == 3
    for candidate in (plan, _plan(tmp_path / "other.csv", count=4)):
        with pytest.raises(rollout_ledger.LedgerError, match="empty dedicated"):
            rollout_postgres.initialize(pg_dsn, candidate)
    with psycopg.connect(pg_dsn) as connection:
        assert connection.execute("SELECT count(*) FROM rollout_events").fetchone()[0] == 3
        assert connection.execute("SELECT count(*) FROM rollout_local_results").fetchone()[0] == 0


def test_concurrent_initializers_cannot_duplicate_rows(tmp_path, pg_dsn):
    plan = _plan(tmp_path / "plan.csv", count=3)

    def create(_):
        try:
            return rollout_postgres.initialize(pg_dsn, plan)["created"]
        except rollout_ledger.LedgerError:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(create, range(4))) == 1
    assert rollout_postgres.verify_plan(pg_dsn, plan)["cells"] == 3


def test_metadata_without_cells_is_not_empty_target(tmp_path, pg_dsn):
    with psycopg.connect(pg_dsn) as connection:
        rollout_postgres.ensure_schema(connection)
        connection.execute("INSERT INTO ledger_metadata VALUES ('old', 'preserve')")
    with pytest.raises(rollout_ledger.LedgerError, match="empty dedicated"):
        rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv"))
    with psycopg.connect(pg_dsn) as connection:
        assert connection.execute("SELECT count(*) FROM rollout_cells").fetchone()[0] == 0
        assert connection.execute("SELECT value FROM ledger_metadata").fetchone()[0] == "preserve"


@pytest.fixture
def owned_cell(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=1))
    cell = rollout_postgres.claim(pg_dsn, worker_id="worker", serving_block="route")
    return {key: cell[key] for key in ("cell_id", "worker_id", "claim_id")}


def _events(dsn):
    with psycopg.connect(dsn) as connection:
        return connection.execute(
            "SELECT event, from_state, to_state, detail_json FROM rollout_events ORDER BY sequence"
        ).fetchall()


@pytest.mark.parametrize("grade_first", [False, True])
def test_real_postgres_acceptance_and_terminal_fencing(pg_dsn, owned_cell, grade_first):
    renewed = rollout_postgres.heartbeat(pg_dsn, **owned_cell, lease_seconds=1800)
    assert (renewed["lease_expires_at"] - renewed["heartbeat_at"]).total_seconds() == 1800
    started = rollout_postgres.start(pg_dsn, **owned_cell, session_id="synthetic-session")
    assert started["state"] == "running" and started["session_id"] == "synthetic-session"
    if grade_first:
        assert rollout_postgres.mark_grading(pg_dsn, **owned_cell)["state"] == "grading"
    receipt = "a" * 64
    accepted = rollout_postgres.accept(pg_dsn, **owned_cell, receipt_digest=receipt)
    assert accepted["state"] == "accepted" and accepted["result_class"] == "valid"
    assert accepted["receipt_digest"] == receipt and accepted["lease_expires_at"] is None
    before = _events(pg_dsn)
    for operation in (
        lambda: rollout_postgres.heartbeat(pg_dsn, **owned_cell),
        lambda: rollout_postgres.accept(pg_dsn, **owned_cell, receipt_digest=receipt),
        lambda: rollout_postgres.request_retry_review(pg_dsn, **owned_cell, failure_code="error"),
    ):
        with pytest.raises(rollout_ledger.LedgerError, match="not active"):
            operation()
    assert _events(pg_dsn) == before
    assert rollout_postgres.claim(pg_dsn, worker_id="later", serving_block="route") is None
    assert rollout_postgres.active_claims(pg_dsn)["owners"] == []
    assert rollout_postgres.summary(pg_dsn)["by_state"]["accepted"] == 1


@pytest.mark.parametrize("state", ["claimed", "running", "grading"])
def test_review_preserves_nonrepeatable_outcome_from_each_active_state(pg_dsn, owned_cell, state):
    if state != "claimed":
        rollout_postgres.start(pg_dsn, **owned_cell, session_id="synthetic-session")
    if state == "grading":
        rollout_postgres.mark_grading(pg_dsn, **owned_cell)
    held = rollout_postgres.request_retry_review(pg_dsn, **owned_cell, failure_code="output_limit")
    assert held["state"] == "retry_review" and held["lease_expires_at"] is None
    assert held["failure_code"] == "output_limit" and held["receipt_digest"] is None
    assert rollout_postgres.claim(pg_dsn, worker_id="no-retry", serving_block="route") is None
    event = _events(pg_dsn)[-1]
    assert event[:3] == ("retry_review_requested", state, "retry_review")
    assert json.loads(event[3]) == {"failure_code": "output_limit"}


@pytest.mark.parametrize("field", ["cell_id", "worker_id", "claim_id"])
def test_wrong_owner_or_missing_cell_cannot_write(pg_dsn, owned_cell, field):
    before = _events(pg_dsn)
    owner = {**owned_cell, field: "not-the-owner"}
    with pytest.raises(rollout_ledger.LedgerError):
        rollout_postgres.start(pg_dsn, **owner, session_id="forbidden")
    assert _events(pg_dsn) == before
    assert rollout_postgres.summary(pg_dsn)["by_state"]["claimed"] == 1


def test_expired_heartbeat_and_invalid_state_transition_are_atomic(pg_dsn, owned_cell):
    before = _events(pg_dsn)
    with pytest.raises(rollout_ledger.LedgerError, match="cannot transition"):
        rollout_postgres.mark_grading(pg_dsn, **owned_cell)
    with psycopg.connect(pg_dsn) as connection:
        connection.execute(
            "UPDATE rollout_cells SET lease_expires_at = clock_timestamp() - INTERVAL '1 second'"
        )
    with pytest.raises(rollout_ledger.LedgerError, match="lease has expired"):
        rollout_postgres.heartbeat(pg_dsn, **owned_cell)
    assert _events(pg_dsn) == before
    assert rollout_postgres.active_claims(pg_dsn)["stale_active"] == 1
    assert rollout_postgres.claim(pg_dsn, worker_id="no-retry", serving_block="route") is None


def test_claim_skips_locked_and_explicitly_excluded_rows(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=3))
    parsed = urlsplit(pg_dsn)
    options = dict(parse_qsl(parsed.query))
    options["options"] += " -cstatement_timeout=3000"
    # libpq URI values need percent-encoded spaces, not HTML-form '+' spaces.
    bounded_dsn = urlunsplit(parsed._replace(query=urlencode(options, quote_via=quote)))
    with psycopg.connect(pg_dsn) as blocker:
        blocker.execute(
            "SELECT cell_id FROM rollout_cells WHERE task_version_id='version-0' FOR UPDATE"
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                rollout_postgres.claim,
                bounded_dsn,
                worker_id="other",
                serving_block="route",
                excluded_task_versions=["version-1"],
            )
            taken = future.result(timeout=5)
        assert taken["task_version_id"] == "version-2"
    next_cell = rollout_postgres.claim(pg_dsn, worker_id="next", serving_block="route")
    assert next_cell["task_version_id"] == "version-0"


def test_trigger_rejecting_claim_rolls_back_without_event(pg_dsn, owned_cell):
    # A real PostgreSQL trigger, not a fake cursor, exercises unexpected lost UPDATE.
    with psycopg.connect(pg_dsn) as connection:
        connection.execute(
            "UPDATE rollout_cells SET state='pending', worker_id=NULL, claim_id=NULL"
        )
        connection.execute("""
            CREATE FUNCTION skip_claim() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RETURN NULL; END $$;
            CREATE TRIGGER skip_claim BEFORE UPDATE ON rollout_cells
            FOR EACH ROW EXECUTE FUNCTION skip_claim();
        """)
    before = _events(pg_dsn)
    with pytest.raises(rollout_ledger.LedgerError, match="claim lost"):
        rollout_postgres.claim(pg_dsn, worker_id="rejected", serving_block="route")
    assert _events(pg_dsn) == before
    assert rollout_postgres.summary(pg_dsn)["by_state"]["pending"] == 1


@pytest.mark.parametrize("defect", ["metadata", "count", "binding"])
def test_verify_plan_rejects_each_independent_drift(tmp_path, pg_dsn, defect):
    plan = _plan(tmp_path / "plan.csv", count=1)
    rollout_postgres.initialize(pg_dsn, plan)
    with psycopg.connect(pg_dsn) as connection:
        if defect == "metadata":
            connection.execute("DELETE FROM ledger_metadata WHERE key='plan_sha256'")
        elif defect == "count":
            connection.execute("DELETE FROM rollout_events")
            connection.execute("DELETE FROM rollout_cells")
        else:
            connection.execute("UPDATE rollout_cells SET model_revision='different'")
    before = _events(pg_dsn)
    with pytest.raises(rollout_ledger.LedgerError, match="differ"):
        rollout_postgres.verify_plan(pg_dsn, plan)
    assert _events(pg_dsn) == before


def _local_record():
    return {
        "execution_id": "sha256:" + "1" * 64,
        "execution_generation": 1,
        "run_id": "synthetic-run",
        "session_id": "synthetic-session",
        "verifier_execution_id": "synthetic-verifier",
        "score": 0.25,
        "config_sha256": "2" * 64,
        "artifact_directory": "attempts/synthetic",
        **{
            f"{name}_path": name + ".json"
            for name in ("trace", "result", "reward", "session_ingest", "cleanup")
        },
        **{
            f"{name}_sha256": "3" * 64
            for name in ("trace", "result", "reward", "session_ingest", "cleanup")
        },
        "session_ingest_status": "completed",
        "agent_exit_code": 0,
        "agent_termination": "completed",
        "elapsed_seconds": 12.5,
    }


def test_local_result_is_private_idempotent_fenced_and_collision_safe(pg_dsn, owned_cell):
    record = _local_record()

    def store(value):
        return rollout_postgres.record_local_result(pg_dsn, **owned_cell, record=value)

    first = store(record)
    again = store(record)
    assert first["created"] is True and again == {**first, "created": False}
    before = _events(pg_dsn)
    with pytest.raises(rollout_ledger.LedgerError, match="different evidence"):
        store({**record, "score": 0.75})
    with pytest.raises(psycopg.errors.UniqueViolation):
        store({**record, "execution_id": "sha256:" + "4" * 64})
    with pytest.raises(rollout_ledger.LedgerError, match="does not own"):
        rollout_postgres.record_local_result(
            pg_dsn, **{**owned_cell, "claim_id": "stale"}, record=record
        )
    assert _events(pg_dsn) == before
    summary = rollout_postgres.summary(pg_dsn)
    assert summary["local_results"] == 1 and summary["by_state"]["accepted"] == 0
    assert "score" not in json.dumps(summary) and "score" not in before[-1][3]
    # Catalog failure does not erase the local copy or silently accept/retry it.
    rollout_postgres.request_retry_review(pg_dsn, **owned_cell, failure_code="catalog_failed")
    with pytest.raises(rollout_ledger.LedgerError, match="not active"):
        store(record)
    assert rollout_postgres.summary(pg_dsn)["local_results"] == 1


@pytest.mark.parametrize("session_id", [None, "synthetic-session"])
def test_shared_normalizer_preserves_sqlite_postgres_result_digests(
    tmp_path, pg_dsn, owned_cell, session_id
):
    database = tmp_path / "preserved.sqlite3"
    rollout_ledger.initialize(database, tmp_path / "plan.csv")
    old_cell = rollout_ledger.claim(database, worker_id="legacy", serving_block="route")
    assert old_cell["cell_id"] == owned_cell["cell_id"]
    old_owner = {key: old_cell[key] for key in ("cell_id", "worker_id", "claim_id")}
    record = {**_local_record(), "session_id": session_id}
    preserved = rollout_ledger.record_local_result(database, **old_owner, record=record)
    current = rollout_postgres.record_local_result(pg_dsn, **owned_cell, record=record)
    # Timestamp representations differ by storage engine, not scientific evidence.
    assert {k: v for k, v in preserved.items() if k != "recorded_at"} == {
        k: v for k, v in current.items() if k != "recorded_at"
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_generation", True),
        ("execution_generation", 0),
        ("execution_generation", 1.5),
        ("agent_exit_code", False),
        ("agent_exit_code", "0"),
        ("elapsed_seconds", True),
        ("elapsed_seconds", "1"),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", float("nan")),
        ("elapsed_seconds", float("inf")),
        ("execution_id", "not-a-digest"),
        ("score", float("nan")),
        ("trace_path", "../private"),
        ("trace_sha256", "bad"),
    ],
)
def test_local_result_validation_stops_before_database(monkeypatch, field, value):
    monkeypatch.setattr(rollout_postgres, "_connect", lambda _: pytest.fail("opened database"))
    with pytest.raises(rollout_ledger.LedgerError):
        rollout_postgres.record_local_result(
            "postgresql://not-contacted",
            cell_id="cell",
            worker_id="worker",
            claim_id="claim",
            record={**_local_record(), field: value},
        )


def test_normalization_allows_missing_catalog_session_but_not_missing_verifier():
    record = _local_record()
    normalized = rollout_ledger.normalize_local_result(
        {**record, "session_id": None}, cell_id="cell"
    )
    assert normalized["session_id"] is None
    with pytest.raises(rollout_ledger.LedgerError):
        rollout_ledger.normalize_local_result(
            {**record, "verifier_execution_id": None}, cell_id="cell"
        )


@pytest.mark.parametrize("operation", ["claim", "heartbeat", "start"])
def test_short_lease_stops_before_database(monkeypatch, operation):
    monkeypatch.setattr(rollout_postgres, "_connect", lambda _: pytest.fail("opened database"))
    kwargs = {"worker_id": "worker", "lease_seconds": 29}
    if operation == "claim":
        kwargs["serving_block"] = "route"
    else:
        kwargs.update(cell_id="cell", claim_id="claim")
        if operation == "start":
            kwargs["session_id"] = "session"
    with pytest.raises(rollout_ledger.LedgerError, match="at least 30"):
        getattr(rollout_postgres, operation)("postgresql://not-contacted", **kwargs)


def test_invalid_scheme_and_duplicate_exclusions_fail_before_connect(monkeypatch):
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: pytest.fail("opened database"))
    with pytest.raises(rollout_ledger.LedgerError, match="scheme"):
        rollout_postgres.summary("sqlite://not-postgres")
    with pytest.raises(rollout_ledger.LedgerError, match="duplicate excluded"):
        rollout_postgres.claim(
            "postgresql://not-contacted",
            worker_id="worker",
            serving_block="route",
            excluded_task_versions=["v1", "v1"],
        )
