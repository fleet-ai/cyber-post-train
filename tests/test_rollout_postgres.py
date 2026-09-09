"""Integration tests only against an explicitly opted-in, local disposable database."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
