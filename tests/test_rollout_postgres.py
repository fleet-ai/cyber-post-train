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

from evals.fleet import (
    reviewed_recovery,
    reviewed_recovery_v2,
    rollout_ledger,
    rollout_postgres,
    rollout_postgres_migrate,
    rollout_postgres_status,
    stored_session_reconciliation,
    stored_session_reconciliation_v2,
)


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
    for operation in (
        rollout_postgres.summary,
        rollout_postgres.active_claims,
        rollout_postgres_status.retry_review_summary,
    ):
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


def test_retry_review_can_be_approved_once_but_never_silently_repeated(pg_dsn, owned_cell):
    rollout_postgres.request_retry_review(pg_dsn, **owned_cell, failure_code="output_limit")
    approved = rollout_postgres.approve_retry(
        pg_dsn, cell_id=owned_cell["cell_id"], reconciliation_digest="a" * 64
    )
    assert approved["state"] == "pending" and approved["retry_count"] == 1
    retried = rollout_postgres.claim(pg_dsn, worker_id="retry", serving_block="route")
    owner = {key: retried[key] for key in ("cell_id", "worker_id", "claim_id")}
    rollout_postgres.request_retry_review(pg_dsn, **owner, failure_code="output_limit")
    with pytest.raises(rollout_ledger.LedgerError, match="exhausted"):
        rollout_postgres.approve_retry(
            pg_dsn, cell_id=owned_cell["cell_id"], reconciliation_digest="b" * 64
        )


def test_reviewed_recovery_claim_checks_full_roster_digest_atomically(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=2))
    owners = [
        rollout_postgres.claim(pg_dsn, worker_id=f"review-{index}", serving_block="route")
        for index in range(2)
    ]
    cell_ids = [owner["cell_id"] for owner in owners]
    for owner in owners:
        rollout_postgres.request_retry_review(
            pg_dsn,
            cell_id=owner["cell_id"],
            worker_id=owner["worker_id"],
            claim_id=owner["claim_id"],
            failure_code="post_claim.connecterror",
        )
    intent_body = {
        "schema_version": reviewed_recovery.INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": reviewed_recovery.runtime_identity(),
        "serving_block": "route",
        "selected_cell_ids": cell_ids,
    }
    intent = reviewed_recovery.ReviewedRecoveryIntent(
        evaluation_plan_sha256=intent_body["evaluation_plan_sha256"],
        runtime_files_sha256=intent_body["runtime_files_sha256"],
        serving_block="route",
        selected_cell_ids=tuple(cell_ids),
        sha256=reviewed_recovery._body_digest(intent_body),  # noqa: SLF001
    )
    apply_receipt = reviewed_recovery.apply_intent(pg_dsn, intent=intent)
    assert apply_receipt["reviewed_intent_sha256"] == intent.sha256
    summary = rollout_postgres.summary(pg_dsn)["by_state"]
    assert summary["retry_review"] == 2 and summary["pending"] == 0
    assert rollout_postgres.claim(pg_dsn, worker_id="ordinary", serving_block="route") is None

    subset_body = {**intent_body, "selected_cell_ids": cell_ids[:1]}
    subset = reviewed_recovery.ReviewedRecoveryIntent(
        evaluation_plan_sha256=subset_body["evaluation_plan_sha256"],
        runtime_files_sha256=subset_body["runtime_files_sha256"],
        serving_block="route",
        selected_cell_ids=tuple(cell_ids[:1]),
        sha256=reviewed_recovery._body_digest(subset_body),  # noqa: SLF001
    )
    with pytest.raises(rollout_ledger.LedgerError, match="reconciliation digest differs"):
        reviewed_recovery.claim(pg_dsn, intent=subset, worker_id="subset", serving_block="route")

    with psycopg.connect(pg_dsn) as connection:
        connection.execute(
            "UPDATE rollout_cells SET reconciliation_digest = %s WHERE cell_id = %s",
            ("c" * 64, cell_ids[0]),
        )
    with pytest.raises(rollout_ledger.LedgerError, match="reconciliation digest differs"):
        reviewed_recovery.claim(pg_dsn, intent=intent, worker_id="guarded", serving_block="route")
    assert rollout_postgres.summary(pg_dsn)["by_state"]["retry_review"] == 2

    with psycopg.connect(pg_dsn) as connection:
        connection.execute(
            "UPDATE rollout_cells SET reconciliation_digest = %s WHERE cell_id = %s",
            (intent.sha256, cell_ids[0]),
        )
    claimed = reviewed_recovery.claim(
        pg_dsn, intent=intent, worker_id="guarded", serving_block="route"
    )
    assert claimed["cell_id"] in cell_ids and claimed["retry_count"] == 1
    with psycopg.connect(pg_dsn) as connection:
        receipts = connection.execute(
            "SELECT kind, receipt_json FROM ledger_reconciliations"
        ).fetchall()
    assert {row[0] for row in receipts} == {
        reviewed_recovery.APPLY_SCHEMA,
        reviewed_recovery.PRECLAIM_SCHEMA,
    }
    for _, raw in receipts:
        receipt = json.loads(raw)
        assert receipt["selected_cell_count"] == 2
        serialized = json.dumps(receipt).lower()
        assert not any(word in serialized for word in ("cell_id", "task_version", "session_id"))


def test_provisioning_timeout_recovery_requires_prior_five_and_claims_only_exact_two(
    tmp_path, pg_dsn
):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=7))
    with psycopg.connect(pg_dsn) as connection:
        rows = connection.execute("SELECT cell_id FROM rollout_cells ORDER BY cell_id").fetchall()
        prior_digest = "c" * 64
        connection.execute(
            """
            UPDATE rollout_cells
            SET state = 'accepted', result_class = 'valid', reconciliation_digest = %s
            WHERE cell_id = ANY(%s::text[])
            """,
            (prior_digest, [row[0] for row in rows[:5]]),
        )
        connection.execute(
            """
            UPDATE rollout_cells
            SET state = 'retry_review', result_class = 'infrastructure_invalid',
                failure_code = %s
            WHERE cell_id = ANY(%s::text[])
            """,
            (reviewed_recovery_v2.SOURCE_FAILURE_CODE, [row[0] for row in rows[5:]]),
        )
        connection.execute(
            """
            INSERT INTO ledger_reconciliations (
                receipt_sha256, kind, receipt_json, created_at
            ) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (
                "d" * 64,
                "fleet-stored-session-reconciliation-v2",
                json.dumps({"reviewed_intent_sha256": prior_digest}),
            ),
        )
    selected = tuple(
        reviewed_recovery_v2.SelectedCell(
            cell_id=row[0],
            claim_file_sha256="1" * 64,
            binding_file_sha256="2" * 64,
            prompt_file_sha256="3" * 64,
            failure_file_sha256="4" * 64,
            cleanup_file_sha256="5" * 64,
        )
        for row in rows[5:]
    )
    body = {
        "schema_version": reviewed_recovery_v2.INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": reviewed_recovery_v2.runtime_identity(),
        "serving_block": "route",
        "source_output_root": "/mnt/sfs/jobs/source-eval",
        "source_database": "source_eval",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "prior_stored_session_intent_sha256": prior_digest,
        "selected_cells": [cell.as_dict() for cell in selected],
    }
    intent = reviewed_recovery_v2.ProvisioningTimeoutIntent(
        evaluation_plan_sha256=body["evaluation_plan_sha256"],
        runtime_files_sha256=body["runtime_files_sha256"],
        serving_block=body["serving_block"],
        source_output_root=body["source_output_root"],
        source_database=body["source_database"],
        source_job_uid=body["source_job_uid"],
        source_job_terminal_receipt_sha256=body["source_job_terminal_receipt_sha256"],
        prior_stored_session_intent_sha256=prior_digest,
        selected_cells=selected,
        sha256=reviewed_recovery_v2._body_digest(body),  # noqa: SLF001
    )
    observations = []
    for cell in selected:
        observation = {
            "schema_version": reviewed_recovery_v2.OBSERVATION_SCHEMA,
            **cell.as_dict(),
            "source_file_set_exact": True,
            "local_result_absent": True,
            "authoritative_exact_execution_session_absent": True,
            "model_execution_artifacts_absent": True,
            "scoring_artifacts_absent": True,
            "provisioning_method": "POST",
            "provisioning_http_status": 504,
        }
        observation["receipt_sha256"] = reviewed_recovery_v2.crypto.digest_without(
            observation, "receipt_sha256"
        )
        observations.append(observation)
    receipt = reviewed_recovery_v2.apply_intent(pg_dsn, intent=intent, observations=observations)
    assert receipt["selected_cell_count"] == 2
    assert rollout_postgres.claim(pg_dsn, worker_id="ordinary", serving_block="route") is None
    claimed = reviewed_recovery_v2.claim(
        pg_dsn, intent=intent, worker_id="repair", serving_block="route"
    )
    assert claimed["cell_id"] in intent.selected_cell_ids
    assert claimed["retry_count"] == 1
    assert claimed["reconciliation_digest"] == intent.sha256


def test_reviewed_outcome_can_be_accepted_without_replaying_it(pg_dsn, owned_cell):
    rollout_postgres.start(pg_dsn, **owned_cell, session_id="authoritative-session")
    rollout_postgres.request_retry_review(pg_dsn, **owned_cell, failure_code="catalog_lag")
    accepted = rollout_postgres.accept_reviewed(
        pg_dsn,
        cell_id=owned_cell["cell_id"],
        session_id="authoritative-session",
        receipt_digest="c" * 64,
        reconciliation_digest="d" * 64,
    )
    assert accepted["state"] == "accepted" and accepted["result_class"] == "valid"
    assert accepted["failure_code"] is None and accepted["lease_expires_at"] is None
    assert _events(pg_dsn)[-1][0] == "reviewed_outcome_accepted"


def test_reviewed_outcome_can_be_terminally_closed(pg_dsn, owned_cell):
    rollout_postgres.request_retry_review(pg_dsn, **owned_cell, failure_code="invalid")
    terminal = rollout_postgres.mark_terminal(
        pg_dsn, cell_id=owned_cell["cell_id"], reconciliation_digest="e" * 64
    )
    assert terminal["state"] == "terminal" and terminal["failure_code"] == "invalid"
    assert _events(pg_dsn)[-1][0] == "terminally_closed"


def _stored_session_intent(cell_ids):
    body = {
        "schema_version": stored_session_reconciliation.INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": stored_session_reconciliation.runtime_identity(),
        "serving_block": "route",
        "source_output_root": "/mnt/sfs/jobs/source-eval",
        "source_database": "stored_session_test",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "selected_cell_ids": list(cell_ids),
    }
    return stored_session_reconciliation.StoredSessionIntent(
        evaluation_plan_sha256=body["evaluation_plan_sha256"],
        runtime_files_sha256=body["runtime_files_sha256"],
        serving_block=body["serving_block"],
        source_output_root=body["source_output_root"],
        source_database=body["source_database"],
        source_job_uid=body["source_job_uid"],
        source_job_terminal_receipt_sha256=body["source_job_terminal_receipt_sha256"],
        selected_cell_ids=tuple(body["selected_cell_ids"]),
        sha256=stored_session_reconciliation._body_digest(body),  # noqa: SLF001
    )


def _stored_session_observation(
    cell_id, session_id, record_sha256, *, task_version_id, model, verifier_execution_id
):
    body = {
        "schema_version": "fleet-stored-session-cell-observation-v1",
        "identity_binding_sha256": stored_session_reconciliation._body_digest(  # noqa: SLF001
            {
                "cell_id": cell_id,
                "session_id": session_id,
                "task_version_id": task_version_id,
                "model": model,
                "verifier_execution_id": verifier_execution_id,
            }
        ),
        "config_sha256": "2" * 64,
        "local_record_sha256": record_sha256,
        "artifact_binding_sha256": "d" * 64,
        "authoritative_session_metadata_sha256": "e" * 64,
        "authoritative_score_finite_and_equal_to_private_local_result": True,
        "authoritative_pinned_task_version_metadata_only": True,
        "reference_trace_content_returned": False,
        "model_generation_performed": False,
        "scoring_call_performed": False,
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return {
        "cell_id": cell_id,
        "session_id": session_id,
        "local_record_sha256": record_sha256,
        "receipt": {
            **body,
            "receipt_sha256": reviewed_recovery.crypto.digest_without(body, "receipt_sha256"),
        },
    }


def test_stored_sessions_are_accepted_atomically_without_retry_or_scoring(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=2))
    owners = [
        rollout_postgres.claim(pg_dsn, worker_id=f"stored-{index}", serving_block="route")
        for index in range(2)
    ]
    observations = []
    for index, owner in enumerate(owners):
        record = {
            **_local_record(),
            "execution_id": "sha256:" + str(index + 1) * 64,
            "run_id": f"stored-run-{index}",
            "session_id": f"stored-session-{index}",
            "verifier_execution_id": f"stored-verifier-{index}",
            "agent_exit_code": 1,
            "agent_termination": "process_error",
        }
        created = rollout_postgres.record_local_result(
            pg_dsn,
            cell_id=owner["cell_id"],
            worker_id=owner["worker_id"],
            claim_id=owner["claim_id"],
            record=record,
        )
        observations.append(
            _stored_session_observation(
                owner["cell_id"],
                record["session_id"],
                created["record_sha256"],
                task_version_id=f"version-{index}",
                model="endpoint",
                verifier_execution_id=record["verifier_execution_id"],
            )
        )
    rollout_postgres.request_retry_review(
        pg_dsn,
        cell_id=owners[0]["cell_id"],
        worker_id=owners[0]["worker_id"],
        claim_id=owners[0]["claim_id"],
        failure_code="authoritative_scoring_started.runtimeerror",
    )
    with psycopg.connect(pg_dsn) as connection:
        connection.execute(
            "UPDATE rollout_cells SET lease_expires_at = CURRENT_TIMESTAMP - INTERVAL '1 minute' "
            "WHERE cell_id = %s",
            (owners[1]["cell_id"],),
        )
    intent = _stored_session_intent([owner["cell_id"] for owner in owners])
    swapped = [
        {**observations[0], "receipt": observations[1]["receipt"]},
        {**observations[1], "receipt": observations[0]["receipt"]},
    ]
    with pytest.raises(rollout_ledger.LedgerError, match="receipt is invalid"):
        stored_session_reconciliation.accept_roster(pg_dsn, intent=intent, observations=swapped)
    assert rollout_postgres.summary(pg_dsn)["by_state"].get("accepted", 0) == 0
    receipt = stored_session_reconciliation.accept_roster(
        pg_dsn, intent=intent, observations=observations
    )
    assert receipt["prior_retry_review_count"] == 1
    assert receipt["prior_stale_active_count"] == 1
    assert receipt["accepted_existing_completed_session_count"] == 2
    assert receipt["model_generation_performed"] is False
    assert receipt["scoring_call_performed"] is False
    summary = rollout_postgres.summary(pg_dsn)
    assert summary["by_state"]["accepted"] == 2
    assert summary["stale_active"] == 0
    with psycopg.connect(pg_dsn) as connection:
        persisted = dict(
            connection.execute(
                "SELECT cell_id, receipt_digest FROM rollout_cells ORDER BY cell_id"
            ).fetchall()
        )
    expected_digests = {
        observation["cell_id"]: observation["receipt"]["receipt_sha256"]
        for observation in observations
    }
    assert persisted == expected_digests
    assert len(set(persisted.values())) == 2
    serialized = json.dumps(receipt)
    assert all(owner["cell_id"] not in serialized for owner in owners)
    assert all(f"stored-session-{index}" not in serialized for index in range(2))
    assert (
        stored_session_reconciliation.accept_roster(
            pg_dsn, intent=intent, observations=observations
        )
        == receipt
    )


def test_stored_session_roster_rejects_a_live_owner_atomically(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=1))
    owner = rollout_postgres.claim(pg_dsn, worker_id="live", serving_block="route")
    record = {
        **_local_record(),
        "session_id": "stored-session",
        "agent_exit_code": 1,
        "agent_termination": "process_error",
    }
    created = rollout_postgres.record_local_result(
        pg_dsn,
        cell_id=owner["cell_id"],
        worker_id=owner["worker_id"],
        claim_id=owner["claim_id"],
        record=record,
    )
    intent = _stored_session_intent([owner["cell_id"]])
    observation = _stored_session_observation(
        owner["cell_id"],
        record["session_id"],
        created["record_sha256"],
        task_version_id="version-0",
        model="endpoint",
        verifier_execution_id=record["verifier_execution_id"],
    )
    with pytest.raises(rollout_ledger.LedgerError, match="not review-held or stale"):
        stored_session_reconciliation.accept_roster(
            pg_dsn, intent=intent, observations=[observation]
        )
    assert rollout_postgres.summary(pg_dsn)["by_state"]["claimed"] == 1


def _stored_session_subset_intent(selected, unselected, counts):
    body = {
        "schema_version": stored_session_reconciliation_v2.SUBSET_INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": stored_session_reconciliation_v2.runtime_identity(),
        "serving_block": "route",
        "source_output_root": "/mnt/sfs/jobs/source-eval",
        "source_database": "stored_session_test",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "selected_cell_ids": list(selected),
        "unselected_cell_ids": list(unselected),
        "expected_arm_state_counts": counts,
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
    }
    return stored_session_reconciliation_v2.ExactStoredSessionSubsetIntent(
        evaluation_plan_sha256=body["evaluation_plan_sha256"],
        runtime_files_sha256=body["runtime_files_sha256"],
        serving_block=body["serving_block"],
        source_output_root=body["source_output_root"],
        source_database=body["source_database"],
        source_job_uid=body["source_job_uid"],
        source_job_terminal_receipt_sha256=body["source_job_terminal_receipt_sha256"],
        selected_cell_ids=tuple(body["selected_cell_ids"]),
        unselected_cell_ids=tuple(body["unselected_cell_ids"]),
        expected_arm_state_counts=body["expected_arm_state_counts"],
        expected_agent_exit_code=body["expected_agent_exit_code"],
        expected_agent_termination=body["expected_agent_termination"],
        expected_failure_code=body["expected_failure_code"],
        sha256=stored_session_reconciliation_v2._body_digest(body),  # noqa: SLF001
    )


def _stored_session_local_gap_intent(selected, unselected, counts, missing):
    body = {
        "schema_version": (stored_session_reconciliation_v2.SUBSET_LOCAL_GAPS_INTENT_SCHEMA),
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": stored_session_reconciliation_v2.runtime_identity(),
        "serving_block": "route",
        "source_output_root": "/mnt/sfs/jobs/source-eval",
        "source_database": "stored_session_test",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "selected_cell_ids": list(selected),
        "unselected_cell_ids": list(unselected),
        "expected_arm_state_counts": counts,
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
        "expected_local_result_count": 15,
        "missing_local_result_cell_ids": list(missing),
        "expected_missing_local_result_failure_code": (
            stored_session_reconciliation_v2.MISSING_LOCAL_RESULT_FAILURE_CODE
        ),
    }
    return stored_session_reconciliation_v2.ExactStoredSessionSubsetIntent(
        evaluation_plan_sha256=body["evaluation_plan_sha256"],
        runtime_files_sha256=body["runtime_files_sha256"],
        serving_block=body["serving_block"],
        source_output_root=body["source_output_root"],
        source_database=body["source_database"],
        source_job_uid=body["source_job_uid"],
        source_job_terminal_receipt_sha256=body["source_job_terminal_receipt_sha256"],
        selected_cell_ids=tuple(body["selected_cell_ids"]),
        unselected_cell_ids=tuple(body["unselected_cell_ids"]),
        expected_arm_state_counts=body["expected_arm_state_counts"],
        expected_agent_exit_code=body["expected_agent_exit_code"],
        expected_agent_termination=body["expected_agent_termination"],
        expected_failure_code=body["expected_failure_code"],
        sha256=stored_session_reconciliation_v2._body_digest(body),  # noqa: SLF001
        expected_local_result_count=body["expected_local_result_count"],
        missing_local_result_cell_ids=tuple(body["missing_local_result_cell_ids"]),
        expected_missing_local_result_failure_code=body[
            "expected_missing_local_result_failure_code"
        ],
    )


def test_subset_stored_session_acceptance_preserves_full_complement(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=3))
    owners = [
        rollout_postgres.claim(pg_dsn, worker_id=f"subset-{index}", serving_block="route")
        for index in range(3)
    ]
    selected_owner = owners[0]
    record = {
        **_local_record(),
        "session_id": "subset-selected-session",
        "verifier_execution_id": "subset-selected-verifier",
        "agent_exit_code": 0,
        "agent_termination": "output_limit",
    }
    created = rollout_postgres.record_local_result(
        pg_dsn,
        cell_id=selected_owner["cell_id"],
        worker_id=selected_owner["worker_id"],
        claim_id=selected_owner["claim_id"],
        record=record,
    )
    rollout_postgres.request_retry_review(
        pg_dsn,
        cell_id=selected_owner["cell_id"],
        worker_id=selected_owner["worker_id"],
        claim_id=selected_owner["claim_id"],
        failure_code="authoritative_scoring_started.runtimeerror",
    )
    unselected_retry = owners[1]
    rollout_postgres.request_retry_review(
        pg_dsn,
        cell_id=unselected_retry["cell_id"],
        worker_id=unselected_retry["worker_id"],
        claim_id=unselected_retry["claim_id"],
        failure_code="post_claim.connecterror",
    )
    accepted = owners[2]
    rollout_postgres.start(
        pg_dsn,
        cell_id=accepted["cell_id"],
        worker_id=accepted["worker_id"],
        claim_id=accepted["claim_id"],
        session_id="already-accepted-session",
    )
    rollout_postgres.accept(
        pg_dsn,
        cell_id=accepted["cell_id"],
        worker_id=accepted["worker_id"],
        claim_id=accepted["claim_id"],
        receipt_digest="f" * 64,
    )
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=2)
    unselected_ids = [unselected_retry["cell_id"], accepted["cell_id"]]
    intent = _stored_session_subset_intent([selected_owner["cell_id"]], unselected_ids, counts)
    observation = _stored_session_observation(
        selected_owner["cell_id"],
        record["session_id"],
        created["record_sha256"],
        task_version_id="version-0",
        model="endpoint",
        verifier_execution_id=record["verifier_execution_id"],
    )
    with psycopg.connect(pg_dsn, row_factory=psycopg.rows.dict_row) as connection:
        complement_before = connection.execute(
            "SELECT * FROM rollout_cells WHERE cell_id = ANY(%s::text[]) ORDER BY cell_id",
            (unselected_ids,),
        ).fetchall()
        event_count_before = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_events"
        ).fetchone()["count"]
    receipt = stored_session_reconciliation_v2.accept_roster(
        pg_dsn, intent=intent, observations=[observation]
    )
    assert receipt["prior_arm_state_counts"] == counts
    assert receipt["post_arm_state_counts"]["accepted"] == 2
    assert receipt["post_arm_state_counts"]["retry_review"] == 1
    assert receipt["nonselected_cells_preserved"] is True
    with psycopg.connect(pg_dsn, row_factory=psycopg.rows.dict_row) as connection:
        complement_after = connection.execute(
            "SELECT * FROM rollout_cells WHERE cell_id = ANY(%s::text[]) ORDER BY cell_id",
            (unselected_ids,),
        ).fetchall()
        event_count_after = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_events"
        ).fetchone()["count"]
        reconciliation_count = connection.execute(
            "SELECT COUNT(*) AS count FROM ledger_reconciliations"
        ).fetchone()["count"]
    assert complement_after == complement_before
    assert event_count_after == event_count_before + 1
    assert reconciliation_count == 1
    assert rollout_postgres.summary(pg_dsn)["by_state"] == {
        **{state: 0 for state in rollout_ledger.STATES},
        "accepted": 2,
        "retry_review": 1,
    }
    second_receipt = stored_session_reconciliation_v2.accept_roster(
        pg_dsn, intent=intent, observations=[observation]
    )
    assert second_receipt == receipt
    with psycopg.connect(pg_dsn) as connection:
        assert connection.execute("SELECT COUNT(*) FROM rollout_events").fetchone()[0] == (
            event_count_after
        )
        assert connection.execute("SELECT COUNT(*) FROM ledger_reconciliations").fetchone()[0] == 1


def test_exact_two_unselected_local_result_gaps_are_locked_and_preserved(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=17))
    owners = [
        rollout_postgres.claim(pg_dsn, worker_id=f"gap-{index}", serving_block="route")
        for index in range(17)
    ]
    selected = owners[:4]
    missing = owners[4:6]
    partial = owners[6]
    accepted = owners[7:]
    observations = []

    for index, owner in enumerate([*selected, partial, *accepted]):
        record = {
            **_local_record(),
            "execution_id": "sha256:" + f"{index + 1:064x}",
            "run_id": f"gap-run-{index}",
            "session_id": f"gap-session-{index}",
            "verifier_execution_id": f"gap-verifier-{index}",
            "agent_exit_code": 0,
            "agent_termination": "output_limit",
        }
        created = rollout_postgres.record_local_result(
            pg_dsn,
            cell_id=owner["cell_id"],
            worker_id=owner["worker_id"],
            claim_id=owner["claim_id"],
            record=record,
        )
        if owner in selected:
            observations.append(
                _stored_session_observation(
                    owner["cell_id"],
                    record["session_id"],
                    created["record_sha256"],
                    task_version_id=owner["task_version_id"],
                    model="endpoint",
                    verifier_execution_id=record["verifier_execution_id"],
                )
            )

    for owner in selected:
        rollout_postgres.request_retry_review(
            pg_dsn,
            cell_id=owner["cell_id"],
            worker_id=owner["worker_id"],
            claim_id=owner["claim_id"],
            failure_code="authoritative_scoring_started.runtimeerror",
        )
    for owner in missing:
        rollout_postgres.request_retry_review(
            pg_dsn,
            cell_id=owner["cell_id"],
            worker_id=owner["worker_id"],
            claim_id=owner["claim_id"],
            failure_code=(stored_session_reconciliation_v2.MISSING_LOCAL_RESULT_FAILURE_CODE),
        )
    rollout_postgres.request_retry_review(
        pg_dsn,
        cell_id=partial["cell_id"],
        worker_id=partial["worker_id"],
        claim_id=partial["claim_id"],
        failure_code="manual-review-required",
    )
    for owner in accepted:
        rollout_postgres.start(
            pg_dsn,
            cell_id=owner["cell_id"],
            worker_id=owner["worker_id"],
            claim_id=owner["claim_id"],
            session_id=f"accepted-{owner['cell_id']}",
        )
        rollout_postgres.accept(
            pg_dsn,
            cell_id=owner["cell_id"],
            worker_id=owner["worker_id"],
            claim_id=owner["claim_id"],
            receipt_digest="f" * 64,
        )

    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=10, retry_review=7)
    unselected_ids = [owner["cell_id"] for owner in owners[4:]]
    intent = _stored_session_local_gap_intent(
        [owner["cell_id"] for owner in selected],
        unselected_ids,
        counts,
        [owner["cell_id"] for owner in missing],
    )
    with psycopg.connect(pg_dsn, row_factory=psycopg.rows.dict_row) as connection:
        complement_before = connection.execute(
            "SELECT * FROM rollout_cells WHERE cell_id = ANY(%s::text[]) ORDER BY cell_id",
            (unselected_ids,),
        ).fetchall()
        local_before = connection.execute(
            "SELECT cell_id FROM rollout_local_results ORDER BY cell_id"
        ).fetchall()

    receipt = stored_session_reconciliation_v2.accept_roster(
        pg_dsn, intent=intent, observations=observations
    )

    assert receipt["source_local_result_count"] == 15
    assert receipt["missing_local_result_count"] == 2
    assert receipt["selected_cells_have_local_results"] is True
    assert receipt["post_arm_state_counts"]["accepted"] == 14
    assert receipt["post_arm_state_counts"]["retry_review"] == 3
    with psycopg.connect(pg_dsn, row_factory=psycopg.rows.dict_row) as connection:
        complement_after = connection.execute(
            "SELECT * FROM rollout_cells WHERE cell_id = ANY(%s::text[]) ORDER BY cell_id",
            (unselected_ids,),
        ).fetchall()
        local_after = connection.execute(
            "SELECT cell_id FROM rollout_local_results ORDER BY cell_id"
        ).fetchall()
    assert complement_after == complement_before
    assert local_after == local_before
    assert rollout_postgres.summary(pg_dsn)["local_results"] == 15


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


def test_retry_review_summary_is_grouped_score_blind_and_read_only(tmp_path, pg_dsn):
    rollout_postgres.initialize(pg_dsn, _plan(tmp_path / "plan.csv", count=3))

    first = rollout_postgres.claim(pg_dsn, worker_id="worker-1", serving_block="route")
    first_owner = {key: first[key] for key in ("cell_id", "worker_id", "claim_id")}
    rollout_postgres.record_local_result(pg_dsn, **first_owner, record=_local_record())
    rollout_postgres.request_retry_review(
        pg_dsn,
        **first_owner,
        failure_code="authoritative_scoring_started.runtimeerror",
    )

    second = rollout_postgres.claim(pg_dsn, worker_id="worker-2", serving_block="route")
    second_owner = {key: second[key] for key in ("cell_id", "worker_id", "claim_id")}
    rollout_postgres.request_retry_review(
        pg_dsn,
        **second_owner,
        failure_code="post_claim.connecterror",
    )

    third = rollout_postgres.claim(pg_dsn, worker_id="worker-3", serving_block="route")
    third_owner = {key: third[key] for key in ("cell_id", "worker_id", "claim_id")}
    rollout_postgres.record_local_result(
        pg_dsn,
        **third_owner,
        record={
            **_local_record(),
            "execution_id": "sha256:" + "4" * 64,
            "session_id": "private-session-value",
            "agent_termination": "private-termination-value",
            "session_ingest_status": "private-ingest-value",
        },
    )
    rollout_postgres.request_retry_review(
        pg_dsn,
        **third_owner,
        failure_code="authoritative_scoring_started.runtimeerror",
    )

    before = _events(pg_dsn)
    observed = rollout_postgres_status.retry_review_summary(pg_dsn)
    assert observed == {
        "retry_review": 3,
        "with_local_result": 2,
        "groups": [
            {
                "failure_code": "authoritative_scoring_started.runtimeerror",
                "result_class": "infrastructure_invalid",
                "agent_termination": "completed",
                "agent_exit_code": 0,
                "session_ingest_status": "completed",
                "has_local_result": True,
                "has_session": True,
                "count": 1,
            },
            {
                "failure_code": "authoritative_scoring_started.runtimeerror",
                "result_class": "infrastructure_invalid",
                "agent_termination": "other",
                "agent_exit_code": 0,
                "session_ingest_status": "other",
                "has_local_result": True,
                "has_session": True,
                "count": 1,
            },
            {
                "failure_code": "post_claim.connecterror",
                "result_class": "infrastructure_invalid",
                "agent_termination": "missing",
                "agent_exit_code": None,
                "session_ingest_status": "missing",
                "has_local_result": False,
                "has_session": False,
                "count": 1,
            },
        ],
    }
    serialized = json.dumps(observed, sort_keys=True)
    for private_value in (
        first["cell_id"],
        second["cell_id"],
        third["cell_id"],
        "synthetic-session",
        "synthetic-verifier",
        "private-session-value",
        "private-termination-value",
        "private-ingest-value",
        '"score"',
        "trace_path",
    ):
        assert private_value not in serialized
    assert _events(pg_dsn) == before


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
