from __future__ import annotations

import copy
import json
import uuid
from contextlib import contextmanager

import pytest

from evals.fleet import evaluate, rollout_ledger
from evals.fleet import stored_session_reconciliation_v2 as reconciliation


class _Result:
    def __init__(self, rows=None, *, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return copy.deepcopy(self._rows)


class _Connection:
    def __init__(self, cells, events, receipts, *, local_result_cell_ids=None, fail_receipt=False):
        self.cells = cells
        self.events = events
        self.receipts = receipts
        self.local_result_cell_ids = set(local_result_cell_ids or ())
        self.fail_receipt = fail_receipt

    def execute(self, statement, params=None):
        sql = " ".join(statement.split())
        if sql.startswith("SET LOCAL"):
            return _Result()
        if sql.startswith("SELECT * FROM rollout_cells ORDER BY cell_id"):
            return _Result([self.cells[key] for key in sorted(self.cells)])
        if sql.startswith("SELECT cell_id FROM rollout_local_results"):
            return _Result([{"cell_id": cell_id} for cell_id in sorted(self.local_result_cell_ids)])
        if sql.startswith("SELECT receipt_json FROM ledger_reconciliations"):
            return _Result([{"receipt_json": value} for value in self.receipts])
        if sql.startswith("UPDATE rollout_cells"):
            session_id, receipt_digest, intent_sha256, cell_id = params
            row = self.cells[cell_id]
            if row["state"] != "retry_review":
                return _Result(rowcount=0)
            row.update(
                state="accepted",
                session_id=session_id,
                completed_at="changed",
                heartbeat_at="changed",
                lease_expires_at=None,
                result_class="valid",
                receipt_digest=receipt_digest,
                failure_code=None,
                reconciliation_digest=intent_sha256,
                updated_at="changed",
            )
            return _Result(rowcount=1)
        if sql.startswith("INSERT INTO ledger_reconciliations"):
            if self.fail_receipt:
                raise RuntimeError("injected receipt failure")
            self.receipts.append(params[2])
            return _Result(rowcount=1)
        raise AssertionError(f"unexpected SQL: {sql}")


def _intent(selected, unselected, counts):
    body = {
        "schema_version": reconciliation.SUBSET_INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": reconciliation.runtime_identity(),
        "serving_block": "base",
        "source_output_root": "/mnt/sfs/jobs/source",
        "source_database": "source_database",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "selected_cell_ids": selected,
        "unselected_cell_ids": unselected,
        "expected_arm_state_counts": counts,
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
    }
    return reconciliation.ExactStoredSessionSubsetIntent(
        evaluation_plan_sha256=body["evaluation_plan_sha256"],
        runtime_files_sha256=body["runtime_files_sha256"],
        serving_block=body["serving_block"],
        source_output_root=body["source_output_root"],
        source_database=body["source_database"],
        source_job_uid=body["source_job_uid"],
        source_job_terminal_receipt_sha256=body["source_job_terminal_receipt_sha256"],
        selected_cell_ids=tuple(selected),
        unselected_cell_ids=tuple(unselected),
        expected_arm_state_counts=counts,
        expected_agent_exit_code=0,
        expected_agent_termination="output_limit",
        expected_failure_code="authoritative_scoring_started.runtimeerror",
        sha256=reconciliation._body_digest(body),  # noqa: SLF001
    )


def _local_gap_intent(selected, unselected, counts, missing):
    body = {
        "schema_version": reconciliation.SUBSET_LOCAL_GAPS_INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": reconciliation.runtime_identity(),
        "serving_block": "base",
        "source_output_root": "/mnt/sfs/jobs/source",
        "source_database": "source_database",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "selected_cell_ids": selected,
        "unselected_cell_ids": unselected,
        "expected_arm_state_counts": counts,
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
        "expected_local_result_count": 15,
        "missing_local_result_cell_ids": missing,
        "expected_missing_local_result_failure_code": (
            reconciliation.MISSING_LOCAL_RESULT_FAILURE_CODE
        ),
    }
    return reconciliation.ExactStoredSessionSubsetIntent(
        evaluation_plan_sha256=body["evaluation_plan_sha256"],
        runtime_files_sha256=body["runtime_files_sha256"],
        serving_block=body["serving_block"],
        source_output_root=body["source_output_root"],
        source_database=body["source_database"],
        source_job_uid=body["source_job_uid"],
        source_job_terminal_receipt_sha256=body["source_job_terminal_receipt_sha256"],
        selected_cell_ids=tuple(selected),
        unselected_cell_ids=tuple(unselected),
        expected_arm_state_counts=counts,
        expected_agent_exit_code=0,
        expected_agent_termination="output_limit",
        expected_failure_code="authoritative_scoring_started.runtimeerror",
        sha256=reconciliation._body_digest(body),  # noqa: SLF001
        expected_local_result_count=15,
        missing_local_result_cell_ids=tuple(missing),
        expected_missing_local_result_failure_code=(
            reconciliation.MISSING_LOCAL_RESULT_FAILURE_CODE
        ),
    )


def _row(cell_id, state, *, local_session=None, failure=None):
    return {
        "cell_id": cell_id,
        "serving_block": "base",
        "state": state,
        "worker_id": "worker",
        "claim_id": "claim",
        "session_id": None,
        "completed_at": "original",
        "heartbeat_at": "original",
        "lease_expires_at": None,
        "result_class": "infrastructure_invalid" if state == "retry_review" else "valid",
        "receipt_digest": "old" if state == "accepted" else None,
        "failure_code": failure,
        "reconciliation_digest": None,
        "updated_at": "original",
        "local_session_id": local_session,
        "record_sha256": "c" * 64,
        "session_ingest_status": "completed",
        "agent_exit_code": 0,
        "agent_termination": "output_limit",
    }


def _install(
    monkeypatch,
    cells,
    *,
    local_result_cell_ids=None,
    fail_event=False,
    fail_receipt=False,
):
    events = []
    receipts = []
    connection = _Connection(
        cells,
        events,
        receipts,
        local_result_cell_ids=local_result_cell_ids,
        fail_receipt=fail_receipt,
    )

    @contextmanager
    def transaction(_dsn):
        before = (copy.deepcopy(cells), copy.deepcopy(events), copy.deepcopy(receipts))
        try:
            yield connection
        except BaseException:
            cells.clear()
            cells.update(before[0])
            events[:] = before[1]
            receipts[:] = before[2]
            raise

    def rows(_connection, intent, *, lock):
        assert lock is True
        return [copy.deepcopy(cells[cell_id]) for cell_id in intent.selected_cell_ids]

    def validate_observation(row, observation):
        assert observation["cell_id"] == row["cell_id"]
        return {"receipt_sha256": observation["receipt_sha256"]}

    def event(_connection, **kwargs):
        events.append(kwargs)
        if fail_event:
            raise RuntimeError("injected event failure")

    monkeypatch.setattr(reconciliation.rollout_postgres, "_transaction", transaction)
    monkeypatch.setattr(reconciliation.rollout_postgres, "_event", event)
    monkeypatch.setattr(reconciliation.legacy, "_rows", rows)
    monkeypatch.setattr(reconciliation.legacy, "_validate_record_digest", lambda row: row)
    monkeypatch.setattr(reconciliation.legacy, "_validate_cell_observation", validate_observation)
    return events, receipts


def _observations(selected):
    return [
        {
            "cell_id": cell_id,
            "session_id": f"session-{index}",
            "local_record_sha256": "c" * 64,
            "receipt_sha256": f"{index + 1}" * 64,
        }
        for index, cell_id in enumerate(selected)
    ]


def test_observe_binds_terminal_ledger_plan_identity_not_plan_manifest(monkeypatch, tmp_path):
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4())]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=1)
    intent = _intent(selected, unselected, counts)
    monkeypatch.setattr(
        evaluate,
        "checked_preflight",
        lambda _path: (
            {"sha256": "b" * 64, "tasks": []},
            {"task_bindings": []},
        ),
    )
    monkeypatch.setattr(rollout_ledger, "_plan_rows", lambda _path: [object()])
    monkeypatch.setattr(rollout_ledger, "_plan_digest", lambda _rows: intent.evaluation_plan_sha256)
    monkeypatch.setattr(
        reconciliation.rollout_postgres,
        "verify_plan",
        lambda _dsn, _path: {
            "created": False,
            "cells": 2,
            "plan_sha256": intent.evaluation_plan_sha256,
        },
    )

    class ObservationContinued(RuntimeError):
        pass

    @contextmanager
    def read_transaction(_dsn):
        yield object()

    monkeypatch.setattr(reconciliation.rollout_postgres, "_read_transaction", read_transaction)
    monkeypatch.setattr(
        reconciliation.legacy,
        "_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ObservationContinued()),
    )
    with pytest.raises(ObservationContinued):
        reconciliation.observe(
            "unused",
            intent=intent,
            evaluation_directory=tmp_path,
            client=object(),
        )


def test_observe_rejects_local_ledger_plan_identity_drift_before_database(monkeypatch, tmp_path):
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4())]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=1)
    intent = _intent(selected, unselected, counts)
    monkeypatch.setattr(
        evaluate,
        "checked_preflight",
        lambda _path: (
            {"sha256": "b" * 64, "tasks": []},
            {"task_bindings": []},
        ),
    )

    monkeypatch.setattr(rollout_ledger, "_plan_rows", lambda _path: [object()])
    monkeypatch.setattr(rollout_ledger, "_plan_digest", lambda _rows: "c" * 64)

    def database_must_not_be_accessed(*_args, **_kwargs):
        raise AssertionError("database must not be accessed after local ledger plan drift")

    monkeypatch.setattr(
        reconciliation.rollout_postgres, "verify_plan", database_must_not_be_accessed
    )
    with pytest.raises(rollout_ledger.LedgerError, match="local source ledger plan"):
        reconciliation.observe(
            "unused",
            intent=intent,
            evaluation_directory=tmp_path,
            client=object(),
        )


def test_observe_rejects_database_ledger_plan_identity_drift_before_cell_observation(
    monkeypatch, tmp_path
):
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4())]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=1)
    intent = _intent(selected, unselected, counts)
    monkeypatch.setattr(
        evaluate,
        "checked_preflight",
        lambda _path: (
            {"sha256": "b" * 64, "tasks": []},
            {"task_bindings": []},
        ),
    )
    monkeypatch.setattr(rollout_ledger, "_plan_rows", lambda _path: [object()])
    monkeypatch.setattr(rollout_ledger, "_plan_digest", lambda _rows: intent.evaluation_plan_sha256)
    monkeypatch.setattr(
        reconciliation.rollout_postgres,
        "verify_plan",
        lambda _dsn, _path: {"created": False, "cells": 2, "plan_sha256": "c" * 64},
    )

    def database_rows_must_not_be_read(*_args, **_kwargs):
        raise AssertionError("cell rows must not be read after ledger plan drift")

    monkeypatch.setattr(
        reconciliation.rollout_postgres, "_read_transaction", database_rows_must_not_be_read
    )
    with pytest.raises(rollout_ledger.LedgerError, match="database differs"):
        reconciliation.observe(
            "unused",
            intent=intent,
            evaluation_directory=tmp_path,
            client=object(),
        )


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "empty", "overlap", "bool-count", "negative", "missing", "extra", "active"],
)
def test_subset_intent_rejects_malformed_full_census(fault):
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4()), str(uuid.uuid4())]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=2)
    if fault == "duplicate":
        unselected[1] = unselected[0]
    elif fault == "empty":
        unselected = []
        counts.update(accepted=0, retry_review=1)
    elif fault == "overlap":
        unselected[0] = selected[0]
    elif fault == "bool-count":
        counts["accepted"] = True
    elif fault == "negative":
        counts["accepted"] = -1
    elif fault == "missing":
        counts.pop("terminal")
    elif fault == "extra":
        counts["other"] = 0
    else:
        counts.update(accepted=0, running=1)
    with pytest.raises(rollout_ledger.LedgerError, match="census"):
        _intent(selected, unselected, counts)


def test_subset_acceptance_mutates_only_selected_and_is_idempotent(monkeypatch):
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4()), str(uuid.uuid4())]
    cells = {
        selected[0]: _row(
            selected[0],
            "retry_review",
            local_session="session-0",
            failure="authoritative_scoring_started.runtimeerror",
        ),
        unselected[0]: _row(unselected[0], "retry_review", failure="manual"),
        unselected[1]: _row(unselected[1], "accepted"),
    }
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=2)
    intent = _intent(selected, unselected, counts)
    complement_before = {cell: copy.deepcopy(cells[cell]) for cell in unselected}
    events, receipts = _install(monkeypatch, cells)
    receipt = reconciliation.accept_roster(
        "unused", intent=intent, observations=_observations(selected)
    )
    assert cells[selected[0]]["state"] == "accepted"
    assert {cell: cells[cell] for cell in unselected} == complement_before
    assert len(events) == 1 and len(receipts) == 1
    assert receipt == reconciliation._subset_receipt(intent)  # noqa: SLF001
    assert receipt["post_arm_state_counts"]["accepted"] == 2
    assert receipt["post_arm_state_counts"]["retry_review"] == 1
    serialized = json.dumps(receipt)
    assert all(cell not in serialized for cell in selected + unselected)
    assert (
        reconciliation.accept_roster("unused", intent=intent, observations=_observations(selected))
        == receipt
    )
    assert len(events) == 1 and len(receipts) == 1


@pytest.mark.parametrize("fault", ["missing", "extra", "duplicate"])
def test_subset_acceptance_rejects_nonexact_observation_roster(monkeypatch, fault):
    selected = [str(uuid.uuid4()), str(uuid.uuid4())]
    unselected = [str(uuid.uuid4())]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=2)
    intent = _intent(selected, unselected, counts)
    cells = {
        cell_id: _row(
            cell_id,
            "retry_review",
            local_session=f"session-{index}",
            failure="authoritative_scoring_started.runtimeerror",
        )
        for index, cell_id in enumerate(selected)
    }
    cells[unselected[0]] = _row(unselected[0], "accepted")
    before = copy.deepcopy(cells)
    _install(monkeypatch, cells)
    observations = _observations(selected)
    if fault == "missing":
        observations.pop()
    elif fault == "extra":
        observations.append(
            {
                "cell_id": str(uuid.uuid4()),
                "session_id": "other",
                "local_record_sha256": "c" * 64,
                "receipt_sha256": "f" * 64,
            }
        )
    else:
        observations[1] = copy.deepcopy(observations[0])
    with pytest.raises(rollout_ledger.LedgerError, match="observations"):
        reconciliation.accept_roster("unused", intent=intent, observations=observations)
    assert cells == before


@pytest.mark.parametrize("fault", ["missing", "extra", "route", "state"])
def test_subset_acceptance_rejects_full_database_census_drift(monkeypatch, fault):
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4())]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(retry_review=2)
    intent = _intent(selected, unselected, counts)
    cells = {
        selected[0]: _row(
            selected[0],
            "retry_review",
            local_session="session-0",
            failure="authoritative_scoring_started.runtimeerror",
        ),
        unselected[0]: _row(unselected[0], "retry_review", failure="manual"),
    }
    if fault == "missing":
        cells.pop(unselected[0])
    elif fault == "extra":
        extra = str(uuid.uuid4())
        cells[extra] = _row(extra, "retry_review", failure="manual")
    elif fault == "route":
        cells[unselected[0]]["serving_block"] = "other"
    else:
        cells[unselected[0]]["state"] = "accepted"
    before = copy.deepcopy(cells)
    events, receipts = _install(monkeypatch, cells)
    with pytest.raises(rollout_ledger.LedgerError, match="census|route"):
        reconciliation.accept_roster("unused", intent=intent, observations=_observations(selected))
    assert cells == before
    assert events == [] and receipts == []


@pytest.mark.parametrize("failure", ["event", "receipt"])
def test_subset_acceptance_rolls_back_every_write(monkeypatch, failure):
    selected = [str(uuid.uuid4()), str(uuid.uuid4())]
    unselected = [str(uuid.uuid4())]
    cells = {
        cell_id: _row(
            cell_id,
            "retry_review",
            local_session=f"session-{index}",
            failure="authoritative_scoring_started.runtimeerror",
        )
        for index, cell_id in enumerate(selected)
    }
    cells[unselected[0]] = _row(unselected[0], "accepted")
    before = copy.deepcopy(cells)
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=1, retry_review=2)
    intent = _intent(selected, unselected, counts)
    events, receipts = _install(
        monkeypatch,
        cells,
        fail_event=failure == "event",
        fail_receipt=failure == "receipt",
    )
    with pytest.raises(RuntimeError, match="injected"):
        reconciliation.accept_roster("unused", intent=intent, observations=_observations(selected))
    assert cells == before
    assert events == []
    assert receipts == []


def test_subset_idempotency_rejects_tampered_stored_receipt(monkeypatch):
    selected = [str(uuid.uuid4())]
    unselected = [str(uuid.uuid4())]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=0, retry_review=2)
    intent = _intent(selected, unselected, counts)
    cells = {
        selected[0]: _row(
            selected[0],
            "retry_review",
            local_session="session-0",
            failure="authoritative_scoring_started.runtimeerror",
        ),
        unselected[0]: _row(unselected[0], "retry_review", failure="manual"),
    }
    _events, receipts = _install(monkeypatch, cells)
    reconciliation.accept_roster("unused", intent=intent, observations=_observations(selected))
    value = json.loads(receipts[0])
    value["nonselected_cells_preserved"] = False
    receipts[0] = json.dumps(value)
    with pytest.raises(rollout_ledger.LedgerError, match="ambiguous"):
        reconciliation.accept_roster("unused", intent=intent, observations=_observations(selected))


def _local_gap_case():
    selected = [str(uuid.uuid4()) for _ in range(4)]
    unselected = [str(uuid.uuid4()) for _ in range(13)]
    missing = unselected[:2]
    counts = {state: 0 for state in rollout_ledger.STATES}
    counts.update(accepted=10, retry_review=7)
    cells = {
        cell_id: _row(
            cell_id,
            "retry_review",
            local_session=f"session-{index}",
            failure="authoritative_scoring_started.runtimeerror",
        )
        for index, cell_id in enumerate(selected)
    }
    for index, cell_id in enumerate(unselected):
        if cell_id in missing:
            cells[cell_id] = _row(
                cell_id,
                "retry_review",
                failure=reconciliation.MISSING_LOCAL_RESULT_FAILURE_CODE,
            )
        elif index == 2:
            cells[cell_id] = _row(
                cell_id,
                "retry_review",
                local_session="partial-session",
                failure="manual-review-required",
            )
        else:
            cells[cell_id] = _row(
                cell_id,
                "accepted",
                local_session=f"accepted-session-{index}",
            )
    local_result_cell_ids = set(selected + unselected) - set(missing)
    return (
        selected,
        unselected,
        missing,
        counts,
        cells,
        local_result_cell_ids,
    )


def test_exact_two_unselected_local_result_gaps_are_bound_and_preserved(monkeypatch):
    selected, unselected, missing, counts, cells, local_result_cell_ids = _local_gap_case()
    intent = _local_gap_intent(selected, unselected, counts, missing)
    complement_before = {cell_id: copy.deepcopy(cells[cell_id]) for cell_id in unselected}
    events, receipts = _install(
        monkeypatch,
        cells,
        local_result_cell_ids=local_result_cell_ids,
    )

    receipt = reconciliation.accept_roster(
        "unused", intent=intent, observations=_observations(selected)
    )

    assert all(cells[cell_id]["state"] == "accepted" for cell_id in selected)
    assert {cell_id: cells[cell_id] for cell_id in unselected} == complement_before
    assert len(events) == 4 and len(receipts) == 1
    assert receipt["source_local_result_count"] == 15
    assert receipt["missing_local_result_count"] == 2
    assert receipt["missing_local_results_are_unselected"] is True
    assert receipt["selected_cells_have_local_results"] is True
    assert receipt["missing_local_result_cells_preserved"] is True
    assert receipt["model_generation_performed"] is False
    assert receipt["scoring_call_performed"] is False
    serialized = json.dumps(receipt)
    assert all(cell_id not in serialized for cell_id in selected + unselected)

    assert (
        reconciliation.accept_roster("unused", intent=intent, observations=_observations(selected))
        == receipt
    )
    assert len(events) == 4 and len(receipts) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "one-missing",
        "three-missing",
        "selected-missing",
        "wrong-local-count",
        "wrong-failure-code",
        "wrong-arm-size",
        "wrong-selected-count",
        "wrong-state-counts",
    ],
)
def test_local_result_gap_intent_is_not_a_generic_weakening(fault):
    selected, unselected, missing, counts, _cells, _local = _local_gap_case()
    if fault == "one-missing":
        missing = missing[:1]
    elif fault == "three-missing":
        missing = [*missing, unselected[2]]
    elif fault == "selected-missing":
        missing[0] = selected[0]
    elif fault == "wrong-arm-size":
        unselected.pop()
        counts.update(accepted=9)
    elif fault == "wrong-selected-count":
        unselected.insert(0, selected.pop())
    elif fault == "wrong-state-counts":
        counts.update(accepted=9, retry_review=8)

    body = {
        "schema_version": reconciliation.SUBSET_LOCAL_GAPS_INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": reconciliation.runtime_identity(),
        "serving_block": "base",
        "source_output_root": "/mnt/sfs/jobs/source",
        "source_database": "source_database",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": "b" * 64,
        "selected_cell_ids": selected,
        "unselected_cell_ids": unselected,
        "expected_arm_state_counts": counts,
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
        "expected_local_result_count": 14 if fault == "wrong-local-count" else 15,
        "missing_local_result_cell_ids": missing,
        "expected_missing_local_result_failure_code": (
            "other"
            if fault == "wrong-failure-code"
            else reconciliation.MISSING_LOCAL_RESULT_FAILURE_CODE
        ),
    }
    with pytest.raises(rollout_ledger.LedgerError, match="local-result exception"):
        reconciliation.ExactStoredSessionSubsetIntent(
            evaluation_plan_sha256=body["evaluation_plan_sha256"],
            runtime_files_sha256=body["runtime_files_sha256"],
            serving_block=body["serving_block"],
            source_output_root=body["source_output_root"],
            source_database=body["source_database"],
            source_job_uid=body["source_job_uid"],
            source_job_terminal_receipt_sha256=body["source_job_terminal_receipt_sha256"],
            selected_cell_ids=tuple(selected),
            unselected_cell_ids=tuple(unselected),
            expected_arm_state_counts=counts,
            expected_agent_exit_code=0,
            expected_agent_termination="output_limit",
            expected_failure_code="authoritative_scoring_started.runtimeerror",
            sha256=reconciliation._body_digest(body),  # noqa: SLF001
            expected_local_result_count=body["expected_local_result_count"],
            missing_local_result_cell_ids=tuple(missing),
            expected_missing_local_result_failure_code=body[
                "expected_missing_local_result_failure_code"
            ],
        )


@pytest.mark.parametrize(
    "fault",
    ["missing-gained-result", "other-lost-result", "selected-lost-result", "failure-code"],
)
def test_local_result_gap_runtime_rejects_any_roster_or_failure_drift(monkeypatch, fault):
    selected, unselected, missing, counts, cells, local_result_cell_ids = _local_gap_case()
    intent = _local_gap_intent(selected, unselected, counts, missing)
    if fault == "missing-gained-result":
        local_result_cell_ids.add(missing[0])
    elif fault == "other-lost-result":
        local_result_cell_ids.remove(unselected[2])
    elif fault == "selected-lost-result":
        local_result_cell_ids.remove(selected[0])
    else:
        cells[missing[0]]["failure_code"] = "other"
    before = copy.deepcopy(cells)
    events, receipts = _install(
        monkeypatch,
        cells,
        local_result_cell_ids=local_result_cell_ids,
    )
    with pytest.raises(rollout_ledger.LedgerError, match="local-result roster"):
        reconciliation.accept_roster("unused", intent=intent, observations=_observations(selected))
    assert cells == before
    assert events == [] and receipts == []
