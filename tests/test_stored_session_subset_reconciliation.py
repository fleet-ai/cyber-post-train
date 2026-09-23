from __future__ import annotations

import copy
import json
import uuid
from contextlib import contextmanager

import pytest

from evals.fleet import rollout_ledger
from evals.fleet import stored_session_reconciliation_v2 as reconciliation


class _Result:
    def __init__(self, rows=None, *, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return copy.deepcopy(self._rows)


class _Connection:
    def __init__(self, cells, events, receipts, *, fail_receipt=False):
        self.cells = cells
        self.events = events
        self.receipts = receipts
        self.fail_receipt = fail_receipt

    def execute(self, statement, params=None):
        sql = " ".join(statement.split())
        if sql.startswith("SET LOCAL"):
            return _Result()
        if sql.startswith("SELECT * FROM rollout_cells ORDER BY cell_id"):
            return _Result([self.cells[key] for key in sorted(self.cells)])
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


def _install(monkeypatch, cells, *, fail_event=False, fail_receipt=False):
    events = []
    receipts = []
    connection = _Connection(cells, events, receipts, fail_receipt=fail_receipt)

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
