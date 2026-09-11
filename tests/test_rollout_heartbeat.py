"""Synthetic lease lifecycle tests; no cluster, task content, or credentials."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evals.fleet import rollout_ledger, rollout_postgres
from evals.fleet import rollout_worker as worker

CELL = {
    "cell_id": "cell",
    "worker_id": "worker",
    "claim_id": "claim",
    "model_id": "model",
    "task_version_id": "task",
    "attempt": 1,
}


def test_heartbeat_renews_before_background_thread_starts(monkeypatch):
    ledger = SimpleNamespace(heartbeat=Mock())
    pulse = worker._Heartbeat("unused", CELL, ledger)
    monkeypatch.setattr(pulse.thread, "start", Mock())
    pulse.__enter__()
    ledger.heartbeat.assert_called_once_with(
        "unused", cell_id="cell", worker_id="worker", claim_id="claim", lease_seconds=300
    )


def test_transient_heartbeat_error_retries_without_disclosing_error(monkeypatch):
    ledger = SimpleNamespace(heartbeat=Mock(side_effect=[OSError("private"), None]))
    pulse = worker._Heartbeat("unused", CELL, ledger)
    monkeypatch.setattr(pulse.stop, "wait", Mock(side_effect=[False, False, True]))
    pulse._run()
    assert ledger.heartbeat.call_count == 2
    pulse.check()


def test_owner_fencing_error_is_not_retried(monkeypatch):
    ledger = SimpleNamespace(heartbeat=Mock(side_effect=rollout_ledger.LedgerError("private")))
    pulse = worker._Heartbeat("unused", CELL, ledger)
    monkeypatch.setattr(pulse.stop, "wait", Mock(return_value=False))
    pulse._run()
    assert ledger.heartbeat.call_count == 1
    with pytest.raises(RuntimeError, match="heartbeat_failed:ledgererror") as error:
        pulse.check()
    assert "private" not in str(error.value)


def test_retry_budget_exhaustion_does_not_revive_expired_lease(monkeypatch):
    ledger = SimpleNamespace(heartbeat=Mock(side_effect=OSError("private")))
    pulse = worker._Heartbeat("unused", CELL, ledger)
    monkeypatch.setattr(pulse.stop, "wait", Mock(return_value=False))
    monkeypatch.setattr(worker.time, "monotonic", lambda: pulse.last_success + 241)
    pulse._run()
    ledger.heartbeat.assert_not_called()
    with pytest.raises(RuntimeError, match="heartbeat_lease_window_exhausted"):
        pulse.check()


def test_close_joins_and_renews_once_before_acceptance():
    ledger = SimpleNamespace(heartbeat=Mock())
    with worker._Heartbeat("unused", CELL, ledger) as pulse:
        pulse.close()
        assert not pulse.thread.is_alive()
        assert ledger.heartbeat.call_count == 2
    assert ledger.heartbeat.call_count == 2


def test_database_heartbeat_rejects_expired_lease_without_event(monkeypatch):
    connection = Mock()
    connection.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(rollout_postgres, "_transaction", lambda _: nullcontext(connection))
    monkeypatch.setattr(rollout_postgres, "_owned_active_row", lambda *a, **k: {"state": "claimed"})
    event = Mock()
    monkeypatch.setattr(rollout_postgres, "_event", event)
    with pytest.raises(rollout_ledger.LedgerError, match="expired"):
        rollout_postgres.heartbeat("unused", cell_id="cell", worker_id="worker", claim_id="claim")
    statements = [call.args[0] for call in connection.execute.call_args_list]
    assert any("lease_expires_at > clock_timestamp()" in sql for sql in statements)
    assert any("statement_timeout" in sql for sql in statements)
    assert any("lock_timeout" in sql for sql in statements)
    event.assert_not_called()


@pytest.mark.parametrize("fail_at", [None, "validation"])
def test_heartbeat_covers_config_run_local_and_validation(monkeypatch, tmp_path, fail_at):
    events = []
    active = False

    class Pulse:
        def __init__(self, *args):
            pass

        def __enter__(self):
            nonlocal active
            active = True
            events.append("heartbeat_enter")
            return self

        def check(self):
            assert active

        def close(self):
            nonlocal active
            active = False
            events.append("heartbeat_close")

        def __exit__(self, *args):
            self.close()

    config = {"execution": {"execution_id": "sha256:synthetic"}}

    def stage(name, result=None):
        def call(*args, **kwargs):
            assert active, name
            events.append(name)
            if name == fail_at:
                raise RuntimeError("synthetic validation failure")
            return result

        return call

    def accept(*args, **kwargs):
        assert not active
        events.append("accept")

    ledger = SimpleNamespace(
        claim=Mock(return_value=CELL),
        start=stage("start"),
        mark_grading=stage("grading"),
        accept=accept,
        request_retry_review=Mock(),
    )
    monkeypatch.setenv("FLEET_API_KEY", "synthetic-not-a-credential")
    monkeypatch.setattr(worker, "_Heartbeat", Pulse)
    monkeypatch.setattr(worker.httpx, "Client", lambda **kwargs: nullcontext(object()))
    monkeypatch.setattr(worker, "_selection_index", lambda _: {"task": {}})
    monkeypatch.setattr(worker, "build_config", stage("config", config))
    monkeypatch.setattr(worker, "_claim_receipt", lambda *args: {})
    monkeypatch.setattr(worker.self_hosted, "write_json_once", lambda *args: None)
    monkeypatch.setattr(worker.self_hosted, "run", stage("run", {"session_id": "session"}))
    monkeypatch.setattr(worker, "_record_local_result", stage("local"))
    monkeypatch.setattr(
        worker, "_accepted_receipt", stage("validation", {"receipt_sha256": "digest"})
    )
    monkeypatch.setattr(worker, "_safe_write_once", lambda *args: None)
    result = worker.run_one(
        database="unused",
        ledger=ledger,
        campaign={},
        selection={},
        universe_index={("model", "task", 1): {}},
        serving_block="route",
        worker_id="worker",
        output_root=tmp_path,
        claim_root=tmp_path,
        proxy_script=tmp_path / "unused",
    )
    assert result["accepted"] is (fail_at is None)
    assert "validation" in events
    if fail_at:
        ledger.request_retry_review.assert_called_once()
        assert "accept" not in events
    else:
        assert events.index("heartbeat_close") < events.index("accept")
        ledger.request_retry_review.assert_not_called()
