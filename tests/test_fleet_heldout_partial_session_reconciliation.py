from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

import pytest

from evals.fleet import heldout_partial_session_reconciliation as recovery
from evals.fleet import rollout_ledger


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _intent_value(tmp_path: Path, **overrides):
    values = {
        "evaluation_plan_sha256": _sha("plan"),
        "serving_block": "qwen-base-dedicated",
        "source_output_root": "/mnt/sfs/jobs/source-eval",
        "source_database": "source_eval",
        "source_job_uid": str(uuid.uuid4()),
        "source_job_terminal_receipt_sha256": _sha("terminal"),
        "cell_id": str(uuid.uuid4()),
        "row_binding_sha256": _sha("row"),
        "config_sha256": _sha("config"),
        "recovery_output_root": "/mnt/sfs/jobs/recovery-eval",
    }
    values.update(overrides)
    return recovery.build_intent_value(**values)


def _load(tmp_path: Path, value: dict) -> recovery.Intent:
    path = tmp_path / "intent.json"
    path.write_text(json.dumps(value))
    path.chmod(0o600)
    return recovery.load_intent(path)


def test_private_intent_binds_single_source_and_destination(tmp_path: Path) -> None:
    value = _intent_value(tmp_path)
    intent = _load(tmp_path, value)
    assert intent.selected_cell_ids == (value["cell_id"],)
    assert intent.source_output_root != intent.recovery_output_root


def test_intent_rejects_digest_drift_alias_and_public_mode(tmp_path: Path) -> None:
    value = _intent_value(tmp_path)
    value["config_sha256"] = _sha("changed")
    with pytest.raises(rollout_ledger.LedgerError, match="intent digest differs"):
        _load(tmp_path, value)

    value = _intent_value(
        tmp_path,
        recovery_output_root="/mnt/sfs/jobs/source-eval",
    )
    with pytest.raises(rollout_ledger.LedgerError, match="aliases"):
        _load(tmp_path, value)

    path = tmp_path / "public.json"
    path.write_text(json.dumps(_intent_value(tmp_path)))
    path.chmod(0o644)
    with pytest.raises(rollout_ledger.LedgerError, match="not private"):
        recovery.load_intent(path)


def test_source_terminal_observation_binds_job_database_plan_and_privacy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    intent = _load(tmp_path, _intent_value(tmp_path))
    object.__setattr__(intent, "source_output_root", str(source))
    observation = {
        "schema": recovery.TERMINAL_OBSERVATION_SCHEMA,
        "job": {"uid": intent.source_job_uid},
        "output_root": {"exists": True, "path": str(source)},
        "database": {
            "name": intent.source_database,
            "summary": {"plan_sha256": intent.evaluation_plan_sha256},
        },
        "decision": {
            "rollout_retry_performed": False,
            "score_blind_reconciliation_required": True,
            "score_read_or_generated": False,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_responses_flags_rewards_or_trace_content_included": False,
            "score_values_included": False,
        },
    }
    observation["sha256"] = recovery.crypto.digest_without(observation, "sha256")
    (source / "TERMINAL_OBSERVATION.json").write_text(json.dumps(observation))
    object.__setattr__(
        intent,
        "source_job_terminal_receipt_sha256",
        observation["sha256"].removeprefix("sha256:"),
    )
    recovery._validate_source_terminal(intent)

    observation["decision"]["score_read_or_generated"] = True
    observation["sha256"] = recovery.crypto.digest_without(observation, "sha256")
    (source / "TERMINAL_OBSERVATION.json").write_text(json.dumps(observation))
    object.__setattr__(
        intent,
        "source_job_terminal_receipt_sha256",
        observation["sha256"].removeprefix("sha256:"),
    )
    with pytest.raises(rollout_ledger.LedgerError, match="binding differs"):
        recovery._validate_source_terminal(intent)


def test_source_row_requires_exact_failed_partial_session(tmp_path: Path, monkeypatch) -> None:
    intent = _load(tmp_path, _intent_value(tmp_path))
    row = {field: None for field in recovery.ROW_BINDING_FIELDS}
    row.update(
        cell_id=intent.cell_id,
        state="retry_review",
        failure_code=recovery.FAILURE_CODE,
        reconciliation_digest=None,
        session_ingest_status="failed",
        agent_exit_code=0,
        agent_termination="completed",
        serving_block=intent.serving_block,
        config_sha256=intent.config_sha256,
    )
    object.__setattr__(intent, "row_binding_sha256", recovery._row_binding(row))
    monkeypatch.setattr(recovery.stored, "_rows", lambda *_a, **_k: [row])
    monkeypatch.setattr(recovery.stored, "_validate_record_digest", lambda value: value)
    assert recovery._source_row_in_connection(object(), intent, lock=False) is row

    row["session_ingest_status"] = "completed"
    with pytest.raises(rollout_ledger.LedgerError, match="database binding differs"):
        recovery._source_row_in_connection(object(), intent, lock=False)


def test_run_resumes_same_session_before_accepting(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "out"
    intent = _load(
        tmp_path,
        _intent_value(
            tmp_path,
            recovery_output_root=str(output).replace(str(tmp_path), "/mnt/sfs/jobs"),
        ),
    )
    # Keep the unit test provider-free while exercising the ordered control path.
    object.__setattr__(intent, "recovery_output_root", str(output))
    row = {"record_sha256": _sha("record")}
    config = {"config_sha256": intent.config_sha256}
    source = {
        "session_id": "session",
        "verifier_execution_id": "verifier",
        "original_ingest_sha256": _sha("ingest"),
    }
    calls = []
    monkeypatch.setattr(
        recovery,
        "_validate_source_terminal",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        recovery,
        "_frozen_config",
        lambda *_a, **_k: (row, config),
    )
    monkeypatch.setattr(
        recovery,
        "_validated_source",
        lambda *_a, **_k: (tmp_path / "attempt", source),
    )

    def observe(*_a, **_k):
        calls.append("observe")
        return {"receipt_sha256": _sha("observer")}

    def resume(*_a, **_k):
        calls.append("resume")
        return {
            "resumed": True,
            "session_id": "session",
            "verifier_execution_id": "verifier",
            "same_session_id_preserved": True,
            "model_or_verifier_replayed": False,
            "receipt_sha256": _sha("resumed"),
        }

    def accept(*_a, **_k):
        calls.append("accept")
        assert calls == ["observe", "resume", "accept"]
        return {
            "accepted_same_completed_session_count": 1,
            "receipt_sha256": _sha("accepted"),
        }

    monkeypatch.setattr(recovery.self_hosted, "observe_partial_session_resume", observe)
    monkeypatch.setattr(recovery.self_hosted, "resume_partial_session_trace", resume)
    monkeypatch.setattr(recovery, "_accept", accept)
    receipt = recovery.run("unused", intent=intent, evaluation_directory=tmp_path, client=object())
    assert receipt["accepted_same_completed_session_count"] == 1
    assert calls == ["observe", "resume", "accept"]
    assert (output / "TERMINAL.json").is_file()


def test_run_holds_when_resume_identity_is_ambiguous(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "out"
    intent = _load(
        tmp_path,
        _intent_value(
            tmp_path,
            recovery_output_root=str(output).replace(str(tmp_path), "/mnt/sfs/jobs"),
        ),
    )
    object.__setattr__(intent, "recovery_output_root", str(output))
    monkeypatch.setattr(
        recovery,
        "_validate_source_terminal",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        recovery,
        "_frozen_config",
        lambda *_a, **_k: (
            {"record_sha256": _sha("record")},
            {"config_sha256": intent.config_sha256},
        ),
    )
    monkeypatch.setattr(
        recovery,
        "_validated_source",
        lambda *_a, **_k: (
            tmp_path / "attempt",
            {
                "session_id": "expected",
                "verifier_execution_id": "verifier",
                "original_ingest_sha256": _sha("ingest"),
            },
        ),
    )
    monkeypatch.setattr(
        recovery.self_hosted,
        "observe_partial_session_resume",
        lambda *_a, **_k: {"receipt_sha256": _sha("observer")},
    )
    monkeypatch.setattr(
        recovery.self_hosted,
        "resume_partial_session_trace",
        lambda *_a, **_k: {
            "resumed": True,
            "session_id": "different",
            "verifier_execution_id": "verifier",
            "same_session_id_preserved": True,
            "model_or_verifier_replayed": False,
            "receipt_sha256": _sha("resumed"),
        },
    )
    accepted = False

    def forbidden(*_a, **_k):
        nonlocal accepted
        accepted = True

    monkeypatch.setattr(recovery, "_accept", forbidden)
    with pytest.raises(rollout_ledger.LedgerError, match="resume receipt differs"):
        recovery.run("unused", intent=intent, evaluation_directory=tmp_path, client=object())
    assert accepted is False


def test_accept_holds_before_transaction_when_authority_changes(
    tmp_path: Path, monkeypatch
) -> None:
    intent = _load(tmp_path, _intent_value(tmp_path))
    monkeypatch.setattr(recovery, "_source_row", lambda *_a, **_k: {"cell_id": intent.cell_id})
    observations = iter(({"receipt": "first"}, {"receipt": "second"}))
    monkeypatch.setattr(
        recovery.stored,
        "_session_receipt",
        lambda *_a, **_k: next(observations),
    )

    def forbidden(*_a, **_k):
        raise AssertionError("database transaction must not start")

    monkeypatch.setattr(recovery.rollout_postgres, "_transaction", forbidden)
    with pytest.raises(rollout_ledger.LedgerError, match="authority changed"):
        recovery._accept(
            "unused",
            intent,
            object(),
            {"config_sha256": intent.config_sha256},
            _sha("recovery"),
        )


def test_main_resolves_real_dedicated_database_dsn(tmp_path: Path, monkeypatch, capsys) -> None:
    intent_path = tmp_path / "intent.json"
    intent_path.write_text(json.dumps(_intent_value(tmp_path)))
    intent_path.chmod(0o600)
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    admin = "postgresql://worker:secret@postgres.internal:5432/postgres?sslmode=require"
    expected = recovery.stored.dedicated_dsn(admin, "source_eval")
    observed = {}

    def run(dsn, *, intent, evaluation_directory, client):
        observed.update(
            dsn=dsn,
            database=intent.source_database,
            evaluation_directory=evaluation_directory,
        )
        return {
            "accepted_same_completed_session_count": 1,
            "receipt_sha256": "sha256:" + _sha("accepted"),
        }

    monkeypatch.setenv("ROLLOUT_DATABASE_URL", admin)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setattr(recovery, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "heldout_partial_session_reconciliation",
            "--intent",
            str(intent_path),
            "--evaluation-directory",
            str(evaluation),
            "--postgres-database",
            "source_eval",
        ],
    )
    assert recovery.main() == 0
    assert observed == {
        "dsn": expected,
        "database": "source_eval",
        "evaluation_directory": evaluation,
    }
    output = json.loads(capsys.readouterr().out)
    assert output["accepted_same_completed_session_count"] == 1
    assert output["new_session_created"] is False
