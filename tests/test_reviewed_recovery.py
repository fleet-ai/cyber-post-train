from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import evaluate, reviewed_recovery, reviewed_recovery_worker, rollout_ledger

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/fleet-reviewed-recovery-digest-guard-20260921.json"


def _intent_value(**changes) -> dict:
    body = {
        "schema_version": reviewed_recovery.INTENT_SCHEMA,
        "evaluation_plan_sha256": "a" * 64,
        "runtime_files_sha256": reviewed_recovery.runtime_identity(),
        "serving_block": "synthetic-route",
        "selected_cell_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
    }
    body.update(changes)
    return {**body, "sha256": reviewed_recovery._body_digest(body)}  # noqa: SLF001


def _intent_file(path: Path, **changes) -> Path:
    path.write_text(json.dumps(_intent_value(**changes)), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_private_self_digesting_intent_loads_and_binds_runtime(tmp_path: Path):
    intent = reviewed_recovery.load_intent(_intent_file(tmp_path / "intent.json"))
    assert intent.evaluation_plan_sha256 == "a" * 64
    assert intent.runtime_files_sha256 == reviewed_recovery.runtime_identity()
    assert intent.serving_block == "synthetic-route"
    assert len(intent.selected_cell_ids) == 2


def test_intent_rejects_public_permissions(tmp_path: Path):
    path = _intent_file(tmp_path / "intent.json")
    path.chmod(0o644)
    with pytest.raises(rollout_ledger.LedgerError, match="group or other"):
        reviewed_recovery.load_intent(path)


def test_intent_rejects_digest_drift_unknown_fields_and_runtime_drift(tmp_path: Path):
    path = _intent_file(tmp_path / "intent.json")
    value = json.loads(path.read_text())
    value["serving_block"] = "changed"
    path.write_text(json.dumps(value))
    with pytest.raises(rollout_ledger.LedgerError, match="self digest differs"):
        reviewed_recovery.load_intent(path)

    path = _intent_file(tmp_path / "intent-extra.json")
    value = json.loads(path.read_text())
    value["score"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(rollout_ledger.LedgerError, match="unknown fields"):
        reviewed_recovery.load_intent(path)

    runtime = reviewed_recovery.runtime_identity()
    runtime["reviewed_recovery.py"] = "b" * 64
    with pytest.raises(rollout_ledger.LedgerError, match="runtime identity differs"):
        reviewed_recovery.load_intent(
            _intent_file(tmp_path / "runtime-drift.json", runtime_files_sha256=runtime)
        )


def test_intent_rejects_duplicate_cells(tmp_path: Path):
    cell_id = str(uuid.uuid4())
    with pytest.raises(rollout_ledger.LedgerError, match="repeats a cell"):
        reviewed_recovery.load_intent(
            _intent_file(tmp_path / "intent.json", selected_cell_ids=[cell_id, cell_id])
        )


def test_intent_digest_commits_complete_roster():
    cells = [str(uuid.uuid4()), str(uuid.uuid4())]
    complete = _intent_value(selected_cell_ids=cells)
    subset = _intent_value(selected_cell_ids=cells[:1])
    reordered = _intent_value(selected_cell_ids=list(reversed(cells)))
    assert len({complete["sha256"], subset["sha256"], reordered["sha256"]}) == 3


def test_apply_and_preclaim_evidence_are_self_digesting_and_sanitized(tmp_path: Path):
    intent = reviewed_recovery.load_intent(_intent_file(tmp_path / "intent.json"))
    receipts = [
        reviewed_recovery._apply_evidence(intent),  # noqa: SLF001
        reviewed_recovery._preclaim_evidence(  # noqa: SLF001
            intent,
            claimable_cell_count=2,
            claim_id=str(uuid.uuid4()),
        ),
    ]
    for receipt in receipts:
        assert receipt["receipt_sha256"] == "sha256:" + digest(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        serialized = json.dumps(receipt).lower()
        assert not any(word in serialized for word in ("cell_id", "task_version", "session_id"))
        assert receipt["score_values_included"] is False


def test_recovery_runtime_is_separate_but_bound_by_the_private_intent():
    assert "reviewed_recovery.py" not in evaluate.RUNTIME_FILES
    assert "reviewed_recovery_worker.py" not in evaluate.RUNTIME_FILES
    assert set(reviewed_recovery.runtime_identity()) == set(reviewed_recovery.RUNTIME_FILES)


def test_worker_result_removes_private_ledger_identity():
    result = reviewed_recovery_worker._sanitized_result(  # noqa: SLF001
        {
            "serving_block": "synthetic-route",
            "claimed": True,
            "accepted": False,
            "ledger_cell_id": str(uuid.uuid4()),
            "failure_code": "post_claim.connecterror",
        }
    )
    assert set(result) == {"serving_block", "claimed", "accepted", "failure_code"}
    assert "ledger_cell_id" not in json.dumps(result)


def test_worker_uses_prepared_eval_path_and_sanitizes_terminal(monkeypatch, tmp_path: Path):
    plan = {
        "sha256": "a" * 64,
        "routes": {"synthetic-route": {"task_versions": ["task-version"], "model": "model"}},
        "models": {"model": {}},
        "tasks": [{"task_version_id": "task-version"}],
        "pass_k": 1,
        "concurrency": 1,
        "images": {"agent": "agent-image", "proxy": "proxy-image"},
    }
    proof = {"task_bindings": {"task-version": []}}
    intent_path = _intent_file(
        tmp_path / "intent.json",
        evaluation_plan_sha256=plan["sha256"],
        selected_cell_ids=[str(uuid.uuid4())],
    )
    evaluation_directory = tmp_path / "prepared"
    evaluation_directory.mkdir()
    calls: list[str] = []
    monkeypatch.setattr(
        evaluate,
        "checked_preflight",
        lambda directory: (plan, proof) if directory == evaluation_directory else None,
    )
    monkeypatch.setattr(evaluate, "check_images", lambda _plan: calls.append("images"))
    monkeypatch.setattr(evaluate, "check_route", lambda *_args: calls.append("route"))
    monkeypatch.setattr(
        reviewed_recovery_worker.httpx,
        "Client",
        lambda **_kwargs: type(
            "Client",
            (),
            {"__enter__": lambda self: self, "__exit__": lambda self, *_args: None},
        )(),
    )
    monkeypatch.setattr(
        reviewed_recovery_worker.rollout_postgres,
        "verify_plan",
        lambda *_args: calls.append("plan"),
    )
    monkeypatch.setattr(
        reviewed_recovery_worker.rollout_postgres,
        "summary",
        lambda _dsn: {"total": 1, "by_state": {"accepted": 1}},
    )
    monkeypatch.setattr(
        reviewed_recovery_worker.rollout_ledger,
        "_plan_rows",
        lambda _path: [
            {
                "model_id": "model",
                "task_version_id": "task-version",
                "attempt": 1,
            }
        ],
    )
    monkeypatch.setattr(
        reviewed_recovery,
        "apply_intent",
        lambda _dsn, *, intent: {
            "receipt_sha256": "sha256:" + "b" * 64,
            "reviewed_intent_sha256": intent.sha256,
        },
    )

    def fake_run_one(**kwargs):
        kwargs["pre_execution_guard"](None)
        return {
            "serving_block": kwargs["serving_block"],
            "claimed": True,
            "accepted": True,
            "ledger_cell_id": str(uuid.uuid4()),
            "receipt_sha256": "sha256:" + "c" * 64,
        }

    monkeypatch.setattr(reviewed_recovery_worker.rollout_worker, "run_one", fake_run_one)
    monkeypatch.setenv("FLEET_API_KEY", "test-only")
    monkeypatch.setenv("AGENT_HARNESS_IMAGE", "before-agent")
    monkeypatch.setenv("FIXED_PROXY_IMAGE", "before-proxy")
    result = reviewed_recovery_worker.run(
        evaluation_directory=evaluation_directory,
        output_root=tmp_path / "output",
        dsn="postgresql://local/test",
        intent_path=intent_path,
        worker_id="synthetic-worker",
    )
    assert calls == ["plan", "images", "route", "route"]
    assert result["accepted"] is True and result["accepted_cells"] == 1
    serialized = json.dumps(result).lower()
    assert "ledger_cell_id" not in serialized
    assert result["private_cell_task_session_or_trace_identifiers_included"] is False


@pytest.mark.parametrize("worker_id", ["synthetic-worker", "../../private-path"])
def test_worker_startup_failure_is_sanitized(monkeypatch, tmp_path: Path, capsys, worker_id):
    output_root = tmp_path / "output"
    monkeypatch.setenv("TEST_RECOVERY_DSN", "postgresql://local/test")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reviewed_recovery_worker",
            "--evaluation-directory",
            "prepared-eval",
            "--output-root",
            str(output_root),
            "--postgres-dsn-env",
            "TEST_RECOVERY_DSN",
            "--reviewed-recovery-intent",
            "private-intent.json",
            "--worker-id",
            worker_id,
        ],
    )

    def fail(**_kwargs):
        raise RuntimeError("private-driver-detail")

    monkeypatch.setattr(reviewed_recovery_worker, "run", fail)
    assert reviewed_recovery_worker.main() == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["controller_failure_code"] == "reviewed_recovery_startup_failed"
    assert "private-driver-detail" not in json.dumps(printed)
    assert not output_root.exists()


def test_sanitized_evidence_is_self_digesting_and_records_no_live_operation():
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["sha256"] == digest(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    assert evidence["privacy"] == {
        "score_values_read_or_recorded": False,
        "capability_outcomes_read_or_recorded": False,
        "live_cell_task_session_or_trace_identifiers_recorded": False,
        "prompt_response_flag_reward_or_trace_content_recorded": False,
        "credentials_recorded": False,
        "final_eight_task_set_accessed": False,
    }
    assert set(evidence["operations"].values()) == {0}
