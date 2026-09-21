from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest

from evals.fleet import (
    stored_session_reconciliation as reconciliation,
)
from evals.fleet import (
    stored_session_reconciliation_job as job,
)

ROOT = Path(__file__).parents[1]
PLAN = ROOT / "configs/evaluation/qwen38-lr30-step76-stored-session-reconciliation-v1.json"
EVIDENCE = (
    ROOT / "docs/evidence/qwen38-lr30-step76-fleet-dev17-seed43-terminal-census-20260921.json"
)


def _intent(path: Path, plan_path: Path = PLAN) -> Path:
    plan = json.loads(plan_path.read_text())
    source = plan["source"]
    body = {
        "schema_version": reconciliation.INTENT_SCHEMA,
        "evaluation_plan_sha256": source["evaluation_plan_sha256"].removeprefix("sha256:"),
        "runtime_files_sha256": reconciliation.runtime_identity(),
        "serving_block": "lr30",
        "source_output_root": source["evaluation_directory"],
        "source_job_uid": source["job_uid"],
        "source_job_terminal_receipt_sha256": source["terminal_receipt_sha256"].removeprefix(
            "sha256:"
        ),
        "selected_cell_ids": [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"stored-session-test-{index}"))
            for index in range(source["selected_cell_count"])
        ],
    }
    value = {**body, "sha256": reconciliation._body_digest(body)}  # noqa: SLF001
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o600)
    return path


def test_terminal_census_is_score_blind_and_self_digesting():
    evidence = json.loads(EVIDENCE.read_text())
    plan = json.loads(PLAN.read_text())
    assert evidence["sha256"] == reconciliation._body_digest(  # noqa: SLF001
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    assert plan["source"]["terminal_receipt_path"] == str(EVIDENCE.relative_to(ROOT))
    assert plan["source"]["terminal_receipt_sha256"] == "sha256:" + evidence["sha256"]
    census = evidence["score_blind_ledger_census"]
    assert census["accepted"] == 6
    assert census["retry_review"] == 10
    assert census["claimed"] == census["stale_active"] == 1
    assert census["unresolved_exact_completed_sessions"] == 11
    assert census["unresolved_exact_pinned_task_version_metadata_matches"] == 11
    assert census["unresolved_reference_trace_payloads_empty"] == 11
    assert census["stored_session_only_reconciliation_eligible"] == 11
    assert census["rollout_regeneration_eligible"] == 0
    assert census["score_values_included"] is False


def test_package_is_cpu_only_create_once_alert_off_and_private(tmp_path):
    intent = _intent(tmp_path / "intent.json")
    package = job.render(repo_root=ROOT, plan_path=PLAN, intent_path=intent)
    assert package.job["metadata"]["annotations"][job.FAILURE_ALERT_ANNOTATION] == "off"
    assert package.job["spec"]["backoffLimit"] == 0
    pod = package.job["spec"]["template"]["spec"]
    assert pod["priorityClassName"] == "c1"
    assert "nvidia.com/gpu" not in json.dumps(pod["containers"][0]["resources"])
    assert package.config_map["immutable"] is True
    assert package.secret["immutable"] is True
    assert package.proof["gpu_request"] == 0
    assert package.proof["model_generation_performed"] is False
    assert package.proof["scoring_call_performed"] is False
    public = json.dumps(package.proof)
    private = reconciliation.load_intent(intent)
    assert all(cell_id not in public for cell_id in private.selected_cell_ids)


def test_intent_permissions_and_reviewed_code_bytes_fail_closed(tmp_path):
    intent = _intent(tmp_path / "intent.json")
    os.chmod(intent, 0o644)
    with pytest.raises(Exception, match="group or other"):
        reconciliation.load_intent(intent)
    os.chmod(intent, 0o600)
    plan = json.loads(PLAN.read_text())
    plan["code_sha256"]["evals/fleet/stored_session_reconciliation.py"] = "0" * 64
    drifted = tmp_path / "plan.json"
    drifted.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(job.PackageError, match="source bytes differ"):
        job.render(repo_root=ROOT, plan_path=drifted, intent_path=intent)


def test_public_plan_private_intent_and_terminal_receipt_share_one_digest(tmp_path):
    intent = _intent(tmp_path / "intent.json")
    plan = json.loads(PLAN.read_text())
    plan["source"]["terminal_receipt_sha256"] = "sha256:" + "0" * 64
    drifted = tmp_path / "plan.json"
    drifted.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(job.PackageError, match="terminal receipt digest differs"):
        job.render(repo_root=ROOT, plan_path=drifted, intent_path=intent)


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("schema",), "wrong_terminal_schema"),
        (("source_controller", "job", "uid"), str(uuid.uuid4())),
        (("source_controller", "job", "active"), 1),
        (("source_controller", "job", "root_failure_alert_annotation"), "on"),
        (("frozen_evaluation", "evaluation_plan_sha256"), "sha256:" + "0" * 64),
        (("score_blind_ledger_census", "stored_session_only_reconciliation_eligible"), 10),
    ],
)
def test_self_digested_terminal_receipt_with_wrong_semantics_fails_closed(
    tmp_path, monkeypatch, field_path, replacement
):
    evidence = json.loads(EVIDENCE.read_text())
    target = evidence
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = replacement
    evidence["sha256"] = reconciliation._body_digest(  # noqa: SLF001
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    plan = json.loads(PLAN.read_text())
    plan["source"]["terminal_receipt_sha256"] = "sha256:" + evidence["sha256"]
    drifted = tmp_path / "plan.json"
    drifted.write_text(json.dumps(plan), encoding="utf-8")
    intent = _intent(tmp_path / "intent.json", drifted)
    original_load = job._load  # noqa: SLF001

    def load(path):
        return evidence if Path(path).resolve() == EVIDENCE.resolve() else original_load(path)

    monkeypatch.setattr(job, "_load", load)
    with pytest.raises(job.PackageError, match="terminal receipt semantics differ"):
        job.render(repo_root=ROOT, plan_path=drifted, intent_path=intent)


def test_self_digested_terminal_receipt_with_unknown_field_fails_closed(tmp_path, monkeypatch):
    evidence = json.loads(EVIDENCE.read_text())
    evidence["source_controller"]["job"]["unknown"] = "not allowed"
    evidence["sha256"] = reconciliation._body_digest(  # noqa: SLF001
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    plan = json.loads(PLAN.read_text())
    plan["source"]["terminal_receipt_sha256"] = "sha256:" + evidence["sha256"]
    drifted = tmp_path / "plan.json"
    drifted.write_text(json.dumps(plan), encoding="utf-8")
    intent = _intent(tmp_path / "intent.json", drifted)
    original_load = job._load  # noqa: SLF001

    def load(path):
        return evidence if Path(path).resolve() == EVIDENCE.resolve() else original_load(path)

    monkeypatch.setattr(job, "_load", load)
    with pytest.raises(job.PackageError, match="Job shape is invalid"):
        job.render(repo_root=ROOT, plan_path=drifted, intent_path=intent)


def test_reference_identity_requires_exact_pinned_version_and_empty_payload(monkeypatch):
    session_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())
    response = {
        "session_id": session_id,
        "job_id": None,
        "eval_task_id": task_id,
        "eval_task_version_id": version_id,
        "task_key": "safe-test-task",
        "status": "no_reference_traces",
        "primary_reference_trace_id": None,
        "reference_trace_count": 0,
        "reference_traces": [],
    }

    def request(_client, _method, path, **_kwargs):
        if path == "/v1/sessions":
            return {
                "sessions": [
                    {
                        "session_id": session_id,
                        "task_key": "safe-test-task",
                        "eval_task_id": task_id,
                        "model": "qwen/safe-test",
                        "status": "completed",
                        "verifier_execution": {"id": "verifier-run", "score": 0.5},
                    }
                ],
                "has_more": False,
            }
        assert path == f"/v1/sessions/{session_id}/reference-traces"
        return response

    monkeypatch.setattr(reconciliation.self_hosted, "_request", request)
    row = {
        "cell_id": str(uuid.uuid4()),
        "local_session_id": session_id,
        "task_key": "safe-test-task",
        "task_version_id": version_id,
        "endpoint_model_id": "qwen/safe-test",
        "verifier_execution_id": "verifier-run",
        "score": 0.5,
        "config_sha256": "a" * 64,
        "record_sha256": "b" * 64,
    }
    receipt = reconciliation._session_receipt(  # noqa: SLF001
        object(),
        row=row,
        config={"config_sha256": "a" * 64},
        artifact_binding_sha256="c" * 64,
    )
    assert receipt["receipt"]["authoritative_pinned_task_version_metadata_only"] is True
    assert receipt["receipt"]["reference_trace_content_returned"] is False
    assert reconciliation._validate_cell_observation(row, receipt) == receipt["receipt"]  # noqa: SLF001

    forged = json.loads(json.dumps(receipt))
    forged["receipt"]["unknown_field"] = "not allowed"
    forged["receipt"]["receipt_sha256"] = reconciliation.crypto.digest_without(
        forged["receipt"], "receipt_sha256"
    )
    with pytest.raises(Exception, match="receipt is invalid"):
        reconciliation._validate_cell_observation(row, forged)  # noqa: SLF001

    response["eval_task_version_id"] = str(uuid.uuid4())
    with pytest.raises(Exception, match="version identity"):
        reconciliation._session_receipt(  # noqa: SLF001
            object(),
            row=row,
            config={"config_sha256": "a" * 64},
            artifact_binding_sha256="c" * 64,
        )

    response["eval_task_version_id"] = version_id
    response["reference_trace_count"] = 1
    response["reference_traces"] = [{"private": "must fail closed"}]
    with pytest.raises(Exception, match="contains content"):
        reconciliation._session_receipt(  # noqa: SLF001
            object(),
            row=row,
            config={"config_sha256": "a" * 64},
            artifact_binding_sha256="c" * 64,
        )


def test_observe_rejects_plan_digest_mismatch_before_database_access(tmp_path, monkeypatch):
    intent = reconciliation.load_intent(_intent(tmp_path / "intent.json"))
    monkeypatch.setattr(
        reconciliation.evaluate,
        "checked_preflight",
        lambda _path: ({"sha256": "b" * 64}, {}),
    )

    def database_access(*_args, **_kwargs):
        raise AssertionError("database must not be accessed after plan drift")

    monkeypatch.setattr(reconciliation.rollout_postgres, "verify_plan", database_access)
    with pytest.raises(Exception, match="differs from evaluation plan"):
        reconciliation.observe(
            "unused",
            intent=intent,
            evaluation_directory=tmp_path,
            client=object(),
        )


def test_atomic_acceptance_binds_each_distinct_cell_receipt(monkeypatch, tmp_path):
    intent = reconciliation.load_intent(_intent(tmp_path / "intent.json"))
    rows = []
    observations = []
    for index, cell_id in enumerate(intent.selected_cell_ids):
        row = {
            "cell_id": cell_id,
            "state": "retry_review",
            "reconciliation_digest": None,
            "record_sha256": f"{index + 1:x}" * 64,
            "local_session_id": f"session-{index}",
            "session_id": None,
            "session_ingest_status": "completed",
            "agent_exit_code": 1,
            "agent_termination": "process_error",
            "task_version_id": f"version-{index}",
            "endpoint_model_id": "qwen/test",
            "verifier_execution_id": f"verifier-{index}",
            "config_sha256": "a" * 64,
            "worker_id": f"worker-{index}",
            "claim_id": f"claim-{index}",
            "failure_code": "process_error",
        }
        identity = {
            "cell_id": cell_id,
            "session_id": row["local_session_id"],
            "task_version_id": row["task_version_id"],
            "model": row["endpoint_model_id"],
            "verifier_execution_id": row["verifier_execution_id"],
        }
        body = {
            "schema_version": "fleet-stored-session-cell-observation-v1",
            "identity_binding_sha256": reconciliation._body_digest(identity),  # noqa: SLF001
            "config_sha256": row["config_sha256"],
            "local_record_sha256": row["record_sha256"],
            "artifact_binding_sha256": "b" * 64,
            "authoritative_session_metadata_sha256": f"{index + 2:x}" * 64,
            "authoritative_pinned_task_version_metadata_only": True,
            "reference_trace_content_returned": False,
            "authoritative_score_finite_and_equal_to_private_local_result": True,
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
        }
        receipt = {
            **body,
            "receipt_sha256": reconciliation.crypto.digest_without(body, "receipt_sha256"),
        }
        rows.append(row)
        observations.append(
            {
                "cell_id": cell_id,
                "session_id": row["local_session_id"],
                "local_record_sha256": row["record_sha256"],
                "receipt": receipt,
            }
        )

    class Result:
        rowcount = 1

    class Connection:
        def __init__(self):
            self.updates = []

        def execute(self, statement, parameters=None):
            if str(statement).lstrip().startswith("UPDATE rollout_cells"):
                self.updates.append(parameters)
            return Result()

    connection = Connection()

    @contextmanager
    def transaction(_dsn):
        yield connection

    monkeypatch.setattr(reconciliation.rollout_postgres, "_transaction", transaction)
    monkeypatch.setattr(reconciliation, "_rows", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(reconciliation, "_validate_record_digest", lambda _row: {})
    monkeypatch.setattr(reconciliation.rollout_postgres, "_event", lambda *_args, **_kwargs: None)
    reconciliation.accept_roster("unused", intent=intent, observations=observations)

    persisted = {parameters[3]: parameters[1] for parameters in connection.updates}
    expected = {
        observation["cell_id"]: observation["receipt"]["receipt_sha256"]
        for observation in observations
    }
    assert persisted == expected
    assert len(set(persisted.values())) == len(rows)
