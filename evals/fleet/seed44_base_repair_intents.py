"""Prepare the two private seed-44 Base repair intents without mutating the ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import (
    evaluate,
    reviewed_recovery_v2,
    rollout_ledger,
    rollout_postgres,
    rollout_worker,
    seed44_base_repair_job,
    stored_session_reconciliation,
    stored_session_reconciliation_v2,
)
from evals.fleet import opencode_self_hosted as self_hosted


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("repair plan must be one JSON object")
    return value


def _rows(dsn: str, failure_code: str, *, with_local_result: bool) -> list[dict[str, Any]]:
    join = "JOIN" if with_local_result else "LEFT JOIN"
    with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        rows = connection.execute(
            f"""
            SELECT c.*, r.execution_id, r.execution_generation, r.run_id,
                   r.session_id AS local_session_id, r.verifier_execution_id,
                   r.score, r.config_sha256, r.artifact_directory,
                   r.trace_path, r.trace_sha256, r.result_path, r.result_sha256,
                   r.reward_path, r.reward_sha256, r.session_ingest_path,
                   r.session_ingest_sha256, r.cleanup_path, r.cleanup_sha256,
                   r.session_ingest_status, r.agent_exit_code,
                   r.agent_termination, r.elapsed_seconds, r.record_sha256
            FROM rollout_cells c
            {join} rollout_local_results r ON r.cell_id = c.cell_id
            WHERE c.state = 'retry_review' AND c.failure_code = %s
            ORDER BY c.cell_id
            """,  # noqa: S608
            (failure_code,),
        ).fetchall()
    return rows


def _stored_value(plan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    source = plan["source"]
    body = {
        "schema_version": stored_session_reconciliation_v2.INTENT_SCHEMA,
        "evaluation_plan_sha256": source["evaluation_plan_sha256"].removeprefix("sha256:"),
        "runtime_files_sha256": stored_session_reconciliation_v2.runtime_identity(),
        "serving_block": "base",
        "source_output_root": source["evaluation_directory"],
        "source_database": source["database"],
        "source_job_uid": source["job_uid"],
        "source_job_terminal_receipt_sha256": source["terminal_evidence_sha256"].removeprefix(
            "sha256:"
        ),
        "selected_cell_ids": [row["cell_id"] for row in rows],
        "expected_agent_exit_code": 0,
        "expected_agent_termination": "output_limit",
        "expected_failure_code": "authoritative_scoring_started.runtimeerror",
    }
    return {
        **body,
        "sha256": stored_session_reconciliation_v2._body_digest(body),  # noqa: SLF001
    }


def _selected_cell(
    *, source_root: Path, row: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    execution_name = config["execution"]["execution_id"].removeprefix("sha256:")
    attempt = source_root / "attempts" / execution_name
    claim = source_root / "claims" / f"{execution_name}.json"
    paths = {
        "claim_file_sha256": claim,
        "binding_file_sha256": attempt / "binding.json",
        "prompt_file_sha256": attempt / "prompt.txt",
        "failure_file_sha256": attempt / "failure.json",
        "cleanup_file_sha256": attempt / "cleanup.json",
    }
    return {
        "cell_id": row["cell_id"],
        **{
            field: reviewed_recovery_v2._prefixed_file_sha256(path)  # noqa: SLF001
            for field, path in paths.items()
        },
    }


def _recovery_value(
    plan_packet: dict[str, Any],
    evaluation_plan: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    prior_intent_sha256: str,
    client: httpx.Client,
) -> dict[str, Any]:
    source = plan_packet["source"]
    scientific = reviewed_recovery_v2._scientific_index(  # noqa: SLF001
        evaluation_plan, Path(source["evaluation_directory"]) / "plan.csv"
    )
    selected = {row["task_version_id"]: row for row in evaluation_plan["tasks"]}
    source_root = Path(source["evaluation_directory"]).resolve()
    cells = []
    for row in rows:
        key = (row["model_id"], row["task_version_id"], int(row["attempt"]))
        if key not in scientific or row["task_version_id"] not in selected:
            raise rollout_ledger.LedgerError("repair cell is absent from the frozen plan")
        config = rollout_worker.build_config(
            evaluation_plan,
            row,
            scientific[key],
            selected[row["task_version_id"]],
            client,
        )
        cells.append(_selected_cell(source_root=source_root, row=row, config=config))
    body = {
        "schema_version": reviewed_recovery_v2.INTENT_SCHEMA,
        "evaluation_plan_sha256": source["evaluation_plan_sha256"].removeprefix("sha256:"),
        "runtime_files_sha256": reviewed_recovery_v2.runtime_identity(),
        "serving_block": "base",
        "source_output_root": source["evaluation_directory"],
        "source_database": source["database"],
        "source_job_uid": source["job_uid"],
        "source_job_terminal_receipt_sha256": source["terminal_evidence_sha256"].removeprefix(
            "sha256:"
        ),
        "prior_stored_session_intent_sha256": prior_intent_sha256,
        "selected_cells": cells,
    }
    return {**body, "sha256": reviewed_recovery_v2._body_digest(body)}  # noqa: SLF001


def prepare(
    *,
    repo_root: Path,
    plan_path: Path,
    output_root: Path,
    admin_dsn: str,
    database: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("create-once private repair intent output already exists")
    plan_packet = _load(plan_path)
    if plan_packet.get("schema") != seed44_base_repair_job.SCHEMA:
        raise ValueError("repair plan schema is unsupported")
    if plan_packet.get("sha256") != seed44_base_repair_job._canonical_digest(  # noqa: SLF001
        {key: item for key, item in plan_packet.items() if key != "sha256"}
    ):
        raise ValueError("repair plan self digest differs")
    source = plan_packet["source"]
    if database != source["database"]:
        raise ValueError("runtime database differs from repair plan")
    dsn = stored_session_reconciliation.dedicated_dsn(admin_dsn, database)
    evaluation_directory = Path(source["evaluation_directory"])
    evaluation_plan, proof = evaluate.checked_preflight(evaluation_directory)
    if evaluation_plan["sha256"] != source["evaluation_plan_sha256"].removeprefix("sha256:"):
        raise ValueError("prepared evaluation plan differs from repair plan")
    rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    stored_rows = _rows(dsn, "authoritative_scoring_started.runtimeerror", with_local_result=True)
    recovery_rows = _rows(dsn, reviewed_recovery_v2.SOURCE_FAILURE_CODE, with_local_result=False)
    if len(stored_rows) != 5 or len(recovery_rows) != 2:
        raise rollout_ledger.LedgerError("repair classification counts changed")
    if any(
        row["agent_exit_code"] != 0
        or row["agent_termination"] != "output_limit"
        or row["session_ingest_status"] != "completed"
        or row["retry_count"] != 0
        or row["reconciliation_digest"] is not None
        for row in stored_rows
    ):
        raise rollout_ledger.LedgerError("stored-session repair evidence drifted")
    if any(
        row["execution_id"] is not None
        or row["local_session_id"] is not None
        or row["session_id"] is not None
        or row["receipt_digest"] is not None
        or row["retry_count"] != 0
        or row["reconciliation_digest"] is not None
        for row in recovery_rows
    ):
        raise rollout_ledger.LedgerError("rollout repair absence evidence drifted")
    evaluation_plan["task_bindings"] = proof["task_bindings"]
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise ValueError("FLEET_API_KEY is required")
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=60,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")  # noqa: SLF001
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise rollout_ledger.LedgerError("Fleet team identity required")
        stored_value = _stored_value(plan_packet, stored_rows)
        stored_intent = stored_session_reconciliation_v2.ExactStoredSessionIntent(
            evaluation_plan_sha256=stored_value["evaluation_plan_sha256"],
            runtime_files_sha256=stored_value["runtime_files_sha256"],
            serving_block=stored_value["serving_block"],
            source_output_root=stored_value["source_output_root"],
            source_database=stored_value["source_database"],
            source_job_uid=stored_value["source_job_uid"],
            source_job_terminal_receipt_sha256=stored_value["source_job_terminal_receipt_sha256"],
            selected_cell_ids=tuple(stored_value["selected_cell_ids"]),
            expected_agent_exit_code=stored_value["expected_agent_exit_code"],
            expected_agent_termination=stored_value["expected_agent_termination"],
            expected_failure_code=stored_value["expected_failure_code"],
            sha256=stored_value["sha256"],
        )
        stored_first = stored_session_reconciliation_v2.observe(
            dsn,
            intent=stored_intent,
            evaluation_directory=evaluation_directory,
            client=client,
        )
        stored_second = stored_session_reconciliation_v2.observe(
            dsn,
            intent=stored_intent,
            evaluation_directory=evaluation_directory,
            client=client,
        )
        if stored_first != stored_second:
            raise rollout_ledger.LedgerError("stored-session intent evidence changed")
        recovery_value = _recovery_value(
            plan_packet,
            evaluation_plan,
            recovery_rows,
            prior_intent_sha256=stored_intent.sha256,
            client=client,
        )
        recovery_intent = reviewed_recovery_v2.ProvisioningTimeoutIntent(
            evaluation_plan_sha256=recovery_value["evaluation_plan_sha256"],
            runtime_files_sha256=recovery_value["runtime_files_sha256"],
            serving_block=recovery_value["serving_block"],
            source_output_root=recovery_value["source_output_root"],
            source_database=recovery_value["source_database"],
            source_job_uid=recovery_value["source_job_uid"],
            source_job_terminal_receipt_sha256=recovery_value["source_job_terminal_receipt_sha256"],
            prior_stored_session_intent_sha256=recovery_value["prior_stored_session_intent_sha256"],
            selected_cells=tuple(
                reviewed_recovery_v2.SelectedCell(**row) for row in recovery_value["selected_cells"]
            ),
            sha256=recovery_value["sha256"],
        )
        recovery_first = reviewed_recovery_v2.observe_eligibility(
            dsn,
            intent=recovery_intent,
            plan=evaluation_plan,
            evaluation_directory=evaluation_directory,
            client=client,
        )
        recovery_second = reviewed_recovery_v2.observe_eligibility(
            dsn,
            intent=recovery_intent,
            plan=evaluation_plan,
            evaluation_directory=evaluation_directory,
            client=client,
        )
        if recovery_first != recovery_second:
            raise rollout_ledger.LedgerError("rollout repair intent evidence changed")
    output_root.mkdir(parents=True, mode=0o700)
    stored_path = output_root / "stored-session-intent.json"
    recovery_path = output_root / "rollout-recovery-intent.json"
    self_hosted.write_json_once(stored_path, stored_value)
    self_hosted.write_json_once(recovery_path, recovery_value)
    receipt = {
        "schema_version": "qwen38-fleet-seed44-base-private-repair-intents-v1",
        "repair_plan_sha256": plan_packet["sha256"],
        "stored_session_intent_sha256": "sha256:" + stored_intent.sha256,
        "stored_session_cell_count": 5,
        "rollout_recovery_intent_sha256": "sha256:" + recovery_intent.sha256,
        "rollout_recovery_cell_count": 2,
        "two_identical_metadata_observations_each": True,
        "ledger_mutations": 0,
        "model_calls": 0,
        "scoring_calls": 0,
        "score_values_included": False,
        "private_cell_task_session_or_trace_identifiers_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
    }
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    self_hosted.write_json_once(output_root / "PREPARED.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--postgres-admin-dsn-env", required=True)
    parser.add_argument("--postgres-database", required=True)
    args = parser.parse_args()
    admin_dsn = os.environ.get(args.postgres_admin_dsn_env, "")
    if not admin_dsn:
        parser.error(f"{args.postgres_admin_dsn_env} is required")
    try:
        result = prepare(
            repo_root=args.repo_root,
            plan_path=args.plan,
            output_root=args.output_root,
            admin_dsn=admin_dsn,
            database=args.postgres_database,
        )
    except BaseException as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema_version": "qwen38-fleet-seed44-base-private-repair-intents-v1",
                    "status": "failed",
                    "controller_failure_code": type(exc).__name__.lower(),
                    "ledger_mutations": 0,
                    "model_calls": 0,
                    "scoring_calls": 0,
                    "private_content_included": False,
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
