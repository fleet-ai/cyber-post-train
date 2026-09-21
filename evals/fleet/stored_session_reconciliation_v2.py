"""Reconcile an exact scored-session roster with an intent-bound terminal outcome.

The historical reconciler is deliberately byte-pinned by its completed LR30
packet.  This successor keeps that source immutable and adds one missing
invariant: the private intent binds the exact local agent termination, exit
code, and held failure code.  That lets an already-authoritatively-scored
``output_limit`` rollout be accepted without either regenerating or rescoring
it, while preventing the new path from broadening the historical policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import rollout_ledger, rollout_postgres, rollout_worker
from evals.fleet import stored_session_reconciliation as legacy

INTENT_SCHEMA = "fleet-stored-session-reconciliation-intent-v3"
RECEIPT_SCHEMA = "fleet-stored-session-reconciliation-v2"
RUNTIME_FILES = (
    "stored_session_reconciliation_v2.py",
    "stored_session_reconciliation.py",
    "retry_review_policy.py",
)
INTENT_FIELDS = {
    "schema_version",
    "evaluation_plan_sha256",
    "runtime_files_sha256",
    "serving_block",
    "source_output_root",
    "source_database",
    "source_job_uid",
    "source_job_terminal_receipt_sha256",
    "selected_cell_ids",
    "expected_agent_exit_code",
    "expected_agent_termination",
    "expected_failure_code",
    "sha256",
}
SUPPORTED_AGENT_OUTCOMES = {(0, "output_limit")}


def _body_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_FILES}


@dataclass(frozen=True)
class ExactStoredSessionIntent:
    evaluation_plan_sha256: str
    runtime_files_sha256: dict[str, str]
    serving_block: str
    source_output_root: str
    source_database: str
    source_job_uid: str
    source_job_terminal_receipt_sha256: str
    selected_cell_ids: tuple[str, ...]
    expected_agent_exit_code: int
    expected_agent_termination: str
    expected_failure_code: str
    sha256: str

    def __post_init__(self) -> None:
        plan = rollout_ledger._require_digest(  # noqa: SLF001
            self.evaluation_plan_sha256, "evaluation plan sha256"
        )
        runtime = runtime_identity()
        if self.runtime_files_sha256 != runtime:
            raise rollout_ledger.LedgerError("stored-session v2 runtime identity differs")
        route = rollout_ledger._require_text(self.serving_block, "serving block")  # noqa: SLF001
        output = Path(self.source_output_root)
        if (
            not output.is_absolute()
            or output.parts[:4] != ("/", "mnt", "sfs", "jobs")
            or ".." in output.parts
        ):
            raise rollout_ledger.LedgerError("source output root is not an exact SFS job path")
        database = legacy._database_name(self.source_database)  # noqa: SLF001
        try:
            source_job_uid = str(uuid.UUID(self.source_job_uid))
            cells = tuple(str(uuid.UUID(value)) for value in self.selected_cell_ids)
        except (AttributeError, TypeError, ValueError) as exc:
            raise rollout_ledger.LedgerError("stored-session v2 identity is malformed") from exc
        terminal = rollout_ledger._require_digest(  # noqa: SLF001
            self.source_job_terminal_receipt_sha256,
            "source job terminal receipt sha256",
        )
        if not cells or len(cells) != len(set(cells)):
            raise rollout_ledger.LedgerError("stored-session v2 roster is empty or repeated")
        if (
            isinstance(self.expected_agent_exit_code, bool)
            or (self.expected_agent_exit_code, self.expected_agent_termination)
            not in SUPPORTED_AGENT_OUTCOMES
            or not isinstance(self.expected_failure_code, str)
            or not self.expected_failure_code
            or len(self.expected_failure_code) > 128
        ):
            raise rollout_ledger.LedgerError("stored-session v2 outcome is unsupported")
        body = {
            "schema_version": INTENT_SCHEMA,
            "evaluation_plan_sha256": plan,
            "runtime_files_sha256": runtime,
            "serving_block": route,
            "source_output_root": str(output),
            "source_database": database,
            "source_job_uid": source_job_uid,
            "source_job_terminal_receipt_sha256": terminal,
            "selected_cell_ids": list(cells),
            "expected_agent_exit_code": self.expected_agent_exit_code,
            "expected_agent_termination": self.expected_agent_termination,
            "expected_failure_code": self.expected_failure_code,
        }
        intent_sha256 = rollout_ledger._require_digest(self.sha256, "intent sha256")  # noqa: SLF001
        if intent_sha256 != _body_digest(body):
            raise rollout_ledger.LedgerError("stored-session v2 intent self digest differs")
        object.__setattr__(self, "evaluation_plan_sha256", plan)
        object.__setattr__(self, "runtime_files_sha256", runtime)
        object.__setattr__(self, "serving_block", route)
        object.__setattr__(self, "source_output_root", str(output))
        object.__setattr__(self, "source_database", database)
        object.__setattr__(self, "source_job_uid", source_job_uid)
        object.__setattr__(self, "source_job_terminal_receipt_sha256", terminal)
        object.__setattr__(self, "selected_cell_ids", cells)
        object.__setattr__(self, "sha256", intent_sha256)


def load_intent(path: Path) -> ExactStoredSessionIntent:
    try:
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise rollout_ledger.LedgerError(
                "stored-session v2 intent must not be readable by group or other"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except rollout_ledger.LedgerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError("stored-session v2 intent is unreadable") from exc
    if (
        not isinstance(value, dict)
        or set(value) != INTENT_FIELDS
        or value.get("schema_version") != INTENT_SCHEMA
        or not isinstance(value.get("selected_cell_ids"), list)
    ):
        raise rollout_ledger.LedgerError("stored-session v2 intent schema is unsupported")
    return ExactStoredSessionIntent(
        evaluation_plan_sha256=value["evaluation_plan_sha256"],
        runtime_files_sha256=value["runtime_files_sha256"],
        serving_block=value["serving_block"],
        source_output_root=value["source_output_root"],
        source_database=value["source_database"],
        source_job_uid=value["source_job_uid"],
        source_job_terminal_receipt_sha256=value["source_job_terminal_receipt_sha256"],
        selected_cell_ids=tuple(value["selected_cell_ids"]),
        expected_agent_exit_code=value["expected_agent_exit_code"],
        expected_agent_termination=value["expected_agent_termination"],
        expected_failure_code=value["expected_failure_code"],
        sha256=value["sha256"],
    )


def _validate_artifacts(row: dict[str, Any], intent: ExactStoredSessionIntent) -> str:
    normalized = legacy._validate_record_digest(row)  # noqa: SLF001
    root = Path(intent.source_output_root).resolve()
    attempt = (root / normalized["artifact_directory"]).resolve()
    if attempt.is_symlink() or not attempt.is_dir() or root not in attempt.parents:
        raise rollout_ledger.LedgerError("stored-session v2 artifact directory escapes its source")
    trace = legacy._artifact_file(  # noqa: SLF001
        attempt, normalized["trace_path"], normalized["trace_sha256"]
    )
    result_path = legacy._artifact_file(  # noqa: SLF001
        attempt, normalized["result_path"], normalized["result_sha256"]
    )
    reward_path = legacy._artifact_file(  # noqa: SLF001
        attempt, normalized["reward_path"], normalized["reward_sha256"]
    )
    ingest_path = legacy._artifact_file(  # noqa: SLF001
        attempt, normalized["session_ingest_path"], normalized["session_ingest_sha256"]
    )
    cleanup_path = legacy._artifact_file(  # noqa: SLF001
        attempt, normalized["cleanup_path"], normalized["cleanup_sha256"]
    )
    result, reward, ingest, cleanup = map(
        legacy._json,
        (result_path, reward_path, ingest_path, cleanup_path),  # noqa: SLF001
    )
    local_score = legacy._finite_score(normalized["score"])  # noqa: SLF001
    if any(
        (
            normalized["session_ingest_status"] != "completed",
            normalized["agent_exit_code"] != intent.expected_agent_exit_code,
            normalized["agent_termination"] != intent.expected_agent_termination,
            result.get("run_id") != normalized["run_id"],
            result.get("task_key") != row["task_key"],
            result.get("task_version_id") != row["task_version_id"],
            result.get("session_id") != normalized["session_id"],
            result.get("verifier_execution_id") != normalized["verifier_execution_id"],
            result.get("session_ingest_status") != "completed",
            result.get("agent_exit_code") != intent.expected_agent_exit_code,
            result.get("agent_termination") != intent.expected_agent_termination,
            legacy._finite_score(result.get("score")) != local_score,  # noqa: SLF001
            reward.get("task_key") != row["task_key"],
            reward.get("task_version_id") != row["task_version_id"],
            reward.get("verifier_execution_id") != normalized["verifier_execution_id"],
            legacy._finite_score(reward.get("reward")) != local_score,  # noqa: SLF001
            ingest.get("status") != "completed",
            ingest.get("session_id") != normalized["session_id"],
            cleanup
            != {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
        )
    ):
        raise rollout_ledger.LedgerError("stored-session v2 local lifecycle binding differs")
    return _body_digest(
        {
            "record_sha256": row["record_sha256"],
            "trace_sha256": legacy._file_sha256(trace),  # noqa: SLF001
            "result_sha256": legacy._file_sha256(result_path),  # noqa: SLF001
            "reward_sha256": legacy._file_sha256(reward_path),  # noqa: SLF001
            "session_ingest_sha256": legacy._file_sha256(ingest_path),  # noqa: SLF001
            "cleanup_sha256": legacy._file_sha256(cleanup_path),  # noqa: SLF001
            "score_finite_and_equal_across_private_artifacts": True,
            "agent_exit_code": intent.expected_agent_exit_code,
            "agent_termination": intent.expected_agent_termination,
        }
    )


def observe(
    dsn: str,
    *,
    intent: ExactStoredSessionIntent,
    evaluation_directory: Path,
    client: httpx.Client,
) -> list[dict[str, Any]]:
    from evals.fleet import evaluate

    plan, proof = evaluate.checked_preflight(evaluation_directory)
    observed_plan_sha256 = rollout_ledger._require_digest(  # noqa: SLF001
        plan["sha256"], "plan sha256"
    )
    if observed_plan_sha256 != intent.evaluation_plan_sha256:
        raise rollout_ledger.LedgerError("stored-session v2 intent differs from evaluation plan")
    plan = {
        **plan,
        "task_bindings": proof["task_bindings"],
        "_plan_csv": str(evaluation_directory / "plan.csv"),
    }
    rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        rows = legacy._rows(connection, intent, lock=False)  # noqa: SLF001
    scientific = legacy._scientific_index(plan)  # noqa: SLF001
    selected = {row["task_version_id"]: row for row in plan["tasks"]}
    observations = []
    for row in rows:
        key = (row["model_id"], row["task_version_id"], int(row["attempt"]))
        if key not in scientific or row["task_version_id"] not in selected:
            raise rollout_ledger.LedgerError("stored-session v2 cell is absent from frozen plan")
        config = rollout_worker.build_config(
            plan, row, scientific[key], selected[row["task_version_id"]], client
        )
        observations.append(
            legacy._session_receipt(  # noqa: SLF001
                client,
                row=row,
                config=config,
                artifact_binding_sha256=_validate_artifacts(row, intent),
            )
        )
    return observations


def accept_roster(
    dsn: str,
    *,
    intent: ExactStoredSessionIntent,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {row["cell_id"]: row for row in observations}
    if set(indexed) != set(intent.selected_cell_ids) or len(indexed) != len(observations):
        raise rollout_ledger.LedgerError("stored-session v2 observations differ from intent roster")
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        rows = legacy._rows(connection, intent, lock=True)  # noqa: SLF001
        if all(
            row["state"] == "accepted" and row["reconciliation_digest"] == intent.sha256
            for row in rows
        ):
            existing = connection.execute(
                "SELECT receipt_json FROM ledger_reconciliations WHERE kind = %s",
                (RECEIPT_SCHEMA,),
            ).fetchall()
            matches = [json.loads(row["receipt_json"]) for row in existing]
            matches = [row for row in matches if row.get("reviewed_intent_sha256") == intent.sha256]
            if len(matches) != 1:
                raise rollout_ledger.LedgerError("stored-session v2 terminal receipt is ambiguous")
            return matches[0]
        for row in rows:
            observation = indexed[row["cell_id"]]
            legacy._validate_record_digest(row)  # noqa: SLF001
            legacy._validate_cell_observation(row, observation)  # noqa: SLF001
            if any(
                (
                    row["state"] != "retry_review",
                    row["reconciliation_digest"] not in (None, intent.sha256),
                    row["failure_code"] != intent.expected_failure_code,
                    row["record_sha256"] != observation["local_record_sha256"],
                    row["local_session_id"] != observation["session_id"],
                    row["session_id"] not in (None, observation["session_id"]),
                    row["session_ingest_status"] != "completed",
                    row["agent_exit_code"] != intent.expected_agent_exit_code,
                    row["agent_termination"] != intent.expected_agent_termination,
                )
            ):
                raise rollout_ledger.LedgerError("stored-session v2 database evidence drifted")
        body = {
            "schema_version": RECEIPT_SCHEMA,
            "reviewed_intent_sha256": intent.sha256,
            "evaluation_plan_sha256": intent.evaluation_plan_sha256,
            "source_job_uid_sha256": "sha256:"
            + hashlib.sha256(intent.source_job_uid.encode()).hexdigest(),
            "source_job_terminal_receipt_sha256": intent.source_job_terminal_receipt_sha256,
            "selected_cell_count": len(rows),
            "prior_retry_review_count": len(rows),
            "accepted_existing_completed_session_count": len(rows),
            "source_agent_exit_code": intent.expected_agent_exit_code,
            "source_agent_termination": intent.expected_agent_termination,
            "source_failure_code_sha256": "sha256:"
            + hashlib.sha256(intent.expected_failure_code.encode()).hexdigest(),
            "action": "accept_existing_scored_session",
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
        }
        receipt = {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}
        for row in rows:
            observation = indexed[row["cell_id"]]
            cell_receipt = legacy._validate_cell_observation(row, observation)  # noqa: SLF001
            updated = connection.execute(
                """
                UPDATE rollout_cells
                SET state = 'accepted', session_id = %s,
                    completed_at = CURRENT_TIMESTAMP, heartbeat_at = CURRENT_TIMESTAMP,
                    lease_expires_at = NULL, result_class = 'valid',
                    receipt_digest = %s, failure_code = NULL,
                    reconciliation_digest = %s, updated_at = CURRENT_TIMESTAMP
                WHERE cell_id = %s
                """,
                (
                    observation["session_id"],
                    cell_receipt["receipt_sha256"],
                    intent.sha256,
                    row["cell_id"],
                ),
            )
            if updated.rowcount != 1:
                raise rollout_ledger.LedgerError("stored-session v2 atomic acceptance lost")
            rollout_postgres._event(  # noqa: SLF001
                connection,
                cell_id=row["cell_id"],
                name="stored_scored_session_reconciled",
                from_state="retry_review",
                to_state="accepted",
                worker_id=row["worker_id"],
                claim_id=row["claim_id"],
                detail={
                    "reviewed_intent_sha256": intent.sha256,
                    "cell_receipt_sha256": cell_receipt["receipt_sha256"],
                    "source_job_terminal_receipt_sha256": (
                        intent.source_job_terminal_receipt_sha256
                    ),
                    "action": "accept_existing_scored_session",
                },
            )
        connection.execute(
            """
            INSERT INTO ledger_reconciliations (
                receipt_sha256, kind, receipt_json, created_at
            ) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (
                receipt["receipt_sha256"],
                RECEIPT_SCHEMA,
                json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            ),
        )
    return receipt


def run(
    *,
    evaluation_directory: Path,
    output_root: Path,
    admin_dsn: str,
    database: str,
    intent_path: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError("create-once stored-session v2 output already exists")
    intent = load_intent(intent_path)
    database = legacy._database_name(database)  # noqa: SLF001
    if database != intent.source_database:
        raise rollout_ledger.LedgerError("runtime database differs from stored-session v2 intent")
    dsn = legacy.dedicated_dsn(admin_dsn, database)
    output_root.mkdir(parents=True, mode=0o700)
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
        first = observe(
            dsn, intent=intent, evaluation_directory=evaluation_directory, client=client
        )
        second = observe(
            dsn, intent=intent, evaluation_directory=evaluation_directory, client=client
        )
    if first != second:
        raise rollout_ledger.LedgerError("stored-session v2 evidence changed during observation")
    receipt = accept_roster(dsn, intent=intent, observations=second)
    terminal = {
        "schema_version": "fleet-stored-session-reconciliation-terminal-v2",
        "status": "accepted",
        "action": "accept_existing_scored_session",
        "reviewed_intent_sha256": intent.sha256,
        "reconciliation_receipt_sha256": receipt["receipt_sha256"],
        "selected_cell_count": len(intent.selected_cell_ids),
        "accepted_existing_completed_session_count": len(intent.selected_cell_ids),
        "source_agent_exit_code": intent.expected_agent_exit_code,
        "source_agent_termination": intent.expected_agent_termination,
        "model_generation_performed": False,
        "scoring_call_performed": False,
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    terminal["receipt_sha256"] = crypto.digest_without(terminal, "receipt_sha256")
    self_hosted.write_json_once(output_root / "TERMINAL.json", terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--postgres-admin-dsn-env", required=True)
    parser.add_argument("--postgres-database", required=True)
    parser.add_argument("--intent", type=Path, required=True)
    args = parser.parse_args()
    admin_dsn = os.environ.get(args.postgres_admin_dsn_env, "")
    if not admin_dsn:
        parser.error(f"{args.postgres_admin_dsn_env} is required")
    try:
        result = run(
            evaluation_directory=args.evaluation_directory,
            output_root=args.output_root,
            admin_dsn=admin_dsn,
            database=args.postgres_database,
            intent_path=args.intent,
        )
    except BaseException as exc:  # noqa: BLE001
        result = {
            "schema_version": "fleet-stored-session-reconciliation-terminal-v2",
            "status": "failed",
            "controller_failure_code": type(exc).__name__.lower(),
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
        }
        print(json.dumps(result, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
