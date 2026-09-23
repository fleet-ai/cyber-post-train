"""Finish one preserved Fleet session, then accept its existing ledger cell.

This path is intentionally narrower than a rollout worker.  It reconstructs the
frozen cell config, validates the immutable local trajectory, resumes only the
missing suffix of the already-bound Fleet session, and accepts that same cell
only after the normal Fleet session surface projects a completed verifier.
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
from evals.fleet import stored_session_reconciliation as stored

INTENT_SCHEMA = "fleet-heldout-partial-session-reconciliation-intent-v1"
RECEIPT_SCHEMA = "fleet-heldout-partial-session-reconciled-v1"
FAILURE_CODE = "authoritative_scoring_started.runtimeerror"
TERMINAL_OBSERVATION_SCHEMA = "cyber_fleet_heldout_terminal_observation_v1"
RUNTIME_FILES = (
    "evaluate.py",
    "exact_pass4_crypto.py",
    "heldout_partial_session_reconciliation.py",
    "opencode_self_hosted.py",
    "rollout_ledger.py",
    "rollout_postgres.py",
    "rollout_worker.py",
    "stored_session_reconciliation.py",
)
ROW_BINDING_FIELDS = (
    "cell_id",
    "experiment_id",
    "task_key",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
    "state",
    "failure_code",
    "reconciliation_digest",
    "execution_id",
    "execution_generation",
    "run_id",
    "local_session_id",
    "verifier_execution_id",
    "score",
    "config_sha256",
    "artifact_directory",
    "trace_path",
    "trace_sha256",
    "result_path",
    "result_sha256",
    "reward_path",
    "reward_sha256",
    "session_ingest_path",
    "session_ingest_sha256",
    "cleanup_path",
    "cleanup_sha256",
    "session_ingest_status",
    "agent_exit_code",
    "agent_termination",
    "elapsed_seconds",
    "record_sha256",
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
    "cell_id",
    "row_binding_sha256",
    "config_sha256",
    "recovery_output_root",
    "sha256",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sfs_job_path(value: str, label: str) -> str:
    path = Path(value)
    if (
        not path.is_absolute()
        or path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or ".." in path.parts
    ):
        raise rollout_ledger.LedgerError(f"{label} is not an exact SFS job path")
    return str(path)


def _row_binding(row: dict[str, Any]) -> str:
    if any(field not in row for field in ROW_BINDING_FIELDS):
        raise rollout_ledger.LedgerError("partial-session row lacks an identity field")
    return _digest({field: row[field] for field in ROW_BINDING_FIELDS})


def runtime_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_FILES}


@dataclass(frozen=True)
class Intent:
    evaluation_plan_sha256: str
    runtime_files_sha256: dict[str, str]
    serving_block: str
    source_output_root: str
    source_database: str
    source_job_uid: str
    source_job_terminal_receipt_sha256: str
    cell_id: str
    row_binding_sha256: str
    config_sha256: str
    recovery_output_root: str
    sha256: str

    def __post_init__(self) -> None:
        runtime = runtime_identity()
        if self.runtime_files_sha256 != runtime:
            raise rollout_ledger.LedgerError("partial-session runtime identity differs")
        values = {
            "evaluation_plan_sha256": rollout_ledger._require_digest(  # noqa: SLF001
                self.evaluation_plan_sha256, "evaluation plan digest"
            ),
            "runtime_files_sha256": runtime,
            "serving_block": rollout_ledger._require_text(  # noqa: SLF001
                self.serving_block, "serving block"
            ),
            "source_output_root": _sfs_job_path(self.source_output_root, "source output root"),
            "source_database": stored._database_name(self.source_database),  # noqa: SLF001
            "source_job_uid": str(uuid.UUID(self.source_job_uid)),
            "source_job_terminal_receipt_sha256": rollout_ledger._require_digest(  # noqa: SLF001
                self.source_job_terminal_receipt_sha256, "source terminal receipt digest"
            ),
            "cell_id": str(uuid.UUID(self.cell_id)),
            "row_binding_sha256": rollout_ledger._require_digest(  # noqa: SLF001
                self.row_binding_sha256, "row binding digest"
            ),
            "config_sha256": rollout_ledger._require_digest(  # noqa: SLF001
                self.config_sha256, "config digest"
            ),
            "recovery_output_root": _sfs_job_path(
                self.recovery_output_root, "recovery output root"
            ),
        }
        if values["source_output_root"] == values["recovery_output_root"]:
            raise rollout_ledger.LedgerError("recovery output aliases its immutable source")
        body = {"schema_version": INTENT_SCHEMA, **values}
        digest = rollout_ledger._require_digest(self.sha256, "intent digest")  # noqa: SLF001
        if digest != _digest(body):
            raise rollout_ledger.LedgerError("partial-session intent digest differs")
        for key, value in values.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "sha256", digest)

    @property
    def selected_cell_ids(self) -> tuple[str, ...]:
        return (self.cell_id,)


def load_intent(path: Path) -> Intent:
    try:
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise rollout_ledger.LedgerError("partial-session intent is not private")
        value = json.loads(path.read_text(encoding="utf-8"))
    except rollout_ledger.LedgerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError("partial-session intent is unreadable") from exc
    if not isinstance(value, dict) or set(value) != INTENT_FIELDS:
        raise rollout_ledger.LedgerError("partial-session intent schema is unsupported")
    if value.pop("schema_version") != INTENT_SCHEMA:
        raise rollout_ledger.LedgerError("partial-session intent schema is unsupported")
    return Intent(**value)


def build_intent_value(**values: Any) -> dict[str, Any]:
    """Build a reviewable private intent; callers must supply observed digests."""
    values.setdefault("runtime_files_sha256", runtime_identity())
    for field in (
        "evaluation_plan_sha256",
        "source_job_terminal_receipt_sha256",
        "row_binding_sha256",
        "config_sha256",
    ):
        values[field] = rollout_ledger._require_digest(values[field], field)  # noqa: SLF001
    values["serving_block"] = rollout_ledger._require_text(  # noqa: SLF001
        values["serving_block"], "serving block"
    )
    values["source_output_root"] = _sfs_job_path(values["source_output_root"], "source output root")
    values["source_database"] = stored._database_name(values["source_database"])  # noqa: SLF001
    values["source_job_uid"] = str(uuid.UUID(values["source_job_uid"]))
    values["cell_id"] = str(uuid.UUID(values["cell_id"]))
    values["recovery_output_root"] = _sfs_job_path(
        values["recovery_output_root"], "recovery output root"
    )
    body = {"schema_version": INTENT_SCHEMA, **values}
    return {**body, "sha256": _digest(body)}


def _validate_source_terminal(intent: Intent) -> None:
    path = Path(intent.source_output_root) / "TERMINAL_OBSERVATION.json"
    try:
        if path.is_symlink() or not path.is_file():
            raise rollout_ledger.LedgerError("source terminal observation is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except rollout_ledger.LedgerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError("source terminal observation is unreadable") from exc
    if not isinstance(value, dict) or value.get("schema") != TERMINAL_OBSERVATION_SCHEMA:
        raise rollout_ledger.LedgerError("source terminal observation schema is unsupported")
    observed = crypto.digest_without(value, "sha256").removeprefix("sha256:")
    database = value.get("database")
    summary = database.get("summary") if isinstance(database, dict) else None
    job = value.get("job")
    output = value.get("output_root")
    decision = value.get("decision")
    privacy = value.get("privacy")
    if any(
        (
            value.get("sha256", "").removeprefix("sha256:") != observed,
            observed != intent.source_job_terminal_receipt_sha256,
            not isinstance(job, dict) or job.get("uid") != intent.source_job_uid,
            not isinstance(output, dict)
            or output.get("exists") is not True
            or output.get("path") != intent.source_output_root,
            not isinstance(database, dict) or database.get("name") != intent.source_database,
            not isinstance(summary, dict)
            or str(summary.get("plan_sha256", "")).removeprefix("sha256:")
            != intent.evaluation_plan_sha256,
            not isinstance(decision, dict)
            or decision.get("rollout_retry_performed") is not False
            or decision.get("score_blind_reconciliation_required") is not True
            or decision.get("score_read_or_generated") is not False,
            privacy
            != {
                "credentials_included": False,
                "prompts_responses_flags_rewards_or_trace_content_included": False,
                "score_values_included": False,
            },
        )
    ):
        raise rollout_ledger.LedgerError("source terminal observation binding differs")


def _source_row(dsn: str, intent: Intent, *, lock: bool = False) -> dict[str, Any]:
    context = rollout_postgres._transaction if lock else rollout_postgres._read_transaction  # noqa: SLF001
    with context(dsn) as connection:
        return _source_row_in_connection(connection, intent, lock=lock)


def _source_row_in_connection(connection: Any, intent: Intent, *, lock: bool) -> dict[str, Any]:
    row = stored._rows(connection, intent, lock=lock)[0]  # noqa: SLF001
    if any(
        (
            row["state"] != "retry_review",
            row["failure_code"] != FAILURE_CODE,
            row["reconciliation_digest"] is not None,
            row["session_ingest_status"] != "failed",
            row["agent_exit_code"] != 0,
            row["agent_termination"] != "completed",
            row["serving_block"] != intent.serving_block,
            row["config_sha256"] != intent.config_sha256,
            _row_binding(row) != intent.row_binding_sha256,
        )
    ):
        raise rollout_ledger.LedgerError("partial-session database binding differs")
    stored._validate_record_digest(row)  # noqa: SLF001
    return row


def _frozen_config(
    dsn: str, intent: Intent, evaluation_directory: Path, client: httpx.Client
) -> tuple[dict[str, Any], dict[str, Any]]:
    from evals.fleet import evaluate

    plan, proof = evaluate.checked_preflight(evaluation_directory)
    local_plan = rollout_ledger._plan_digest(  # noqa: SLF001
        rollout_ledger._plan_rows(evaluation_directory / "plan.csv")  # noqa: SLF001
    )
    if local_plan != intent.evaluation_plan_sha256:
        raise rollout_ledger.LedgerError("partial-session intent differs from source plan")
    verified = rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    if (
        rollout_ledger._require_digest(  # noqa: SLF001
            verified["plan_sha256"], "database plan digest"
        )
        != local_plan
    ):
        raise rollout_ledger.LedgerError("partial-session database differs from source plan")
    plan = {
        **plan,
        "task_bindings": proof["task_bindings"],
        "_plan_csv": str(evaluation_directory / "plan.csv"),
    }
    row = _source_row(dsn, intent)
    scientific = stored._scientific_index(plan)  # noqa: SLF001
    selected = {item["task_version_id"]: item for item in plan["tasks"]}
    key = (row["model_id"], row["task_version_id"], int(row["attempt"]))
    if key not in scientific or row["task_version_id"] not in selected:
        raise rollout_ledger.LedgerError("partial-session cell is absent from frozen plan")
    config = rollout_worker.build_config(
        plan,
        row,
        scientific[key],
        selected[row["task_version_id"]],
        client,
    )
    if config["config_sha256"] != intent.config_sha256:
        raise rollout_ledger.LedgerError("partial-session reconstructed config differs")
    return row, config


def _validated_source(
    row: dict[str, Any], intent: Intent, config: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    root = Path(intent.source_output_root).resolve()
    attempt = (root / row["artifact_directory"]).resolve()
    if attempt.is_symlink() or not attempt.is_dir() or root not in attempt.parents:
        raise rollout_ledger.LedgerError("partial-session artifact directory escapes source")
    for path_field, digest_field in (
        ("trace_path", "trace_sha256"),
        ("result_path", "result_sha256"),
        ("reward_path", "reward_sha256"),
        ("session_ingest_path", "session_ingest_sha256"),
        ("cleanup_path", "cleanup_sha256"),
    ):
        stored._artifact_file(attempt, row[path_field], row[digest_field])  # noqa: SLF001
    _chunks, source = self_hosted._partial_recovery_source(config, attempt)  # noqa: SLF001
    if any(
        (
            source["session_id"] != row["local_session_id"],
            source["verifier_execution_id"] != row["verifier_execution_id"],
            source["score"] != stored._finite_score(row["score"]),  # noqa: SLF001
            source["trace_sha256"] != "sha256:" + row["trace_sha256"].removeprefix("sha256:"),
            source["original_ingest_sha256"]
            != "sha256:" + row["session_ingest_sha256"].removeprefix("sha256:"),
        )
    ):
        raise rollout_ledger.LedgerError("partial-session local artifact binding differs")
    return attempt, source


def _accept(
    dsn: str,
    intent: Intent,
    client: httpx.Client,
    config: dict[str, Any],
    recovery_binding_sha256: str,
) -> dict[str, Any]:
    row = _source_row(dsn, intent)
    first = stored._session_receipt(  # noqa: SLF001
        client, row=row, config=config, artifact_binding_sha256=recovery_binding_sha256
    )
    second = stored._session_receipt(  # noqa: SLF001
        client, row=row, config=config, artifact_binding_sha256=recovery_binding_sha256
    )
    if first != second:
        raise rollout_ledger.LedgerError("partial-session authority changed across observations")
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        locked = _source_row_in_connection(connection, intent, lock=True)
        cell_receipt = stored._validate_cell_observation(locked, first)  # noqa: SLF001
        body = {
            "schema_version": RECEIPT_SCHEMA,
            "reviewed_intent_sha256": intent.sha256,
            "evaluation_plan_sha256": intent.evaluation_plan_sha256,
            "source_job_uid_sha256": _digest(intent.source_job_uid),
            "source_job_terminal_receipt_sha256": intent.source_job_terminal_receipt_sha256,
            "row_binding_sha256": intent.row_binding_sha256,
            "config_sha256": intent.config_sha256,
            "recovery_binding_sha256": recovery_binding_sha256,
            "selected_cell_count": 1,
            "accepted_same_completed_session_count": 1,
            "missing_suffix_ingest_resumed": True,
            "model_generation_performed": False,
            "new_session_created": False,
            "fresh_scoring_request_performed": False,
            "score_values_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
        }
        receipt = {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}
        updated = connection.execute(
            """
            UPDATE rollout_cells
            SET state = 'accepted', session_id = %s,
                completed_at = CURRENT_TIMESTAMP, heartbeat_at = CURRENT_TIMESTAMP,
                lease_expires_at = NULL, result_class = 'valid',
                receipt_digest = %s, failure_code = NULL,
                reconciliation_digest = %s, updated_at = CURRENT_TIMESTAMP
            WHERE cell_id = %s AND state = 'retry_review'
            """,
            (first["session_id"], cell_receipt["receipt_sha256"], intent.sha256, intent.cell_id),
        )
        if updated.rowcount != 1:
            raise rollout_ledger.LedgerError("partial-session atomic acceptance lost")
        rollout_postgres._event(  # noqa: SLF001
            connection,
            cell_id=intent.cell_id,
            name="partial_session_ingest_reconciled",
            from_state="retry_review",
            to_state="accepted",
            worker_id=locked["worker_id"],
            claim_id=locked["claim_id"],
            detail={
                "reviewed_intent_sha256": intent.sha256,
                "cell_receipt_sha256": cell_receipt["receipt_sha256"],
                "recovery_binding_sha256": recovery_binding_sha256,
                "action": "resume_existing_session_suffix",
            },
        )
        connection.execute(
            """
            INSERT INTO ledger_reconciliations
                (receipt_sha256, kind, receipt_json, created_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (
                receipt["receipt_sha256"],
                RECEIPT_SCHEMA,
                json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            ),
        )
    return receipt


def run(
    dsn: str, *, intent: Intent, evaluation_directory: Path, client: httpx.Client
) -> dict[str, Any]:
    _validate_source_terminal(intent)
    row, config = _frozen_config(dsn, intent, evaluation_directory, client)
    attempt, source = _validated_source(row, intent, config)
    output = Path(intent.recovery_output_root)
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    observed = self_hosted.observe_partial_session_resume(
        client, config=config, source_dir=attempt, out_dir=output / "observer"
    )
    resumed = self_hosted.resume_partial_session_trace(
        client,
        config=config,
        source_dir=attempt,
        out_dir=output / "resume",
        observer_receipt=observed,
    )
    if any(
        (
            resumed.get("resumed") is not True,
            resumed.get("session_id") != source["session_id"],
            resumed.get("verifier_execution_id") != source["verifier_execution_id"],
            resumed.get("same_session_id_preserved") is not True,
            resumed.get("model_or_verifier_replayed") is not False,
        )
    ):
        raise rollout_ledger.LedgerError("partial-session resume receipt differs")
    recovery_binding = _digest(
        {
            "intent_sha256": intent.sha256,
            "observer_receipt_sha256": observed["receipt_sha256"],
            "resumed_receipt_sha256": resumed["receipt_sha256"],
            "source_record_sha256": row["record_sha256"],
            "source_original_ingest_sha256": source["original_ingest_sha256"],
            "same_session_preserved": True,
            "model_or_verifier_replayed": False,
        }
    )
    receipt = _accept(dsn, intent, client, config, recovery_binding)
    self_hosted.write_json_once(output / "TERMINAL.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--postgres-admin-dsn-env", default="ROLLOUT_DATABASE_URL")
    parser.add_argument("--postgres-database", required=True)
    args = parser.parse_args()
    intent = load_intent(args.intent)
    if args.postgres_database != intent.source_database:
        raise rollout_ledger.LedgerError("partial-session database argument differs")
    admin = os.environ.get(args.postgres_admin_dsn_env)
    key = os.environ.get("FLEET_API_KEY")
    if not admin or not key:
        raise rollout_ledger.LedgerError("required runtime credentials are unavailable")
    dsn = stored.dedicated_dsn(admin, intent.source_database)
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=1800,
    ) as client:
        receipt = run(
            dsn,
            intent=intent,
            evaluation_directory=args.evaluation_directory,
            client=client,
        )
    print(
        json.dumps(
            {
                "accepted_same_completed_session_count": receipt[
                    "accepted_same_completed_session_count"
                ],
                "model_generation_performed": False,
                "new_session_created": False,
                "fresh_scoring_request_performed": False,
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
