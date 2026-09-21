"""Accept exact stored Fleet sessions without another model or scoring call.

The private intent names a complete, reviewed cell roster.  The worker proves
the frozen plan, immutable local artifacts, and one exact completed Fleet
session for every selected cell.  It then accepts the complete roster in one
database transaction.  No model endpoint is consulted and no score value is
written to public evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import (
    evaluate,
    rollout_ledger,
    rollout_postgres,
    rollout_worker,
)
from evals.fleet import (
    exact_pass4_crypto as crypto,
)
from evals.fleet import (
    opencode_self_hosted as self_hosted,
)

INTENT_SCHEMA = "fleet-stored-session-reconciliation-intent-v2"
RECEIPT_SCHEMA = "fleet-stored-session-reconciliation-v1"
TERMINAL_OWNER_FAILURE_CODE = "stored_session.owner_terminal"
RUNTIME_FILES = (
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
    "sha256",
}
LOCAL_RESULT_FIELDS = (
    "execution_id",
    "execution_generation",
    "run_id",
    "session_id",
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
)
REFERENCE_IDENTITY_FIELDS = {
    "session_id",
    "job_id",
    "eval_task_id",
    "eval_task_version_id",
    "task_key",
    "status",
    "primary_reference_trace_id",
    "reference_trace_count",
    "reference_traces",
}
CELL_OBSERVATION_FIELDS = {
    "cell_id",
    "session_id",
    "local_record_sha256",
    "receipt",
}
CELL_RECEIPT_FIELDS = {
    "schema_version",
    "identity_binding_sha256",
    "config_sha256",
    "local_record_sha256",
    "artifact_binding_sha256",
    "authoritative_session_metadata_sha256",
    "authoritative_pinned_task_version_metadata_only",
    "reference_trace_content_returned",
    "authoritative_score_finite_and_equal_to_private_local_result",
    "model_generation_performed",
    "scoring_call_performed",
    "score_values_included",
    "prompt_response_flag_reward_or_trace_content_included",
    "cell_task_session_or_trace_identifiers_included",
    "receipt_sha256",
}


def _body_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_FILES}


@dataclass(frozen=True)
class StoredSessionIntent:
    evaluation_plan_sha256: str
    runtime_files_sha256: dict[str, str]
    serving_block: str
    source_output_root: str
    source_database: str
    source_job_uid: str
    source_job_terminal_receipt_sha256: str
    selected_cell_ids: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        plan = rollout_ledger._require_digest(  # noqa: SLF001
            self.evaluation_plan_sha256, "evaluation plan sha256"
        )
        runtime = runtime_identity()
        if self.runtime_files_sha256 != runtime:
            raise rollout_ledger.LedgerError("stored-session runtime identity differs")
        route = rollout_ledger._require_text(self.serving_block, "serving block")  # noqa: SLF001
        output = Path(self.source_output_root)
        if (
            not output.is_absolute()
            or output.parts[:4] != ("/", "mnt", "sfs", "jobs")
            or ".." in output.parts
        ):
            raise rollout_ledger.LedgerError("source output root is not an exact SFS job path")
        database = _database_name(self.source_database)
        source_job_uid = str(uuid.UUID(self.source_job_uid))
        terminal_receipt = rollout_ledger._require_digest(  # noqa: SLF001
            self.source_job_terminal_receipt_sha256,
            "source job terminal receipt sha256",
        )
        cells = tuple(str(uuid.UUID(value)) for value in self.selected_cell_ids)
        if not cells or len(cells) != len(set(cells)):
            raise rollout_ledger.LedgerError("stored-session roster is empty or repeated")
        expected = _body_digest(
            {
                "schema_version": INTENT_SCHEMA,
                "evaluation_plan_sha256": plan,
                "runtime_files_sha256": runtime,
                "serving_block": route,
                "source_output_root": str(output),
                "source_database": database,
                "source_job_uid": source_job_uid,
                "source_job_terminal_receipt_sha256": terminal_receipt,
                "selected_cell_ids": list(cells),
            }
        )
        value = rollout_ledger._require_digest(self.sha256, "intent sha256")  # noqa: SLF001
        if value != expected:
            raise rollout_ledger.LedgerError("stored-session intent self digest differs")
        object.__setattr__(self, "evaluation_plan_sha256", plan)
        object.__setattr__(self, "runtime_files_sha256", runtime)
        object.__setattr__(self, "serving_block", route)
        object.__setattr__(self, "source_output_root", str(output))
        object.__setattr__(self, "source_database", database)
        object.__setattr__(self, "source_job_uid", source_job_uid)
        object.__setattr__(self, "source_job_terminal_receipt_sha256", terminal_receipt)
        object.__setattr__(self, "selected_cell_ids", cells)
        object.__setattr__(self, "sha256", value)


def load_intent(path: Path) -> StoredSessionIntent:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            raise rollout_ledger.LedgerError(
                "stored-session intent must not be readable by group or other"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except rollout_ledger.LedgerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError("stored-session intent is unreadable") from exc
    if not isinstance(value, dict) or set(value) != INTENT_FIELDS:
        raise rollout_ledger.LedgerError("stored-session intent has missing or unknown fields")
    if value["schema_version"] != INTENT_SCHEMA or not isinstance(value["selected_cell_ids"], list):
        raise rollout_ledger.LedgerError("stored-session intent schema is unsupported")
    return StoredSessionIntent(
        evaluation_plan_sha256=value["evaluation_plan_sha256"],
        runtime_files_sha256=value["runtime_files_sha256"],
        serving_block=value["serving_block"],
        source_output_root=value["source_output_root"],
        source_database=value["source_database"],
        source_job_uid=value["source_job_uid"],
        source_job_terminal_receipt_sha256=value["source_job_terminal_receipt_sha256"],
        selected_cell_ids=tuple(value["selected_cell_ids"]),
        sha256=value["sha256"],
    )


def _database_name(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,62}", value) is None:
        raise rollout_ledger.LedgerError("stored-session database name is invalid")
    return value


def dedicated_dsn(admin_dsn: str, database: str) -> str:
    """Select the exact sealed evaluation database from the cluster administrator DSN."""

    database = _database_name(database)
    try:
        parsed = urlsplit(admin_dsn)
    except ValueError as exc:
        raise rollout_ledger.LedgerError("PostgreSQL administrator DSN is invalid") from exc
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or parsed.fragment:
        raise rollout_ledger.LedgerError("PostgreSQL administrator DSN is invalid")
    return urlunsplit((parsed.scheme, parsed.netloc, "/" + database, parsed.query, ""))


def _rows(connection: Any, intent: StoredSessionIntent, *, lock: bool) -> list[dict[str, Any]]:
    suffix = " FOR UPDATE OF c" if lock else ""
    rows = connection.execute(
        f"""
        SELECT c.*, r.execution_id, r.execution_generation, r.run_id,
               r.session_id AS local_session_id,
               r.verifier_execution_id, r.score, r.config_sha256,
               r.artifact_directory, r.trace_path, r.trace_sha256,
               r.result_path, r.result_sha256, r.reward_path, r.reward_sha256,
               r.session_ingest_path, r.session_ingest_sha256,
               r.cleanup_path, r.cleanup_sha256, r.session_ingest_status,
               r.agent_exit_code, r.agent_termination, r.elapsed_seconds,
               r.record_sha256
        FROM rollout_cells c
        LEFT JOIN rollout_local_results r ON r.cell_id = c.cell_id
        WHERE c.cell_id = ANY(%s::text[])
        ORDER BY c.cell_id{suffix}
        """,  # noqa: S608
        (list(intent.selected_cell_ids),),
    ).fetchall()
    if len(rows) != len(intent.selected_cell_ids) or {row["cell_id"] for row in rows} != set(
        intent.selected_cell_ids
    ):
        raise rollout_ledger.LedgerError("database differs from the stored-session roster")
    if any(row["serving_block"] != intent.serving_block for row in rows):
        raise rollout_ledger.LedgerError("database route differs from stored-session intent")
    if any(row["local_session_id"] is None for row in rows):
        raise rollout_ledger.LedgerError("stored-session roster lacks one local result per cell")
    return rows


def _normalized_local_result(row: dict[str, Any]) -> dict[str, Any]:
    record = {
        field: row["local_session_id"] if field == "session_id" else row[field]
        for field in LOCAL_RESULT_FIELDS
    }
    return rollout_ledger.normalize_local_result(record, cell_id=row["cell_id"])


def _validate_record_digest(row: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalized_local_result(row)
    observed = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed != row["record_sha256"]:
        raise rollout_ledger.LedgerError("stored local result record digest differs")
    return normalized


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _artifact_file(root: Path, relative: str, expected_sha256: str) -> Path:
    path = root / relative
    if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
        raise rollout_ledger.LedgerError("stored-session artifact path is not a regular child file")
    if _file_sha256(path) != rollout_ledger._require_digest(  # noqa: SLF001
        expected_sha256, "artifact sha256"
    ):
        raise rollout_ledger.LedgerError("stored-session artifact digest differs")
    return path


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError("stored-session JSON artifact is unreadable") from exc
    if not isinstance(value, dict):
        raise rollout_ledger.LedgerError("stored-session JSON artifact is not an object")
    return value


def _finite_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise rollout_ledger.LedgerError("stored-session score is not numeric")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise rollout_ledger.LedgerError("stored-session score is outside the valid range")
    return score


def _validate_artifacts(row: dict[str, Any], intent: StoredSessionIntent) -> str:
    normalized = _validate_record_digest(row)
    root = Path(intent.source_output_root).resolve()
    attempt = (root / normalized["artifact_directory"]).resolve()
    if attempt.is_symlink() or not attempt.is_dir() or root not in attempt.parents:
        raise rollout_ledger.LedgerError("stored-session artifact directory escapes its source")
    trace = _artifact_file(attempt, normalized["trace_path"], normalized["trace_sha256"])
    result_path = _artifact_file(attempt, normalized["result_path"], normalized["result_sha256"])
    reward_path = _artifact_file(attempt, normalized["reward_path"], normalized["reward_sha256"])
    ingest_path = _artifact_file(
        attempt, normalized["session_ingest_path"], normalized["session_ingest_sha256"]
    )
    cleanup_path = _artifact_file(attempt, normalized["cleanup_path"], normalized["cleanup_sha256"])
    result, reward, ingest, cleanup = map(
        _json, (result_path, reward_path, ingest_path, cleanup_path)
    )
    local_score = _finite_score(normalized["score"])
    if any(
        (
            normalized["session_ingest_status"] != "completed",
            normalized["agent_exit_code"] != 1,
            normalized["agent_termination"] != "process_error",
            result.get("run_id") != normalized["run_id"],
            result.get("task_key") != row["task_key"],
            result.get("task_version_id") != row["task_version_id"],
            result.get("session_id") != normalized["session_id"],
            result.get("verifier_execution_id") != normalized["verifier_execution_id"],
            result.get("session_ingest_status") != "completed",
            result.get("agent_exit_code") != 1,
            result.get("agent_termination") != "process_error",
            _finite_score(result.get("score")) != local_score,
            reward.get("task_key") != row["task_key"],
            reward.get("task_version_id") != row["task_version_id"],
            reward.get("verifier_execution_id") != normalized["verifier_execution_id"],
            _finite_score(reward.get("reward")) != local_score,
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
        raise rollout_ledger.LedgerError("stored-session local lifecycle binding differs")
    return _body_digest(
        {
            "record_sha256": row["record_sha256"],
            "trace_sha256": _file_sha256(trace),
            "result_sha256": _file_sha256(result_path),
            "reward_sha256": _file_sha256(reward_path),
            "session_ingest_sha256": _file_sha256(ingest_path),
            "cleanup_sha256": _file_sha256(cleanup_path),
            "score_finite_and_equal_across_private_artifacts": True,
        }
    )


def _scientific_index(plan: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rollout_ledger._plan_rows(Path(plan["_plan_csv"])):  # noqa: SLF001
        execution_cell = "sha256:" + digest(row)
        result[(row["model_id"], row["task_version_id"], int(row["attempt"]))] = {
            **row,
            "initial_execution": {
                "cell_id": execution_cell,
                "execution_id": "sha256:" + digest({"cell": execution_cell, "generation": 1}),
                "execution_generation": 1,
            },
        }
    return result


def _session_receipt(
    client: httpx.Client,
    *,
    row: dict[str, Any],
    config: dict[str, Any],
    artifact_binding_sha256: str,
) -> dict[str, Any]:
    sessions = self_hosted._task_sessions(client, row["task_key"])  # noqa: SLF001
    matches = [
        session for session in sessions if session.get("session_id") == row["local_session_id"]
    ]
    if len(matches) != 1:
        raise rollout_ledger.LedgerError("stored-session inventory is missing or ambiguous")
    session = matches[0]
    version_identity = self_hosted._request(  # noqa: SLF001
        client,
        "GET",
        f"/v1/sessions/{row['local_session_id']}/reference-traces",
    )
    if (
        not isinstance(version_identity, dict)
        or set(version_identity) != REFERENCE_IDENTITY_FIELDS
        or version_identity.get("session_id") != row["local_session_id"]
        or version_identity.get("eval_task_id") != session.get("eval_task_id")
        or version_identity.get("eval_task_version_id") != row["task_version_id"]
        or version_identity.get("task_key") != row["task_key"]
        or version_identity.get("status") != "no_reference_traces"
        or version_identity.get("primary_reference_trace_id") is not None
        or version_identity.get("reference_trace_count") != 0
        or version_identity.get("reference_traces") != []
    ):
        raise rollout_ledger.LedgerError(
            "stored authoritative session version identity is unavailable or contains content"
        )
    verifier = session.get("verifier_execution")
    authoritative_score = (
        _finite_score(verifier.get("score")) if isinstance(verifier, dict) else None
    )
    local_score = _finite_score(row["score"])
    if any(
        (
            session.get("status") != "completed",
            session.get("task_key") != row["task_key"],
            session.get("model") != row["endpoint_model_id"],
            not isinstance(verifier, dict),
            verifier.get("id") != row["verifier_execution_id"]
            if isinstance(verifier, dict)
            else True,
            authoritative_score != local_score,
            rollout_ledger._require_digest(  # noqa: SLF001
                config["config_sha256"], "expected config sha256"
            )
            != row["config_sha256"],
        )
    ):
        raise rollout_ledger.LedgerError("stored authoritative session binding differs")
    identity = {
        "cell_id": row["cell_id"],
        "session_id": row["local_session_id"],
        "task_version_id": row["task_version_id"],
        "model": row["endpoint_model_id"],
        "verifier_execution_id": row["verifier_execution_id"],
    }
    body = {
        "schema_version": "fleet-stored-session-cell-observation-v1",
        "identity_binding_sha256": _body_digest(identity),
        "config_sha256": row["config_sha256"],
        "local_record_sha256": row["record_sha256"],
        "artifact_binding_sha256": artifact_binding_sha256,
        "authoritative_session_metadata_sha256": _body_digest(
            {
                "session_id": session["session_id"],
                "task_key": session["task_key"],
                "eval_task_id": session["eval_task_id"],
                "eval_task_version_id": version_identity["eval_task_version_id"],
                "model": session["model"],
                "status": session["status"],
                "verifier_execution_id": verifier["id"],
            }
        ),
        "authoritative_pinned_task_version_metadata_only": True,
        "reference_trace_content_returned": False,
        "authoritative_score_finite_and_equal_to_private_local_result": True,
        "model_generation_performed": False,
        "scoring_call_performed": False,
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return {
        "cell_id": row["cell_id"],
        "session_id": row["local_session_id"],
        "local_record_sha256": row["record_sha256"],
        "receipt": {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")},
    }


def observe(
    dsn: str,
    *,
    intent: StoredSessionIntent,
    evaluation_directory: Path,
    client: httpx.Client,
) -> list[dict[str, Any]]:
    plan, proof = evaluate.checked_preflight(evaluation_directory)
    observed_plan_sha256 = rollout_ledger._require_digest(  # noqa: SLF001
        plan["sha256"], "plan sha256"
    )
    if observed_plan_sha256 != intent.evaluation_plan_sha256:
        raise rollout_ledger.LedgerError("stored-session intent differs from evaluation plan")
    plan = {
        **plan,
        "task_bindings": proof["task_bindings"],
        "_plan_csv": str(evaluation_directory / "plan.csv"),
    }
    rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        rows = _rows(connection, intent, lock=False)
    scientific = _scientific_index(plan)
    selected = {row["task_version_id"]: row for row in plan["tasks"]}
    observations: list[dict[str, Any]] = []
    for row in rows:
        key = (row["model_id"], row["task_version_id"], int(row["attempt"]))
        if key not in scientific or row["task_version_id"] not in selected:
            raise rollout_ledger.LedgerError("stored-session cell is absent from frozen plan")
        config = rollout_worker.build_config(
            plan, row, scientific[key], selected[row["task_version_id"]], client
        )
        artifact_binding = _validate_artifacts(row, intent)
        observations.append(
            _session_receipt(
                client,
                row=row,
                config=config,
                artifact_binding_sha256=artifact_binding,
            )
        )
    return observations


def _validate_cell_observation(row: dict[str, Any], observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict) or set(observation) != CELL_OBSERVATION_FIELDS:
        raise rollout_ledger.LedgerError("stored-session cell observation is invalid")
    cell_receipt = observation.get("receipt")
    if not isinstance(cell_receipt, dict) or set(cell_receipt) != CELL_RECEIPT_FIELDS:
        raise rollout_ledger.LedgerError("stored-session cell receipt is invalid")
    expected_identity = _body_digest(
        {
            "cell_id": row["cell_id"],
            "session_id": row["local_session_id"],
            "task_version_id": row["task_version_id"],
            "model": row["endpoint_model_id"],
            "verifier_execution_id": row["verifier_execution_id"],
        }
    )
    digests = (
        "identity_binding_sha256",
        "config_sha256",
        "local_record_sha256",
        "artifact_binding_sha256",
        "authoritative_session_metadata_sha256",
    )
    try:
        normalized_digests = {
            field: rollout_ledger._require_digest(cell_receipt[field], field)  # noqa: SLF001
            for field in digests
        }
        rollout_ledger._require_digest(  # noqa: SLF001
            cell_receipt["receipt_sha256"], "receipt_sha256"
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise rollout_ledger.LedgerError("stored-session cell receipt is invalid") from exc
    if any(
        (
            observation["cell_id"] != row["cell_id"],
            observation["session_id"] != row["local_session_id"],
            observation["local_record_sha256"] != row["record_sha256"],
            normalized_digests["identity_binding_sha256"] != expected_identity,
            normalized_digests["config_sha256"] != row["config_sha256"],
            normalized_digests["local_record_sha256"] != row["record_sha256"],
            cell_receipt.get("schema_version") != "fleet-stored-session-cell-observation-v1",
            cell_receipt["receipt_sha256"] != crypto.digest_without(cell_receipt, "receipt_sha256"),
            cell_receipt.get("authoritative_pinned_task_version_metadata_only") is not True,
            cell_receipt.get("reference_trace_content_returned") is not False,
            cell_receipt.get("authoritative_score_finite_and_equal_to_private_local_result")
            is not True,
            cell_receipt.get("model_generation_performed") is not False,
            cell_receipt.get("scoring_call_performed") is not False,
            cell_receipt.get("score_values_included") is not False,
            cell_receipt.get("prompt_response_flag_reward_or_trace_content_included") is not False,
            cell_receipt.get("cell_task_session_or_trace_identifiers_included") is not False,
        )
    ):
        raise rollout_ledger.LedgerError("stored-session cell receipt is invalid")
    return cell_receipt


def accept_roster(
    dsn: str,
    *,
    intent: StoredSessionIntent,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {row["cell_id"]: row for row in observations}
    if set(indexed) != set(intent.selected_cell_ids) or len(indexed) != len(observations):
        raise rollout_ledger.LedgerError("stored-session observations differ from intent roster")
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        rows = _rows(connection, intent, lock=True)
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
                raise rollout_ledger.LedgerError("stored-session terminal receipt is ambiguous")
            return matches[0]
        retry_review_count = 0
        stale_active_count = 0
        for row in rows:
            observation = indexed[row["cell_id"]]
            _validate_record_digest(row)
            _validate_cell_observation(row, observation)
            stale_active = (
                row["state"] in rollout_ledger.ACTIVE_STATES
                and row["lease_expires_at"] is not None
                and row["lease_expires_at"]
                < connection.execute("SELECT CURRENT_TIMESTAMP AS now").fetchone()["now"]
            )
            if row["state"] == "retry_review":
                retry_review_count += 1
            elif stale_active:
                stale_active_count += 1
            else:
                raise rollout_ledger.LedgerError("stored-session row is not review-held or stale")
            if any(
                (
                    row["reconciliation_digest"] not in (None, intent.sha256),
                    row["record_sha256"] != observation["local_record_sha256"],
                    row["local_session_id"] != observation["session_id"],
                    row["session_id"] not in (None, observation["session_id"]),
                    row["session_ingest_status"] != "completed",
                    row["agent_exit_code"] != 1,
                    row["agent_termination"] != "process_error",
                )
            ):
                raise rollout_ledger.LedgerError("stored-session database evidence drifted")
        body = {
            "schema_version": RECEIPT_SCHEMA,
            "reviewed_intent_sha256": intent.sha256,
            "evaluation_plan_sha256": intent.evaluation_plan_sha256,
            "source_job_uid_sha256": "sha256:"
            + hashlib.sha256(intent.source_job_uid.encode()).hexdigest(),
            "source_job_terminal_receipt_sha256": intent.source_job_terminal_receipt_sha256,
            "selected_cell_count": len(rows),
            "prior_retry_review_count": retry_review_count,
            "prior_stale_active_count": stale_active_count,
            "accepted_existing_completed_session_count": len(rows),
            "model_generation_performed": False,
            "scoring_call_performed": False,
            "score_values_included": False,
            "prompt_response_flag_reward_or_trace_content_included": False,
            "cell_task_session_or_trace_identifiers_included": False,
        }
        receipt = {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}
        for row in rows:
            observation = indexed[row["cell_id"]]
            cell_receipt = _validate_cell_observation(row, observation)
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
                raise rollout_ledger.LedgerError("stored-session atomic acceptance lost")
            rollout_postgres._event(  # noqa: SLF001
                connection,
                cell_id=row["cell_id"],
                name="stored_session_reconciled",
                from_state=row["state"],
                to_state="accepted",
                worker_id=row["worker_id"],
                claim_id=row["claim_id"],
                detail={
                    "reviewed_intent_sha256": intent.sha256,
                    "cell_receipt_sha256": cell_receipt["receipt_sha256"],
                    "source_job_terminal_receipt_sha256": intent.source_job_terminal_receipt_sha256,
                    "failure_code": (
                        TERMINAL_OWNER_FAILURE_CODE
                        if row["state"] in rollout_ledger.ACTIVE_STATES
                        else row["failure_code"]
                    ),
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
        raise FileExistsError("create-once stored-session output already exists")
    intent = load_intent(intent_path)
    database = _database_name(database)
    if database != intent.source_database:
        raise rollout_ledger.LedgerError("runtime database differs from stored-session intent")
    dsn = dedicated_dsn(admin_dsn, database)
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
            dsn,
            intent=intent,
            evaluation_directory=evaluation_directory,
            client=client,
        )
        second = observe(
            dsn,
            intent=intent,
            evaluation_directory=evaluation_directory,
            client=client,
        )
    if first != second:
        raise rollout_ledger.LedgerError("stored-session evidence changed during observation")
    receipt = accept_roster(dsn, intent=intent, observations=second)
    terminal = {
        "schema_version": "fleet-stored-session-reconciliation-terminal-v1",
        "status": "accepted",
        "reviewed_intent_sha256": intent.sha256,
        "reconciliation_receipt_sha256": receipt["receipt_sha256"],
        "selected_cell_count": len(intent.selected_cell_ids),
        "accepted_existing_completed_session_count": len(intent.selected_cell_ids),
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
            "schema_version": "fleet-stored-session-reconciliation-terminal-v1",
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
