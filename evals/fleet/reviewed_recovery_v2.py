"""Evidence-bound recovery for a provisioning POST 504 before model execution.

``post_claim.fleetrequesterror`` is intentionally *not* a general retry class.
This module admits only a private, self-digesting roster whose original attempt
has an exact sanitized provisioning ``POST``/``504`` receipt, no local result,
no exact authoritative session, and no file that proves model execution or
scoring began.  The evidence is observed twice before the complete roster is
bound atomically, and every claim consumes the frozen campaign's sole retry.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import opencode_self_hosted as self_hosted
from evals.fleet import rollout_ledger, rollout_postgres, rollout_worker

INTENT_SCHEMA = "fleet-reviewed-provisioning-timeout-recovery-intent-v2"
APPLY_SCHEMA = "fleet-reviewed-provisioning-timeout-recovery-apply-v2"
PRECLAIM_SCHEMA = "fleet-reviewed-provisioning-timeout-recovery-preclaim-v2"
OBSERVATION_SCHEMA = "fleet-reviewed-provisioning-timeout-observation-v1"
SOURCE_FAILURE_CODE = "post_claim.fleetrequesterror"
RUNTIME_FILES = (
    "reviewed_recovery_v2.py",
    "reviewed_recovery_worker_v2.py",
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
    "prior_stored_session_intent_sha256",
    "selected_cells",
    "sha256",
}
SELECTED_CELL_FIELDS = {
    "cell_id",
    "claim_file_sha256",
    "binding_file_sha256",
    "prompt_file_sha256",
    "failure_file_sha256",
    "cleanup_file_sha256",
}
OBSERVATION_FIELDS = {
    "schema_version",
    "cell_id",
    "claim_file_sha256",
    "binding_file_sha256",
    "prompt_file_sha256",
    "failure_file_sha256",
    "cleanup_file_sha256",
    "source_file_set_exact",
    "local_result_absent",
    "authoritative_exact_execution_session_absent",
    "model_execution_artifacts_absent",
    "scoring_artifacts_absent",
    "provisioning_method",
    "provisioning_http_status",
    "receipt_sha256",
}
REQUIRED_ATTEMPT_FILES = {
    "binding.json",
    "prompt.txt",
    "failure.json",
    "cleanup.json",
}


def _body_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _prefixed_file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def runtime_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_FILES}


def _exact_sfs_root(value: Any) -> str:
    path = Path(value) if isinstance(value, str) else Path()
    if (
        not isinstance(value, str)
        or not path.is_absolute()
        or path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or ".." in path.parts
    ):
        raise rollout_ledger.LedgerError("recovery source is not an exact SFS job path")
    return str(path)


@dataclass(frozen=True)
class SelectedCell:
    cell_id: str
    claim_file_sha256: str
    binding_file_sha256: str
    prompt_file_sha256: str
    failure_file_sha256: str
    cleanup_file_sha256: str

    def __post_init__(self) -> None:
        try:
            cell_id = str(uuid.UUID(self.cell_id))
        except (AttributeError, TypeError, ValueError) as exc:
            raise rollout_ledger.LedgerError(
                "reviewed recovery cell identity is malformed"
            ) from exc
        for field in SELECTED_CELL_FIELDS - {"cell_id"}:
            value = rollout_ledger._require_digest(getattr(self, field), field)  # noqa: SLF001
            object.__setattr__(self, field, "sha256:" + value)
        object.__setattr__(self, "cell_id", cell_id)

    def as_dict(self) -> dict[str, Any]:
        return {field: getattr(self, field) for field in sorted(SELECTED_CELL_FIELDS)}


@dataclass(frozen=True)
class ProvisioningTimeoutIntent:
    evaluation_plan_sha256: str
    runtime_files_sha256: dict[str, str]
    serving_block: str
    source_output_root: str
    source_database: str
    source_job_uid: str
    source_job_terminal_receipt_sha256: str
    prior_stored_session_intent_sha256: str
    selected_cells: tuple[SelectedCell, ...]
    sha256: str

    def __post_init__(self) -> None:
        plan = rollout_ledger._require_digest(  # noqa: SLF001
            self.evaluation_plan_sha256, "evaluation plan sha256"
        )
        runtime = runtime_identity()
        if self.runtime_files_sha256 != runtime:
            raise rollout_ledger.LedgerError("reviewed recovery v2 runtime identity differs")
        route = rollout_ledger._require_text(self.serving_block, "serving block")  # noqa: SLF001
        source_output = _exact_sfs_root(self.source_output_root)
        source_database = _database_name(self.source_database)
        try:
            source_job_uid = str(uuid.UUID(self.source_job_uid))
        except (AttributeError, TypeError, ValueError) as exc:
            raise rollout_ledger.LedgerError("reviewed recovery source Job UID is invalid") from exc
        terminal = rollout_ledger._require_digest(  # noqa: SLF001
            self.source_job_terminal_receipt_sha256,
            "source job terminal receipt sha256",
        )
        prior = rollout_ledger._require_digest(  # noqa: SLF001
            self.prior_stored_session_intent_sha256,
            "prior stored-session intent sha256",
        )
        if not self.selected_cells:
            raise rollout_ledger.LedgerError("reviewed recovery v2 intent must select cells")
        cell_ids = [cell.cell_id for cell in self.selected_cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise rollout_ledger.LedgerError("reviewed recovery v2 intent repeats a cell")
        body = {
            "schema_version": INTENT_SCHEMA,
            "evaluation_plan_sha256": plan,
            "runtime_files_sha256": runtime,
            "serving_block": route,
            "source_output_root": source_output,
            "source_database": source_database,
            "source_job_uid": source_job_uid,
            "source_job_terminal_receipt_sha256": terminal,
            "prior_stored_session_intent_sha256": prior,
            "selected_cells": [cell.as_dict() for cell in self.selected_cells],
        }
        intent_sha256 = rollout_ledger._require_digest(self.sha256, "intent sha256")  # noqa: SLF001
        if intent_sha256 != _body_digest(body):
            raise rollout_ledger.LedgerError("reviewed recovery v2 intent self digest differs")
        object.__setattr__(self, "evaluation_plan_sha256", plan)
        object.__setattr__(self, "runtime_files_sha256", runtime)
        object.__setattr__(self, "serving_block", route)
        object.__setattr__(self, "source_output_root", source_output)
        object.__setattr__(self, "source_database", source_database)
        object.__setattr__(self, "source_job_uid", source_job_uid)
        object.__setattr__(self, "source_job_terminal_receipt_sha256", terminal)
        object.__setattr__(self, "prior_stored_session_intent_sha256", prior)
        object.__setattr__(self, "sha256", intent_sha256)

    @property
    def selected_cell_ids(self) -> tuple[str, ...]:
        return tuple(cell.cell_id for cell in self.selected_cells)

    @property
    def selected_index(self) -> dict[str, SelectedCell]:
        return {cell.cell_id: cell for cell in self.selected_cells}


def _database_name(value: Any) -> str:
    import re

    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,62}", value) is None:
        raise rollout_ledger.LedgerError("reviewed recovery database name is invalid")
    return value


def load_intent(path: Path) -> ProvisioningTimeoutIntent:
    try:
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise rollout_ledger.LedgerError(
                "reviewed recovery v2 intent must not be readable by group or other"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except rollout_ledger.LedgerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError("reviewed recovery v2 intent is unreadable") from exc
    if (
        not isinstance(value, dict)
        or set(value) != INTENT_FIELDS
        or value.get("schema_version") != INTENT_SCHEMA
        or not isinstance(value.get("selected_cells"), list)
        or any(
            not isinstance(row, dict) or set(row) != SELECTED_CELL_FIELDS
            for row in value["selected_cells"]
        )
    ):
        raise rollout_ledger.LedgerError("reviewed recovery v2 intent schema is unsupported")
    cells = tuple(SelectedCell(**row) for row in value["selected_cells"])
    return ProvisioningTimeoutIntent(
        evaluation_plan_sha256=value["evaluation_plan_sha256"],
        runtime_files_sha256=value["runtime_files_sha256"],
        serving_block=value["serving_block"],
        source_output_root=value["source_output_root"],
        source_database=value["source_database"],
        source_job_uid=value["source_job_uid"],
        source_job_terminal_receipt_sha256=value["source_job_terminal_receipt_sha256"],
        prior_stored_session_intent_sha256=value["prior_stored_session_intent_sha256"],
        selected_cells=cells,
        sha256=value["sha256"],
    )


def _rows(
    connection: Any, intent: ProvisioningTimeoutIntent, *, lock: bool
) -> list[dict[str, Any]]:
    suffix = " FOR UPDATE" if lock else ""
    rows = connection.execute(
        f"""
        SELECT * FROM rollout_cells
        WHERE cell_id = ANY(%s::text[])
        ORDER BY cell_id{suffix}
        """,  # noqa: S608
        (list(intent.selected_cell_ids),),
    ).fetchall()
    if len(rows) != len(intent.selected_cell_ids) or {row["cell_id"] for row in rows} != set(
        intent.selected_cell_ids
    ):
        raise rollout_ledger.LedgerError("database differs from reviewed recovery v2 roster")
    if any(row["serving_block"] != intent.serving_block for row in rows):
        raise rollout_ledger.LedgerError("database route differs from reviewed recovery v2 intent")
    return rows


def _json_file(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise rollout_ledger.LedgerError(f"reviewed recovery {label} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError(f"reviewed recovery {label} is unreadable") from exc
    if not isinstance(value, dict):
        raise rollout_ledger.LedgerError(f"reviewed recovery {label} is not an object")
    return value


def _exact_session_matches(
    sessions: list[dict[str, Any]], *, config: dict[str, Any]
) -> list[dict[str, Any]]:
    cell_id = config["execution"]["cell_id"]
    execution_id = config["execution"]["execution_id"]
    run_id = config["run_id"]
    matches = []
    for session in sessions:
        metadata = session.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if (
            metadata.get("cell_id") == cell_id
            or metadata.get("execution_id") == execution_id
            or metadata.get("run_id") == run_id
        ):
            matches.append(session)
    return matches


def _scientific_index(
    plan: dict[str, Any], plan_csv: Path
) -> dict[tuple[str, str, int], dict[str, Any]]:
    result = {}
    for row in rollout_ledger._plan_rows(plan_csv):  # noqa: SLF001
        cell_id = "sha256:" + digest(row)
        result[(row["model_id"], row["task_version_id"], int(row["attempt"]))] = {
            **row,
            "initial_execution": {
                "cell_id": cell_id,
                "execution_id": "sha256:" + digest({"cell": cell_id, "generation": 1}),
                "execution_generation": 1,
            },
        }
    return result


def _observe_cell(
    *,
    row: dict[str, Any],
    selected_cell: SelectedCell,
    config: dict[str, Any],
    source_root: Path,
    client: httpx.Client,
) -> dict[str, Any]:
    execution_name = config["execution"]["execution_id"].removeprefix("sha256:")
    claim_path = source_root / "claims" / f"{execution_name}.json"
    attempt = source_root / "attempts" / execution_name
    if attempt.is_symlink() or not attempt.is_dir() or source_root not in attempt.resolve().parents:
        raise rollout_ledger.LedgerError("reviewed recovery source attempt is invalid")
    observed_files = {
        path.name for path in attempt.iterdir() if path.is_file() and not path.is_symlink()
    }
    if observed_files != REQUIRED_ATTEMPT_FILES:
        raise rollout_ledger.LedgerError("reviewed recovery source file set differs")
    paths = {
        "claim_file_sha256": claim_path,
        "binding_file_sha256": attempt / "binding.json",
        "prompt_file_sha256": attempt / "prompt.txt",
        "failure_file_sha256": attempt / "failure.json",
        "cleanup_file_sha256": attempt / "cleanup.json",
    }
    for field, path in paths.items():
        if (
            path.is_symlink()
            or not path.is_file()
            or _prefixed_file_sha256(path) != getattr(selected_cell, field)
        ):
            raise rollout_ledger.LedgerError("reviewed recovery source digest differs")
    claim = _json_file(claim_path, "claim receipt")
    if claim != rollout_worker._claim_receipt(config, row):  # noqa: SLF001
        raise rollout_ledger.LedgerError("reviewed recovery claim receipt differs")
    binding = _json_file(attempt / "binding.json", "binding")
    expected_binding = {
        "schema_version": config["schema_version"],
        "run_id": config["run_id"],
        "source_job_id": config["source_job_id"],
        "task": config["task"],
        "environment": config["environment"],
        "verifier": config["verifier"],
        "authority": config["authority"],
        "model": config["model"],
        "harness": config["harness"],
    }
    if set(binding) != {*expected_binding, "authority_gate"} or any(
        binding.get(field) != value for field, value in expected_binding.items()
    ):
        raise rollout_ledger.LedgerError("reviewed recovery source binding differs")
    failure = _json_file(attempt / "failure.json", "failure receipt")
    expected_failure_fields = {
        "error_type",
        "elapsed_seconds",
        "run_id",
        "method",
        "route",
        "http_status",
        "response_sha256",
        "reason",
    }
    elapsed = failure.get("elapsed_seconds")
    if any(
        (
            set(failure) != expected_failure_fields,
            failure.get("error_type") != "FleetRequestError",
            isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)),
            not math.isfinite(float(elapsed)) if isinstance(elapsed, (int, float)) else True,
            failure.get("run_id") != config["run_id"],
            failure.get("method") != "POST",
            failure.get("route") != self_hosted.authoritative_route(config, "provisioning"),
            failure.get("http_status") != 504,
            not isinstance(failure.get("reason"), str),
            not isinstance(failure.get("response_sha256"), str),
        )
    ):
        raise rollout_ledger.LedgerError("reviewed recovery is not an exact provisioning 504")
    rollout_ledger._require_digest(failure["response_sha256"], "response sha256")  # noqa: SLF001
    cleanup = _json_file(attempt / "cleanup.json", "cleanup receipt")
    if cleanup != {
        "instance_created": False,
        "instance_closed": False,
        "containers_removed": True,
    }:
        raise rollout_ledger.LedgerError("reviewed recovery cleanup proves later execution")
    sessions = self_hosted._task_sessions(client, row["task_key"])  # noqa: SLF001
    if _exact_session_matches(sessions, config=config):
        raise rollout_ledger.LedgerError("reviewed recovery exact authoritative session exists")
    body = {
        "schema_version": OBSERVATION_SCHEMA,
        "cell_id": row["cell_id"],
        **{field: getattr(selected_cell, field) for field in paths},
        "source_file_set_exact": True,
        "local_result_absent": True,
        "authoritative_exact_execution_session_absent": True,
        "model_execution_artifacts_absent": True,
        "scoring_artifacts_absent": True,
        "provisioning_method": "POST",
        "provisioning_http_status": 504,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def observe_eligibility(
    dsn: str,
    *,
    intent: ProvisioningTimeoutIntent,
    plan: dict[str, Any],
    evaluation_directory: Path,
    client: httpx.Client,
) -> list[dict[str, Any]]:
    observed_plan_sha256 = rollout_ledger._require_digest(  # noqa: SLF001
        plan["sha256"], "plan sha256"
    )
    if observed_plan_sha256 != intent.evaluation_plan_sha256:
        raise rollout_ledger.LedgerError("reviewed recovery v2 intent differs from frozen plan")
    rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        rows = _rows(connection, intent, lock=False)
        local_results = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_local_results WHERE cell_id = ANY(%s::text[])",
            (list(intent.selected_cell_ids),),
        ).fetchone()["count"]
    if local_results:
        raise rollout_ledger.LedgerError("reviewed recovery v2 roster has a local result")
    scientific = _scientific_index(plan, evaluation_directory / "plan.csv")
    selected = {row["task_version_id"]: row for row in plan["tasks"]}
    source_root = Path(intent.source_output_root).resolve()
    observations = []
    for row in rows:
        if any(
            (
                row["state"] != "retry_review",
                row["failure_code"] != SOURCE_FAILURE_CODE,
                row["result_class"] != "infrastructure_invalid",
                row["session_id"] is not None,
                row["receipt_digest"] is not None,
                row["reconciliation_digest"] not in (None, intent.sha256),
                int(row["retry_count"]) != 0,
                int(row["max_retries"]) != 1,
            )
        ):
            raise rollout_ledger.LedgerError("reviewed recovery v2 database evidence drifted")
        key = (row["model_id"], row["task_version_id"], int(row["attempt"]))
        scientific_cell = scientific.get(key)
        task = selected.get(row["task_version_id"])
        if scientific_cell is None or task is None:
            raise rollout_ledger.LedgerError("reviewed recovery cell is absent from frozen plan")
        config = rollout_worker.build_config(plan, row, scientific_cell, task, client)
        observations.append(
            _observe_cell(
                row=row,
                selected_cell=intent.selected_index[row["cell_id"]],
                config=config,
                source_root=source_root,
                client=client,
            )
        )
    return observations


def _validate_observations(
    intent: ProvisioningTimeoutIntent, observations: list[dict[str, Any]]
) -> None:
    indexed = {row.get("cell_id"): row for row in observations if isinstance(row, dict)}
    if set(indexed) != set(intent.selected_cell_ids) or len(indexed) != len(observations):
        raise rollout_ledger.LedgerError("reviewed recovery v2 observations differ from roster")
    for cell_id, observation in indexed.items():
        selected = intent.selected_index[cell_id]
        if set(observation) != OBSERVATION_FIELDS or any(
            (
                observation.get("schema_version") != OBSERVATION_SCHEMA,
                observation.get("receipt_sha256")
                != crypto.digest_without(observation, "receipt_sha256"),
                any(
                    observation.get(field) != getattr(selected, field)
                    for field in SELECTED_CELL_FIELDS - {"cell_id"}
                ),
                any(
                    observation.get(field) is not True
                    for field in (
                        "source_file_set_exact",
                        "local_result_absent",
                        "authoritative_exact_execution_session_absent",
                        "model_execution_artifacts_absent",
                        "scoring_artifacts_absent",
                    )
                ),
                observation.get("provisioning_method") != "POST",
                observation.get("provisioning_http_status") != 504,
            )
        ):
            raise rollout_ledger.LedgerError("reviewed recovery v2 observation is invalid")


def _apply_evidence(intent: ProvisioningTimeoutIntent) -> dict[str, Any]:
    body = {
        "schema_version": APPLY_SCHEMA,
        "reviewed_intent_sha256": intent.sha256,
        "prior_stored_session_intent_sha256": intent.prior_stored_session_intent_sha256,
        "source_job_terminal_receipt_sha256": intent.source_job_terminal_receipt_sha256,
        "selected_cell_count": len(intent.selected_cell_ids),
        "eligibility_policy": "exact_provisioning_post_504_before_model_or_scoring_v1",
        "source_failure_code_sha256": "sha256:"
        + hashlib.sha256(SOURCE_FAILURE_CODE.encode()).hexdigest(),
        "rows_remain_quarantined_from_ordinary_claims": True,
        "local_result_count_at_apply": 0,
        "authoritative_exact_execution_session_count_at_apply": 0,
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def apply_intent(
    dsn: str,
    *,
    intent: ProvisioningTimeoutIntent,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_observations(intent, observations)
    evidence = _apply_evidence(intent)
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        rows = _rows(connection, intent, lock=True)
        digests = {row["reconciliation_digest"] for row in rows}
        if digests == {intent.sha256}:
            stored = connection.execute(
                "SELECT kind, receipt_json FROM ledger_reconciliations WHERE receipt_sha256 = %s",
                (evidence["receipt_sha256"],),
            ).fetchone()
            if (
                stored is None
                or stored["kind"] != APPLY_SCHEMA
                or json.loads(stored["receipt_json"]) != evidence
            ):
                raise rollout_ledger.LedgerError("reviewed recovery v2 apply receipt differs")
            return evidence
        if digests != {None}:
            raise rollout_ledger.LedgerError("reviewed recovery v2 reconciliation digest differs")
        prior_rows = connection.execute(
            """
            SELECT COUNT(*) AS count FROM rollout_cells
            WHERE reconciliation_digest = %s AND state = 'accepted'
            """,
            (intent.prior_stored_session_intent_sha256,),
        ).fetchone()["count"]
        prior_receipts = connection.execute(
            "SELECT receipt_json FROM ledger_reconciliations WHERE kind = %s",
            ("fleet-stored-session-reconciliation-v2",),
        ).fetchall()
        prior_matches = [
            json.loads(row["receipt_json"])
            for row in prior_receipts
            if json.loads(row["receipt_json"]).get("reviewed_intent_sha256")
            == intent.prior_stored_session_intent_sha256
        ]
        local_results = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_local_results WHERE cell_id = ANY(%s::text[])",
            (list(intent.selected_cell_ids),),
        ).fetchone()["count"]
        if (
            prior_rows != 5
            or len(prior_matches) != 1
            or local_results != 0
            or any(
                row["state"] != "retry_review"
                or int(row["retry_count"]) != 0
                or int(row["max_retries"]) != 1
                or row["result_class"] != "infrastructure_invalid"
                or row["failure_code"] != SOURCE_FAILURE_CODE
                or row["session_id"] is not None
                or row["receipt_digest"] is not None
                for row in rows
            )
        ):
            raise rollout_ledger.LedgerError("reviewed recovery v2 roster is ineligible")
        updated = connection.execute(
            """
            UPDATE rollout_cells
            SET reconciliation_digest = %s, updated_at = clock_timestamp()
            WHERE cell_id = ANY(%s::text[]) AND state = 'retry_review'
              AND reconciliation_digest IS NULL
            """,
            (intent.sha256, list(intent.selected_cell_ids)),
        )
        if updated.rowcount != len(rows):
            raise rollout_ledger.LedgerError("reviewed recovery v2 atomic apply lost")
        connection.execute(
            """
            INSERT INTO ledger_reconciliations (
                receipt_sha256, kind, receipt_json, created_at
            ) VALUES (%s, %s, %s, clock_timestamp())
            """,
            (
                evidence["receipt_sha256"],
                APPLY_SCHEMA,
                json.dumps(evidence, sort_keys=True, separators=(",", ":")),
            ),
        )
        for row in rows:
            rollout_postgres._event(  # noqa: SLF001
                connection,
                cell_id=row["cell_id"],
                name="reviewed_provisioning_timeout_recovery_bound",
                from_state="retry_review",
                to_state="retry_review",
                detail={
                    "reviewed_intent_sha256": intent.sha256,
                    "apply_receipt_sha256": evidence["receipt_sha256"],
                },
            )
    return evidence


def _preclaim_evidence(
    intent: ProvisioningTimeoutIntent, *, claimable_cell_count: int, claim_id: str
) -> dict[str, Any]:
    body = {
        "schema_version": PRECLAIM_SCHEMA,
        "reviewed_intent_sha256": intent.sha256,
        "selected_cell_count": len(intent.selected_cell_ids),
        "matching_reconciliation_digest_count": len(intent.selected_cell_ids),
        "claimable_cell_count_before_claim": claimable_cell_count,
        "claim_id_sha256": "sha256:" + hashlib.sha256(claim_id.encode()).hexdigest(),
        "claim_selected_in_same_transaction": True,
        "eligibility_policy": "exact_provisioning_post_504_before_model_or_scoring_v1",
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def claim(
    dsn: str,
    *,
    intent: ProvisioningTimeoutIntent,
    worker_id: str,
    serving_block: str,
    lease_seconds: int = 900,
) -> dict[str, Any] | None:
    worker = rollout_ledger._require_text(worker_id, "worker_id")  # noqa: SLF001
    route = rollout_ledger._require_text(serving_block, "serving_block")  # noqa: SLF001
    if route != intent.serving_block:
        raise rollout_ledger.LedgerError("worker route differs from reviewed recovery v2 intent")
    if lease_seconds < 30:
        raise rollout_ledger.LedgerError("lease_seconds must be at least 30")
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        rows = _rows(connection, intent, lock=True)
        if any(row["reconciliation_digest"] != intent.sha256 for row in rows):
            raise rollout_ledger.LedgerError("reviewed recovery v2 digest fence differs")
        claimable = sorted(
            (
                row
                for row in rows
                if row["state"] == "retry_review"
                and int(row["retry_count"]) < int(row["max_retries"])
            ),
            key=lambda row: (row["task_version_id"], row["model_id"], int(row["attempt"])),
        )
        if not claimable:
            return None
        row = claimable[0]
        claim_id = str(uuid.uuid4())
        claimed = connection.execute(
            """
            UPDATE rollout_cells
            SET state = 'claimed', worker_id = %s, claim_id = %s,
                session_id = NULL, started_at = NULL,
                heartbeat_at = clock_timestamp(),
                lease_expires_at = clock_timestamp() + (%s * INTERVAL '1 second'),
                completed_at = NULL, retry_count = retry_count + 1,
                result_class = NULL, receipt_digest = NULL,
                updated_at = clock_timestamp()
            WHERE cell_id = %s AND state = 'retry_review'
              AND retry_count < max_retries AND reconciliation_digest = %s
            RETURNING *
            """,
            (worker, claim_id, lease_seconds, row["cell_id"], intent.sha256),
        ).fetchone()
        if claimed is None:
            raise rollout_ledger.LedgerError("reviewed recovery v2 atomic claim lost")
        evidence = _preclaim_evidence(
            intent, claimable_cell_count=len(claimable), claim_id=claim_id
        )
        connection.execute(
            """
            INSERT INTO ledger_reconciliations (
                receipt_sha256, kind, receipt_json, created_at
            ) VALUES (%s, %s, %s, clock_timestamp())
            """,
            (
                evidence["receipt_sha256"],
                PRECLAIM_SCHEMA,
                json.dumps(evidence, sort_keys=True, separators=(",", ":")),
            ),
        )
        rollout_postgres._event(  # noqa: SLF001
            connection,
            cell_id=row["cell_id"],
            name="reviewed_provisioning_timeout_recovery_claimed",
            from_state="retry_review",
            to_state="claimed",
            worker_id=worker,
            claim_id=claim_id,
            detail={
                "lease_seconds": lease_seconds,
                "reviewed_intent_sha256": intent.sha256,
                "preclaim_receipt_sha256": evidence["receipt_sha256"],
            },
        )
        return claimed


class Ledger:
    def __init__(self, intent: ProvisioningTimeoutIntent) -> None:
        self.intent = intent

    def claim(
        self,
        dsn: str,
        *,
        worker_id: str,
        serving_block: str,
        lease_seconds: int = 900,
    ) -> dict[str, Any] | None:
        return claim(
            dsn,
            intent=self.intent,
            worker_id=worker_id,
            serving_block=serving_block,
            lease_seconds=lease_seconds,
        )

    heartbeat = staticmethod(rollout_postgres.heartbeat)
    record_local_result = staticmethod(rollout_postgres.record_local_result)
    start = staticmethod(rollout_postgres.start)
    mark_grading = staticmethod(rollout_postgres.mark_grading)
    accept = staticmethod(rollout_postgres.accept)
    request_retry_review = staticmethod(rollout_postgres.request_retry_review)
