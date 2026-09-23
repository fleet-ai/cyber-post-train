"""Reconcile an exact scored-session roster with an intent-bound terminal outcome.

The historical reconciler is deliberately byte-pinned by its completed LR30
packet.  This successor keeps that source immutable and adds one missing
invariant: the private intent binds the exact local agent termination, exit
code, and held failure code.  That lets an already-authoritatively-scored
``output_limit`` or ``process_error`` rollout be accepted without either
regenerating or rescoring it, while preventing the new path from broadening
the historical policy.
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
SUBSET_INTENT_SCHEMA = "fleet-stored-session-subset-reconciliation-intent-v1"
SUBSET_LOCAL_GAPS_INTENT_SCHEMA = "fleet-stored-session-subset-reconciliation-intent-v2"
STALE_METADATA_INTENT_SCHEMA = "fleet-stale-session-metadata-reconciliation-intent-v1"
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
SUBSET_INTENT_FIELDS = INTENT_FIELDS | {"expected_arm_state_counts", "unselected_cell_ids"}
SUBSET_LOCAL_GAPS_INTENT_FIELDS = SUBSET_INTENT_FIELDS | {
    "expected_local_result_count",
    "missing_local_result_cell_ids",
    "expected_missing_local_result_failure_code",
}
STALE_METADATA_INTENT_FIELDS = {
    "schema_version",
    "evaluation_plan_sha256",
    "runtime_files_sha256",
    "serving_block",
    "source_output_root",
    "source_database",
    "source_job_uid",
    "source_job_terminal_receipt_sha256",
    "selected_cells",
    "unselected_cell_ids",
    "expected_arm_state_counts",
    "expected_terminal_local_result_count",
    "expected_terminal_stale_active_count",
    "unselected_missing_local_result_cell_ids",
    "expected_unselected_missing_local_result_failure_code",
    "sha256",
}
STALE_METADATA_CELL_FIELDS = {
    "cell_id",
    "execution_id",
    "expected_agent_exit_code",
    "expected_agent_termination",
    "local_result_initially_present",
    "local_record_sha256",
    "claim_sha256",
    "binding_sha256",
    "runtime_binding_sha256",
    "scoring_intent_sha256",
    "result_sha256",
    "reward_sha256",
    "session_ingest_sha256",
    "cleanup_sha256",
    "trace_manifest_sha256",
    "accepted_receipt_sha256",
}
SUPPORTED_AGENT_OUTCOMES = frozenset({(0, "output_limit"), (1, "process_error")})
MISSING_LOCAL_RESULT_FAILURE_CODE = "authoritative_scoring_started.fleetrequesterror"
STALE_METADATA_UNSELECTED_FAILURE_CODE = "post_claim.connecterror"


def _body_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_FILES}


def _normalized_common(
    *,
    evaluation_plan_sha256: str,
    runtime_files_sha256: dict[str, str],
    serving_block: str,
    source_output_root: str,
    source_database: str,
    source_job_uid: str,
    source_job_terminal_receipt_sha256: str,
    selected_cell_ids: tuple[str, ...],
    expected_agent_exit_code: int,
    expected_agent_termination: str,
    expected_failure_code: str,
) -> dict[str, Any]:
    plan = rollout_ledger._require_digest(  # noqa: SLF001
        evaluation_plan_sha256, "evaluation plan sha256"
    )
    runtime = runtime_identity()
    if runtime_files_sha256 != runtime:
        raise rollout_ledger.LedgerError("stored-session v2 runtime identity differs")
    route = rollout_ledger._require_text(serving_block, "serving block")  # noqa: SLF001
    output = Path(source_output_root)
    if (
        not output.is_absolute()
        or output.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or ".." in output.parts
    ):
        raise rollout_ledger.LedgerError("source output root is not an exact SFS job path")
    database = legacy._database_name(source_database)  # noqa: SLF001
    try:
        job_uid = str(uuid.UUID(source_job_uid))
        cells = tuple(str(uuid.UUID(value)) for value in selected_cell_ids)
    except (AttributeError, TypeError, ValueError) as exc:
        raise rollout_ledger.LedgerError("stored-session v2 identity is malformed") from exc
    terminal = rollout_ledger._require_digest(  # noqa: SLF001
        source_job_terminal_receipt_sha256,
        "source job terminal receipt sha256",
    )
    if not cells or len(cells) != len(set(cells)):
        raise rollout_ledger.LedgerError("stored-session v2 roster is empty or repeated")
    if (
        type(expected_agent_exit_code) is not int
        or (expected_agent_exit_code, expected_agent_termination) not in SUPPORTED_AGENT_OUTCOMES
        or not isinstance(expected_failure_code, str)
        or not expected_failure_code
        or len(expected_failure_code) > 128
    ):
        raise rollout_ledger.LedgerError("stored-session v2 outcome is unsupported")
    return {
        "evaluation_plan_sha256": plan,
        "runtime_files_sha256": runtime,
        "serving_block": route,
        "source_output_root": str(output),
        "source_database": database,
        "source_job_uid": job_uid,
        "source_job_terminal_receipt_sha256": terminal,
        "selected_cell_ids": cells,
        "expected_agent_exit_code": expected_agent_exit_code,
        "expected_agent_termination": expected_agent_termination,
        "expected_failure_code": expected_failure_code,
    }


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
        normalized = _normalized_common(
            evaluation_plan_sha256=self.evaluation_plan_sha256,
            runtime_files_sha256=self.runtime_files_sha256,
            serving_block=self.serving_block,
            source_output_root=self.source_output_root,
            source_database=self.source_database,
            source_job_uid=self.source_job_uid,
            source_job_terminal_receipt_sha256=self.source_job_terminal_receipt_sha256,
            selected_cell_ids=self.selected_cell_ids,
            expected_agent_exit_code=self.expected_agent_exit_code,
            expected_agent_termination=self.expected_agent_termination,
            expected_failure_code=self.expected_failure_code,
        )
        body = {
            "schema_version": INTENT_SCHEMA,
            **normalized,
            "selected_cell_ids": list(normalized["selected_cell_ids"]),
        }
        intent_sha256 = rollout_ledger._require_digest(self.sha256, "intent sha256")  # noqa: SLF001
        if intent_sha256 != _body_digest(body):
            raise rollout_ledger.LedgerError("stored-session v2 intent self digest differs")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sha256", intent_sha256)


@dataclass(frozen=True)
class ExactStoredSessionSubsetIntent:
    evaluation_plan_sha256: str
    runtime_files_sha256: dict[str, str]
    serving_block: str
    source_output_root: str
    source_database: str
    source_job_uid: str
    source_job_terminal_receipt_sha256: str
    selected_cell_ids: tuple[str, ...]
    unselected_cell_ids: tuple[str, ...]
    expected_arm_state_counts: dict[str, int]
    expected_agent_exit_code: int
    expected_agent_termination: str
    expected_failure_code: str
    sha256: str
    expected_local_result_count: int | None = None
    missing_local_result_cell_ids: tuple[str, ...] = ()
    expected_missing_local_result_failure_code: str | None = None

    def __post_init__(self) -> None:
        normalized = _normalized_common(
            evaluation_plan_sha256=self.evaluation_plan_sha256,
            runtime_files_sha256=self.runtime_files_sha256,
            serving_block=self.serving_block,
            source_output_root=self.source_output_root,
            source_database=self.source_database,
            source_job_uid=self.source_job_uid,
            source_job_terminal_receipt_sha256=self.source_job_terminal_receipt_sha256,
            selected_cell_ids=self.selected_cell_ids,
            expected_agent_exit_code=self.expected_agent_exit_code,
            expected_agent_termination=self.expected_agent_termination,
            expected_failure_code=self.expected_failure_code,
        )
        try:
            unselected = tuple(str(uuid.UUID(value)) for value in self.unselected_cell_ids)
        except (AttributeError, TypeError, ValueError) as exc:
            raise rollout_ledger.LedgerError("stored-session subset identity is malformed") from exc
        selected = normalized["selected_cell_ids"]
        states = tuple(rollout_ledger.STATES)
        counts = self.expected_arm_state_counts
        if (
            not unselected
            or len(unselected) != len(set(unselected))
            or set(selected) & set(unselected)
            or not isinstance(counts, dict)
            or set(counts) != set(states)
            or any(type(counts[state]) is not int or counts[state] < 0 for state in states)
            or sum(counts.values()) != len(selected) + len(unselected)
            or counts["retry_review"] < len(selected)
            or any(
                counts[state] != 0
                for state in ("pending", *rollout_ledger.ACTIVE_STATES, "terminal")
            )
        ):
            raise rollout_ledger.LedgerError("stored-session subset arm census is invalid")
        normalized_counts = {state: counts[state] for state in states}
        local_gap_values_supplied = any(
            (
                self.expected_local_result_count is not None,
                bool(self.missing_local_result_cell_ids),
                self.expected_missing_local_result_failure_code is not None,
            )
        )
        missing_local_results: tuple[str, ...] = ()
        if local_gap_values_supplied:
            try:
                missing_local_results = tuple(
                    str(uuid.UUID(value)) for value in self.missing_local_result_cell_ids
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise rollout_ledger.LedgerError(
                    "stored-session subset local-result identity is malformed"
                ) from exc
            total = len(selected) + len(unselected)
            if (
                total != 17
                or len(selected) != 4
                or len(unselected) != 13
                or normalized_counts["accepted"] != 10
                or normalized_counts["retry_review"] != 7
                or type(self.expected_local_result_count) is not int
                or self.expected_local_result_count != 15
                or len(missing_local_results) != 2
                or len(set(missing_local_results)) != 2
                or not set(missing_local_results).issubset(unselected)
                or self.expected_local_result_count != total - len(missing_local_results)
                or self.expected_missing_local_result_failure_code
                != MISSING_LOCAL_RESULT_FAILURE_CODE
            ):
                raise rollout_ledger.LedgerError(
                    "stored-session subset local-result exception is unsupported"
                )
        body = {
            "schema_version": (
                SUBSET_LOCAL_GAPS_INTENT_SCHEMA
                if local_gap_values_supplied
                else SUBSET_INTENT_SCHEMA
            ),
            **normalized,
            "selected_cell_ids": list(selected),
            "unselected_cell_ids": list(unselected),
            "expected_arm_state_counts": normalized_counts,
        }
        if local_gap_values_supplied:
            body.update(
                expected_local_result_count=self.expected_local_result_count,
                missing_local_result_cell_ids=list(missing_local_results),
                expected_missing_local_result_failure_code=(
                    self.expected_missing_local_result_failure_code
                ),
            )
        intent_sha256 = rollout_ledger._require_digest(self.sha256, "intent sha256")  # noqa: SLF001
        if intent_sha256 != _body_digest(body):
            raise rollout_ledger.LedgerError("stored-session subset intent self digest differs")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "unselected_cell_ids", unselected)
        object.__setattr__(self, "expected_arm_state_counts", normalized_counts)
        object.__setattr__(self, "missing_local_result_cell_ids", missing_local_results)
        object.__setattr__(self, "sha256", intent_sha256)


@dataclass(frozen=True)
class ExactStaleSessionMetadataIntent:
    """One hard-scoped recovery for the four terminal s52 owners.

    This is intentionally not a generic stale-claim repair.  Its immutable shape
    matches the reviewed s52 terminal snapshot: four complete selected sessions,
    one missing selected local-result index, and three unrelated pre-model gaps
    that remain untouched.
    """

    evaluation_plan_sha256: str
    runtime_files_sha256: dict[str, str]
    serving_block: str
    source_output_root: str
    source_database: str
    source_job_uid: str
    source_job_terminal_receipt_sha256: str
    selected_cells: tuple[dict[str, Any], ...]
    unselected_cell_ids: tuple[str, ...]
    expected_arm_state_counts: dict[str, int]
    expected_terminal_local_result_count: int
    expected_terminal_stale_active_count: int
    unselected_missing_local_result_cell_ids: tuple[str, ...]
    expected_unselected_missing_local_result_failure_code: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.selected_cells, tuple) or any(
            not isinstance(row, dict) for row in self.selected_cells
        ):
            raise rollout_ledger.LedgerError("stale-session metadata cell binding is invalid")
        common = _normalized_common(
            evaluation_plan_sha256=self.evaluation_plan_sha256,
            runtime_files_sha256=self.runtime_files_sha256,
            serving_block=self.serving_block,
            source_output_root=self.source_output_root,
            source_database=self.source_database,
            source_job_uid=self.source_job_uid,
            source_job_terminal_receipt_sha256=self.source_job_terminal_receipt_sha256,
            selected_cell_ids=tuple(row.get("cell_id") for row in self.selected_cells),
            expected_agent_exit_code=0,
            expected_agent_termination="output_limit",
            expected_failure_code="terminal_owner_metadata_recovery",
        )
        try:
            unselected = tuple(str(uuid.UUID(value)) for value in self.unselected_cell_ids)
            missing_unselected = tuple(
                str(uuid.UUID(value)) for value in self.unselected_missing_local_result_cell_ids
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise rollout_ledger.LedgerError(
                "stale-session metadata identity is malformed"
            ) from exc
        cells: list[dict[str, Any]] = []
        for value in self.selected_cells:
            if not isinstance(value, dict) or set(value) != STALE_METADATA_CELL_FIELDS:
                raise rollout_ledger.LedgerError("stale-session metadata cell binding is invalid")
            try:
                cell_id = str(uuid.UUID(value["cell_id"]))
            except (AttributeError, TypeError, ValueError) as exc:
                raise rollout_ledger.LedgerError(
                    "stale-session metadata cell identity is malformed"
                ) from exc
            execution_id = rollout_ledger._require_text(  # noqa: SLF001
                value["execution_id"], "execution id"
            )
            if rollout_ledger.EXECUTION_ID_RE.fullmatch(execution_id) is None:
                raise rollout_ledger.LedgerError("stale-session execution identity is malformed")
            if (
                type(value["expected_agent_exit_code"]) is not int
                or value["expected_agent_exit_code"] != 0
                or value["expected_agent_termination"] not in {"completed", "output_limit"}
                or type(value["local_result_initially_present"]) is not bool
            ):
                raise rollout_ledger.LedgerError("stale-session outcome binding is unsupported")
            digests: dict[str, str | None] = {}
            for field in (
                "local_record_sha256",
                "claim_sha256",
                "binding_sha256",
                "runtime_binding_sha256",
                "scoring_intent_sha256",
                "result_sha256",
                "reward_sha256",
                "session_ingest_sha256",
                "cleanup_sha256",
                "trace_manifest_sha256",
                "accepted_receipt_sha256",
            ):
                raw = value[field]
                if raw is None and field in {"local_record_sha256", "accepted_receipt_sha256"}:
                    digests[field] = None
                else:
                    digests[field] = rollout_ledger._require_digest(raw, field)  # noqa: SLF001
            if value["local_result_initially_present"] != (
                digests["local_record_sha256"] is not None
            ):
                raise rollout_ledger.LedgerError("stale-session local-result binding is invalid")
            if (digests["accepted_receipt_sha256"] is not None) != (
                value["expected_agent_termination"] == "completed"
            ):
                raise rollout_ledger.LedgerError(
                    "stale-session accepted-receipt binding is invalid"
                )
            cells.append(
                {
                    **value,
                    "cell_id": cell_id,
                    "execution_id": execution_id,
                    **digests,
                }
            )
        selected = tuple(row["cell_id"] for row in cells)
        counts = self.expected_arm_state_counts
        expected_counts = {state: 0 for state in rollout_ledger.STATES}
        expected_counts.update(accepted=5, claimed=4, retry_review=8)
        if (
            len(cells) != 4
            or len(set(selected)) != 4
            or len(unselected) != 13
            or len(set(unselected)) != 13
            or set(selected) & set(unselected)
            or counts != expected_counts
            or self.expected_terminal_local_result_count != 13
            or self.expected_terminal_stale_active_count != 3
            or len(missing_unselected) != 3
            or len(set(missing_unselected)) != 3
            or not set(missing_unselected).issubset(unselected)
            or self.expected_unselected_missing_local_result_failure_code
            != STALE_METADATA_UNSELECTED_FAILURE_CODE
            or sum(row["local_result_initially_present"] for row in cells) != 3
            or sum(row["expected_agent_termination"] == "completed" for row in cells) != 1
            or sum(row["expected_agent_termination"] == "output_limit" for row in cells) != 3
        ):
            raise rollout_ledger.LedgerError("stale-session metadata incident shape is unsupported")
        body = {
            "schema_version": STALE_METADATA_INTENT_SCHEMA,
            "evaluation_plan_sha256": common["evaluation_plan_sha256"],
            "runtime_files_sha256": common["runtime_files_sha256"],
            "serving_block": common["serving_block"],
            "source_output_root": common["source_output_root"],
            "source_database": common["source_database"],
            "source_job_uid": common["source_job_uid"],
            "source_job_terminal_receipt_sha256": common["source_job_terminal_receipt_sha256"],
            "selected_cells": cells,
            "unselected_cell_ids": list(unselected),
            "expected_arm_state_counts": expected_counts,
            "expected_terminal_local_result_count": 13,
            "expected_terminal_stale_active_count": 3,
            "unselected_missing_local_result_cell_ids": list(missing_unselected),
            "expected_unselected_missing_local_result_failure_code": (
                STALE_METADATA_UNSELECTED_FAILURE_CODE
            ),
        }
        digest = rollout_ledger._require_digest(self.sha256, "intent sha256")  # noqa: SLF001
        if digest != _body_digest(body):
            raise rollout_ledger.LedgerError("stale-session metadata intent self digest differs")
        for name in (
            "evaluation_plan_sha256",
            "runtime_files_sha256",
            "serving_block",
            "source_output_root",
            "source_database",
            "source_job_uid",
            "source_job_terminal_receipt_sha256",
        ):
            object.__setattr__(self, name, common[name])
        object.__setattr__(self, "selected_cells", tuple(cells))
        object.__setattr__(self, "unselected_cell_ids", unselected)
        object.__setattr__(self, "expected_arm_state_counts", expected_counts)
        object.__setattr__(self, "unselected_missing_local_result_cell_ids", missing_unselected)
        object.__setattr__(self, "sha256", digest)

    @property
    def selected_cell_ids(self) -> tuple[str, ...]:
        return tuple(row["cell_id"] for row in self.selected_cells)


StoredSessionIntent = (
    ExactStoredSessionIntent | ExactStoredSessionSubsetIntent | ExactStaleSessionMetadataIntent
)


def load_intent(path: Path) -> StoredSessionIntent:
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
    if not isinstance(value, dict):
        raise rollout_ledger.LedgerError("stored-session v2 intent schema is unsupported")
    if not isinstance(value.get("selected_cell_ids"), list):
        if (
            value.get("schema_version") != STALE_METADATA_INTENT_SCHEMA
            or set(value) != STALE_METADATA_INTENT_FIELDS
            or not isinstance(value.get("selected_cells"), list)
            or not isinstance(value.get("unselected_cell_ids"), list)
            or not isinstance(value.get("unselected_missing_local_result_cell_ids"), list)
        ):
            raise rollout_ledger.LedgerError("stored-session v2 intent schema is unsupported")
        return ExactStaleSessionMetadataIntent(
            evaluation_plan_sha256=value["evaluation_plan_sha256"],
            runtime_files_sha256=value["runtime_files_sha256"],
            serving_block=value["serving_block"],
            source_output_root=value["source_output_root"],
            source_database=value["source_database"],
            source_job_uid=value["source_job_uid"],
            source_job_terminal_receipt_sha256=value["source_job_terminal_receipt_sha256"],
            selected_cells=tuple(value["selected_cells"]),
            unselected_cell_ids=tuple(value["unselected_cell_ids"]),
            expected_arm_state_counts=value["expected_arm_state_counts"],
            expected_terminal_local_result_count=value["expected_terminal_local_result_count"],
            expected_terminal_stale_active_count=value["expected_terminal_stale_active_count"],
            unselected_missing_local_result_cell_ids=tuple(
                value["unselected_missing_local_result_cell_ids"]
            ),
            expected_unselected_missing_local_result_failure_code=value[
                "expected_unselected_missing_local_result_failure_code"
            ],
            sha256=value["sha256"],
        )
    subset_schema = value.get("schema_version")
    subset_fields = set(value)
    if (subset_schema == SUBSET_INTENT_SCHEMA and subset_fields == SUBSET_INTENT_FIELDS) or (
        subset_schema == SUBSET_LOCAL_GAPS_INTENT_SCHEMA
        and subset_fields == SUBSET_LOCAL_GAPS_INTENT_FIELDS
    ):
        if not isinstance(value.get("unselected_cell_ids"), list):
            raise rollout_ledger.LedgerError("stored-session v2 intent schema is unsupported")
        return ExactStoredSessionSubsetIntent(
            evaluation_plan_sha256=value["evaluation_plan_sha256"],
            runtime_files_sha256=value["runtime_files_sha256"],
            serving_block=value["serving_block"],
            source_output_root=value["source_output_root"],
            source_database=value["source_database"],
            source_job_uid=value["source_job_uid"],
            source_job_terminal_receipt_sha256=value["source_job_terminal_receipt_sha256"],
            selected_cell_ids=tuple(value["selected_cell_ids"]),
            unselected_cell_ids=tuple(value["unselected_cell_ids"]),
            expected_arm_state_counts=value["expected_arm_state_counts"],
            expected_agent_exit_code=value["expected_agent_exit_code"],
            expected_agent_termination=value["expected_agent_termination"],
            expected_failure_code=value["expected_failure_code"],
            sha256=value["sha256"],
            expected_local_result_count=value.get("expected_local_result_count"),
            missing_local_result_cell_ids=tuple(value.get("missing_local_result_cell_ids", ())),
            expected_missing_local_result_failure_code=value.get(
                "expected_missing_local_result_failure_code"
            ),
        )
    if set(value) != INTENT_FIELDS or value.get("schema_version") != INTENT_SCHEMA:
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


def _validate_artifacts(row: dict[str, Any], intent: StoredSessionIntent) -> str:
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


def _stale_metadata_rows(
    connection: Any, intent: ExactStaleSessionMetadataIntent, *, lock: bool
) -> list[dict[str, Any]]:
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
    if (
        len(rows) != 4
        or {row["cell_id"] for row in rows} != set(intent.selected_cell_ids)
        or any(row["serving_block"] != intent.serving_block for row in rows)
    ):
        raise rollout_ledger.LedgerError("stale-session selected roster differs")
    return rows


def _exact_private_file(root: Path, name: str, expected_sha256: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
        raise rollout_ledger.LedgerError("stale-session artifact file is not a regular child")
    if legacy._file_sha256(path) != expected_sha256:  # noqa: SLF001
        raise rollout_ledger.LedgerError("stale-session artifact digest differs")
    return path


def _accepted_receipt_binding(
    path: Path | None,
    *,
    descriptor: dict[str, Any],
    row: dict[str, Any],
    config: dict[str, Any],
    result: dict[str, Any],
) -> None:
    expected = descriptor["accepted_receipt_sha256"]
    if expected is None:
        if path is not None and path.exists():
            raise rollout_ledger.LedgerError("unexpected stale-session accepted receipt exists")
        return
    if path is None:
        raise rollout_ledger.LedgerError("stale-session accepted receipt is missing")
    _exact_private_file(path.parent, path.name, expected)
    value = legacy._json(path)  # noqa: SLF001
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if any(
        (
            value.get("schema_version") != rollout_worker.ACCEPTED_SCHEMA,
            value.get("receipt_sha256") != crypto.digest_without(body, "receipt_sha256"),
            value.get("accepted") is not True,
            value.get("campaign_id") != config["campaign_id"],
            value.get("cell_id") != config["execution"]["cell_id"],
            value.get("execution_id") != descriptor["execution_id"],
            value.get("ledger_cell_id") != row["cell_id"],
            value.get("run_id") != config["run_id"],
            value.get("serving_block") != row["serving_block"],
            value.get("session_id") != result.get("session_id"),
            value.get("verifier_execution_id") != result.get("verifier_execution_id"),
            value.get("config_sha256") != config["config_sha256"],
            value.get("session_ingest_completed") is not True,
            value.get("cleanup_completed") is not True,
            value.get("score_persisted_privately") is not True,
            value.get("scores_included") is not False,
            value.get("prompts_or_traces_included") is not False,
        )
    ):
        raise rollout_ledger.LedgerError("stale-session accepted receipt binding differs")


def _validate_stale_runtime_bindings(
    *,
    binding: dict[str, Any],
    runtime: dict[str, Any],
    scoring_intent: dict[str, Any],
    trace_manifest: dict[str, Any],
    result: dict[str, Any],
    reward: dict[str, Any],
    descriptor: dict[str, Any],
    config: dict[str, Any],
) -> None:
    binding_fields = {
        "schema_version",
        "run_id",
        "source_job_id",
        "task",
        "environment",
        "verifier",
        "authority",
        "authority_gate",
        "model",
        "harness",
    }
    static_binding_fields = binding_fields - {"authority_gate"}
    authority = config["authority"]
    expected_routes = sorted(
        (
            authority["provisioning_route_template"],
            authority["scoring_route_template"],
        )
    )
    authority_gate = binding.get("authority_gate")
    valid_authority_gate = authority_gate in (
        {"mode": "openapi", "routes": expected_routes},
        {
            "mode": "behavioral_method_not_allowed",
            "method": "GET",
            "statuses": {"provisioning": 405, "scoring": 405},
        },
    )
    if (
        set(binding) != binding_fields
        or any(binding.get(field) != config.get(field) for field in static_binding_fields)
        or not valid_authority_gate
    ):
        raise rollout_ledger.LedgerError("stale-session frozen runtime binding differs")

    environment = config["environment"]
    runtime_fields = {
        "instance_id",
        "evidence_run_id",
        "env_key",
        "environment_version",
        "data_key",
        "data_version",
        "tool_names",
        "tool_catalog_sha256",
    }
    if environment.get("version_id") is not None:
        runtime_fields.add("environment_version_id")
    if (
        set(runtime) != runtime_fields
        or runtime.get("instance_id") != result.get("instance_id")
        or runtime.get("evidence_run_id") != result.get("evidence_run_id")
        or runtime.get("env_key") != environment["id"]
        or runtime.get("environment_version") != environment["version"]
        or runtime.get("data_key") != environment["data_id"]
        or runtime.get("data_version") != environment["data_version"]
        or runtime.get("environment_version_id") != environment.get("version_id")
        or runtime.get("tool_names") != config["execution"]["required_task_tools"]
        or rollout_ledger._require_digest(  # noqa: SLF001
            runtime.get("tool_catalog_sha256", ""), "tool catalog sha256"
        )
        != rollout_ledger._require_digest(  # noqa: SLF001
            config["execution"]["required_task_tool_catalog_sha256"],
            "required tool catalog sha256",
        )
    ):
        raise rollout_ledger.LedgerError("stale-session environment runtime binding differs")

    scoring_fields = {
        "schema_version",
        "run_id",
        "task_key",
        "task_version_id",
        "instance_id",
        "evidence_run_id",
        "scoring_payload_mode",
        "request_keys",
        "request_sha256",
        "scoring_intent_sha256",
    }
    request_keys = {"instance_id", "scoring_mode", "multi_app_aggregation_mode"}
    if authority.get("scoring_payload_mode") != self_hosted.RUNTIME_EVIDENCE_ONLY_V3:
        request_keys.update({"final_answer", "conversation"})
    rollout_ledger._require_digest(  # noqa: SLF001
        scoring_intent.get("request_sha256", ""), "scoring request sha256"
    )
    if (
        set(scoring_intent) != scoring_fields
        or scoring_intent.get("schema_version") != self_hosted.SCORING_INTENT_SCHEMA
        or scoring_intent.get("scoring_intent_sha256")
        != crypto.digest_without(scoring_intent, "scoring_intent_sha256")
        or scoring_intent.get("run_id") != config["run_id"]
        or scoring_intent.get("task_key") != config["task"]["key"]
        or scoring_intent.get("task_version_id") != config["task"]["version_id"]
        or scoring_intent.get("instance_id") != runtime["instance_id"]
        or scoring_intent.get("evidence_run_id") != runtime["evidence_run_id"]
        or scoring_intent.get("scoring_payload_mode") != authority.get("scoring_payload_mode")
        or scoring_intent.get("request_keys") != sorted(request_keys)
    ):
        raise rollout_ledger.LedgerError("stale-session scoring intent binding differs")

    if any(
        (
            result.get("harness") != config["harness"]["name"],
            reward.get("instance_id") != runtime["instance_id"],
            trace_manifest.get("harness") != config["harness"]["name"],
            trace_manifest.get("agent_termination") != descriptor["expected_agent_termination"],
        )
    ):
        raise rollout_ledger.LedgerError("stale-session local runtime identity differs")


def _reconstruct_stale_local_result(
    row: dict[str, Any],
    descriptor: dict[str, Any],
    config: dict[str, Any],
    intent: ExactStaleSessionMetadataIntent,
) -> tuple[dict[str, Any], str, str]:
    if descriptor["execution_id"] != config["execution"]["execution_id"]:
        raise rollout_ledger.LedgerError("stale-session execution binding differs")
    execution_name = descriptor["execution_id"].removeprefix("sha256:")
    root = Path(intent.source_output_root).resolve()
    attempt = (root / "attempts" / execution_name).resolve()
    claim_path = (root / "claims" / f"{execution_name}.json").resolve()
    if (
        attempt.is_symlink()
        or not attempt.is_dir()
        or root not in attempt.parents
        or claim_path.is_symlink()
        or not claim_path.is_file()
        or root not in claim_path.parents
        or legacy._file_sha256(claim_path) != descriptor["claim_sha256"]  # noqa: SLF001
    ):
        raise rollout_ledger.LedgerError("stale-session attempt or claim binding differs")
    claim = legacy._json(claim_path)  # noqa: SLF001
    expected_claim = rollout_worker._claim_receipt(config, row)  # noqa: SLF001
    if claim != expected_claim:
        raise rollout_ledger.LedgerError("stale-session claim receipt differs")

    binding_path = _exact_private_file(attempt, "binding.json", descriptor["binding_sha256"])
    runtime_path = _exact_private_file(
        attempt, "runtime-binding.json", descriptor["runtime_binding_sha256"]
    )
    scoring_intent_path = _exact_private_file(
        attempt, "scoring-intent.json", descriptor["scoring_intent_sha256"]
    )
    result_path = _exact_private_file(attempt, "result.json", descriptor["result_sha256"])
    reward_path = _exact_private_file(attempt, "reward-result.json", descriptor["reward_sha256"])
    ingest_path = _exact_private_file(
        attempt, "session-ingest.json", descriptor["session_ingest_sha256"]
    )
    cleanup_path = _exact_private_file(attempt, "cleanup.json", descriptor["cleanup_sha256"])
    manifest_path = _exact_private_file(
        attempt, "trace-manifest.json", descriptor["trace_manifest_sha256"]
    )
    binding, runtime, scoring_intent, result, reward, ingest, cleanup, manifest = map(
        legacy._json,  # noqa: SLF001
        (
            binding_path,
            runtime_path,
            scoring_intent_path,
            result_path,
            reward_path,
            ingest_path,
            cleanup_path,
            manifest_path,
        ),
    )
    trace_relative = Path(str(manifest.get("canonical_trace") or ""))
    if not trace_relative.parts or trace_relative.is_absolute() or ".." in trace_relative.parts:
        raise rollout_ledger.LedgerError("stale-session canonical trace path is unsafe")
    trace_sha256 = rollout_ledger._require_digest(  # noqa: SLF001
        manifest.get("canonical_trace_sha256"), "canonical trace sha256"
    )
    trace_path = _exact_private_file(attempt, trace_relative.as_posix(), trace_sha256)
    _validate_stale_runtime_bindings(
        binding=binding,
        runtime=runtime,
        scoring_intent=scoring_intent,
        trace_manifest=manifest,
        result=result,
        reward=reward,
        descriptor=descriptor,
        config=config,
    )
    score = legacy._finite_score(result.get("score"))  # noqa: SLF001
    if any(
        (
            result.get("run_id") != config["run_id"],
            result.get("task_key") != row["task_key"],
            result.get("task_version_id") != row["task_version_id"],
            result.get("agent_exit_code") != descriptor["expected_agent_exit_code"],
            result.get("agent_termination") != descriptor["expected_agent_termination"],
            result.get("session_ingest_status") != "completed",
            not isinstance(result.get("elapsed_seconds"), (int, float)),
            reward.get("task_key") != row["task_key"],
            reward.get("task_version_id") != row["task_version_id"],
            reward.get("instance_id") != result.get("instance_id"),
            reward.get("verifier_execution_id") != result.get("verifier_execution_id"),
            legacy._finite_score(reward.get("reward")) != score,  # noqa: SLF001
            ingest.get("status") != "completed",
            ingest.get("session_id") != result.get("session_id"),
            cleanup
            != {
                "instance_created": True,
                "instance_closed": True,
                "containers_removed": True,
            },
        )
    ):
        raise rollout_ledger.LedgerError("stale-session private lifecycle binding differs")
    normalized = rollout_ledger.normalize_local_result(
        {
            "execution_id": descriptor["execution_id"],
            "execution_generation": 1,
            "run_id": config["run_id"],
            "session_id": result.get("session_id"),
            "verifier_execution_id": result.get("verifier_execution_id"),
            "score": score,
            "config_sha256": config["config_sha256"],
            "artifact_directory": f"attempts/{execution_name}",
            "trace_path": trace_relative.as_posix(),
            "trace_sha256": trace_sha256,
            "result_path": "result.json",
            "result_sha256": descriptor["result_sha256"],
            "reward_path": "reward-result.json",
            "reward_sha256": descriptor["reward_sha256"],
            "session_ingest_path": "session-ingest.json",
            "session_ingest_sha256": descriptor["session_ingest_sha256"],
            "cleanup_path": "cleanup.json",
            "cleanup_sha256": descriptor["cleanup_sha256"],
            "session_ingest_status": "completed",
            "agent_exit_code": descriptor["expected_agent_exit_code"],
            "agent_termination": descriptor["expected_agent_termination"],
            "elapsed_seconds": result["elapsed_seconds"],
        },
        cell_id=row["cell_id"],
    )
    record_sha256 = _body_digest(normalized)
    if (
        descriptor["local_result_initially_present"]
        and descriptor["local_record_sha256"] != (row["record_sha256"])
    ):
        raise rollout_ledger.LedgerError("stale-session indexed record binding differs")
    existing = row["local_session_id"] is not None
    if existing:
        observed = legacy._normalized_local_result(row)  # noqa: SLF001
        if observed != normalized or row["record_sha256"] != record_sha256:
            raise rollout_ledger.LedgerError("stale-session indexed metadata differs")
    elif descriptor["local_result_initially_present"]:
        raise rollout_ledger.LedgerError("stale-session expected indexed metadata is missing")
    _accepted_receipt_binding(
        attempt / "ACCEPTED.json",
        descriptor=descriptor,
        row=row,
        config=config,
        result=result,
    )
    artifact_binding = _body_digest(
        {
            "claim_sha256": descriptor["claim_sha256"],
            "binding_sha256": descriptor["binding_sha256"],
            "runtime_binding_sha256": descriptor["runtime_binding_sha256"],
            "scoring_intent_sha256": descriptor["scoring_intent_sha256"],
            "result_sha256": descriptor["result_sha256"],
            "reward_sha256": descriptor["reward_sha256"],
            "session_ingest_sha256": descriptor["session_ingest_sha256"],
            "cleanup_sha256": descriptor["cleanup_sha256"],
            "trace_manifest_sha256": descriptor["trace_manifest_sha256"],
            "trace_sha256": legacy._file_sha256(trace_path),  # noqa: SLF001
            "record_sha256": record_sha256,
            "accepted_receipt_sha256": descriptor["accepted_receipt_sha256"],
            "private_lifecycle_complete": True,
        }
    )
    return normalized, record_sha256, artifact_binding


def _observe_stale_metadata(
    dsn: str,
    *,
    intent: ExactStaleSessionMetadataIntent,
    evaluation_directory: Path,
    client: httpx.Client,
) -> dict[str, Any]:
    from evals.fleet import evaluate

    plan, proof = evaluate.checked_preflight(evaluation_directory)
    plan = {
        **plan,
        "task_bindings": proof["task_bindings"],
        "_plan_csv": str(evaluation_directory / "plan.csv"),
    }
    local_plan = rollout_ledger._plan_digest(  # noqa: SLF001
        rollout_ledger._plan_rows(evaluation_directory / "plan.csv")  # noqa: SLF001
    )
    if local_plan != intent.evaluation_plan_sha256:
        raise rollout_ledger.LedgerError("stale-session intent differs from local ledger plan")
    observed_plan = rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    if observed_plan.get("plan_sha256") != local_plan:
        raise rollout_ledger.LedgerError("stale-session database differs from local ledger plan")
    with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        rows = _stale_metadata_rows(connection, intent, lock=False)
    scientific = legacy._scientific_index(plan)  # noqa: SLF001
    selected_tasks = {row["task_version_id"]: row for row in plan["tasks"]}
    descriptors = {row["cell_id"]: row for row in intent.selected_cells}
    observations: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        key = (row["model_id"], row["task_version_id"], int(row["attempt"]))
        if key not in scientific or row["task_version_id"] not in selected_tasks:
            raise rollout_ledger.LedgerError("stale-session cell is absent from frozen plan")
        config = rollout_worker.build_config(
            plan, row, scientific[key], selected_tasks[row["task_version_id"]], client
        )
        normalized, record_sha256, artifact_binding = _reconstruct_stale_local_result(
            row, descriptors[row["cell_id"]], config, intent
        )
        private_row = {
            **row,
            **normalized,
            "local_session_id": normalized["session_id"],
            "record_sha256": record_sha256,
        }
        observations.append(
            legacy._session_receipt(  # noqa: SLF001
                client,
                row=private_row,
                config=config,
                artifact_binding_sha256=artifact_binding,
            )
        )
        records.append(
            {
                "cell_id": row["cell_id"],
                "record": normalized,
                "record_sha256": record_sha256,
                "initially_present": descriptors[row["cell_id"]]["local_result_initially_present"],
            }
        )
    return {"observations": observations, "records": records}


def observe(
    dsn: str,
    *,
    intent: StoredSessionIntent,
    evaluation_directory: Path,
    client: httpx.Client,
) -> list[dict[str, Any]] | dict[str, Any]:
    if isinstance(intent, ExactStaleSessionMetadataIntent):
        return _observe_stale_metadata(
            dsn, intent=intent, evaluation_directory=evaluation_directory, client=client
        )
    from evals.fleet import evaluate

    plan, proof = evaluate.checked_preflight(evaluation_directory)
    plan = {
        **plan,
        "task_bindings": proof["task_bindings"],
        "_plan_csv": str(evaluation_directory / "plan.csv"),
    }
    local_ledger_plan_sha256 = rollout_ledger._plan_digest(  # noqa: SLF001
        rollout_ledger._plan_rows(evaluation_directory / "plan.csv")  # noqa: SLF001
    )
    if local_ledger_plan_sha256 != intent.evaluation_plan_sha256:
        raise rollout_ledger.LedgerError(
            "stored-session v2 intent differs from local source ledger plan"
        )
    ledger_plan = rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    # The terminal source receipt exposes the score-blind PostgreSQL plan identity,
    # which is the normalized plan.csv digest.  EVALUATION_PLAN.json has a separate
    # self digest and is already checked by checked_preflight() above.
    observed_plan_sha256 = rollout_ledger._require_digest(  # noqa: SLF001
        ledger_plan["plan_sha256"], "ledger plan sha256"
    )
    if observed_plan_sha256 != local_ledger_plan_sha256:
        raise rollout_ledger.LedgerError(
            "stored-session v2 database differs from local source ledger plan"
        )
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


def _accept_full_roster(
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


def _state_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {state: sum(row["state"] == state for row in rows) for state in rollout_ledger.STATES}


def _locked_arm_rows(
    connection: Any, intent: ExactStoredSessionSubsetIntent
) -> list[dict[str, Any]]:
    rows = connection.execute("SELECT * FROM rollout_cells ORDER BY cell_id FOR UPDATE").fetchall()
    expected_ids = set(intent.selected_cell_ids) | set(intent.unselected_cell_ids)
    if (
        len(rows) != len(expected_ids)
        or {row["cell_id"] for row in rows} != expected_ids
        or any(row["serving_block"] != intent.serving_block for row in rows)
    ):
        raise rollout_ledger.LedgerError("stored-session subset full arm census differs")
    return rows


def _locked_local_result_cell_ids(
    connection: Any,
    intent: ExactStoredSessionSubsetIntent,
    arm_rows: list[dict[str, Any]],
) -> tuple[str, ...] | None:
    """Prove the one supported partial local-result roster under the arm lock."""

    if intent.expected_local_result_count is None:
        return None
    # ``_locked_arm_rows`` already holds every parent cell ``FOR UPDATE``.
    # ``record_local_result`` takes the same cell lock before inserting, so an
    # absent result cannot appear between this census and transaction commit.
    rows = connection.execute(
        """
        SELECT cell_id FROM rollout_local_results
        WHERE cell_id = ANY(%s::text[])
        ORDER BY cell_id
        FOR UPDATE
        """,
        ([row["cell_id"] for row in arm_rows],),
    ).fetchall()
    observed = tuple(row["cell_id"] for row in rows)
    observed_set = set(observed)
    all_ids = {row["cell_id"] for row in arm_rows}
    missing = set(intent.missing_local_result_cell_ids)
    selected = set(intent.selected_cell_ids)
    arm_by_id = {row["cell_id"]: row for row in arm_rows}
    if (
        len(observed) != intent.expected_local_result_count
        or len(observed_set) != len(observed)
        or observed_set != all_ids - missing
        or not selected.issubset(observed_set)
        or any(
            arm_by_id[cell_id]["state"] != "retry_review"
            or arm_by_id[cell_id]["failure_code"]
            != intent.expected_missing_local_result_failure_code
            or arm_by_id[cell_id]["reconciliation_digest"] is not None
            for cell_id in missing
        )
    ):
        raise rollout_ledger.LedgerError("stored-session subset local-result roster differs")
    return observed


def _expected_post_counts(intent: ExactStoredSessionSubsetIntent) -> dict[str, int]:
    result = dict(intent.expected_arm_state_counts)
    result["retry_review"] -= len(intent.selected_cell_ids)
    result["accepted"] += len(intent.selected_cell_ids)
    return result


def _subset_receipt(intent: ExactStoredSessionSubsetIntent) -> dict[str, Any]:
    post_counts = _expected_post_counts(intent)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "reviewed_intent_sha256": intent.sha256,
        "evaluation_plan_sha256": intent.evaluation_plan_sha256,
        "source_job_uid_sha256": "sha256:"
        + hashlib.sha256(intent.source_job_uid.encode()).hexdigest(),
        "source_job_terminal_receipt_sha256": intent.source_job_terminal_receipt_sha256,
        "selected_cell_count": len(intent.selected_cell_ids),
        "source_total_cell_count": len(intent.selected_cell_ids) + len(intent.unselected_cell_ids),
        "prior_retry_review_count": intent.expected_arm_state_counts["retry_review"],
        "prior_arm_state_counts": intent.expected_arm_state_counts,
        "post_arm_state_counts": post_counts,
        "nonselected_cell_count": len(intent.unselected_cell_ids),
        "nonselected_cells_preserved": True,
        "accepted_existing_completed_session_count": len(intent.selected_cell_ids),
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
    if intent.expected_local_result_count is not None:
        body.update(
            source_local_result_count=intent.expected_local_result_count,
            missing_local_result_count=len(intent.missing_local_result_cell_ids),
            missing_local_results_are_unselected=True,
            selected_cells_have_local_results=True,
            missing_local_result_cells_preserved=True,
            missing_local_result_failure_code_sha256="sha256:"
            + hashlib.sha256(
                intent.expected_missing_local_result_failure_code.encode()
            ).hexdigest(),
        )
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def _existing_receipt(connection: Any, intent: ExactStoredSessionSubsetIntent) -> dict[str, Any]:
    existing = connection.execute(
        "SELECT receipt_json FROM ledger_reconciliations WHERE kind = %s",
        (RECEIPT_SCHEMA,),
    ).fetchall()
    matches = [json.loads(row["receipt_json"]) for row in existing]
    matches = [row for row in matches if row.get("reviewed_intent_sha256") == intent.sha256]
    expected = _subset_receipt(intent)
    if len(matches) != 1 or matches[0] != expected:
        raise rollout_ledger.LedgerError("stored-session v2 terminal receipt is ambiguous")
    return matches[0]


def _accept_subset_roster(
    dsn: str,
    *,
    intent: ExactStoredSessionSubsetIntent,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {row["cell_id"]: row for row in observations}
    if set(indexed) != set(intent.selected_cell_ids) or len(indexed) != len(observations):
        raise rollout_ledger.LedgerError("stored-session v2 observations differ from intent roster")
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        arm_before = _locked_arm_rows(connection, intent)
        local_results_before = _locked_local_result_cell_ids(connection, intent, arm_before)
        selected_rows = legacy._rows(connection, intent, lock=True)  # noqa: SLF001
        selected_ids = set(intent.selected_cell_ids)
        unselected_before = {
            row["cell_id"]: dict(row) for row in arm_before if row["cell_id"] not in selected_ids
        }
        if all(
            row["state"] == "accepted" and row["reconciliation_digest"] == intent.sha256
            for row in selected_rows
        ):
            if _state_counts(arm_before) != _expected_post_counts(intent):
                raise rollout_ledger.LedgerError("stored-session subset post-state census drifted")
            receipt = _existing_receipt(connection, intent)
            arm_after = connection.execute(
                "SELECT * FROM rollout_cells ORDER BY cell_id"
            ).fetchall()
            local_results_after = _locked_local_result_cell_ids(connection, intent, arm_after)
            unselected_after = {
                row["cell_id"]: dict(row) for row in arm_after if row["cell_id"] not in selected_ids
            }
            if unselected_after != unselected_before or local_results_after != local_results_before:
                raise rollout_ledger.LedgerError("stored-session subset complement changed")
            return receipt
        if _state_counts(arm_before) != intent.expected_arm_state_counts:
            raise rollout_ledger.LedgerError("stored-session subset pre-state census drifted")
        for row in selected_rows:
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
        post_counts = _expected_post_counts(intent)
        receipt = _subset_receipt(intent)
        for row in selected_rows:
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
                WHERE cell_id = %s AND state = 'retry_review'
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
        arm_after = connection.execute("SELECT * FROM rollout_cells ORDER BY cell_id").fetchall()
        local_results_after = _locked_local_result_cell_ids(connection, intent, arm_after)
        unselected_after = {
            row["cell_id"]: dict(row) for row in arm_after if row["cell_id"] not in selected_ids
        }
        if unselected_after != unselected_before or local_results_after != local_results_before:
            raise rollout_ledger.LedgerError("stored-session subset complement changed")
        if _state_counts(arm_after) != post_counts:
            raise rollout_ledger.LedgerError("stored-session subset post-state census drifted")
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


def _stale_metadata_receipt(intent: ExactStaleSessionMetadataIntent) -> dict[str, Any]:
    post_counts = {state: 0 for state in rollout_ledger.STATES}
    post_counts.update(accepted=9, retry_review=8)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "reviewed_intent_sha256": intent.sha256,
        "evaluation_plan_sha256": intent.evaluation_plan_sha256,
        "source_job_uid_sha256": "sha256:"
        + hashlib.sha256(intent.source_job_uid.encode()).hexdigest(),
        "source_job_terminal_receipt_sha256": intent.source_job_terminal_receipt_sha256,
        "selected_cell_count": 4,
        "source_total_cell_count": 17,
        "prior_arm_state_counts": intent.expected_arm_state_counts,
        "post_arm_state_counts": post_counts,
        "prior_stale_claim_count": 4,
        "source_terminal_stale_active_count": intent.expected_terminal_stale_active_count,
        "source_local_result_count": intent.expected_terminal_local_result_count,
        "post_local_result_count": 14,
        "existing_selected_local_result_count": 3,
        "restored_selected_local_result_count": 1,
        "preserved_unselected_missing_local_result_count": 3,
        "accepted_existing_completed_session_count": 4,
        "nonselected_cell_count": 13,
        "nonselected_cells_preserved": True,
        "action": "restore_one_local_result_and_accept_existing_scored_sessions",
        "model_generation_performed": False,
        "scoring_call_performed": False,
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def _accept_stale_metadata(
    dsn: str,
    *,
    intent: ExactStaleSessionMetadataIntent,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(evidence, dict) or set(evidence) != {"observations", "records"}:
        raise rollout_ledger.LedgerError("stale-session observation package is invalid")
    observations = evidence["observations"]
    records = evidence["records"]
    if (
        not isinstance(observations, list)
        or len(observations) != 4
        or any(not isinstance(row, dict) for row in observations)
        or not isinstance(records, list)
        or len(records) != 4
        or any(not isinstance(row, dict) for row in records)
    ):
        raise rollout_ledger.LedgerError("stale-session observation package is invalid")
    indexed = {row.get("cell_id"): row for row in observations}
    record_index = {row.get("cell_id"): row for row in records}
    selected = set(intent.selected_cell_ids)
    if (
        set(indexed) != selected
        or len(indexed) != 4
        or set(record_index) != selected
        or len(record_index) != 4
    ):
        raise rollout_ledger.LedgerError("stale-session observations differ from intent roster")
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        arm_before = connection.execute(
            "SELECT * FROM rollout_cells ORDER BY cell_id FOR UPDATE"
        ).fetchall()
        expected_ids = selected | set(intent.unselected_cell_ids)
        if (
            len(arm_before) != 17
            or {row["cell_id"] for row in arm_before} != expected_ids
            or any(row["serving_block"] != intent.serving_block for row in arm_before)
        ):
            raise rollout_ledger.LedgerError("stale-session full arm census differs")
        selected_rows = _stale_metadata_rows(connection, intent, lock=True)
        local_before = connection.execute(
            "SELECT * FROM rollout_local_results ORDER BY execution_id FOR UPDATE"
        ).fetchall()
        local_before_ids = {row["cell_id"] for row in local_before}
        if len(local_before_ids) != len(local_before):
            raise rollout_ledger.LedgerError("stale-session local-result roster is ambiguous")
        initially_missing_selected = {
            row["cell_id"]
            for row in intent.selected_cells
            if not row["local_result_initially_present"]
        }
        missing_unselected = set(intent.unselected_missing_local_result_cell_ids)
        expected_pre_local = expected_ids - initially_missing_selected - missing_unselected
        expected_post_local = expected_ids - missing_unselected
        pre_state = _state_counts(arm_before) == intent.expected_arm_state_counts
        post_counts = {state: 0 for state in rollout_ledger.STATES}
        post_counts.update(accepted=9, retry_review=8)
        post_state = _state_counts(arm_before) == post_counts
        terminal_receipt = _stale_metadata_receipt(intent)
        if post_state:
            if (
                any(
                    row["state"] != "accepted" or row["reconciliation_digest"] != intent.sha256
                    for row in selected_rows
                )
                or local_before_ids != expected_post_local
            ):
                raise rollout_ledger.LedgerError("stale-session post-state evidence drifted")
            existing = connection.execute(
                "SELECT receipt_json FROM ledger_reconciliations WHERE kind = %s",
                (RECEIPT_SCHEMA,),
            ).fetchall()
            matches = [json.loads(row["receipt_json"]) for row in existing]
            matches = [row for row in matches if row.get("reviewed_intent_sha256") == intent.sha256]
            if matches != [terminal_receipt]:
                raise rollout_ledger.LedgerError("stale-session terminal receipt is ambiguous")
            return matches[0]
        if not pre_state or local_before_ids != expected_pre_local:
            raise rollout_ledger.LedgerError("stale-session pre-state evidence drifted")
        now = connection.execute("SELECT CURRENT_TIMESTAMP AS now").fetchone()["now"]
        arm_by_id = {row["cell_id"]: row for row in arm_before}
        if any(
            arm_by_id[cell_id]["state"] != "retry_review"
            or arm_by_id[cell_id]["failure_code"]
            != intent.expected_unselected_missing_local_result_failure_code
            or arm_by_id[cell_id]["reconciliation_digest"] is not None
            for cell_id in missing_unselected
        ):
            raise rollout_ledger.LedgerError("stale-session preserved local-result gaps differ")
        complement_before = {
            row["cell_id"]: dict(row) for row in arm_before if row["cell_id"] not in selected
        }
        local_complement_before = [
            dict(row) for row in local_before if row["cell_id"] not in selected
        ]
        descriptors = {row["cell_id"]: row for row in intent.selected_cells}
        for row in selected_rows:
            evidence_row = record_index[row["cell_id"]]
            normalized = evidence_row.get("record")
            descriptor = descriptors[row["cell_id"]]
            if (
                set(evidence_row)
                != {
                    "cell_id",
                    "record",
                    "record_sha256",
                    "initially_present",
                }
                or not isinstance(normalized, dict)
                or type(evidence_row.get("initially_present")) is not bool
                or evidence_row["initially_present"] != descriptor["local_result_initially_present"]
            ):
                raise rollout_ledger.LedgerError("stale-session private record is invalid")
            expected_digest = _body_digest(normalized)
            if evidence_row.get("record_sha256") != expected_digest:
                raise rollout_ledger.LedgerError("stale-session private record digest differs")
            observation = indexed[row["cell_id"]]
            private_row = {
                **row,
                **normalized,
                "local_session_id": normalized["session_id"],
                "record_sha256": expected_digest,
            }
            legacy._validate_cell_observation(private_row, observation)  # noqa: SLF001
            if any(
                (
                    row["state"] != "claimed",
                    row["worker_id"] is None,
                    row["claim_id"] is None,
                    row["lease_expires_at"] is None,
                    row["lease_expires_at"] >= now,
                    row["reconciliation_digest"] is not None,
                    row["session_id"] not in (None, normalized["session_id"]),
                    row["failure_code"] is not None,
                )
            ):
                raise rollout_ledger.LedgerError("stale-session owner is not terminal and stale")
            present = row["local_session_id"] is not None
            if present != evidence_row["initially_present"]:
                raise rollout_ledger.LedgerError("stale-session initial metadata presence drifted")
            if present and (
                legacy._normalized_local_result(row) != normalized  # noqa: SLF001
                or row["record_sha256"] != expected_digest
            ):
                raise rollout_ledger.LedgerError("stale-session indexed metadata drifted")
        missing_cell = next(iter(initially_missing_selected))
        missing_record = record_index[missing_cell]["record"]
        missing_digest = record_index[missing_cell]["record_sha256"]
        if missing_digest != _body_digest(missing_record):
            raise rollout_ledger.LedgerError("stale-session reconstructed record digest differs")
        columns = [*missing_record, "record_sha256"]
        inserted = connection.execute(
            f"""
            INSERT INTO rollout_local_results ({", ".join(columns)}, recorded_at)
            VALUES ({", ".join("%s" for _ in columns)}, CURRENT_TIMESTAMP)
            """,  # noqa: S608
            [*(missing_record[column] for column in missing_record), missing_digest],
        )
        if inserted.rowcount != 1:
            raise rollout_ledger.LedgerError("stale-session metadata restoration lost")
        source_row = next(row for row in selected_rows if row["cell_id"] == missing_cell)
        rollout_postgres._event(  # noqa: SLF001
            connection,
            cell_id=missing_cell,
            name="local_result_metadata_restored",
            from_state="claimed",
            to_state="claimed",
            worker_id=source_row["worker_id"],
            claim_id=source_row["claim_id"],
            detail={
                "reviewed_intent_sha256": intent.sha256,
                "record_sha256": missing_digest,
                "action": "restore_missing_local_result_index_from_exact_private_artifacts",
            },
        )
        for row in selected_rows:
            observation = indexed[row["cell_id"]]
            cell_receipt = legacy._validate_cell_observation(  # noqa: SLF001
                {
                    **row,
                    **record_index[row["cell_id"]]["record"],
                    "local_session_id": record_index[row["cell_id"]]["record"]["session_id"],
                    "record_sha256": record_index[row["cell_id"]]["record_sha256"],
                },
                observation,
            )
            updated = connection.execute(
                """
                UPDATE rollout_cells
                SET state = 'accepted', session_id = %s,
                    completed_at = CURRENT_TIMESTAMP, heartbeat_at = CURRENT_TIMESTAMP,
                    lease_expires_at = NULL, result_class = 'valid',
                    receipt_digest = %s, failure_code = NULL,
                    reconciliation_digest = %s, updated_at = CURRENT_TIMESTAMP
                WHERE cell_id = %s AND state = 'claimed'
                """,
                (
                    observation["session_id"],
                    cell_receipt["receipt_sha256"],
                    intent.sha256,
                    row["cell_id"],
                ),
            )
            if updated.rowcount != 1:
                raise rollout_ledger.LedgerError("stale-session atomic acceptance lost")
            rollout_postgres._event(  # noqa: SLF001
                connection,
                cell_id=row["cell_id"],
                name="stored_scored_session_reconciled",
                from_state="claimed",
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
        arm_after = connection.execute("SELECT * FROM rollout_cells ORDER BY cell_id").fetchall()
        local_after = connection.execute(
            "SELECT * FROM rollout_local_results ORDER BY execution_id"
        ).fetchall()
        complement_after = {
            row["cell_id"]: dict(row) for row in arm_after if row["cell_id"] not in selected
        }
        local_complement_after = [
            dict(row) for row in local_after if row["cell_id"] not in selected
        ]
        if (
            complement_after != complement_before
            or local_complement_after != local_complement_before
            or _state_counts(arm_after) != post_counts
            or {row["cell_id"] for row in local_after} != expected_post_local
            or len(local_after) != 14
        ):
            raise rollout_ledger.LedgerError("stale-session post-state or complement changed")
        connection.execute(
            """
            INSERT INTO ledger_reconciliations (
                receipt_sha256, kind, receipt_json, created_at
            ) VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            """,
            (
                terminal_receipt["receipt_sha256"],
                RECEIPT_SCHEMA,
                json.dumps(terminal_receipt, sort_keys=True, separators=(",", ":")),
            ),
        )
    return terminal_receipt


def accept_roster(
    dsn: str,
    *,
    intent: StoredSessionIntent,
    observations: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    if isinstance(intent, ExactStaleSessionMetadataIntent):
        if not isinstance(observations, dict):
            raise rollout_ledger.LedgerError("stale-session observation package is invalid")
        return _accept_stale_metadata(dsn, intent=intent, evidence=observations)
    if isinstance(intent, ExactStoredSessionSubsetIntent):
        return _accept_subset_roster(dsn, intent=intent, observations=observations)
    return _accept_full_roster(dsn, intent=intent, observations=observations)


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
        "model_generation_performed": False,
        "scoring_call_performed": False,
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    if isinstance(intent, ExactStaleSessionMetadataIntent):
        terminal.update(
            action="restore_one_local_result_and_accept_existing_scored_sessions",
            restored_selected_local_result_count=1,
            mixed_source_agent_terminations_bound=True,
        )
    else:
        terminal.update(
            source_agent_exit_code=intent.expected_agent_exit_code,
            source_agent_termination=intent.expected_agent_termination,
        )
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
