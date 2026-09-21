"""Atomic database guard for privately reviewed Fleet rollout recoveries.

This module is deliberately outside ``evaluate.RUNTIME_FILES``. It does not
change a frozen evaluator plan and is not an automatic-retry mechanism. A
reviewer creates one private, self-digesting intent whose digest commits the
complete approved cell roster. Applying the intent leaves those cells in
``retry_review``, where ordinary workers cannot claim them. A recovery claim
then locks the complete roster, checks that digest, records sanitized evidence,
and moves one approved cell directly from ``retry_review`` to ``claimed`` in
the same bounded transaction.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import retry_review_policy, rollout_ledger, rollout_postgres

INTENT_SCHEMA = "fleet-reviewed-recovery-intent-v1"
APPLY_SCHEMA = "fleet-reviewed-recovery-apply-v1"
PRECLAIM_SCHEMA = "fleet-reviewed-recovery-preclaim-v1"
RUNTIME_FILES = (
    "reviewed_recovery.py",
    "reviewed_recovery_worker.py",
    "retry_review_policy.py",
)
INTENT_FIELDS = {
    "schema_version",
    "evaluation_plan_sha256",
    "runtime_files_sha256",
    "serving_block",
    "selected_cell_ids",
    "sha256",
}


def _body_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_identity() -> dict[str, str]:
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in RUNTIME_FILES}


@dataclass(frozen=True)
class ReviewedRecoveryIntent:
    """Private exact roster produced by one completed score-blind review."""

    evaluation_plan_sha256: str
    runtime_files_sha256: dict[str, str]
    serving_block: str
    selected_cell_ids: tuple[str, ...]
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.evaluation_plan_sha256, str):
            raise rollout_ledger.LedgerError("evaluation plan sha256 must be a SHA-256 digest")
        plan_sha256 = rollout_ledger._require_digest(  # noqa: SLF001
            self.evaluation_plan_sha256, "evaluation plan sha256"
        )
        if plan_sha256 != self.evaluation_plan_sha256:
            raise rollout_ledger.LedgerError("evaluation plan sha256 must be canonical")
        if not isinstance(self.runtime_files_sha256, dict):
            raise rollout_ledger.LedgerError("reviewed recovery runtime identity is malformed")
        runtime = runtime_identity()
        if self.runtime_files_sha256 != runtime:
            raise rollout_ledger.LedgerError("reviewed recovery runtime identity differs")
        route = rollout_ledger._require_text(self.serving_block, "serving_block")  # noqa: SLF001
        if not self.selected_cell_ids:
            raise rollout_ledger.LedgerError("reviewed recovery intent must select cells")
        cells: list[str] = []
        for raw in self.selected_cell_ids:
            if not isinstance(raw, str):
                raise rollout_ledger.LedgerError("reviewed recovery cell identity is malformed")
            try:
                parsed = str(uuid.UUID(raw))
            except ValueError as exc:
                raise rollout_ledger.LedgerError(
                    "reviewed recovery cell identity is malformed"
                ) from exc
            if parsed != raw:
                raise rollout_ledger.LedgerError("reviewed recovery cell identity is not canonical")
            cells.append(raw)
        if len(cells) != len(set(cells)):
            raise rollout_ledger.LedgerError("reviewed recovery intent repeats a cell")
        if not isinstance(self.sha256, str):
            raise rollout_ledger.LedgerError("intent sha256 must be a SHA-256 digest")
        intent_sha256 = rollout_ledger._require_digest(self.sha256, "intent sha256")  # noqa: SLF001
        expected = _body_digest(
            {
                "schema_version": INTENT_SCHEMA,
                "evaluation_plan_sha256": self.evaluation_plan_sha256,
                "runtime_files_sha256": self.runtime_files_sha256,
                "serving_block": self.serving_block,
                "selected_cell_ids": list(self.selected_cell_ids),
            }
        )
        if intent_sha256 != expected:
            raise rollout_ledger.LedgerError("reviewed recovery intent self digest differs")
        object.__setattr__(self, "evaluation_plan_sha256", plan_sha256)
        object.__setattr__(self, "runtime_files_sha256", dict(runtime))
        object.__setattr__(self, "serving_block", route)
        object.__setattr__(self, "selected_cell_ids", tuple(cells))
        object.__setattr__(self, "sha256", intent_sha256)


def load_intent(path: Path) -> ReviewedRecoveryIntent:
    """Load one strict private intent without exposing its cell roster."""

    try:
        if path.stat().st_mode & 0o077:
            raise rollout_ledger.LedgerError(
                "reviewed recovery intent must not be readable by group or other"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except rollout_ledger.LedgerError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise rollout_ledger.LedgerError("reviewed recovery intent is unreadable") from exc
    if not isinstance(value, dict) or set(value) != INTENT_FIELDS:
        raise rollout_ledger.LedgerError("reviewed recovery intent has missing or unknown fields")
    if value["schema_version"] != INTENT_SCHEMA:
        raise rollout_ledger.LedgerError("reviewed recovery intent schema is unsupported")
    if not isinstance(value["selected_cell_ids"], list):
        raise rollout_ledger.LedgerError("reviewed recovery cell roster is malformed")
    return ReviewedRecoveryIntent(
        evaluation_plan_sha256=value["evaluation_plan_sha256"],
        runtime_files_sha256=value["runtime_files_sha256"],
        serving_block=value["serving_block"],
        selected_cell_ids=tuple(value["selected_cell_ids"]),
        sha256=value["sha256"],
    )


def _locked_rows(connection: Any, intent: ReviewedRecoveryIntent) -> list[dict[str, Any]]:
    connection.execute("SET LOCAL statement_timeout = '30s'")
    connection.execute("SET LOCAL lock_timeout = '5s'")
    rows = connection.execute(
        """
        SELECT * FROM rollout_cells
        WHERE cell_id = ANY(%s::text[])
        ORDER BY cell_id
        FOR UPDATE
        """,
        (list(intent.selected_cell_ids),),
    ).fetchall()
    observed = {str(row["cell_id"]) for row in rows}
    if len(rows) != len(intent.selected_cell_ids) or observed != set(intent.selected_cell_ids):
        raise rollout_ledger.LedgerError("database differs from the reviewed recovery roster")
    if any(row["serving_block"] != intent.serving_block for row in rows):
        raise rollout_ledger.LedgerError("database route differs from reviewed recovery intent")
    return rows


def _apply_evidence(intent: ReviewedRecoveryIntent) -> dict[str, Any]:
    body = {
        "schema_version": APPLY_SCHEMA,
        "reviewed_intent_sha256": intent.sha256,
        "selected_cell_count": len(intent.selected_cell_ids),
        "rows_remain_quarantined_from_ordinary_claims": True,
        "local_result_count_at_apply": 0,
        "allowed_failure_codes": sorted(retry_review_policy.ROLLOUT_RETRY_FAILURE_CODES),
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def apply_intent(dsn: str, *, intent: ReviewedRecoveryIntent) -> dict[str, Any]:
    """Bind a complete reviewed roster without exposing it to ordinary claims."""

    evidence = _apply_evidence(intent)
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        rows = _locked_rows(connection, intent)
        digests = {row["reconciliation_digest"] for row in rows}
        if digests == {intent.sha256}:
            stored = connection.execute(
                """
                SELECT kind, receipt_json FROM ledger_reconciliations
                WHERE receipt_sha256 = %s
                """,
                (evidence["receipt_sha256"],),
            ).fetchone()
            if stored is None or stored["kind"] != APPLY_SCHEMA:
                raise rollout_ledger.LedgerError("reviewed recovery apply receipt is missing")
            if json.loads(stored["receipt_json"]) != evidence:
                raise rollout_ledger.LedgerError("reviewed recovery apply receipt differs")
            return evidence
        if digests != {None}:
            raise rollout_ledger.LedgerError(
                "database reconciliation digest differs from reviewed recovery intent"
            )
        local_results = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_local_results WHERE cell_id = ANY(%s::text[])",
            (list(intent.selected_cell_ids),),
        ).fetchone()["count"]
        if (
            any(
                row["state"] != "retry_review"
                or int(row["retry_count"]) >= int(row["max_retries"])
                or row["result_class"] != "infrastructure_invalid"
                or row["failure_code"] not in retry_review_policy.ROLLOUT_RETRY_FAILURE_CODES
                or row["session_id"] is not None
                or row["receipt_digest"] is not None
                for row in rows
            )
            or local_results
        ):
            raise rollout_ledger.LedgerError("reviewed recovery roster contains an ineligible cell")
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
            raise rollout_ledger.LedgerError("atomic reviewed recovery apply lost unexpectedly")
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
                name="reviewed_recovery_bound",
                from_state="retry_review",
                to_state="retry_review",
                detail={
                    "reviewed_intent_sha256": intent.sha256,
                    "apply_receipt_sha256": evidence["receipt_sha256"],
                },
            )
    return evidence


def _preclaim_evidence(
    intent: ReviewedRecoveryIntent,
    *,
    claimable_cell_count: int,
    claim_id: str,
) -> dict[str, Any]:
    body = {
        "schema_version": PRECLAIM_SCHEMA,
        "reviewed_intent_sha256": intent.sha256,
        "selected_cell_count": len(intent.selected_cell_ids),
        "matching_reconciliation_digest_count": len(intent.selected_cell_ids),
        "claimable_cell_count_before_claim": claimable_cell_count,
        "claim_id_sha256": "sha256:" + hashlib.sha256(claim_id.encode()).hexdigest(),
        "claim_selected_in_same_transaction": True,
        "score_values_included": False,
        "prompt_response_flag_reward_or_trace_content_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}


def claim(
    dsn: str,
    *,
    intent: ReviewedRecoveryIntent,
    worker_id: str,
    serving_block: str,
    lease_seconds: int = 900,
) -> dict[str, Any] | None:
    """Claim one quarantined reviewed cell after an atomic full-roster check."""

    worker = rollout_ledger._require_text(worker_id, "worker_id")  # noqa: SLF001
    route = rollout_ledger._require_text(serving_block, "serving_block")  # noqa: SLF001
    if route != intent.serving_block:
        raise rollout_ledger.LedgerError("worker route differs from reviewed recovery intent")
    if lease_seconds < 30:
        raise rollout_ledger.LedgerError("lease_seconds must be at least 30")
    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        rows = _locked_rows(connection, intent)
        if any(row["reconciliation_digest"] != intent.sha256 for row in rows):
            raise rollout_ledger.LedgerError(
                "database reconciliation digest differs from reviewed recovery intent"
            )
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
            raise rollout_ledger.LedgerError("atomic reviewed recovery claim lost unexpectedly")
        evidence = _preclaim_evidence(
            intent,
            claimable_cell_count=len(claimable),
            claim_id=claim_id,
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
            name="reviewed_recovery_claimed",
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
    """Ledger adapter for ``rollout_worker.run_one`` reviewed recovery calls."""

    def __init__(self, intent: ReviewedRecoveryIntent) -> None:
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
