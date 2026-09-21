"""Seal one score-blind lineage receipt after a narrow held-out repair."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import reviewed_recovery_v2, rollout_ledger, rollout_postgres

LINEAGE_SCHEMA = "fleet-heldout-narrow-repair-lineage-v1"


def _job_uid_sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def finalize(
    dsn: str,
    *,
    intent: reviewed_recovery_v2.ProvisioningTimeoutIntent,
) -> dict[str, Any]:
    """Require the original 10 + reconciled 5 + rerolled 2, then seal once."""

    with rollout_postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute("SET LOCAL statement_timeout = '30s'")
        connection.execute("SET LOCAL lock_timeout = '5s'")
        metadata = connection.execute(
            "SELECT value FROM ledger_metadata WHERE key = 'plan_sha256'"
        ).fetchone()
        states = {
            row["state"]: int(row["count"])
            for row in connection.execute(
                "SELECT state, COUNT(*) AS count FROM rollout_cells GROUP BY state"
            ).fetchall()
        }
        counts = connection.execute(
            """
            SELECT
              COUNT(*) AS total,
              COUNT(*) FILTER (WHERE state = 'accepted') AS accepted,
              COUNT(*) FILTER (
                WHERE state = 'accepted' AND reconciliation_digest = %s
              ) AS reconciled_existing,
              COUNT(*) FILTER (
                WHERE state = 'accepted' AND reconciliation_digest = %s
                  AND retry_count = 1
              ) AS rerolled_once,
              COUNT(*) FILTER (
                WHERE state = 'accepted'
                  AND reconciliation_digest IS NULL
                  AND retry_count = 0
              ) AS preserved_original
            FROM rollout_cells
            """,
            (
                intent.prior_stored_session_intent_sha256,
                intent.sha256,
            ),
        ).fetchone()
        local_results = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_local_results"
        ).fetchone()["count"]
        repair_events = {
            row["event"]: int(row["count"])
            for row in connection.execute(
                """
                SELECT event, COUNT(*) AS count FROM rollout_events
                WHERE event IN (
                  'stored_scored_session_reconciled',
                  'reviewed_provisioning_timeout_recovery_claimed'
                )
                GROUP BY event
                """
            ).fetchall()
        }
        receipts = connection.execute(
            "SELECT kind, receipt_json FROM ledger_reconciliations"
        ).fetchall()
        stored_receipts = [
            json.loads(row["receipt_json"])
            for row in receipts
            if row["kind"] == "fleet-stored-session-reconciliation-v2"
            and json.loads(row["receipt_json"]).get("reviewed_intent_sha256")
            == intent.prior_stored_session_intent_sha256
        ]
        apply_receipts = [
            json.loads(row["receipt_json"])
            for row in receipts
            if row["kind"] == reviewed_recovery_v2.APPLY_SCHEMA
            and json.loads(row["receipt_json"]).get("reviewed_intent_sha256") == intent.sha256
        ]
        if any(
            (
                metadata is None or metadata["value"] != intent.evaluation_plan_sha256,
                counts["total"] != 17,
                counts["accepted"] != 17,
                states != {"accepted": 17},
                int(local_results) != 17,
                counts["preserved_original"] != 10,
                counts["reconciled_existing"] != 5,
                counts["rerolled_once"] != 2,
                repair_events.get("stored_scored_session_reconciled") != 5,
                repair_events.get("reviewed_provisioning_timeout_recovery_claimed") != 2,
                len(stored_receipts) != 1,
                len(apply_receipts) != 1,
            )
        ):
            raise rollout_ledger.LedgerError("narrow repair has not produced one complete arm")
        body = {
            "schema_version": LINEAGE_SCHEMA,
            "status": "accepted",
            "evaluation_plan_sha256": intent.evaluation_plan_sha256,
            "source_job_uid_sha256": _job_uid_sha256(intent.source_job_uid),
            "source_job_terminal_receipt_sha256": (intent.source_job_terminal_receipt_sha256),
            "source_terminal_counts": {
                "accepted": 10,
                "retry_review": 7,
                "local_results": 15,
            },
            "repair": {
                "accepted_existing_scored_sessions": 5,
                "new_rollouts": 2,
                "new_rollouts_retry_count_each": 1,
                "stored_session_intent_sha256": intent.prior_stored_session_intent_sha256,
                "stored_session_receipt_sha256": stored_receipts[0]["receipt_sha256"],
                "rollout_recovery_intent_sha256": intent.sha256,
                "rollout_recovery_apply_receipt_sha256": apply_receipts[0]["receipt_sha256"],
            },
            "final_counts": {
                "total": 17,
                "accepted": 17,
                "preserved_original_accepted": 10,
                "accepted_existing_scored_session": 5,
                "single_rollout_repair": 2,
                "local_results": 17,
                "unresolved": 0,
            },
            "policy": {
                "accepted_cells_replayed": 0,
                "stored_sessions_rescored": 0,
                "stored_sessions_regenerated": 0,
                "no_session_cells_rerolled_more_than_once": 0,
                "final_eight_accessed": False,
            },
            "privacy": {
                "score_values_included": False,
                "cell_task_session_or_trace_identifiers_included": False,
                "prompt_response_flag_reward_or_trace_content_included": False,
            },
        }
        receipt = {**body, "receipt_sha256": crypto.digest_without(body, "receipt_sha256")}
        existing = connection.execute(
            "SELECT kind, receipt_json FROM ledger_reconciliations WHERE receipt_sha256 = %s",
            (receipt["receipt_sha256"],),
        ).fetchone()
        if existing is not None:
            if (
                existing["kind"] != LINEAGE_SCHEMA
                or json.loads(existing["receipt_json"]) != receipt
            ):
                raise rollout_ledger.LedgerError("narrow repair lineage receipt differs")
            return receipt
        connection.execute(
            """
            INSERT INTO ledger_reconciliations (
                receipt_sha256, kind, receipt_json, created_at
            ) VALUES (%s, %s, %s, clock_timestamp())
            """,
            (
                receipt["receipt_sha256"],
                LINEAGE_SCHEMA,
                json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            ),
        )
    return receipt
