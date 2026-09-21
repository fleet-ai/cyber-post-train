"""Read-only preflight for a unique seed-44 Stage B successor.

The predecessor Stage B controller can fail before it creates its output root.
This check proves that such a failure did not bind or claim the two reviewed
cells, that no exact authoritative session appeared, and that both predecessor
and successor output roots are absent.  It never applies the recovery intent,
claims a cell, generates a model response, or invokes scoring.
"""

from __future__ import annotations

import argparse
import hashlib
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
    stored_session_reconciliation,
)
from evals.fleet import opencode_self_hosted as self_hosted

SCHEMA = "qwen38-fleet-seed44-stageb-successor-preflight-v1"


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _receipt(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256"),
    }


def run(
    *,
    evaluation_directory: Path,
    predecessor_output_root: Path,
    successor_output_root: Path,
    admin_dsn: str,
    database: str,
    intent_path: Path,
    harness_tar: Path,
    harness_tar_sha256: str,
    harness_receipt: Path,
    harness_receipt_sha256: str,
) -> dict[str, Any]:
    for label, path in (
        ("predecessor", predecessor_output_root),
        ("successor", successor_output_root),
    ):
        if path.exists():
            raise FileExistsError(f"{label} output root already exists")

    intent = reviewed_recovery_v2.load_intent(intent_path)
    if len(intent.selected_cell_ids) != 2:
        raise rollout_ledger.LedgerError("successor requires the exact two-cell roster")
    dsn = stored_session_reconciliation.dedicated_dsn(admin_dsn, database)
    plan, proof = evaluate.checked_preflight(evaluation_directory)
    if intent.evaluation_plan_sha256 != plan["sha256"]:
        raise rollout_ledger.LedgerError("successor intent differs from frozen plan")
    ledger_proof = rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    if _file_sha256(harness_tar) != harness_tar_sha256:
        raise ValueError("harness image archive digest differs")
    if _file_sha256(harness_receipt) != harness_receipt_sha256:
        raise ValueError("harness build receipt digest differs")

    plan["task_bindings"] = proof["task_bindings"]
    endpoint = plan["routes"][intent.serving_block]
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise ValueError("FLEET_API_KEY is required")
    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=60,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")  # noqa: SLF001
        if (
            account.get("team_name") != "fleet"
            or account.get("team_id") != self_hosted.FLEET_TEAM_ID
        ):
            raise rollout_ledger.LedgerError("Fleet team identity required")
        route = evaluate.check_route(endpoint, plan["models"][endpoint["model"]], client)
        first = reviewed_recovery_v2.observe_eligibility(
            dsn,
            intent=intent,
            plan=plan,
            evaluation_directory=evaluation_directory,
            client=client,
        )
        second = reviewed_recovery_v2.observe_eligibility(
            dsn,
            intent=intent,
            plan=plan,
            evaluation_directory=evaluation_directory,
            client=client,
        )
    if first != second:
        raise rollout_ledger.LedgerError("successor eligibility evidence changed")

    with rollout_postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        rows = reviewed_recovery_v2._rows(connection, intent, lock=False)  # noqa: SLF001
        selected_local_results = connection.execute(
            "SELECT COUNT(*) AS count FROM rollout_local_results WHERE cell_id = ANY(%s::text[])",
            (list(intent.selected_cell_ids),),
        ).fetchone()["count"]
        apply_receipts = connection.execute(
            "SELECT receipt_json FROM ledger_reconciliations WHERE kind = %s",
            (reviewed_recovery_v2.APPLY_SCHEMA,),
        ).fetchall()
    matching_apply_receipts = sum(
        json.loads(row["receipt_json"]).get("reviewed_intent_sha256") == intent.sha256
        for row in apply_receipts
    )
    if any(
        row["state"] != "retry_review"
        or int(row["retry_count"]) != 0
        or int(row["max_retries"]) != 1
        or row["reconciliation_digest"] is not None
        or row["session_id"] is not None
        or row["receipt_digest"] is not None
        for row in rows
    ):
        raise rollout_ledger.LedgerError("successor database evidence drifted")
    if selected_local_results != 0 or matching_apply_receipts != 0:
        raise rollout_ledger.LedgerError("predecessor mutated the reviewed recovery roster")

    summary = rollout_postgres.summary(dsn)
    expected_states = {state: 0 for state in rollout_ledger.STATES}
    expected_states.update({"accepted": 15, "retry_review": 2})
    if any(
        (
            summary["total"] != 17,
            summary["local_results"] != 15,
            summary["by_state"] != expected_states,
            summary["stale_active"] != 0,
            summary["plan_sha256"] != ledger_proof["plan_sha256"],
        )
    ):
        raise rollout_ledger.LedgerError("successor ledger census differs")

    body = {
        "schema": SCHEMA,
        "evaluation_plan_sha256": "sha256:" + plan["sha256"],
        "ledger_plan_sha256": "sha256:" + ledger_proof["plan_sha256"],
        "reviewed_intent_sha256": "sha256:" + intent.sha256,
        "selected_cell_count": 2,
        "ledger_census": {
            "total": 17,
            "accepted": 15,
            "retry_review": 2,
            "local_results": 15,
            "active": 0,
            "stale_active": 0,
        },
        "selected_roster": {
            "retry_review": 2,
            "retry_count_zero": 2,
            "reconciliation_digest_null": 2,
            "session_id_null": 2,
            "receipt_digest_null": 2,
            "local_result_count": 0,
            "matching_recovery_apply_receipt_count": 0,
            "exact_authoritative_session_count": 0,
            "model_execution_artifact_count": 0,
            "scoring_artifact_count": 0,
        },
        "predecessor_output_absent": True,
        "successor_output_absent": True,
        "two_identical_eligibility_observations": True,
        "route_profile_sha256": "sha256:" + route["profile_sha256"],
        "route_ready": route["ready"],
        "ledger_mutations": 0,
        "model_generation_calls": 0,
        "scoring_calls": 0,
        "accepted_cell_replays": 0,
        "scores_included": False,
        "prompts_responses_flags_rewards_or_traces_included": False,
        "cell_task_session_or_trace_identifiers_included": False,
    }
    return _receipt(body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--predecessor-output-root", type=Path, required=True)
    parser.add_argument("--successor-output-root", type=Path, required=True)
    parser.add_argument("--postgres-admin-dsn-env", required=True)
    parser.add_argument("--postgres-database", required=True)
    parser.add_argument("--intent", type=Path, required=True)
    parser.add_argument("--harness-tar", type=Path, required=True)
    parser.add_argument("--harness-tar-sha256", required=True)
    parser.add_argument("--harness-receipt", type=Path, required=True)
    parser.add_argument("--harness-receipt-sha256", required=True)
    args = parser.parse_args()
    admin_dsn = os.environ.get(args.postgres_admin_dsn_env, "")
    if not admin_dsn:
        parser.error(f"{args.postgres_admin_dsn_env} is required")
    try:
        result = run(
            evaluation_directory=args.evaluation_directory,
            predecessor_output_root=args.predecessor_output_root,
            successor_output_root=args.successor_output_root,
            admin_dsn=admin_dsn,
            database=args.postgres_database,
            intent_path=args.intent,
            harness_tar=args.harness_tar,
            harness_tar_sha256=args.harness_tar_sha256,
            harness_receipt=args.harness_receipt,
            harness_receipt_sha256=args.harness_receipt_sha256,
        )
    except BaseException as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "accepted": False,
                    "failure_type": type(exc).__name__.lower(),
                    "failure_message_sha256": hashlib.sha256(str(exc).encode()).hexdigest(),
                    "ledger_mutations": 0,
                    "model_generation_calls": 0,
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
