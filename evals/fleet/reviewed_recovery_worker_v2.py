"""Run an exact two-cell provisioning-timeout repair and seal its lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import (
    cluster_entry,
    evaluate,
    repair_lineage,
    reviewed_recovery_v2,
    rollout_ledger,
    rollout_postgres,
    rollout_worker,
    stored_session_reconciliation,
)
from evals.fleet import opencode_self_hosted as self_hosted


def _repair_universe_index(
    plan: dict[str, Any], plan_csv: Path
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Use a fresh execution identity while preserving each scientific cell."""

    universe = reviewed_recovery_v2._scientific_index(plan, plan_csv)  # noqa: SLF001
    result = {}
    for key, row in universe.items():
        cell_id = row["initial_execution"]["cell_id"]
        result[key] = {
            **row,
            "initial_execution": {
                "cell_id": cell_id,
                "execution_id": "sha256:" + digest({"cell": cell_id, "generation": 2}),
                "execution_generation": 2,
            },
        }
    return result


def _sanitized_result(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "serving_block",
        "claimed",
        "accepted",
        "failure_code",
        "controller_failure_code",
        "receipt_sha256",
    }
    return {key: value[key] for key in sorted(allowed & value.keys())}


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _stage_and_check_images(
    *,
    plan: dict[str, Any],
    harness_tar: Path,
    harness_tar_sha256: str,
    harness_receipt: Path,
    harness_receipt_sha256: str,
) -> None:
    """Populate a fresh DinD daemon, then run the normal image preflight."""

    if _file_sha256(harness_tar) != harness_tar_sha256:
        raise ValueError("harness image archive digest differs")
    cluster_entry.stage_images(
        config={"images": plan["images"], "harness": plan["treatment"]},
        harness_tar=harness_tar,
        harness_receipt=harness_receipt,
        receipt_sha256=harness_receipt_sha256,
    )
    evaluate.check_images(plan)


def run(
    *,
    evaluation_directory: Path,
    output_root: Path,
    admin_dsn: str,
    database: str,
    harness_tar: Path,
    harness_tar_sha256: str,
    harness_receipt: Path,
    harness_receipt_sha256: str,
    intent_path: Path,
    worker_id: str,
) -> dict[str, Any]:
    evaluate._name(worker_id)  # noqa: SLF001
    if output_root.exists():
        raise FileExistsError("create-once reviewed recovery v2 output already exists")
    intent = reviewed_recovery_v2.load_intent(intent_path)
    if database != intent.source_database:
        raise ValueError("runtime database differs from reviewed recovery v2 intent")
    dsn = stored_session_reconciliation.dedicated_dsn(admin_dsn, database)
    plan, proof = evaluate.checked_preflight(evaluation_directory)
    if intent.evaluation_plan_sha256 != plan["sha256"]:
        raise ValueError("reviewed recovery v2 intent differs from the frozen plan")
    if intent.serving_block not in plan["routes"]:
        raise ValueError("reviewed recovery v2 route is absent from the frozen plan")
    limit = len(intent.selected_cell_ids)
    if limit != 2:
        raise ValueError("reviewed recovery v2 requires the exact two-cell roster")
    route_limit = len(plan["routes"][intent.serving_block]["task_versions"]) * plan["pass_k"]
    if limit > route_limit:
        raise ValueError("reviewed recovery v2 roster exceeds the frozen route")
    rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    _stage_and_check_images(
        plan=plan,
        harness_tar=harness_tar,
        harness_tar_sha256=harness_tar_sha256,
        harness_receipt=harness_receipt,
        harness_receipt_sha256=harness_receipt_sha256,
    )
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
        evaluate.check_route(endpoint, plan["models"][endpoint["model"]], client)
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
        raise rollout_ledger.LedgerError("reviewed recovery v2 evidence changed")
    output_root.mkdir(parents=True, exist_ok=False)
    rollout_worker._safe_write_once(  # noqa: SLF001
        output_root / f"STARTED-{worker_id}.json",
        {
            "plan_sha256": plan["sha256"],
            "reviewed_intent_sha256": intent.sha256,
            "selected_cell_count": limit,
            "eligibility_policy": "exact_provisioning_post_504_before_model_or_scoring_v1",
            "scores_included": False,
            "prompts_or_traces_included": False,
        },
    )
    apply_receipt = reviewed_recovery_v2.apply_intent(dsn, intent=intent, observations=second)
    universe_index = _repair_universe_index(plan, evaluation_directory / "plan.csv")
    os.environ["AGENT_HARNESS_IMAGE"] = plan["images"]["agent"]
    os.environ["FIXED_PROXY_IMAGE"] = plan["images"]["proxy"]
    ledger = reviewed_recovery_v2.Ledger(intent)

    def one(index: int) -> dict[str, Any]:
        def guard(client: httpx.Client) -> None:
            evaluate.check_route(endpoint, plan["models"][endpoint["model"]], client)

        result = rollout_worker.run_one(
            database=dsn,
            ledger=ledger,
            campaign=plan,
            selection={"tasks": plan["tasks"]},
            universe_index=universe_index,
            serving_block=intent.serving_block,
            worker_id=f"{worker_id}-{index}",
            output_root=output_root,
            claim_root=output_root / "claims",
            proxy_script=Path(evaluate.__file__).with_name("fixed_proxy.py"),
            pre_execution_guard=guard,
        )
        return _sanitized_result(result)

    with ThreadPoolExecutor(max_workers=limit) as pool:
        results = list(pool.map(one, range(limit)))
    accepted = len(results) == limit and all(row.get("accepted") is True for row in results)
    lineage = repair_lineage.finalize(dsn, intent=intent) if accepted else None
    if lineage is not None:
        self_hosted.write_json_once(output_root / "LINEAGE.json", lineage)
    body = {
        "schema_version": "fleet-reviewed-recovery-controller-terminal-v2",
        "worker_id": worker_id,
        "plan_sha256": plan["sha256"],
        "reviewed_intent_sha256": intent.sha256,
        "apply_receipt_sha256": apply_receipt["receipt_sha256"],
        "requested_cells": limit,
        "accepted_cells": sum(row.get("accepted") is True for row in results),
        "accepted": accepted,
        "results": results,
        "ledger_status": rollout_postgres.summary(dsn),
        "lineage_receipt_sha256": (lineage["receipt_sha256"] if lineage is not None else None),
        "scores_included": False,
        "prompts_or_traces_included": False,
        "private_cell_task_session_or_trace_identifiers_included": False,
    }
    terminal = output_root / f"TERMINAL-{worker_id}.json"
    rollout_worker._safe_write_once(terminal, body)  # noqa: SLF001
    return rollout_worker._load(terminal)  # noqa: SLF001


def _failure(worker_id: str) -> dict[str, Any]:
    return {
        "schema_version": "fleet-reviewed-recovery-controller-terminal-v2",
        "accepted": False,
        "worker_id": worker_id if worker_id.isidentifier() else "invalid-worker-id",
        "controller_failure_code": "reviewed_recovery_v2_startup_failed",
        "scores_included": False,
        "prompts_or_traces_included": False,
        "private_cell_task_session_or_trace_identifiers_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--postgres-admin-dsn-env", required=True)
    parser.add_argument("--postgres-database", required=True)
    parser.add_argument("--harness-tar", type=Path, required=True)
    parser.add_argument("--harness-tar-sha256", required=True)
    parser.add_argument("--harness-receipt", type=Path, required=True)
    parser.add_argument("--harness-receipt-sha256", required=True)
    parser.add_argument("--reviewed-recovery-intent", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
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
            harness_tar=args.harness_tar,
            harness_tar_sha256=args.harness_tar_sha256,
            harness_receipt=args.harness_receipt,
            harness_receipt_sha256=args.harness_receipt_sha256,
            intent_path=args.reviewed_recovery_intent,
            worker_id=args.worker_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("accepted") is True else 1
    except BaseException:  # noqa: BLE001
        body = _failure(args.worker_id)
        print(json.dumps(body, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
