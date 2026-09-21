"""Run one exact privately reviewed recovery against a prepared Fleet evaluation."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import (
    evaluate,
    reviewed_recovery,
    rollout_ledger,
    rollout_postgres,
    rollout_worker,
)


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


def run(
    *,
    evaluation_directory: Path,
    output_root: Path,
    dsn: str,
    intent_path: Path,
    worker_id: str,
) -> dict[str, Any]:
    evaluate._name(worker_id)  # noqa: SLF001
    if output_root.exists():
        raise FileExistsError("create-once recovery output already exists")
    intent = reviewed_recovery.load_intent(intent_path)
    plan, proof = evaluate.checked_preflight(evaluation_directory)
    if intent.evaluation_plan_sha256 != plan["sha256"]:
        raise ValueError("reviewed recovery intent differs from the frozen evaluation plan")
    if intent.serving_block not in plan["routes"]:
        raise ValueError("reviewed recovery route is absent from the frozen plan")
    limit = len(intent.selected_cell_ids)
    route_limit = len(plan["routes"][intent.serving_block]["task_versions"]) * plan["pass_k"]
    if limit > route_limit:
        raise ValueError("reviewed recovery roster exceeds the frozen route")
    rollout_postgres.verify_plan(dsn, evaluation_directory / "plan.csv")
    evaluate.check_images(plan)
    plan["task_bindings"] = proof["task_bindings"]
    endpoint = plan["routes"][intent.serving_block]
    api_key = os.environ.get("FLEET_API_KEY")
    if not api_key:
        raise ValueError("FLEET_API_KEY is required")
    with httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        timeout=60,
    ) as client:
        evaluate.check_route(endpoint, plan["models"][endpoint["model"]], client)
    universe_index = {}
    for row in rollout_ledger._plan_rows(evaluation_directory / "plan.csv"):  # noqa: SLF001
        cell_id = "sha256:" + digest(row)
        universe_index[(row["model_id"], row["task_version_id"], row["attempt"])] = {
            **row,
            "initial_execution": {
                "cell_id": cell_id,
                "execution_id": "sha256:" + digest({"cell": cell_id, "generation": 1}),
                "execution_generation": 1,
            },
        }
    os.environ["AGENT_HARNESS_IMAGE"] = plan["images"]["agent"]
    os.environ["FIXED_PROXY_IMAGE"] = plan["images"]["proxy"]
    output_root.mkdir(parents=True, exist_ok=False)
    rollout_worker._safe_write_once(  # noqa: SLF001
        output_root / f"STARTED-{worker_id}.json",
        {
            "plan_sha256": plan["sha256"],
            "reviewed_intent_sha256": intent.sha256,
            "selected_cell_count": limit,
            "scores_included": False,
            "prompts_or_traces_included": False,
        },
    )
    apply_receipt = reviewed_recovery.apply_intent(dsn, intent=intent)
    ledger = reviewed_recovery.Ledger(intent)

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

    with ThreadPoolExecutor(max_workers=min(limit, plan["concurrency"])) as pool:
        results = list(pool.map(one, range(limit)))
    body = {
        "schema_version": rollout_worker.TERMINAL_SCHEMA,
        "worker_id": worker_id,
        "plan_sha256": plan["sha256"],
        "reviewed_intent_sha256": intent.sha256,
        "apply_receipt_sha256": apply_receipt["receipt_sha256"],
        "requested_cells": limit,
        "accepted_cells": sum(row.get("accepted") is True for row in results),
        "accepted": bool(results) and all(row.get("accepted") is True for row in results),
        "results": results,
        "ledger_status": rollout_postgres.summary(dsn),
        "scores_included": False,
        "prompts_or_traces_included": False,
        "private_cell_task_session_or_trace_identifiers_included": False,
    }
    terminal = output_root / f"TERMINAL-{worker_id}.json"
    rollout_worker._safe_write_once(terminal, body)  # noqa: SLF001
    return rollout_worker._load(terminal)  # noqa: SLF001


def _failure(worker_id: str) -> dict[str, Any]:
    return {
        "schema_version": rollout_worker.TERMINAL_SCHEMA,
        "accepted": False,
        "worker_id": worker_id if worker_id.isidentifier() else "invalid-worker-id",
        "controller_failure_code": "reviewed_recovery_startup_failed",
        "scores_included": False,
        "prompts_or_traces_included": False,
        "private_cell_task_session_or_trace_identifiers_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--postgres-dsn-env", required=True)
    parser.add_argument("--reviewed-recovery-intent", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    dsn = os.environ.get(args.postgres_dsn_env, "")
    if not dsn:
        parser.error(f"{args.postgres_dsn_env} is required")
    try:
        evaluate._name(args.worker_id)  # noqa: SLF001
        result = run(
            evaluation_directory=args.evaluation_directory,
            output_root=args.output_root,
            dsn=dsn,
            intent_path=args.reviewed_recovery_intent,
            worker_id=args.worker_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return int(any("controller_failure_code" in row for row in result["results"]))
    except BaseException:  # noqa: BLE001
        body = _failure(args.worker_id)
        print(json.dumps(body, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
