"""In-cluster entrypoint for one authorized visible-action collection Job.

The outer launcher proves and creates one CPU-only Kubernetes Job.  This
entrypoint then performs the scientific mutations in their required order:

1. verify and stage the exact local Docker images;
2. exclusively create the authorized SFS operation root;
3. complete the Fleet-team, route and task-binding preflight;
4. durably record the collection create intent;
5. create one dedicated PostgreSQL database and initialize its bound ledger;
6. execute the frozen cell universe once; and
7. write a content-free aggregate terminal receipt.

It never prints prompts, responses, traces, flags, scores, credentials, or
private record paths.  An exception is reduced to its class name at the CLI
boundary so Kubernetes logs cannot accidentally become a private trace sink.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import digest
from evals.fleet import cluster_entry
from evals.fleet import evaluate
from evals.fleet import rollout_postgres
from evals.fleet import rollout_worker
from evals.fleet import visible_action_collection_v2 as collection

TERMINAL_SCHEMA = "cyber_fleet_visible_action_collection_job_terminal_v1"
TERMINAL_FILE = "COLLECTION_JOB_TERMINAL.json"
ACTIVE_STATES = {"pending", "claimed", "running", "grading"}


def _file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def _safe_summary(value: object, *, planned_cells: int, plan_sha256: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "total",
        "local_results",
        "by_state",
        "by_serving_block",
        "stale_active",
        "plan_sha256",
    }:
        raise RuntimeError("collection ledger summary has an unexpected shape")
    if (
        value.get("total") != planned_cells
        or value.get("plan_sha256") != plan_sha256
        or type(value.get("local_results")) is not int
        or not 0 <= value["local_results"] <= planned_cells
        or type(value.get("stale_active")) is not int
        or value["stale_active"] < 0
    ):
        raise RuntimeError("collection ledger summary differs from the authorized plan")
    by_state = value.get("by_state")
    if not isinstance(by_state, dict) or any(
        not isinstance(state, str) or type(count) is not int or count < 0
        for state, count in by_state.items()
    ):
        raise RuntimeError("collection ledger state census is invalid")
    if sum(by_state.values()) != planned_cells:
        raise RuntimeError("collection ledger state census is incomplete")
    by_block = value.get("by_serving_block")
    if not isinstance(by_block, list) or any(
        not isinstance(row, dict)
        or set(row) != {"serving_block", "state", "count"}
        or not isinstance(row["serving_block"], str)
        or not isinstance(row["state"], str)
        or type(row["count"]) is not int
        or row["count"] < 0
        for row in by_block
    ):
        raise RuntimeError("collection ledger route census is invalid")
    return {
        "total": planned_cells,
        "local_results": value["local_results"],
        "by_state": dict(sorted(by_state.items())),
        "by_serving_block": sorted(
            by_block, key=lambda row: (row["serving_block"], row["state"])
        ),
        "stale_active": value["stale_active"],
        "plan_sha256": plan_sha256,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    authorization_path = Path(args.authorization)
    private_root = Path(args.private_root)
    config = evaluate.read_mapping(config_path)
    authorization_input = evaluate.read_mapping(authorization_path)
    plan = collection.compile_eval(config, relative_to=config_path.resolve().parent)
    authorization = collection.validate_operation_authorization(authorization_input, plan)
    if args.limit != plan["planned_cells"]:
        raise ValueError("Job must execute the complete authorized cell universe")
    if args.route not in plan["routes"]:
        raise ValueError("Job route is absent from the authorized plan")
    if _file_sha256(Path(args.harness_tar)) != args.harness_tar_sha256:
        raise ValueError("harness image archive digest differs")

    cluster_entry.stage_images(
        config=config,
        harness_tar=Path(args.harness_tar),
        harness_receipt=Path(args.harness_receipt),
        receipt_sha256=args.harness_receipt_sha256,
    )
    prepared = collection.prepare(
        config,
        authorization,
        private_root,
        relative_to=config_path.resolve().parent,
    )
    directory = Path(prepared["prepared"])
    preflight = collection.preflight(directory)

    admin_dsn = os.environ.get("ROLLOUT_DATABASE_URL")
    if not admin_dsn:
        raise ValueError("ROLLOUT_DATABASE_URL is required")
    dedicated_dsn = cluster_entry.dedicated_dsn(admin_dsn, args.database)

    def initialize(
        supplied_admin_dsn: str,
        plan_path: Path,
        exact_authorization: dict[str, Any],
    ) -> dict[str, Any]:
        observed = cluster_entry.create_database_once(supplied_admin_dsn, args.database)
        if observed != dedicated_dsn:
            raise RuntimeError("dedicated database identity changed")
        return collection._initialize_authorized_ledger(  # noqa: SLF001
            dedicated_dsn, plan_path, exact_authorization
        )

    ledger_receipt = collection.initialize_once(
        directory,
        dsn=admin_dsn,
        initializer=initialize,
    )
    result = collection.run(
        directory,
        dsn=dedicated_dsn,
        route=args.route,
        worker_id=args.worker_id,
        limit=args.limit,
    )
    summary = _safe_summary(
        rollout_postgres.summary(dedicated_dsn),
        planned_cells=plan["planned_cells"],
        plan_sha256=ledger_receipt["ledger_plan_sha256"],
    )
    unresolved = sum(summary["by_state"].get(state, 0) for state in ACTIVE_STATES)
    body = {
        "schema": TERMINAL_SCHEMA,
        "campaign_id": authorization["campaign_id"],
        "plan_sha256": authorization["plan_sha256"],
        "planned_cell_universe_sha256": authorization["planned_cell_universe_sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
        "database": args.database,
        "preflight_receipt_sha256": "sha256:" + preflight["receipt_sha256"],
        "ledger_receipt_sha256": ledger_receipt["sha256"],
        "planned_cells": plan["planned_cells"],
        "accepted_cells": result["accepted"],
        "ledger_summary": summary,
        "unresolved_cells": unresolved,
        "model_calls_not_counted_or_exposed": True,
        "score_read_or_generated": False,
        "private_content_included": False,
    }
    receipt = {**body, "sha256": "sha256:" + digest(body)}
    rollout_worker._safe_write_once(directory / TERMINAL_FILE, receipt)  # noqa: SLF001
    if unresolved:
        raise RuntimeError("collection ended with unresolved ledger cells")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--harness-tar", required=True)
    parser.add_argument("--harness-tar-sha256", required=True)
    parser.add_argument("--harness-receipt", required=True)
    parser.add_argument("--harness-receipt-sha256", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--limit", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        receipt = execute(parse_args())
    except BaseException as error:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "schema": "cyber_fleet_visible_action_collection_job_failure_v1",
                    "status": "failed_no_automatic_retry",
                    "failure_code": type(error).__name__.lower(),
                    "private_content_included": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "status": "terminal",
                "planned_cells": receipt["planned_cells"],
                "accepted_cells": receipt["accepted_cells"],
                "unresolved_cells": receipt["unresolved_cells"],
                "receipt_sha256": receipt["sha256"],
                "private_content_included": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
