"""Completion-budget-safe successor for visible-action collection.

V1 and v2 remain byte-for-byte historical inputs.  V3 preserves the same
non-thinking OpenCode science and exactly-once operation boundary while binding
the separate chat-completion budget proxy and its content-free output-limit
classification overlay.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import collection_completion_budget as budget
from evals.fleet import evaluate
from evals.fleet import opencode_self_hosted as harness
from evals.fleet import rollout_ledger as ledger
from evals.fleet import rollout_postgres as postgres
from evals.fleet import rollout_worker as worker
from evals.fleet import visible_action_collection_v2 as v2

PLAN_SCHEMA = "cyber_fleet_visible_action_collection_eval_v3"
PREFLIGHT_SCHEMA = "cyber_fleet_visible_action_collection_preflight_v3"
RUNTIME_SCHEMA = "cyber_visible_action_collection_runtime_v3"
THINKING_DISABLED = v2.THINKING_DISABLED
EXECUTION_MODE = v2.EXECUTION_MODE
PROXY_FILE = budget.PROXY_FILE
OPERATION_AUTHORIZATION_FILE = v2.OPERATION_AUTHORIZATION_FILE
CREATE_INTENT_FILE = v2.CREATE_INTENT_FILE
LEDGER_RECEIPT_FILE = v2.LEDGER_RECEIPT_FILE
RUNTIME_FILES = (
    "visible_action_collection.py",
    "visible_action_collection_v2.py",
    "visible_action_collection_v3.py",
    "collection_completion_budget.py",
    PROXY_FILE,
)
RUNTIME_FIELDS = v2.RUNTIME_FIELDS | {"completion_budget"}
COMPLETION_BUDGET_POLICY = {
    "scope": "chat_completions_only_v1",
    "maximum_completions_source": "harness.max_model_requests",
    "discovery_requests_consume_completion_budget": False,
    "total_request_headroom": budget.TOTAL_REQUEST_HEADROOM,
    "exhaustion_evidence_schema": budget.proxy.BUDGET_RECEIPT_SCHEMA,
    "exhaustion_termination": "output_limit",
    "exhausted_trajectory_training_data_eligible": False,
}


def runtime_identity() -> dict[str, Any]:
    return {
        "base_evaluator": evaluate.runtime_identity(),
        "collection_adapter": {
            name: budget.file_sha256(Path(__file__).with_name(name)) for name in RUNTIME_FILES
        },
    }


def _runtime_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUNTIME_FIELDS:
        raise ValueError("collection v3 runtime contract has unknown or missing fields")
    expected = {
        "schema": RUNTIME_SCHEMA,
        "source_template_sha256": value.get("source_template_sha256"),
        "reasoning_generation": THINKING_DISABLED,
        "reasoning_request_override": {"chat_template_kwargs": {"enable_thinking": False}},
        "opencode_model_reasoning": False,
        "opencode_cli_thinking_flag": False,
        "maximum_planned_cells": value.get("maximum_planned_cells"),
        "operation_authorization_required": True,
        "canonical_private_operation_root_required": True,
        "exclusive_pre_mutation_intent_required": True,
        "dedicated_empty_ledger_required": True,
        "automatic_replay_of_ambiguous_cells": False,
        "external_submission": False,
        "execution_mode": EXECUTION_MODE,
        "cluster_wrapper_supported": True,
        "completion_budget": COMPLETION_BUDGET_POLICY,
    }
    if value != expected:
        raise ValueError("collection v3 runtime is not the exact completion-budget rail")
    v2._sha(value["source_template_sha256"], "source template")  # noqa: SLF001
    maximum = value["maximum_planned_cells"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("collection cell ceiling must be a positive integer")
    return copy.deepcopy(value)


def _as_v2_config(config: dict[str, Any]) -> dict[str, Any]:
    converted = copy.deepcopy(config)
    runtime = _runtime_contract(converted["collection_runtime"])
    converted["collection_runtime"] = {
        key: item for key, item in runtime.items() if key != "completion_budget"
    }
    converted["collection_runtime"]["schema"] = v2.RUNTIME_SCHEMA
    return converted


def compile_eval(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    runtime = _runtime_contract(config.get("collection_runtime"))
    plan = v2.compile_eval(_as_v2_config(config), relative_to=relative_to)
    plan.pop("sha256")
    plan["schema"] = PLAN_SCHEMA
    plan["collection_runtime"] = runtime
    plan["runtime_files"] = runtime_identity()
    return {**plan, "sha256": digest(plan)}


plan_rows = v2.plan_rows
cell_universe_sha256 = v2.cell_universe_sha256
identity_map = v2.identity_map
identity_map_sha256 = v2.identity_map_sha256
build_operation_authorization = v2.build_operation_authorization
validate_operation_authorization = v2.validate_operation_authorization
canonical_operation_root = v2.canonical_operation_root


def prepare(
    config: dict[str, Any],
    authorization: dict[str, Any],
    private_root: Path,
    *,
    relative_to: Path,
) -> dict[str, Any]:
    plan = compile_eval(config, relative_to=relative_to)
    authorization = validate_operation_authorization(authorization, plan)
    directory = canonical_operation_root(private_root, authorization)
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError as error:
        raise FileExistsError("canonical collection operation root already exists") from error
    for name in ("claims", "attempts"):
        (directory / name).mkdir(mode=0o700)
    harness.write_json_once(directory / "EVAL.json", plan)
    rows = plan_rows(plan)
    with (directory / "plan.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    normalized = ledger._plan_rows(directory / "plan.csv")  # noqa: SLF001
    observed = []
    for row in normalized:
        scientific = "sha256:" + digest(row)
        observed.append(
            {
                "campaign_id": row["experiment_id"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "model_id": row["model_id"],
                "attempt": row["attempt"],
                "ledger_cell_id": row["cell_id"],
                "scientific_cell_id": scientific,
                "execution_id": "sha256:" + digest({"cell": scientific, "generation": 1}),
            }
        )
    if "sha256:" + digest(observed) != authorization["identity_map_sha256"]:
        raise ValueError("written plan changed the authorized cell identity map")
    v2._write_once_fsynced(directory / OPERATION_AUTHORIZATION_FILE, authorization)  # noqa: SLF001
    return {
        "prepared": str(directory),
        "sessions": len(rows),
        "submitted": False,
        "plan_sha256": plan["sha256"],
        "operation_authorization_sha256": authorization["sha256"],
    }


def load(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = evaluate.read_mapping(directory / "EVAL.json")
    unsigned = {key: item for key, item in plan.items() if key != "sha256"}
    if plan.get("schema") != PLAN_SCHEMA or plan.get("sha256") != digest(unsigned):
        raise ValueError("visible-action collection v3 plan identity changed")
    if plan.get("runtime_files") != runtime_identity():
        raise ValueError("collection v3 runtime changed; prepare a new immutable operation")
    _runtime_contract(plan.get("collection_runtime"))
    authorization = validate_operation_authorization(
        evaluate.read_mapping(directory / OPERATION_AUTHORIZATION_FILE), plan
    )
    if directory.resolve() != canonical_operation_root(directory.parent, authorization):
        raise ValueError("collection operation is outside its canonical private root")
    actual = ledger._plan_rows(directory / "plan.csv")  # noqa: SLF001
    expected = plan_rows(plan)
    fields = list(expected[0])
    normalized = [{key: row[key] for key in fields} for row in actual]
    if normalized != sorted(
        expected,
        key=lambda row: (
            row["experiment_id"],
            row["task_version_id"],
            row["model_id"],
            row["attempt"],
        ),
    ):
        raise ValueError("CSV and visible-action collection v3 plan differ")
    if len(actual) != plan.get("planned_cells"):
        raise ValueError("visible-action collection v3 cell count drift")
    if identity_map_sha256(plan) != authorization["identity_map_sha256"]:
        raise ValueError("collection v3 identity map changed")
    return plan, authorization


def preflight(directory: Path) -> dict[str, Any]:
    plan, authorization = load(directory)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise ValueError("FLEET_API_KEY is required")
    proof = {
        "schema": PREFLIGHT_SCHEMA,
        "plan_sha256": plan["sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "task_bindings": {},
        "routes": {},
        "images": plan["images"],
        "reasoning_generation": THINKING_DISABLED,
        "completion_budget": COMPLETION_BUDGET_POLICY,
        "model_calls": 0,
    }
    evaluate.check_images(plan)
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60) as client:
        account = harness._request(client, "GET", "/v1/account")  # noqa: SLF001
        if account.get("team_name") != "fleet" or account.get("team_id") != harness.FLEET_TEAM_ID:
            raise RuntimeError("Fleet team identity required")
        for block, route in plan["routes"].items():
            proof["routes"][block] = evaluate.check_route(
                route, plan["models"][route["model"]], client
            )
        for task in plan["tasks"]:
            binding = worker._task_binding(client, task, task)  # noqa: SLF001
            if binding[0]["cyber_contract"] != worker.AUTHORITY["required_cyber_contract"]:
                raise RuntimeError("task cyber contract differs")
            proof["task_bindings"][task["task_version_id"]] = list(binding)
    proof["sha256"] = digest(proof)
    harness.write_json_once(directory / "EVAL_PREFLIGHT.json", proof)
    return {
        "status": "passed",
        "tasks": len(plan["tasks"]),
        "routes": len(plan["routes"]),
        "scored_sessions": 0,
        "model_calls": 0,
        "receipt_sha256": proof["sha256"],
    }


def checked_preflight(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, authorization = load(directory)
    proof = evaluate.read_mapping(directory / "EVAL_PREFLIGHT.json")
    unsigned = {key: item for key, item in proof.items() if key != "sha256"}
    if proof.get("plan_sha256") != plan["sha256"] or proof.get("sha256") != digest(unsigned):
        raise ValueError("visible-action collection v3 preflight does not match")
    if (
        proof.get("schema") != PREFLIGHT_SCHEMA
        or proof.get("operation_authorization_sha256") != authorization["sha256"]
        or proof.get("images") != plan["images"]
        or proof.get("reasoning_generation") != THINKING_DISABLED
        or proof.get("completion_budget") != COMPLETION_BUDGET_POLICY
        or proof.get("model_calls") != 0
        or set(proof.get("task_bindings", {}))
        != {task["task_version_id"] for task in plan["tasks"]}
        or set(proof.get("routes", {})) != set(plan["routes"])
    ):
        raise ValueError("visible-action collection v3 preflight is incomplete")
    return plan, authorization, proof


def initialize_once(
    directory: Path,
    *,
    dsn: str,
    initializer: Callable[[str, Path, dict[str, Any]], dict[str, Any]] = (
        v2._initialize_authorized_ledger  # noqa: SLF001
    ),
) -> dict[str, Any]:
    plan, authorization, proof = checked_preflight(directory)
    if not dsn:
        raise ValueError("ROLLOUT_DATABASE_URL is required")
    intent_path = directory / CREATE_INTENT_FILE
    receipt_path_value = directory / LEDGER_RECEIPT_FILE
    if (
        intent_path.exists()
        or intent_path.is_symlink()
        or receipt_path_value.exists()
        or receipt_path_value.is_symlink()
    ):
        raise RuntimeError("collection create intent exists; reconcile, never retry")
    body = {
        "schema": v2.CREATE_INTENT_SCHEMA,
        "state": "CREATE_INTENT_DO_NOT_RETRY",
        "campaign_id": authorization["campaign_id"],
        "plan_sha256": authorization["plan_sha256"],
        "planned_cell_universe_sha256": authorization["planned_cell_universe_sha256"],
        "identity_map_sha256": authorization["identity_map_sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
        "preflight_receipt_sha256": "sha256:" + proof["sha256"],
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    intent = {**body, "sha256": "sha256:" + digest(body)}
    v2._write_once_fsynced(intent_path, intent)  # noqa: SLF001
    created = initializer(dsn, directory / "plan.csv", authorization)
    receipt_body = {
        "schema": v2.LEDGER_RECEIPT_SCHEMA,
        "campaign_id": authorization["campaign_id"],
        "plan_sha256": authorization["plan_sha256"],
        "planned_cell_universe_sha256": authorization["planned_cell_universe_sha256"],
        "identity_map_sha256": authorization["identity_map_sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
        "create_intent_sha256": intent["sha256"],
        "ledger_plan_sha256": created["plan_sha256"],
        "created_cells": created["cells"],
        "created_at": datetime.now(UTC).isoformat(),
    }
    receipt = {**receipt_body, "sha256": "sha256:" + digest(receipt_body)}
    v2._write_once_fsynced(receipt_path_value, receipt)  # noqa: SLF001
    return receipt


_validated_ledger_receipt = v2._validated_ledger_receipt  # noqa: SLF001
_initialize_authorized_ledger = v2._initialize_authorized_ledger  # noqa: SLF001
_verify_authorized_ledger = v2._verify_authorized_ledger  # noqa: SLF001


def run(
    directory: Path,
    *,
    dsn: str,
    route: str,
    worker_id: str,
    limit: int,
) -> dict[str, Any]:
    plan, authorization, proof = checked_preflight(directory)
    _validated_ledger_receipt(directory, plan, authorization)
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", worker_id):
        raise ValueError("safe lowercase worker ID is required")
    if (
        not dsn
        or route not in plan["routes"]
        or type(limit) is not int
        or limit < 1
        or limit > len(plan["routes"][route]["task_versions"]) * plan["pass_k"]
    ):
        raise ValueError("database, route and bounded positive session limit required")
    _verify_authorized_ledger(dsn, directory / "plan.csv", authorization)
    evaluate.check_images(plan)
    worker._safe_write_once(  # noqa: SLF001
        directory / f"STARTED-{worker_id}.json",
        {
            "plan_sha256": plan["sha256"],
            "operation_authorization_sha256": authorization["sha256"],
        },
    )
    plan["task_bindings"] = proof["task_bindings"]
    index = {}
    for row in ledger._plan_rows(directory / "plan.csv"):  # noqa: SLF001
        scientific = "sha256:" + digest(row)
        index[(row["model_id"], row["task_version_id"], row["attempt"])] = {
            **row,
            "initial_execution": {
                "cell_id": scientific,
                "execution_id": "sha256:" + digest({"cell": scientific, "generation": 1}),
                "execution_generation": 1,
            },
        }
    os.environ["AGENT_HARNESS_IMAGE"] = plan["images"]["agent"]
    os.environ["FIXED_PROXY_IMAGE"] = plan["images"]["proxy"]
    budget.activate_runtime()

    def one(index_number: int) -> dict[str, Any]:
        endpoint = plan["routes"][route]

        def guard(client: httpx.Client) -> None:
            evaluate.check_route(endpoint, plan["models"][endpoint["model"]], client)

        return worker.run_one(
            database=dsn,
            ledger=postgres,
            campaign=plan,
            selection={"tasks": plan["tasks"]},
            universe_index=index,
            serving_block=route,
            worker_id=f"{worker_id}-{index_number}",
            output_root=directory,
            claim_root=directory / "claims",
            proxy_script=Path(__file__).with_name(PROXY_FILE),
            pre_execution_guard=guard,
        )

    with ThreadPoolExecutor(max_workers=min(limit, plan["concurrency"])) as pool:
        results = list(pool.map(one, range(limit)))
    terminal = {
        "worker_id": worker_id,
        "plan_sha256": plan["sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "results": results,
        "accepted": sum(result.get("accepted") is True for result in results),
    }
    worker._safe_write_once(directory / f"TERMINAL-{worker_id}.json", terminal)  # noqa: SLF001
    return terminal


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("authorization", type=Path)
    prepare_parser.add_argument("--private-root", type=Path, required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("directory", type=Path)
    init_parser = commands.add_parser("init")
    init_parser.add_argument("directory", type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("directory", type=Path)
    run_parser.add_argument("route")
    run_parser.add_argument("worker_id")
    run_parser.add_argument("--limit", type=int, default=1)
    commands.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            evaluate.read_mapping(args.config),
            evaluate.read_mapping(args.authorization),
            args.private_root,
            relative_to=args.config.resolve().parent,
        )
    elif args.command == "preflight":
        result = preflight(args.directory)
    elif args.command == "init":
        result = initialize_once(args.directory, dsn=os.environ["ROLLOUT_DATABASE_URL"])
    elif args.command == "run":
        result = run(
            args.directory,
            dsn=os.environ["ROLLOUT_DATABASE_URL"],
            route=args.route,
            worker_id=args.worker_id,
            limit=args.limit,
        )
    else:
        result = postgres.summary(os.environ["ROLLOUT_DATABASE_URL"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
