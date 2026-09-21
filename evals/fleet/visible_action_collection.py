"""Isolated non-thinking OpenCode runtime for visible-action collection.

Historical Fleet evaluation plans bind the exact bytes of the shared evaluator,
worker, harness, and proxy.  This successor leaves those files unchanged.  It
wraps their validated task/route/ledger machinery with a new plan schema and a
collection-only runtime that disables Qwen thinking at three boundaries:

* OpenCode declares the selected model as non-reasoning;
* the OpenCode invocation omits its thinking-display flag; and
* the standalone proxy forces ``chat_template_kwargs.enable_thinking=false``.

Preparation is source-only.  Preflight performs metadata/image reads but no
model request.  ``run`` is the only command in this module that creates Fleet
environments or model traffic.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import evaluate
from evals.fleet import opencode_self_hosted as harness
from evals.fleet import rollout_ledger as ledger
from evals.fleet import rollout_postgres as postgres
from evals.fleet import rollout_worker as worker

PLAN_SCHEMA = "cyber_fleet_visible_action_collection_eval_v1"
PREFLIGHT_SCHEMA = "cyber_fleet_visible_action_collection_preflight_v1"
RUNTIME_SCHEMA = "cyber_visible_action_collection_runtime_v1"
DUPLICATE_CENSUS_SCHEMA = "cyber_visible_action_collection_duplicate_census_v1"
DUPLICATE_CENSUS_FILE = "COLLECTION_DUPLICATE_CENSUS.json"
DUPLICATE_CENSUS_MAX_AGE_SECONDS = 600
DUPLICATE_CENSUS_COVERAGE = "all_authoritative_collection_ledgers_and_fleet_sessions_v1"
THINKING_DISABLED = "disabled"
EXECUTION_MODE = "authorized_cpu_worker_local_v1"
PROXY_FILE = "collection_fixed_proxy.py"
ADAPTER_FILES = ("visible_action_collection.py", PROXY_FILE)
RUNTIME_FIELDS = {
    "schema",
    "source_template_sha256",
    "reasoning_generation",
    "reasoning_request_override",
    "opencode_model_reasoning",
    "opencode_cli_thinking_flag",
    "maximum_planned_cells",
    "prelaunch_exact_cell_duplicate_census_required",
    "duplicate_census_max_age_seconds",
    "duplicate_census_required_coverage",
    "automatic_replay_of_ambiguous_cells",
    "external_submission",
    "execution_mode",
    "cluster_wrapper_supported",
}

_BASE_OPENCODE_SETTINGS = harness.opencode_settings
_BASE_DOCKER = harness._docker  # noqa: SLF001


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_identity() -> dict[str, Any]:
    """Bind both the historical base rail and the new collection adapter."""
    return {
        "base_evaluator": evaluate.runtime_identity(),
        "collection_adapter": {
            name: _file_sha256(Path(__file__).with_name(name)) for name in ADAPTER_FILES
        },
    }


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"exact {label} SHA-256 is required")
    return value


def _runtime_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUNTIME_FIELDS:
        raise ValueError("collection runtime contract has unknown or missing fields")
    if value != {
        "schema": RUNTIME_SCHEMA,
        "source_template_sha256": value.get("source_template_sha256"),
        "reasoning_generation": THINKING_DISABLED,
        "reasoning_request_override": {"chat_template_kwargs": {"enable_thinking": False}},
        "opencode_model_reasoning": False,
        "opencode_cli_thinking_flag": False,
        "maximum_planned_cells": value.get("maximum_planned_cells"),
        "prelaunch_exact_cell_duplicate_census_required": True,
        "duplicate_census_max_age_seconds": DUPLICATE_CENSUS_MAX_AGE_SECONDS,
        "duplicate_census_required_coverage": DUPLICATE_CENSUS_COVERAGE,
        "automatic_replay_of_ambiguous_cells": False,
        "external_submission": False,
        "execution_mode": EXECUTION_MODE,
        "cluster_wrapper_supported": False,
    }:
        raise ValueError("collection runtime contract is not the qualified non-thinking rail")
    _sha(value["source_template_sha256"], "source template")
    maximum = value["maximum_planned_cells"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("collection cell ceiling must be a positive integer")
    return copy.deepcopy(value)


def compile_eval(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
    """Compile a new-schema plan through the unchanged shared validator."""
    if not isinstance(config, dict) or "collection_runtime" not in config:
        raise ValueError("collection_runtime is required")
    runtime = _runtime_contract(config["collection_runtime"])
    base = copy.deepcopy(config)
    base.pop("collection_runtime")
    treatment = base.get("harness")
    if not isinstance(treatment, dict) or set(treatment) != evaluate.TREATMENT_FIELDS | {
        "thinking_mode"
    }:
        raise ValueError("collection needs the complete OpenCode non-thinking treatment")
    if treatment.get("thinking_mode") != THINKING_DISABLED:
        raise ValueError("visible-action collection requires disabled thinking")
    base["harness"] = {key: item for key, item in treatment.items() if key != "thinking_mode"}
    plan = evaluate.compile_eval(base, relative_to=relative_to)
    if (
        plan["training_data_eligible"] is not True
        or plan["pass_k"] != 4
        or plan["max_reviewed_infrastructure_retries"] != 0
        or plan["automatic_retry"] is not False
    ):
        raise ValueError("collection plan lost its pass@4 create-once eligibility contract")
    planned_cells = len(evaluate.plan_rows(plan))
    if planned_cells > runtime["maximum_planned_cells"]:
        raise ValueError("collection plan exceeds its frozen cell ceiling")
    plan.pop("sha256")
    plan.update(
        {
            "schema": PLAN_SCHEMA,
            "treatment": copy.deepcopy(treatment),
            "collection_runtime": runtime,
            "planned_cells": planned_cells,
            "runtime_files": runtime_identity(),
            "interpretation": "verified-success visible-action trajectory collection",
        }
    )
    return {**plan, "sha256": digest(plan)}


def plan_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return evaluate.plan_rows(plan)


def cell_universe_sha256(plan: dict[str, Any]) -> str:
    return "sha256:" + digest(plan_rows(plan))


def validate_duplicate_census(
    directory: Path,
    plan: dict[str, Any],
    *,
    require_fresh: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate, but never mint, the external exact-cell duplicate census."""
    value = evaluate.read_mapping(directory / DUPLICATE_CENSUS_FILE)
    fields = {
        "schema",
        "plan_sha256",
        "planned_cell_universe_sha256",
        "planned_cells",
        "coverage",
        "authority_snapshot_sha256",
        "issuer",
        "completed_at",
        "exact_duplicate_count",
        "ambiguous_cell_count",
        "receipt_sha256",
    }
    if set(value) != fields or value.get("schema") != DUPLICATE_CENSUS_SCHEMA:
        raise ValueError("duplicate census has unknown or missing fields")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value["receipt_sha256"] != "sha256:" + digest(unsigned):
        raise ValueError("duplicate census receipt identity changed")
    if (
        value["plan_sha256"] != plan["sha256"]
        or value["planned_cell_universe_sha256"] != cell_universe_sha256(plan)
        or value["planned_cells"] != plan["planned_cells"]
    ):
        raise ValueError("duplicate census is for a different exact cell universe")
    if value["coverage"] != DUPLICATE_CENSUS_COVERAGE:
        raise ValueError("duplicate census coverage is incomplete")
    _sha(value["authority_snapshot_sha256"], "duplicate-census authority snapshot")
    if not isinstance(value["issuer"], str) or not value["issuer"].strip():
        raise ValueError("duplicate census issuer is required")
    if value["exact_duplicate_count"] != 0 or value["ambiguous_cell_count"] != 0:
        raise ValueError("duplicate or ambiguous collection cells require reconciliation")
    try:
        completed = datetime.fromisoformat(value["completed_at"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError("duplicate census needs an RFC 3339 completion time") from error
    if completed.tzinfo is None:
        raise ValueError("duplicate census completion time needs a timezone")
    if require_fresh:
        current = now or datetime.now(UTC)
        age = (current - completed).total_seconds()
        if age < 0 or age > DUPLICATE_CENSUS_MAX_AGE_SECONDS:
            raise ValueError("duplicate census is not fresh enough for preflight")
    return value


def prepare(config: dict[str, Any], directory: Path, *, relative_to: Path) -> dict[str, Any]:
    plan = compile_eval(config, relative_to=relative_to)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in ("claims", "attempts"):
        (directory / name).mkdir(mode=0o700)
    harness.write_json_once(directory / "EVAL.json", plan)
    rows = plan_rows(plan)
    with (directory / "plan.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ledger._plan_rows(directory / "plan.csv")  # noqa: SLF001
    return {
        "prepared": str(directory),
        "sessions": len(rows),
        "submitted": False,
        "plan_sha256": plan["sha256"],
    }


def load(directory: Path) -> dict[str, Any]:
    plan = evaluate.read_mapping(directory / "EVAL.json")
    unsigned = {key: item for key, item in plan.items() if key != "sha256"}
    if plan.get("schema") != PLAN_SCHEMA or plan.get("sha256") != digest(unsigned):
        raise ValueError("visible-action collection plan identity changed")
    if plan.get("runtime_files") != runtime_identity():
        raise ValueError("collection runtime changed; prepare a new immutable plan")
    _runtime_contract(plan.get("collection_runtime"))
    actual = ledger._plan_rows(directory / "plan.csv")  # noqa: SLF001
    expected = plan_rows(plan)
    fields = list(expected[0])
    if [{key: row[key] for key in fields} for row in actual] != sorted(
        expected,
        key=lambda row: (
            row["experiment_id"],
            row["task_version_id"],
            row["model_id"],
            row["attempt"],
        ),
    ):
        raise ValueError("CSV and visible-action collection plan differ")
    if len(actual) != plan.get("planned_cells"):
        raise ValueError("visible-action collection cell count drift")
    return plan


def nonthinking_opencode_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Derive OpenCode settings while turning off its reasoning declaration."""
    settings = _BASE_OPENCODE_SETTINGS(config)
    model_id = config["model"]["served_id"]
    model = settings["provider"]["fleet-cluster"]["models"][model_id]
    model["reasoning"] = False
    model.pop("interleaved", None)
    return settings


def collection_docker_args(arguments: tuple[Any, ...]) -> tuple[Any, ...]:
    """Remove only the pinned OpenCode thinking-display flag."""
    marker = "opencode run --format json --thinking "
    replacement = "opencode run --format json "
    changed = 0
    rendered: list[Any] = []
    for argument in arguments:
        if isinstance(argument, str) and marker in argument:
            changed += argument.count(marker)
            argument = argument.replace(marker, replacement)
        rendered.append(argument)
    contains_command = any(
        isinstance(argument, str) and "opencode run --format json" in argument
        for argument in arguments
    )
    if changed > 1 or (contains_command and changed != 1):
        raise RuntimeError("ambiguous OpenCode collection command")
    return tuple(rendered)


def _collection_docker(*arguments: Any, **kwargs: Any) -> Any:
    return _BASE_DOCKER(*collection_docker_args(arguments), **kwargs)


def activate_runtime() -> None:
    """Install the collection-only overrides before any worker thread starts."""
    if harness.opencode_settings not in {_BASE_OPENCODE_SETTINGS, nonthinking_opencode_settings}:
        raise RuntimeError("OpenCode settings were already modified by another runtime")
    if harness._docker not in {_BASE_DOCKER, _collection_docker}:  # noqa: SLF001
        raise RuntimeError("OpenCode Docker launcher was already modified by another runtime")
    harness.opencode_settings = nonthinking_opencode_settings
    harness._docker = _collection_docker  # type: ignore[attr-defined]  # noqa: SLF001


def preflight(directory: Path) -> dict[str, Any]:
    """Read exact metadata/images and write a score-free preflight receipt."""
    plan = load(directory)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise ValueError("FLEET_API_KEY is required")
    census = validate_duplicate_census(directory, plan, require_fresh=True)
    proof = {
        "schema": PREFLIGHT_SCHEMA,
        "plan_sha256": plan["sha256"],
        "task_bindings": {},
        "routes": {},
        "images": plan["images"],
        "reasoning_generation": THINKING_DISABLED,
        "duplicate_census_receipt_sha256": census["receipt_sha256"],
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


def checked_preflight(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load(directory)
    census = validate_duplicate_census(directory, plan, require_fresh=False)
    proof = evaluate.read_mapping(directory / "EVAL_PREFLIGHT.json")
    unsigned = {key: item for key, item in proof.items() if key != "sha256"}
    if proof.get("plan_sha256") != plan["sha256"] or proof.get("sha256") != digest(unsigned):
        raise ValueError("visible-action collection preflight does not match")
    if (
        proof.get("schema") != PREFLIGHT_SCHEMA
        or proof.get("images") != plan["images"]
        or proof.get("reasoning_generation") != THINKING_DISABLED
        or proof.get("duplicate_census_receipt_sha256") != census["receipt_sha256"]
        or proof.get("model_calls") != 0
        or set(proof.get("task_bindings", {}))
        != {task["task_version_id"] for task in plan["tasks"]}
        or set(proof.get("routes", {})) != set(plan["routes"])
    ):
        raise ValueError("visible-action collection preflight is incomplete")
    return plan, proof


def run(
    directory: Path,
    *,
    dsn: str,
    route: str,
    worker_id: str,
    limit: int,
) -> dict[str, Any]:
    """Execute bounded collection cells through the isolated runtime."""
    plan, proof = checked_preflight(directory)
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
    postgres.verify_plan(dsn, directory / "plan.csv")
    evaluate.check_images(plan)
    worker._safe_write_once(  # noqa: SLF001
        directory / f"STARTED-{worker_id}.json", {"plan_sha256": plan["sha256"]}
    )
    plan["task_bindings"] = proof["task_bindings"]
    index = {}
    for row in ledger._plan_rows(directory / "plan.csv"):  # noqa: SLF001
        cell_id = "sha256:" + digest(row)
        index[(row["model_id"], row["task_version_id"], row["attempt"])] = {
            **row,
            "initial_execution": {
                "cell_id": cell_id,
                "execution_id": "sha256:" + digest({"cell": cell_id, "generation": 1}),
                "execution_generation": 1,
            },
        }
    os.environ["AGENT_HARNESS_IMAGE"] = plan["images"]["agent"]
    os.environ["FIXED_PROXY_IMAGE"] = plan["images"]["proxy"]
    activate_runtime()

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
        "results": results,
        "accepted": sum(result.get("accepted") is True for result in results),
    }
    worker._safe_write_once(directory / f"TERMINAL-{worker_id}.json", terminal)  # noqa: SLF001
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("config", type=Path)
    prepare_parser.add_argument("--output", type=Path, required=True)
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
            args.output,
            relative_to=args.config.resolve().parent,
        )
    elif args.command == "preflight":
        result = preflight(args.directory)
    elif args.command == "init":
        plan, _proof = checked_preflight(args.directory)
        validate_duplicate_census(args.directory, plan, require_fresh=True)
        result = postgres.initialize(
            os.environ["ROLLOUT_DATABASE_URL"], args.directory / "plan.csv"
        )
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
