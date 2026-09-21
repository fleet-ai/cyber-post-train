"""Exactly-once local rail for the visible-action collection campaign.

This is an immutable v2 successor to :mod:`visible_action_collection`.  It
preserves the task, model, harness, sampling and pass@4 science while replacing
the unavailable global duplicate-census gate with one operation we can prove:

* one deterministic campaign/plan/cell universe;
* one canonical private operation directory;
* one create-once intent written before the first external mutation;
* one empty, dedicated PostgreSQL ledger initialized in the same transaction
  as its operation binding; and
* no retry after an uncertain initialization result.

Historical sessions have different deterministic cell identities and therefore
belong to different experiments.  This rail never searches them, adopts them,
or treats their existence as permission to replay this operation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import stat
from collections.abc import Callable
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
from evals.fleet import visible_action_collection as v1

PLAN_SCHEMA = "cyber_fleet_visible_action_collection_eval_v2"
PREFLIGHT_SCHEMA = "cyber_fleet_visible_action_collection_preflight_v2"
RUNTIME_SCHEMA = "cyber_visible_action_collection_runtime_v2"
OPERATION_AUTHORIZATION_SCHEMA = "cyber_fleet_visible_action_collection_operation_authorization_v1"
CREATE_INTENT_SCHEMA = "cyber_fleet_visible_action_collection_create_intent_v1"
LEDGER_RECEIPT_SCHEMA = "cyber_fleet_visible_action_collection_ledger_created_v1"
OPERATION_AUTHORIZATION_FILE = "COLLECTION_OPERATION_AUTHORIZATION.json"
CREATE_INTENT_FILE = "COLLECTION_CREATE_INTENT.json"
LEDGER_RECEIPT_FILE = "COLLECTION_LEDGER_CREATED.json"
THINKING_DISABLED = v1.THINKING_DISABLED
EXECUTION_MODE = "authorized_amd64_cpu_job_v1"
PROXY_FILE = v1.PROXY_FILE
ADAPTER_FILES = (
    "visible_action_collection.py",
    "visible_action_collection_v2.py",
    PROXY_FILE,
)
LEDGER_CELL_DERIVATION = "fleet-cyber-rollout-cell-v1"
SCIENTIFIC_CELL_DERIVATION = "sha256-canonical-ledger-row-v1"
EXECUTION_DERIVATION = "sha256-scientific-cell-generation-v1"
RUNTIME_FIELDS = {
    "schema",
    "source_template_sha256",
    "reasoning_generation",
    "reasoning_request_override",
    "opencode_model_reasoning",
    "opencode_cli_thinking_flag",
    "maximum_planned_cells",
    "operation_authorization_required",
    "canonical_private_operation_root_required",
    "exclusive_pre_mutation_intent_required",
    "dedicated_empty_ledger_required",
    "automatic_replay_of_ambiguous_cells",
    "external_submission",
    "execution_mode",
    "cluster_wrapper_supported",
}


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError(f"exact {label} SHA-256 is required")
    return value


def _safe_name(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", value) is None:
        raise ValueError(f"safe lowercase {label} is required")
    return value


def _sealed(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value.get("sha256") != "sha256:" + digest(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError(f"invalid sealed {schema}")


def runtime_identity() -> dict[str, Any]:
    """Bind the historical evaluator plus both collection adapters."""
    return {
        "base_evaluator": evaluate.runtime_identity(),
        "collection_adapter": {
            name: _file_sha256(Path(__file__).with_name(name)) for name in ADAPTER_FILES
        },
    }


def _runtime_contract(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != RUNTIME_FIELDS:
        raise ValueError("collection v2 runtime contract has unknown or missing fields")
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
    }
    if value != expected:
        raise ValueError("collection v2 runtime is not the qualified exactly-once rail")
    _sha(value["source_template_sha256"], "source template")
    maximum = value["maximum_planned_cells"]
    if type(maximum) is not int or maximum < 1:
        raise ValueError("collection cell ceiling must be a positive integer")
    return copy.deepcopy(value)


def compile_eval(config: dict[str, Any], *, relative_to: Path) -> dict[str, Any]:
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


def identity_map(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ordered ledger/scientific/execution identity map."""
    rows: list[dict[str, Any]] = []
    for raw in plan_rows(plan):
        row = {
            **raw,
            "task_key": raw.get("task_key"),
            "max_retries": int(raw.get("max_retries", 0)),
        }
        row["cell_id"] = ledger._cell_id(row)  # noqa: SLF001
        stored = {field: row[field] for field in ledger.PLAN_STORED_COLUMNS}
        scientific_cell_id = "sha256:" + digest(stored)
        rows.append(
            {
                "campaign_id": row["experiment_id"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "model_id": row["model_id"],
                "attempt": row["attempt"],
                "ledger_cell_id": row["cell_id"],
                "scientific_cell_id": scientific_cell_id,
                "execution_id": "sha256:" + digest({"cell": scientific_cell_id, "generation": 1}),
            }
        )
    rows.sort(
        key=lambda row: (
            row["campaign_id"],
            row["task_version_id"],
            row["model_id"],
            row["attempt"],
        )
    )
    return rows


def identity_map_sha256(plan: dict[str, Any]) -> str:
    return "sha256:" + digest(identity_map(plan))


def build_operation_authorization(plan: dict[str, Any]) -> dict[str, Any]:
    campaign_id = _safe_name(plan.get("campaign_id"), "campaign ID")
    plan_sha = _sha("sha256:" + str(plan.get("sha256", "")), "plan")
    universe = cell_universe_sha256(plan)
    identity_sha = identity_map_sha256(plan)
    root_name = f"{campaign_id}-{plan_sha.removeprefix('sha256:')[:12]}"
    ledger_id = "sha256:" + digest(
        {
            "schema": "cyber_fleet_visible_action_collection_dedicated_ledger_v1",
            "campaign_id": campaign_id,
            "plan_sha256": plan_sha,
            "planned_cell_universe_sha256": universe,
            "identity_map_sha256": identity_sha,
        }
    )
    body = {
        "schema": OPERATION_AUTHORIZATION_SCHEMA,
        "campaign_id": campaign_id,
        "plan_sha256": plan_sha,
        "planned_cell_universe_sha256": universe,
        "planned_cells": plan["planned_cells"],
        "operation_root_name": root_name,
        "dedicated_ledger_id": ledger_id,
        "identity_map_sha256": identity_sha,
        "identity_derivation": {
            "ledger_cell_id": LEDGER_CELL_DERIVATION,
            "scientific_cell_id": SCIENTIFIC_CELL_DERIVATION,
            "execution_id": EXECUTION_DERIVATION,
            "execution_generation": 1,
        },
        "execution_contract": {
            "canonical_private_operation_root_required": True,
            "exclusive_pre_mutation_intent_required": True,
            "dedicated_empty_ledger_required": True,
            "same_path_retry_allowed": False,
            "alternate_path_retry_allowed": False,
            "ambiguous_external_mutation_replay_allowed": False,
        },
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_operation_authorization(value: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    _sealed(value, OPERATION_AUTHORIZATION_SCHEMA)
    if set(value) != {
        "schema",
        "campaign_id",
        "plan_sha256",
        "planned_cell_universe_sha256",
        "planned_cells",
        "operation_root_name",
        "dedicated_ledger_id",
        "identity_map_sha256",
        "identity_derivation",
        "execution_contract",
        "sha256",
    }:
        raise ValueError("operation authorization has unknown or missing fields")
    expected = build_operation_authorization(plan)
    if value != expected:
        raise ValueError("operation authorization differs from the exact campaign identity")
    return copy.deepcopy(value)


def canonical_operation_root(private_root: Path, authorization: dict[str, Any]) -> Path:
    try:
        metadata = private_root.lstat()
    except OSError as error:
        raise ValueError("private collection root is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("private collection root must be a real directory")
    return private_root.resolve() / _safe_name(
        authorization.get("operation_root_name"), "operation root name"
    )


def _write_once_fsynced(path: Path, value: dict[str, Any]) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


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
    observed_map = []
    for row in normalized:
        scientific_cell_id = "sha256:" + digest(row)
        observed_map.append(
            {
                "campaign_id": row["experiment_id"],
                "task_key": row["task_key"],
                "task_version_id": row["task_version_id"],
                "model_id": row["model_id"],
                "attempt": row["attempt"],
                "ledger_cell_id": row["cell_id"],
                "scientific_cell_id": scientific_cell_id,
                "execution_id": "sha256:" + digest({"cell": scientific_cell_id, "generation": 1}),
            }
        )
    if "sha256:" + digest(observed_map) != authorization["identity_map_sha256"]:
        raise ValueError("written plan changed the authorized cell identity map")
    _write_once_fsynced(directory / OPERATION_AUTHORIZATION_FILE, authorization)
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
        raise ValueError("visible-action collection v2 plan identity changed")
    if plan.get("runtime_files") != runtime_identity():
        raise ValueError("collection v2 runtime changed; prepare a new immutable operation")
    _runtime_contract(plan.get("collection_runtime"))
    authorization = validate_operation_authorization(
        evaluate.read_mapping(directory / OPERATION_AUTHORIZATION_FILE), plan
    )
    if directory.resolve() != canonical_operation_root(directory.parent, authorization):
        raise ValueError("collection operation is outside its canonical private root")
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
        raise ValueError("CSV and visible-action collection v2 plan differ")
    if len(actual) != plan.get("planned_cells"):
        raise ValueError("visible-action collection v2 cell count drift")
    if (
        "sha256:"
        + digest(
            [
                {
                    "campaign_id": row["experiment_id"],
                    "task_key": row["task_key"],
                    "task_version_id": row["task_version_id"],
                    "model_id": row["model_id"],
                    "attempt": row["attempt"],
                    "ledger_cell_id": row["cell_id"],
                    "scientific_cell_id": "sha256:" + digest(row),
                    "execution_id": "sha256:"
                    + digest({"cell": "sha256:" + digest(row), "generation": 1}),
                }
                for row in actual
            ]
        )
        != authorization["identity_map_sha256"]
    ):
        raise ValueError("collection v2 identity map changed")
    return plan, authorization


def preflight(directory: Path) -> dict[str, Any]:
    """Read exact metadata/images and write a score-free preflight receipt."""
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
        raise ValueError("visible-action collection v2 preflight does not match")
    if (
        proof.get("schema") != PREFLIGHT_SCHEMA
        or proof.get("operation_authorization_sha256") != authorization["sha256"]
        or proof.get("images") != plan["images"]
        or proof.get("reasoning_generation") != THINKING_DISABLED
        or proof.get("model_calls") != 0
        or set(proof.get("task_bindings", {}))
        != {task["task_version_id"] for task in plan["tasks"]}
        or set(proof.get("routes", {})) != set(plan["routes"])
    ):
        raise ValueError("visible-action collection v2 preflight is incomplete")
    return plan, authorization, proof


def _initialize_authorized_ledger(
    dsn: str, plan_path: Path, authorization: dict[str, Any]
) -> dict[str, Any]:
    rows = ledger._plan_rows(plan_path)  # noqa: SLF001
    plan_digest = ledger._plan_digest(rows)  # noqa: SLF001
    with postgres._transaction(dsn) as connection:  # noqa: SLF001
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))",
            (authorization["dedicated_ledger_id"],),
        )
        postgres.ensure_schema(connection)
        for table in (
            "ledger_metadata",
            "rollout_cells",
            "rollout_events",
            "rollout_local_results",
            "ledger_migrations",
            "ledger_reconciliations",
        ):
            if connection.execute(
                f"SELECT EXISTS(SELECT 1 FROM {table}) AS present"  # noqa: S608
            ).fetchone()["present"]:
                raise ledger.LedgerError("authorization requires an empty dedicated ledger")
        columns = ledger.PLAN_STORED_COLUMNS
        with connection.cursor() as cursor:
            cursor.executemany(
                f"INSERT INTO rollout_cells ({', '.join(columns)}, state, created_at, updated_at) "
                f"VALUES ({', '.join(['%s'] * len(columns))}, 'pending', CURRENT_TIMESTAMP, "
                "CURRENT_TIMESTAMP)",  # noqa: S608
                [tuple(row[field] for field in columns) for row in rows],
            )
            cursor.executemany(
                "INSERT INTO rollout_events (cell_id, recorded_at, event, to_state, detail_json) "
                "VALUES (%s, CURRENT_TIMESTAMP, 'initialized', 'pending', '{}')",
                [(row["cell_id"],) for row in rows],
            )
            cursor.executemany(
                "INSERT INTO ledger_metadata (key, value) VALUES (%s, %s)",
                [
                    ("schema_version", "fleet_cyber_rollout_ledger_v3"),
                    ("plan_sha256", plan_digest),
                    ("campaign_id", authorization["campaign_id"]),
                    ("operation_authorization_sha256", authorization["sha256"]),
                    ("planned_cell_universe_sha256", authorization["planned_cell_universe_sha256"]),
                    ("identity_map_sha256", authorization["identity_map_sha256"]),
                    ("dedicated_ledger_id", authorization["dedicated_ledger_id"]),
                ],
            )
    return {
        "created": True,
        "cells": len(rows),
        "plan_sha256": plan_digest,
        "operation_authorization_sha256": authorization["sha256"],
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
    }


def _verify_authorized_ledger(
    dsn: str, plan_path: Path, authorization: dict[str, Any]
) -> dict[str, Any]:
    result = postgres.verify_plan(dsn, plan_path)
    expected = {
        "campaign_id": authorization["campaign_id"],
        "operation_authorization_sha256": authorization["sha256"],
        "planned_cell_universe_sha256": authorization["planned_cell_universe_sha256"],
        "identity_map_sha256": authorization["identity_map_sha256"],
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
    }
    with postgres._read_transaction(dsn) as connection:  # noqa: SLF001
        observed = {
            row["key"]: row["value"]
            for row in connection.execute(
                "SELECT key, value FROM ledger_metadata WHERE key = ANY(%s::text[])",
                (list(expected),),
            ).fetchall()
        }
    if observed != expected:
        raise ledger.LedgerError("dedicated ledger operation binding differs")
    return {**result, **expected}


def initialize_once(
    directory: Path,
    *,
    dsn: str,
    initializer: Callable[[str, Path, dict[str, Any]], dict[str, Any]] = (
        _initialize_authorized_ledger
    ),
) -> dict[str, Any]:
    plan, authorization, proof = checked_preflight(directory)
    if not dsn:
        raise ValueError("ROLLOUT_DATABASE_URL is required")
    intent_path = directory / CREATE_INTENT_FILE
    receipt_path = directory / LEDGER_RECEIPT_FILE
    if (
        intent_path.exists()
        or intent_path.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise RuntimeError("collection create intent exists; reconcile, never retry")
    intent_body = {
        "schema": CREATE_INTENT_SCHEMA,
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
    intent = {**intent_body, "sha256": "sha256:" + digest(intent_body)}
    _write_once_fsynced(intent_path, intent)
    created = initializer(dsn, directory / "plan.csv", authorization)
    receipt_body = {
        "schema": LEDGER_RECEIPT_SCHEMA,
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
    _write_once_fsynced(receipt_path, receipt)
    return receipt


def _validated_ledger_receipt(
    directory: Path, plan: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, Any]:
    intent = evaluate.read_mapping(directory / CREATE_INTENT_FILE)
    receipt = evaluate.read_mapping(directory / LEDGER_RECEIPT_FILE)
    _sealed(intent, CREATE_INTENT_SCHEMA)
    _sealed(receipt, LEDGER_RECEIPT_SCHEMA)
    if (
        receipt.get("campaign_id") != authorization["campaign_id"]
        or receipt.get("plan_sha256") != authorization["plan_sha256"]
        or receipt.get("planned_cell_universe_sha256")
        != authorization["planned_cell_universe_sha256"]
        or receipt.get("identity_map_sha256") != authorization["identity_map_sha256"]
        or receipt.get("operation_authorization_sha256") != authorization["sha256"]
        or receipt.get("dedicated_ledger_id") != authorization["dedicated_ledger_id"]
        or receipt.get("create_intent_sha256") != intent["sha256"]
        or receipt.get("created_cells") != plan["planned_cells"]
    ):
        raise ValueError("dedicated ledger receipt differs from the authorized operation")
    return receipt


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
        scientific_cell_id = "sha256:" + digest(row)
        index[(row["model_id"], row["task_version_id"], row["attempt"])] = {
            **row,
            "initial_execution": {
                "cell_id": scientific_cell_id,
                "execution_id": "sha256:" + digest({"cell": scientific_cell_id, "generation": 1}),
                "execution_generation": 1,
            },
        }
    os.environ["AGENT_HARNESS_IMAGE"] = plan["images"]["agent"]
    os.environ["FIXED_PROXY_IMAGE"] = plan["images"]["proxy"]
    v1.activate_runtime()

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
