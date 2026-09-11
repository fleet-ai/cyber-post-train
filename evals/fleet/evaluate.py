"""Configurable Fleet evaluations using the existing fenced PostgreSQL worker.

Preparation is offline. Preflight reads exact task/model metadata; it does not
create a challenge or score a task. Initialization writes only an empty dedicated
database. Workers claim pending rows and never retry ambiguous or valid outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from cyber_post_train.jobs import digest
from evals.fleet import opencode_self_hosted as harness
from evals.fleet import rollout_ledger as ledger
from evals.fleet import rollout_postgres as postgres
from evals.fleet import rollout_worker as worker
from evals.fleet.fixed_proxy import completion_overrides
from training.sft import _known, read_mapping

CATALOG_FIELDS = {"engine", "precision", "tensor_parallel_size"}
MODEL_FIELDS = {"model_path", "model_type", "architectures"}
SERVER_FIELDS = {
    "model_path",
    "context_length",
    "tp_size",
    "quantization",
    "kv_cache_dtype",
    "reasoning_parser",
    "tool_call_parser",
}
SERVER_OPTIONAL_FIELDS = {"dp_size", "load_balance_method"}
TASK_FIELDS = {
    "task_key",
    "task_version_id",
    "env_key",
    "env_version",
    "environment_version_id",
    "data_key",
    "data_version",
}
RUNTIME_FILES = (
    "evaluate.py",
    "rollout_worker.py",
    "rollout_postgres.py",
    "rollout_ledger.py",
    "opencode_self_hosted.py",
    "fixed_proxy.py",
    "exact_pass4_crypto.py",
)


def runtime_identity() -> dict:
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in RUNTIME_FILES
    }


TREATMENT_FIELDS = {
    "harness",
    "harness_version",
    "release_asset_sha256",
    "provider_adapter",
    "context_management",
    "context_window_size",
    "compaction_headroom_tokens",
    "max_output_tokens",
    "max_model_requests",
    "timeout_seconds",
    "tools",
    "tool_catalog_sha256",
}


def _sha(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError("expected a SHA-256 identity")


def _name(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", value):
        raise ValueError("expected an owner-specific lowercase name")


def compile_eval(config: dict, *, relative_to: Path) -> dict:
    _known(
        config,
        {
            "name",
            "task_set",
            "models",
            "routes",
            "harness",
            "images",
            "pass_k",
            "concurrency",
            "training_data_eligible",
            "sampling",
        },
        "evaluation",
    )
    _name(config["name"])
    selection = read_mapping(relative_to / config["task_set"])
    tasks = worker._selection_index(selection)
    # Preserve only the versioned runtime tuple, not historical task outcomes.
    selected = []
    for task in tasks.values():
        if any(not isinstance(task.get(k), str) or not task[k] for k in TASK_FIELDS):
            raise ValueError("task set lacks an exact task/environment/data tuple")
        for field in ("task_version_id", "environment_version_id"):
            if str(uuid.UUID(task[field])) != task[field]:
                raise ValueError("task and environment versions must be canonical UUIDs")
        selected.append({k: task[k] for k in sorted(TASK_FIELDS)})
    selected.sort(key=lambda x: x["task_version_id"])
    treatment = config["harness"]
    if set(treatment) != TREATMENT_FIELDS or treatment["tools"] != ["bash", "submit_report"]:
        raise ValueError("require the complete, ordered Fleet blackbox harness treatment")
    if treatment["provider_adapter"] != "@ai-sdk/openai-compatible":
        raise ValueError("this harness uses the OpenAI-compatible provider adapter")
    _sha(treatment["release_asset_sha256"])
    _sha(treatment["tool_catalog_sha256"])
    for field in ("max_model_requests", "timeout_seconds"):
        if type(treatment[field]) is not int or treatment[field] <= 0:
            raise ValueError("request/time budgets must be positive integers")
    if treatment["timeout_seconds"] > 28800:
        raise ValueError("runtime budget exceeds the supported instance lifetime")
    harness.opencode_settings(
        {
            "harness": {
                "name": treatment["harness"],
                "version": treatment["harness_version"],
                **treatment,
            },
            "model": {"served_id": "check"},
        }
    )
    images = config["images"]
    completion_overrides(
        {"model": "check", "max_tokens": treatment["max_output_tokens"], **config["sampling"]}
    )
    if set(config["sampling"]) != {"temperature", "top_p", "seed"}:
        raise ValueError("sampling requires temperature, top_p and seed")
    if set(images) != {"agent", "proxy"} or any(
        not isinstance(v, str) or not re.fullmatch(r"(?:[^\s]+@)?sha256:[a-f0-9]{64}", v)
        for v in images.values()
    ):
        raise ValueError("agent and fixed-proxy images must be immutable")
    models, routes = config["models"], config["routes"]
    if not models or not routes:
        raise ValueError("models and serving routes are required")
    assignments = {model: set() for model in models}
    for name, model in models.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,63}", name):
            raise ValueError("model aliases must be lowercase names; dots and hyphens are allowed")
        if (
            set(model) != {"repository", "revision", "session_model"}
            or not re.fullmatch(r"[a-f0-9]{40}", model["revision"])
            or not all(isinstance(v, str) and v for v in model.values())
        ):
            raise ValueError("model needs repository, exact revision and catalog session identity")
    for block, route in routes.items():
        _name(block)
        if set(route) != {
            "model",
            "served_id",
            "task_versions",
            "catalog",
            "model_info",
            "server_info",
            "endpoint_origin",
        }:
            raise ValueError("route requires explicit assignment and serving profile")
        # Fleet credentials may only reach the Fleet gateway, never a config-supplied host.
        if route["endpoint_origin"] != "https://inference.flt.build":
            raise ValueError("this adapter uses only the Fleet inference gateway")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", route["served_id"]):
            raise ValueError("unsafe served model identifier")
        for field, keys in (
            ("catalog", CATALOG_FIELDS),
            ("model_info", MODEL_FIELDS),
            ("server_info", SERVER_FIELDS),
        ):
            actual = set(route[field])
            optional = SERVER_OPTIONAL_FIELDS if field == "server_info" else set()
            if not keys <= actual or actual - keys - optional:
                raise ValueError("serving profile must bind every required runtime field")
        if route["server_info"]["context_length"] != treatment["context_window_size"]:
            raise ValueError("serving and harness context lengths differ")
        versions = route["task_versions"]
        model = route["model"]
        if model not in models or not isinstance(versions, list) or not versions:
            raise ValueError("route has no valid model/task assignment")
        if len(set(versions)) != len(versions) or set(versions) - tasks.keys():
            raise ValueError("route tasks are duplicate or outside the frozen set")
        if assignments[model] & set(versions):
            raise ValueError("a model/task cannot be split across serving blocks")
        assignments[model].update(versions)
    if any(versions != tasks.keys() for versions in assignments.values()):
        raise ValueError("every model must cover the full task set exactly once")
    for field, default, maximum in (("pass_k", 1, 64), ("concurrency", 1, 32)):
        value = config.get(field, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("pass_k/concurrency outside supported bounds")
    eligible = config.get("training_data_eligible", False)
    if type(eligible) is not bool:
        raise ValueError("training eligibility must be explicit boolean")
    plan = {
        "schema": "cyber_fleet_eval_v1",
        "campaign_id": config["name"],
        "run_prefix": config["name"],
        "selection": {"source_job_id": selection.get("source_job_id")},
        "tasks": selected,
        "models": models,
        "routes": routes,
        "treatment": treatment,
        "images": images,
        "pass_k": config.get("pass_k", 1),
        "concurrency": config.get("concurrency", 1),
        "training_data_eligible": eligible,
        "automatic_retry": False,
        "runtime_files": runtime_identity(),
        "sampling": config["sampling"],
        "interpretation": "serving-block descriptive evaluation",
    }
    return {**plan, "sha256": digest(plan)}


def plan_rows(plan: dict) -> list[dict]:
    tasks = {t["task_version_id"]: t for t in plan["tasks"]}
    return [
        {
            "experiment_id": plan["campaign_id"],
            "task_key": tasks[version]["task_key"],
            "task_version_id": version,
            "model_id": route["model"],
            "model_revision": plan["models"][route["model"]]["revision"],
            "serving_block": block,
            "endpoint_model_id": route["served_id"],
            "harness_id": "protocol-" + plan["sha256"],
            "attempt": attempt,
            "max_retries": 0,
        }
        for block, route in sorted(plan["routes"].items())
        for version in sorted(route["task_versions"])
        for attempt in range(1, plan["pass_k"] + 1)
    ]


def prepare(config: dict, directory: Path, *, relative_to: Path) -> dict:
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
    ledger._plan_rows(directory / "plan.csv")
    return {
        "prepared": str(directory),
        "sessions": len(rows),
        "submitted": False,
        "plan_sha256": plan["sha256"],
    }


def load(directory: Path) -> dict:
    plan = read_mapping(directory / "EVAL.json")
    if plan.get("schema") != "cyber_fleet_eval_v1" or plan.get("sha256") != digest(
        {k: v for k, v in plan.items() if k != "sha256"}
    ):
        raise ValueError("evaluation plan identity changed")
    if plan.get("runtime_files") != runtime_identity():
        raise ValueError("evaluation code changed; do not run a frozen plan with different code")
    actual = ledger._plan_rows(directory / "plan.csv")
    expected = plan_rows(plan)
    if [{k: row[k] for k in expected[0]} for row in actual] != sorted(
        expected,
        key=lambda r: (r["experiment_id"], r["task_version_id"], r["model_id"], r["attempt"]),
    ):
        raise ValueError("CSV and frozen evaluation plan differ")
    return plan


def check_route(route: dict, model: dict, client: httpx.Client) -> dict:
    def get(path):
        response = client.get(
            "https://inference.flt.build" + path, headers={"X-Fleet-Model": route["served_id"]}
        )
        if response.status_code != 200:
            raise RuntimeError(f"serving identity HTTP {response.status_code}")
        return response.json()

    matches = [
        r
        for r in get("/fleet/v1/model-catalog").get("data", [])
        if r.get("id") == route["served_id"]
    ]
    if len(matches) != 1:
        raise RuntimeError("serving route absent or ambiguous")
    catalog = matches[0]
    expected = {
        **route["catalog"],
        "model_revision": model["revision"],
        "routed": True,
        "status": "ready",
    }
    if any(catalog.get(k) != v for k, v in expected.items()) or not (
        catalog.get("ready_replicas", 0) > 0 and "tool_calling" in catalog.get("capabilities", [])
    ):
        raise RuntimeError("serving identity/readiness drift")
    for field in ("model_info", "server_info"):
        actual = get("/" + field)
        if any(actual.get(k) != v for k, v in route[field].items()):
            raise RuntimeError("serving runtime drift")
        if field == "server_info" and actual.get("served_model_name") != route["served_id"]:
            raise RuntimeError("serving model name drift")
    return {
        "served_id": route["served_id"],
        "revision": model["revision"],
        "profile_sha256": digest(route),
        "ready": True,
    }


def check_images(plan: dict) -> None:
    # No pull or provision: both containers must already be available on this worker.
    for image in plan["images"].values():
        result = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError("immutable harness image is not staged on this worker")
        info = json.loads(result.stdout)
        if (
            image not in [info.get("Id"), *(info.get("RepoDigests") or [])]
            or info.get("Os") != "linux"
            or info.get("Architecture") != "amd64"
        ):
            raise RuntimeError("harness image bytes or platform differ")
        if image == plan["images"]["agent"]:
            labels = info.get("Config", {}).get("Labels") or {}
            if (
                labels.get("cyber.opencode.release-sha256")
                != plan["treatment"]["release_asset_sha256"]
            ):
                raise RuntimeError("harness release identity differs")
    # Version alone never initializes OpenCode's data directories. Exercise the
    # same uid, HOME and private mount as a real agent, offline and before claims.
    with tempfile.TemporaryDirectory(prefix="cpt-agent-preflight-") as home:
        if os.geteuid() == 0:
            os.chown(home, 1000, 1000)
        startup = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--platform",
                "linux/amd64",
                "--network",
                "none",
                "--pull",
                "never",
                *harness.agent_container_user_args(),
                "-v",
                f"{home}:/home/node",
                plan["images"]["agent"],
                "bash",
                "-lc",
                "opencode --version && opencode db path",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    if startup.returncode or startup.stdout.splitlines() != [
        plan["treatment"]["harness_version"],
        "/home/node/.local/share/opencode/opencode.db",
    ]:
        raise RuntimeError("harness version or writable agent-home startup differs")


def preflight(directory: Path) -> dict:
    plan = load(directory)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise ValueError("FLEET_API_KEY is required")
    proof = {
        "schema": "cyber_fleet_eval_preflight_v1",
        "plan_sha256": plan["sha256"],
        "task_bindings": {},
        "routes": {},
        "images": plan["images"],
    }
    check_images(plan)
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60) as client:
        account = harness._request(client, "GET", "/v1/account")
        if account.get("team_name") != "fleet" or account.get("team_id") != harness.FLEET_TEAM_ID:
            raise RuntimeError("Fleet team identity required")
        for block, route in plan["routes"].items():
            proof["routes"][block] = check_route(route, plan["models"][route["model"]], client)
        for task in plan["tasks"]:
            binding = worker._task_binding(client, task, task)
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
        "receipt_sha256": proof["sha256"],
    }


def checked_preflight(directory: Path) -> tuple[dict, dict]:
    plan = load(directory)
    proof = read_mapping(directory / "EVAL_PREFLIGHT.json")
    if proof.get("plan_sha256") != plan["sha256"] or proof.get("sha256") != digest(
        {k: v for k, v in proof.items() if k != "sha256"}
    ):
        raise ValueError("evaluation preflight does not match")
    if (
        proof.get("schema") != "cyber_fleet_eval_preflight_v1"
        or proof.get("images") != plan["images"]
        or set(proof.get("task_bindings", {})) != {t["task_version_id"] for t in plan["tasks"]}
        or set(proof.get("routes", {})) != set(plan["routes"])
    ):
        raise ValueError("evaluation preflight is incomplete")
    return plan, proof


def run(directory: Path, *, dsn: str, route: str, worker_id: str, limit: int) -> dict:
    plan, proof = checked_preflight(directory)
    _name(worker_id)
    if (
        not dsn
        or route not in plan["routes"]
        or type(limit) is not int
        or limit < 1
        or limit > len(plan["routes"].get(route, {}).get("task_versions", [])) * plan["pass_k"]
    ):
        raise ValueError("database, route and bounded positive session limit required")
    postgres.verify_plan(dsn, directory / "plan.csv")
    # Preflight may have run on a different host: check this Docker daemon before claims.
    check_images(plan)
    # Each controller has a create-once identity; PG remains the only claim authority.
    worker._safe_write_once(
        directory / f"STARTED-{worker_id}.json", {"plan_sha256": plan["sha256"]}
    )
    plan["task_bindings"] = proof["task_bindings"]
    index = {}
    for row in ledger._plan_rows(directory / "plan.csv"):
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

    def one(index_number):
        try:
            with httpx.Client(
                headers={"Authorization": "Bearer " + os.environ["FLEET_API_KEY"]}, timeout=60
            ) as client:
                endpoint = plan["routes"][route]
                check_route(endpoint, plan["models"][endpoint["model"]], client)
        except Exception as exc:
            return {
                "claimed": False,
                "accepted": False,
                "serving_block": route,
                "controller_failure_code": type(exc).__name__.lower(),
            }
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
            proxy_script=Path(__file__).with_name("fixed_proxy.py"),
        )

    with ThreadPoolExecutor(max_workers=min(limit, plan["concurrency"])) as pool:
        results = list(pool.map(one, range(limit)))
    terminal = {
        "worker_id": worker_id,
        "plan_sha256": plan["sha256"],
        "results": results,
        "accepted": sum(r.get("accepted") is True for r in results),
    }
    worker._safe_write_once(directory / f"TERMINAL-{worker_id}.json", terminal)
    return terminal
