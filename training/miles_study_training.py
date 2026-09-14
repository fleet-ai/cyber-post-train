"""Future Miles comparative-study runtime with an exact public W&B contract.

This is a new, inert revision.  It does not alter the already-qualified Dev8
compiler or bundle.  Any use requires fresh native-image, CPU, Dev GPU,
checkpoint, and reload qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import bundled_request, digest

from . import miles_training as v1
from . import miles_wandb

SCHEMA = "cyber_miles_training_v2"
MODULE = "training.miles_study_training"
RUNTIME_FILES = (*v1.RUNTIME_FILES, "training/miles_wandb.py", "training/miles_study_training.py")


def _runtime() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _legacy_plan(plan: dict) -> dict:
    value = copy.deepcopy(plan)
    value["schema"] = v1.SCHEMA
    value["runtime_sha256"] = digest(v1._runtime())
    value.pop("wandb")
    return value


def compile_rl(config: dict, *, relative_to: Path) -> dict:
    """Compile via the qualified v1 science path, then add only W&B v2."""
    wandb = copy.deepcopy(config.get("wandb"))
    if not isinstance(wandb, dict):
        raise ValueError("W&B study contract is required")
    legacy = copy.deepcopy(config)
    legacy["wandb"] = {
        key: wandb[key] for key in ("entity", "project", "run_id")
    }
    plan = v1.compile_rl(legacy, relative_to=relative_to)
    plan["schema"] = SCHEMA
    plan["wandb"] = wandb
    plan["runtime_sha256"] = digest(_runtime())
    miles_wandb.build_contract(wandb, plan, "sha256:" + digest(plan))
    job_request(plan)
    return plan


def job_request(plan: dict) -> dict:
    if (
        plan.get("schema") != SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("execution", {}).get("cluster_target") not in {"dev", "prod"}
    ):
        raise ValueError("Miles study plan/runtime/cluster drift")
    legacy_request = v1.job_request(_legacy_plan(plan))
    contract = miles_wandb.build_contract(
        plan["wandb"], plan, "sha256:" + digest(plan)
    )
    files = _runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    env = {
        key: value
        for key, value in legacy_request["env"].items()
        if not key.startswith("CYBER_RUNTIME_BUNDLE")
    }
    env.update(miles_wandb.request_env(contract))
    return bundled_request(
        {
            key: legacy_request[key]
            for key in (
                "name",
                "run_dir",
                "image",
                "workers",
                "gpus_per_worker",
                "resources",
                "priority_class",
                "requeueIfPreempted",
                "secrets",
            )
        }
        | {"title": plan["run_name"] + " comparative native Miles RL", "env": env},
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def check_artifacts(plan: dict):
    return v1.check_artifacts(plan)


def native_source():
    return v1.native_source()


def native_args(plan: dict):
    args = v1.native_args(_legacy_plan(plan))
    contract = miles_wandb.build_contract(
        plan["wandb"], plan, "sha256:" + digest(plan)
    )
    args.wandb_group = contract["group"]
    args.wandb_run_id = contract["run_id"]
    return args


def preflight(plan: dict) -> dict:
    legacy = v1.preflight(_legacy_plan(plan))
    contract = miles_wandb.build_contract(
        plan["wandb"], plan, "sha256:" + digest(plan)
    )
    return {
        **legacy,
        "schema": "cyber_miles_training_cpu_preflight_v2",
        "plan_sha256": digest(plan),
        "request_sha256": digest(job_request(plan)),
        "wandb_contract_sha256": contract["sha256"],
        "wandb_history_keys": list(miles_wandb.HISTORY_KEYS),
        "fresh_dev_qualification_required": True,
        "rl_qualified": False,
    }


def _native(plan: dict) -> None:
    import ray
    from miles.utils.tracking_utils.tracking import finish_tracking

    from .miles_promotion import validate_embedded_promotion

    validate_embedded_promotion(plan, check_files=False)
    check_artifacts(plan)
    args = native_args(plan)
    contract = miles_wandb.build_contract(
        plan["wandb"], plan, "sha256:" + digest(plan)
    )
    miles_wandb.install(contract)
    source = native_source()
    spec = importlib.util.spec_from_file_location("cyber_native_miles_train", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ray.init(
        address="auto",
        log_to_driver=False,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(Path(__file__).resolve().parents[1])
                + ":"
                + os.environ.get("PYTHONPATH", ""),
                "MILES_USE_LEGACY_ROLLOUT_V1": "0",
                **miles_wandb.worker_env(contract),
            },
            "worker_process_setup_hook": "training.miles_wandb.install_worker",
        },
    )
    try:
        asyncio.run(module.train(args))
    except BaseException as exc:
        from .rl_runtime import native_failure, native_rejection

        if native_rejection(plan, exc):
            return
        with suppress(Exception):
            native_failure(plan, exc)
        raise
    finally:
        try:
            finish_tracking()
        finally:
            ray.shutdown()


def native_result(plan: dict) -> dict:
    value = v1.native_result(plan)
    value["wandb_contract_sha256"] = miles_wandb.build_contract(
        plan["wandb"], plan, "sha256:" + digest(plan)
    )["sha256"]
    return value


def run(plan: dict, plan_path: Path):
    from .rl_runtime import run as supervised_run

    return supervised_run(plan, plan_path, backend=sys.modules[__name__])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        if args.native:
            _native(plan)
        else:
            result = run(plan, args.plan)
            print(json.dumps({key: result[key] for key in ("status", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
