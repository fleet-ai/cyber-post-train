"""Idempotent execution wrapper for a sealed post-training plan.

The wrapper is intentionally framework-thin: GLM-5.2 needs a cluster-validated
NeMo/Megatron integration before its exact entry point is trustworthy. Commands
are injected as argv strings by the cluster job, while this module owns stage
ordering, dry-run defaults, receipts, resume behavior and eval hooks.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shlex
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import atomic_write_json, digest_json
from .secrets import contains_secret, secret_values

COMMAND_ENV = {
    "sft": "SFT_TRAIN_COMMAND",
    "pre_rl_eval": "CYBER_EVAL_HOOK",
    "online_rl": "RL_TRAIN_COMMAND",
    "post_rl_eval": "CYBER_EVAL_HOOK",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def _event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def stage_argv(stage_name: str, env: Mapping[str, str]) -> list[str]:
    variable = COMMAND_ENV[stage_name]
    command = env.get(variable, "").strip()
    if not command:
        raise ValueError(f"{variable} must contain the cluster-validated command")
    argv = shlex.split(command)
    if not argv:
        raise ValueError(f"{variable} parsed to an empty command")
    if contains_secret(argv, secret_values()):
        raise ValueError(f"{variable} embeds a secret; inject credentials through environment")
    return argv


def run_plan(
    plan_path: Path,
    work_dir: Path,
    *,
    execute: bool = False,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    plan = _load(plan_path)
    if plan.get("schema") != "cyber_post_train_run_plan_v1":
        raise ValueError("invalid run plan schema")
    plan_digest = plan.get("plan_digest")
    unsigned = {key: value for key, value in plan.items() if key != "plan_digest"}
    if plan_digest != digest_json(unsigned):
        raise ValueError("run plan digest mismatch")
    environment = dict(os.environ if env is None else env)
    work_dir.mkdir(parents=True, exist_ok=True)
    events_path = work_dir / "events.jsonl"
    outcomes: list[dict[str, Any]] = []

    for stage in plan.get("stages") or []:
        name = str(stage.get("name"))
        if name not in COMMAND_ENV:
            raise ValueError(f"unsupported stage in plan: {name}")
        receipt_path = work_dir / "receipts" / f"{name}.json"
        if receipt_path.exists() and _load(receipt_path).get("status") == "succeeded":
            outcomes.append({"stage": name, "status": "already_succeeded"})
            continue
        argv = stage_argv(name, environment)
        public_argv = [str(item) for item in argv]
        if not execute:
            outcomes.append({"stage": name, "status": "dry_run", "argv": public_argv})
            continue

        stage_env = dict(environment)
        stage_env.update(
            {
                "CYBER_RUN_PLAN": str(plan_path.resolve()),
                "CYBER_RUN_PLAN_DIGEST": str(plan_digest),
                "CYBER_STAGE": name,
                "CYBER_EVAL_PHASE": name if name.endswith("eval") else "",
            }
        )
        started = dt.datetime.now(dt.UTC).isoformat()
        _event(events_path, {"stage": name, "event": "started", "at": started})
        completed = subprocess.run(argv, env=stage_env, check=False)
        finished = dt.datetime.now(dt.UTC).isoformat()
        receipt = {
            "schema": "cyber_post_train_stage_receipt_v1",
            "plan_digest": plan_digest,
            "stage": name,
            "status": "succeeded" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "started_at": started,
            "finished_at": finished,
        }
        receipt["receipt_digest"] = digest_json(receipt)
        atomic_write_json(receipt_path, receipt)
        _event(
            events_path,
            {
                "stage": name,
                "event": receipt["status"],
                "at": finished,
                "receipt_digest": receipt["receipt_digest"],
            },
        )
        outcomes.append({"stage": name, "status": receipt["status"]})
        if completed.returncode:
            raise RuntimeError(f"stage {name} failed with return code {completed.returncode}")
    return outcomes
