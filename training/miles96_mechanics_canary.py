"""A small, current Miles/FTI mechanics canary for Qwen3.8-27B.

This is deliberately not a replacement for the 262K SkyRL science rail.  It
uses Fleet's maintained 96K Qwen Miles recipe to prove a much narrower fact:
one fresh V1 task group can obtain real, varying verifier rewards, make one
GRPO update, export an HF checkpoint, and reload that checkpoint on one GPU.

The module only renders Jobs API requests and supplies their runtime entry
points.  It does not contact Fleet or create a workload.  An operator must
first bind a fresh identity, a fresh authoritative task observation, and a
prepared SFS model tree, then perform the normal server-preview/absence gates.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from cyber_post_train.jobs import bundled_request

SCHEMA = "cyber_qwen38_miles96_mechanics_canary_v1"
TRAIN_RECEIPT_SCHEMA = "cyber_qwen38_miles96_update_export_receipt_v1"
RELOAD_RECEIPT_SCHEMA = "cyber_qwen38_miles96_reload_receipt_v1"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "ee273bee346ad8e1cea63d18026b2bc5703bbd2e14d3c65efbbd74f19854874a"
)
THESEUS_COMMIT = "f2b0cb5db7c0a9dcc2210943f2ee1df31f9b50fc"
MILES_COMMIT = "9e178ca16839b0600155f3927f57ce0670b8f453"
FTI_VERSION = "0.10.10"
RUN_FLEET_SHA256 = "6f396d9fce6344ac7a8d6a3b5884c2258d50b2760c343d2d37eded2c5c25d582"
RECIPE = "qwen3.8-27b"
MODEL = "Qwen/Qwen3.8-27B"
CONTEXT_TOKENS = 98_304
RESPONSE_TOKENS = 81_920
REWARD_FILTER = "fti.trainers.miles.filters.check_no_aborted_and_nonzero_std"
TRAIN_RESOURCES = {
    "cpu_request": "48",
    "cpu_limit": "64",
    "memory_request": "1500Gi",
    "memory_limit": "2650Gi",
}
RELOAD_RESOURCES = {
    "cpu_request": "12",
    "cpu_limit": "16",
    "memory_request": "128Gi",
    "memory_limit": "256Gi",
}
LAUNCH_FILE = "LAUNCH.json"
UPDATE_FILE = "UPDATE_AND_EXPORT.json"
RELOAD_FILE = "RELOAD_VALIDATED.json"


def digest(value: Any) -> str:
    """Canonical digest used for plans and sanitized receipts."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ValueError(f"{field} must be a sha256 digest")
    return value


def _uuid(value: object, field: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"{field} must be a UUID") from None


def _name(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", value
    ):
        raise ValueError(f"{field} must be a DNS label of at most 31 characters")
    return value


def _sfs_path(value: object, field: str, *, exact_leaf: str | None = None) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an SFS path")
    path = PurePosixPath(value)
    if (
        path.parts[:4] != ("/", "mnt", "sfs", "jobs")
        or len(path.parts) < 5
        or ".." in path.parts
        or str(path) != value
        or (exact_leaf is not None and path.name != exact_leaf)
    ):
        raise ValueError(f"{field} must be a canonical path below /mnt/sfs/jobs")
    return value


def _require_relative(path: Path, root: Path, field: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"{field} escapes its run directory") from None


def _write_once(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def build_plan(
    *,
    name: str,
    reload_name: str,
    model_root: str,
    model_binding_sha256: str,
    task_binding: dict[str, str],
) -> dict[str, Any]:
    """Make one fresh plan from independently observed inputs.

    ``task_binding`` must come from an immediately preceding read-only
    authoritative task/version check.  This function intentionally cannot
    infer a task, reuse a predecessor, or stage a private row from history.
    """
    plan = {
        "schema": SCHEMA,
        "identity": {
            "name": name,
            "run_dir": f"/mnt/sfs/jobs/{name}",
            "reload_name": reload_name,
            "reload_run_dir": f"/mnt/sfs/jobs/{reload_name}",
        },
        "trainer": {
            "model": MODEL,
            "recipe": RECIPE,
            "image": IMAGE,
            "theseus_commit": THESEUS_COMMIT,
            "miles_commit": MILES_COMMIT,
            "fti_version": FTI_VERSION,
            "run_fleet_sha256": RUN_FLEET_SHA256,
        },
        "prepared_model": {
            "root": model_root,
            "binding_sha256": model_binding_sha256,
        },
        "task_binding": task_binding,
        "episode": {
            "max_turns": 32,
            "max_tokens_per_turn": 8192,
            "max_concurrent_envs": 8,
            "request_timeout_s": 120,
            "ready_timeout_s": 600,
            "episode_timeout_s": 2400,
            "tool_timeout_s": 330,
            "grade_timeout_s": 900,
            "ttl_seconds": 7200,
            "tool_output_max_chars": 50000,
            "vision": False,
            "pass_conversation_to_verifier": False,
        },
        "optimization": {
            "prompt_groups": 1,
            "samples_per_prompt": 8,
            "optimizer_steps": 1,
            "learning_rate": 1e-6,
            "checkpoint_interval": 1,
            "seed": 20260921,
            "reward_filter": REWARD_FILTER,
        },
        "wandb": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": name,
        },
        "cluster": {
            "nodes": 1,
            "gpus_per_node": 8,
            "priority": "c1",
            "topology_mode": "preferred",
            "resources": TRAIN_RESOURCES,
        },
        "acceptance": {
            "fresh_v1_train_rows": False,
            "finite_rewards": False,
            "reward_variation": False,
            "optimizer_update": False,
            "hf_export": False,
            "one_gpu_reload": False,
            "instances_released": False,
            "gpus_released": False,
        },
    }
    return validate_plan(plan)


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Reject any drift from the bounded one-node mechanics contract."""
    value = json.loads(json.dumps(plan))
    expected_top = {
        "schema",
        "identity",
        "trainer",
        "prepared_model",
        "task_binding",
        "episode",
        "optimization",
        "wandb",
        "cluster",
        "acceptance",
    }
    if set(value) != expected_top or value.get("schema") != SCHEMA:
        raise ValueError("unsupported Miles96 mechanics plan")

    identity = value["identity"]
    if set(identity) != {"name", "run_dir", "reload_name", "reload_run_dir"}:
        raise ValueError("mechanics identity fields changed")
    name = _name(identity["name"], "training name")
    reload_name = _name(identity["reload_name"], "reload name")
    if name == reload_name:
        raise ValueError("training and reload identities must differ")
    _sfs_path(identity["run_dir"], "training run_dir", exact_leaf=name)
    _sfs_path(identity["reload_run_dir"], "reload run_dir", exact_leaf=reload_name)

    expected_trainer = {
        "model": MODEL,
        "recipe": RECIPE,
        "image": IMAGE,
        "theseus_commit": THESEUS_COMMIT,
        "miles_commit": MILES_COMMIT,
        "fti_version": FTI_VERSION,
        "run_fleet_sha256": RUN_FLEET_SHA256,
    }
    if value["trainer"] != expected_trainer:
        raise ValueError("maintained 96K Miles trainer binding drift")

    prepared = value["prepared_model"]
    if set(prepared) != {"root", "binding_sha256"}:
        raise ValueError("prepared-model binding fields changed")
    _sfs_path(prepared["root"], "prepared model root")
    _sha256(prepared["binding_sha256"], "prepared model binding")
    if prepared["root"].startswith(identity["run_dir"] + "/"):
        raise ValueError("prepared model must not be a training output")

    task = value["task_binding"]
    expected_task_fields = {
        "task_key",
        "task_version_id",
        "task_set_sha256",
        "tool_catalog_sha256",
        "authority_receipt_sha256",
    }
    if set(task) != expected_task_fields:
        raise ValueError("task binding fields changed")
    if not isinstance(task["task_key"], str) or not task["task_key"].strip():
        raise ValueError("task binding needs an exact task key")
    _uuid(task["task_version_id"], "task version")
    for field in expected_task_fields - {"task_key", "task_version_id"}:
        _sha256(task[field], field)

    expected_episode = {
        "max_turns": 32,
        "max_tokens_per_turn": 8192,
        "max_concurrent_envs": 8,
        "request_timeout_s": 120,
        "ready_timeout_s": 600,
        "episode_timeout_s": 2400,
        "tool_timeout_s": 330,
        "grade_timeout_s": 900,
        "ttl_seconds": 7200,
        "tool_output_max_chars": 50000,
        "vision": False,
        "pass_conversation_to_verifier": False,
    }
    if value["episode"] != expected_episode:
        raise ValueError("bounded V1 episode contract drift")

    expected_optimization = {
        "prompt_groups": 1,
        "samples_per_prompt": 8,
        "optimizer_steps": 1,
        "learning_rate": 1e-6,
        "checkpoint_interval": 1,
        "seed": 20260921,
        "reward_filter": REWARD_FILTER,
    }
    if value["optimization"] != expected_optimization:
        raise ValueError("one-update reward-variation mechanics contract drift")
    if value["wandb"] != {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "run_id": name,
    }:
        raise ValueError("W&B identity drift")
    if value["cluster"] != {
        "nodes": 1,
        "gpus_per_node": 8,
        "priority": "c1",
        "topology_mode": "preferred",
        "resources": TRAIN_RESOURCES,
    }:
        raise ValueError("one-node c1 cluster contract drift")
    required_acceptance = {
        "fresh_v1_train_rows",
        "finite_rewards",
        "reward_variation",
        "optimizer_update",
        "hf_export",
        "one_gpu_reload",
        "instances_released",
        "gpus_released",
    }
    if set(value["acceptance"]) != required_acceptance or any(value["acceptance"].values()):
        raise ValueError("acceptance starts false and is set only by independent evidence")
    return value


def task_rows(plan: dict[str, Any]) -> bytes:
    """Return a newly written V1 row without a task prompt or a stored trace."""
    plan = validate_plan(plan)
    row = {
        "messages": [{"role": "user", "content": "Run the versioned Fleet task."}],
        "label": plan["task_binding"]["task_key"],
        "metadata": {
            "task_key": plan["task_binding"]["task_key"],
            "task_version_id": plan["task_binding"]["task_version_id"],
            "fleet": plan["episode"],
        },
    }
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _wandb_args(plan: dict[str, Any]) -> str:
    """Use W&B without ever placing its credential in an argv string."""
    wandb = plan["wandb"]
    return " ".join(
        (
            "--use-wandb",
            "--disable-wandb-random-suffix",
            f"--wandb-team {wandb['entity']}",
            f"--wandb-project {wandb['project']}",
            f"--wandb-group {plan['identity']['name']}",
            f"--wandb-run-id {wandb['run_id']}",
        )
    )


def install_safe_wandb() -> None:
    """Patch the maintained helper before it serializes WANDB_API_KEY into argv."""
    import miles.utils.external_utils.command_utils as command_utils

    command_utils.get_default_wandb_args = lambda *_args, **_kwargs: (
        "--use-wandb --disable-wandb-random-suffix" if os.environ.get("WANDB_API_KEY") else ""
    )


SITECUSTOMIZE = (
    "from training.miles96_mechanics_canary import install_safe_wandb\ninstall_safe_wandb()\n"
)


def native_arguments(plan: dict[str, Any]) -> list[str]:
    """Use the maintained direct ``run_fleet`` route, not its legacy run.sh."""
    plan = validate_plan(plan)
    identity, optim = plan["identity"], plan["optimization"]
    run_dir = identity["run_dir"]
    extra = " ".join(
        (
            "--num-rollout 1",
            "--save-interval 1",
            f"--save-hf {run_dir}/hf/step-{{rollout_id}}",
            f"--lr {optim['learning_rate']}",
            f"--seed {optim['seed']}",
            f"--rollout-seed {optim['seed']}",
            f"--dynamic-sampling-filter-path {REWARD_FILTER}",
            "--custom-megatron-post-save-hook-path "
            "training.miles96_mechanics_canary.post_save_hook",
            _wandb_args(plan),
        )
    )
    runtime_env = json.dumps(
        {
            "PYTHONPATH": run_dir + "/.runtime",
            "CYBER_RUNTIME_DIR": run_dir + "/.runtime",
            "CYBER_PLAN_SHA256": digest(plan),
        },
        separators=(",", ":"),
    )
    return [
        "-m",
        "fti.trainers.miles.run_fleet",
        "--model-name",
        RECIPE,
        "--platform",
        "v1",
        "--mode",
        "normal",
        "--num-nodes",
        "1",
        "--num-gpus-per-node",
        "8",
        "--rollout-batch-size",
        "1",
        "--n-samples-per-prompt",
        "8",
        "--run-id",
        identity["name"],
        "--dataset-dir",
        run_dir + "/data",
        "--model-dir",
        plan["prepared_model"]["root"],
        "--data-dir",
        run_dir,
        "--output-dir",
        run_dir,
        "--checkpoint-dir",
        run_dir + "/model-output/checkpoints",
        "--extra-env-vars",
        runtime_env,
        "--extra-args",
        extra,
    ]


def _runtime_files(plan: dict[str, Any]) -> dict[str, str]:
    return {
        "sitecustomize.py": SITECUSTOMIZE,
        "training/__init__.py": "",
        "training/miles96_mechanics_canary.py": Path(__file__).read_text(),
        "plan.json": json.dumps(plan, sort_keys=True, separators=(",", ":")),
    }


def job_request(plan: dict[str, Any]) -> dict[str, Any]:
    """Render a one-node request; this function cannot submit it."""
    plan = validate_plan(plan)
    identity = plan["identity"]
    request = {
        "name": identity["name"],
        "title": "Qwen3.8 96K Miles V1 mechanics canary",
        "run_dir": identity["run_dir"],
        "image": IMAGE,
        "command": "placeholder",
        "workers": 1,
        "gpus_per_worker": 8,
        "priority_class": "c1",
        "topology_mode": "preferred",
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "privileged": True,
        "resources": TRAIN_RESOURCES,
        "secrets": ["fleet-api", "wandb-api"],
        "image_pull_secrets": ["ecr-pull"],
        "env": {
            "MODEL_DIR": plan["prepared_model"]["root"],
            "FLEET_MODEL_DIR": identity["run_dir"] + "/model-output",
            "PYTHONPATH": identity["run_dir"] + "/.runtime",
            "MILES_SCRIPT_EXTERNAL_RAY": "1",
            "WANDB_ENTITY": plan["wandb"]["entity"],
            "WANDB_MODE": "online",
            "WANDB_RESUME": "never",
            "WANDB_DISABLE_CODE": "true",
            "WANDB_CONSOLE": "off",
            "CYBER_PLAN_SHA256": digest(plan),
            "CYBER_MODEL_BINDING_SHA256": plan["prepared_model"]["binding_sha256"],
            "CYBER_TASK_BINDING_SHA256": "sha256:" + digest(plan["task_binding"]),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        },
    }
    return bundled_request(
        request,
        _runtime_files(plan),
        "training.miles96_mechanics_canary",
        ["--train", "--plan", "plan.json", "--sha256", digest(plan)],
    )


def _runtime_recipe_binding() -> None:
    import fti
    from fti.trainers.miles import run_fleet

    if fti.__version__ != FTI_VERSION:
        raise ValueError("FTI version drift")
    if file_sha256(Path(run_fleet.__file__)) != RUN_FLEET_SHA256:
        raise ValueError("maintained run_fleet source drift")
    recipe = run_fleet._RECIPES.get(RECIPE)
    expected = {
        "max_context_len": CONTEXT_TOKENS,
        "max_response_len": RESPONSE_TOKENS,
        "max_tokens_per_gpu": 49_152,
        "rollout_num_gpus_per_engine": 1,
        "megatron_model_type": "qwen3.8-27B",
        "vision": False,
    }
    if recipe is None or any(getattr(recipe, key, None) != item for key, item in expected.items()):
        raise ValueError("maintained Qwen 96K recipe drift")
    shape = recipe.parallel_args_by_shape.get((1, 8))
    if (
        not isinstance(shape, str)
        or "--tensor-model-parallel-size 4" not in shape
        or ("--context-parallel-size 2" not in shape)
    ):
        raise ValueError("maintained one-node TP4/CP2 shape drift")


def _prepared_model_exists(plan: dict[str, Any]) -> None:
    root = Path(plan["prepared_model"]["root"])
    if not (root / "Qwen3.8-27B" / "config.json").is_file():
        raise ValueError("prepared HF model is absent")
    tracker = root / "qwen3.8-27B_torch_dist" / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text().strip() != "release":
        raise ValueError("prepared Megatron reference is absent or incomplete")


def _runtime_paths(plan: dict[str, Any]) -> tuple[Path, Path, Path]:
    run_dir = Path(plan["identity"]["run_dir"])
    data_dir = run_dir / "data"
    output_dir = run_dir / "model-output"
    if str(run_dir) != plan["identity"]["run_dir"]:
        raise ValueError("run directory changed")
    for item, name in ((data_dir, "data path"), (output_dir, "output path")):
        _require_relative(item, run_dir, name)
    return run_dir, data_dir, output_dir


def _stage_train_inputs(plan: dict[str, Any], expected_digest: str) -> None:
    run_dir, data_dir, output_dir = _runtime_paths(plan)
    if not run_dir.is_dir():
        raise ValueError("Jobs API did not initialize the create-once run directory")
    if data_dir.exists() or output_dir.exists() or (run_dir / LAUNCH_FILE).exists():
        raise FileExistsError("a mechanics-canary output already exists")
    if os.environ.get("RUN_DIR") != str(run_dir):
        raise ValueError("runtime output identity drift")
    if os.environ.get("MODEL_DIR") != plan["prepared_model"]["root"]:
        raise ValueError("prepared model path drift")
    if os.environ.get("FLEET_MODEL_DIR") != str(output_dir):
        raise ValueError("model-output path drift")
    if os.environ.get("CYBER_PLAN_SHA256") != expected_digest:
        raise ValueError("runtime plan identity drift")
    if os.environ.get("CYBER_MODEL_BINDING_SHA256") != plan["prepared_model"]["binding_sha256"]:
        raise ValueError("prepared model binding drift")
    if os.environ.get("CYBER_TASK_BINDING_SHA256") != "sha256:" + digest(plan["task_binding"]):
        raise ValueError("task binding drift")
    _runtime_recipe_binding()
    _prepared_model_exists(plan)
    data_dir.mkdir(mode=0o700)
    output_dir.mkdir(mode=0o700)
    rows = task_rows(plan)
    _write_once(data_dir / "train.jsonl", rows)
    launch = {
        "schema": "cyber_qwen38_miles96_launch_v1",
        "plan_sha256": expected_digest,
        "dataset_sha256": "sha256:" + hashlib.sha256(rows).hexdigest(),
        "row_count": 1,
        "trainer_image": IMAGE,
        "fti_version": FTI_VERSION,
        "recipe": RECIPE,
        "context_tokens": CONTEXT_TOKENS,
        "compaction": False,
    }
    _write_once(
        run_dir / LAUNCH_FILE,
        json.dumps(launch, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def run_train(plan: dict[str, Any], expected_digest: str) -> None:
    plan = validate_plan(plan)
    if digest(plan) != expected_digest:
        raise ValueError("runtime plan digest mismatch")
    _stage_train_inputs(plan, expected_digest)
    os.execvpe(sys.executable, [sys.executable, *native_arguments(plan)], os.environ)


def _load_runtime_plan() -> dict[str, Any]:
    runtime_dir = Path(os.environ.get("CYBER_RUNTIME_DIR", ""))
    if not runtime_dir.is_dir():
        raise ValueError("Miles worker runtime source is absent")
    plan = json.loads((runtime_dir / "plan.json").read_text())
    return validate_plan(plan)


def _train_receipt(
    plan: dict[str, Any], *, rollout_id: int, checkpoint_dir: str, hf_dir: str
) -> dict[str, Any]:
    run_dir = Path(plan["identity"]["run_dir"])
    checkpoint = Path(checkpoint_dir)
    expected_hf = run_dir / "hf" / "step-1"
    if rollout_id != 1 or Path(hf_dir) != expected_hf:
        raise ValueError("post-save hook received an unexpected checkpoint identity")
    _require_relative(checkpoint, run_dir / "model-output" / "checkpoints", "checkpoint path")
    if not checkpoint.exists() or not (expected_hf / ".complete").is_file():
        raise ValueError("post-save hook lacks a completed checkpoint/export")
    body = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "optimizer_steps": 1,
        "reward_filter": REWARD_FILTER,
        "checkpoint_path_sha256": "sha256:" + hashlib.sha256(str(checkpoint).encode()).hexdigest(),
        "hf_export_path_sha256": "sha256:" + hashlib.sha256(str(expected_hf).encode()).hexdigest(),
        "hf_export_complete": True,
        "save_hook_after_checkpoint": True,
    }
    return {**body, "receipt_sha256": "sha256:" + digest(body)}


def post_save_hook(
    args: Any, rollout_id: int, checkpoint_dir: str, hf_checkpoint_dir: str | None
) -> None:
    """Maintained Miles rank-zero hook: seal only a successful first update/export."""
    del args
    plan = _load_runtime_plan()
    if hf_checkpoint_dir is None:
        raise ValueError("Miles did not provide its HF export path")
    if os.environ.get("CYBER_PLAN_SHA256") != digest(plan):
        raise ValueError("post-save plan digest drift")
    receipt = _train_receipt(
        plan,
        rollout_id=rollout_id,
        checkpoint_dir=checkpoint_dir,
        hf_dir=hf_checkpoint_dir,
    )
    _write_once(
        Path(plan["identity"]["run_dir"]) / UPDATE_FILE,
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def validate_train_receipt(plan: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    plan = validate_plan(plan)
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "optimizer_steps": 1,
        "reward_filter": REWARD_FILTER,
        "hf_export_complete": True,
        "save_hook_after_checkpoint": True,
    }
    expected_fields = set(expected) | {"checkpoint_path_sha256", "hf_export_path_sha256"}
    if (
        set(body) != expected_fields
        or any(body[key] != item for key, item in expected.items())
        or receipt.get("receipt_sha256") != "sha256:" + digest(body)
    ):
        raise ValueError("invalid update/export receipt")
    _sha256(body["checkpoint_path_sha256"], "checkpoint path")
    _sha256(body["hf_export_path_sha256"], "HF export path")
    return receipt


def reload_request(plan: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Render the separate one-GPU, zero-optimizer reload observer."""
    plan = validate_plan(plan)
    receipt = validate_train_receipt(plan, receipt)
    identity = plan["identity"]
    request = {
        "name": identity["reload_name"],
        "title": "Qwen3.8 96K Miles checkpoint reload observer",
        "run_dir": identity["reload_run_dir"],
        "image": IMAGE,
        "command": "placeholder",
        "workers": 1,
        "gpus_per_worker": 1,
        "priority_class": "c1",
        "topology_mode": "preferred",
        "requeueIfPreempted": False,
        "failureAlerts": False,
        "privileged": False,
        "resources": RELOAD_RESOURCES,
        "secrets": ["fleet-api"],
        "image_pull_secrets": ["ecr-pull"],
        "env": {
            "PYTHONPATH": identity["reload_run_dir"] + "/.runtime",
            "CYBER_PLAN_SHA256": digest(plan),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        },
    }
    files = _runtime_files(plan)
    files["receipt.json"] = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        request,
        files,
        "training.miles96_mechanics_canary",
        ["--reload", "--plan", "plan.json", "--receipt", "receipt.json"],
    )


def _reload_runtime_binding() -> None:
    _runtime_recipe_binding()
    from fti.miles.v1 import client_recording

    signature = inspect.signature(client_recording.generate)
    if tuple(signature.parameters) != ("input",):
        raise ValueError("maintained V1 reload API drift")


def _wait_health(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 900
    while True:
        if process.poll() is not None:
            raise RuntimeError("SGLang stopped before its health check")
        if time.monotonic() >= deadline:
            raise TimeoutError("SGLang did not become ready")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health_generate", timeout=5):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(5)


async def _reload_episode(plan: dict[str, Any], model_path: Path) -> dict[str, bool]:
    from fti.miles.v1.client_recording import generate
    from miles.rollout.base_types import GenerateFnInput
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
    from miles.utils.chat_template_utils import resolve_fixed_chat_template
    from miles.utils.http_utils import init_http_client
    from miles.utils.types import Sample

    template_path, _ = resolve_fixed_chat_template("qwen35")
    args = argparse.Namespace(
        hf_checkpoint=str(model_path),
        chat_template_path=template_path,
        custom_generate_function_path="fti.miles.v1.client_recording.generate",
        fleet_tito_model="qwen35",
        partial_rollout=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        sglang_router_policy="round_robin",
        sglang_server_concurrency=1,
        sglang_speculative_algorithm=None,
        rollout_num_gpus=1,
        eval_num_gpus=0,
        rollout_num_gpus_per_engine=1,
        rollout_temperature=1,
        rollout_top_p=1,
        rollout_top_k=-1,
        rollout_max_response_len=RESPONSE_TOKENS,
        rollout_max_context_len=CONTEXT_TOKENS,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_distributed_post=False,
    )
    init_http_client(args)
    binding = plan["task_binding"]
    sample = Sample(
        index=0,
        prompt=[{"role": "user", "content": "Run the versioned Fleet task."}],
        metadata={
            "task_key": binding["task_key"],
            "task_version_id": binding["task_version_id"],
            "fleet": plan["episode"],
        },
    )
    state = GenerateState(args)
    generated = GenerateFnInput(state, sample, state.sampling_params, evaluation=True)
    output = await asyncio.wait_for(
        generate(generated),
        timeout=plan["episode"]["episode_timeout_s"],
    )
    samples = output.samples if isinstance(output.samples, list) else [output.samples]
    if len(samples) != 1:
        raise ValueError("reload observer returned an unexpected sample count")
    item = samples[0]
    fleet = (item.metadata or {}).get("fleet_v1")
    if (
        item.status != Sample.Status.COMPLETED
        or not isinstance(item.reward, (int, float))
        or isinstance(item.reward, bool)
        or not math.isfinite(float(item.reward))
        or not isinstance(fleet, dict)
        or fleet.get("graded") is not True
        or fleet.get("cleanup_error") is not None
        or not isinstance(fleet.get("tool_calls"), int)
        or fleet["tool_calls"] < 1
        or not isinstance(item.response_length, int)
        or item.response_length < 1
    ):
        raise ValueError("reload observer did not receive a finite, cleaned V1 reward")
    return {
        "finite_reward": True,
        "verifier_graded": True,
        "task_cleanup_confirmed": True,
        "tool_call_seen": True,
    }


def run_reload(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
    plan = validate_plan(plan)
    validate_train_receipt(plan, receipt)
    identity = plan["identity"]
    run_dir = Path(identity["reload_run_dir"])
    model_path = Path(identity["run_dir"]) / "hf" / "step-1"
    if os.environ.get("RUN_DIR") != str(run_dir) or os.environ.get("CYBER_PLAN_SHA256") != digest(
        plan
    ):
        raise ValueError("reload runtime identity drift")
    if not run_dir.is_dir() or (run_dir / RELOAD_FILE).exists():
        raise FileExistsError("reload output is not create-once")
    if not (model_path / ".complete").is_file() or not (model_path / "config.json").is_file():
        raise ValueError("the first HF export is not complete")
    _reload_runtime_binding()
    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError("reload observer must have exactly one GPU")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            "30000",
            "--mem-fraction-static",
            "0.75",
            "--context-length",
            str(CONTEXT_TOKENS),
            "--tp-size",
            "1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_health(30000, process)
        checks = asyncio.run(_reload_episode(plan, model_path))
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    body = {
        "schema": RELOAD_RECEIPT_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "checkpoint_receipt_sha256": receipt["receipt_sha256"],
        "context_tokens": CONTEXT_TOKENS,
        "optimizer_steps": 0,
        "engine_stopped": True,
        **checks,
    }
    payload = {**body, "receipt_sha256": "sha256:" + digest(body)}
    _write_once(
        run_dir / RELOAD_FILE,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--reload", action="store_true")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--receipt")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if args.train:
        if not args.sha256 or args.receipt:
            parser.error("--train requires --sha256 and forbids --receipt")
        run_train(plan, args.sha256)
    else:
        if args.sha256 or not args.receipt:
            parser.error("--reload requires --receipt and forbids --sha256")
        run_reload(plan, json.loads(Path(args.receipt).read_text()))


if __name__ == "__main__":
    main()
