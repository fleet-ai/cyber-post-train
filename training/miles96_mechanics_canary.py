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
import ctypes
import errno
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "cyber_qwen38_miles96_mechanics_canary_v1"
TRAIN_RECEIPT_SCHEMA = "cyber_qwen38_miles96_update_export_receipt_v1"
RELOAD_RECEIPT_SCHEMA = "cyber_qwen38_miles96_reload_receipt_v1"
REWARD_GATE_SCHEMA = "cyber_qwen38_miles96_selected_reward_group_v1"
PRIVATE_EPISODE_SCHEMA = "cyber_qwen38_miles96_private_episode_v1"
COMPLETE_MODEL_SCHEMA = "cyber_qwen38_miles96_complete_model_v1"
CHECKPOINT_MANIFEST_SCHEMA = "cyber_qwen38_miles96_checkpoint_manifest_v1"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "ee273bee346ad8e1cea63d18026b2bc5703bbd2e14d3c65efbbd74f19854874a"
)
# This is the source commit used to build the pinned 0.10.9 image.  The later
# 0.10.10 source exists in Theseus, but its changelog explicitly says that no
# image was built.  Never claim 0.10.10 for the 0.10.9 image below.
THESEUS_COMMIT = "224d6b81cb698f4785fb16123314583b981493f3"
MILES_COMMIT = "9e178ca16839b0600155f3927f57ce0670b8f453"
FTI_VERSION = "0.10.9"
RUN_FLEET_SHA256 = "340be7e1d1976fdd5f42f1ed934c07ae8aa98cb520d9af21bda2933ddd146faa"
FTI_COMMON_SHA256 = "0a6801afe0cea5e4c0b6a53ffe07c083cc7e3691cf231dff460889e79c1626b0"
FTI_CLIENT_RECORDING_SHA256 = "41292533ec356a51a722c2c98f92bdfd98c56fdce429537a816957bf7172e9dc"
MILES_INFERENCE_ROLLOUT_SHA256 = "96e3cba12ae033527e823ed3dd8cb43c31d244775756ecf0bab4ad81eb4f06a4"
MILES_HTTP_UTILS_SHA256 = "da630d6594c86d76e89a262d1060c7f81238da9659899dea917987c5f39fab86"
MILES_MEGATRON_ACTOR_SHA256 = "eecd72a4387511916add2c97d9e2dad6716db9097fd6ec471468c1f7edc074b9"
MILES_HF_EXPORT_SHA256 = "4986684bb62acf2ccd0e18a5ab7cf6bd50391f42d42be35ad3a377b36bbd4a34"
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
REWARD_GATE_FILE = "REWARD_GATE.json"
COMPLETE_MODEL_FILE = "COMPLETE_MODEL.json"
PRIVATE_EVIDENCE_DIR = ".private-reward-evidence"
PRIVATE_SELECTED_FILE = "SELECTED_GROUP.json"
PRIVATE_CHECKPOINT_FILE = "CHECKPOINT_MANIFEST.json"
RAW_HF_STEP = "hf/step-0"
COMPLETE_HF_STEP = "hf-complete/step-0"
FROZEN_TENSOR_PREFIXES = ("model.visual.", "mtp.")
PROD_JOBS_API = "https://api.ft.flt.build"
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
NAMESPACE = "fleet-train-jobs"


def digest(value: Any) -> str:
    """Canonical digest used for plans and sanitized receipts."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


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


def _private_directory(run_dir: Path) -> Path:
    path = run_dir / PRIVATE_EVIDENCE_DIR
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o700:
        raise ValueError("private evidence directory is not mode 0700")
    return path


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
            "fti_common_sha256": FTI_COMMON_SHA256,
            "fti_client_recording_sha256": FTI_CLIENT_RECORDING_SHA256,
            "miles_inference_rollout_sha256": MILES_INFERENCE_ROLLOUT_SHA256,
            "miles_http_utils_sha256": MILES_HTTP_UTILS_SHA256,
            "miles_megatron_actor_sha256": MILES_MEGATRON_ACTOR_SHA256,
            "miles_hf_export_sha256": MILES_HF_EXPORT_SHA256,
        },
        "execution": {
            "cluster_target": "prod",
            "jobs_api_base_url": PROD_JOBS_API,
            "kubernetes_context": PROD_CONTEXT,
            "namespace": NAMESPACE,
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
        "execution",
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
        "fti_common_sha256": FTI_COMMON_SHA256,
        "fti_client_recording_sha256": FTI_CLIENT_RECORDING_SHA256,
        "miles_inference_rollout_sha256": MILES_INFERENCE_ROLLOUT_SHA256,
        "miles_http_utils_sha256": MILES_HTTP_UTILS_SHA256,
        "miles_megatron_actor_sha256": MILES_MEGATRON_ACTOR_SHA256,
        "miles_hf_export_sha256": MILES_HF_EXPORT_SHA256,
    }
    if value["trainer"] != expected_trainer:
        raise ValueError("maintained 96K Miles trainer binding drift")
    if value["execution"] != {
        "cluster_target": "prod",
        "jobs_api_base_url": PROD_JOBS_API,
        "kubernetes_context": PROD_CONTEXT,
        "namespace": NAMESPACE,
    }:
        raise ValueError("production execution binding drift")

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
        "verifier_version_id",
        "task_set_sha256",
        "tool_catalog_sha256",
        "authority_receipt_sha256",
    }
    if set(task) != expected_task_fields:
        raise ValueError("task binding fields changed")
    if not isinstance(task["task_key"], str) or not task["task_key"].strip():
        raise ValueError("task binding needs an exact task key")
    _uuid(task["task_version_id"], "task version")
    _uuid(task["verifier_version_id"], "verifier version")
    for field in expected_task_fields - {"task_key", "task_version_id", "verifier_version_id"}:
        _sha256(task[field], field)
    authority = {
        "task_key": task["task_key"],
        "task_version_id": task["task_version_id"],
        "verifier_version_id": task["verifier_version_id"],
        "task_set_sha256": task["task_set_sha256"],
        "tool_catalog_sha256": task["tool_catalog_sha256"],
    }
    if task["authority_receipt_sha256"] != "sha256:" + digest(authority):
        raise ValueError("task authority receipt digest is not self-consistent")

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
            # FTI 0.10.9 defaults to two candidate prompt groups.  This
            # mechanics canary deliberately collects exactly one group of
            # eight episodes, so bind the Miles override explicitly.
            "--over-sampling-batch-size 1",
            "--save-interval 1",
            f"--save-hf {run_dir}/hf/step-{{rollout_id}}",
            f"--lr {optim['learning_rate']}",
            f"--seed {optim['seed']}",
            f"--rollout-seed {optim['seed']}",
            f"--dynamic-sampling-filter-path {REWARD_FILTER}",
            "--custom-generate-function-path "
            "training.miles96_mechanics_canary.generate_with_evidence",
            "--rollout-sample-filter-path "
            "training.miles96_mechanics_canary.record_selected_reward_group",
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
    from cyber_post_train.jobs import bundled_request

    plan = validate_plan(plan)
    identity = plan["identity"]
    request = {
        "name": identity["name"],
        "title": f"Qwen3.8 96K Miles V1 mechanics canary: {identity['name']}",
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
    import miles
    from fti.miles.v1 import client_recording
    from fti.miles.v1 import common as fti_common
    from fti.trainers.miles import run_fleet

    if fti.__version__ != FTI_VERSION:
        raise ValueError("FTI version drift")
    if file_sha256(Path(run_fleet.__file__)) != RUN_FLEET_SHA256:
        raise ValueError("maintained run_fleet source drift")
    if file_sha256(Path(fti_common.__file__)) != FTI_COMMON_SHA256:
        raise ValueError("maintained V1 task-session source drift")
    if file_sha256(Path(client_recording.__file__)) != FTI_CLIENT_RECORDING_SHA256:
        raise ValueError("maintained V1 rollout source drift")
    miles_root = Path(miles.__file__).resolve().parent
    if (
        file_sha256(miles_root / "rollout/inference_rollout/inference_rollout_common.py")
        != MILES_INFERENCE_ROLLOUT_SHA256
    ):
        raise ValueError("maintained Miles inference-rollout source drift")
    if file_sha256(miles_root / "utils/http_utils.py") != MILES_HTTP_UTILS_SHA256:
        raise ValueError("maintained Miles HTTP source drift")
    if file_sha256(miles_root / "backends/megatron_utils/actor.py") != MILES_MEGATRON_ACTOR_SHA256:
        raise ValueError("maintained Miles checkpoint/export source drift")
    if file_sha256(miles_root / "backends/megatron_utils/hf_export.py") != MILES_HF_EXPORT_SHA256:
        raise ValueError("maintained Miles HF-export source drift")
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
        or recipe.extra_sglang_args != "--sglang-attention-backend triton "
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


_EVIDENCE_SESSION_CLASS: type | None = None


def _evidence_session_class() -> type:
    """Build the maintained V1 session subclass lazily inside the pinned image."""
    global _EVIDENCE_SESSION_CLASS
    if _EVIDENCE_SESSION_CLASS is not None:
        return _EVIDENCE_SESSION_CLASS

    from fti.fleet import GradeResult
    from fti.fleet.v1 import PlatformError
    from fti.miles.v1.common import TaskSession, numeric_reward

    class EvidenceTaskSession(TaskSession):
        verifier_execution_id: str | None = None
        authority_evidence_sha256: str | None = None

        def open(self) -> None:
            super().open()
            plan = _load_runtime_plan()
            binding = plan["task_binding"]
            if (
                self.task_key != binding["task_key"]
                or self.task_version_id != binding["task_version_id"]
                or self.verifier_version_id != binding["verifier_version_id"]
                or "sha256:" + digest(self.tools) != binding["tool_catalog_sha256"]
            ):
                self.close()
                raise ValueError("live V1 task authority differs from the immutable plan")

        def grade(self, answer, reset_ack=None, close_final_step=False):
            del reset_ack, close_final_step
            if self.closed.is_set():
                raise PlatformError("episode is closed")
            if self.instance is None or self.verifier_version_id is None:
                raise PlatformError("episode authority is incomplete")
            result = self.client.execute_verifier(
                self.verifier_version_id,
                self.instance.instance_id,
                final_answer=answer,
                conversation=self.conversation if self.cfg.pass_conversation_to_verifier else None,
                timeout_s=self.cfg.grade_timeout_s,
            )
            if result.get("success") is not True:
                raise PlatformError("registered verifier execution failed")
            execution_id = (
                result.get("verifier_execution_id") or result.get("job_id") or result.get("id")
            )
            self.verifier_execution_id = _uuid(execution_id, "verifier execution identity")
            verdict = result.get("result")
            value = verdict.get("result") if isinstance(verdict, dict) else verdict
            return GradeResult(reward=numeric_reward(value))

    _EVIDENCE_SESSION_CLASS = EvidenceTaskSession
    return EvidenceTaskSession


def _evidence_episode_metadata(session: Any, result: Any, stats: Any) -> dict[str, Any]:
    """Persist exact V1 authority privately and expose only its digest to Miles."""
    if session.authority_evidence_sha256 is not None:
        return {
            "authority_evidence_sha256": session.authority_evidence_sha256,
            "authority_verified": True,
            "cleanup_confirmed": True,
            "done_reason": result.done_reason,
            "turns": stats.turns,
            "round_number": stats.turns,
            "tool_calls": stats.tool_calls,
            "images": stats.images,
        }
    plan = _load_runtime_plan()
    if (
        session.instance is None
        or session.verifier_execution_id is None
        or not session.deleted
        or session.cleanup_error is not None
        or result.grade is None
    ):
        raise ValueError("V1 episode authority or cleanup evidence is incomplete")
    reward = float(result.grade.reward)
    if not math.isfinite(reward):
        raise ValueError("V1 episode reward is not finite")
    body = {
        "schema": PRIVATE_EPISODE_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "task_key": session.task_key,
        "task_version_id": _uuid(session.task_version_id, "task version"),
        "verifier_version_id": _uuid(session.verifier_version_id, "verifier version"),
        "instance_id": _uuid(session.instance.instance_id, "instance identity"),
        "verifier_execution_id": session.verifier_execution_id,
        "reward": reward,
        "cleanup_confirmed": True,
    }
    payload = {**body, "sha256": "sha256:" + digest(body)}
    evidence_dir = _private_directory(
        Path(os.environ.get("CYBER_EVIDENCE_RUN_DIR", plan["identity"]["run_dir"]))
    )
    _write_once(
        evidence_dir / (payload["sha256"].removeprefix("sha256:") + ".json"),
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    session.authority_evidence_sha256 = payload["sha256"]
    return {
        "authority_evidence_sha256": payload["sha256"],
        "authority_verified": True,
        "cleanup_confirmed": True,
        "done_reason": result.done_reason,
        "turns": stats.turns,
        "round_number": stats.turns,
        "tool_calls": stats.tool_calls,
        "images": stats.images,
    }


async def generate_with_evidence(input: Any) -> Any:
    """Run the maintained V1 generator with exact authority evidence attached."""
    from fti.miles.v1 import client_recording

    client_recording.TaskSession = _evidence_session_class()
    client_recording.episode_metadata = _evidence_episode_metadata
    return await client_recording.generate(input)


def _generate_add_arguments(parser: Any) -> None:
    from fti.miles.v1.client_recording import add_arguments

    add_arguments(parser)


generate_with_evidence.add_arguments = _generate_add_arguments


def _selected_samples(data: list[Any]) -> list[Any]:
    samples: list[Any] = []
    for group in data:
        if not isinstance(group, list):
            raise ValueError("selected reward group is malformed")
        for item in group:
            samples.extend(item if isinstance(item, list) else [item])
    return samples


def record_selected_reward_group(args: Any, data: list[Any]) -> None:
    """Seal the exact post-filter group that will feed the sole optimizer update."""
    plan = _load_runtime_plan()
    if (
        getattr(args, "rollout_batch_size", None) != 1
        or getattr(args, "n_samples_per_prompt", None) != 8
        or len(data) != 1
    ):
        raise ValueError("selected reward group differs from the one-update plan")
    refs: set[str] = set()
    for sample in _selected_samples(data):
        metadata = getattr(sample, "metadata", None) or {}
        fleet = metadata.get("fleet_v1") if isinstance(metadata, dict) else None
        ref = fleet.get("authority_evidence_sha256") if isinstance(fleet, dict) else None
        if isinstance(ref, str):
            _sha256(ref, "authority evidence")
            refs.add(ref)
    if len(refs) != plan["optimization"]["samples_per_prompt"]:
        raise ValueError("selected group lacks eight unique authoritative episodes")

    evidence_dir = _private_directory(Path(plan["identity"]["run_dir"]))
    episodes = []
    for ref in sorted(refs):
        path = evidence_dir / (ref.removeprefix("sha256:") + ".json")
        value = json.loads(path.read_text())
        body = {key: item for key, item in value.items() if key != "sha256"}
        if (
            value.get("schema") != PRIVATE_EPISODE_SCHEMA
            or value.get("sha256") != "sha256:" + digest(body)
            or value.get("sha256") != ref
            or value.get("plan_sha256") != "sha256:" + digest(plan)
            or value.get("task_key") != plan["task_binding"]["task_key"]
            or value.get("task_version_id") != plan["task_binding"]["task_version_id"]
            or value.get("verifier_version_id") != plan["task_binding"]["verifier_version_id"]
            or value.get("cleanup_confirmed") is not True
        ):
            raise ValueError("private episode authority evidence is invalid")
        episodes.append(value)
    execution_ids = {item["verifier_execution_id"] for item in episodes}
    instance_ids = {item["instance_id"] for item in episodes}
    rewards = [item["reward"] for item in episodes]
    if (
        len(execution_ids) != len(episodes)
        or len(instance_ids) != len(episodes)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in rewards
        )
        or max(rewards) == min(rewards)
    ):
        raise ValueError("selected verifier rewards lack unique authority or variation")

    private_body = {
        "schema": REWARD_GATE_SCHEMA + "_private",
        "plan_sha256": "sha256:" + digest(plan),
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "episode_receipts": sorted(refs),
        "verifier_execution_ids": sorted(execution_ids),
        "instance_ids": sorted(instance_ids),
        "rewards": rewards,
    }
    private = {**private_body, "sha256": "sha256:" + digest(private_body)}
    _write_once(
        evidence_dir / PRIVATE_SELECTED_FILE,
        json.dumps(private, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    public_body = {
        "schema": REWARD_GATE_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "selected_episode_count": len(episodes),
        "finite_rewards": True,
        "reward_variation": True,
        "unique_verifier_executions": len(execution_ids),
        "all_instances_released": True,
        "private_evidence_sha256": private["sha256"],
    }
    public = {**public_body, "sha256": "sha256:" + digest(public_body)}
    _write_once(
        Path(plan["identity"]["run_dir"]) / REWARD_GATE_FILE,
        json.dumps(public, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )


def _hf_index(root: Path) -> dict[str, str]:
    from safetensors import safe_open

    try:
        value = json.loads((root / "model.safetensors.index.json").read_text())
        weight_map = value["weight_map"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("HF safetensors index is absent or malformed") from exc
    if (
        not isinstance(weight_map, dict)
        or not weight_map
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or not filename.endswith(".safetensors")
            for key, filename in weight_map.items()
        )
    ):
        raise ValueError("HF safetensors weight map is invalid")
    indexed_by_shard: dict[str, set[str]] = {}
    for key, filename in weight_map.items():
        indexed_by_shard.setdefault(filename, set()).add(key)
    for filename, indexed_keys in indexed_by_shard.items():
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError("HF safetensors shard is absent or unsafe")
        with safe_open(path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
        if actual_keys != indexed_keys:
            raise ValueError("HF safetensors shard contents differ from its index")
    return weight_map


def _tensor_record(root: Path, filename: str, key: str) -> tuple[dict[str, Any], Any]:
    import torch
    from safetensors import safe_open

    with safe_open(root / filename, framework="pt", device="cpu") as handle:
        # ``safe_open`` exposes keys(), but is intentionally not iterable.
        if key not in handle.keys():  # noqa: SIM118
            raise ValueError("HF index points to a missing tensor")
        tensor = handle.get_tensor(key).contiguous()
    raw = tensor.view(torch.uint8).numpy()
    value = hashlib.sha256()
    value.update(memoryview(raw))
    record = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "tensor_sha256": "sha256:" + value.hexdigest(),
    }
    return record, tensor


def _copy_sidecars(base: Path, destination: Path, weight_files: set[str]) -> list[str]:
    copied = []
    excluded = weight_files | {"model.safetensors.index.json", ".complete", COMPLETE_MODEL_FILE}
    for source in sorted(base.iterdir(), key=lambda path: path.name):
        if source.name in excluded:
            continue
        target = destination / source.name
        if source.is_symlink():
            raise ValueError("base model sidecars may not be symlinks")
        if source.is_file():
            shutil.copy2(source, target)
        elif source.is_dir():
            shutil.copytree(source, target, symlinks=False)
        else:
            raise ValueError("base model contains an unsupported sidecar")
        copied.append(source.name)
    if "config.json" not in copied:
        raise ValueError("complete model lacks the exact base config")
    return copied


def _all_file_records(
    root: Path, *, exclude_complete_model_markers: bool = True
) -> list[dict[str, Any]]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("complete model contains a symlink")
        if not path.is_file() or (
            exclude_complete_model_markers and path.name in {COMPLETE_MODEL_FILE, ".complete"}
        ):
            continue
        relative = path.relative_to(root).as_posix()
        result.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": "sha256:" + file_sha256(path),
            }
        )
    return result


def _rename_noreplace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2 is required for create-once publication")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))
        return
    # Local Darwin tests cannot call renameat2.  The exclusive durable lock
    # preserves create-once behavior for cooperating publishers; cluster
    # execution always takes the stronger Linux kernel path above.
    lock = destination.with_name(destination.name + ".publish.lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("complete model destination already exists")
        os.rename(source, destination)
    finally:
        lock.unlink(missing_ok=True)


def compose_complete_model(plan: dict[str, Any], raw_hf: Path) -> dict[str, Any]:
    """Compose trained text tensors with exact frozen Qwen VLM tensors and sidecars."""
    import safetensors.torch
    import torch

    plan = validate_plan(plan)
    run_dir = Path(plan["identity"]["run_dir"])
    base = Path(plan["prepared_model"]["root"]) / "Qwen3.8-27B"
    final = run_dir / COMPLETE_HF_STEP
    partial = final.with_name(final.name + ".partial-" + uuid.uuid4().hex)
    _require_relative(partial, run_dir, "complete model partial path")
    if final.exists() or final.is_symlink():
        raise FileExistsError("complete model destination already exists")
    if Path(raw_hf) != run_dir / RAW_HF_STEP or not (raw_hf / ".complete").is_file():
        raise ValueError("native step-0 HF export is absent or misbound")
    base_map, trained_map = _hf_index(base), _hf_index(raw_hf)
    base_keys, trained_keys = set(base_map), set(trained_map)
    if not trained_keys < base_keys:
        raise ValueError("native HF export is not a strict subset of the exact base model")
    frozen_keys = base_keys - trained_keys
    if not frozen_keys or any(not key.startswith(FROZEN_TENSOR_PREFIXES) for key in frozen_keys):
        raise ValueError("native HF export complement is not limited to frozen VLM/MTP tensors")

    partial.mkdir(parents=True, mode=0o700)
    weight_map: dict[str, str] = {}
    tensor_manifest: dict[str, dict[str, Any]] = {}
    total_tensor_bytes = 0
    changed = 0
    raw_shards = sorted(set(trained_map.values()))
    renamed_shards = {
        filename: f"trained-{index:05d}.safetensors"
        for index, filename in enumerate(raw_shards, start=1)
    }
    for filename, renamed in renamed_shards.items():
        shutil.copy2(raw_hf / filename, partial / renamed)
    for key in sorted(trained_keys):
        trained_record, trained_tensor = _tensor_record(raw_hf, trained_map[key], key)
        base_record, base_tensor = _tensor_record(base, base_map[key], key)
        if (
            trained_record["shape"] != base_record["shape"]
            or trained_record["dtype"] != base_record["dtype"]
            or not bool(torch.isfinite(trained_tensor).all())
        ):
            raise ValueError("trained tensor shape, dtype, or finiteness is invalid")
        if trained_record["tensor_sha256"] != base_record["tensor_sha256"]:
            changed += 1
        weight_map[key] = renamed_shards[trained_map[key]]
        total_tensor_bytes += trained_tensor.numel() * trained_tensor.element_size()
        tensor_manifest[key] = {
            **trained_record,
            "source": "trained_step_0",
            "base_tensor_sha256": base_record["tensor_sha256"],
        }
        del trained_tensor, base_tensor
    if changed < 1:
        raise ValueError("optimizer update produced no changed trained tensor")

    frozen_by_shard: dict[str, list[str]] = {}
    for key in sorted(frozen_keys):
        frozen_by_shard.setdefault(base_map[key], []).append(key)
    for index, (source_shard, keys) in enumerate(sorted(frozen_by_shard.items()), start=1):
        destination_shard = f"frozen-{index:05d}.safetensors"
        tensors = {}
        for key in keys:
            record, tensor = _tensor_record(base, source_shard, key)
            tensors[key] = tensor
            total_tensor_bytes += tensor.numel() * tensor.element_size()
            weight_map[key] = destination_shard
            tensor_manifest[key] = {**record, "source": "exact_frozen_base"}
        safetensors.torch.save_file(tensors, partial / destination_shard)
        del tensors

    sidecars = _copy_sidecars(base, partial, set(base_map.values()))
    (partial / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_tensor_bytes}, "weight_map": weight_map},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    body = {
        "schema": COMPLETE_MODEL_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "base_binding_sha256": plan["prepared_model"]["binding_sha256"],
        "base_tensor_count": len(base_keys),
        "trained_tensor_count": len(trained_keys),
        "frozen_tensor_count": len(frozen_keys),
        "changed_trained_tensor_count": changed,
        "all_trained_tensors_finite": True,
        "total_tensor_bytes": total_tensor_bytes,
        "tensor_manifest": tensor_manifest,
        "sidecars": sidecars,
        "files": _all_file_records(partial),
    }
    manifest = {**body, "sha256": "sha256:" + digest(body)}
    _write_once(
        partial / COMPLETE_MODEL_FILE,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    _write_once(partial / ".complete", (manifest["sha256"] + "\n").encode())
    _rename_noreplace(partial, final)
    return manifest


def validate_complete_model(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    """Re-hash the complete model before a one-GPU reload is attempted."""
    import torch

    plan = validate_plan(plan)
    if Path(root) != Path(plan["identity"]["run_dir"]) / COMPLETE_HF_STEP:
        raise ValueError("complete model path is not plan-bound")
    try:
        value = json.loads((root / COMPLETE_MODEL_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("complete model manifest is absent") from exc
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (
        value.get("schema") != COMPLETE_MODEL_SCHEMA
        or value.get("plan_sha256") != "sha256:" + digest(plan)
        or value.get("native_rollout_id") != 0
        or value.get("optimizer_updates") != 1
        or value.get("all_trained_tensors_finite") is not True
        or type(value.get("changed_trained_tensor_count")) is not int
        or value["changed_trained_tensor_count"] < 1
        or value.get("sha256") != "sha256:" + digest(body)
        or (root / ".complete").read_text().strip() != value.get("sha256")
    ):
        raise ValueError("complete model manifest identity is invalid")
    expected_files = value.get("files")
    if not isinstance(expected_files, list) or expected_files != _all_file_records(root):
        raise ValueError("complete model file manifest changed")
    weight_map = _hf_index(root)
    tensors = value.get("tensor_manifest")
    if (
        not isinstance(tensors, dict)
        or set(tensors) != set(weight_map)
        or value.get("base_tensor_count") != len(tensors)
        or type(value.get("trained_tensor_count")) is not int
        or type(value.get("frozen_tensor_count")) is not int
        or value["trained_tensor_count"] + value["frozen_tensor_count"] != len(tensors)
        or value["changed_trained_tensor_count"] > value["trained_tensor_count"]
        or type(value.get("total_tensor_bytes")) is not int
        or value["total_tensor_bytes"] < 1
    ):
        raise ValueError("complete model tensor manifest changed")
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("complete model index metadata changed") from exc
    if index.get("metadata") != {"total_size": value["total_tensor_bytes"]}:
        raise ValueError("complete model index metadata changed")
    trained_keys = {key for key, item in tensors.items() if item.get("source") == "trained_step_0"}
    frozen_keys = {
        key for key, item in tensors.items() if item.get("source") == "exact_frozen_base"
    }
    if (
        trained_keys | frozen_keys != set(tensors)
        or trained_keys & frozen_keys
        or len(trained_keys) != value["trained_tensor_count"]
        or len(frozen_keys) != value["frozen_tensor_count"]
        or any(not key.startswith(FROZEN_TENSOR_PREFIXES) for key in frozen_keys)
    ):
        raise ValueError("complete model tensor provenance changed")
    observed_tensor_bytes = 0
    for key in sorted(weight_map):
        observed, tensor = _tensor_record(root, weight_map[key], key)
        observed_tensor_bytes += tensor.numel() * tensor.element_size()
        expected = tensors[key]
        if any(observed[field] != expected.get(field) for field in observed) or not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError("complete model tensor bytes changed")
        del tensor
    if observed_tensor_bytes != value["total_tensor_bytes"]:
        raise ValueError("complete model tensor byte count changed")
    return value


def _checkpoint_manifest(plan: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    run_dir = Path(plan["identity"]["run_dir"])
    _require_relative(checkpoint, run_dir / "model-output" / "checkpoints", "checkpoint path")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError("native step-0 checkpoint is absent")
    files = _all_file_records(checkpoint, exclude_complete_model_markers=False)
    if not files:
        raise ValueError("native step-0 checkpoint is empty")
    body = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "checkpoint_path_sha256": "sha256:" + hashlib.sha256(str(checkpoint).encode()).hexdigest(),
        "files": files,
    }
    manifest = {**body, "sha256": "sha256:" + digest(body)}
    _write_once(
        _private_directory(run_dir) / PRIVATE_CHECKPOINT_FILE,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n",
    )
    return manifest


def _train_receipt(
    plan: dict[str, Any], *, rollout_id: int, checkpoint_dir: str, hf_dir: str
) -> dict[str, Any]:
    run_dir = Path(plan["identity"]["run_dir"])
    checkpoint = Path(checkpoint_dir)
    expected_hf = run_dir / RAW_HF_STEP
    if rollout_id != 0 or Path(hf_dir) != expected_hf:
        raise ValueError("post-save hook received an unexpected checkpoint identity")
    if not checkpoint.exists() or not (expected_hf / ".complete").is_file():
        raise ValueError("post-save hook lacks a completed checkpoint/export")
    reward_gate = json.loads((run_dir / REWARD_GATE_FILE).read_text())
    reward_body = {key: item for key, item in reward_gate.items() if key != "sha256"}
    private_reward = json.loads((_private_directory(run_dir) / PRIVATE_SELECTED_FILE).read_text())
    private_reward_body = {key: item for key, item in private_reward.items() if key != "sha256"}
    private_refs = private_reward.get("episode_receipts")
    private_executions = private_reward.get("verifier_execution_ids")
    private_instances = private_reward.get("instance_ids")
    private_rewards = private_reward.get("rewards")
    private_lists_valid = all(
        isinstance(items, list) and len(items) == 8
        for items in (private_refs, private_executions, private_instances, private_rewards)
    )
    if private_lists_valid:
        try:
            private_lists_valid = (
                len({_sha256(item, "episode authority evidence") for item in private_refs}) == 8
                and len({_uuid(item, "verifier execution identity") for item in private_executions})
                == 8
                and len({_uuid(item, "task instance identity") for item in private_instances}) == 8
                and all(
                    not isinstance(item, bool)
                    and isinstance(item, (int, float))
                    and math.isfinite(float(item))
                    for item in private_rewards
                )
                and max(private_rewards) != min(private_rewards)
            )
        except (TypeError, ValueError):
            private_lists_valid = False
    if (
        reward_gate.get("schema") != REWARD_GATE_SCHEMA
        or reward_gate.get("plan_sha256") != "sha256:" + digest(plan)
        or reward_gate.get("native_rollout_id") != 0
        or reward_gate.get("optimizer_updates") != 1
        or reward_gate.get("selected_episode_count") != 8
        or reward_gate.get("finite_rewards") is not True
        or reward_gate.get("reward_variation") is not True
        or reward_gate.get("unique_verifier_executions") != 8
        or reward_gate.get("all_instances_released") is not True
        or reward_gate.get("sha256") != "sha256:" + digest(reward_body)
        or private_reward.get("schema") != REWARD_GATE_SCHEMA + "_private"
        or private_reward.get("plan_sha256") != "sha256:" + digest(plan)
        or private_reward.get("native_rollout_id") != 0
        or private_reward.get("optimizer_updates") != 1
        or private_reward.get("sha256") != "sha256:" + digest(private_reward_body)
        or reward_gate.get("private_evidence_sha256") != private_reward.get("sha256")
        or not private_lists_valid
    ):
        raise ValueError("selected reward gate is absent or invalid")
    checkpoint_manifest = _checkpoint_manifest(plan, checkpoint)
    complete = compose_complete_model(plan, expected_hf)
    body = {
        "schema": TRAIN_RECEIPT_SCHEMA,
        "plan_sha256": "sha256:" + digest(plan),
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "reward_filter": REWARD_FILTER,
        "reward_gate_sha256": reward_gate["sha256"],
        "checkpoint_manifest_sha256": checkpoint_manifest["sha256"],
        "complete_model_manifest_sha256": complete["sha256"],
        "complete_model_path_sha256": "sha256:"
        + hashlib.sha256(str(run_dir / COMPLETE_HF_STEP).encode()).hexdigest(),
        "changed_trained_tensor_count": complete["changed_trained_tensor_count"],
        "all_trained_tensors_finite": complete["all_trained_tensors_finite"],
        "wandb_run_id": plan["wandb"]["run_id"],
        "wandb_resume": "never",
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
        "native_rollout_id": 0,
        "optimizer_updates": 1,
        "reward_filter": REWARD_FILTER,
        "wandb_run_id": plan["wandb"]["run_id"],
        "wandb_resume": "never",
        "hf_export_complete": True,
        "save_hook_after_checkpoint": True,
    }
    expected_fields = set(expected) | {
        "reward_gate_sha256",
        "checkpoint_manifest_sha256",
        "complete_model_manifest_sha256",
        "complete_model_path_sha256",
        "changed_trained_tensor_count",
        "all_trained_tensors_finite",
    }
    if (
        set(body) != expected_fields
        or any(body[key] != item for key, item in expected.items())
        or body.get("all_trained_tensors_finite") is not True
        or receipt.get("receipt_sha256") != "sha256:" + digest(body)
    ):
        raise ValueError("invalid update/export receipt")
    for field in (
        "reward_gate_sha256",
        "checkpoint_manifest_sha256",
        "complete_model_manifest_sha256",
        "complete_model_path_sha256",
    ):
        _sha256(body[field], field)
    if (
        type(body["changed_trained_tensor_count"]) is not int
        or body["changed_trained_tensor_count"] < 1
    ):
        raise ValueError("invalid update/export receipt")
    return receipt


def reload_request(plan: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Render the separate one-GPU, zero-optimizer reload observer."""
    from cyber_post_train.jobs import bundled_request

    plan = validate_plan(plan)
    receipt = validate_train_receipt(plan, receipt)
    identity = plan["identity"]
    request = {
        "name": identity["reload_name"],
        "title": f"Qwen3.8 96K Miles checkpoint reload observer: {identity['reload_name']}",
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
            "CYBER_EVIDENCE_RUN_DIR": identity["reload_run_dir"],
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


async def _reload_episode(plan: dict[str, Any], model_path: Path) -> dict[str, Any]:
    from miles.rollout.base_types import GenerateFnInput
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
    from miles.utils.chat_template_utils import resolve_fixed_chat_template
    from miles.utils.http_utils import init_http_client
    from miles.utils.types import Sample

    template_path, _ = resolve_fixed_chat_template("qwen35")
    args = argparse.Namespace(
        hf_checkpoint=str(model_path),
        chat_template_path=template_path,
        custom_generate_function_path="training.miles96_mechanics_canary.generate_with_evidence",
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
        generate_with_evidence(generated),
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
        or fleet.get("authority_verified") is not True
        or fleet.get("cleanup_confirmed") is not True
        or not isinstance(fleet.get("authority_evidence_sha256"), str)
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
        "authority_evidence_sha256": _sha256(
            fleet["authority_evidence_sha256"], "reload authority evidence"
        ),
    }


def _reload_server_arguments(model_path: Path) -> list[str]:
    """Use the pinned recipe's one-engine attention/context settings."""
    return [
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
        "0.8",
        "--context-length",
        str(CONTEXT_TOKENS),
        "--tp-size",
        "1",
        "--attention-backend",
        "triton",
    ]


def run_reload(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
    plan = validate_plan(plan)
    validate_train_receipt(plan, receipt)
    identity = plan["identity"]
    run_dir = Path(identity["reload_run_dir"])
    model_path = Path(identity["run_dir"]) / COMPLETE_HF_STEP
    if os.environ.get("RUN_DIR") != str(run_dir) or os.environ.get("CYBER_PLAN_SHA256") != digest(
        plan
    ):
        raise ValueError("reload runtime identity drift")
    if not run_dir.is_dir() or (run_dir / RELOAD_FILE).exists():
        raise FileExistsError("reload output is not create-once")
    complete = validate_complete_model(plan, model_path)
    if complete["sha256"] != receipt["complete_model_manifest_sha256"]:
        raise ValueError("reload model differs from the update receipt")
    _reload_runtime_binding()
    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError("reload observer must have exactly one GPU")
    private_log = _private_directory(run_dir) / "reload-sglang.log"
    descriptor = os.open(private_log, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as log:
        process = subprocess.Popen(
            _reload_server_arguments(model_path),
            stdout=log,
            stderr=subprocess.STDOUT,
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
        "native_rollout_id": 0,
        "source_optimizer_updates": 1,
        "complete_model_manifest_sha256": complete["sha256"],
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
