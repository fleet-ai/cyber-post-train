"""Thin, immutable Miles RL launch. Native Miles owns the complete training loop.

A returned native loop plus checkpoint inventory is not scientific acceptance:
reward, optimizer updates and checkpoint reload need independent run evidence.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

from cyber_post_train.jobs import bundled_request, digest, quantity

from . import miles
from .miles_conversion import _hash, check_inputs, inventory
from .rl_runtime import sealed as _sealed

SCHEMA = "cyber_miles_training_v1"
MODULE = "training.miles_training"
# Colocated generation, Megatron and CPU offload have a different RAM peak
# from SFT. The real Qwen canary exhausted its inherited 768-GiB limit.
RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "128",
    "memory_request": "1536Gi",
    "memory_limit": "2048Gi",
}
NATIVE_DRIVER_SHA256 = "85dbfd31d41a84f9c2e79a2918583851fb53925630afa229e9cd0a154b170f46"
RUNTIME_FILES = (
    "training/miles_training.py",
    "training/miles.py",
    "training/miles_conversion.py",
    "training/miles_rollout.py",
    "training/miles_text.py",
    "training/rl_episode.py",
    "training/rl_runtime.py",
    "training/sft_runtime.py",
    "evals/fleet/opencode_self_hosted.py",
    "cyber_post_train/jobs.py",
)


def _runtime():
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def compile_rl(config: dict, *, relative_to: Path) -> dict:
    from .models import bound_model
    from .sft import _known, _sfs_root, read_mapping

    _known(
        config,
        {
            "backend",
            "name",
            "output_root",
            "model",
            "data",
            "checkpoint",
            "recipe",
            "wandb",
            "cluster",
        },
        "RL",
    )
    if config["backend"] != "miles":
        raise ValueError("this RL path uses Miles; no silent backend substitution")
    model, data, checkpoint = (config[k] for k in ("model", "data", "checkpoint"))
    _known(model, {"lock", "weights", "root"}, "model")
    _known(data, {"manifest", "root"}, "data")
    _known(checkpoint, {"manifest", "sha256"}, "checkpoint")
    w, cluster = config["wandb"], config.get("cluster", {})
    _known(w, {"entity", "project", "run_id"}, "W&B")
    _known(cluster, {"priority", "resources"}, "cluster")
    recipe = config.get("recipe", {})
    _known(
        recipe,
        {
            "nodes",
            "steps",
            "groups",
            "samples_per_prompt",
            "lr",
            "eval_interval",
            "checkpoint_interval",
            "seed",
        },
        "Miles recipe",
    )
    bound = bound_model(
        read_mapping(relative_to / model["lock"]),
        read_mapping(relative_to / model["weights"]),
        _sfs_root(model["root"], "model root"),
    )
    metadata = read_mapping(relative_to / data["manifest"])
    _sealed(metadata, "cyber_miles_data_v1")
    cp_path = relative_to / checkpoint["manifest"]
    if _hash(cp_path) != checkpoint["sha256"].removeprefix("sha256:"):
        raise ValueError("native checkpoint manifest file digest mismatch")
    cp = read_mapping(cp_path)
    _sealed(cp, "cyber_miles_checkpoint_v1")
    if cp["model"] != bound or cp["image"] != miles.IMAGE or cp["optimizer_steps"] != 0:
        raise ValueError("RL-from-base requires the exact native base conversion")
    root = Path(_sfs_root(data["root"], "data root"))
    limits = metadata["limits"]
    args = miles.MilesConfig(
        name=config["name"],
        output_root=_sfs_root(config["output_root"], "output root"),
        model=bound["repo"],
        model_root=bound["root"],
        torch_dist_root=cp["root"],
        train_data=str(root / metadata["files"]["train"]["path"]),
        dev_data=str(root / metadata["files"]["dev"]["path"]),
        data_manifest=str(root / "manifest.json"),
        wandb_entity=w["entity"],
        wandb_project=w["project"],
        wandb_run_id=w["run_id"],
        context_tokens=limits["context_tokens"],
        response_tokens=limits["response_tokens"],
        tokens_per_turn=limits["max_tokens_per_turn"],
        **recipe,
    )
    args.validate()
    if (
        metadata["name"] != args.name
        or metadata["tokenizer"]["repo"] != bound["repo"]
        or metadata["tokenizer"]["revision"] != bound["revision"]
        or metadata["template_sha256"] != "sha256:" + miles.TEMPLATE_SHA256
        or not 0 < args.groups <= metadata["files"]["train"]["rows"]
        or metadata["files"]["dev"]["rows"] < 1
    ):
        raise ValueError("run/model/data/batch identity mismatch")
    for item in metadata["files"].values():
        if Path(item["path"]).name != item["path"]:
            raise ValueError("data file must be directly inside its immutable root")
    layout = miles.topology(args)
    # Dev evaluation runs the whole frozen set at one sample per prompt, so the
    # larger of the two phases sets the run's simultaneous environment demand.
    layout["concurrent_environments"] = max(
        layout["concurrent_train_environments"], metadata["files"]["dev"]["rows"]
    )
    plan = {
        "schema": SCHEMA,
        "run_name": args.name,
        "output_root": args.output_root,
        "model": bound,
        "data": metadata,
        "checkpoint": cp,
        "arguments": dataclasses.asdict(args),
        "topology": layout,
        "runtime_sha256": digest(_runtime()),
        "native_driver_sha256": NATIVE_DRIVER_SHA256,
        "execution": {
            "image": miles.IMAGE,
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
        },
    }
    job_request(plan)
    return plan


def job_request(plan):
    args = miles.MilesConfig(**plan["arguments"])
    args.validate()
    layout, derived = plan.get("topology", {}), miles.topology(args)
    if (
        plan["schema"] != SCHEMA
        or plan["runtime_sha256"] != digest(_runtime())
        or plan["native_driver_sha256"] != NATIVE_DRIVER_SHA256
        or plan["execution"]["image"] != miles.IMAGE
        or plan["run_name"] != args.name
        or plan["output_root"] != args.output_root
        or any(layout.get(key) != value for key, value in derived.items())
        or layout.get("concurrent_environments", 0) < derived["concurrent_train_environments"]
    ):
        raise ValueError("Miles plan/runtime drift")
    resources = plan["execution"]["resources"]
    if any(
        quantity(resources[key]) < quantity(RESOURCES[key])
        for key in ("cpu_request", "memory_request", "memory_limit")
    ):
        raise ValueError("native Qwen RL requires its reviewed colocated RAM reservation and limit")
    files = _runtime()
    files.update(
        {p + "/__init__.py": "" for p in ("training", "evals", "evals/fleet", "cyber_post_train")}
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": args.name,
            "title": args.name + " native Miles RL",
            "run_dir": args.output_root,
            "image": miles.IMAGE,
            "workers": layout["nodes"],
            "gpus_per_worker": layout["gpus_per_node"],
            "resources": resources,
            "priority_class": plan["execution"]["priority"],
            "requeueIfPreempted": False,
            "secrets": ["fleet-api", "wandb-api"],
            "env": {
                # The editable Megatron install omits post_training. Native
                # conversion and training both need the complete source tree.
                "PYTHONPATH": "/root/Megatron-LM",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "MILES_USE_LEGACY_ROLLOUT_V1": "0",
                "WANDB_MODE": "online",
                "WANDB_RUN_ID": args.wandb_run_id,
                "WANDB_DISABLE_CODE": "true",
                "WANDB_CONSOLE": "off",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        "training.miles_training",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def check_artifacts(plan):
    """Read private payloads mechanically; return no task text or model tensors."""
    from .rl_episode import _validate

    check_inputs(plan)
    cp = plan["checkpoint"]
    _sealed(cp, "cyber_miles_checkpoint_v1")
    root = Path(cp["root"])
    if inventory(root) != [{k: f[k] for k in ("path", "size")} for f in cp["files"]]:
        raise ValueError("native checkpoint inventory changed")
    for item in cp["files"]:
        path = root / item["path"]
        if path.stat().st_size != item["size"] or _hash(path) != item["sha256"]:
            raise ValueError("native checkpoint payload changed")
    args = plan["arguments"]
    path = Path(args["data_manifest"])
    data = json.loads(path.read_text())
    if data != plan["data"]:
        raise ValueError("staged data manifest changed")
    selected = {"train": [], "dev": []}
    for split, item in data["files"].items():
        p = path.parent / item["path"]
        if _hash(p) != item["sha256"].removeprefix("sha256:"):
            raise ValueError("staged RL prompts changed")
        rows = [json.loads(line) for line in p.read_bytes().splitlines()]
        if len(rows) != item["rows"] or not rows:
            raise ValueError("RL task count changed")
        for row in rows:
            meta = row["metadata"]
            cfg = meta["cyber_config"]
            _validate(cfg)
            if (
                meta["split"] != split
                or cfg["run_id"] != args["name"]
                or cfg["model"]["repo"] != plan["model"]["repo"]
                or cfg["model"]["revision"] != plan["model"]["revision"]
                or cfg["model"]["root"] != args["model_root"]
                or cfg["rl"] != {k: v for k, v in data["limits"].items() if k != "response_tokens"}
                or cfg["execution"]["required_task_tool_catalog_sha256"]
                != data["tool_catalog_sha256"]
                or cfg["initial_prompt_sha256"]
                != "sha256:" + hashlib.sha256(row["input"].encode()).hexdigest()
            ):
                raise ValueError("episode identity changed")
            selected[split].append(row)
    return selected


def native_source():
    from miles.utils.external_utils.command_utils import repo_base_dir

    path = Path(repo_base_dir) / "train.py"
    if _hash(path) != NATIVE_DRIVER_SHA256:
        raise ValueError("native Miles training driver changed")
    return path


def native_args(plan):
    os.environ["MILES_USE_LEGACY_ROLLOUT_V1"] = "0"
    from miles.utils.arguments import parse_args

    previous = sys.argv
    try:
        sys.argv = [str(native_source()), *miles.arguments(miles.MilesConfig(**plan["arguments"]))]
        args = parse_args()
        expected = {
            "data_source_path": "training.miles_text.TextDataSource",
            "tool_key": "tools",
            "start_rollout_id": 0,
            "load": plan["checkpoint"]["root"],
            "ref_load": plan["checkpoint"]["root"],
            "num_rollout": plan["arguments"]["steps"],
            "num_steps_per_rollout": 1,
            "actor_num_nodes": plan["topology"]["nodes"],
            "actor_num_gpus_per_node": plan["topology"]["gpus_per_node"],
            "global_batch_size": plan["arguments"]["groups"]
            * plan["arguments"]["samples_per_prompt"],
        }
        if any(getattr(args, k) != v for k, v in expected.items()):
            raise ValueError("native parsed training/load semantics changed")
        return args
    finally:
        sys.argv = previous


def check_native_namespace(pythonpath):
    """Resolve the native namespace in a fresh child without importing CUDA."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util;"
            "assert importlib.util.find_spec('megatron.post_training.checkpointing');"
            "assert importlib.util.find_spec('model_provider')",
        ],
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        timeout=30,
    )
    if result.returncode:
        raise ValueError("native Megatron source namespace unavailable")


def preflight(plan):
    import torch
    from miles.utils.data import Dataset

    from .miles_text import TextDataSource
    from .rl_data import selection

    if torch.cuda.is_available():
        raise ValueError("Miles preflight is CPU-only")
    request = job_request(plan)
    check_native_namespace(request["env"]["PYTHONPATH"])
    if Path(plan["output_root"]).exists():
        raise FileExistsError("RL output already exists")
    rows = check_artifacts(plan)
    data_root = Path(plan["arguments"]["data_manifest"]).parent
    split = json.loads((data_root / "split.json").read_text())
    task_set = json.loads((data_root / "task-set.json").read_text())
    if (
        split["sha256"] != plan["data"]["split_sha256"]
        or task_set["sha256"] != plan["data"]["selection_sha256"]
    ):
        raise ValueError("reviewed task-set/split changed")
    expected = {
        (r["split"], r["task_key"], r["task_version_id"]): r["lineage"]
        for r in selection(task_set, split)
    }
    actual = {}
    for group, values in rows.items():
        for row in values:
            meta = row["metadata"]
            task = meta["cyber_config"]["task"]
            key = (group, task["key"], task["version_id"])
            if key in actual:
                raise ValueError("duplicate RL task version")
            actual[key] = meta["lineage"]
    if actual != expected:
        raise ValueError("selected RL task families differ from frozen split")
    # Megatron's parser imports Transformer Engine/libcuda even before training.
    # The CPU gate checks the real FTI builder; the unchanged native parser runs
    # first in the bounded GPU child. Do not mock CUDA to claim CPU qualification.
    config = miles.MilesConfig(**plan["arguments"])
    argv = miles.arguments(config)
    native_source()
    source = TextDataSource(
        SimpleNamespace(
            rollout_global_dataset=True,
            apply_chat_template=False,
            multimodal_keys=None,
            tool_key="tools",
            label_key=None,
            input_key="input",
            metadata_key="metadata",
            hf_checkpoint=config.model_root,
            chat_template_path=argv[argv.index("--chat-template-path") + 1],
            prompt_data=config.train_data,
            rollout_max_prompt_len=config.context_tokens - config.response_tokens,
            rollout_seed=config.seed,
            rollout_shuffle=True,
        )
    )
    if hashlib.sha256(source.tokenizer.chat_template.encode()).hexdigest() != miles.TEMPLATE_SHA256:
        raise ValueError("loaded Miles chat template changed")
    for split in ("train", "dev"):
        path = plan["arguments"][split + "_data"]
        dataset = (
            source.dataset
            if split == "train"
            else Dataset(
                path,
                source.tokenizer,
                None,
                config.context_tokens - config.response_tokens,
                prompt_key="input",
                metadata_key="metadata",
                apply_chat_template=False,
            )
        )
        if len(dataset) != len(rows[split]) or any(
            sample.prompt != row["input"] or sample.metadata != row["metadata"]
            for sample, row in zip(dataset.origin_samples, rows[split], strict=True)
        ):
            raise ValueError("native Miles silently filtered or changed a task")
    return {
        "schema": "cyber_miles_training_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "planned_steps": config.steps,
        "planned_global_batch": config.groups * config.samples_per_prompt,
        "native_arguments_sha256": digest(argv),
        "native_parser_checked": False,
        "native_megatron_namespace_checked": True,
        "native_text_source_checked": True,
        "counts": {k: len(v) for k, v in rows.items()},
        "rl_qualified": False,
    }


def _native(plan):
    import ray
    from miles.utils.tracking_utils.tracking import finish_tracking

    # Keep slow payload validation in the child so the parent bounds startup,
    # including stalled storage, before Ray or model loading begins.
    check_artifacts(plan)
    args = native_args(plan)
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
            }
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


def native_result(plan):
    root = Path(plan["output_root"])
    count = plan["arguments"]["steps"]
    expected = [
        "dev-baseline-r0",
        *(f"train-r{i}" for i in range(count)),
        *(
            f"dev-after-r{i}"
            for i in range(count)
            if (i + 1) % plan["arguments"]["eval_interval"] == 0
        ),
    ]
    for name in expected:
        path = root / "episodes/batches" / name
        value = json.loads((path / "COLLECTED.json").read_text())
        _sealed(value, "cyber_miles_batch_v1")
        if (
            (path / "FAILED.json").exists()
            or value["batch_id"] != name
            or value["data_sha256"] != plan["data"]["sha256"]
        ):
            raise ValueError("native batch completion evidence mismatch")
    pointer = (root / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip()
    # Native Miles passes zero-based rollout_id to Megatron save_checkpoint.
    if pointer != str(count - 1):
        raise ValueError("native final checkpoint rollout index mismatch")
    return {
        "status": "native_loop_returned",
        "plan_sha256": digest(plan),
        "checkpoint_rollout_index": int(pointer),
        "completed_batches": len(expected),
        "completed_at": time.time(),
        "optimizer_update_independently_verified": False,
        "checkpoint_reload_verified": False,
    }


def run(plan, plan_path):
    from .rl_runtime import run as supervised_run

    return supervised_run(plan, plan_path, backend=sys.modules[__name__])


def main():
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
            print(json.dumps({k: result[k] for k in ("status", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
