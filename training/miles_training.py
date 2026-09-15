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
import re
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
LONG_CONTEXT_SCHEMA = "cyber_miles_training_v2"
MODULE = "training.miles_training"
# Colocated generation, Megatron and CPU offload have a different RAM peak
# from SFT. The real Qwen canary exhausted its inherited 768-GiB limit.
RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "128",
    "memory_request": "1536Gi",
    "memory_limit": "2048Gi",
}
MINIMUM_CPU_REQUEST = "32"
NATIVE_DRIVER_SHA256 = "85dbfd31d41a84f9c2e79a2918583851fb53925630afa229e9cd0a154b170f46"
RUNTIME_FILES = (
    "training/miles_training.py",
    "training/miles.py",
    "training/miles_conversion.py",
    "training/miles_opencode.py",
    "training/miles_promotion.py",
    "training/miles_rollout.py",
    "training/miles_text.py",
    "training/rl_episode.py",
    "training/rl_runtime.py",
    "training/sft_runtime.py",
    "evals/fleet/opencode_self_hosted.py",
    "cyber_post_train/jobs.py",
)
LONG_RUNTIME_BASE_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "b713f93d8da719a08aa20f1c45752d32a40d72b95159f410cdb6046d5ed1cf5d"
)
LONG_RUNTIME_CHECKS = frozenset(
    {
        "installed_source_digest_checked",
        "opencode_binary_checked",
        "native_256k_parser_checked",
        "qwen38_profile_template_checked",
        "native_qwen38_tito_backport_checked",
        "native_radix_affinity_route_checked",
        "forced_repeated_compaction_checked",
        "more_than_1024_nodes_checked",
        "one_reward_per_rollout_checked",
        "summary_tokens_excluded_checked",
        "no_primary_tool_prefix_truncation_checked",
    }
)


def _runtime():
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def _validate_long_runtime_receipt(
    receipt: object, *, image: str, build_source_sha256: str | None = None
) -> dict:
    """Validate an image qualification, never a scientific RL acceptance."""
    if not isinstance(receipt, dict):
        raise ValueError("long-context runtime image qualification is absent")
    _sealed(receipt, "cyber_miles_opencode_runtime_qualification_v1")
    if (
        receipt.get("status") != "image_qualified_for_dev"
        or receipt.get("image") != image
        or receipt.get("base_image") != LONG_RUNTIME_BASE_IMAGE
        or receipt.get("fti_version") != "0.8.4"
        or receipt.get("native_profile") != "qwen3.8-27b-256k"
        or receipt.get("model_config_sha256")
        != "sha256:191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
        or receipt.get("parser_checkpoint_root")
        != "/mnt/sfs/jobs/chris-cpt-cleanup-q38-miles-base-v1/torch-dist"
        or receipt.get("parser_checkpoint_receipt_file_sha256")
        != "sha256:b3d772de9121f442ea7b9a4c9a996f2a0a99cab8c49fe3083c148fe3eebd089c"
        or receipt.get("parser_checkpoint_receipt_sha256")
        != "19c8e93482530170e0f648815ab74233719e6f2b3bb7879a6564b42c3abec371"
        or receipt.get("opencode_version") != "1.18.27"
        or receipt.get("opencode_source_commit") != "4b7e19e315cca414121ba1d61523fef74bb3ae8b"
        or receipt.get("opencode_binary_sha256")
        != "sha256:bddf894e5c2bc3d8cf452bd6e5ab2273bbe4a37eeeb9aec848d3d7d20db1f256"
        or receipt.get("miles_source_commit") != "9e178ca16839b0600155f3927f57ce0670b8f453"
        or receipt.get("miles_tito_backport_commit")
        != "257992eb52bfa1f5248b5a5ae8f5a959be500788"
        or receipt.get("installed_tito_source_sha256")
        != "sha256:72650e3b337d69d237088c03cafa12b066a2c31fe1ffd96fab2d49d832f4a33c"
        or receipt.get("miles_tree_source_sha256")
        != "sha256:fd978a1ef2617f4bf30850fedd197e546cdc9c6542b00b03df502cbb285fc732"
        or receipt.get("native_driver_sha256") != "sha256:" + miles.LONG_NATIVE_DRIVER_SHA256
        or receipt.get("native_converter_sha256") != "sha256:" + miles.LONG_NATIVE_CONVERTER_SHA256
        or receipt.get("installed_session_tree_sha256")
        != "sha256:" + miles.LONG_INSTALLED_SESSION_TREE_SHA256
        or set(receipt.get("checks", {})) != LONG_RUNTIME_CHECKS
        or not all(receipt["checks"].values())
        or not re.fullmatch(r"[^@\s]+@sha256:[a-f0-9]{64}", image)
        or not re.fullmatch(r"sha256:[a-f0-9]{64}", str(receipt.get("build_source_sha256", "")))
        or (
            build_source_sha256 is not None
            and receipt.get("build_source_sha256") != build_source_sha256
        )
    ):
        raise ValueError("long-context runtime image has not passed exact qualification")
    return receipt


def _bind_long_runtime(value: object, relative_to: Path) -> dict:
    """Bind a built and parser-qualified image for the first dev canary."""
    from .sft import _known, read_mapping

    if not isinstance(value, dict):
        raise ValueError("long-context runtime image qualification is absent")
    _known(value, {"image", "receipt", "sha256"}, "long-context runtime")
    if not isinstance(value.get("sha256"), str) or not re.fullmatch(
        r"sha256:[a-f0-9]{64}", value["sha256"]
    ):
        raise ValueError("long-context runtime receipt digest is malformed")
    path = relative_to / value["receipt"]
    if _hash(path) != value["sha256"].removeprefix("sha256:"):
        raise ValueError("long-context runtime receipt file changed")
    receipt = read_mapping(path)
    root = Path(__file__).resolve().parents[1]
    build_files = {
        str(path.relative_to(root)): path.read_text()
        for path in sorted((root / "training/images/miles-opencode-long-context").glob("*"))
        if path.is_file()
    }
    build_source_sha256 = "sha256:" + digest(build_files)
    return _validate_long_runtime_receipt(
        receipt, image=value["image"], build_source_sha256=build_source_sha256
    )


def compile_rl(config: dict, *, relative_to: Path) -> dict:
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
            "production_promotion",
            "runtime",
        },
        "RL",
    )
    if config["backend"] != "miles":
        raise ValueError("this RL path uses Miles; no silent backend substitution")
    model, data, checkpoint = (config[k] for k in ("model", "data", "checkpoint"))
    _known(
        model,
        {
            "lock",
            "weights",
            "root",
            "export",
            "gpu_check",
            "sft_source",
            "runtime_stage",
        },
        "model",
    )
    _known(data, {"manifest", "root"}, "data")
    _known(checkpoint, {"manifest", "sha256"}, "checkpoint")
    w, cluster = config["wandb"], config.get("cluster", {})
    _known(w, {"entity", "project", "run_id"}, "W&B")
    _known(cluster, {"priority", "resources", "target"}, "cluster")
    target = cluster.get("target")
    if target not in {None, "dev", "prod"}:
        raise ValueError("Miles cluster target must be dev or prod")
    from .miles_promotion import bind_production_promotion

    production_promotion = bind_production_promotion(config, relative_to)
    recipe = config.get("recipe", {})
    _known(
        recipe,
        {
            "nodes",
            "gpus_per_node",
            "steps",
            "groups",
            "samples_per_prompt",
            "lr",
            "temperature",
            "kl_loss_coef",
            "max_tokens_per_gpu",
            "eval_interval",
            "checkpoint_interval",
            "seed",
            "native_profile",
            "harness",
            "session_node_cap",
        },
        "Miles recipe",
    )
    from .miles_conversion import bind_model_source

    bound = bind_model_source(model, relative_to=relative_to)
    policy_identity_root = bound["root"]
    if initial_policy := bound.get("initial_policy"):
        policy_identity_root = initial_policy.get("accepted_root")
        if not isinstance(policy_identity_root, str):
            raise ValueError("accepted SFT policy identity is absent")
    metadata = read_mapping(relative_to / data["manifest"])
    long_horizon = metadata.get("schema") == "cyber_miles_data_v2"
    _sealed(metadata, "cyber_miles_data_v2" if long_horizon else "cyber_miles_data_v1")
    runtime = _bind_long_runtime(config.get("runtime"), relative_to) if long_horizon else None
    if not long_horizon and config.get("runtime") is not None:
        raise ValueError("legacy Miles does not accept an alternate runtime")
    cp_path = relative_to / checkpoint["manifest"]
    if _hash(cp_path) != checkpoint["sha256"].removeprefix("sha256:"):
        raise ValueError("native checkpoint manifest file digest mismatch")
    cp = read_mapping(cp_path)
    _sealed(cp, "cyber_miles_checkpoint_v1")
    runtime_image = runtime["image"] if runtime is not None else miles.IMAGE
    if cp["model"] != bound or cp["image"] != runtime_image or cp["optimizer_steps"] != 0:
        raise ValueError("RL requires an exact zero-step conversion of its initial policy")
    root = Path(_sfs_root(data["root"], "data root"))
    limits = metadata["limits"]
    args = miles.MilesConfig(
        name=config["name"],
        output_root=_sfs_root(config["output_root"], "output root"),
        model=bound["repo"],
        model_root=bound["root"],
        policy_identity_root=policy_identity_root,
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
        runtime_image=runtime_image,
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
    plan = {
        "schema": LONG_CONTEXT_SCHEMA if long_horizon else SCHEMA,
        "run_name": args.name,
        "output_root": args.output_root,
        "model": bound,
        "data": metadata,
        "checkpoint": cp,
        "arguments": dataclasses.asdict(args),
        "runtime_sha256": digest(_runtime()),
        "native_driver_sha256": (
            miles.LONG_NATIVE_DRIVER_SHA256 if long_horizon else NATIVE_DRIVER_SHA256
        ),
        "execution": {
            "image": runtime_image,
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
            **({"cluster_target": target} if target is not None else {}),
            **(
                {"production_promotion": production_promotion}
                if production_promotion is not None
                else {}
            ),
        },
        **({"runtime_qualification": runtime} if runtime is not None else {}),
    }
    job_request(plan)
    return plan


def job_request(plan):
    from .miles_promotion import validate_embedded_promotion

    validate_embedded_promotion(plan, check_files=True)
    args = miles.MilesConfig(**plan["arguments"])
    args.validate()
    if args.harness == "opencode":
        receipt = plan.get("runtime_qualification")
        _validate_long_runtime_receipt(receipt, image=args.runtime_image)
    elif "runtime_qualification" in plan:
        raise ValueError("legacy plan carries an alternate runtime qualification")
    if (
        plan["schema"] != (LONG_CONTEXT_SCHEMA if args.harness == "opencode" else SCHEMA)
        or plan["runtime_sha256"] != digest(_runtime())
        or plan["native_driver_sha256"]
        != (miles.LONG_NATIVE_DRIVER_SHA256 if args.harness == "opencode" else NATIVE_DRIVER_SHA256)
        or plan["execution"]["image"] != args.runtime_image
        or "priority_reason" in plan["execution"]
        or plan["execution"].get("cluster_target") not in {None, "dev", "prod"}
        or plan["run_name"] != args.name
        or plan["output_root"] != args.output_root
        or (args.harness == "opencode" and plan["execution"]["priority"] != "c1")
    ):
        raise ValueError("Miles plan/runtime drift")
    resources = plan["execution"]["resources"]
    minimums = {**RESOURCES, "cpu_request": MINIMUM_CPU_REQUEST}
    if any(
        quantity(resources[key]) < quantity(minimums[key])
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
            "image": args.runtime_image,
            "workers": args.nodes,
            "gpus_per_worker": args.gpus_per_node,
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
                "MILES_SESSION_MAX_NODES": str(args.session_node_cap),
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
    from . import miles_opencode
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
    long_horizon = data.get("schema") == "cyber_miles_data_v2"
    if data != plan["data"]:
        raise ValueError("staged data manifest changed")
    runtime_model_root = plan["model"]["root"]
    if args["model_root"] != runtime_model_root:
        raise ValueError("runtime model identity changed")
    initial_policy = plan["model"].get("initial_policy")
    episode_model_root = runtime_model_root
    if initial_policy is not None:
        # A validated runtime stage changes byte location, not scientific policy identity.
        episode_model_root = initial_policy.get("accepted_root")
        if not isinstance(episode_model_root, str):
            raise ValueError("accepted SFT episode identity is absent")
        if "runtime_stage" not in initial_policy and runtime_model_root != episode_model_root:
            raise ValueError("unstaged SFT runtime identity changed")
    policy_identity_root = args.get("policy_identity_root") or runtime_model_root
    if policy_identity_root != episode_model_root:
        raise ValueError("policy identity changed")
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
                or cfg["model"]["root"] != policy_identity_root
                or (
                    long_horizon
                    and (
                        cfg["model"].get("tito_family") != miles_opencode.TITO_FAMILY
                        or cfg["model"].get("served_id") != "model"
                    )
                )
                or cfg["rl"]
                != (
                    data["limits"]
                    if long_horizon
                    else {k: v for k, v in data["limits"].items() if k != "response_tokens"}
                )
                or (cfg.get("harness") != (data.get("harness") if long_horizon else None))
                or cfg["execution"]["required_task_tool_catalog_sha256"]
                != data["tool_catalog_sha256"]
                or cfg["initial_prompt_sha256"]
                != "sha256:" + hashlib.sha256(row["input"].encode()).hexdigest()
            ):
                raise ValueError("episode identity changed")
            selected[split].append(row)
    return selected


def _native_source(expected):
    from miles.utils.external_utils.command_utils import repo_base_dir

    path = Path(repo_base_dir) / "train.py"
    if _hash(path) != expected:
        raise ValueError("native Miles training driver changed")
    return path


def native_source():
    return _native_source(NATIVE_DRIVER_SHA256)


def native_source_for_plan(plan):
    """Plan-aware hook used by the shared lifecycle before spawning Miles."""
    if plan["arguments"]["harness"] == "opencode":
        return _native_source(miles.LONG_NATIVE_DRIVER_SHA256)
    return native_source()


def watchdog_hard_seconds(plan):
    if plan["arguments"]["harness"] == "opencode":
        from .miles_opencode import JOB_HARD_SECONDS

        return JOB_HARD_SECONDS
    return None


def native_args(plan):
    os.environ["MILES_USE_LEGACY_ROLLOUT_V1"] = "0"
    from miles.utils.arguments import parse_args

    previous = sys.argv
    try:
        sys.argv = [
            str(native_source_for_plan(plan)),
            *miles.arguments(miles.MilesConfig(**plan["arguments"])),
        ]
        args = parse_args()
        expected = {
            "data_source_path": "training.miles_text.TextDataSource",
            "tool_key": "tools",
            "start_rollout_id": 0,
            "load": plan["checkpoint"]["root"],
            "ref_load": plan["checkpoint"]["root"],
            "hf_checkpoint": plan["model"]["root"],
            "fleet_policy_identity_root": miles.resolved_policy_identity_root(
                miles.MilesConfig(**plan["arguments"])
            ),
            "num_rollout": plan["arguments"]["steps"],
            "num_steps_per_rollout": 1,
            "global_batch_size": plan["arguments"]["groups"]
            * plan["arguments"]["samples_per_prompt"],
            "calculate_per_token_loss": True,
            "grpo_std_normalization": False,
        }
        if plan["arguments"]["harness"] == "opencode":
            from . import miles_opencode

            expected.update(
                {
                    "use_session_server": "v2",
                    "max_seq_len": 262_144,
                    "tito_model": miles_opencode.TITO_FAMILY,
                    "custom_agent_function_path": "training.miles_opencode.run",
                    "session_sample_picker_path": (
                        "training.miles_opencode.pick_compaction_segments"
                    ),
                    "session_sample_postprocessor_path": (
                        "training.miles_opencode.postprocess_compaction_segments"
                    ),
                    "sglang_router_policy": "consistent_hashing",
                }
            )
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
    native_source_for_plan(plan)
    if config.harness == "opencode":
        from miles.utils.chat_template_utils.tito_tokenizer import (
            resolve_fixed_chat_template,
            resolve_reasoning_and_tool_call_parser,
        )

        from . import miles_opencode

        tito_template, kwargs = resolve_fixed_chat_template(miles_opencode.TITO_FAMILY)
        if (
            tito_template is None
            or _hash(Path(tito_template)) != miles.LONG_TITO_TEMPLATE_SHA256
            or kwargs != {"preserve_thinking": True, "reasoning_effort": "xhigh"}
        ):
            raise ValueError("native Qwen3.8 TITO family changed")
        if resolve_reasoning_and_tool_call_parser(miles_opencode.TITO_FAMILY) != (
            "qwen3",
            "qwen3_coder",
        ):
            raise ValueError("native Qwen3.8 reasoning/tool parser changed")
    chat_template_path = (
        tito_template
        if config.harness == "opencode"
        else argv[argv.index("--chat-template-path") + 1]
    )
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
            chat_template_path=chat_template_path,
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

    from .miles_promotion import validate_embedded_promotion

    # The immutable receipt and compiled candidate are revalidated inside the
    # allocated process. Deep evidence reopening already ran immediately before
    # submission and is intentionally not bundled into the training runtime.
    validate_embedded_promotion(plan, check_files=False)

    # Keep slow payload validation in the child so the parent bounds startup,
    # including stalled storage, before Ray or model loading begins.
    check_artifacts(plan)
    args = native_args(plan)
    source = native_source_for_plan(plan)
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
                "MILES_SESSION_MAX_NODES": str(plan["arguments"]["session_node_cap"]),
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
