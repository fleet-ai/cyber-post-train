"""Production-only SkyRL compiler with plan-bound watchdog and launch gates.

The existing one-step reward-canary compiler is intentionally untouched so its
frozen request remains reproducible.  Full arms use this separate schema and
runtime entrypoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import API_URLS, JobsError, bundled_request, digest, quantity

from . import skyrl
from . import skyrl_production as production
from . import skyrl_training as base

SCHEMA = "cyber_skyrl_production_training_v1"
MODULE = "training.skyrl_production_training"
RUNTIME_FILES = (
    *base.RUNTIME_FILES,
    "training/skyrl_production.py",
    "training/skyrl_production_training.py",
)


def _runtime() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    if len(RUNTIME_FILES) != len(set(RUNTIME_FILES)):
        raise ValueError("SkyRL production runtime file list contains duplicates")
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def compile_rl(config: dict, *, relative_to: Path) -> dict:
    from .models import bound_model
    from .sft import RESOURCES, _known, _sfs_root, read_mapping

    _known(
        config,
        {
            "backend",
            "name",
            "output_root",
            "model",
            "data",
            "recipe",
            "wandb",
            "cluster",
            "qualification",
        },
        "production RL",
    )
    if config.get("backend") != "skyrl" or config.get("qualification") is None:
        raise ValueError("production SkyRL requires its qualification closure")
    model, data, wandb = (config[key] for key in ("model", "data", "wandb"))
    cluster, recipe = config.get("cluster", {}), config.get("recipe", {})
    _known(model, {"lock", "weights", "root"}, "model")
    _known(data, {"manifest", "root"}, "data")
    _known(wandb, {"entity", "project", "run_id"}, "W&B")
    _known(cluster, {"priority", "resources", "target"}, "cluster")
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
            "keep_checkpoints",
            "seed",
        },
        "SkyRL recipe",
    )
    bound = bound_model(
        read_mapping(relative_to / model["lock"]),
        read_mapping(relative_to / model["weights"]),
        _sfs_root(model["root"], "model root"),
    )
    metadata = read_mapping(Path(data["manifest"]))
    from .rl_runtime import sealed

    sealed(metadata, production.MANIFEST_SCHEMA)
    if set(metadata.get("files", {})) != {"train", "dev"} or any(
        item.get("path") != split + ".jsonl" for split, item in metadata["files"].items()
    ):
        raise ValueError("exact native train/dev files required")
    root, limits = Path(_sfs_root(data["root"], "data root")), metadata["limits"]
    qualification = production.validate_run_config(
        config,
        metadata,
        bound,
        read_mapping(relative_to / config["qualification"]),
    )
    args = skyrl.SkyRLConfig(
        name=config["name"],
        output_root=_sfs_root(config["output_root"], "output root"),
        model=bound["repo"],
        model_root=bound["root"],
        train_data=str(root / "train.jsonl"),
        dev_data=str(root / "dev.jsonl"),
        data_manifest=str(root / "manifest.json"),
        train_rows=metadata["files"]["train"]["rows"],
        dev_rows=metadata["files"]["dev"]["rows"],
        wandb_entity=wandb["entity"],
        wandb_project=wandb["project"],
        wandb_run_id=wandb["run_id"],
        context_tokens=limits["context_tokens"],
        response_tokens=limits["response_tokens"],
        tokens_per_turn=limits["max_tokens_per_turn"],
        max_turns=limits["max_turns"],
        **recipe,
    )
    if metadata["name"] != args.name or any(
        metadata["tokenizer"][key] != bound[key] for key in ("repo", "revision")
    ):
        raise ValueError("production run/model/data identity mismatch")
    arguments = dataclasses.asdict(args)
    for key in (
        "generation_chunk_tokens",
        "compaction_trigger_tokens",
        "compaction_summary_tokens",
        "compaction_enabled",
    ):
        arguments.pop(key)
    plan = {
        "schema": SCHEMA,
        "run_name": args.name,
        "output_root": args.output_root,
        "model": bound,
        "data": metadata,
        "arguments": arguments,
        "native_overrides": skyrl.overrides(args),
        "native_sources": base.NATIVE,
        "runtime_sha256": digest(_runtime()),
        "execution": {
            "image": qualification["image"],
            "priority": cluster["priority"],
            "resources": {**RESOURCES, **cluster["resources"]},
            "cluster_target": qualification["cluster_target"],
            "jobs_api_base_url": qualification["jobs_api_base_url"],
        },
        "watchdog": qualification["watchdog"],
        "qualification": qualification,
    }
    job_request(plan)
    return plan


def job_request(plan: dict) -> dict:
    args = skyrl.SkyRLConfig(**plan["arguments"])
    binding = production.validate_plan_binding(
        plan.get("qualification"), plan["data"], plan["arguments"], plan["model"]
    )
    if (
        plan.get("schema") != SCHEMA
        or plan.get("runtime_sha256") != digest(_runtime())
        or plan.get("native_sources") != base.NATIVE
        or plan.get("native_overrides") != skyrl.overrides(args)
        or plan.get("watchdog") != binding["watchdog"]
        or plan.get("execution")
        != {
            "image": binding["image"],
            "priority": "c1",
            "resources": production.RESOURCES,
            "cluster_target": "prod",
            "jobs_api_base_url": API_URLS["prod"],
        }
        or plan.get("run_name") != args.name
        or plan.get("output_root") != args.output_root
    ):
        raise ValueError("SkyRL production plan/runtime drift")
    resources = plan["execution"]["resources"]
    if quantity(resources["cpu_request"]) < 64 or quantity(resources["memory_request"]) < quantity(
        "512Gi"
    ):
        raise ValueError("native Qwen RL requires its reviewed loading reservation")
    files = _runtime()
    files.update(
        {
            path + "/__init__.py": ""
            for path in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": args.name,
            "title": args.name + " native SkyRL RL",
            "run_dir": args.output_root,
            "image": binding["image"],
            "workers": args.nodes,
            "gpus_per_worker": 8,
            "resources": resources,
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": ["fleet-api", "wandb-api"],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "WANDB_MODE": "online",
                "WANDB_RUN_ID": args.wandb_run_id,
                "WANDB_DISABLE_CODE": "true",
                "WANDB_CONSOLE": "off",
                "PYTHONUNBUFFERED": "1",
                "CYBER_EXPECTED_RUNTIME_UID": "1000",
                "CYBER_EXPECTED_RUNTIME_GID": "100",
                **binding["environment"],
            },
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def validate_preview(plan: dict, request: dict, preview: dict) -> dict:
    if job_request(plan) != request:
        raise JobsError("SkyRL production preview request differs from its immutable plan")
    return base.validate_gpu_runtime_preview(request, preview)


def preflight(plan: dict) -> dict:
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("SkyRL production preflight must run as image user 1000:100")
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise ValueError("SkyRL production preflight is CPU-only")
    request = job_request(plan)
    staged = production.validate_staged_data(plan)
    rows = base.check_artifacts(plan)
    base.native_source()
    skyrl.native_config(skyrl.SkyRLConfig(**plan["arguments"]))
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], trust_remote_code=False, local_files_only=True
    )
    if "sha256:" + base.digest_template(tokenizer) != plan["data"]["template_sha256"]:
        raise ValueError("native template changed")
    for split in rows:
        base.dataset(plan, tokenizer, split, rows[split])
    return {
        "schema": "cyber_skyrl_production_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "staged_data_validation_sha256": staged["sha256"],
        "fresh_sfs_output_absence": staged["output_absent"],
        "native_parser_checked": True,
        "counts": {key: len(value) for key, value in rows.items()},
        "planned_steps": plan["arguments"]["steps"],
        "rl_qualified": False,
    }


def native_source() -> dict:
    return base.native_source()


def native_result(plan: dict) -> dict:
    return base.native_result(plan)


def _native(plan: dict) -> None:
    base._native(plan)


def run(plan: dict, plan_path: Path) -> dict:
    """Apply only this plan's reviewed watchdog to the outer RL process."""
    from . import sft_runtime
    from .rl_runtime import run as supervised_run

    watchdog = production.validate_plan_binding(
        plan.get("qualification"), plan["data"], plan["arguments"], plan["model"]
    )["watchdog"]
    names = {
        "WATCHDOG_POLL_SECONDS": "poll_seconds",
        "WATCHDOG_STARTUP_SECONDS": "startup_seconds",
        "WATCHDOG_IDLE_SECONDS": "idle_seconds",
        "WATCHDOG_HARD_SECONDS": "hard_seconds",
        "WATCHDOG_DRAIN_SECONDS": "drain_seconds",
    }
    previous = {name: getattr(sft_runtime, name) for name in names}
    try:
        for name, field in names.items():
            setattr(sft_runtime, name, watchdog[field])
        return supervised_run(plan, plan_path, backend=sys.modules[__name__])
    finally:
        for name, value in previous.items():
            setattr(sft_runtime, name, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        job_request(plan)
        base.validate_gpu_runtime_user()
        if args.native:
            try:
                _native(plan)
            except BaseException as exc:
                from .rl_runtime import native_failure, native_rejection

                if native_rejection(plan, exc):
                    return
                with suppress(Exception):
                    native_failure(plan, exc)
                raise
        else:
            result = run(plan, args.plan)
            print(json.dumps({key: result[key] for key in ("status", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
