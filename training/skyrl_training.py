"""Exact Fleet GRPO through native SkyRL, using the shared Jobs API lifecycle.

Preparation/CPU checks are not real reward, optimizer or recovery qualification.
The initial profile starts from the pinned base; it never auto-resumes a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import numbers
import os
import re
import sys
import time
from contextlib import suppress
from pathlib import Path

from cyber_post_train.jobs import API_URLS, JobsError, bundled_request, digest, quantity

from . import skyrl
from .miles_conversion import _hash, check_inputs
from .rl_runtime import sealed
from .skyrl_episode import _module

SCHEMA = "cyber_skyrl_training_v1"
MODULE = "training.skyrl_training"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
NATIVE = {
    **skyrl.NATIVE_SOURCES,
    "skyrl.train.entrypoints.main_base": (
        "aee8976aa5d18a0c19be93e0b99fc1d1868688af0fe26b08b8a863c0aed027d7"
    ),
    # The immutable image adds phase-utilization and log-probability telemetry
    # to upstream f5bc3b78; optimizer and checkpoint semantics are unchanged.
    "skyrl.train.trainer": "0e3e2e4d1bbe28f8920ddd3c73999fae27308f949b60c8ea8062d41527dc3b21",
    "skyrl.train.dataset.dataset": (
        "ff041e24a24e7d99c9c20052ae643b137acb0f1777016ddba79260a42226466c"
    ),
}
RUNTIME_FILES = (
    "training/skyrl_training.py",
    "training/skyrl.py",
    "training/skyrl_rollout.py",
    "training/skyrl_episode.py",
    "training/rl_episode.py",
    "training/rl_reward_canary.py",
    "training/rl_runtime.py",
    "training/rl_data.py",
    "training/sft_runtime.py",
    "training/dense.py",
    "training/io.py",
    "training/sft.py",
    "training/models.py",
    "training/corpus.py",
    "training/splits.py",
    "training/qwen_tools.py",
    "training/miles_conversion.py",
    "training/miles.py",
    "evals/fleet/opencode_self_hosted.py",
    "cyber_post_train/jobs.py",
)


def _runtime():
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_text() for name in RUNTIME_FILES}


def compile_rl(config, *, relative_to):
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
        "RL",
    )
    if config["backend"] != "skyrl":
        raise ValueError("no silent RL backend substitution")
    model, data, w = (config[k] for k in ("model", "data", "wandb"))
    cluster, recipe = config.get("cluster", {}), config.get("recipe", {})
    _known(model, {"lock", "weights", "root"}, "model")
    _known(data, {"manifest", "root"}, "data")
    _known(w, {"entity", "project", "run_id"}, "W&B")
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
    metadata = read_mapping(relative_to / data["manifest"])
    sealed(metadata, "cyber_skyrl_data_v1")
    if set(metadata["files"]) != {"train", "dev"} or any(
        item["path"] != split + ".jsonl" for split, item in metadata["files"].items()
    ):
        raise ValueError("exact native train/dev files required")
    root, limits = Path(_sfs_root(data["root"], "data root")), metadata["limits"]
    qualification = None
    if config.get("qualification") is not None:
        from .rl_reward_canary import validate_run_config

        qualification = validate_run_config(
            config,
            metadata,
            bound,
            relative_to=relative_to,
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
        wandb_entity=w["entity"],
        wandb_project=w["project"],
        wandb_run_id=w["run_id"],
        context_tokens=limits["context_tokens"],
        response_tokens=limits["response_tokens"],
        tokens_per_turn=limits["max_tokens_per_turn"],
        max_turns=limits["max_turns"],
        **recipe,
    )
    if metadata["name"] != args.name or any(
        metadata["tokenizer"][k] != bound[k] for k in ("repo", "revision")
    ):
        raise ValueError("run/model/data identity mismatch")
    cluster_target = qualification["cluster_target"] if qualification else cluster.get(
        "target", "prod"
    )
    if cluster_target not in API_URLS:
        raise ValueError("cluster target must be dev or prod")
    plan = {
        "schema": SCHEMA,
        "run_name": args.name,
        "output_root": args.output_root,
        "model": bound,
        "data": metadata,
        "arguments": dataclasses.asdict(args),
        "native_overrides": skyrl.overrides(args),
        "native_sources": NATIVE,
        "runtime_sha256": digest(_runtime()),
        "execution": {
            "image": qualification["image"] if qualification else IMAGE,
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
            "cluster_target": cluster_target,
            "jobs_api_base_url": API_URLS[cluster_target],
        },
    }
    if qualification is not None:
        plan["qualification"] = qualification
    job_request(plan)
    return plan


def job_request(plan):
    args = skyrl.SkyRLConfig(**plan["arguments"])
    qualification = plan.get("qualification")
    image, extra_env = IMAGE, {}
    bound_route = None
    if qualification is not None:
        from .rl_reward_canary import validate_plan_binding

        binding = validate_plan_binding(qualification, plan["data"], plan["arguments"])
        image, extra_env = binding["image"], binding["environment"]
        bound_route = (binding["cluster_target"], binding["jobs_api_base_url"])
    if (
        plan["schema"] != SCHEMA
        or plan["runtime_sha256"] != digest(_runtime())
        or plan["native_sources"] != NATIVE
        or plan["native_overrides"] != skyrl.overrides(args)
        or plan["execution"]["image"] != image
        or plan["run_name"] != args.name
        or plan["output_root"] != args.output_root
        or plan["execution"].get("cluster_target") not in API_URLS
        or plan["execution"].get("jobs_api_base_url")
        != API_URLS[plan["execution"]["cluster_target"]]
        or (
            bound_route is not None
            and (
                plan["execution"]["cluster_target"],
                plan["execution"]["jobs_api_base_url"],
            )
            != bound_route
        )
    ):
        raise ValueError("SkyRL plan/runtime drift")
    resources = plan["execution"]["resources"]
    if quantity(resources["cpu_request"]) < 64 or quantity(resources["memory_request"]) < quantity(
        "512Gi"
    ):
        raise ValueError("native Qwen RL requires its reviewed loading reservation")
    files = _runtime()
    files.update(
        {p + "/__init__.py": "" for p in ("training", "evals", "evals/fleet", "cyber_post_train")}
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": args.name,
            "title": args.name + " native SkyRL RL",
            "run_dir": args.output_root,
            "image": image,
            "workers": args.nodes,
            "gpus_per_worker": 8,
            "resources": resources,
            "priority_class": plan["execution"]["priority"],
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
                **extra_env,
            },
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def validate_gpu_runtime_user() -> None:
    """Fail before model loading if the GPU process is not the reviewed user."""
    expected = (
        os.environ.get("CYBER_EXPECTED_RUNTIME_UID"),
        os.environ.get("CYBER_EXPECTED_RUNTIME_GID"),
    )
    if expected != ("1000", "100") or (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("SkyRL GPU runtime must use the pinned image user 1000:100")


def validate_preview(plan: dict, request: dict, preview: dict) -> dict:
    """Require the rendered GPU containers to run as UID 1000/GID 100.

    Generic preview validation covers resources, queueing, image, secrets and
    command identity.  This profile gate additionally checks the effective pod
    and container security contexts.  Missing identity is not inferred from an
    image Dockerfile; the preview must prove it.
    """
    if job_request(plan) != request:
        raise JobsError("SkyRL preview request differs from its immutable plan")
    return validate_gpu_runtime_preview(request, preview)


def validate_gpu_runtime_preview(request: dict, preview: dict) -> dict:
    """Validate the common pinned-image runtime identity for SkyRL GPU jobs."""
    import yaml

    from cyber_post_train.jobs import validate_preview as validate_generic_preview

    result = validate_generic_preview(request, preview)
    try:
        obj = yaml.safe_load(preview["manifest_yaml"])
        cluster = obj["spec"]["rayClusterSpec"]
        templates = [cluster["headGroupSpec"]["template"]] + [
            group["template"]
            for group in cluster.get("workerGroupSpecs", [])
            if group.get("replicas", 0)
        ]
        checked = 0
        for template in templates:
            pod = template["spec"]
            pod_context = pod.get("securityContext", {})
            containers = [
                container
                for container in pod["containers"]
                if quantity(
                    container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0)
                )
                > 0
            ]
            if len(containers) != 1:
                raise JobsError("SkyRL preview must contain one GPU container per pod")
            context = {**pod_context, **containers[0].get("securityContext", {})}
            if (
                context.get("runAsUser") != 1000
                or context.get("runAsGroup") != 100
                or context.get("runAsNonRoot") is not True
            ):
                raise JobsError("SkyRL preview does not prove runtime user 1000:100")
            checked += 1
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("malformed SkyRL Jobs API preview") from exc
    if checked != request["workers"]:
        raise JobsError("SkyRL preview runtime-user worker count drift")
    return {**result, "runtime_user": {"uid": 1000, "gid": 100}, "workers_checked": checked}


def check_artifacts(plan):
    """Hash payloads and reconcile exact task families without exposing task text."""
    from .rl_data import selection
    from .rl_episode import _validate

    check_inputs(plan)
    data, args = plan["data"], plan["arguments"]
    root = Path(args["data_manifest"]).parent
    if json.loads((root / "manifest.json").read_bytes()) != data:
        raise ValueError("staged data manifest changed")
    split, task_set = (json.loads((root / p).read_bytes()) for p in ("split.json", "task-set.json"))
    if split["sha256"] != data["split_sha256"] or task_set["sha256"] != data["selection_sha256"]:
        raise ValueError("reviewed task-set/split changed")
    expected = {
        (r["split"], r["task_key"], r["task_version_id"]) for r in selection(task_set, split)
    }
    rows, selected = {}, set()
    for group, item in data["files"].items():
        path = root / item["path"]
        if _hash(path) != item["sha256"].removeprefix("sha256:"):
            raise ValueError("RL data payload changed")
        rows[group] = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(rows[group]) != item["rows"] or not rows[group]:
            raise ValueError("RL data count changed")
        for row in rows[group]:
            cfg = json.loads(row["cyber_config_json"])
            _validate(cfg)
            key = (group, cfg["task"]["key"], cfg["task"]["version_id"])
            if (
                key in selected
                or row["split"] != group
                or row["env_class"] != cfg["environment"]["id"]
                or cfg["run_id"] != args["name"]
                or len(row["prompt"]) != 1
                or row["prompt"][0]["role"] != "user"
                or "sha256:" + hashlib.sha256(row["prompt"][0]["content"].encode()).hexdigest()
                != cfg["task"]["prompt_sha256"]
                or any(cfg["model"][k] != plan["model"][k] for k in ("repo", "revision", "root"))
                or cfg["model"]["runtime_chat_template_sha256"] != data["template_sha256"]
                or cfg["rl"] != {k: v for k, v in data["limits"].items() if k != "response_tokens"}
                or cfg["execution"]["required_task_tool_catalog_sha256"]
                != data["tool_catalog_sha256"]
            ):
                raise ValueError("RL episode identity changed")
            selected.add(key)
    if selected != expected:
        raise ValueError("selected RL task families differ from frozen split")
    return rows


def native_source():
    from .sft_runtime import validate_runtime_sources

    validate_runtime_sources()
    return {name: _module(name, sha) for name, sha in NATIVE.items()}


def dataset(plan, tokenizer, split, rows):
    native = _module("skyrl.train.dataset.dataset", NATIVE["skyrl.train.dataset.dataset"])
    value = native.PromptDataset(
        plan["arguments"][split + "_data"],
        tokenizer,
        plan["native_overrides"]["trainer.max_prompt_length"],
        num_workers=1,
    )
    if len(value) != len(rows) or any(
        value[i]
        != (
            row["prompt"],
            row["env_class"],
            {k: v for k, v in row.items() if k not in {"prompt", "env_class"}},
            str(i),
        )
        for i, row in enumerate(rows)
    ):
        raise ValueError("native SkyRL silently filtered or changed a task")
    return value


def preflight(plan):
    # The pinned GPU image is UID 1000/GID 100. Root can read private staging
    # files that its trainer cannot; such a preflight is not representative.
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("SkyRL CPU preflight must use the pinned image user 1000:100, not root")
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise ValueError("SkyRL preflight is CPU-only")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("RL output already exists")
    rows = check_artifacts(plan)
    native_source()
    skyrl.native_config(skyrl.SkyRLConfig(**plan["arguments"]))
    tokenizer = AutoTokenizer.from_pretrained(
        plan["model"]["root"], trust_remote_code=False, local_files_only=True
    )
    if "sha256:" + digest_template(tokenizer) != plan["data"]["template_sha256"]:
        raise ValueError("native template changed")
    for split in rows:
        dataset(plan, tokenizer, split, rows[split])
    return {
        "schema": "cyber_skyrl_training_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "native_parser_checked": True,
        "counts": {k: len(v) for k, v in rows.items()},
        "planned_steps": plan["arguments"]["steps"],
        "rl_qualified": False,
    }


def digest_template(tokenizer):
    return hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()


class ScalarTracking:
    """Native tracker interface with no text/tables and explicit outer finalization."""

    def __init__(self, plan):
        import wandb

        from .sft_runtime import _configure_wandb

        args = plan["arguments"]
        _configure_wandb(
            {
                "output_root": plan["output_root"],
                "wandb": {
                    "entity": args["wandb_entity"],
                    "group": args["name"],
                    "run_id": args["wandb_run_id"],
                },
            }
        )
        self.root, self.logger = Path(plan["output_root"]), wandb
        run = wandb.init(
            entity=args["wandb_entity"],
            project=args["wandb_project"],
            id=args["wandb_run_id"],
            name=args["name"],
            resume="never",
            config={"plan_sha256": digest(plan), "arguments": args},
        )
        if (run.id, run.entity, run.project) != (
            args["wandb_run_id"],
            args["wandb_entity"],
            args["wandb_project"],
        ):
            raise ValueError("W&B run identity mismatch")

    def log(self, data, step, commit=False):
        if (
            type(step) is not int
            or step < 0
            or any(
                not isinstance(k, str)
                or not re.fullmatch(r"[A-Za-z0-9_./@() -]+", k)
                or isinstance(v, bool)
                or not isinstance(v, numbers.Real)
                or not math.isfinite(v)
                for k, v in data.items()
            )
        ):
            raise ValueError("native telemetry must contain finite scalars only")
        data = {k: float(v) for k, v in data.items()}
        with (self.root / "metrics.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"optimizer_step": step, "time": time.time(), **data}, allow_nan=False)
                + "\n"
            )
            stream.flush()
        self.logger.log(data=data, step=step, commit=commit)

    def finish(self):
        pass  # only the outer run knows whether checkpoint/evidence checks passed

    def log_exception(self, error, step=0):
        pass  # native otherwise uploads complete private tracebacks to W&B

    def log_samples_to_table(self, *args, **kwargs):
        raise ValueError("trajectory uploads are disabled")


def _native(plan):
    import ray
    import wandb
    from skyrl.backends.skyrl_train.utils.ppo_utils import sync_registries

    from .skyrl_rollout import Generator

    rows, modules = check_artifacts(plan), native_source()
    args = skyrl.SkyRLConfig(**plan["arguments"])
    cfg = skyrl.native_config(args)
    base = modules["skyrl.train.entrypoints.main_base"].BasePPOExp

    class Experiment(base):
        def get_train_dataset(self):
            return dataset(plan, self.tokenizer, "train", rows["train"])

        def get_eval_dataset(self):
            return dataset(plan, self.tokenizer, "dev", rows["dev"])

        def get_generator(self, cfg, tokenizer, engine):
            return Generator(
                args.data_manifest,
                plan["data"]["sha256"],
                tokenizer,
                engine,
                Path(args.output_root) / "episodes",
                response_tokens=args.response_tokens,
                repetitions={"train": args.samples_per_prompt, "eval": 1},
                concurrency=args.groups * args.samples_per_prompt,
            )

        def get_tracker(self):
            return ScalarTracking(plan)

        def get_trajectory_logger(self):
            return None

    env = modules["skyrl.train.utils.utils"].prepare_runtime_environment(cfg)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1]) + ":" + os.environ.get("PYTHONPATH", "")
    )
    ray.init(address="auto", log_to_driver=False, runtime_env={"env_vars": env})
    code = 1
    try:
        sync_registries()
        experiment = Experiment(cfg)
        experiment.run()
        if experiment.trainer.global_step != args.steps:
            raise ValueError("native optimizer step limit changed")
        native_result(plan)
        code = 0
    finally:
        try:
            if wandb.run is not None:
                wandb.finish(exit_code=code)
        finally:
            ray.shutdown()


def native_result(plan):
    root, args = Path(plan["output_root"]), plan["arguments"]
    expected = (
        {("train", i) for i in range(1, args["steps"] + 1)}
        | {("eval", 0), ("eval", args["steps"])}
        | {("eval", i) for i in range(1, args["steps"] + 1) if i % args["eval_interval"] == 0}
    )
    seen = set()
    for directory in (root / "episodes/batches").iterdir():
        value = json.loads((directory / "COLLECTED.json").read_bytes())
        sealed(value, "cyber_skyrl_batch_v1")
        key = (value["phase"], value["global_step"])
        if (
            key not in expected
            or key in seen
            or (directory / "FAILED.json").exists()
            or value["data_sha256"] != plan["data"]["sha256"]
        ):
            raise ValueError("native batch completion evidence mismatch")
        seen.add(key)
    if seen != expected:
        raise ValueError("native training/dev batch missing")
    checkpoint = root / f"checkpoints/global_step_{args['steps']}"
    if (
        (root / "checkpoints/latest_ckpt_global_step.txt").read_text().strip() != str(args["steps"])
        or any(
            not (checkpoint / p).is_file() or (checkpoint / p).stat().st_size == 0
            for p in ("data.pt", "trainer_state.pt")
        )
        or not any(p.is_file() and p.stat().st_size for p in (checkpoint / "policy").rglob("*"))
    ):
        raise ValueError("native final checkpoint/sampler missing")
    return {
        "status": "native_loop_returned",
        "plan_sha256": digest(plan),
        "checkpoint_global_step": args["steps"],
        "completed_batches": len(seen),
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
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        job_request(plan)
        validate_gpu_runtime_user()
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
            print(json.dumps({k: result[k] for k in ("status", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
