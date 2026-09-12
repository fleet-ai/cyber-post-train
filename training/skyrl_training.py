"""Exact Fleet GRPO through native SkyRL, using the shared Jobs API lifecycle.

Preparation/CPU checks are not real reward, optimizer or recovery qualification.
The initial profile starts from the pinned base; it never auto-resumes a run.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import numbers
import os
import re
import sys
import time
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from pathlib import Path

from cyber_post_train.jobs import bundled_request, digest, quantity

from . import skyrl
from .miles_conversion import _hash, _write, check_inputs
from .rl_runtime import HardDeadlineExceeded, hard_deadline, sealed
from .skyrl_episode import _module

SCHEMA = "cyber_skyrl_training_v1"
ENGINE_DIAGNOSTIC_SCHEMA = "cyber_skyrl_engine_start_diagnostic_v1"
ENGINE_DIAGNOSTIC_WORKERS = 2
ENGINE_DIAGNOSTIC_GPUS_PER_WORKER = 4
_CREDENTIAL_ENV_NAME = re.compile(
    r"(?:^|_)(?:TOKENS?|PASSWORDS?|PASSWD|CREDENTIALS?|SECRETS?|API_KEYS?|"
    r"ACCESS_KEYS?|PRIVATE_KEYS?|DATABASE_URL|AUTH(?:ORIZATION)?)(?:_|$)",
    re.IGNORECASE,
)
_ALWAYS_SCRUB_WORKER_ENV = frozenset({"FLEET_API_KEY", "WANDB_API_KEY"})
MODULE = "training.skyrl_training"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "9b6f43938f9b28aaff7ba91edd59be3d18b01f9a22078ba5475cdac5e6bfcca6"
)
ENGINE_IMAGE_CPU_QUALIFICATION = {
    "schema": "cyber_q38_rl_image_cleanpull_cpu_qualification_v1",
    "status": "qualified",
    "classification": "operational_gate",
    "source_commit": "de9e6b7cf087d12c5ca371c9ec916a54757d369f",
    "receipt_sha256": "3ebe51bc1c28c9143cc31d1f5271e20badee0a18db0409754880f30164cacfc7",
    "evidence_path": (
        "docs/evidence/qwen38-study/2026-09-12-skyrl-replacement-image-cpu-qualification-v1.json"
    ),
}
_ENGINE_START_DISQUALIFIED = frozenset(
    {
        (
            "Qwen/Qwen3.8-27B",
            "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
            "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4",
        )
    }
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
    "training/rl_runtime.py",
    "training/rl_data.py",
    "training/sft_runtime.py",
    "training/dense.py",
    "training/io.py",
    "training/sft.py",
    "training/models.py",
    "training/corpus.py",
    "training/source_coverage.py",
    "training/study_data.py",
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
        {"backend", "name", "output_root", "model", "data", "recipe", "wandb", "cluster"},
        "RL",
    )
    if config["backend"] != "skyrl":
        raise ValueError("no silent RL backend substitution")
    model, data, w = (config[k] for k in ("model", "data", "wandb"))
    cluster, recipe = config.get("cluster", {}), config.get("recipe", {})
    _known(model, {"lock", "weights", "root"}, "model")
    _known(data, {"manifest", "root"}, "data")
    _known(w, {"entity", "project", "run_id"}, "W&B")
    _known(cluster, {"priority", "resources"}, "cluster")
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
            "engine_start_timeout_seconds",
            "engine_cleanup_timeout_seconds",
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
            "image": IMAGE,
            "image_cpu_qualification": dict(ENGINE_IMAGE_CPU_QUALIFICATION),
            "priority": cluster.get("priority", "c1"),
            "resources": {**RESOURCES, **cluster.get("resources", {})},
        },
    }
    job_request(plan)
    return plan


def job_request(plan):
    args = skyrl.SkyRLConfig(**plan["arguments"])
    if (
        plan["schema"] != SCHEMA
        or plan["runtime_sha256"] != digest(_runtime())
        or plan["native_sources"] != NATIVE
        or plan["native_overrides"] != skyrl.overrides(args)
        or plan["execution"]["image"] != IMAGE
        or plan["execution"].get("image_cpu_qualification") != ENGINE_IMAGE_CPU_QUALIFICATION
        or plan["run_name"] != args.name
        or plan["output_root"] != args.output_root
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
            "image": IMAGE,
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
            },
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def _diagnostic_shape(plan, cfg=None):
    """Bind the fragmented-dev diagnostic shape, never the training shape."""
    overrides = plan["native_overrides"]
    if (
        plan["arguments"]["nodes"] != 1
        or overrides.get("generator.inference_engine.num_engines") != 2
        or overrides.get("generator.inference_engine.tensor_parallel_size") != 4
    ):
        raise ValueError("engine diagnostic requires the exact one-node 2xTP4 engine plan")
    if cfg is not None:
        engine = cfg.generator.inference_engine
        if (
            engine.num_engines != 2
            or engine.tensor_parallel_size != 4
            or engine.pipeline_parallel_size != 1
            or engine.data_parallel_size != 1
        ):
            raise ValueError("native engine shape differs from diagnostic cluster binding")
    return {
        "diagnostic_workers": ENGINE_DIAGNOSTIC_WORKERS,
        "diagnostic_gpus_per_worker": ENGINE_DIAGNOSTIC_GPUS_PER_WORKER,
        "diagnostic_total_gpus": (ENGINE_DIAGNOSTIC_WORKERS * ENGINE_DIAGNOSTIC_GPUS_PER_WORKER),
        "num_engines": 2,
        "tensor_parallel_size": 4,
    }


def engine_diagnostic_request(plan):
    """Build a no-rollout, no-secret engine-start diagnostic for a fresh plan.

    The caller must compile the source configuration under a new diagnostic
    name/output root.  Reusing a failed run directory is intentionally not
    supported.
    """
    job_request(plan)  # exact plan/runtime/image/resource validation
    shape = _diagnostic_shape(plan)
    args = skyrl.SkyRLConfig(**plan["arguments"])
    resources = plan["execution"]["resources"]
    files = _runtime()
    files.update(
        {p + "/__init__.py": "" for p in ("training", "evals", "evals/fleet", "cyber_post_train")}
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": args.name,
            "title": args.name + " SkyRL engine-start diagnostic",
            "run_dir": args.output_root,
            "image": IMAGE,
            "workers": shape["diagnostic_workers"],
            "gpus_per_worker": shape["diagnostic_gpus_per_worker"],
            "resources": resources,
            "priority_class": plan["execution"]["priority"],
            "requeueIfPreempted": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        MODULE,
        ["--plan", "plan.json", "--sha256", digest(plan), "--engine-diagnostic"],
    )


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


def _require_engine_start_qualified_image(plan):
    """Reject only exact model/image pairs disproven by terminal dev evidence."""
    identity = (plan["model"]["repo"], plan["execution"]["image"])
    if identity in _ENGINE_START_DISQUALIFIED:
        raise ValueError(
            "Qwen3.8 SkyRL image is engine-start disqualified by the sealed "
            "dev5/dev6 evidence; qualify and pin a replacement image before GPU submission"
        )


def _validate_vllm_startup_error_transport():
    """Prove the installed safe startup wrapper crosses Ray's pickle boundary.

    vLLM raises this exception inside a Ray actor.  Ray 2.56 checks only that a
    cause can be serialized before placing it in ``RayTaskError``; the driver
    can still lose the cause later if reconstructing the exception fails.  A
    GPU engine smoke cannot be the first place that constructor mismatch is
    discovered.
    """
    import pickle

    ray_errors = importlib.import_module("ray.exceptions")
    ray_pickle = importlib.import_module("ray.cloudpickle")
    startup = importlib.import_module("vllm.v1.engine.fleet_startup_error")
    error_type = getattr(startup, "FleetVllmStartupError", None)
    schema = getattr(startup, "SCHEMA", None)
    if not isinstance(error_type, type) or schema != "fleet_vllm_startup_error_v1":
        raise ValueError("installed vLLM startup exception contract changed")

    expected = {
        "schema": schema,
        "exception_class": "TypeError",
        "frame": {"source_id": "model", "line": 1},
    }
    secret = "SYNTHETIC_PRIVATE_STARTUP_TRANSPORT_PROBE"
    poisoned = {
        "schema": schema,
        "exception_class": secret,
        "frame": {"source_id": secret, "line": secret},
        "message": secret,
    }
    probes = (
        ("engine_core", expected, expected),
        (
            secret,
            poisoned,
            {"schema": schema, "exception_class": "Other", "frame": None},
        ),
    )
    for codec_name, codec in (("pickle", pickle), ("ray.cloudpickle", ray_pickle)):
        for stage, cause, normalized in probes:
            error = error_type(stage, cause)
            try:
                payload = codec.dumps(error)
                restored = codec.loads(payload)
            except Exception as exc:
                raise ValueError(
                    f"installed vLLM startup exception failed {codec_name} round-trip"
                ) from exc
            if (
                type(restored) is not error_type
                or getattr(restored, "sanitized_cause", None) != normalized
                or secret.encode() in payload
                or secret in str(restored)
                or secret in repr(getattr(restored, "sanitized_cause", None))
            ):
                raise ValueError(
                    f"installed vLLM startup exception failed {codec_name} privacy contract"
                )
    for stage, cause, normalized in probes:
        error = error_type(stage, cause)
        envelope = ray_errors.RayTaskError(
            "engine_start",
            "synthetic safe traceback",
            error,
            proctitle="engine-start-preflight",
            pid=1,
            ip="127.0.0.1",
        )
        try:
            payload = envelope.to_bytes()
            restored = ray_errors.RayError.from_bytes(payload)
        except Exception as exc:
            raise ValueError(
                "installed vLLM startup exception failed RayTaskError round-trip"
            ) from exc
        restored_cause = getattr(restored, "cause", None)
        if (
            isinstance(restored, ray_errors.UnserializableException)
            or type(restored_cause) is not error_type
            or getattr(restored_cause, "sanitized_cause", None) != normalized
            or secret.encode() in payload
            or secret in str(restored_cause)
            or secret in repr(getattr(restored_cause, "sanitized_cause", None))
        ):
            raise ValueError(
                "installed vLLM startup exception failed RayTaskError privacy contract"
            )


def preflight(plan):
    # The pinned GPU image is UID 1000/GID 100. Root can read private staging
    # files that its trainer cannot; such a preflight is not representative.
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("SkyRL CPU preflight must use the pinned image user 1000:100, not root")
    _require_engine_start_qualified_image(plan)
    import torch
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise ValueError("SkyRL preflight is CPU-only")
    request = job_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("RL output already exists")
    rows = check_artifacts(plan)
    native_source()
    _validate_vllm_startup_error_transport()
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
        "startup_error_transport_checked": True,
        "counts": {k: len(v) for k, v in rows.items()},
        "planned_steps": plan["arguments"]["steps"],
        "rl_qualified": False,
    }


def engine_diagnostic_preflight(plan):
    """Validate the engine-only request without reading task rows or using GPUs."""
    if (os.geteuid(), os.getegid()) != (1000, 100):
        raise ValueError("SkyRL CPU preflight must use the pinned image user 1000:100, not root")
    _require_engine_start_qualified_image(plan)
    import torch

    if torch.cuda.is_available():
        raise ValueError("SkyRL engine diagnostic preflight is CPU-only")
    from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args

    request = engine_diagnostic_request(plan)
    if Path(plan["output_root"]).exists():
        raise FileExistsError("RL diagnostic output already exists")
    check_inputs(plan)
    native_source()
    _validate_vllm_startup_error_transport()
    cfg = skyrl.diagnostic_native_config(skyrl.SkyRLConfig(**plan["arguments"]))
    shape = _diagnostic_shape(plan, cfg)
    build_vllm_cli_args(cfg)
    return {
        "schema": "cyber_skyrl_engine_diagnostic_cpu_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "runtime_user": {"uid": os.geteuid(), "gid": os.getegid()},
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "native_parser_checked": True,
        "engine_cli_args_checked": True,
        "startup_error_transport_checked": True,
        "model_files": len(plan["model"]["files"]),
        "task_rows_read": 0,
        "rollouts": False,
        "verifier_calls": False,
        "optimizer_updates": False,
        "checkpoints": False,
        "wandb": False,
        "engine_start_qualified": False,
        **shape,
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


def _prepare_infra_log(plan):
    """Create the one private shared file used by SkyRL infrastructure actors."""
    directory = Path(plan["output_root"]) / "private-native-logs"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    path = directory / "infra.log"
    with path.open("x"):
        pass
    os.chmod(path, 0o600)
    return path


def _infra_log_evidence(path):
    """Return metadata only; the infrastructure log always remains private."""
    signature = b"Engine core initialization failed. See root cause above."
    hasher = hashlib.sha256()
    size = 0
    overlap = b""
    signature_present = False
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
            window = overlap + chunk
            signature_present = signature_present or signature in window
            overlap = window[-(len(signature) - 1) :]
    return {
        "path": "private-native-logs/infra.log",
        "bytes": size,
        "sha256": hasher.hexdigest(),
        "engine_failure_signature_present": signature_present,
    }


def _string_environment(env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(env, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()
    ):
        raise ValueError("SkyRL Ray environment must contain only string variables")
    return dict(env)


def _credential_names(*environments: Mapping[str, str]) -> tuple[str, ...]:
    names = set(_ALWAYS_SCRUB_WORKER_ENV)
    for environment in environments:
        names.update(
            key for key in _string_environment(environment) if _CREDENTIAL_ENV_NAME.search(key)
        )
    return tuple(sorted(names))


def _scrubbed_actor_environment(environment, scrubbed, overlay=None):
    """Merge an explicit actor environment and blank credentials last."""
    value = _string_environment(environment)
    if overlay is not None:
        value.update(_string_environment(overlay))
    names = set(scrubbed)
    names.update(_credential_names(value))
    value.update(dict.fromkeys(sorted(names), ""))
    return value


@contextmanager
def _scrubbed_process_environment(names):
    """Temporarily blank credentials while a router child is launched."""
    missing = object()
    previous = {name: os.environ.get(name, missing) for name in names}
    try:
        for name in names:
            os.environ[name] = ""
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _DiagnosticContractRejected(ValueError):
    """An explicitly recognized diagnostic incompatibility, not a runtime defect."""


def _router_child_entry(target, names, probe_state, start_method, args, kwargs):
    """Picklable entrypoint: prove isolation before calling the unchanged target.

    The original native target must remain its module's global function so that
    Python's spawn pickler can resolve it.  No credentials or environment values
    are copied into the child arguments or the shared, integer-only probe.
    """
    import multiprocessing

    if multiprocessing.get_start_method() != start_method:
        probe_state.value = -2
        raise RuntimeError("router child process context changed")
    if any(os.environ.get(name) for name in set(names) | set(_credential_names(os.environ))):
        probe_state.value = -1
        raise RuntimeError("router child credential isolation failed")
    probe_state.value = 1
    return target(*args, **kwargs)


class _RouterMultiprocessing:
    """Intercept only this native router module's Process constructor."""

    def __init__(self, native, target, names, probe_state, start_method):
        self._native = native
        self._target = target
        self._names = tuple(names)
        self._probe_state = probe_state
        self._start_method = start_method

    def __getattr__(self, name):
        return getattr(self._native, name)

    def Process(self, *, target, args=(), kwargs=None, **options):  # noqa: N802
        if target is not self._target:
            raise _DiagnosticContractRejected("native router process target changed")
        if self._native.get_start_method() != self._start_method:
            raise _DiagnosticContractRejected("native router process context changed")
        return self._native.Process(
            target=_router_child_entry,
            args=(
                target,
                self._names,
                self._probe_state,
                self._start_method,
                args,
                {} if kwargs is None else kwargs,
            ),
            **options,
        )


def _ray_environment(plan, cfg, native, *, diagnostic=False):
    env = _string_environment(native["skyrl.train.utils.utils"].prepare_runtime_environment(cfg))
    scrubbed = _credential_names(os.environ, env) if diagnostic else ()
    for name in scrubbed:
        env[name] = ""
    log = _prepare_infra_log(plan)
    env["SKYRL_LOG_FILE"] = str(log)
    env["PYTHONPATH"] = (
        str(Path(__file__).resolve().parents[1]) + ":" + os.environ.get("PYTHONPATH", "")
    )
    return env, log, scrubbed


def _diagnostic_runtime_files(plan):
    files = _runtime()
    files.update(
        {p + "/__init__.py": "" for p in ("training", "evals", "evals/fleet", "cyber_post_train")}
    )
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return files


def _diagnostic_output_evidence(plan, *, started):
    """Prove the immutable bundle and absence of task/training artifacts."""
    root = Path(plan["output_root"])
    if not root.is_dir() or root.is_symlink():
        raise ValueError("engine diagnostic output root is not a regular directory")
    allowed = {".runtime"}
    if started:
        allowed.update({"ENGINE_DIAGNOSTIC_STARTED.json", "private-native-logs"})
    if {path.name for path in root.iterdir()} != allowed:
        raise ValueError("engine diagnostic output contains an unexpected artifact")

    runtime = root / ".runtime"
    if runtime.is_symlink() or not runtime.is_dir():
        raise ValueError("engine diagnostic runtime bundle is not a regular directory")
    expected = _diagnostic_runtime_files(plan)
    actual = {}
    actual_directories = set()
    for path in runtime.rglob("*"):
        if path.is_symlink() or (not path.is_dir() and not path.is_file()):
            raise ValueError("engine diagnostic runtime bundle contains an unsafe entry")
        if path.is_file():
            actual[str(path.relative_to(runtime))] = path
        else:
            actual_directories.add(str(path.relative_to(runtime)))
    expected_directories = {
        str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."
    }
    if (
        set(actual) != set(expected)
        or actual_directories != expected_directories
        or any(actual[name].read_text() != value for name, value in expected.items())
    ):
        raise ValueError("engine diagnostic runtime bundle changed")
    if digest({name: actual[name].read_text() for name in RUNTIME_FILES}) != plan.get(
        "runtime_sha256"
    ):
        raise ValueError("engine diagnostic runtime sources changed")

    if started:
        value = json.loads((root / "ENGINE_DIAGNOSTIC_STARTED.json").read_bytes())
        sealed(value, ENGINE_DIAGNOSTIC_SCHEMA)
        if value.get("status") != "started" or value.get("plan_sha256") != digest(plan):
            raise ValueError("engine diagnostic start receipt changed")
        logs = root / "private-native-logs"
        if logs.is_symlink() or not logs.is_dir() or (logs.stat().st_mode & 0o777) != 0o700:
            raise ValueError("engine diagnostic private log directory changed")
        for path in logs.rglob("*"):
            if path.is_symlink() or (not path.is_dir() and not path.is_file()):
                raise ValueError("engine diagnostic private logs contain an unsafe entry")
            if path.is_dir() or (
                path.name != "infra.log"
                and re.fullmatch(r"router-\d{6}_\d{6}\.log", path.name) is None
            ):
                raise ValueError("engine diagnostic private logs contain an unexpected entry")
        infra = logs / "infra.log"
        if not infra.is_file() or (infra.stat().st_mode & 0o777) != 0o600:
            raise ValueError("engine diagnostic infrastructure log changed")
    return {
        "runtime_files_unchanged": True,
        "unexpected_output_artifacts": 0,
        "checkpoint_artifacts": 0,
        "episode_artifacts": 0,
        "task_artifacts": 0,
    }


def _add_identity(values, value):
    if value is not None and all(existing is not value for existing in values):
        values.append(value)


@dataclasses.dataclass
class _DiagnosticOwnership:
    groups: list = dataclasses.field(default_factory=list)
    routers: list = dataclasses.field(default_factory=list)
    actors: list = dataclasses.field(default_factory=list)
    engine_actors: list = dataclasses.field(default_factory=list)
    placement_groups: list = dataclasses.field(default_factory=list)
    job_id: str | None = None
    ray_gpu_nodes_discovered: int | None = None
    ray_gpu_nodes_probed: int = 0
    ray_actor_environment_probes_passed: int | None = None
    ray_actor_environment_probe_failures: int | None = None
    ray_actor_nonempty_scrubbed_credentials: int | None = None
    router_start_attempts: int = 0
    router_credential_probes_passed: int = 0
    router_credential_probe_failures: int = 0
    router_multiprocessing_start_method: str | None = None

    def absorb_setup(self, setup):
        if setup is None:
            return
        try:
            _add_identity(self.routers, setup.router)
        except HardDeadlineExceeded:
            raise
        except Exception:
            pass
        try:
            for group in setup.server_groups:
                _add_identity(self.groups, group)
        except HardDeadlineExceeded:
            raise
        except Exception:
            pass

    def absorb_group_resources(self):
        for group in tuple(self.groups):
            try:
                for actor in group.get_actors():
                    _add_identity(self.actors, actor)
                    _add_identity(self.engine_actors, actor)
            except HardDeadlineExceeded:
                raise
            except Exception:
                pass
            for name in ("_internal_pg", "_external_pg"):
                try:
                    value = getattr(group, name)
                    _add_identity(self.placement_groups, getattr(value, "pg", value))
                except HardDeadlineExceeded:
                    raise
                except Exception:
                    pass


class _TrackedActorClass:
    def __init__(self, actor_class, ownership):
        self._actor_class = actor_class
        self._ownership = ownership

    def remote(self, *args, **kwargs):
        actor = self._actor_class.remote(*args, **kwargs)
        _add_identity(self._ownership.actors, actor)
        _add_identity(self._ownership.engine_actors, actor)
        return actor

    def options(self, *args, **kwargs):
        return type(self)(self._actor_class.options(*args, **kwargs), self._ownership)

    def __getattr__(self, name):
        return getattr(self._actor_class, name)


@contextmanager
def _instrument_engine_ownership(ownership, environment, scrubbed):
    """Track exact native objects even when setup loses its partial locals."""
    setup_module = importlib.import_module("skyrl.backends.skyrl_train.inference_servers.setup")
    group_module = importlib.import_module(
        "skyrl.backends.skyrl_train.inference_servers.server_group"
    )
    router_module = importlib.import_module(
        "skyrl.backends.skyrl_train.inference_servers.vllm_router"
    )
    patches = []

    def patch(module, name, value):
        patches.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    original_group = group_module.ServerGroup
    original_router = router_module.VLLMRouter
    original_engine_env = group_module.build_engine_runtime_env
    original_router_target = router_module._run_router_with_logging
    original_multiprocessing = router_module.multiprocessing
    start_method = original_multiprocessing.get_start_method()
    ownership.router_multiprocessing_start_method = start_method
    # A pre-existing forkserver can inherit an older, unsanitized environment.
    # Neither changing the framework's context nor trusting that server is safe.
    if start_method not in {"fork", "spawn"}:
        raise _DiagnosticContractRejected("unsupported native router process context")
    router_probe_state = original_multiprocessing.RawValue("b", 0)

    class TrackedServerGroup(original_group):
        def __init__(self, *args, **kwargs):
            _add_identity(ownership.groups, self)
            super().__init__(*args, **kwargs)

        def _create_actor_class(self, *args, **kwargs):
            actor_class = super()._create_actor_class(*args, **kwargs)
            return _TrackedActorClass(actor_class, ownership)

    class TrackedRouter(original_router):
        def __init__(self, *args, **kwargs):
            _add_identity(ownership.routers, self)
            super().__init__(*args, **kwargs)

        def start(self, *args, **kwargs):
            ownership.router_start_attempts += 1
            router_probe_state.value = 0
            try:
                with _scrubbed_process_environment(scrubbed):
                    return super().start(*args, **kwargs)
            finally:
                if router_probe_state.value == 1:
                    ownership.router_credential_probes_passed += 1
                elif router_probe_state.value in {-1, -2}:
                    ownership.router_credential_probe_failures += 1

    def tracked_placement_group(original):
        def create(*args, **kwargs):
            value = original(*args, **kwargs)
            _add_identity(ownership.placement_groups, value)
            return value

        return create

    def scrubbed_engine_environment(*args, **kwargs):
        value = original_engine_env(*args, **kwargs) or {}
        if not isinstance(value, Mapping):
            raise ValueError("native engine runtime environment changed")
        value = dict(value)
        value["env_vars"] = _scrubbed_actor_environment(
            environment,
            scrubbed,
            value.get("env_vars", {}),
        )
        return value

    try:
        patch(group_module, "ServerGroup", TrackedServerGroup)
        patch(router_module, "VLLMRouter", TrackedRouter)
        patch(
            router_module,
            "multiprocessing",
            _RouterMultiprocessing(
                original_multiprocessing,
                original_router_target,
                scrubbed,
                router_probe_state,
                start_method,
            ),
        )
        # Patch any import-time aliases too.  The pinned implementation reads
        # these module globals while ``create_inference_servers`` is running.
        if getattr(setup_module, "ServerGroup", None) is original_group:
            patch(setup_module, "ServerGroup", TrackedServerGroup)
        if getattr(setup_module, "VLLMRouter", None) is original_router:
            patch(setup_module, "VLLMRouter", TrackedRouter)
        if getattr(group_module, "VLLMRouter", None) is original_router:
            patch(group_module, "VLLMRouter", TrackedRouter)
        patch(
            setup_module,
            "ray_placement_group",
            tracked_placement_group(setup_module.ray_placement_group),
        )
        patch(
            group_module,
            "placement_group",
            tracked_placement_group(group_module.placement_group),
        )
        patch(group_module, "build_engine_runtime_env", scrubbed_engine_environment)
        yield setup_module
    finally:
        for module, name, value in reversed(patches):
            setattr(module, name, value)


@contextmanager
def _bounded_ray_get(ray, deadline):
    """Give every native bare ``ray.get`` the current hard-deadline remainder."""
    original = ray.get

    def bounded(refs, *, timeout=None, **kwargs):
        remaining = deadline.remaining()
        plan_limited = timeout is None or remaining <= float(timeout)
        timeout = remaining if timeout is None else min(remaining, float(timeout))
        try:
            return original(refs, timeout=timeout, **kwargs)
        except BaseException as exc:
            if plan_limited and (
                isinstance(exc, TimeoutError)
                or type(exc).__name__ in {"GetTimeoutError", "RayTimeoutError"}
            ):
                deadline.raise_expired()
            raise

    ray.get = bounded
    try:
        yield
    finally:
        ray.get = original


class _CredentialProbe:
    def inspect(self, names):
        import ray

        return {
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "nonempty": sum(bool(os.environ.get(name)) for name in names),
        }

    def shutdown(self):
        return None


def _live_gpu_node_ids(ray, expected_nodes, expected_gpus_per_node):
    """Return the exact sorted live GPU node IDs or fail on topology uncertainty."""
    rows = ray.nodes()
    if not isinstance(rows, (list, tuple)):
        raise ValueError("Ray node table changed")
    node_ids = []
    for row in rows:
        if not isinstance(row, Mapping) or type(row.get("Alive")) is not bool:
            raise ValueError("Ray node record changed")
        if not row["Alive"]:
            continue
        resources = row.get("Resources")
        if not isinstance(resources, Mapping):
            raise ValueError("Ray live-node resources changed")
        gpus = resources.get("GPU", 0)
        if (
            isinstance(gpus, bool)
            or not isinstance(gpus, numbers.Real)
            or not math.isfinite(gpus)
            or gpus < 0
        ):
            raise ValueError("Ray live-node GPU resources changed")
        if gpus == 0:
            continue
        node_id = row.get("NodeID")
        if (
            not isinstance(node_id, str)
            or re.fullmatch(r"[0-9a-fA-F]+", node_id) is None
            or len(node_id) % 2
            or gpus != expected_gpus_per_node
        ):
            raise ValueError("Ray GPU node topology differs from diagnostic request")
        node_ids.append(node_id.lower())
    if len(node_ids) != expected_nodes or len(set(node_ids)) != len(node_ids):
        raise ValueError("Ray GPU node topology differs from diagnostic request")
    return tuple(sorted(node_ids))


def _node_affinity_strategy(ray, node_id):
    strategies = getattr(getattr(ray, "util", None), "scheduling_strategies", None)
    value = getattr(strategies, "NodeAffinitySchedulingStrategy", None)
    if value is None:
        value = importlib.import_module(
            "ray.util.scheduling_strategies"
        ).NodeAffinitySchedulingStrategy
    return value(node_id=node_id, soft=False)


def _probe_worker_credentials(
    ray,
    ownership,
    environment,
    scrubbed,
    *,
    expected_nodes,
    expected_gpus_per_node,
):
    node_ids = _live_gpu_node_ids(ray, expected_nodes, expected_gpus_per_node)
    ownership.ray_gpu_nodes_discovered = len(node_ids)
    actor_environment = _scrubbed_actor_environment(environment, scrubbed)
    actors = []
    for node_id in node_ids:
        actor = (
            ray.remote(_CredentialProbe)
            .options(
                num_cpus=0,
                num_gpus=0,
                max_restarts=0,
                runtime_env={"env_vars": dict(actor_environment)},
                scheduling_strategy=_node_affinity_strategy(ray, node_id),
            )
            .remote()
        )
        _add_identity(ownership.actors, actor)
        actors.append(actor)
    results = ray.get([actor.inspect.remote(scrubbed) for actor in actors])
    if not isinstance(results, (list, tuple)) or len(results) != len(node_ids):
        raise ValueError("Ray credential probe result changed")
    normalized = []
    for result in results:
        if (
            not isinstance(result, Mapping)
            or set(result) != {"node_id", "nonempty"}
            or not isinstance(result["node_id"], str)
            or re.fullmatch(r"[0-9a-fA-F]+", result["node_id"]) is None
            or len(result["node_id"]) % 2
            or type(result["nonempty"]) is not int
            or not 0 <= result["nonempty"] <= len(scrubbed)
        ):
            raise ValueError("Ray credential probe result changed")
        normalized.append((result["node_id"].lower(), result["nonempty"]))
    ownership.ray_gpu_nodes_probed = len(normalized)
    ownership.ray_actor_nonempty_scrubbed_credentials = sum(count for _, count in normalized)
    ownership.ray_actor_environment_probes_passed = sum(count == 0 for _, count in normalized)
    ownership.ray_actor_environment_probe_failures = sum(count != 0 for _, count in normalized)
    if tuple(sorted(node_id for node_id, _ in normalized)) != node_ids:
        raise ValueError("Ray credential probes did not cover the exact GPU nodes")
    if _live_gpu_node_ids(ray, expected_nodes, expected_gpus_per_node) != node_ids:
        raise ValueError("Ray GPU node topology changed during credential probes")
    if ownership.ray_actor_nonempty_scrubbed_credentials != 0:
        raise ValueError("Ray actor retained a scrubbed credential value")
    return 0


def _validate_complete_engine_setup(setup, engine_config):
    expected_groups = engine_config.num_engines
    expected_servers = expected_groups * engine_config.data_parallel_size
    groups = tuple(setup.server_groups)
    urls = tuple(setup.server_urls)
    actor_counts = tuple(len(tuple(group.get_actors())) for group in groups)
    if (
        type(expected_groups) is not int
        or type(expected_servers) is not int
        or expected_groups < 1
        or expected_servers < 1
        or len(groups) != expected_groups
        or len(urls) != expected_servers
        or actor_counts != (engine_config.data_parallel_size,) * expected_groups
        or any(not isinstance(url, str) or not url for url in urls)
        or setup.router is None
        or not isinstance(setup.proxy_url, str)
        or not setup.proxy_url
    ):
        raise ValueError("native engine setup returned a partial topology")


def _active_owned_resources(job_id, timeout):
    """Read exact current-job ownership without Ray's dashboard server.

    Ray 2.56's public state list APIs route through the optional dashboard at
    port 8265, which the pinned training cluster does not run.  Its internal
    actor table and raw GCS placement-group table are dashboard-independent.
    The caller's process alarm bounds both raw calls because they expose no
    per-request timeout; an error or deadline remains fatal uncertainty.
    Treat every state except the one terminal state as active so a new or
    unfamiliar state can never be mistaken for successful cleanup.
    """
    job_id = _ray_job_id_hex(job_id)
    if timeout <= 0:
        raise ValueError("invalid Ray cleanup ownership query")
    ray_state = importlib.import_module("ray._private.state")
    actors = ray_state.actors()
    if not isinstance(actors, Mapping):
        raise ValueError("Ray actor ownership table changed")
    active_actors = 0
    for value in actors.values():
        if not isinstance(value, Mapping) or "JobID" not in value or "State" not in value:
            raise ValueError("Ray actor ownership record changed")
        if _ray_job_id_hex(value["JobID"]) == job_id and value["State"] != "DEAD":
            active_actors += 1

    gcs_pb2 = importlib.import_module("ray.core.generated.gcs_pb2")
    binary_to_hex = importlib.import_module("ray._common.utils").binary_to_hex
    accessor = ray_state.state._connect_and_get_accessor()
    payloads = accessor.get_placement_group_table()
    if not isinstance(payloads, (list, tuple)):
        raise ValueError("Ray placement-group ownership table changed")
    active_groups = 0
    for payload in payloads:
        row = gcs_pb2.PlacementGroupTableData.FromString(payload)
        creator_job_id = _ray_job_id_hex(binary_to_hex(row.creator_job_id))
        if creator_job_id != job_id:
            continue
        try:
            state_name = gcs_pb2.PlacementGroupTableData.PlacementGroupState.Name(row.state)
        except (TypeError, ValueError) as exc:
            raise ValueError("Ray placement-group state changed") from exc
        if state_name != "REMOVED":
            active_groups += 1
    return active_actors, active_groups


def _ray_job_id_hex(value):
    """Normalize Ray's JobID object without trusting its display string.

    Ray 2.56 renders ``str(JobID)`` as ``JobID(<hex>)``.  The raw ``hex()``
    method is the stable identity used by the actor and placement-group tables.
    """
    if isinstance(value, str):
        result = value
    else:
        method = getattr(value, "hex", None)
        if not callable(method):
            raise ValueError("Ray job identity changed")
        result = method()
    if (
        not isinstance(result, str)
        or re.fullmatch(r"[0-9a-fA-F]+", result) is None
        or len(result) % 2
    ):
        raise ValueError("Ray job identity changed")
    return result.lower()


def _prove_owned_resources_released(ownership, deadline):
    if not ownership.job_id:
        raise ValueError("Ray job identity unavailable for cleanup proof")
    while True:
        actors, groups = _active_owned_resources(ownership.job_id, deadline.remaining())
        if actors == groups == 0:
            return {"active_owned_actors": 0, "active_owned_placement_groups": 0}
        time.sleep(min(0.1, deadline.remaining() / 2))


def _router_released(router, shutdown_returned):
    marker = object()
    process = getattr(router, "_process", marker)
    if process is marker:
        return shutdown_returned
    if process is not None and process.is_alive():
        return False
    return all(
        getattr(router, name, None) is None
        for name in ("_port_reservation", "_prometheus_port_reservation")
    )


def _cleanup_engine_diagnostic(setup, ray, ownership, timeout_seconds):
    """Bound, force, and independently prove all diagnostic-owned cleanup."""
    with (
        hard_deadline(timeout_seconds, "SkyRL engine cleanup") as deadline,
        _bounded_ray_get(ray, deadline),
    ):
        ownership.absorb_setup(setup)
        ownership.absorb_group_resources()
        router_results = {}
        for router in ownership.routers:
            try:
                router.shutdown()
            except HardDeadlineExceeded:
                raise
            except Exception:
                router_results[id(router)] = False
            else:
                router_results[id(router)] = True

        shutdown_refs = []
        graceful = True
        for actor in ownership.actors:
            try:
                shutdown_refs.append(actor.shutdown.remote())
            except HardDeadlineExceeded:
                raise
            except Exception:
                graceful = False
        if shutdown_refs:
            try:
                ray.get(
                    shutdown_refs,
                    timeout=min(30.0, max(0.1, deadline.remaining() / 3)),
                )
            except HardDeadlineExceeded:
                raise
            except Exception:
                graceful = False
        for actor in ownership.actors:
            deadline.remaining()
            try:
                ray.kill(actor, no_restart=True)
            except HardDeadlineExceeded:
                raise
            except Exception:
                pass

        remove_placement_group = importlib.import_module(
            "ray.util.placement_group"
        ).remove_placement_group
        for group in ownership.placement_groups:
            deadline.remaining()
            try:
                remove_placement_group(group)
            except HardDeadlineExceeded:
                raise
            except Exception:
                pass

        if ownership.job_id:
            proof = _prove_owned_resources_released(ownership, deadline)
        elif any(
            (ownership.actors, ownership.groups, ownership.placement_groups, ownership.routers)
        ):
            raise ValueError("Ray job identity unavailable for cleanup proof")
        else:
            proof = {"active_owned_actors": 0, "active_owned_placement_groups": 0}
        if any(
            not _router_released(router, router_results.get(id(router), False))
            for router in ownership.routers
        ):
            raise ValueError("engine diagnostic router cleanup could not be proven")
        ray.shutdown()
        deadline.remaining()
    return {
        **proof,
        "tracked_ray_actors": len(ownership.actors),
        "tracked_engine_actors": len(ownership.engine_actors),
        "tracked_placement_groups": len(ownership.placement_groups),
        "tracked_routers": len(ownership.routers),
        "graceful_actor_shutdown": graceful,
        "cleanup_proven": True,
    }


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

    env, _, _ = _ray_environment(plan, cfg, modules)
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


def engine_diagnostic(plan):
    """Start and stop only the exact vLLM engines; never load task rows or train."""
    root = Path(plan["output_root"])
    if os.environ.get("RUN_DIR") != str(root):
        raise ValueError("Jobs API output binding mismatch")
    args = skyrl.SkyRLConfig(**plan["arguments"])
    ownership = _DiagnosticOwnership()
    setup = None
    ray = None
    log = None
    scrubbed = ()
    shape = None
    create_called = False
    ray_initialization_attempted = False
    startup_error = None
    startup_deadline = None
    startup_phase = "plan_validation"
    ray_actor_environment_isolation_proven = False
    try:
        # One wall-clock budget covers all work done after GPU allocation,
        # including the exact 27B model re-hash.  Cleanup owns a separate alarm
        # only after this context has restored the process signal state.
        with hard_deadline(
            args.engine_start_timeout_seconds, "SkyRL engine startup"
        ) as startup_deadline:
            job_request(plan)
            startup_phase = "runtime_validation"
            _diagnostic_output_evidence(plan, started=False)
            startup_phase = "input_validation"
            check_inputs(plan)
            startup_phase = "native_validation"
            modules = native_source()
            cfg = skyrl.diagnostic_native_config(args)
            shape = _diagnostic_shape(plan, cfg)
            _write(
                root / "ENGINE_DIAGNOSTIC_STARTED.json",
                {
                    "schema": ENGINE_DIAGNOSTIC_SCHEMA,
                    "status": "started",
                    "plan_sha256": digest(plan),
                    "optimizer_steps": 0,
                    "rollouts": 0,
                    "verifier_calls": 0,
                    "checkpoints_created": 0,
                    "engine_start_timeout_seconds": args.engine_start_timeout_seconds,
                    "engine_cleanup_timeout_seconds": args.engine_cleanup_timeout_seconds,
                    **shape,
                },
            )
            startup_phase = "ray_environment"
            import ray as ray_module
            from skyrl.backends.skyrl_train.inference_servers.utils import (
                build_vllm_cli_args,
            )

            ray = ray_module
            env, log, scrubbed = _ray_environment(plan, cfg, modules, diagnostic=True)
            startup_phase = "ownership_instrumentation"
            with (
                _instrument_engine_ownership(ownership, env, scrubbed) as setup_module,
                _bounded_ray_get(ray, startup_deadline),
            ):
                startup_phase = "ray_initialization"
                ray_initialization_attempted = True
                ray.init(address="auto", log_to_driver=False, runtime_env={"env_vars": env})
                ownership.job_id = _ray_job_id_hex(ray.get_runtime_context().get_job_id())
                startup_phase = "ray_actor_environment_probe"
                _probe_worker_credentials(
                    ray,
                    ownership,
                    env,
                    scrubbed,
                    expected_nodes=shape["diagnostic_workers"],
                    expected_gpus_per_node=shape["diagnostic_gpus_per_worker"],
                )
                ray_actor_environment_isolation_proven = True
                startup_phase = "engine_argument_build"
                vllm_args = build_vllm_cli_args(cfg)
                engine_config = cfg.generator.inference_engine
                engine_log_path = cfg.trainer.log_path
                create_called = True
                startup_phase = "engine_creation"
                setup = setup_module.create_inference_servers(
                    engine_config,
                    vllm_args,
                    log_path=engine_log_path,
                )
                startup_phase = "engine_validation"
                _validate_complete_engine_setup(setup, engine_config)
                if (
                    ownership.router_start_attempts != 1
                    or ownership.router_credential_probes_passed != 1
                    or ownership.router_credential_probe_failures != 0
                ):
                    raise ValueError("router child credential isolation was not proven")
                startup_deadline.remaining()
                startup_phase = "complete"
    except BaseException as exc:
        startup_error = exc

    ownership.absorb_setup(setup)
    if startup_error is None:
        engine_start_state, engine_started = "all", True
        status = "passed"
    elif setup is not None or ownership.groups or ownership.engine_actors or ownership.routers:
        engine_start_state, engine_started = "partial_or_unknown", None
        status = None
    else:
        engine_start_state, engine_started = "none", False
        status = None

    timed_out = bool(startup_deadline is not None and startup_deadline.expired)
    router_isolation_proven = ownership.router_start_attempts == 0 or (
        ownership.router_start_attempts == ownership.router_credential_probes_passed
        and ownership.router_credential_probe_failures == 0
    )
    # Only a typed, recognized contract rejection before any Ray initialization
    # is a clean pre-Ray outcome.  Import errors and arbitrary ValueErrors still
    # fail, and any ambiguous allocation continues to require exact cleanup.
    pre_ray_contract_rejection = (
        isinstance(startup_error, _DiagnosticContractRejected)
        and startup_phase == "ownership_instrumentation"
        and not ray_initialization_attempted
        and not any(
            (
                ownership.job_id,
                ownership.groups,
                ownership.routers,
                ownership.actors,
                ownership.engine_actors,
                ownership.placement_groups,
            )
        )
    )
    terminal_evidence_candidate = pre_ray_contract_rejection or (
        isinstance(startup_error, Exception) and bool(ownership.job_id)
    )
    clean_candidate = pre_ray_contract_rejection or (
        terminal_evidence_candidate
        and ray_actor_environment_isolation_proven
        and router_isolation_proven
    )
    if terminal_evidence_candidate:
        if pre_ray_contract_rejection:
            status = "diagnostic_contract_rejected"
        elif timed_out:
            status = "engine_start_timeout"
        elif not ray_actor_environment_isolation_proven or not router_isolation_proven:
            status = "environment_isolation_rejected"
        elif create_called:
            status = "engine_start_rejected"
        else:
            status = "pre_engine_rejected"

    if ray is not None:
        try:
            cleanup = _cleanup_engine_diagnostic(
                setup,
                ray,
                ownership,
                args.engine_cleanup_timeout_seconds,
            )
        except BaseException as cleanup_error:
            if startup_error is not None:
                raise BaseExceptionGroup(
                    "engine diagnostic startup and cleanup failed",
                    [startup_error, cleanup_error],
                ) from None
            raise
    elif startup_error is None:
        raise RuntimeError("engine diagnostic did not initialize Ray")

    if startup_error is not None and not terminal_evidence_candidate:
        raise startup_error

    from .rl_runtime import sanitized_causes

    output = _diagnostic_output_evidence(plan, started=True)
    if log is None:
        raise RuntimeError("engine diagnostic infrastructure log unavailable")
    if shape is None:
        raise RuntimeError("engine diagnostic cluster shape unavailable")
    router_isolation = (
        "proven"
        if ownership.router_start_attempts == ownership.router_credential_probes_passed == 1
        and ownership.router_credential_probe_failures == 0
        else "not_started"
        if ownership.router_start_attempts == 0
        else "failed_or_unknown"
    )
    result = {
        "schema": ENGINE_DIAGNOSTIC_SCHEMA,
        "status": status,
        "engine_start_state": engine_start_state,
        "engine_started": engine_started,
        "startup_phase": startup_phase,
        "causes": [] if startup_error is None else sanitized_causes(startup_error),
    }
    result.update(
        {
            "plan_sha256": digest(plan),
            "private_log": _infra_log_evidence(log),
            "task_rows_read": 0,
            "verifier_calls": 0,
            "optimizer_steps": 0,
            "rollouts": 0,
            "checkpoints_created": 0,
            "checkpoint_created": False,
            "wandb_initialized": False,
            "diagnostic_completed": True,
            "engine_start_qualified": result["status"] == "passed",
            "training_qualified": False,
            "production_training_shape_qualified": False,
            "engine_start_timeout_seconds": args.engine_start_timeout_seconds,
            "engine_cleanup_timeout_seconds": args.engine_cleanup_timeout_seconds,
            "credential_variables_scrubbed": len(scrubbed),
            "credential_environment_isolation_proven": (
                ray_actor_environment_isolation_proven and router_isolation_proven
            ),
            "ray_actor_environment_isolation_proven": (ray_actor_environment_isolation_proven),
            "ray_initialization_attempted": ray_initialization_attempted,
            "ray_gpu_nodes_expected": shape["diagnostic_workers"],
            "ray_gpu_nodes_discovered": ownership.ray_gpu_nodes_discovered,
            "ray_gpu_nodes_probed": ownership.ray_gpu_nodes_probed,
            "ray_actor_environment_probes_passed": (ownership.ray_actor_environment_probes_passed),
            "ray_actor_environment_probe_failures": (
                ownership.ray_actor_environment_probe_failures
            ),
            "ray_actor_nonempty_scrubbed_credentials": (
                ownership.ray_actor_nonempty_scrubbed_credentials
            ),
            "router_child_credential_environment_isolation": router_isolation,
            "router_start_attempts": ownership.router_start_attempts,
            "router_multiprocessing_start_method": ownership.router_multiprocessing_start_method,
            "router_environment_probes_passed": ownership.router_credential_probes_passed,
            "router_environment_probe_failures": ownership.router_credential_probe_failures,
            "registry_actors_created": 0,
            "service_account_token_isolation_proven": False,
            "service_account_rbac_write_access_tested": False,
            **shape,
            "cleanup": cleanup,
            "output_postconditions": output,
            "completed_at": time.time(),
        }
    )
    result = _write(root / "ENGINE_DIAGNOSTIC.json", result)
    if startup_error is not None and not clean_candidate:
        raise startup_error
    return result


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--native", action="store_true")
    mode.add_argument("--engine-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_bytes())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        if args.engine_diagnostic:
            result = engine_diagnostic(plan)
            print(json.dumps({k: result[k] for k in ("status", "sha256")}))
        else:
            job_request(plan)
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
