"""One exact FTI 0.10.27 Qwen3.8 HF-to-Miles conversion.

This is a format conversion only.  It owns one eight-GPU node because the
maintained Miles converter needs eight ranks to partition the model while it
loads.  It never creates an optimizer or performs a training step.  The
project-owned prepared-model tree is assembled later, on CPU, by
``miles96_model_stage``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cyber_post_train.jobs import bundled_request, digest, quantity
from training import miles_conversion as base

SCHEMA = "cyber_qwen38_miles96_phase2_conversion_v1"
RUN_NAME = "chris-q38-m96-p2-convert-v1"
OUTPUT_ROOT = "/mnt/sfs/jobs/chris-q38-m96-p2-convert-v1"
MODEL_ROOT = "/mnt/sfs/models/qwen3.8-27b-1d4bf0f2"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "5bdf98161971959295b24efa54d0ad91445e6d6e409d9f1384cb886992dbb064"
)
FTI_VERSION = "0.10.27"
FTI_SOURCE_COMMIT = "d23116f018cb9213f0a3ee6c228d213abcd85d80"
FTI_RUN_FLEET_SHA256 = "ae82e3f03d14c81e52009bc2cbff857d9e844baf07a9d82d8266608cb8cba2b1"
MODEL_BINDING_SHA256 = "dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
DEADLINE_SECONDS = base.DEADLINE_SECONDS
RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "64",
    "memory_request": "512Gi",
    "memory_limit": "768Gi",
}


def compile_plan(*, root: Path | None = None) -> dict:
    """Bind the exact local manifests and fixed SFS source without I/O there."""
    from training.models import bound_model
    from training.sft import read_mapping

    root = root or Path(__file__).resolve().parents[1]
    model = bound_model(
        read_mapping(root / "configs/models/qwen38-27b-1d4bf0f2.lock.json"),
        read_mapping(root / "configs/models/qwen38-27b-1d4bf0f2.weights.json"),
        MODEL_ROOT,
    )
    plan = {
        "schema": SCHEMA,
        "run_name": RUN_NAME,
        "output_root": OUTPUT_ROOT,
        "model": model,
        "runtime_sha256": base._hash(Path(__file__)),
        "support_runtime_sha256": base._hash(Path(base.__file__)),
        "native_converter_sha256": base.CONVERTER_SHA256,
        "fti_version": FTI_VERSION,
        "fti_source_commit": FTI_SOURCE_COMMIT,
        "fti_run_fleet_sha256": FTI_RUN_FLEET_SHA256,
        "optimizer_steps": 0,
        "deadline_seconds": DEADLINE_SECONDS,
        "execution": {
            "image": IMAGE,
            "priority": "c1",
            "queue_priority": "q1",
            "resources": RESOURCES,
        },
    }
    job_request(plan)
    return plan


def _validate_plan(plan: dict) -> None:
    if any(
        (
            plan.get("schema") != SCHEMA,
            plan.get("run_name") != RUN_NAME,
            plan.get("output_root") != OUTPUT_ROOT,
            plan.get("runtime_sha256") != base._hash(Path(__file__)),
            plan.get("support_runtime_sha256") != base._hash(Path(base.__file__)),
            plan.get("native_converter_sha256") != base.CONVERTER_SHA256,
            plan.get("fti_version") != FTI_VERSION,
            plan.get("fti_source_commit") != FTI_SOURCE_COMMIT,
            plan.get("fti_run_fleet_sha256") != FTI_RUN_FLEET_SHA256,
            plan.get("optimizer_steps") != 0,
            plan.get("deadline_seconds") != DEADLINE_SECONDS,
            plan.get("execution", {}).get("image") != IMAGE,
            plan.get("execution", {}).get("priority") != "c1",
            plan.get("execution", {}).get("queue_priority") != "q1",
            plan.get("execution", {}).get("resources") != RESOURCES,
            plan.get("model", {}).get("root") != MODEL_ROOT,
            plan.get("model", {}).get("repo") != "Qwen/Qwen3.8-27B",
            plan.get("model", {}).get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            digest(plan.get("model")) != MODEL_BINDING_SHA256,
        )
    ):
        raise ValueError("Miles96 phase-2 conversion plan drift")


def job_request(plan: dict) -> dict:
    _validate_plan(plan)
    resources = plan["execution"]["resources"]
    if quantity(resources["cpu_request"]) < 64 or quantity(resources["memory_request"]) < quantity(
        "512Gi"
    ):
        raise ValueError("conversion loading envelope drift")
    root = Path(__file__).resolve().parents[1]
    files = {
        name: (root / name).read_text()
        for name in (
            "training/miles96_phase2_conversion.py",
            "training/miles_conversion.py",
            "training/miles.py",
            "cyber_post_train/jobs.py",
        )
    }
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    return bundled_request(
        {
            "name": RUN_NAME,
            "title": RUN_NAME + " FTI 0.10.27 zero-step conversion",
            "run_dir": OUTPUT_ROOT,
            "image": IMAGE,
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": resources,
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "WANDB_MODE": "disabled",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        "training.miles96_phase2_conversion",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )


def validate_runtime(plan: dict, *, check_model: bool = True) -> list[str]:
    import fti
    from fti.trainers.miles import run_fleet

    _validate_plan(plan)
    if (
        fti.__version__ != FTI_VERSION
        or base._hash(Path(run_fleet.__file__)) != FTI_RUN_FLEET_SHA256
    ):
        raise ValueError("FTI 0.10.27 source identity drift")
    if check_model:
        base.check_inputs(plan)
    return base.native_arguments(plan)


def preflight(plan: dict) -> dict:
    import torch
    from transformers import AutoConfig

    if torch.cuda.is_available():
        raise ValueError("conversion preflight must not allocate GPUs")
    if Path(OUTPUT_ROOT).exists():
        raise FileExistsError("conversion output already exists")
    argv = validate_runtime(plan)
    config = AutoConfig.from_pretrained(
        MODEL_ROOT, trust_remote_code=False, local_files_only=True
    ).get_text_config()
    if (config.num_hidden_layers, config.hidden_size, config.vocab_size) != (
        64,
        5120,
        248320,
    ):
        raise ValueError("staged architecture differs from Qwen3.8-27B")
    return {
        "schema": "cyber_qwen38_miles96_phase2_conversion_preflight_v1",
        "status": "passed",
        "gpus": 0,
        "plan_sha256": digest(plan),
        "request_sha256": digest(job_request(plan)),
        "native_argv_sha256": digest(argv),
        "model_files": len(plan["model"]["files"]),
        "gpu_conversion_verified": False,
    }


def run(plan: dict) -> dict:
    # ``base.run`` performs the one full source hash while the GPUs are owned.
    # Do not make that expensive I/O pass twice.
    validate_runtime(plan, check_model=False)
    return base.run(plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    ray = None
    task = None
    try:
        plan = json.loads(args.plan.read_text())
        if digest(plan) != args.sha256:
            raise ValueError("plan digest mismatch")
        _validate_plan(plan)
        import ray

        ray.init(
            address="auto",
            log_to_driver=False,
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1])
                    + ":"
                    + os.environ.get("PYTHONPATH", ""),
                }
            },
        )
        task = ray.remote(num_cpus=1, num_gpus=8)(run).remote(plan)
        result = ray.get(task, timeout=DEADLINE_SECONDS + 120)
        print(json.dumps({k: result[k] for k in ("status", "optimizer_steps", "sha256")}))
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    finally:
        if ray is not None and ray.is_initialized():
            if task is not None:
                ray.cancel(task, force=True)
            ray.shutdown()


if __name__ == "__main__":
    main()
