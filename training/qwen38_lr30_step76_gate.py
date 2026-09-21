"""Bounded zero-update GPU forward for the accepted Qwen3.8 LR30 export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .checkpoints import receipt
from .export_check import check
from .sft_runtime import _checked_file, write_receipt

EXPORT = Path("/mnt/sfs/jobs/chris-q38-lr30-resume76-prod-v1/hf-export-step76-v1/EXPORT.json")
EXPORT_FILE_SHA256 = "39cedfb644aa955b450e42eb98494276c1bdce85a9760b2b038b67823ff7b9e5"
EXPORT_RECEIPT_SHA256 = "983bf6daa2f8b7680560c834cbfb76fb8fe7fba5a273f116b9b3e80ef9f97538"
SOURCE_MANIFEST_FILE_SHA256 = "cc4e2997e120a80bb1e209dc027f18e12ce1ebe12639594d28e23b08c459cf1a"
SOURCE_CHECKPOINT_RECEIPT_SHA256 = (
    "1252a1a9d5556fa118f03d1b2fe5687e0e9505ccf622e2f86743a26b77dffa64"
)
CHECKER_SHA256 = "a04811409178eedc6969e34766ca70c82d84c27b718944ecab93407609c4dfe7"
HARD_CHILD_SECONDS = 1200
RUN_NAME = "chris-q38-lr30-s76-gpu-v1"
RUN_DIR = "/mnt/sfs/jobs/chris-q38-lr30-step76-hfcheck-v1"
IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:"
    "ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
RUNTIME_FILES = (
    "training/qwen38_lr30_step76_gate.py",
    "training/checkpoints.py",
    "training/export_check.py",
    "training/io.py",
    "training/post_sft_artifacts.py",
    "training/sft_runtime.py",
)


def job_request() -> dict:
    """Return the immutable one-GPU request; this does not call any API."""
    from cyber_post_train.jobs import bundled_request

    root = Path(__file__).resolve().parents[1]
    files = {name: (root / name).read_text() for name in RUNTIME_FILES}
    files["training/__init__.py"] = ""
    return bundled_request(
        {
            "name": RUN_NAME,
            "title": "Qwen3.8 LR30 step76 exact BF16 one-GPU zero-update reload v1",
            "run_dir": RUN_DIR,
            "image": IMAGE,
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": {
                "cpu_request": "8",
                "cpu_limit": "8",
                "memory_request": "64Gi",
                "memory_limit": "128Gi",
            },
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": [],
            "image_pull_secrets": [],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": RUN_DIR + "/.runtime:/opt/skyrl",
            },
        },
        files,
        "training.qwen38_lr30_step76_gate",
        [],
    )


def validate_source() -> dict:
    _checked_file(EXPORT, EXPORT_FILE_SHA256)
    proof = receipt(EXPORT)
    if (
        proof.get("receipt_sha256") != EXPORT_RECEIPT_SHA256
        or proof.get("source_manifest_file_sha256") != SOURCE_MANIFEST_FILE_SHA256
        or proof.get("source_checkpoint_receipt_sha256") != SOURCE_CHECKPOINT_RECEIPT_SHA256
        or proof.get("optimizer_step") != 76
        or proof.get("optimizer_steps_executed") != 0
        or proof.get("source_inventory_sizes_mtimes_unchanged") is not True
        or proof.get("output_root") != str(EXPORT.parent)
    ):
        raise ValueError("LR30 export differs from the accepted binding")
    return proof


def validate_gpu_receipt(value: dict) -> None:
    if (
        value.get("checker_sha256") != CHECKER_SHA256
        or value.get("export_receipt_sha256") != EXPORT_RECEIPT_SHA256
        or value.get("finite_logits") is not True
        or value.get("generated_tokens") != 2
        or value.get("gpu_reload_verified") is not True
        or value.get("source_unchanged") is not True
        or value.get("optimizer_steps_executed") != 0
        or value.get("serving_qualified") is not False
    ):
        raise ValueError("LR30 GPU receipt differs from the zero-update forward gate")


def run_child(output: Path) -> None:
    value = check(EXPORT, EXPORT_FILE_SHA256, output / "GPU_CHECK.json", gpu=True)
    validate_gpu_receipt(value)


def run_parent(output: Path) -> None:
    write_receipt(
        output / "STARTED.json",
        {
            "schema": "cyber_qwen38_lr30_step76_gpu_check_start_v1",
            "started_at": time.time(),
            "export_file_sha256": EXPORT_FILE_SHA256,
            "export_receipt_sha256": EXPORT_RECEIPT_SHA256,
            "source_manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
            "gpus": 1,
            "optimizer_steps_executed": 0,
            "hard_child_seconds": HARD_CHILD_SECONDS,
        },
    )
    try:
        with (output / "private-synthetic-loader.log").open("x") as log:
            os.chmod(log.name, 0o600)
            subprocess.run(
                [sys.executable, "-m", "training.qwen38_lr30_step76_gate", "--child"],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=HARD_CHILD_SECONDS,
                check=True,
            )
        value = receipt(output / "GPU_CHECK.json")
        validate_gpu_receipt(value)
        write_receipt(
            output / "COMPLETE.json",
            {
                "schema": "cyber_qwen38_lr30_step76_gpu_check_complete_v1",
                "completed_at": time.time(),
                "gpu_check_receipt_sha256": value["receipt_sha256"],
                "export_file_sha256": EXPORT_FILE_SHA256,
                "export_receipt_sha256": EXPORT_RECEIPT_SHA256,
                "gpus": 1,
                "optimizer_steps_executed": 0,
                "serving_qualified": False,
            },
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "gpu_check_receipt_sha256": value["receipt_sha256"],
                    "gpus": 1,
                    "optimizer_steps_executed": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException as exc:
        write_receipt(
            output / "FAILED.json",
            {
                "schema": "cyber_qwen38_lr30_step76_gpu_check_failure_v1",
                "error_class": type(exc).__name__,
                "gpus": 1,
                "optimizer_steps_executed": 0,
            },
        )
        print(
            json.dumps({"status": "failed", "error_class": type(exc).__name__}, sort_keys=True),
            flush=True,
        )
        raise


def main() -> None:
    validate_source()
    output = Path(os.environ["RUN_DIR"])
    if "--child" in sys.argv:
        run_child(output)
    else:
        run_parent(output)


if __name__ == "__main__":
    main()
