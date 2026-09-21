"""Sanitized development control-plane helpers for the Qwen3.8 export lane."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cyber_post_train.jobs import digest

from .qwen38_lora_artifacts import (
    validate_continuation_checkpoint_receipt,
    validate_export_receipt,
)
from .qwen38_lora_export import (
    PROMOTION_CHECKPOINT_FILENAME,
    RECEIPT_FILENAME,
    _hash,
    _read_checkpoint,
    _read_continuation_manifest,
    _runtime_source_inventory,
    _write_new,
    job_request,
    seal_continuation_plan,
    seal_plan,
)

PREFLIGHT_SCHEMA = "cyber_qwen38_megatron_lora_zero_step_export_preflight_v1"
VALIDATION_SCHEMA = "cyber_qwen38_megatron_lora_zero_step_export_validation_v1"


def _claim_control_root(control: Path) -> None:
    if control.is_symlink():
        raise ValueError("control root must not be a symlink")
    if not control.exists():
        os.mkdir(control, 0o700)
        return
    stat = control.stat()
    if (
        not control.is_dir()
        or stat.st_uid != os.getuid()
        or stat.st_gid != os.getgid()
        or stat.st_mode & 0o777 != 0o700
        or any(control.iterdir())
    ):
        raise ValueError("pre-created control root must be empty and privately owned")


def seal(args: argparse.Namespace) -> dict:
    control = Path(args.control_dir)
    run_dir = Path(args.run_dir)
    if run_dir.exists():
        raise ValueError("output root must be fresh")
    if args.checkpoint_manifest:
        if not args.checkpoint_runtime_path:
            raise ValueError("continuation seal requires its exact SFS runtime manifest path")
        plan = seal_continuation_plan(
            Path(args.checkpoint_manifest),
            checkpoint_file_sha256=args.checkpoint_file_sha256,
            checkpoint_runtime_path=args.checkpoint_runtime_path,
            run_name=args.run_name,
            run_dir=args.run_dir,
        )
        reference = plan["checkpoint_manifest"]
        reference_kind = "native_continuation_manifest"
    else:
        if args.checkpoint_runtime_path:
            raise ValueError("step-one seal does not accept a continuation runtime path")
        plan = seal_plan(
            Path(args.checkpoint_receipt),
            checkpoint_file_sha256=args.checkpoint_file_sha256,
            run_name=args.run_name,
            run_dir=args.run_dir,
        )
        reference = plan["checkpoint_receipt"]
        reference_kind = "accepted_step_one_receipt"
    request = job_request(plan)
    sources = _runtime_source_inventory()
    _claim_control_root(control)
    _write_new(control / "plan.json", plan)
    _write_new(control / "request.json", request)
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "accepted",
        "run_name": plan["run_name"],
        "run_dir": plan["run_dir"],
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "checkpoint_reference_kind": reference_kind,
        "checkpoint_receipt_sha256": reference["receipt_sha256"],
        "checkpoint_file_sha256": reference["file_sha256"],
        "image": plan["image"],
        "installed_source_sha256": {name: row["sha256"] for name, row in sources.items()},
        "optimizer_steps_executed": 0,
        "external_evaluation": False,
        "priority_class": "c1",
        "gpus_per_worker": 8,
        "deadline_seconds": plan["deadline_seconds"],
        "output_root_absent": not run_dir.exists(),
        "create_once": True,
    }
    _write_new(control / "preflight.json", result)
    return result


def verify(args: argparse.Namespace) -> dict:
    control = Path(args.control_dir)
    if args.checkpoint_manifest:
        if not args.checkpoint_runtime_path:
            raise ValueError("continuation verify requires its exact SFS runtime manifest path")
        source, source_identity = _read_continuation_manifest(
            Path(args.checkpoint_manifest),
            args.checkpoint_file_sha256,
            require_canonical_name=False,
        )
        checkpoint_path = Path(args.run_dir) / PROMOTION_CHECKPOINT_FILENAME
        checkpoint_file_sha256 = _hash(checkpoint_path)
        checkpoint = json.loads(checkpoint_path.read_text())
        if _hash(checkpoint_path) != checkpoint_file_sha256:
            raise ValueError("promotion receipt changed while it was read")
        checkpoint_identity = validate_continuation_checkpoint_receipt(checkpoint)
        if (
            checkpoint["source_native_manifest"] != source
            or checkpoint["source_native_manifest_path"] != args.checkpoint_runtime_path
            or checkpoint["source_native_manifest_file_sha256"] != args.checkpoint_file_sha256
            or checkpoint_identity["source_plan_sha256"] != source_identity["source_plan_sha256"]
        ):
            raise ValueError("promotion receipt binds a different continuation checkpoint")
        reference_kind = "native_continuation_manifest"
    else:
        if args.checkpoint_runtime_path:
            raise ValueError("step-one verify does not accept a continuation runtime path")
        checkpoint, _ = _read_checkpoint(Path(args.checkpoint_receipt), args.checkpoint_file_sha256)
        checkpoint_file_sha256 = args.checkpoint_file_sha256
        reference_kind = "accepted_step_one_receipt"
    export_path = Path(args.run_dir) / RECEIPT_FILENAME
    export = json.loads(export_path.read_text())
    identity = validate_export_receipt(
        export,
        checkpoint,
        checkpoint_file_sha256=checkpoint_file_sha256,
    )
    resume_field = (
        "optimizer_scheduler_resume_verified"
        if reference_kind == "native_continuation_manifest"
        else "optimizer_resume_verified"
    )
    result = {
        "schema": VALIDATION_SCHEMA,
        "status": "accepted",
        "export_receipt_sha256": identity["receipt_sha256"],
        "export_receipt_file_sha256": _hash(export_path),
        "checkpoint_reference_kind": reference_kind,
        "checkpoint_receipt_sha256": export["source_checkpoint_receipt_sha256"],
        "checkpoint_receipt_file_sha256": checkpoint_file_sha256,
        "optimizer_steps_executed": export["optimizer_steps_executed"],
        "finite_logits": export["finite_logits"],
        "gpu_reload_verified": export["gpu_reload_verified"],
        resume_field: export[resume_field],
        "source_checkpoint_unchanged": export["source_checkpoint_unchanged"],
        "source_base_unchanged": export["source_base_unchanged"],
        "deterministic_merge": export["deterministic_merge"],
        "all_output_tensors_reopened_equal": export["all_output_tensors_reopened_equal"],
    }
    _write_new(control / "terminal-validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("seal", "verify"):
        command = subparsers.add_parser(action)
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--checkpoint-receipt")
        source.add_argument("--checkpoint-manifest")
        command.add_argument("--checkpoint-file-sha256", required=True)
        command.add_argument("--checkpoint-runtime-path")
        command.add_argument("--run-dir", required=True)
        command.add_argument("--control-dir", required=True)
        if action == "seal":
            command.add_argument("--run-name", required=True)
    args = parser.parse_args()
    result = seal(args) if args.action == "seal" else verify(args)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
