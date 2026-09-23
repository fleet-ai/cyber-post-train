"""Append-only one-GPU reload adapter for exact Fast3 checkpoints.

This module reuses the frozen prod9 loader and launch-neutral request shape,
but owns the plan/schema boundary so a Fast3 manifest is never represented as
a prod9 plan. It performs no provider call or workload creation.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from types import FunctionType
from typing import Any

from cyber_post_train.jobs import bundled_request, digest, validate_request

from . import checkpoints, sft_runtime, skyrl_fast3_posttrain, skyrl_fast3_training
from . import skyrl_prod9_reload as historical

SPEC_SCHEMA = "cyber_skyrl_fast3_reload_spec_v1"
MODULE = "training.skyrl_fast3_reload"
IMAGE = historical.IMAGE
MAXIMUM_SECONDS = historical.MAXIMUM_SECONDS
HARD_CHILD_SECONDS = historical.HARD_CHILD_SECONDS
RUNTIME_FILES = skyrl_fast3_training.RUNTIME_FILES


def _runtime_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {name: historical._file_sha256(root / name) for name in RUNTIME_FILES}


def _adapt(function, **bindings):
    namespace = {
        **function.__globals__,
        "SPEC_SCHEMA": SPEC_SCHEMA,
        "MODULE": MODULE,
        "IMAGE": IMAGE,
        "MAXIMUM_SECONDS": MAXIMUM_SECONDS,
        "HARD_CHILD_SECONDS": HARD_CHILD_SECONDS,
        "RUNTIME_FILES": RUNTIME_FILES,
        "_runtime_hashes": _runtime_hashes,
        "__file__": __file__,
        **bindings,
    }
    adapted = FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    adapted.__kwdefaults__ = function.__kwdefaults__
    return adapted


_SPEC_IDENTITY = _adapt(historical._spec_identity)


def _spec_identity(spec: object) -> dict[str, Any]:
    return _SPEC_IDENTITY(spec)


def build_spec(
    plan: dict[str, Any],
    checkpoint_manifest: dict[str, Any],
    export_receipt: dict[str, Any],
    *,
    checkpoint_manifest_file_sha256: str,
    export_file_sha256: str,
) -> dict[str, Any]:
    """Bind an exact Fast3 plan, checkpoint manifest, and BF16 export."""
    if plan.get("schema") != skyrl_fast3_training.SCHEMA:
        raise ValueError("Fast3 reload requires the exact training schema")
    args = skyrl_fast3_training._validated(plan)
    skyrl_fast3_posttrain.verify_manifest(checkpoint_manifest, check_files=False)
    export = historical._receipt(export_receipt)
    checkpoint_file = historical._hex(checkpoint_manifest_file_sha256, prefix=False)
    export_file = historical._hex(export_file_sha256, prefix=False)
    expected_name = plan["run_name"] + f"-p{args.steps}-reload-v1"
    expected_reload_root = historical._reload_run_dir(plan, args.steps)
    expected_export_root = plan["output_root"] + f"/hf-export-step{args.steps}-v1"
    expected_manifest_path = plan["output_root"] + f"/checkpoint-seals-v1/step-{args.steps}.json"
    expected_checkpoint_path = plan["output_root"] + f"/checkpoints/global_step_{args.steps}"
    if (
        checkpoint_manifest.get("source_plan") != plan
        or checkpoint_manifest.get("source_plan_sha256") != digest(plan)
        or checkpoint_manifest.get("checkpoint_path") != expected_checkpoint_path
        or checkpoint_manifest.get("optimizer_step") != args.steps
        or export.get("schema") != skyrl_fast3_posttrain.EXPORT_SCHEMA
        or export.get("source_checkpoint_receipt_sha256")
        != checkpoint_manifest.get("receipt_sha256")
        or export.get("source_manifest_file_sha256") != checkpoint_file
        or export.get("source_plan_sha256") != digest(plan)
        or export.get("model_repo") != plan["model"]["repo"]
        or export.get("model_revision") != plan["model"]["revision"]
        or export.get("output_root") != expected_export_root
        or export.get("optimizer_step") != args.steps
        or export.get("optimizer_steps_executed") != 0
        or export.get("gpu_reload_verified") is not False
        or export.get("dtype") != "BF16"
        or export.get("all_output_tensors_reopened_equal") is not True
        or export.get("source_inventory_sizes_mtimes_unchanged") is not True
    ):
        raise ValueError("Fast3 reload checkpoint/export identity changed")
    return historical._seal(
        {
            "schema": SPEC_SCHEMA,
            "plan_sha256": digest(plan),
            "name": expected_name,
            "run_dir": expected_reload_root,
            "image": IMAGE,
            "model": {
                "repo": plan["model"]["repo"],
                "revision": plan["model"]["revision"],
                "base_root": plan["model"]["root"],
                "export_root": expected_export_root,
            },
            "checkpoint": {
                "manifest_path": expected_manifest_path,
                "manifest_file_sha256": checkpoint_file,
                "receipt_sha256": checkpoint_manifest["receipt_sha256"],
                "checkpoint_path": expected_checkpoint_path,
                "optimizer_step": args.steps,
            },
            "export": {
                "receipt_path": expected_export_root + "/EXPORT.json",
                "file_sha256": export_file,
                "receipt_sha256": export["receipt_sha256"],
                "source_checkpoint_receipt_sha256": export["source_checkpoint_receipt_sha256"],
                "source_manifest_file_sha256": export["source_manifest_file_sha256"],
                "source_plan_sha256": export["source_plan_sha256"],
                "optimizer_step": args.steps,
            },
            "runtime": {
                "module": MODULE,
                "files_sha256": _runtime_hashes(),
                "hard_child_seconds": HARD_CHILD_SECONDS,
            },
            "resources": {
                "nodes": 1,
                "gpus": 1,
                "priority_class": "c1",
                "queue_priority_class": "q1",
                "maximum_seconds": MAXIMUM_SECONDS,
            },
            "scientific_work": {
                "optimizer_steps": 0,
                "synthetic_only": True,
                "serving_qualified": False,
            },
        }
    )


def job_request(spec: dict[str, Any]) -> dict[str, Any]:
    """Build the exact one-GPU reload bundle without contacting a provider."""
    value = _spec_identity(spec)
    root = Path(__file__).resolve().parents[1]
    files = {name: (root / name).read_text() for name in RUNTIME_FILES}
    files["training/__init__.py"] = ""
    files["spec.json"] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    source_name = (
        "q38-rld-" + value["plan_sha256"][:8] + "-" + str(value["checkpoint"]["optimizer_step"])
    )
    request = bundled_request(
        {
            "name": source_name,
            "title": "Qwen3.8 Fast3 exact BF16 one-GPU reload " + value["plan_sha256"][:12],
            "run_dir": value["run_dir"],
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
                "PYTHONPATH": value["run_dir"] + "/.runtime:/opt/skyrl",
            },
        },
        files,
        MODULE,
        ["--spec", "spec.json", "--sha256", value["sha256"]],
    )
    claim = value["run_dir"] + "/.fast3-reload-create-claim-v1"
    request["command"] = "mkdir " + shlex.quote(claim) + " && exec " + request["command"]
    validate_request(request)
    return request


_INSPECT_EXPORT = _adapt(historical._inspect_export, _spec_identity=_spec_identity)


def _inspect_export(spec: dict[str, Any]):
    return _INSPECT_EXPORT(spec)


def validate_source(spec: dict[str, Any]):
    """Reopen one exact Fast3 checkpoint receipt and BF16 export."""
    value = _spec_identity(spec)
    checkpoint = value["checkpoint"]
    manifest_path = Path(checkpoint["manifest_path"])
    sft_runtime._checked_file(manifest_path, checkpoint["manifest_file_sha256"])
    manifest = checkpoints.receipt(manifest_path)
    skyrl_fast3_posttrain.verify_manifest(manifest, check_files=False)
    source_plan = manifest.get("source_plan")
    source_model = source_plan.get("model") if isinstance(source_plan, dict) else None
    if (
        manifest.get("schema") != skyrl_fast3_posttrain.MANIFEST_SCHEMA
        or manifest.get("receipt_sha256") != checkpoint["receipt_sha256"]
        or not isinstance(source_plan, dict)
        or digest(source_plan) != value["plan_sha256"]
        or manifest.get("source_plan_sha256") != value["plan_sha256"]
        or source_plan.get("schema") != skyrl_fast3_training.SCHEMA
        or not isinstance(source_model, dict)
        or source_model.get("repo") != value["model"]["repo"]
        or source_model.get("revision") != value["model"]["revision"]
        or source_model.get("root") != value["model"]["base_root"]
        or source_plan.get("run_name") + f"-p{checkpoint['optimizer_step']}-reload-v1"
        != value["name"]
        or manifest.get("checkpoint_path") != checkpoint["checkpoint_path"]
        or manifest.get("optimizer_step") != checkpoint["optimizer_step"]
        or manifest.get("optimizer_update_verified") is not True
        or manifest.get("source_inputs_unchanged") is not True
    ):
        raise ValueError("Fast3 reload checkpoint identity changed")
    return _inspect_export(value)


_RUN_CHECK = _adapt(
    historical.run_check,
    _spec_identity=_spec_identity,
    validate_source=validate_source,
    _inspect_export=_inspect_export,
)
_RUN_PARENT = _adapt(historical.run_parent, _spec_identity=_spec_identity)


def run_check(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    return _RUN_CHECK(spec, output)


def run_parent(spec: dict[str, Any]) -> None:
    _RUN_PARENT(spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    value = _spec_identity(json.loads(args.spec.read_bytes()))
    if args.sha256 != value["sha256"]:
        raise ValueError("Fast3 reload runtime specification digest changed")
    if args.child:
        run_check(value, Path(value["run_dir"]) / "GPU_CHECK.json")
    else:
        run_parent(value)


if __name__ == "__main__":
    main()
