"""Create-once zero-update Qwen3.8 Megatron-LoRA merge/export lane.

The qualified SkyRL image exposes native TP8 checkpoint load and Hugging Face
export calls, but both use non-strict state-dict handling.  This producer closes
that boundary with the already accepted checkpoint receipt: every live adapter
shard and frozen-base rank digest is reconciled after load, then two independent
exports must contain the same complete BF16 tensor values and exact base layout.

This module never trains, evaluates, downloads, or selects a checkpoint.  Its
only model input is an accepted ``QWEN38_LORA_CHECKPOINT.json`` and the exact
base/checkpoint paths already bound by that receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any

from cyber_post_train.jobs import bundled_request, digest, quantity

from .qwen38_lora_artifacts import (
    EXPORT_SCHEMA,
    validate_checkpoint_receipt,
    validate_export_receipt,
)
from .sft_runtime import QWEN38_MEGATRON_IMAGE, QWEN38_MEGATRON_SOURCE_SHA256

PLAN_SCHEMA = "cyber_qwen38_megatron_lora_zero_step_export_plan_v1"
CHECKPOINT_FILENAME = "QWEN38_LORA_CHECKPOINT.json"
RECEIPT_FILENAME = "QWEN38_LORA_MERGED_HF_EXPORT.json"
MERGE_METHOD = "megatron_bridge_lora_merge_v1"
DEADLINE_SECONDS = 1800
SHA256 = re.compile(r"^[a-f0-9]{64}$")
RUN_NAME = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?$")
RESOURCES = {
    "cpu_request": "64",
    "cpu_limit": "64",
    "memory_request": "512Gi",
    "memory_limit": "768Gi",
}
NATIVE_API = {
    "checkpoint_load": "WorkerDispatch.load_checkpoint",
    "adapter_census": "WorkerDispatch.collect_lora_qualification_snapshots",
    "merged_hf_export": "WorkerDispatch.save_hf_model",
    "model_reload": "transformers.AutoModelForCausalLM.from_pretrained",
    "tokenizer_reload": "transformers.AutoTokenizer.from_pretrained",
}
CHECKPOINT_IDENTITY_FIELDS = {
    "source_plan_sha256",
    "checkpoint_path",
    "optimizer_step",
    "model_repository",
    "base_model_revision",
    "checkpoint_inventory_sha256",
    "target_census_sha256",
    "adapter_parameter_inventory_sha256",
    "base_model_inventory_sha256",
}
_FAILURE_STAGE = "entry"


def _stage(name: str) -> None:
    global _FAILURE_STAGE
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name):
        raise ValueError("invalid sanitized export stage")
    _FAILURE_STAGE = name


def _hash(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or indirect file")
    before = path.stat()
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
        if getattr(before, field) != getattr(after, field):
            raise ValueError("file changed while it was hashed")
    return value


def _canonical_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/") or "//" in value:
        raise ValueError(f"{label} must be one canonical absolute path")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(f"{label} must be one canonical absolute path")
    return Path(value)


def _code_sha256() -> dict[str, str]:
    root = Path(__file__).parent
    return {
        name: _hash(root / name)
        for name in ("qwen38_lora_export.py", "qwen38_lora_artifacts.py", "sft_runtime.py", "io.py")
    }


def _read_checkpoint(path: Path, expected_file_sha256: str) -> tuple[dict, dict]:
    if path.name != CHECKPOINT_FILENAME or not SHA256.fullmatch(expected_file_sha256):
        raise ValueError("checkpoint input must be one exact QWEN38_LORA_CHECKPOINT.json")
    if _hash(path) != expected_file_sha256:
        raise ValueError("checkpoint receipt file digest mismatch")
    value = json.loads(path.read_text())
    return value, validate_checkpoint_receipt(value)


def seal_plan(
    checkpoint_receipt: Path,
    *,
    checkpoint_file_sha256: str,
    run_name: str,
    run_dir: str,
) -> dict:
    """Bind a runnable plan only from one accepted checkpoint receipt."""
    receipt, identity = _read_checkpoint(checkpoint_receipt, checkpoint_file_sha256)
    root = _canonical_path(run_dir, "run directory")
    if root.parts[:4] != ("/", "mnt", "sfs", "jobs") or len(root.parts) != 5:
        raise ValueError("run directory must be one direct child of /mnt/sfs/jobs")
    if not RUN_NAME.fullmatch(run_name) or root.name != run_name:
        raise ValueError("run name and create-once run directory must agree")
    source_root = Path(receipt["source_plan"]["output_root"])
    if root == source_root or root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError("export run and source training run must not overlap")
    plan = {
        "schema": PLAN_SCHEMA,
        "run_name": run_name,
        "run_dir": str(root),
        "output_root": str(root / "merged-hf"),
        "checkpoint_receipt": {
            "path": str(checkpoint_receipt),
            "file_sha256": checkpoint_file_sha256,
            "receipt_sha256": identity["receipt_sha256"],
        },
        "checkpoint_identity": {
            key: identity[key]
            for key in (
                "source_plan_sha256",
                "checkpoint_path",
                "optimizer_step",
                "model_repository",
                "base_model_revision",
                "checkpoint_inventory_sha256",
                "target_census_sha256",
                "adapter_parameter_inventory_sha256",
                "base_model_inventory_sha256",
            )
        },
        "image": QWEN38_MEGATRON_IMAGE,
        "source_files_sha256": dict(QWEN38_MEGATRON_SOURCE_SHA256),
        "code_sha256": _code_sha256(),
        "native_api": dict(NATIVE_API),
        "optimizer_steps_executed": 0,
        "external_evaluation": False,
        "create_once": True,
        "priority_class": "c1",
        "resources": dict(RESOURCES),
        "deadline_seconds": DEADLINE_SECONDS,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict, *, check_code: bool = True) -> dict:
    expected = {
        "schema",
        "run_name",
        "run_dir",
        "output_root",
        "checkpoint_receipt",
        "checkpoint_identity",
        "image",
        "source_files_sha256",
        "code_sha256",
        "native_api",
        "optimizer_steps_executed",
        "external_evaluation",
        "create_once",
        "priority_class",
        "resources",
        "deadline_seconds",
    }
    if not isinstance(plan, dict) or set(plan) != expected or plan["schema"] != PLAN_SCHEMA:
        raise ValueError("zero-step export plan has unknown or missing fields")
    root = _canonical_path(plan["run_dir"], "run directory")
    output = _canonical_path(plan["output_root"], "output root")
    reference = plan["checkpoint_receipt"]
    identity = plan["checkpoint_identity"]
    if (
        not RUN_NAME.fullmatch(plan["run_name"])
        or root.name != plan["run_name"]
        or output != root / "merged-hf"
        or not isinstance(reference, dict)
        or set(reference) != {"path", "file_sha256", "receipt_sha256"}
        or _canonical_path(reference["path"], "checkpoint receipt").name != CHECKPOINT_FILENAME
        or not SHA256.fullmatch(reference["file_sha256"])
        or not SHA256.fullmatch(reference["receipt_sha256"])
        or not isinstance(identity, dict)
        or set(identity) != CHECKPOINT_IDENTITY_FIELDS
        or identity.get("optimizer_step") != 1
        or plan["image"] != QWEN38_MEGATRON_IMAGE
        or plan["source_files_sha256"] != QWEN38_MEGATRON_SOURCE_SHA256
        or plan["native_api"] != NATIVE_API
        or plan["optimizer_steps_executed"] != 0
        or plan["external_evaluation"] is not False
        or plan["create_once"] is not True
        or plan["priority_class"] != "c1"
        or plan["resources"] != RESOURCES
        or plan["deadline_seconds"] != DEADLINE_SECONDS
    ):
        raise ValueError("zero-step export plan differs from the reviewed exact-image lane")
    if check_code and plan["code_sha256"] != _code_sha256():
        raise ValueError("zero-step export producer bytes changed after plan sealing")
    return plan


def job_request(plan: dict) -> dict:
    validate_plan(plan)
    root = Path(__file__).resolve().parents[1]
    names = (
        "training/qwen38_lora_export.py",
        "training/qwen38_lora_artifacts.py",
        "training/sft_runtime.py",
        "training/io.py",
        "cyber_post_train/jobs.py",
    )
    files = {name: (root / name).read_text() for name in names}
    files.update({"training/__init__.py": "", "cyber_post_train/__init__.py": ""})
    files["plan.json"] = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    request = bundled_request(
        {
            "name": plan["run_name"],
            "title": plan["run_name"] + " Qwen3.8 LoRA zero-step merge/export",
            "run_dir": plan["run_dir"],
            "image": plan["image"],
            "workers": 1,
            "gpus_per_worker": 8,
            "resources": plan["resources"],
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "secrets": [],
            "image_pull_secrets": ["ghcr-pull"],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "WANDB_MODE": "disabled",
                "FLA_TILELANG": "0",
                "SKYRL_PYTHONPATH_EXPORT": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": str(Path(plan["run_dir"]) / ".runtime") + ":/opt/skyrl",
            },
        },
        files,
        "training.qwen38_lora_export",
        ["--plan", "plan.json", "--sha256", digest(plan)],
    )
    if quantity(request["resources"]["memory_request"]) < quantity("512Gi"):
        raise ValueError("Qwen3.8 export lost its reviewed memory envelope")
    return request


def _expected_snapshot_rows(receipt: dict) -> dict[int, dict[str, str]]:
    expected = {rank: {} for rank in range(8)}
    for name, row in receipt["adapter_parameters"].items():
        for shard in row["rank_shards"]:
            expected[shard["tp_rank"]][name] = shard["checkpoint_sha256"]
    return expected


def verify_reloaded_snapshots(receipt: dict, snapshots: list[dict]) -> dict:
    """Close the native strict=False gap using the accepted rank census."""
    if not isinstance(snapshots, list) or len(snapshots) != 8:
        raise ValueError("checkpoint reload did not return every TP8 rank")
    expected_adapters = _expected_snapshot_rows(receipt)
    trainable = {
        row["tp_rank"]: row["after_sha256"]
        for row in receipt["trainable_parameter_census"]["rank_manifests"]
    }
    frozen = {
        row["tp_rank"]: row["after_sha256"] for row in receipt["frozen_base"]["rank_manifests"]
    }
    seen = set()
    for snapshot in snapshots:
        rank = snapshot.get("rank", {}).get("tp_rank")
        adapters = snapshot.get("adapters")
        if type(rank) is not int or rank in seen or rank not in expected_adapters:
            raise ValueError("checkpoint reload returned an invalid or duplicate TP rank")
        observed = (
            {name: row.get("sha256") for name, row in adapters.items()}
            if isinstance(adapters, dict)
            else None
        )
        if observed != expected_adapters[rank]:
            raise ValueError("live adapter values differ from the accepted TP8 checkpoint census")
        if (
            snapshot.get("trainable", {}).get("manifest_sha256") != trainable[rank]
            or snapshot.get("frozen_base", {}).get("manifest_sha256") != frozen[rank]
            or snapshot.get("successful_optimizer_updates") != 0
            or snapshot.get("last_gradient_norm") is not None
        ):
            raise ValueError("reload changed base/trainable state or executed an optimizer update")
        seen.add(rank)
    if seen != set(range(8)):
        raise ValueError("checkpoint reload did not cover TP8")
    return {"tp_ranks": sorted(seen), "optimizer_steps_executed": 0, "strict_census": True}


def _inventory_bound_files(root: Path, rows: list[dict]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        name = row["path"]
        path = root / name
        expected = str(row["sha256"]).removeprefix("sha256:")
        actual = _hash(path)
        if actual != expected:
            raise ValueError("bound source file differs from its accepted receipt")
        result[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return result


def _checkpoint_inventory(receipt: dict) -> dict[str, dict[str, Any]]:
    root = Path(receipt["checkpoint_path"])
    rows = []
    for name, expected in receipt["files"].items():
        path = root / name
        actual = _hash(path)
        if actual != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("checkpoint payload differs from its accepted receipt")
        rows.append((name, {"bytes": expected["bytes"], "sha256": actual}))
    return dict(rows)


def _runtime_source_inventory() -> dict[str, dict[str, Any]]:
    spec = importlib.util.find_spec("skyrl")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("pinned SkyRL package is unavailable")
    root = Path(next(iter(spec.submodule_search_locations))).parent
    rows = {}
    for name, expected in sorted(QWEN38_MEGATRON_SOURCE_SHA256.items()):
        path = root / name
        actual = _hash(path)
        if actual != expected:
            raise ValueError("installed SkyRL source differs from the exact image binding")
        rows[name] = {"bytes": path.stat().st_size, "sha256": actual}
    return rows


def _tensor_sha256(tensor) -> str:
    value = tensor.detach().contiguous().view(-1).view(__import__("torch").uint8).cpu()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def inspect_hf_export(root: Path, *, allowed_files: set[str] | None = None) -> dict:
    """Independently reopen every indexed tensor and every flat payload file."""
    from safetensors import safe_open

    if root.is_symlink() or not root.is_dir():
        raise ValueError("HF export root is missing or indirect")
    if allowed_files is None:
        paths = sorted(path for path in root.rglob("*") if path.is_file())
    else:
        if not allowed_files or any(
            PurePosixPath(name).is_absolute()
            or len(PurePosixPath(name).parts) != 1
            or str(PurePosixPath(name)) != name
            for name in allowed_files
        ):
            raise ValueError("bound base inference surface contains an unsafe path")
        # The staged model root can contain unrelated cache metadata. Read only
        # the exact accepted inference surface and never traverse that cache.
        paths = [root / name for name in sorted(allowed_files)]
    if any(path.is_symlink() or path.parent != root or path.stat().st_size <= 0 for path in paths):
        raise ValueError("HF export must be one complete, flat, nonempty directory")
    files = {path.name: {"bytes": path.stat().st_size, "sha256": _hash(path)} for path in paths}
    required = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    }
    if not required <= set(files) or any("adapter" in name.lower() for name in files):
        raise ValueError("HF export lacks required model/tokenizer files or retains an adapter")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("HF export has no tensor index")
    shards = set(weight_map.values())
    actual_shards = {path.name for path in paths if path.suffix == ".safetensors"}
    if shards != actual_shards or any(
        not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", name) for name in shards
    ):
        raise ValueError("HF export shard set differs from its tensor index")
    tensors = {}
    for shard in sorted(shards):
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if set(keys) != {name for name, mapped in weight_map.items() if mapped == shard}:
                raise ValueError("HF shard keys differ from the tensor index")
            for name in keys:
                if name in tensors:
                    raise ValueError("HF export contains a duplicate tensor")
                tensor = handle.get_tensor(name)
                if str(tensor.dtype) != "torch.bfloat16" or tensor.numel() <= 0:
                    raise ValueError("merged HF export must contain nonempty BF16 tensors only")
                tensors[name] = {
                    "dtype": "BF16",
                    "shape": list(tensor.shape),
                    "values": tensor.numel(),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "sha256": _tensor_sha256(tensor),
                }
    layout = [
        {"name": name, "dtype": row["dtype"], "shape": row["shape"]}
        for name, row in sorted(tensors.items())
    ]
    sidecars = {
        name: row["sha256"]
        for name, row in files.items()
        if not name.endswith(".safetensors") and name != "model.safetensors.index.json"
    }
    return {
        "files": files,
        "sidecars": sidecars,
        "tensors": tensors,
        "layout_sha256": digest(layout),
        "tensor_count": len(tensors),
        "tensor_values": sum(row["values"] for row in tensors.values()),
        "tensor_bytes": sum(row["bytes"] for row in tensors.values()),
    }


def _distributed_export(plan: dict, receipt: dict, first: Path, second: Path) -> dict:
    """Use the exact native TP8 APIs; no training loop or optimizer call exists here."""
    import ray

    from .sft_runtime import _make_trainer_class, build_runtime_configs

    runtime_plan = copy.deepcopy(receipt["source_plan"])
    runtime_plan["run_name"] = plan["run_name"]
    runtime_plan["output_root"] = plan["run_dir"]
    runtime_plan["plan_sha256"] = digest(runtime_plan)
    cfg, skyrl_cfg = build_runtime_configs(runtime_plan)
    megatron = skyrl_cfg.trainer.policy.megatron_config
    if (
        megatron.tensor_model_parallel_size != 8
        or megatron.pipeline_model_parallel_size != 1
        or megatron.context_parallel_size != 1
        or megatron.lora_config.merge_lora is not True
    ):
        raise ValueError("native runtime lost the exact TP8 merged-LoRA export configuration")
    skyrl_cfg.trainer.log_path = str(Path(plan["run_dir"]) / "private_logs")
    parent = _make_trainer_class()

    class ExportTrainer(parent):
        def _init_tracker(self):
            self.tracker = None

        def train_step(self, *args, **kwargs):
            raise RuntimeError("zero-step export must never enter a training step")

    trainer = ExportTrainer(cfg, skyrl_cfg, runtime_plan)
    try:
        trainer.setup()
        trainer.dispatch.load_checkpoint(
            "policy",
            str(Path(receipt["checkpoint_path"]) / "policy"),
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
        )
        actor = trainer.dispatch._actor_groups["policy"]
        learning_rates = ray.get(actor.async_run_ray_method("pass_through", "get_lr"))
        if (
            not isinstance(learning_rates, list)
            or len(learning_rates) != 8
            or any(
                type(value) not in {int, float} or not math.isfinite(value) or value <= 0
                for value in learning_rates
            )
        ):
            raise ValueError("checkpoint optimizer/scheduler reload lacks all-rank evidence")
        snapshots = trainer.dispatch.collect_lora_qualification_snapshots("policy")
        evidence = verify_reloaded_snapshots(receipt, snapshots)
        trainer.dispatch.save_hf_model("policy", str(first), trainer.tokenizer)
        trainer.dispatch.save_hf_model("policy", str(second), trainer.tokenizer)
        return {**evidence, "optimizer_resume_verified": True}
    finally:
        trainer.shutdown()


def _reload_model(root: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise ValueError("independent reload must own exactly one visible GPU")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    encoded = tokenizer("deterministic reload qualification", return_tensors="pt")
    encoded = {name: tensor.to("cuda:0") for name, tensor in encoded.items()}
    with torch.inference_mode():
        logits = model(**encoded).logits
    if logits.numel() <= 0 or not torch.isfinite(logits).all().item():
        raise ValueError("independent merged-model reload produced nonfinite logits")
    if tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False) == "":
        raise ValueError("independent tokenizer reload produced an empty decode")
    return {
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "logit_values": logits.numel(),
        "finite_logits": True,
    }


def _write_new(path: Path, value: dict) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(plan: dict) -> dict:
    """Execute one create-once producer. A failure is preserved, never retried in place."""
    import ray
    from skyrl.train.utils.utils import initialize_ray

    _stage("plan_validation")
    validate_plan(plan)
    _stage("runtime_binding")
    if os.environ.get("RUN_DIR") != plan["run_dir"]:
        raise ValueError("Jobs API run-directory binding drift")
    run_root = Path(plan["run_dir"])
    final = Path(plan["output_root"])
    second = run_root / ".determinism-second"
    _stage("destination_check")
    if final.exists() or final.is_symlink() or second.exists() or second.is_symlink():
        raise FileExistsError("create-once export destination already exists")
    _stage("checkpoint_validation")
    checkpoint_path = Path(plan["checkpoint_receipt"]["path"])
    receipt, identity = _read_checkpoint(checkpoint_path, plan["checkpoint_receipt"]["file_sha256"])
    if identity["receipt_sha256"] != plan["checkpoint_receipt"]["receipt_sha256"] or any(
        identity.get(key) != value for key, value in plan["checkpoint_identity"].items()
    ):
        raise ValueError("runtime checkpoint identity differs from the sealed plan")
    source_root = Path(receipt["source_plan"]["output_root"])
    if (
        run_root == source_root
        or run_root.is_relative_to(source_root)
        or source_root.is_relative_to(run_root)
    ):
        raise ValueError("export run and source training run overlap")
    _stage("source_input_hashes")
    base_root = Path(receipt["source_plan"]["model"]["root"])
    base_before = _inventory_bound_files(base_root, receipt["source_plan"]["model"]["files"])
    checkpoint_before = _checkpoint_inventory(receipt)
    runtime_before = _runtime_source_inventory()
    checkpoint_receipt_before = _hash(checkpoint_path)
    final.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    _, cfg = build_runtime_configs_for_ray(receipt, plan)
    _stage("ray_initialization")
    initialize_ray(cfg)
    try:
        # The coordinator task owns the eight TP actors. Returning from the task
        # releases that actor ownership before the independent one-GPU reload.
        _stage("distributed_reload_and_dual_export")
        reload_evidence = ray.get(
            ray.remote(num_cpus=1)(_distributed_export).remote(plan, receipt, final, second),
            timeout=DEADLINE_SECONDS,
        )
        _stage("first_export_reopen")
        first_inspection = inspect_hf_export(final)
        _stage("second_export_reopen")
        second_inspection = inspect_hf_export(second)
        _stage("deterministic_export_compare")
        if (
            first_inspection["tensors"] != second_inspection["tensors"]
            or first_inspection["sidecars"] != second_inspection["sidecars"]
        ):
            raise ValueError("two native merge/exports are not deterministic")
        _stage("published_export_reopen")
        reopened = inspect_hf_export(final)
        _stage("published_export_stability")
        if reopened != first_inspection:
            raise ValueError("published output changed during independent tensor reopen")
        _stage("base_export_reopen")
        base = inspect_hf_export(
            base_root,
            allowed_files={row["path"] for row in receipt["source_plan"]["model"]["files"]},
        )
        _stage("base_layout_compare")
        if base["layout_sha256"] != first_inspection["layout_sha256"]:
            raise ValueError("merged export tensor layout differs from the exact base")
        changed = sum(
            base["tensors"][name]["sha256"] != row["sha256"]
            for name, row in first_inspection["tensors"].items()
        )
        _stage("adapter_change_check")
        if changed <= 0:
            raise ValueError("merged export contains no adapter-derived tensor changes")
        _stage("second_export_cleanup")
        shutil.rmtree(second)
        _stage("full_model_tokenizer_reload")
        reload_result = ray.get(
            ray.remote(num_cpus=4, num_gpus=1)(_reload_model).remote(str(final)),
            timeout=1800,
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()
    _stage("source_immutability_recheck")
    base_after = _inventory_bound_files(base_root, receipt["source_plan"]["model"]["files"])
    checkpoint_after = _checkpoint_inventory(receipt)
    runtime_after = _runtime_source_inventory()
    _stage("source_immutability_compare")
    if (
        base_before != base_after
        or checkpoint_before != checkpoint_after
        or runtime_before != runtime_after
        or checkpoint_receipt_before != _hash(checkpoint_path)
    ):
        raise ValueError("base, checkpoint, or exact-image source changed during export")
    _stage("plan_revalidation")
    validate_plan(plan)
    value = {
        "schema": EXPORT_SCHEMA,
        "source_checkpoint_receipt_sha256": identity["receipt_sha256"],
        "source_manifest_file_sha256": plan["checkpoint_receipt"]["file_sha256"],
        "source_plan_sha256": identity["source_plan_sha256"],
        "code_sha256": plan["code_sha256"],
        "model_repo": identity["model_repository"],
        "model_revision": identity["base_model_revision"],
        "output_root": str(final),
        "optimizer_step": identity["optimizer_step"],
        "optimizer_steps_executed": 0,
        "merge_method": MERGE_METHOD,
        "dtype": "BF16",
        "adapter_parameter_inventory_sha256": identity["adapter_parameter_inventory_sha256"],
        "target_census_sha256": identity["target_census_sha256"],
        "base_model_inventory_sha256": identity["base_model_inventory_sha256"],
        "base_tensor_layout_sha256": base["layout_sha256"],
        "merged_tensor_layout_sha256": first_inspection["layout_sha256"],
        "base_tensor_count": base["tensor_count"],
        "merged_tensor_count": first_inspection["tensor_count"],
        "adapter_updated_tensor_count": changed,
        "tensor_values": first_inspection["tensor_values"],
        "tensor_bytes": first_inspection["tensor_bytes"],
        "sidecars": first_inspection["sidecars"],
        "files": first_inspection["files"],
        "source_checkpoint_unchanged": True,
        "source_base_unchanged": True,
        "adapter_checkpoint_reload_verified": reload_evidence["strict_census"],
        "optimizer_resume_verified": reload_evidence["optimizer_resume_verified"],
        "all_output_tensors_reopened_equal": True,
        "adapter_payloads_absent": True,
        "deterministic_merge": True,
        "merged_model_reload_verified": bool(reload_result["model_class"]),
        "finite_logits": reload_result["finite_logits"],
        "gpu_reload_verified": True,
    }
    signed = {**value, "receipt_sha256": digest(value)}
    _stage("receipt_validation")
    validate_export_receipt(
        signed,
        receipt,
        checkpoint_file_sha256=plan["checkpoint_receipt"]["file_sha256"],
    )
    _stage("receipt_write")
    _write_new(run_root / RECEIPT_FILENAME, signed)
    return signed


def build_runtime_configs_for_ray(receipt: dict, plan: dict):
    """Render the source config solely to initialize the pinned Ray topology."""
    from .sft_runtime import build_runtime_configs

    runtime_plan = copy.deepcopy(receipt["source_plan"])
    runtime_plan["run_name"] = plan["run_name"]
    runtime_plan["output_root"] = plan["run_dir"]
    return build_runtime_configs(runtime_plan)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if digest(plan) != args.sha256:
        raise ValueError("zero-step export plan digest mismatch")
    started = time.monotonic()
    try:
        result = run(plan)
        print(json.dumps({"status": "accepted", "receipt_sha256": result["receipt_sha256"]}))
    except BaseException as exc:
        root = Path(plan.get("run_dir", os.environ.get("RUN_DIR", ".")))
        failure = root / "FAILED.json"
        if root.is_dir() and not failure.exists():
            _write_new(
                failure,
                {
                    "status": "failed",
                    "error_class": type(exc).__name__,
                    "failure_stage": _FAILURE_STAGE,
                    "plan_sha256": digest(plan),
                    "optimizer_steps_executed": 0,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
