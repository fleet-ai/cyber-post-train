"""Seal native SkyRL checkpoints for an exact, separately authorized handoff.

This CPU-only operation hashes files and checks saved step/sampler metadata. It
does not load a model, run an optimizer, certify GPU reload, or alter the source.
Only use checkpoints produced by the bound trusted training run: native PyTorch
checkpoint metadata is pickle, not an untrusted interchange format.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .sft_runtime import _unsigned_digest, digest, validate_plan, write_receipt


def receipt(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("symlink receipt")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("receipt_sha256") != _unsigned_digest(
        {k: v for k, v in value.items() if k != "receipt_sha256"}
    ):
        raise ValueError("receipt digest mismatch")
    return value


def checkpoint_files(root: Path, world_size: int) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint root must be a real directory")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("checkpoint contains a symlink")
        if path.is_file():
            files[str(path.relative_to(root))] = path
    _validate_names(set(files), world_size)
    config = json.loads(files["policy/fsdp_config.json"].read_text())
    # The pinned implementation uses FSDP2, but its serialized strategy enum is
    # "fsdp". This is checked against the native producer in CPU tests.
    if config != {"fsdp_strategy": "fsdp", "world_size": world_size}:
        raise ValueError("checkpoint topology differs from the plan")
    if any(path.stat().st_size <= 0 for path in files.values()):
        raise ValueError("empty checkpoint payload")
    return dict(sorted(files.items()))


def _validate_names(names: set[str], world_size: int) -> None:
    required = {"data.pt", "trainer_state.pt", "policy/fsdp_config.json"}
    required |= {
        f"policy/{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(world_size)
    }
    if (
        names - required != {name for name in names if name.startswith("policy/huggingface/")}
        or not required <= names
    ):
        raise ValueError("incomplete or unexpected native checkpoint layout")
    if "policy/huggingface/config.json" not in names:
        raise ValueError("missing native model configuration")


def seal(plan: dict, step: int, output: Path, *, progress=None) -> dict:
    import torch

    if torch.cuda.is_available():
        raise ValueError("checkpoint sealing does not need a GPU")
    validate_plan(plan, check_files=False)
    if type(step) is not int or not 0 < step <= plan["recipe"]["max_steps"]:
        raise ValueError("checkpoint step outside the frozen plan")
    run = Path(plan["output_root"])
    root = run / "checkpoints" / f"global_step_{step}"
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("never write into the source checkpoint")
    if output.exists() or output.is_symlink():
        raise FileExistsError("checkpoint manifest already exists")
    saved = receipt(run / "checkpoint_receipts" / f"step-{step:06d}.json")
    expected = {
        "plan_sha256": _unsigned_digest(plan),
        "optimizer_step": step,
        "checkpoint_path": str(root),
    }
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint receipt differs from source plan/step/path")
    size = plan["recipe"]["nodes"] * plan["recipe"]["gpus_per_node"]
    files = checkpoint_files(root, size)
    before = {name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in files.items()}
    # Only the two small, trusted producer metadata files are deserialized.
    trainer = torch.load(files["trainer_state.pt"], map_location="cpu", weights_only=False)
    sampler = torch.load(files["data.pt"], map_location="cpu", weights_only=False)
    if trainer.get("global_step") != step:
        raise ValueError("saved trainer step differs from checkpoint receipt")
    epoch_steps = math.ceil(plan["datasets"]["train"]["rows"] / plan["recipe"]["batch_size"])
    yielded = (step - 1) % epoch_steps + 1
    if sampler.get("_num_yielded") != yielded:
        raise ValueError("saved sampler cursor differs from completed optimizer step")
    inventory = {}
    total = 0
    for name, path in files.items():
        inventory[name] = {"bytes": before[name][0], "sha256": digest(path)}
        total += before[name][0]
        if progress:
            progress(len(inventory), total)
    after = checkpoint_files(root, size)
    if set(after) != set(files) or any(
        (p.stat().st_size, p.stat().st_mtime_ns) != before[name] for name, p in after.items()
    ):
        raise ValueError("checkpoint changed while sealing")
    result = {
        "schema": "cyber_skyrl_checkpoint_manifest_v1",
        "source_plan_sha256": _unsigned_digest(plan),
        "source_plan": plan,
        "checkpoint_path": str(root),
        "optimizer_step": step,
        "world_size": size,
        "sampler_batches_in_epoch": yielded,
        "files": inventory,
        "total_bytes": total,
        "gpu_reload_verified": False,
    }
    write_receipt(output, result)
    return receipt(output)


def verify(manifest: dict, *, check_files: bool = True) -> None:
    if (
        manifest.get("receipt_sha256")
        != _unsigned_digest({k: v for k, v in manifest.items() if k != "receipt_sha256"})
        or manifest.get("schema") != "cyber_skyrl_checkpoint_manifest_v1"
    ):
        raise ValueError("checkpoint manifest digest/schema mismatch")
    plan = manifest["source_plan"]
    validate_plan(plan, check_files=False)
    step = manifest["optimizer_step"]
    size = plan["recipe"]["nodes"] * plan["recipe"]["gpus_per_node"]
    expected = Path(plan["output_root"]) / "checkpoints" / f"global_step_{step}"
    if (
        type(step) is not int
        or not 0 < step <= plan["recipe"]["max_steps"]
        or manifest["source_plan_sha256"] != _unsigned_digest(plan)
        or manifest["checkpoint_path"] != str(expected)
        or type(manifest["world_size"]) is not int
        or manifest["world_size"] != size
        or manifest["gpu_reload_verified"] is not False
        or type(manifest["sampler_batches_in_epoch"]) is not int
        or manifest["sampler_batches_in_epoch"]
        != (step - 1) % math.ceil(plan["datasets"]["train"]["rows"] / plan["recipe"]["batch_size"])
        + 1
        or manifest["total_bytes"] != sum(v["bytes"] for v in manifest["files"].values())
    ):
        raise ValueError("checkpoint manifest bindings disagree")
    for name, spec in manifest["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != name:
            raise ValueError("checkpoint file escapes source root")
        if type(spec["bytes"]) is not int or spec["bytes"] <= 0:
            raise ValueError("invalid checkpoint file size")
        if not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
            raise ValueError("invalid checkpoint file digest")
    _validate_names(set(manifest["files"]), size)
    if check_files:
        files = checkpoint_files(expected, size)
        if set(files) != set(manifest["files"]):
            raise ValueError("checkpoint inventory changed")
        for name, path in files.items():
            spec = manifest["files"][name]
            if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError("checkpoint file digest/size mismatch")
