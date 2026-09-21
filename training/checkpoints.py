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
from dataclasses import asdict
from pathlib import Path

from .sft_runtime import (
    _is_qwen38_lora,
    _unsigned_digest,
    digest,
    validate_plan,
    write_receipt,
)

QWEN38_MEGATRON_SCHEMA = "cyber_qwen38_megatron_checkpoint_manifest_v1"


def receipt(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError("symlink receipt")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("receipt_sha256") != _unsigned_digest(
        {k: v for k, v in value.items() if k != "receipt_sha256"}
    ):
        raise ValueError("receipt digest mismatch")
    return value


def verify_recovery_terminal(plan: dict, step: int) -> dict | None:
    """Verify a resumed run's terminal step from canonical plan bytes.

    Prepared plan files deliberately do not embed their own digest.  The
    terminal receipt binds to the canonical digest of those bytes instead.
    Keeping this check in the checkpoint sealer prevents ad-hoc operators from
    accidentally treating a missing ``plan_sha256`` field as a run defect.
    """
    recovery = plan.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("mode") != "resume":
        return None
    source = recovery.get("checkpoint")
    if not isinstance(source, dict) or type(source.get("optimizer_step")) is not int:
        raise ValueError("recovery source step is missing")
    source_step = source["optimizer_step"]
    if plan.get("pause_after_step") != step or step <= source_step:
        raise ValueError("recovery seal must target the exact planned pause")

    run = Path(plan["output_root"])
    terminal_path = run / "TRAINING_PAUSED.json"
    terminal = receipt(terminal_path)
    expected = {
        "status": "training_paused",
        "plan_sha256": _unsigned_digest(plan),
        "optimizer_step": step,
        "planned_optimizer_steps": plan["recipe"]["max_steps"],
        "optimizer_steps_executed": step - source_step,
        "checkpoint_path": str(run / "checkpoints" / f"global_step_{step}"),
    }
    if any(terminal.get(key) != value for key, value in expected.items()):
        raise ValueError("recovery terminal receipt differs from the canonical plan")

    metrics_path = run / "metrics.jsonl"
    if metrics_path.is_symlink() or not metrics_path.is_file():
        raise ValueError("recovery metrics are missing")
    try:
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    except json.JSONDecodeError as exc:
        raise ValueError("recovery metrics are malformed") from exc
    expected_steps = list(range(source_step + 1, step + 1))
    if [row.get("train/global_step") for row in metrics] != expected_steps:
        raise ValueError("recovery metrics do not cover every resumed optimizer step")
    for row in metrics:
        loss = row.get("train/loss")
        gradient = row.get("train/grad_norm")
        learning_rate = row.get("train/lr")
        if (
            type(loss) not in (int, float)
            or type(gradient) not in (int, float)
            or type(learning_rate) not in (int, float)
            or not math.isfinite(loss)
            or not math.isfinite(gradient)
            or not math.isfinite(learning_rate)
            or gradient <= 0
            or learning_rate <= 0
            or not math.isclose(learning_rate, plan["recipe"]["lr"], rel_tol=1e-12)
        ):
            raise ValueError("recovery metrics contain an invalid optimizer step")
    return {
        "source_optimizer_step": source_step,
        "optimizer_step": step,
        "optimizer_steps_executed": step - source_step,
        "terminal_receipt_sha256": terminal["receipt_sha256"],
        "terminal_file_sha256": digest(terminal_path),
        "metrics_file_sha256": digest(metrics_path),
    }


def checkpoint_files(root: Path, world_size: int, *, adapter: bool = False) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("checkpoint root must be a real directory")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("checkpoint contains a symlink")
        if path.is_file():
            files[str(path.relative_to(root))] = path
    _validate_names(set(files), world_size, adapter=adapter)
    # The pinned implementation uses FSDP2, but its serialized strategy enum is
    # "fsdp". This is checked against the native producer in CPU tests.
    if not adapter:
        config = json.loads(files["policy/fsdp_config.json"].read_text())
        if config != {"fsdp_strategy": "fsdp", "world_size": world_size}:
            raise ValueError("checkpoint topology differs from the plan")
    if any(path.stat().st_size <= 0 for path in files.values()):
        raise ValueError("empty checkpoint payload")
    return dict(sorted(files.items()))


def qwen38_megatron_checkpoint_files(root: Path, world_size: int) -> dict[str, Path]:
    """Reopen the exact TP8 Megatron-LoRA checkpoint layout.

    This is intentionally separate from the FSDP and PEFT layouts above.  A
    Megatron distributed checkpoint is not interchangeable with either one.
    """
    if world_size != 8 or root.is_symlink() or not root.is_dir():
        raise ValueError("Qwen3.8 Megatron checkpoint requires one real TP8 root")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("checkpoint contains a symlink")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    required = {
        "data.pt",
        "trainer_state.pt",
        "policy/.metadata",
        "policy/common.pt",
        "policy/metadata.json",
        "policy/huggingface/chat_template.jinja",
        "policy/huggingface/config.json",
        "policy/huggingface/generation_config.json",
        "policy/huggingface/processor_config.json",
        "policy/huggingface/tokenizer.json",
        "policy/huggingface/tokenizer_config.json",
    }
    required |= {f"policy/__{rank}_0.distcp" for rank in range(8)}
    required |= {f"policy/adapter_tp{rank}_pp0_cp0_dp0_ep0_etp{rank}.pt" for rank in range(8)}
    if set(files) != required or any(path.stat().st_size <= 0 for path in files.values()):
        raise ValueError("incomplete or unexpected Qwen3.8 Megatron checkpoint layout")
    metadata = json.loads(files["policy/metadata.json"].read_text())
    if metadata != {
        "common_backend": "torch",
        "common_backend_version": 1,
        "sharded_backend": "torch_dist",
        "sharded_backend_version": 1,
    }:
        raise ValueError("Qwen3.8 Megatron checkpoint backend metadata drift")
    return dict(sorted(files.items()))


def _validate_names(names: set[str], world_size: int, *, adapter: bool = False) -> None:
    if adapter:
        required = {"data.pt", "trainer_state.pt"} | {
            "policy/" + name
            for name in (
                "COMPLETE.json",
                "adapter_tensors.safetensors",
                "training_state.pt",
                "peft_config.json",
            )
        }
        if names != required:
            raise ValueError("incomplete or unexpected adapter checkpoint layout")
        return
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


def _adapter_binding(plan: dict, step: int, root: Path, inventory: dict) -> None:
    from .glm_runtime import TARGETS, identity_from_plan

    inner = receipt(root / "policy/COMPLETE.json")
    per_epoch = math.ceil(plan["datasets"]["train"]["rows"] / plan["recipe"]["batch_size"])
    expected = {
        "schema": "glm53_lora_resumable_checkpoint_v1",
        "base": asdict(identity_from_plan(plan)),
        "plan_sha256": _unsigned_digest(plan),
        "world_size": plan["recipe"]["nodes"] * plan["recipe"]["gpus_per_node"],
        "optimizer_step": step,
        "next_batch": (step - 1) % per_epoch + 1,
        "epoch": (step - 1) // per_epoch,
        "frozen_base_saved": False,
        "payloads": {
            name.removeprefix("policy/"): spec
            for name, spec in inventory.items()
            if name.startswith("policy/") and name != "policy/COMPLETE.json"
        },
    }
    if any(inner.get(k) != v for k, v in expected.items()):
        raise ValueError("adapter receipt differs from bound base, step, cursor or payloads")
    peft = json.loads((root / "policy/peft_config.json").read_text())
    if (
        peft != inner.get("peft_config")
        or peft.get("r") != plan["lora"]["rank"]
        or peft.get("lora_alpha") != plan["lora"]["alpha"]
        or set(peft.get("target_modules", [])) != set(TARGETS)
    ):
        raise ValueError("adapter configuration differs from the plan")


def _seal_qwen38_megatron(plan: dict, step: int, output: Path, *, progress=None) -> dict:
    """Seal one trusted native TP8 Megatron checkpoint without loading weights."""
    import torch

    run = Path(plan["output_root"])
    root = run / "checkpoints" / f"global_step_{step}"
    saved = receipt(run / "checkpoint_receipts" / f"step-{step:06d}.json")
    expected = {
        "plan_sha256": _unsigned_digest(plan),
        "optimizer_step": step,
        "checkpoint_path": str(root),
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint receipt differs from source plan/step/path")
    files = qwen38_megatron_checkpoint_files(root, 8)
    before = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in files.items()}
    trainer = torch.load(files["trainer_state.pt"], map_location="cpu", weights_only=False)
    sampler = torch.load(files["data.pt"], map_location="cpu", weights_only=False)
    if trainer.get("global_step") != step:
        raise ValueError("saved trainer step differs from checkpoint receipt")
    epoch_steps = math.ceil(plan["datasets"]["train"]["rows"] / plan["recipe"]["batch_size"])
    yielded = (step - 1) % epoch_steps + 1
    if sampler.get("_num_yielded") != yielded:
        raise ValueError("saved sampler cursor differs from completed optimizer step")
    inventory, total = {}, 0
    for name, path in files.items():
        inventory[name] = {"bytes": before[name][0], "sha256": digest(path)}
        total += before[name][0]
        if progress:
            progress(len(inventory), total)
    after = qwen38_megatron_checkpoint_files(root, 8)
    if set(after) != set(files) or any(
        (path.stat().st_size, path.stat().st_mtime_ns) != before[name]
        for name, path in after.items()
    ):
        raise ValueError("checkpoint changed while sealing")
    result = {
        "schema": QWEN38_MEGATRON_SCHEMA,
        "source_plan_sha256": _unsigned_digest(plan),
        "source_plan": plan,
        "checkpoint_path": str(root),
        "optimizer_step": step,
        "world_size": 8,
        "tensor_parallel_size": 8,
        "sampler_batches_in_epoch": yielded,
        "files": inventory,
        "total_bytes": total,
        "gpu_reload_verified": False,
    }
    if "training_progress" in saved:
        result["training_progress"] = saved["training_progress"]
        _validate_progress(result)
    write_receipt(output, result)
    return receipt(output)


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
    verify_recovery_terminal(plan, step)
    if _is_qwen38_lora(plan):
        return _seal_qwen38_megatron(plan, step, output, progress=progress)
    saved = receipt(run / "checkpoint_receipts" / f"step-{step:06d}.json")
    expected = {
        "plan_sha256": _unsigned_digest(plan),
        "optimizer_step": step,
        "checkpoint_path": str(root),
    }
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError("checkpoint receipt differs from source plan/step/path")
    size = plan["recipe"]["nodes"] * plan["recipe"]["gpus_per_node"]
    adapter = "lora" in plan
    files = checkpoint_files(root, size, adapter=adapter)
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
    if adapter:
        _adapter_binding(plan, step, root, inventory)
    after = checkpoint_files(root, size, adapter=adapter)
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
    if "training_progress" in saved:
        result["training_progress"] = saved["training_progress"]
        _validate_progress(result)
    write_receipt(output, result)
    return receipt(output)


def verify(manifest: dict, *, check_files: bool = True) -> None:
    if manifest.get("schema") == QWEN38_MEGATRON_SCHEMA:
        return _verify_qwen38_megatron(manifest, check_files=check_files)
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
    adapter = "lora" in plan
    if "training_progress" in manifest:
        _validate_progress(manifest)
    _validate_names(set(manifest["files"]), size, adapter=adapter)
    if check_files:
        files = checkpoint_files(expected, size, adapter=adapter)
        if set(files) != set(manifest["files"]):
            raise ValueError("checkpoint inventory changed")
        for name, path in files.items():
            spec = manifest["files"][name]
            if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError("checkpoint file digest/size mismatch")
        if adapter:
            _adapter_binding(plan, step, expected, manifest["files"])


def _verify_qwen38_megatron(manifest: dict, *, check_files: bool) -> None:
    if manifest.get("receipt_sha256") != _unsigned_digest(
        {key: value for key, value in manifest.items() if key != "receipt_sha256"}
    ):
        raise ValueError("checkpoint manifest digest mismatch")
    expected_fields = {
        "schema",
        "source_plan_sha256",
        "source_plan",
        "checkpoint_path",
        "optimizer_step",
        "world_size",
        "tensor_parallel_size",
        "sampler_batches_in_epoch",
        "files",
        "total_bytes",
        "gpu_reload_verified",
        "receipt_sha256",
    }
    if "training_progress" in manifest:
        expected_fields.add("training_progress")
    if set(manifest) != expected_fields:
        raise ValueError("Qwen3.8 Megatron checkpoint manifest fields drift")
    plan = manifest["source_plan"]
    validate_plan(plan, check_files=False)
    if not _is_qwen38_lora(plan):
        raise ValueError("Megatron checkpoint source is not the qualified Qwen3.8 LoRA plan")
    step = manifest["optimizer_step"]
    root = Path(plan["output_root"]) / "checkpoints" / f"global_step_{step}"
    epoch_steps = math.ceil(plan["datasets"]["train"]["rows"] / plan["recipe"]["batch_size"])
    if (
        type(step) is not int
        or not 0 < step <= plan["recipe"]["max_steps"]
        or manifest["source_plan_sha256"] != _unsigned_digest(plan)
        or manifest["checkpoint_path"] != str(root)
        or manifest["world_size"] != 8
        or manifest["tensor_parallel_size"] != 8
        or manifest["sampler_batches_in_epoch"] != (step - 1) % epoch_steps + 1
        or manifest["gpu_reload_verified"] is not False
        or manifest["total_bytes"] != sum(spec["bytes"] for spec in manifest["files"].values())
    ):
        raise ValueError("Qwen3.8 Megatron checkpoint manifest bindings disagree")
    for name, spec in manifest["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != name:
            raise ValueError("checkpoint file escapes source root")
        if set(spec) != {"bytes", "sha256"} or type(spec["bytes"]) is not int or spec["bytes"] <= 0:
            raise ValueError("invalid checkpoint file record")
        if not re.fullmatch(r"[a-f0-9]{64}", spec["sha256"]):
            raise ValueError("invalid checkpoint file digest")
    if "training_progress" in manifest:
        _validate_progress(manifest)
    if check_files:
        files = qwen38_megatron_checkpoint_files(root, 8)
        if set(files) != set(manifest["files"]):
            raise ValueError("checkpoint inventory changed")
        for name, path in files.items():
            spec = manifest["files"][name]
            if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
                raise ValueError("checkpoint file digest/size mismatch")


def _validate_progress(manifest: dict) -> None:
    progress = manifest["training_progress"]
    if not isinstance(progress, dict) or set(progress) != {"supervised_tokens", "best"}:
        raise ValueError("invalid checkpoint training progress")
    tokens = progress["supervised_tokens"]
    source = manifest["source_plan"]
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("checkpoint supervised-token count must be positive")
    per_epoch = source["datasets"]["train"].get("supervised_tokens")
    if per_epoch is not None and tokens > per_epoch * source["recipe"]["epochs"]:
        raise ValueError("checkpoint supervised-token count exceeds the frozen corpus budget")
    best = progress["best"]
    if best is not None and (
        not isinstance(best, dict)
        or set(best)
        != {"optimizer_step", "task_macro_loss", "token_weighted_loss", "checkpoint_path"}
        or type(best["optimizer_step"]) is not int
        or not 0 <= best["optimizer_step"] <= manifest["optimizer_step"]
        or any(
            type(best[key]) not in (int, float) or not math.isfinite(best[key]) or best[key] < 0
            for key in ("task_macro_loss", "token_weighted_loss")
        )
        or not isinstance(best["checkpoint_path"], str)
        or not Path(best["checkpoint_path"]).is_absolute()
    ):
        raise ValueError("invalid checkpoint selection progress")
