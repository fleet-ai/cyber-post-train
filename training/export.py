"""CPU-only native-checkpoint → complete BF16 Hugging Face export.

One tensor is reconstructed at a time from trusted, hash-bound SkyRL rank files.
No trainer, optimizer, collective, GPU, or remote model code is executed. The
qualified Qwen causal loader omits the base's frozen MTP head; restore only that
exact allowlist, never an arbitrary missing weight. GPU reload is a later gate.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from .checkpoints import checkpoint_files, receipt, verify
from .post_sft_artifacts import QWEN36_EXACT_MTP_OMISSION_KEYS as FROZEN_MTP_KEYS
from .post_sft_artifacts import _safetensor_layout
from .post_sft_cast import (
    _fsync_directory,
    _fsync_file,
    _load_tensor,
    _rename_noreplace,
    _shard_groups,
)
from .sft_runtime import _checked_file, digest, write_receipt

MAX_SOURCE_BYTES = 8 * 1024**3
MAX_SHARD_BYTES = 3 * 1024**3
SIDECARS = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
}


def reassemble(parts: list, shape: list[int]):
    """Join a one-dimensional native FSDP shard; reject unsupported placements."""
    import torch
    from torch.distributed.tensor import DTensor, Replicate, Shard

    if not parts or math.prod(shape) * 4 > MAX_SOURCE_BYTES:
        raise ValueError("source tensor exceeds the bounded export contract")
    first = parts[0]
    if isinstance(first, DTensor):
        for tensor in parts:
            if (
                not isinstance(tensor, DTensor)
                or tensor.device_mesh.mesh.tolist() != list(range(len(parts)))
                or len(tensor.placements) != 1
                or tensor.placements != first.placements
                or list(tensor.shape) != shape
            ):
                raise ValueError("rank tensor mesh/placement/shape mismatch")
        placement = first.placements[0]
        local = [tensor.to_local() for tensor in parts]
    elif len(parts) == 1 and isinstance(first, torch.Tensor):
        placement, local = Replicate(), parts
    else:
        raise ValueError("multi-rank checkpoint must contain native DTensors")
    if any(t.device.type != "cpu" or t.dtype != torch.float32 for t in local):
        raise ValueError("expected CPU FP32 trained shards")
    if isinstance(placement, Shard):
        dim = placement.dim
        if not 0 <= dim < len(shape):
            raise ValueError("invalid shard dimension")
        chunk = math.ceil(shape[dim] / len(local))
        for rank, tensor in enumerate(local):
            expected = list(shape)
            expected[dim] = max(0, min(chunk, shape[dim] - rank * chunk))
            if list(tensor.shape) != expected:
                raise ValueError("local shard shape mismatch")
        value = torch.cat(local, dim=dim)
    elif isinstance(placement, Replicate):
        if any(not torch.equal(local[0], tensor) for tensor in local[1:]):
            raise ValueError("replicated rank tensors disagree")
        value = local[0]
    else:
        raise ValueError("partial/reduction placements are unsupported")
    if list(value.shape) != shape or not torch.isfinite(value).all():
        raise ValueError("reconstructed tensor is malformed or nonfinite")
    return value.to(torch.bfloat16).contiguous()


def export(manifest_path: Path, expected_sha256: str, output: Path, *, progress=None) -> dict:
    import torch
    from safetensors.torch import save_file

    if torch.cuda.is_available():
        raise ValueError("checkpoint export does not need a GPU")
    _checked_file(manifest_path, expected_sha256)
    manifest = receipt(manifest_path)
    plan = manifest["source_plan"]
    if plan["model"]["repo"] != "Qwen/Qwen3.8-27B":
        raise ValueError("model needs a separately qualified export profile")
    source, base = Path(manifest["checkpoint_path"]), Path(plan["model"]["root"])
    partial = output.with_name(output.name + ".partial")
    for path in (output, partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError("export destination or partial already exists")
        if path.resolve().is_relative_to(source.resolve()) or path.resolve().is_relative_to(
            base.resolve()
        ):
            raise ValueError("export must not write inside source checkpoint/base")
    if progress:
        progress("verify_source", 0, 0)
    verify(manifest)
    files = checkpoint_files(source, manifest["world_size"])
    for entry in plan["model"]["files"]:
        path = base / entry["path"]
        _checked_file(path, entry["sha256"])
        files["base/" + entry["path"]] = path
    before = {name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in files.items()}
    layout, base_shards = _safetensor_layout(base)
    if set(base_shards) | {"model.safetensors.index.json"} | SIDECARS != {
        entry["path"] for entry in plan["model"]["files"]
    }:
        raise ValueError("base weights/index/runtime sidecars are not exhaustively bound")
    if any(spec["dtype"] != "BF16" for spec in layout.values()):
        raise ValueError("export requires an exact BF16 base")
    states = [
        torch.load(
            source / "policy" / f"model_world_size_{manifest['world_size']}_rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for rank in range(manifest["world_size"])
    ]
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("checkpoint ranks disagree on tensor keys")
    missing = set(layout) - keys
    if keys - set(layout) or missing != set(FROZEN_MTP_KEYS):
        raise ValueError("only the exact frozen-base MTP tensors may be restored")
    groups = _shard_groups(layout, MAX_SHARD_BYTES)
    partial.mkdir(parents=True, mode=0o700)
    index, total, checked = {}, 0, 0
    for number, group in enumerate(groups, 1):
        name = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        tensors = {}
        for key in group:
            spec = layout[key]
            tensor = (
                _load_tensor(base, spec["shard"], key).clone()
                if key in missing
                else reassemble([state[key] for state in states], spec["shape"])
            )
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite export tensor")
            tensors[key] = tensor
            index[key] = name
            total += tensor.numel() * tensor.element_size()
        save_file(tensors, partial / name, metadata={"format": "pt"})
        _fsync_file(partial / name)
        for key, expected in tensors.items():
            reopened = _load_tensor(partial, name, key)
            if not torch.equal(reopened, expected):
                raise ValueError("export tensor differs after write/reopen")
            checked += 1
        del tensors, tensor, expected, reopened
        if progress:
            progress("export_shards", checked, total)
    (partial / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": index}, sort_keys=True) + "\n"
    )
    sidecars = {}
    for entry in plan["model"]["files"]:
        name = entry["path"]
        if name in base_shards or name == "model.safetensors.index.json":
            continue
        shutil.copyfile(base / name, partial / name)
        _checked_file(partial / name, entry["sha256"])
        sidecars[name] = digest(partial / name)
    exported, _ = _safetensor_layout(partial)
    if {k: (v["shape"], v["dtype"]) for k, v in layout.items()} != {
        k: (v["shape"], v["dtype"]) for k, v in exported.items()
    }:
        raise ValueError("export layout differs from exact base")
    if any((p.stat().st_size, p.stat().st_mtime_ns) != before[k] for k, p in files.items()):
        raise ValueError("source checkpoint/base changed during export")
    payload = {}
    for path in sorted(partial.iterdir()):
        _fsync_file(path)
        payload[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    result = {
        "schema": "cyber_native_checkpoint_hf_export_v1",
        "source_checkpoint_receipt_sha256": manifest["receipt_sha256"],
        "source_manifest_file_sha256": digest(manifest_path),
        "source_plan_sha256": manifest["source_plan_sha256"],
        "code_sha256": {
            name: digest(Path(__file__).with_name(name))
            for name in (
                "export.py",
                "checkpoints.py",
                "sft_runtime.py",
                "post_sft_cast.py",
                "post_sft_artifacts.py",
                "post_sft_base_surface.py",
                "io.py",
            )
        },
        "model_repo": plan["model"]["repo"],
        "model_revision": plan["model"]["revision"],
        "output_root": str(output),
        "optimizer_step": manifest["optimizer_step"],
        "optimizer_steps_executed": 0,
        "gpu_reload_verified": False,
        "trained_tensors": len(keys),
        "restored_base_tensors": sorted(missing),
        "tensor_values": total // 2,
        "tensor_bytes": total,
        "dtype": "BF16",
        "sidecars": sidecars,
        "files": payload,
        "source_inventory_sizes_mtimes_unchanged": True,
        "all_output_tensors_reopened_equal": True,
    }
    write_receipt(partial / "EXPORT.json", result)
    _fsync_directory(partial)
    _rename_noreplace(partial, output)
    _fsync_directory(output.parent)
    return receipt(output / "EXPORT.json")
