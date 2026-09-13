"""Create-once Miles DCP -> Hugging Face export and a separate GPU load gate.

The exporter consumes the already accepted Miles checkpoint and native-reload
receipts.  It uses the exact offline converter shipped by the pinned Miles
image, restores only frozen visual/MTP tensors that the text-only Megatron
actor never owned, and publishes by a no-replace rename.  The GPU gate is a
separate process and does no rollout, forward, backward, optimizer, or save.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import API_URLS, digest

from . import miles
from .miles_conversion import _hash, _write
from .post_sft_artifacts import _safetensor_layout
from .post_sft_cast import (
    _fsync_directory,
    _fsync_file,
    _load_tensor,
    _rename_noreplace,
    _shard_groups,
)

EXPORT_SCHEMA = "cyber_miles_native_hf_export_v1"
RELOAD_SCHEMA = "cyber_miles_hf_zero_update_reload_v1"
RELOAD_CONTROLLER_SCHEMA = "cyber_miles_hf_reload_controller_terminal_v1"
RELOAD_RELEASE_SCHEMA = "cyber_miles_hf_reload_external_release_v1"
RELOAD_ACCEPTED_SCHEMA = "cyber_miles_hf_reload_accepted_v1"
CHECKPOINT_SCHEMA = "cyber_miles_training_checkpoint_v1"
TERMINAL_SCHEMA = "cyber_miles_reward_canary_terminal_v1"
NATIVE_RELOAD_SCHEMA = "cyber_miles_rl_reload_accepted_v1"
MODEL_REPO = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MILES_SOURCE_COMMIT = "2799fe386320c156334bf763ad4d7ca0f85dca4e"
MAX_SHARD_BYTES = 3 * 1024**3
CONVERTER_TIMEOUT_SECONDS = 4 * 60 * 60
HASH_CHUNK_BYTES = 8 * 1024**2
VOCAB_SIZE = 248_320
FROZEN_AUXILIARY_PREFIXES = ("model.visual.", "mtp.")
RELOAD_NAMESPACE = "fleet-train-jobs"

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,53}")


def _fields(value: str) -> frozenset[str]:
    return frozenset(value.split())  # noqa: SIM905 - compact reviewed schemas


_RELOAD_WORK_FIELDS = _fields(
    "optimizer_updates rollouts verifier_calls forwards backwards checkpoint_writes wandb_events"
)
_RELOAD_RESULT_FIELDS = _fields(
    "schema status run_name image export_path export_file_sha256 export_receipt_sha256 "
    "export_tensor_inventory_sha256 model_repo model_revision runtime_tensor_count "
    "artifact_only_mtp_tensor_count runtime_layout_sha256 all_runtime_weights_loaded "
    "source_export_unchanged optimizer_updates rollouts verifier_calls forwards backwards "
    "checkpoint_writes wandb_events gpus gpu_name peak_memory_bytes "
    "external_gpu_release_verified serving_qualified completed_at sha256"
)
_CONTROLLER_FIELDS = _fields(
    "schema status cluster api_base_url kube_context namespace namespace_uid run_name "
    "reload_result_path reload_result_file_sha256 reload_result_sha256 api_run_id "
    "api_run_name rayjob_name rayjob_uid workload_name workload_uid "
    "workload_owner_rayjob_uid raycluster_name raycluster_uid raycluster_owner_rayjob_uid "
    "pods api_status controller_status effective_priority automatic_requeue workers "
    "gpus_per_worker total_gpus observed_at sha256"
)
_POD_FIELDS = _fields(
    "name uid owner_raycluster_uid phase exit_code termination_reason terminated_at "
    "runtime_image_id container_restarts gpus"
)
_RELEASE_FIELDS = _fields(
    "schema status cluster api_base_url kube_context namespace namespace_uid run_name "
    "reload_result_path reload_result_file_sha256 reload_result_sha256 "
    "controller_terminal_path controller_terminal_file_sha256 controller_terminal_sha256 "
    "api_run_id api_run_name rayjob_name rayjob_uid workload_name workload_uid "
    "raycluster_name raycluster_uid pod_uids api_status controller_status rayjob_present "
    "workload_present quota_reservation_present raycluster_present gpu_pods_present "
    "active_gpu_pod_uids active_gpus observed_at sha256"
)
_ACCEPTED_FIELDS = _fields(
    "schema status export_path export_file_sha256 export_receipt_sha256 "
    "export_tensor_inventory_sha256 reload_result_path reload_result_file_sha256 "
    "reload_result controller_terminal_path controller_terminal_file_sha256 "
    "controller_terminal external_release_path external_release_file_sha256 external_release "
    "work_executed exact_hf_reload_verified source_export_unchanged_after_release "
    "external_gpu_release_verified post_export_promotion_requires_this_receipt "
    "serving_qualified sha256"
)

# The image digest pins the whole runtime; these hashes additionally pin the
# entrypoint plus the Qwen mapping and processor sources it exercises.
MILES_CONVERTER_SOURCES = {
    "tools/convert_torch_dist_to_hf.py": (
        "332bf9eedb5c72f83de69ac8e993d43c4ab4c2c30d4d2e60f6ba466ac5ea6390"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/__init__.py": (
        "adc2f476e0285700956bf5d32db3133667182fb9a4622a7f44081f9f4126fac7"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/qwen3_5.py": (
        "0fc2b8800b606a5386c065de562ed4dbb294f52caf93ecad6bc5094b36ebb2c1"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/__init__.py": (
        "65d2515cc094a4eabd0d2b991959056f19aa401b3e175dd79f1a37640dac1fe4"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/padding_remover.py": (
        "d45dc9f0235b6d95f15a3e4906d3873fbf69d6b95eaf5b54157b8b070e44927c"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py": (
        "8e5988739e8b1a60aaa19b5e858e50c47de2690210178836c193751e043d23df"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py": (
        "6a27879b2d4e1b0dc7adef696ddfbe3da1eb561a13ac77f22dd659b2559e3e88"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py": (
        "18411f746379e9c55f46579497a0e63b9953497206914d8079e8afd820f64f5b"
    ),
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py": (
        "45cf128d6e518fb537cc7b2359ce113b83add70085192ac8ec140fb82b862770"
    ),
    "miles/utils/hf_config.py": (
        "603617e36cf713e41356c0c2c20247443d87ecff7cd963480e1b4636a5a3f643"
    ),
}


def _unsigned(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "sha256"}


def _sealed(value: Any, schema: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("sha256", "").removeprefix("sha256:") != digest(_unsigned(value))
    ):
        raise ValueError("receipt schema or self-digest mismatch")
    return value


def _snapshot_json(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    expected = expected_sha256.removeprefix("sha256:")
    if _SHA256.fullmatch(expected) is None or path.is_symlink() or not path.is_file():
        raise ValueError("bound JSON file is missing or indirect")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in attributes):
        raise ValueError("bound JSON file changed while reading")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("bound JSON file digest mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("bound JSON file is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("bound JSON file must contain an object")
    return value, expected


def _terminal_checkpoint_reference(terminal: dict[str, Any]) -> dict[str, str]:
    _sealed(terminal, TERMINAL_SCHEMA)
    reference = terminal.get("checkpoint_manifest")
    if (
        terminal.get("status") != "accepted"
        or terminal.get("reward_values_included") is not False
        or terminal.get("task_content_included") is not False
        or terminal.get("production_promotion_requires_reload_acceptance") is not True
        or not isinstance(reference, dict)
        or set(reference) != {"path", "file_sha256", "receipt_sha256"}
    ):
        raise ValueError("Miles terminal acceptance is incomplete")
    return reference


def bind_source(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    terminal_path: Path,
    terminal_sha256: str,
    native_reload_path: Path,
    native_reload_sha256: str,
) -> dict[str, Any]:
    """Bind one accepted source without copying private terminal evidence."""
    from .miles_reload import _verify_checkpoint
    from .miles_reload_acceptance import validate_accepted

    checkpoint, checkpoint_file_sha256 = _snapshot_json(checkpoint_path, checkpoint_sha256)
    terminal, terminal_file_sha256 = _snapshot_json(terminal_path, terminal_sha256)
    native_reload, native_reload_file_sha256 = _snapshot_json(
        native_reload_path, native_reload_sha256
    )
    _sealed(checkpoint, CHECKPOINT_SCHEMA)
    reference = _terminal_checkpoint_reference(terminal)
    _sealed(native_reload, NATIVE_RELOAD_SCHEMA)
    validated_reload = validate_accepted(native_reload, check_files=True)
    checkpoint_self_sha256 = checkpoint["sha256"].removeprefix("sha256:")
    if (
        Path(reference["path"]) != checkpoint_path
        or reference["file_sha256"].removeprefix("sha256:") != checkpoint_file_sha256
        or reference["receipt_sha256"].removeprefix("sha256:") != checkpoint_self_sha256
        or terminal.get("source_plan_sha256", "").removeprefix("sha256:")
        != checkpoint.get("source", {}).get("plan_sha256")
        or validated_reload.get("source_manifest_sha256", "").removeprefix("sha256:")
        != checkpoint_self_sha256
        or validated_reload.get("source_terminal_acceptance_sha256", "").removeprefix("sha256:")
        != terminal["sha256"].removeprefix("sha256:")
        or native_reload.get("reload_plan", {}).get("source_manifest") != checkpoint
    ):
        raise ValueError("terminal/reload acceptance does not bind the selected checkpoint")
    source = checkpoint.get("source", {})
    arguments = source.get("arguments", {})
    model = checkpoint.get("model", {})
    if (
        checkpoint.get("image") != miles.IMAGE
        or checkpoint.get("world_size") != 8
        or checkpoint.get("topology") != {"nodes": 1, "gpus_per_node": 8}
        or arguments.get("nodes") != 1
        or arguments.get("gpus_per_node") != 8
        or model.get("repo") != MODEL_REPO
        or model.get("revision") != MODEL_REVISION
    ):
        raise ValueError("source is not the exact Qwen3.8 TP4 x CP2 Miles profile")
    _verify_checkpoint(checkpoint, hashes=True)
    return {
        "checkpoint": {
            "path": str(checkpoint_path),
            "file_sha256": checkpoint_file_sha256,
            "receipt_sha256": checkpoint_self_sha256,
        },
        "terminal_acceptance": {
            "path": str(terminal_path),
            "file_sha256": terminal_file_sha256,
            "receipt_sha256": terminal["sha256"].removeprefix("sha256:"),
        },
        "native_reload_acceptance": {
            "path": str(native_reload_path),
            "file_sha256": native_reload_file_sha256,
            "receipt_sha256": native_reload["sha256"].removeprefix("sha256:"),
        },
        "source_plan_sha256": checkpoint["source"]["plan_sha256"],
        "checkpoint_manifest": checkpoint,
    }


def _miles_root() -> Path:
    from miles.utils.external_utils.command_utils import repo_base_dir

    root = Path(repo_base_dir)
    for name, expected in MILES_CONVERTER_SOURCES.items():
        if _hash(root / name) != expected:
            raise ValueError("pinned Miles converter source changed")
    return root


def _sidecar_names(model: dict[str, Any]) -> set[str]:
    names = {item["path"] for item in model["files"]}
    weights = {name for name in names if name.endswith(".safetensors")}
    index = "model.safetensors.index.json"
    sidecars = names - weights - {index}
    if (
        index not in names
        or not weights
        or not sidecars
        or any(Path(name).name != name for name in names)
    ):
        raise ValueError("exact base inference inventory is malformed")
    return sidecars


def _verify_base(model: dict[str, Any]) -> dict[str, tuple[int, int, int, int, int]]:
    root = Path(model["root"])
    if root.is_symlink() or not root.is_dir():
        raise ValueError("exact base model is missing or indirect")
    snapshot = {}
    for item in model["files"]:
        path = root / item["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("exact base model contains a missing or indirect file")
        if _hash(path) != item["sha256"].removeprefix("sha256:"):
            raise ValueError("exact base model file digest mismatch")
        stat = path.stat()
        snapshot[item["path"]] = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    config = json.loads((root / "config.json").read_text())
    if config.get("auto_map") or config.get("text_config", {}).get("auto_map"):
        raise ValueError("remote model code is forbidden")
    return snapshot


def _hash_range(path: Path, start: int, size: int) -> str:
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, HASH_CHUNK_BYTES))
            if not chunk:
                raise ValueError("truncated safetensors payload")
            result.update(chunk)
            remaining -= len(chunk)
    if path.stat() != before:
        raise ValueError("safetensors shard changed while hashing")
    return result.hexdigest()


def tensor_inventory(root: Path, selected_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Hash every tensor payload from safetensors byte ranges, never values."""
    layout, shards = _safetensor_layout(root)
    result = []
    for shard_name in shards:
        path = root / shard_name
        if path.is_symlink() or not path.is_file():
            raise ValueError("safetensors shard is missing or indirect")
        before = path.stat()
        with path.open("rb") as stream:
            raw_size = stream.read(8)
            if len(raw_size) != 8:
                raise ValueError("invalid safetensors header")
            header_size = int.from_bytes(raw_size, "little")
            if not 2 <= header_size <= 64 * 1024**2:
                raise ValueError("safetensors header exceeds the bound")
            header = json.loads(stream.read(header_size))
        rows = {key: value for key, value in header.items() if key != "__metadata__"}
        expected_keys = {key for key, value in layout.items() if value["shard"] == shard_name}
        if set(rows) != expected_keys:
            raise ValueError("safetensors header/layout key mismatch")
        intervals = []
        for key, value in rows.items():
            offsets = value.get("data_offsets") if isinstance(value, dict) else None
            if (
                value.get("dtype") != "BF16"
                or value.get("shape") != layout[key]["shape"]
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] < offsets[1]
            ):
                raise ValueError("safetensors tensor metadata is invalid")
            size = offsets[1] - offsets[0]
            if size != 2 * math.prod(value["shape"]):
                raise ValueError("BF16 tensor byte size differs from shape")
            intervals.append((offsets[0], offsets[1], key))
            if selected_names is None or key in selected_names:
                result.append(
                    {
                        "name": key,
                        "shape": value["shape"],
                        "dtype": "BF16",
                        "shard": shard_name,
                        "bytes": size,
                        "sha256": _hash_range(path, 8 + header_size + offsets[0], size),
                    }
                )
        ordered = sorted(intervals)
        data_size = path.stat().st_size - 8 - header_size
        if (
            not ordered
            or ordered[0][0] != 0
            or ordered[-1][1] != data_size
            or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:], strict=False))
        ):
            raise ValueError("safetensors payload has a gap or overlap")
        if path.stat() != before:
            raise ValueError("safetensors shard changed during tensor inventory")
    if selected_names is not None and {row["name"] for row in result} != selected_names:
        raise ValueError("selected tensor inventory is incomplete")
    return sorted(result, key=lambda row: row["name"])


def _raw_contract(base: Path, candidate: Path) -> tuple[dict[str, Any], list[str]]:
    base_layout, _ = _safetensor_layout(base)
    candidate_layout, _ = _safetensor_layout(candidate)
    auxiliary = {key for key in base_layout if key.startswith(FROZEN_AUXILIARY_PREFIXES)}
    missing = set(base_layout) - set(candidate_layout)
    shape_drift = {
        key
        for key in set(base_layout) & set(candidate_layout)
        if base_layout[key]["shape"] != candidate_layout[key]["shape"]
    }
    if (
        not auxiliary
        or not any(key.startswith("model.visual.") for key in auxiliary)
        or not any(key.startswith("mtp.") for key in auxiliary)
        or set(candidate_layout) - set(base_layout)
        or missing != auxiliary
        or shape_drift
        or any(value["dtype"] != "BF16" for value in candidate_layout.values())
        or any(base_layout[key]["dtype"] != "BF16" for key in auxiliary)
    ):
        raise ValueError("raw Miles export differs outside frozen visual/MTP auxiliaries")
    return base_layout, sorted(auxiliary)


def _restore_auxiliary(
    base: Path, candidate: Path, base_layout: dict[str, Any], names: list[str]
) -> None:
    from safetensors.torch import save_file

    index_path = candidate / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map) & set(names):
        raise ValueError("raw Miles index cannot accept frozen auxiliaries")
    selected = {key: base_layout[key] for key in names}
    groups = _shard_groups(selected, MAX_SHARD_BYTES)
    restored_bytes = 0
    for number, group in enumerate(groups, 1):
        shard_name = f"model-frozen-aux-{number:05d}-of-{len(groups):05d}.safetensors"
        if (candidate / shard_name).exists():
            raise FileExistsError("frozen auxiliary shard already exists")
        tensors = {key: _load_tensor(base, base_layout[key]["shard"], key).clone() for key in group}
        if any(tensor.dtype.is_floating_point is False for tensor in tensors.values()):
            raise ValueError("frozen auxiliary tensor is not floating point")
        save_file(tensors, candidate / shard_name, metadata={"format": "pt"})
        _fsync_file(candidate / shard_name)
        for key, tensor in tensors.items():
            weight_map[key] = shard_name
            restored_bytes += tensor.numel() * tensor.element_size()
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or type(metadata.get("total_size")) is not int:
        raise ValueError("raw Miles index total size is invalid")
    metadata["total_size"] += restored_bytes
    index_path.write_text(json.dumps(index, sort_keys=True) + "\n")
    _fsync_file(index_path)


def _model_files(root: Path, sidecars: set[str]) -> dict[str, dict[str, Any]]:
    _, shards = _safetensor_layout(root)
    expected = set(shards) | sidecars | {"model.safetensors.index.json"}
    actual = {path.name for path in root.iterdir()}
    if actual != expected or any(
        path.is_symlink() or not path.is_file() for path in root.iterdir()
    ):
        raise ValueError("final HF export contains an unknown or indirect entry")
    return {
        name: {"bytes": (root / name).stat().st_size, "sha256": _hash(root / name)}
        for name in sorted(expected)
    }


def _invoke_converter(source: Path, destination: Path, metadata: Path, log: Path) -> None:
    miles_root = _miles_root()
    argv = [
        sys.executable,
        str(miles_root / "tools/convert_torch_dist_to_hf.py"),
        "--input-dir",
        str(source),
        "--output-dir",
        str(destination),
        "--origin-hf-dir",
        str(metadata),
        "--chunk-size",
        str(MAX_SHARD_BYTES),
        "--vocab-size",
        str(VOCAB_SIZE),
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with log.open("x") as stream:
        os.chmod(log, 0o600)
        result = subprocess.run(
            argv,
            cwd=miles_root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=CONVERTER_TIMEOUT_SECONDS,
            check=False,
        )
    if result.returncode:
        raise RuntimeError("pinned Miles conversion failed; private log preserved")


def export(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    terminal_path: Path,
    terminal_sha256: str,
    native_reload_path: Path,
    native_reload_sha256: str,
    output: Path,
) -> dict[str, Any]:
    """Create one exact HF directory; never reuse or replace a destination."""
    import torch

    if torch.cuda.is_available():
        raise ValueError("Miles HF export is CPU-only; GPU reload is a separate gate")
    source = bind_source(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        terminal_path=terminal_path,
        terminal_sha256=terminal_sha256,
        native_reload_path=native_reload_path,
        native_reload_sha256=native_reload_sha256,
    )
    checkpoint = source["checkpoint_manifest"]
    base = Path(checkpoint["model"]["root"])
    checkpoint_root = Path(checkpoint["root"])
    generation = checkpoint_root / f"iter_{checkpoint['rollout_index']:07d}"
    attempt = output.with_name(output.name + ".partial")
    candidate = attempt / "model"
    metadata = attempt / "base-metadata"
    private_log = attempt / "private-converter.log"
    if any(path.exists() or path.is_symlink() for path in (output, attempt)):
        raise FileExistsError("export destination or attempt artifact already exists")
    resolved_output = output.resolve()
    if any(
        resolved_output == root.resolve()
        or resolved_output.is_relative_to(root.resolve())
        or root.resolve().is_relative_to(resolved_output)
        for root in (base, checkpoint_root)
    ):
        raise ValueError("export destination must be disjoint from source and exact base")
    sidecars = _sidecar_names(checkpoint["model"])
    base_before = _verify_base(checkpoint["model"])
    attempt.mkdir(mode=0o700)
    metadata.mkdir(mode=0o700)
    for name in sidecars:
        shutil.copyfile(base / name, metadata / name)
    _invoke_converter(generation, candidate, metadata, private_log)
    base_layout, restored = _raw_contract(base, candidate)
    base_auxiliary_tensors = tensor_inventory(base, set(restored))
    _restore_auxiliary(base, candidate, base_layout, restored)
    final_layout, _ = _safetensor_layout(candidate)
    if {key: (value["shape"], value["dtype"]) for key, value in base_layout.items()} != {
        key: (value["shape"], value["dtype"]) for key, value in final_layout.items()
    }:
        raise ValueError("final HF tensor key/shape/BF16 parity failed")
    for name in sidecars:
        expected = next(item for item in checkpoint["model"]["files"] if item["path"] == name)
        if _hash(candidate / name) != expected["sha256"].removeprefix("sha256:"):
            raise ValueError("final HF sidecar differs from exact base")
    tensors = tensor_inventory(candidate)
    restored_set = set(restored)
    for row in tensors:
        row["source"] = "frozen_base_auxiliary" if row["name"] in restored_set else "trained"
    final_by_name = {row["name"]: row for row in tensors}
    if any(final_by_name[row["name"]]["sha256"] != row["sha256"] for row in base_auxiliary_tensors):
        raise ValueError("restored frozen auxiliary tensor differs from exact base")
    files = _model_files(candidate, sidecars)
    from .miles_reload import _verify_checkpoint

    _verify_checkpoint(checkpoint, hashes=True)
    if _verify_base(checkpoint["model"]) != base_before:
        raise ValueError("exact base changed during export")
    value = _write(
        candidate / "EXPORT.json",
        {
            "schema": EXPORT_SCHEMA,
            "status": "exported",
            "source": {key: item for key, item in source.items() if key != "checkpoint_manifest"},
            "miles": {
                "image": miles.IMAGE,
                "source_commit": MILES_SOURCE_COMMIT,
                "converter_sources": MILES_CONVERTER_SOURCES,
            },
            "model": {
                "repo": MODEL_REPO,
                "revision": MODEL_REVISION,
                "base_root": str(base),
                "native_parallelism": {"tensor": 4, "context": 2, "world_size": 8},
            },
            "output_root": str(output),
            "files": files,
            "tensor_inventory": tensors,
            "tensor_inventory_sha256": digest(tensors),
            "sidecars": {name: files[name] for name in sorted(sidecars)},
            "trained_tensor_count": len(tensors) - len(restored),
            "restored_base_tensor_count": len(restored),
            "restored_base_tensor_inventory_sha256": digest(base_auxiliary_tensors),
            "tensor_bytes": sum(row["bytes"] for row in tensors),
            "dtype": "BF16",
            "base_layout_sha256": digest(
                {
                    key: {"shape": value["shape"], "dtype": value["dtype"]}
                    for key, value in sorted(base_layout.items())
                }
            ),
            "output_layout_sha256": digest(
                {
                    key: {"shape": value["shape"], "dtype": value["dtype"]}
                    for key, value in sorted(final_layout.items())
                }
            ),
            "optimizer_updates_executed": 0,
            "source_checkpoint_unchanged": True,
            "base_model_unchanged": True,
            "create_only": True,
            "gpu_reload_verified": False,
        },
    )
    _fsync_file(candidate / "EXPORT.json")
    _fsync_directory(candidate)
    shutil.rmtree(metadata)
    private_log.unlink()
    _rename_noreplace(candidate, output)
    attempt.rmdir()
    _fsync_directory(output.parent)
    return value


def inspect_export(path: Path, expected_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value, _ = _snapshot_json(path, expected_sha256)
    _sealed(value, EXPORT_SCHEMA)
    root = path.parent
    if (
        path.name != "EXPORT.json"
        or value.get("status") != "exported"
        or value.get("output_root") != str(root)
        or value.get("dtype") != "BF16"
        or value.get("optimizer_updates_executed") != 0
        or value.get("source_checkpoint_unchanged") is not True
        or value.get("base_model_unchanged") is not True
        or value.get("create_only") is not True
        or value.get("gpu_reload_verified") is not False
        or value.get("miles")
        != {
            "image": miles.IMAGE,
            "source_commit": MILES_SOURCE_COMMIT,
            "converter_sources": MILES_CONVERTER_SOURCES,
        }
        or {item.name for item in root.iterdir()} != set(value.get("files", {})) | {"EXPORT.json"}
    ):
        raise ValueError("Miles HF export receipt is incomplete")
    for name, item in value["files"].items():
        candidate = root / name
        if (
            candidate.name != name
            or candidate.stat().st_size != item["bytes"]
            or _hash(candidate) != item["sha256"]
        ):
            raise ValueError("Miles HF export file inventory changed")
    tensors = tensor_inventory(root)
    by_name = {item["name"]: item["source"] for item in value["tensor_inventory"]}
    for row in tensors:
        row["source"] = by_name.get(row["name"])
    if tensors != value["tensor_inventory"] or digest(tensors) != value["tensor_inventory_sha256"]:
        raise ValueError("Miles HF tensor inventory changed")
    return value, tensors


def gpu_reload(
    export_path: Path, export_sha256: str, output: Path, *, run_name: str
) -> dict[str, Any]:
    """Load the complete HF artifact on one GPU without executing model work."""
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("HF reload run name is invalid")
    if output.exists() or output.is_symlink():
        raise FileExistsError("GPU reload receipt already exists")
    if output.resolve().is_relative_to(export_path.parent.resolve()):
        raise ValueError("GPU reload receipt must be outside the immutable export")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("HF reload gate requires exactly one visible GPU")
    before, tensors = inspect_export(export_path, export_sha256)
    root = export_path.parent
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if getattr(config, "auto_map", None) or getattr(config.get_text_config(), "auto_map", None):
        raise ValueError("remote model code is forbidden")
    torch.cuda.set_device(0)
    model, info = AutoModelForImageTextToText.from_pretrained(
        root,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    artifact = {row["name"]: row for row in tensors}
    state = model.state_dict()
    artifact_only = set(artifact) - set(state)
    if (
        any(not name.startswith("mtp.") for name in artifact_only)
        or set(state) - set(artifact)
        or any(list(tensor.shape) != artifact[name]["shape"] for name, tensor in state.items())
        or any(
            tensor.dtype != torch.bfloat16 or tensor.device.type != "cuda"
            for tensor in state.values()
        )
        or info.get("missing_keys")
        or info.get("mismatched_keys")
        or info.get("error_msgs")
        or set(info.get("unexpected_keys", [])) != artifact_only
    ):
        raise ValueError("HF GPU loader key/shape/BF16 contract failed")
    runtime_layout = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in sorted(state.items())
    }
    peak_memory = torch.cuda.max_memory_allocated(0)
    gpu_name = torch.cuda.get_device_name(0)
    del state, model
    torch.cuda.empty_cache()
    after, _ = inspect_export(export_path, export_sha256)
    if after != before:
        raise ValueError("HF export changed during GPU reload")
    return _write(
        output,
        {
            "schema": RELOAD_SCHEMA,
            "status": "reload_validated",
            "run_name": run_name,
            "image": miles.IMAGE,
            "export_path": str(export_path),
            "export_file_sha256": export_sha256.removeprefix("sha256:"),
            "export_receipt_sha256": before["sha256"].removeprefix("sha256:"),
            "export_tensor_inventory_sha256": before["tensor_inventory_sha256"],
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "runtime_tensor_count": len(runtime_layout),
            "artifact_only_mtp_tensor_count": len(artifact_only),
            "runtime_layout_sha256": digest(runtime_layout),
            "all_runtime_weights_loaded": True,
            "source_export_unchanged": True,
            "optimizer_updates": 0,
            "rollouts": 0,
            "verifier_calls": 0,
            "forwards": 0,
            "backwards": 0,
            "checkpoint_writes": 0,
            "wandb_events": 0,
            "gpus": 1,
            "gpu_name": gpu_name,
            "peak_memory_bytes": peak_memory,
            "external_gpu_release_verified": False,
            "serving_qualified": False,
            "completed_at": time.time(),
        },
    )


def _snapshot(path: Path) -> tuple[dict[str, Any], str]:
    """Read one regular JSON evidence file once and bind its exact bytes."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("HF reload evidence is missing or indirect")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in attributes):
        raise ValueError("HF reload evidence changed while being read")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HF reload evidence is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("HF reload evidence must contain an object")
    return value, hashlib.sha256(raw).hexdigest()


def _evidence_time(value: object) -> float:
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("HF reload evidence timestamp is invalid") from exc
        if parsed.tzinfo is not None:
            return parsed.timestamp()
    raise ValueError("HF reload evidence timestamp is invalid")


def _exact_integer(value: object, expected: int) -> bool:
    return type(value) is int and value == expected


def _runtime_image_digest(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("HF reload runtime image ID is absent")
    match = re.fullmatch(
        r"(?:containerd|docker-pullable)://(?:[^@\s]+@)?sha256:([a-f0-9]{64})", value
    )
    if match is None:
        raise ValueError("HF reload runtime image ID is not immutable")
    return match.group(1)


def _validate_reload_result(value: dict[str, Any]) -> float:
    _sealed(value, RELOAD_SCHEMA)
    if (
        set(value) != _RELOAD_RESULT_FIELDS
        or value.get("status") != "reload_validated"
        or _RUN_NAME.fullmatch(str(value.get("run_name", ""))) is None
        or value.get("image") != miles.IMAGE
        or value.get("model_repo") != MODEL_REPO
        or value.get("model_revision") != MODEL_REVISION
        or _SHA256.fullmatch(str(value.get("export_file_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("export_receipt_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("export_tensor_inventory_sha256", ""))) is None
        or _SHA256.fullmatch(str(value.get("runtime_layout_sha256", ""))) is None
        or type(value.get("runtime_tensor_count")) is not int
        or value["runtime_tensor_count"] < 1
        or type(value.get("artifact_only_mtp_tensor_count")) is not int
        or value["artifact_only_mtp_tensor_count"] < 0
        or value.get("all_runtime_weights_loaded") is not True
        or value.get("source_export_unchanged") is not True
        or any(not _exact_integer(value.get(key), 0) for key in _RELOAD_WORK_FIELDS)
        or not _exact_integer(value.get("gpus"), 1)
        or not isinstance(value.get("gpu_name"), str)
        or not value["gpu_name"]
        or type(value.get("peak_memory_bytes")) is not int
        or value["peak_memory_bytes"] <= 0
        or value.get("external_gpu_release_verified") is not False
        or value.get("serving_qualified") is not False
    ):
        raise ValueError("HF reload process receipt is incomplete or conflicting")
    return _evidence_time(value.get("completed_at"))


def _validate_controller(
    value: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    completed_at: float,
) -> float:
    _sealed(value, RELOAD_CONTROLLER_SCHEMA)
    run_id = value.get("api_run_id")
    expected_name = f"{result['run_name']}-{str(run_id)[:8]}"
    pods = value.get("pods")
    observed_at = _evidence_time(value.get("observed_at"))
    expected_image = miles.IMAGE.rsplit("@sha256:", 1)[-1]
    if (
        set(value) != _CONTROLLER_FIELDS
        or value.get("status") != "succeeded"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != API_URLS["dev"]
        or not isinstance(value.get("kube_context"), str)
        or not value["kube_context"]
        or value.get("namespace") != RELOAD_NAMESPACE
        or _UUID.fullmatch(str(value.get("namespace_uid"))) is None
        or value.get("run_name") != result["run_name"]
        or value.get("reload_result_path") != str(result_path)
        or value.get("reload_result_file_sha256") != result_file_sha256
        or value.get("reload_result_sha256") != result["sha256"].removeprefix("sha256:")
        or _UUID.fullmatch(str(run_id)) is None
        or value.get("api_run_name") != expected_name
        or value.get("rayjob_name") != expected_name
        or any(
            _UUID.fullmatch(str(value.get(key))) is None
            for key in ("rayjob_uid", "workload_uid", "raycluster_uid")
        )
        or any(
            not isinstance(value.get(key), str) or not value[key]
            for key in ("workload_name", "raycluster_name")
        )
        or value.get("workload_owner_rayjob_uid") != value.get("rayjob_uid")
        or value.get("raycluster_owner_rayjob_uid") != value.get("rayjob_uid")
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or not _exact_integer(value.get("effective_priority"), 10000)
        or value.get("automatic_requeue") is not False
        or not _exact_integer(value.get("workers"), 1)
        or not _exact_integer(value.get("gpus_per_worker"), 1)
        or not _exact_integer(value.get("total_gpus"), 1)
        or observed_at < completed_at
        or not isinstance(pods, list)
        or len(pods) != 1
    ):
        raise ValueError("HF reload controller evidence is incomplete or mismatched")
    pod = pods[0]
    if (
        not isinstance(pod, dict)
        or set(pod) != _POD_FIELDS
        or not isinstance(pod.get("name"), str)
        or not pod["name"]
        or _UUID.fullmatch(str(pod.get("uid"))) is None
        or pod.get("owner_raycluster_uid") != value["raycluster_uid"]
        or pod.get("phase") != "Succeeded"
        or not _exact_integer(pod.get("exit_code"), 0)
        or pod.get("termination_reason") != "Completed"
        or _evidence_time(pod.get("terminated_at")) < completed_at
        or _evidence_time(pod.get("terminated_at")) > observed_at
        or _runtime_image_digest(pod.get("runtime_image_id")) != expected_image
        or not _exact_integer(pod.get("container_restarts"), 0)
        or not _exact_integer(pod.get("gpus"), 1)
    ):
        raise ValueError("HF reload Pod evidence is incomplete or mismatched")
    identities = (
        run_id,
        value["rayjob_uid"],
        value["workload_uid"],
        value["raycluster_uid"],
        pod["uid"],
    )
    if len(set(identities)) != len(identities):
        raise ValueError("HF reload controller identities are not distinct")
    return observed_at


def _validate_release(
    value: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    controller: dict[str, Any],
    controller_path: Path,
    controller_file_sha256: str,
    not_before: float,
) -> None:
    _sealed(value, RELOAD_RELEASE_SCHEMA)
    identities = (
        "api_run_id",
        "api_run_name",
        "rayjob_name",
        "rayjob_uid",
        "workload_name",
        "workload_uid",
        "raycluster_name",
        "raycluster_uid",
    )
    if (
        set(value) != _RELEASE_FIELDS
        or value.get("status") != "released"
        or value.get("cluster") != "dev"
        or value.get("api_base_url") != API_URLS["dev"]
        or value.get("kube_context") != controller["kube_context"]
        or value.get("namespace") != RELOAD_NAMESPACE
        or value.get("namespace_uid") != controller["namespace_uid"]
        or value.get("run_name") != result["run_name"]
        or value.get("reload_result_path") != str(result_path)
        or value.get("reload_result_file_sha256") != result_file_sha256
        or value.get("reload_result_sha256") != result["sha256"].removeprefix("sha256:")
        or value.get("controller_terminal_path") != str(controller_path)
        or value.get("controller_terminal_file_sha256") != controller_file_sha256
        or value.get("controller_terminal_sha256") != controller["sha256"].removeprefix("sha256:")
        or any(value.get(key) != controller[key] for key in identities)
        or value.get("pod_uids") != [controller["pods"][0]["uid"]]
        or value.get("api_status") != "SUCCEEDED"
        or value.get("controller_status") != "SUCCEEDED"
        or value.get("rayjob_present") is not False
        or value.get("workload_present") is not False
        or value.get("quota_reservation_present") is not False
        or value.get("raycluster_present") is not False
        or value.get("gpu_pods_present") is not False
        or value.get("active_gpu_pod_uids") != []
        or not _exact_integer(value.get("active_gpus"), 0)
        or _evidence_time(value.get("observed_at")) < not_before
    ):
        raise ValueError("HF reload external release is incomplete or mismatched")


def _validate_reload_evidence(
    *,
    result: dict[str, Any],
    result_path: Path,
    result_file_sha256: str,
    controller: dict[str, Any],
    controller_path: Path,
    controller_file_sha256: str,
    release: dict[str, Any],
) -> dict[str, Any]:
    completed_at = _validate_reload_result(result)
    terminal_at = _validate_controller(
        controller, result, result_path, result_file_sha256, completed_at
    )
    _validate_release(
        release,
        result,
        result_path,
        result_file_sha256,
        controller,
        controller_path,
        controller_file_sha256,
        terminal_at,
    )
    export_path = Path(result["export_path"])
    if (
        not export_path.is_absolute()
        or not result_path.is_absolute()
        or result_path.resolve().is_relative_to(export_path.parent.resolve())
    ):
        raise ValueError("HF reload evidence/export paths are not disjoint absolute paths")
    exported, _ = inspect_export(export_path, result["export_file_sha256"])
    if (
        exported["sha256"].removeprefix("sha256:") != result["export_receipt_sha256"]
        or exported["tensor_inventory_sha256"] != result["export_tensor_inventory_sha256"]
    ):
        raise ValueError("HF reload result does not bind the selected export")
    return exported


def accept_gpu_reload(
    *,
    result_path: Path,
    controller_path: Path,
    release_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Join process, controller, and release evidence into one reload gate."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("HF reload acceptance destination already exists")
    expected = {
        controller_path: result_path.parent / "HF_RELOAD_CONTROLLER_TERMINAL.json",
        release_path: result_path.parent / "HF_RELOAD_RELEASE.json",
        output: result_path.parent / "HF_RELOAD_ACCEPTED.json",
    }
    if result_path.name != "HF_RELOAD_VALIDATED.json" or any(
        actual != wanted for actual, wanted in expected.items()
    ):
        raise ValueError("HF reload acceptance evidence is outside its bound root")
    result, result_file_sha256 = _snapshot(result_path)
    controller, controller_file_sha256 = _snapshot(controller_path)
    release, release_file_sha256 = _snapshot(release_path)
    exported = _validate_reload_evidence(
        result=result,
        result_path=result_path,
        result_file_sha256=result_file_sha256,
        controller=controller,
        controller_path=controller_path,
        controller_file_sha256=controller_file_sha256,
        release=release,
    )
    work = {key: result[key] for key in _RELOAD_WORK_FIELDS}
    return _write(
        output,
        {
            "schema": RELOAD_ACCEPTED_SCHEMA,
            "status": "accepted",
            "export_path": result["export_path"],
            "export_file_sha256": result["export_file_sha256"],
            "export_receipt_sha256": result["export_receipt_sha256"],
            "export_tensor_inventory_sha256": exported["tensor_inventory_sha256"],
            "reload_result_path": str(result_path),
            "reload_result_file_sha256": result_file_sha256,
            "reload_result": result,
            "controller_terminal_path": str(controller_path),
            "controller_terminal_file_sha256": controller_file_sha256,
            "controller_terminal": controller,
            "external_release_path": str(release_path),
            "external_release_file_sha256": release_file_sha256,
            "external_release": release,
            "work_executed": work,
            "exact_hf_reload_verified": True,
            "source_export_unchanged_after_release": True,
            "external_gpu_release_verified": True,
            "post_export_promotion_requires_this_receipt": True,
            "serving_qualified": False,
        },
    )


def validate_reload_accepted(value: dict[str, Any], *, check_files: bool = True) -> dict[str, str]:
    """Reopen every accepted HF reload input before a downstream consumer."""
    if not check_files:
        raise ValueError("HF reload acceptance requires reopening every evidence file")
    _sealed(value, RELOAD_ACCEPTED_SCHEMA)
    work = value.get("work_executed")
    if (
        set(value) != _ACCEPTED_FIELDS
        or value.get("status") != "accepted"
        or not isinstance(work, dict)
        or set(work) != _RELOAD_WORK_FIELDS
        or any(not _exact_integer(work.get(key), 0) for key in _RELOAD_WORK_FIELDS)
        or value.get("exact_hf_reload_verified") is not True
        or value.get("source_export_unchanged_after_release") is not True
        or value.get("external_gpu_release_verified") is not True
        or value.get("post_export_promotion_requires_this_receipt") is not True
        or value.get("serving_qualified") is not False
    ):
        raise ValueError("accepted HF reload receipt is incomplete")
    paths = {
        "reload_result": Path(value["reload_result_path"]),
        "controller_terminal": Path(value["controller_terminal_path"]),
        "external_release": Path(value["external_release_path"]),
    }
    reopened = {}
    for key, path in paths.items():
        observed, file_sha256 = _snapshot(path)
        if observed != value[key] or file_sha256 != value[f"{key}_file_sha256"]:
            raise ValueError("accepted HF reload evidence changed")
        reopened[key] = observed
    exported = _validate_reload_evidence(
        result=reopened["reload_result"],
        result_path=paths["reload_result"],
        result_file_sha256=value["reload_result_file_sha256"],
        controller=reopened["controller_terminal"],
        controller_path=paths["controller_terminal"],
        controller_file_sha256=value["controller_terminal_file_sha256"],
        release=reopened["external_release"],
    )
    if (
        value["export_path"] != reopened["reload_result"]["export_path"]
        or value["export_file_sha256"] != reopened["reload_result"]["export_file_sha256"]
        or value["export_receipt_sha256"] != reopened["reload_result"]["export_receipt_sha256"]
        or value["export_tensor_inventory_sha256"] != exported["tensor_inventory_sha256"]
    ):
        raise ValueError("accepted HF reload export binding changed")
    return {
        "export_receipt_sha256": value["export_receipt_sha256"],
        "reload_result_sha256": reopened["reload_result"]["sha256"],
        "controller_terminal_sha256": reopened["controller_terminal"]["sha256"],
        "external_release_sha256": reopened["external_release"]["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    for name in ("checkpoint", "terminal", "native-reload"):
        export_parser.add_argument(f"--{name}", type=Path, required=True)
        export_parser.add_argument(f"--{name}-sha256", required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    reload_parser = commands.add_parser("reload")
    reload_parser.add_argument("--export", type=Path, required=True)
    reload_parser.add_argument("--export-sha256", required=True)
    reload_parser.add_argument("--output", type=Path, required=True)
    reload_parser.add_argument("--run-name", required=True)
    accept_parser = commands.add_parser("accept-reload")
    accept_parser.add_argument("--result", type=Path, required=True)
    accept_parser.add_argument("--controller", type=Path, required=True)
    accept_parser.add_argument("--release", type=Path, required=True)
    accept_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export(
                checkpoint_path=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                terminal_path=args.terminal,
                terminal_sha256=args.terminal_sha256,
                native_reload_path=args.native_reload,
                native_reload_sha256=args.native_reload_sha256,
                output=args.output,
            )
        elif args.command == "reload":
            result = gpu_reload(
                args.export, args.export_sha256, args.output, run_name=args.run_name
            )
        else:
            result = accept_gpu_reload(
                result_path=args.result,
                controller_path=args.controller,
                release_path=args.release,
                output=args.output,
            )
        print(
            json.dumps(
                {
                    "schema": result["schema"],
                    "status": result["status"],
                    "sha256": result["sha256"],
                }
            )
        )
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
