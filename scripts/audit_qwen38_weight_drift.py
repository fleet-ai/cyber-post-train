#!/usr/bin/env python3
"""Read-only, value-free audit of Qwen BF16 exports against their frozen base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

SIDECARS = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
CHUNK_ELEMENTS = 2_000_000
FROZEN_MTP_KEYS = {
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
}
REQUIRED_TRAINED_CATEGORIES = {
    "embedding",
    "lm_head",
    "self_attention",
    "linear_attention",
    "mlp",
    "normalization",
}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path.name}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def tensor_bytes(dtype: str, shape: list[int]) -> int:
    if dtype != "BF16" or not shape or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError("only well-formed BF16 tensors are supported")
    return math.prod(shape) * 2


def safetensor_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensor {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= 128 * 1024**2:
            raise ValueError(f"invalid safetensor header length {path.name}")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError("safetensor header is not an object")
    return 8 + length, header


def layout(root: Path) -> tuple[dict, dict]:
    index_path = root / "model.safetensors.index.json"
    index = strict_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"invalid weight map under {root}")
    shards = set(weight_map.values())
    actual = {path.name for path in root.glob("*.safetensors")}
    if shards != actual or any(Path(name).name != name for name in shards):
        raise ValueError(f"shard inventory mismatch under {root}")
    result, shard_facts = {}, {}
    for shard in sorted(shards):
        path = root / shard
        start, header = safetensor_header(path)
        entries = {key: value for key, value in header.items() if key != "__metadata__"}
        spans = []
        for key, spec in entries.items():
            if not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data_offsets"}:
                raise ValueError(f"malformed tensor header in {shard}")
            offsets = spec["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1]
                or offsets[1] - offsets[0] != tensor_bytes(spec["dtype"], spec["shape"])
            ):
                raise ValueError(f"malformed tensor offsets in {shard}")
            if start + offsets[1] > path.stat().st_size:
                raise ValueError(f"tensor exceeds shard in {shard}")
            spans.append((offsets[0], offsets[1], key))
            result[key] = {
                "path": path,
                "start": start + offsets[0],
                "bytes": offsets[1] - offsets[0],
                "dtype": spec["dtype"],
                "shape": spec["shape"],
            }
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:], strict=False)):
            raise ValueError(f"overlapping tensors in {shard}")
        expected_keys = {key for key, value in weight_map.items() if value == shard}
        if expected_keys != set(entries):
            raise ValueError(f"index/header key mismatch in {shard}")
        shard_facts[shard] = {
            "bytes": path.stat().st_size,
            "tensor_count": len(entries),
            "payload_bytes": sum(end - begin for begin, end, _ in spans),
        }
    if set(result) != set(weight_map):
        raise ValueError("layout key mismatch")
    metadata_total = index.get("metadata", {}).get("total_size")
    total = sum(spec["bytes"] for spec in result.values())
    if metadata_total != total:
        raise ValueError(f"index total_size mismatch under {root}")
    return result, {
        "index_sha256": sha(index_path),
        "layout_sha256": hashlib.sha256(
            canonical(
                {
                    key: {name: value for name, value in spec.items() if name != "path"}
                    for key, spec in result.items()
                }
            )
        ).hexdigest(),
        "tensor_count": len(result),
        "tensor_values": total // 2,
        "tensor_bytes": total,
        "shard_count": len(shards),
        "shard_inventory_sha256": hashlib.sha256(canonical(shard_facts)).hexdigest(),
        "index_header_offsets_valid": True,
    }


def category(name: str) -> str:
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual.") or ".visual." in name:
        return "visual"
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "lm_head"
    if ".self_attn." in name:
        return "self_attention"
    if ".linear_attn." in name or ".mamba." in name:
        return "linear_attention"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "normalization"
    return "other"


def empty_stats() -> dict:
    return {
        "tensors": 0,
        "changed_tensors": 0,
        "elements": 0,
        "changed_elements": 0,
        "reference_zero_elements": 0,
        "candidate_zero_elements": 0,
        "new_candidate_all_zero_tensors": 0,
        "candidate_all_zero_tensors": 0,
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "reference_sum_sq": 0.0,
        "max_abs": 0.0,
        "nonfinite": 0,
    }


def chunk_metrics(
    ref: np.ndarray,
    candidate: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> dict:
    diff = right - left
    finite = np.isfinite(left) & np.isfinite(right)
    result = {
        "elements": int(ref.size),
        "changed_elements": int(np.count_nonzero(ref != candidate)),
        "reference_zero_elements": int(np.count_nonzero((ref & 0x7FFF) == 0)),
        "candidate_zero_elements": int(np.count_nonzero((candidate & 0x7FFF) == 0)),
        "sum_abs": 0.0,
        "sum_sq": 0.0,
        "reference_sum_sq": 0.0,
        "max_abs": 0.0,
        "nonfinite": int(finite.size - np.count_nonzero(finite)),
    }
    if np.any(finite):
        selected = diff[finite]
        reference = left[finite]
        absolute = np.abs(selected)
        result["sum_abs"] = float(np.sum(absolute, dtype=np.float64))
        result["sum_sq"] = float(np.sum(selected * selected, dtype=np.float64))
        result["reference_sum_sq"] = float(np.sum(reference * reference, dtype=np.float64))
        result["max_abs"] = float(np.max(absolute))
    return result


def add(stats: dict, metrics: dict) -> None:
    for key in (
        "elements",
        "changed_elements",
        "reference_zero_elements",
        "candidate_zero_elements",
        "sum_abs",
        "sum_sq",
        "reference_sum_sq",
        "nonfinite",
    ):
        stats[key] += metrics[key]
    stats["max_abs"] = max(stats["max_abs"], metrics["max_abs"])


def finish(stats: dict) -> dict:
    elements = stats["elements"]
    result = {
        "tensors": stats["tensors"],
        "changed_tensors": stats["changed_tensors"],
        "elements": elements,
        "changed_elements": stats["changed_elements"],
        "changed_element_fraction": stats["changed_elements"] / elements if elements else 0.0,
        "reference_zero_elements": stats["reference_zero_elements"],
        "candidate_zero_elements": stats["candidate_zero_elements"],
        "candidate_zero_element_fraction": stats["candidate_zero_elements"] / elements
        if elements
        else 0.0,
        "candidate_all_zero_tensors": stats["candidate_all_zero_tensors"],
        "new_candidate_all_zero_tensors": stats["new_candidate_all_zero_tensors"],
        "mean_abs_diff": stats["sum_abs"] / elements if elements else 0.0,
        "rms_diff": math.sqrt(stats["sum_sq"] / elements) if elements else 0.0,
        "relative_l2": math.sqrt(stats["sum_sq"] / stats["reference_sum_sq"])
        if stats["reference_sum_sq"]
        else 0.0,
        "max_abs_diff": stats["max_abs"],
        "nonfinite_values": stats["nonfinite"],
    }
    return result


def audit_drift(base: dict, exports: list[tuple[str, dict]]) -> dict:
    expected = {key: (value["dtype"], value["shape"]) for key, value in base.items()}
    for label, current in exports:
        if {key: (value["dtype"], value["shape"]) for key, value in current.items()} != expected:
            raise ValueError(f"layout mismatch for {label}")
    labels = [label for label, _ in exports]
    pairs = [("base", label) for label in labels] + list(zip(labels, labels[1:], strict=False))
    stats = {
        pair: {
            "global": empty_stats(),
            "categories": defaultdict(empty_stats),
            "layers": defaultdict(empty_stats),
        }
        for pair in pairs
    }
    roots = [("base", base), *exports]
    handles = {}
    try:
        for key in sorted(base):
            specs = [(label, value[key]) for label, value in roots]
            group = category(key)
            match = LAYER.search(key)
            layer = match.group(1) if match else None
            tensor_changed = {pair: False for pair in pairs}
            tensor_reference_all_zero = {pair: True for pair in pairs}
            tensor_all_zero = {pair: True for pair in pairs}
            remaining = specs[0][1]["bytes"]
            offset = 0
            while remaining:
                length = min(remaining, CHUNK_ELEMENTS * 2)
                arrays = {}
                for label, spec in specs:
                    path = spec["path"]
                    handle = handles.setdefault(path, path.open("rb"))
                    handle.seek(spec["start"] + offset)
                    raw = handle.read(length)
                    if len(raw) != length:
                        raise ValueError("short tensor payload read")
                    arrays[label] = np.frombuffer(raw, dtype="<u2")
                floats = {
                    label: (value.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
                    for label, value in arrays.items()
                }
                for pair in pairs:
                    left, right = arrays[pair[0]], arrays[pair[1]]
                    metrics = chunk_metrics(
                        left,
                        right,
                        floats[pair[0]],
                        floats[pair[1]],
                    )
                    tensor_changed[pair] |= metrics["changed_elements"] > 0
                    tensor_reference_all_zero[pair] &= (
                        metrics["reference_zero_elements"] == metrics["elements"]
                    )
                    tensor_all_zero[pair] &= (
                        metrics["candidate_zero_elements"] == metrics["elements"]
                    )
                    for bucket in (
                        stats[pair]["global"],
                        stats[pair]["categories"][group],
                        stats[pair]["layers"][layer] if layer is not None else None,
                    ):
                        if bucket is not None:
                            add(bucket, metrics)
                remaining -= length
                offset += length
            for pair in pairs:
                for bucket in (
                    stats[pair]["global"],
                    stats[pair]["categories"][group],
                    stats[pair]["layers"][layer] if layer is not None else None,
                ):
                    if bucket is not None:
                        bucket["tensors"] += 1
                        bucket["changed_tensors"] += int(tensor_changed[pair])
                        bucket["candidate_all_zero_tensors"] += int(tensor_all_zero[pair])
                        bucket["new_candidate_all_zero_tensors"] += int(
                            tensor_all_zero[pair] and not tensor_reference_all_zero[pair]
                        )
    finally:
        for handle in handles.values():
            handle.close()
    result = {}
    for pair, value in stats.items():
        layer_values = [finish(item) for item in value["layers"].values()]
        result[f"{pair[0]}_vs_{pair[1]}"] = {
            "global": finish(value["global"]),
            "categories": {key: finish(item) for key, item in sorted(value["categories"].items())},
            "layer_summary": {
                "layers": len(layer_values),
                "layers_with_changed_tensors": sum(
                    item["changed_tensors"] > 0 for item in layer_values
                ),
                "relative_l2_min": min((item["relative_l2"] for item in layer_values), default=0.0),
                "relative_l2_median": float(
                    np.median([item["relative_l2"] for item in layer_values])
                )
                if layer_values
                else 0.0,
                "relative_l2_max": max((item["relative_l2"] for item in layer_values), default=0.0),
            },
        }
    return result


def interpret_drift(drift: dict, labels: list[str]) -> dict:
    issues = []
    warnings = []
    observations = {}
    previous = "base"
    for label in labels:
        base_pair = drift[f"base_vs_{label}"]
        adjacent = drift[f"{previous}_vs_{label}"]
        global_stats = base_pair["global"]
        categories = base_pair["categories"]
        mtp_changed = categories.get("mtp", {}).get("changed_tensors", 0)
        visual_changed = categories.get("visual", {}).get("changed_tensors", 0)
        training_tensor_changes = global_stats["changed_tensors"] - mtp_changed
        adjacent_mtp = adjacent["categories"].get("mtp", {}).get("changed_tensors", 0)
        adjacent_nonaux = adjacent["global"]["changed_tensors"] - adjacent_mtp
        observations[label] = {
            "changed_vs_base": global_stats["changed_tensors"],
            "changed_training_tensors_vs_base": training_tensor_changes,
            "changed_visual_tensors_vs_base": visual_changed,
            "changed_mtp_tensors_vs_base": mtp_changed,
            "changed_non_mtp_tensors_vs_previous": adjacent_nonaux,
            "changed_layers_vs_base": base_pair["layer_summary"]["layers_with_changed_tensors"],
        }
        if mtp_changed:
            issues.append(f"{label}: restored MTP tensors differ from frozen base")
        if training_tensor_changes <= 0:
            issues.append(f"{label}: no trained tensor differs from frozen base")
        if adjacent_nonaux <= 0:
            issues.append(f"{label}: no trained tensor differs from previous checkpoint")
        for group in sorted(REQUIRED_TRAINED_CATEGORIES & set(categories)):
            if categories[group]["changed_tensors"] <= 0:
                warnings.append(f"{label}: trained category {group} has no change from frozen base")
        if (
            base_pair["layer_summary"]["layers_with_changed_tensors"]
            != base_pair["layer_summary"]["layers"]
        ):
            warnings.append(
                f"{label}: at least one transformer layer has no change from frozen base"
            )
        if global_stats["nonfinite_values"]:
            issues.append(f"{label}: nonfinite BF16 values observed")
        if global_stats["new_candidate_all_zero_tensors"]:
            issues.append(f"{label}: tensor became entirely zero relative to frozen base")
        previous = label
    return {
        "structural_issues": issues,
        "diagnostic_warnings": warnings,
        "all_structural_weight_checks_pass": not issues,
        "checkpoints": observations,
    }


def validate_base(base: Path, weights_path: Path, lock_path: Path) -> dict:
    weights, lock = strict_json(weights_path), strict_json(lock_path)
    files = sorted(weights["files"], key=lambda item: item["path"])
    manifest = hashlib.sha256(canonical(files)).hexdigest()
    if manifest != weights["sha256"] or f"sha256:{manifest}" != lock["weights"]["manifest_sha256"]:
        raise ValueError("weight manifest digest mismatch")
    checked = 0
    for item in files:
        path = base / item["path"]
        if path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
            raise ValueError(f"base weight mismatch: {item['path']}")
        checked += 1
    expected = {
        "model.safetensors.index.json": lock["weights"]["index_sha256"].removeprefix("sha256:"),
        "config.json": lock["configuration"]["config_sha256"].removeprefix("sha256:"),
        "generation_config.json": lock["configuration"]["generation_config_sha256"].removeprefix(
            "sha256:"
        ),
        "preprocessor_config.json": lock["configuration"][
            "preprocessor_config_sha256"
        ].removeprefix("sha256:"),
        "video_preprocessor_config.json": lock["configuration"][
            "video_preprocessor_config_sha256"
        ].removeprefix("sha256:"),
        **{item["path"]: item["sha256"] for item in lock["tokenizer"]["files"]},
    }
    for name, expected_sha in expected.items():
        if sha(base / name) != expected_sha:
            raise ValueError(f"base sidecar mismatch: {name}")
    return {
        "revision": lock["revision"],
        "weight_shards_verified": checked,
        "weight_manifest_sha256": manifest,
        "index_and_sidecars_verified": len(expected),
        "base_matches_frozen_lock": True,
    }


def validate_export(
    root: Path,
    base: Path,
    label: str,
    expected_identity: tuple[str, str, str] | None,
) -> dict:
    receipt_path = root / "EXPORT.json"
    receipt = strict_json(receipt_path)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ValueError(f"invalid export receipt for {label}")
    if (
        receipt.get("schema") != "cyber_native_checkpoint_hf_export_v1"
        or receipt.get("optimizer_steps_executed") != 0
        or receipt.get("dtype") != "BF16"
        or receipt.get("all_output_tensors_reopened_equal") is not True
        or receipt.get("source_inventory_sizes_mtimes_unchanged") is not True
        or receipt.get("trained_tensors") != 1184
        or set(receipt.get("restored_base_tensors", [])) != FROZEN_MTP_KEYS
    ):
        raise ValueError(f"incomplete export receipt for {label}")
    files = receipt.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"missing payload inventory for {label}")
    actual = {path.name for path in root.iterdir() if path.name != "EXPORT.json"}
    if actual != set(files):
        raise ValueError(f"payload inventory mismatch for {label}")
    for name, expected in files.items():
        path = root / name
        if (
            not isinstance(expected, dict)
            or set(expected) != {"bytes", "sha256"}
            or path.stat().st_size != expected["bytes"]
            or sha(path) != expected["sha256"]
        ):
            raise ValueError(f"payload digest mismatch for {label}/{name}")
    payload_manifest_sha256 = hashlib.sha256(canonical(files)).hexdigest()
    file_sha256 = sha(receipt_path)
    if (
        expected_identity
        and (
            file_sha256,
            receipt["receipt_sha256"],
            payload_manifest_sha256,
        )
        != expected_identity
    ):
        raise ValueError(f"accepted export identity mismatch for {label}")
    sidecars = {}
    for name in SIDECARS:
        base_sha, export_sha = sha(base / name), sha(root / name)
        if base_sha != export_sha or receipt.get("sidecars", {}).get(name) != export_sha:
            raise ValueError(f"sidecar mismatch for {label}/{name}")
        sidecars[name] = export_sha
    return {
        "optimizer_step": receipt["optimizer_step"],
        "receipt_file_sha256": file_sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "payload_manifest_sha256": payload_manifest_sha256,
        "payload_files_verified": len(files),
        "trained_tensors": receipt["trained_tensors"],
        "restored_base_tensors": len(receipt["restored_base_tensors"]),
        "sidecars_match_base": len(sidecars),
        "source_unchanged": True,
        "reopen_equal": True,
    }


def f32_sample(checkpoint: Path, export: dict) -> dict:
    files = [checkpoint / "policy" / f"model_world_size_8_rank_{rank}.pt" for rank in range(8)]
    if not all(path.is_file() for path in files):
        return {"available": False, "reason": "native_checkpoint_rank_files_absent"}
    try:
        import torch
        from torch.distributed.tensor import DTensor, Replicate, Shard
    except ImportError:
        return {"available": False, "reason": "torch_runtime_absent"}
    states = [torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in files]
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("native checkpoint rank keys disagree")
    groups = defaultdict(list)
    for key in sorted(keys):
        groups[category(key)].append(key)
    selected = []
    for group in (
        "visual",
        "embedding",
        "lm_head",
        "normalization",
        "self_attention",
        "linear_attention",
        "mlp",
    ):
        candidates = groups[group]
        if not candidates:
            continue
        layered = defaultdict(list)
        for key in candidates:
            match = LAYER.search(key)
            layered[int(match.group(1)) if match else -1].append(key)
        layers = sorted(layer for layer in layered if layer >= 0)
        chosen_layers = {
            layers[index]
            for index in (0, len(layers) // 3, 2 * len(layers) // 3, len(layers) - 1)
            if layers
        }
        if -1 in layered:
            chosen_layers.add(-1)
        for layer in sorted(chosen_layers):
            selected.append(min(layered[layer], key=lambda key: math.prod(export[key]["shape"])))
    selected = selected[:24]
    exact, finite, source_fp32 = 0, 0, 0
    by_category = defaultdict(int)
    for key in selected:
        parts = [state[key] for state in states]
        first = parts[0]
        if isinstance(first, DTensor):
            placement = first.placements[0]
            local = [part.to_local() for part in parts]
        elif len(parts) == 1 and isinstance(first, torch.Tensor):
            placement, local = Replicate(), parts
        else:
            raise ValueError("unsupported native tensor")
        source_fp32 += int(
            all(part.dtype == torch.float32 and part.device.type == "cpu" for part in local)
        )
        if isinstance(placement, Shard):
            value = torch.cat(local, dim=placement.dim)
        elif isinstance(placement, Replicate):
            if any(not torch.equal(local[0], part) for part in local[1:]):
                raise ValueError("replicated native tensor disagreement")
            value = local[0]
        else:
            raise ValueError("unsupported native placement")
        value = value.reshape(export[key]["shape"])
        finite += int(bool(torch.isfinite(value).all()))
        spec = export[key]
        with spec["path"].open("rb") as handle:
            handle.seek(spec["start"])
            raw = handle.read(spec["bytes"])
        target = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(spec["shape"])
        exact += int(torch.equal(value.to(torch.bfloat16).contiguous(), target))
        by_category[category(key)] += 1
    return {
        "available": True,
        "sample_count": len(selected),
        "sample_identity_sha256": hashlib.sha256(canonical(selected)).hexdigest(),
        "by_category": dict(sorted(by_category.items())),
        "source_fp32_samples": source_fp32,
        "finite_source_samples": finite,
        "exact_f32_to_bf16_matches": exact,
        "all_samples_exact": bool(selected) and exact == finite == source_fp32 == len(selected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--export", action="append", required=True, help="LABEL=PATH")
    parser.add_argument(
        "--expected-export",
        action="append",
        default=[],
        help="LABEL=RECEIPT_FILE_SHA256,RECEIPT_SHA256,PAYLOAD_MANIFEST_SHA256",
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    exports = []
    for item in args.export:
        label, raw = item.split("=", 1)
        exports.append((label, Path(raw)))
    expected_exports = {}
    for item in args.expected_export:
        label, raw = item.split("=", 1)
        values = tuple(raw.split(","))
        if len(values) != 3 or label in expected_exports:
            raise ValueError("invalid or duplicate expected export identity")
        expected_exports[label] = values
    if expected_exports and set(expected_exports) != {label for label, _ in exports}:
        raise ValueError("expected export identity labels differ from export labels")
    base_layout, base_layout_facts = layout(args.base)
    layouts, layout_facts, receipts = [], {}, {}
    for label, root in exports:
        current, facts = layout(root)
        layouts.append((label, current))
        layout_facts[label] = facts
        receipts[label] = validate_export(root, args.base, label, expected_exports.get(label))
    drift = audit_drift(base_layout, layouts)
    interpretation = interpret_drift(drift, [label for label, _ in layouts])
    result = {
        "schema": "cyber_qwen38_weight_drift_audit_v1",
        "status": "passed" if interpretation["all_structural_weight_checks_pass"] else "rejected",
        "base": {
            **validate_base(args.base, args.weights_manifest, args.model_lock),
            **base_layout_facts,
        },
        "exports": receipts,
        "layouts": layout_facts,
        "drift": drift,
        "interpretation": interpretation,
        "step1000_f32_to_bf16": f32_sample(args.checkpoint, layouts[-1][1])
        if args.checkpoint
        else {"available": False, "reason": "checkpoint_not_requested"},
        "privacy": {
            "tensor_values_emitted": False,
            "tensor_names_emitted": False,
            "raw_weights_emitted": False,
            "prompts_traces_flags_answers_or_scores_included": False,
        },
    }
    result["sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
