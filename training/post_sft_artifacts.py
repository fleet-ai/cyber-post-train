"""Deterministic, read-only manifests for the post-SFT source and HF export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .io import digest_json

TOKENIZER_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

ALLOWED_TRAINER_CONFIG_DRIFT = {
    "dtype",
    "pad_token_id",
    "text_config.dtype",
    "transformers_version",
    "use_cache",
    "vision_config.dtype",
    "vision_config.model_type",
}


def _json_differences(base: Any, candidate: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(base, dict) and isinstance(candidate, dict):
        rows = []
        for key in sorted(set(base) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else key
            rows.extend(_json_differences(base.get(key), candidate.get(key), path))
        return rows
    if base != candidate:
        return [{"path": prefix, "base": base, "candidate": candidate}]
    return []


def compare_model_config_architecture(base_config: Path, candidate_config: Path) -> dict[str, Any]:
    """Allow known trainer/runtime metadata rewrites but no architectural drift."""

    base = json.loads(base_config.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_config.read_text(encoding="utf-8"))
    differences = _json_differences(base, candidate)
    unexpected = [row for row in differences if row["path"] not in ALLOWED_TRAINER_CONFIG_DRIFT]
    if unexpected:
        raise ValueError(
            "candidate model config contains architectural or undeclared drift: "
            + ", ".join(row["path"] for row in unexpected)
        )
    def without_allowed(document: dict[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(document))
        for dotted in ALLOWED_TRAINER_CONFIG_DRIFT:
            parent: Any = result
            parts = dotted.split(".")
            for part in parts[:-1]:
                parent = parent.get(part, {}) if isinstance(parent, dict) else {}
            if isinstance(parent, dict):
                parent.pop(parts[-1], None)
        return result

    normalized_base = without_allowed(base)
    normalized_candidate = without_allowed(candidate)
    if normalized_base != normalized_candidate:  # defensive: the diff check above should imply it
        raise ValueError("normalized model architectures differ")
    return {
        "schema": "cyber_sft_model_config_architecture_equivalence_v1",
        "all_architecture_and_vocab_fields_identical": True,
        "allowed_trainer_metadata_differences": differences,
        "base_config_sha256": sha256_file(base_config),
        "candidate_config_sha256": sha256_file(candidate_config),
        "normalized_architecture_sha256": digest_json(normalized_base),
    }


def _safetensor_layout(root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - cluster image supplies this dependency
        raise ValueError("safetensors is required to inspect an HF export") from exc

    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("HF export has no safetensors weight map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shard_names):
        raise ValueError("HF export weight map contains a non-safetensors shard")
    if sorted(path.name for path in root.glob("*.safetensors")) != shard_names:
        raise ValueError("HF export shards differ from its weight map")

    layout: dict[str, dict[str, Any]] = {}
    for name in shard_names:
        with safe_open(str(root / name), framework="pt", device="cpu") as shard:
            # Current safetensors exposes ``keys()`` but older test doubles and releases were
            # iterable. Supporting both keeps the artifact verifier version-tolerant.
            keys = shard.keys() if hasattr(shard, "keys") else iter(shard)
            for key in keys:
                if key in layout:
                    raise ValueError(f"duplicate tensor {key!r} in HF export")
                tensor_slice = shard.get_slice(key)
                shape = tensor_slice.get_shape()
                if not shape or not all(isinstance(value, int) and value > 0 for value in shape):
                    raise ValueError(f"invalid tensor shape for {key}")
                layout[key] = {
                    "shape": list(shape),
                    "dtype": tensor_slice.get_dtype(),
                    "shard": name,
                }
    if set(layout) != set(weight_map):
        raise ValueError("safetensors index keys differ from shard contents")
    for key, row in layout.items():
        if weight_map[key] != row["shard"]:
            raise ValueError(f"safetensors index points {key!r} at the wrong shard")
    return layout, shard_names


def compare_safetensor_layout(
    base_root: Path,
    candidate_root: Path,
    *,
    require_same_dtype: bool = True,
) -> dict[str, Any]:
    """Prove candidate keys/shapes, and optionally dtypes, agree with the base."""

    base, _ = _safetensor_layout(base_root.resolve(strict=True))
    candidate, _ = _safetensor_layout(candidate_root.resolve(strict=True))
    normalized_base = {
        key: {"shape": row["shape"], "dtype": row["dtype"]} for key, row in base.items()
    }
    normalized_candidate = {
        key: {"shape": row["shape"], "dtype": row["dtype"]}
        for key, row in candidate.items()
    }
    missing = sorted(set(base) - set(candidate))
    unexpected = sorted(set(candidate) - set(base))
    shape_mismatched = sorted(
        key
        for key in set(base) & set(candidate)
        if normalized_base[key]["shape"] != normalized_candidate[key]["shape"]
    )
    dtype_mismatched = sorted(
        key
        for key in set(base) & set(candidate)
        if normalized_base[key]["dtype"] != normalized_candidate[key]["dtype"]
    )
    if missing or unexpected or shape_mismatched or (require_same_dtype and dtype_mismatched):
        raise ValueError(
            "candidate safetensors layout differs from base: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatched={len(shape_mismatched)}, "
            f"dtype_mismatched={len(dtype_mismatched)}"
        )
    return {
        "schema": "cyber_sft_safetensors_layout_equivalence_v2",
        "tensor_count": len(candidate),
        "missing_key_count": 0,
        "unexpected_key_count": 0,
        "shape_mismatch_count": 0,
        "dtype_mismatch_count": len(dtype_mismatched),
        "dtype_match_required": require_same_dtype,
        "base_layout_sha256": digest_json(normalized_base),
        "candidate_layout_sha256": digest_json(normalized_candidate),
        "all_keys_and_shapes_match": True,
        "all_keys_shapes_and_dtypes_match": not dtype_mismatched,
    }


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def structural_manifest(root: Path) -> dict[str, Any]:
    """Hash path, size, and integer mtime without reading checkpoint payload bytes."""

    resolved = root.resolve(strict=True)
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    rows = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "size": path.stat().st_size,
            "mtime_seconds": int(path.stat().st_mtime),
        }
        for path in files
    ]
    return {
        "schema": "cyber_sft_structural_manifest_v1",
        "root": str(resolved),
        "file_count": len(rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }


def structural_tsv_sha256(root: Path) -> str:
    """Reproduce the pre-conversion BusyBox stat receipt without reading payload bytes."""

    resolved = root.resolve(strict=True)
    rows = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(f"{path}\\t{stat.st_size}\\t{int(stat.st_mtime)}\n")
    return "sha256:" + hashlib.sha256("".join(rows).encode()).hexdigest()


def full_file_manifest(root: Path) -> dict[str, Any]:
    """Serialize hashing to one file at a time to limit pressure on shared SFS."""

    resolved = root.resolve(strict=True)
    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    rows = []
    for path in files:
        rows.append(
            {
                "path": path.relative_to(resolved).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path).removeprefix("sha256:"),
            }
        )
    return {
        "schema": "cyber_sft_full_file_manifest_v1",
        "root": str(resolved),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }


def inspect_hf_export(
    root: Path,
    *,
    expected_tokenizer_manifest_sha256: str,
    expected_chat_template_sha256: str,
    expected_config_sha256: str,
    expected_parameter_count: int,
    expected_sidecar_sha256: dict[str, str] | None = None,
    require_base_sidecars: bool = True,
    expected_dtype: str = "BF16",
) -> dict[str, Any]:
    """Hash all HF files and validate safetensors shapes without materializing tensors."""

    resolved = root.resolve(strict=True)
    layout, shard_names = _safetensor_layout(resolved)

    parameter_count = 0
    tensor_count = 0
    weight_rows = []
    for name in shard_names:
        path = resolved / name
        weight_rows.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path).removeprefix("sha256:"),
            }
        )
    normalized_expected_dtype = expected_dtype.upper()
    if normalized_expected_dtype not in {"BF16", "F32"}:
        raise ValueError("expected dtype must be BF16 or F32")
    observed_dtypes = sorted({str(row["dtype"]).upper() for row in layout.values()})
    if observed_dtypes != [normalized_expected_dtype]:
        raise ValueError(
            "HF export tensor dtype differs from expectation: "
            f"expected={normalized_expected_dtype}, observed={observed_dtypes}"
        )
    for row in layout.values():
        parameter_count += math.prod(row["shape"])
        tensor_count += 1
    if parameter_count != expected_parameter_count:
        raise ValueError("HF export parameter count differs from the base architecture")

    tokenizer_rows = []
    for name in TOKENIZER_FILES:
        path = resolved / name
        if path.is_file():
            tokenizer_rows.append(
                {"path": name, "sha256": sha256_file(path).removeprefix("sha256:")}
            )
        elif require_base_sidecars:
            raise ValueError(f"HF export is missing tokenizer file {name}")
    tokenizer_sha256 = digest_json(sorted(tokenizer_rows, key=lambda row: row["path"]))
    tokenizer_matches = tokenizer_sha256 == expected_tokenizer_manifest_sha256
    if require_base_sidecars and not tokenizer_matches:
        raise ValueError("HF export tokenizer manifest differs from the base checkpoint")
    chat_sha256 = sha256_file(resolved / "chat_template.jinja")
    chat_matches = chat_sha256 == expected_chat_template_sha256
    if require_base_sidecars and not chat_matches:
        raise ValueError("HF export chat template differs from the base checkpoint")
    config_sha256 = sha256_file(resolved / "config.json")
    config_matches = config_sha256 == expected_config_sha256
    if require_base_sidecars and not config_matches:
        raise ValueError("HF export model config differs from the base checkpoint")

    observed_sidecars: dict[str, str] = {}
    for path in sorted(item for item in resolved.iterdir() if item.is_file()):
        if path.name.endswith(".safetensors") or path.name == "model.safetensors.index.json":
            continue
        observed_sidecars[path.name] = sha256_file(path)
    sidecar_matches: dict[str, bool] = {}
    if expected_sidecar_sha256:
        sidecar_matches = {
            name: observed_sidecars.get(name) == digest
            for name, digest in sorted(expected_sidecar_sha256.items())
        }
        if require_base_sidecars and not all(sidecar_matches.values()):
            raise ValueError("HF export runtime sidecars differ from the base checkpoint")

    weight_hashes = {row["path"]: row["sha256"] for row in weight_rows}
    file_rows = []
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        relative = path.relative_to(resolved).as_posix()
        file_rows.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": weight_hashes.get(relative) or sha256_file(path).removeprefix("sha256:"),
            }
        )
    return {
        "schema": "cyber_sft_hf_output_inspection_v1",
        "root": str(resolved),
        "format": "safetensors",
        "dtype": normalized_expected_dtype.lower(),
        "shard_count": len(shard_names),
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "weights_manifest_sha256": digest_json(weight_rows),
        "files_manifest_sha256": digest_json(file_rows),
        "tokenizer_manifest_sha256": tokenizer_sha256,
        "tokenizer_manifest_matches_base": tokenizer_matches,
        "chat_template_sha256": chat_sha256,
        "chat_template_matches_base": chat_matches,
        "config_sha256": config_sha256,
        "config_matches_base": config_matches,
        "sidecar_sha256": observed_sidecars,
        "sidecar_matches_base": sidecar_matches,
        "base_sidecars_required": require_base_sidecars,
        "all_shards_present": True,
        "safetensors_load_passed": True,
        "parameter_count_matches": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("structural", "full"):
        child = subparsers.add_parser(command)
        child.add_argument("root", type=Path)
        child.add_argument("--output", type=Path, required=True)
    hf = subparsers.add_parser("hf")
    hf.add_argument("root", type=Path)
    hf.add_argument("--plan", type=Path, required=True)
    hf.add_argument("--output", type=Path, required=True)
    hf.add_argument("--allow-sidecar-drift", action="store_true")
    hf.add_argument("--expected-dtype", choices=("BF16", "F32"), default="BF16")
    args = parser.parse_args()
    if args.command == "structural":
        result = structural_manifest(args.root)
    elif args.command == "full":
        result = full_file_manifest(args.root)
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        model = plan["base_model"]
        result = inspect_hf_export(
            args.root,
            expected_tokenizer_manifest_sha256=str(model["tokenizer_manifest_sha256"]),
            expected_chat_template_sha256=str(model["chat_template_sha256"]),
            expected_config_sha256=str(model["config_sha256"]),
            expected_parameter_count=int(model["parameter_count"]),
            expected_sidecar_sha256=model.get("runtime_sidecar_sha256"),
            require_base_sidecars=not args.allow_sidecar_drift,
            expected_dtype=args.expected_dtype,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
