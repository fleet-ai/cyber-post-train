"""Strict shared validation for the frozen-base inference artifact surface."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .io import digest_json

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RAW_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SURFACE_KEYS = {
    "schema",
    "policy",
    "weight_shard_count",
    "weights_manifest_sha256",
    "index",
    "required_runtime_sidecar_sha256",
    "allowed_non_artifact_top_level_files",
    "excluded_non_artifact_directory_prefixes",
    "unknown_top_level_entries",
    "symlinks",
}


def _top_level_name(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"{field} must be one safe top-level name")
    return value


def validate_base_inference_artifact_manifest(
    manifest_value: Any,
    surface_value: Any,
    *,
    expected_root: str,
) -> dict[str, Any]:
    """Reconstruct and validate the exact frozen-base inference artifact row set."""

    if not isinstance(surface_value, Mapping) or set(surface_value) != SURFACE_KEYS:
        raise ValueError("base inference artifact surface fields differ")
    surface = dict(surface_value)
    if (
        surface.get("schema") != "cyber_sft_base_inference_artifact_surface_v1"
        or surface.get("policy")
        != "exact_top_level_inference_artifacts_with_reviewed_control_exclusions_v1"
        or surface.get("unknown_top_level_entries") != "reject"
        or surface.get("symlinks") != "reject"
    ):
        raise ValueError("base inference artifact surface policy differs")
    shard_count = surface.get("weight_shard_count")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("base inference artifact shard count is invalid")
    weights_digest = surface.get("weights_manifest_sha256")
    if not isinstance(weights_digest, str) or not SHA256_RE.fullmatch(weights_digest):
        raise ValueError("base inference artifact weight digest is invalid")
    index = surface.get("index")
    if not isinstance(index, Mapping) or set(index) != {"path", "sha256"}:
        raise ValueError("base inference artifact index binding differs")
    index_name = _top_level_name(index.get("path"), "base inference index path")
    index_digest = index.get("sha256")
    if (
        index_name != "model.safetensors.index.json"
        or not isinstance(index_digest, str)
        or not SHA256_RE.fullmatch(index_digest)
    ):
        raise ValueError("base inference artifact index binding is invalid")

    sidecars = surface.get("required_runtime_sidecar_sha256")
    controls = surface.get("allowed_non_artifact_top_level_files")
    excluded = surface.get("excluded_non_artifact_directory_prefixes")
    if not isinstance(sidecars, Mapping) or not sidecars:
        raise ValueError("base inference artifact sidecars are invalid")
    if not isinstance(controls, Mapping) or not isinstance(excluded, Mapping):
        raise ValueError("base inference artifact exclusions are invalid")
    for name, digest in sidecars.items():
        _top_level_name(name, "base inference sidecar path")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError("base inference sidecar digest is invalid")
    for name, reason in controls.items():
        _top_level_name(name, "base non-artifact control path")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("base non-artifact control reason is invalid")
    for prefix, reason in excluded.items():
        if not isinstance(prefix, str) or not prefix.endswith("/"):
            raise ValueError("base excluded directory prefix is invalid")
        _top_level_name(prefix[:-1], "base excluded directory prefix")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("base excluded directory reason is invalid")

    shard_names = [
        f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        for number in range(1, shard_count + 1)
    ]
    required_names = set(shard_names) | set(sidecars) | {index_name}
    excluded_names = set(controls) | {prefix[:-1] for prefix in excluded}
    if required_names & excluded_names or len(excluded_names) != len(controls) + len(
        excluded
    ):
        raise ValueError("base inference artifacts and exclusions overlap")

    if not isinstance(manifest_value, Mapping):
        raise ValueError("base inference artifact manifest must be an object")
    manifest = dict(manifest_value)
    expected_manifest_keys = {
        "schema",
        "root",
        "surface_sha256",
        "file_count",
        "total_bytes",
        "files",
        "manifest_sha256",
        "weights_manifest_sha256",
        "runtime_sidecar_sha256",
        "excluded_non_artifact_files",
        "excluded_non_artifact_directory_prefixes",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("base inference artifact manifest fields differ")
    if (
        manifest.get("schema") != "cyber_sft_base_inference_artifact_manifest_v1"
        or manifest.get("root") != expected_root
        or manifest.get("surface_sha256") != digest_json(surface)
    ):
        raise ValueError("base inference artifact manifest identity differs")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("base inference artifact manifest files must be a list")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "size", "sha256"}:
            raise ValueError("base inference artifact file row fields differ")
        path = _top_level_name(row.get("path"), "base inference artifact path")
        size = row.get("size")
        digest = row.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("base inference artifact file size is invalid")
        if not isinstance(digest, str) or not RAW_SHA256_RE.fullmatch(digest):
            raise ValueError("base inference artifact file digest is invalid")
        normalized_rows.append({"path": path, "size": size, "sha256": digest})
    if normalized_rows != sorted(normalized_rows, key=lambda row: row["path"]):
        raise ValueError("base inference artifact rows are not canonically ordered")
    observed_paths = [row["path"] for row in normalized_rows]
    if len(set(observed_paths)) != len(observed_paths) or set(observed_paths) != required_names:
        raise ValueError("base inference artifact row paths differ from the exact surface")
    rows_by_path = {row["path"]: row for row in normalized_rows}
    if rows_by_path[index_name]["sha256"] != index_digest.removeprefix("sha256:"):
        raise ValueError("base inference artifact index row hash differs")
    for name, digest in sidecars.items():
        if rows_by_path[name]["sha256"] != digest.removeprefix("sha256:"):
            raise ValueError(f"base inference artifact sidecar row hash differs: {name}")
    weight_rows = [rows_by_path[name] for name in shard_names]
    expected_excluded_files = [
        {"path": name, "reviewed_reason": controls[name]} for name in sorted(controls)
    ]
    expected_excluded_prefixes = [
        {"path_prefix": prefix, "reviewed_reason": excluded[prefix]}
        for prefix in sorted(excluded)
    ]
    if (
        manifest.get("file_count") != len(normalized_rows)
        or manifest.get("total_bytes") != sum(row["size"] for row in normalized_rows)
        or manifest.get("manifest_sha256") != digest_json(normalized_rows)
        or manifest.get("weights_manifest_sha256") != digest_json(weight_rows)
        or manifest.get("weights_manifest_sha256") != weights_digest
        or manifest.get("runtime_sidecar_sha256") != dict(sidecars)
        or manifest.get("excluded_non_artifact_files") != expected_excluded_files
        or manifest.get("excluded_non_artifact_directory_prefixes")
        != expected_excluded_prefixes
    ):
        raise ValueError("base inference artifact manifest summary or exclusions differ")
    return manifest
