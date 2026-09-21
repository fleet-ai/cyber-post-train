"""Exact-byte package builder for the non-root LoRA CPU preflight."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .lora_cpu_preflight_driver import (
    BUNDLE_FILES,
    ENV_BUNDLE_SHA256,
    ENV_SFS_CONTROL_MOUNT,
    ENV_SFS_OUTPUT_ROOT,
    ENV_SFS_OWNED_ROOT,
    ENV_SOURCE_ARCHIVE,
    ENV_SOURCE_ARCHIVE_SHA256,
    ENV_SOURCE_COMMIT,
    bundle_sha256,
)
from .source_bundle import canonical_source_commit_bytes, verify_source_archive_commit

NAMESPACE = "fleet-train-jobs"
CPU_SFS_OWNED_ROOT_ANNOTATION = "cyber-post-train.fleet.ai/sfs-owned-root"
CPU_SFS_OUTPUT_ROOT_ANNOTATION = "cyber-post-train.fleet.ai/sfs-output-root"
CPU_SFS_MEMORY_FLOOR_MIB_ANNOTATION = "cyber-post-train.fleet.ai/memory-floor-mib"
CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION = "cyber-post-train.fleet.ai/preflight-bundle-sha256"
CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION = "cyber-post-train.fleet.ai/source-archive-sha256"
CPU_SOURCE_COMMIT_ANNOTATION = "cyber-post-train.fleet.ai/source-commit"
CPU_PREFLIGHT_BINDING_ANNOTATIONS = (
    CPU_SFS_OWNED_ROOT_ANNOTATION,
    CPU_SFS_OUTPUT_ROOT_ANNOTATION,
    CPU_SFS_MEMORY_FLOOR_MIB_ANNOTATION,
    CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION,
    CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION,
    CPU_SOURCE_COMMIT_ANNOTATION,
)
CPU_PREFLIGHT_ENV = (
    ENV_SFS_OWNED_ROOT,
    ENV_SFS_OUTPUT_ROOT,
    ENV_SFS_CONTROL_MOUNT,
    ENV_BUNDLE_SHA256,
    ENV_SOURCE_ARCHIVE,
    ENV_SOURCE_ARCHIVE_SHA256,
    ENV_SOURCE_COMMIT,
)
_SOURCE_ROOT = Path(__file__).resolve().parent
_TRACKED_SOURCE = {
    "preflight_driver.py": _SOURCE_ROOT / "lora_cpu_preflight_driver.py",
    "sfs_write_identity.py": _SOURCE_ROOT / "sfs_write_identity.py",
    "source_bundle.py": _SOURCE_ROOT / "source_bundle.py",
}


@dataclass(frozen=True)
class LoraCpuPreflightPackage:
    """The exact immutable ConfigMap, Pod, and local archive evidence."""

    config_map: dict[str, Any]
    pod: dict[str, Any]
    source_archive: Path
    source_commit: str


def tracked_preflight_bundle() -> tuple[dict[str, str], str]:
    """Load the exact repository bytes that will execute in the pinned image."""

    raw = {name: path.read_bytes() for name, path in _TRACKED_SOURCE.items()}
    if set(raw) != set(BUNDLE_FILES):
        raise ValueError("tracked preflight bundle source set is incomplete")
    try:
        data = {name: raw[name].decode("utf-8") for name in BUNDLE_FILES}
    except UnicodeDecodeError:
        raise ValueError("tracked preflight bundle must contain UTF-8 text") from None
    if any(data[name].encode() != raw[name] for name in BUNDLE_FILES):
        raise ValueError("tracked preflight bundle text is not byte-stable")
    return data, bundle_sha256(raw)


def _literal_environment(container: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    entries = container.get("env", [])
    if not isinstance(entries, list):
        raise ValueError("preflight environment is not a list")
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"name", "value"}
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("value"), str)
            or entry["name"] in result
        ):
            raise ValueError("preflight environment must contain literal unique values")
        result[entry["name"]] = entry["value"]
    return result


def _runtime_archive_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("runtime source archive path is required")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to("/mnt/sfs"):
        raise ValueError("runtime source archive must be an absolute read-only SFS path")
    return str(path)


def build_lora_cpu_preflight_package(
    pod: dict,
    *,
    source_archive: Path,
    runtime_source_archive: str,
    source_commit: str,
) -> LoraCpuPreflightPackage:
    """Bind canonical archive bytes and tracked executable bytes to one Pod."""

    local_archive = Path(source_archive)
    source = verify_source_archive_commit(local_archive, source_commit)
    runtime_archive = _runtime_archive_path(runtime_source_archive)
    data, package_sha256 = tracked_preflight_bundle()
    packaged_pod = deepcopy(pod)
    try:
        metadata = packaged_pod["metadata"]
        annotations = metadata["annotations"]
        containers = packaged_pod["spec"]["containers"]
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed LoRA CPU preflight Pod") from exc
    if metadata.get("namespace") != NAMESPACE or not isinstance(metadata.get("name"), str):
        raise ValueError("LoRA CPU preflight Pod namespace/name drift")
    if (
        not isinstance(annotations, dict)
        or not isinstance(containers, list)
        or len(containers) != 1
    ):
        raise ValueError("LoRA CPU preflight Pod shape is invalid")
    if any(name in annotations for name in CPU_PREFLIGHT_BINDING_ANNOTATIONS[3:]):
        raise ValueError("LoRA CPU preflight Pod already contains a source binding")
    environment = _literal_environment(containers[0])
    if set(environment) != {
        ENV_SFS_OWNED_ROOT,
        ENV_SFS_OUTPUT_ROOT,
        ENV_SFS_CONTROL_MOUNT,
    }:
        raise ValueError("LoRA CPU preflight base environment is not the reviewed SFS binding")

    annotations.update(
        {
            CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION: package_sha256,
            CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION: source["source_archive_sha256"],
            CPU_SOURCE_COMMIT_ANNOTATION: source_commit,
        }
    )
    environment.update(
        {
            ENV_BUNDLE_SHA256: package_sha256,
            ENV_SOURCE_ARCHIVE: runtime_archive,
            ENV_SOURCE_ARCHIVE_SHA256: source["source_archive_sha256"],
            ENV_SOURCE_COMMIT: source_commit,
        }
    )
    containers[0]["env"] = [
        {"name": name, "value": environment[name]} for name in CPU_PREFLIGHT_ENV
    ]
    config_map_name = metadata["name"] + "-source"
    config_map = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": config_map_name,
            "namespace": NAMESPACE,
            "annotations": {
                CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION: package_sha256,
                CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION: source["source_archive_sha256"],
                CPU_SOURCE_COMMIT_ANNOTATION: source_commit,
            },
        },
        "immutable": True,
        "data": data,
    }
    package = LoraCpuPreflightPackage(
        config_map=config_map,
        pod=packaged_pod,
        source_archive=local_archive,
        source_commit=source_commit,
    )
    validate_lora_cpu_preflight_package(package)
    return package


def validate_lora_cpu_preflight_package(package: LoraCpuPreflightPackage) -> dict[str, str]:
    """Reopen all local inputs and reject any drift before an API call."""

    if not isinstance(package, LoraCpuPreflightPackage):
        raise ValueError("LoRA CPU preflight requires one reviewed package")
    canonical_source_commit_bytes(package.source_commit)
    source = verify_source_archive_commit(package.source_archive, package.source_commit)
    data, package_sha256 = tracked_preflight_bundle()
    pod = package.pod
    config_map = package.config_map
    try:
        pod_meta = pod["metadata"]
        pod_annotations = pod_meta["annotations"]
        container = pod["spec"]["containers"][0]
        config_meta = config_map["metadata"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("malformed LoRA CPU preflight package") from exc
    config_map_name = pod_meta["name"] + "-source"
    expected_annotations = {
        CPU_PREFLIGHT_BUNDLE_SHA256_ANNOTATION: package_sha256,
        CPU_SOURCE_ARCHIVE_SHA256_ANNOTATION: source["source_archive_sha256"],
        CPU_SOURCE_COMMIT_ANNOTATION: package.source_commit,
    }
    if (
        set(config_map) != {"apiVersion", "kind", "metadata", "immutable", "data"}
        or config_map.get("apiVersion") != "v1"
        or config_map.get("kind") != "ConfigMap"
        or config_map.get("immutable") is not True
        or config_map.get("data") != data
        or config_meta
        != {
            "name": config_map_name,
            "namespace": NAMESPACE,
            "annotations": expected_annotations,
        }
    ):
        raise ValueError("immutable preflight ConfigMap differs from tracked packaged bytes")
    if any(pod_annotations.get(name) != value for name, value in expected_annotations.items()):
        raise ValueError("LoRA CPU preflight Pod source annotations differ from its package")
    environment = _literal_environment(container)
    if (
        set(environment) != set(CPU_PREFLIGHT_ENV)
        or environment[ENV_BUNDLE_SHA256] != package_sha256
        or environment[ENV_SOURCE_ARCHIVE_SHA256] != source["source_archive_sha256"]
        or environment[ENV_SOURCE_COMMIT] != package.source_commit
        or _runtime_archive_path(environment[ENV_SOURCE_ARCHIVE]) != environment[ENV_SOURCE_ARCHIVE]
    ):
        raise ValueError("LoRA CPU preflight runtime environment differs from its package")
    volumes = pod["spec"].get("volumes", [])
    bundle_volumes = [
        item for item in volumes if isinstance(item, dict) and item.get("name") == "bundle"
    ]
    if bundle_volumes != [
        {
            "name": "bundle",
            "configMap": {"name": config_map_name, "defaultMode": 292},
        }
    ]:
        raise ValueError("LoRA CPU preflight Pod does not mount its exact immutable ConfigMap")
    return {
        "preflight_bundle_sha256": package_sha256,
        "source_archive_sha256": source["source_archive_sha256"],
        "source_commit": package.source_commit,
        "config_map_name": config_map_name,
    }


def validate_lora_cpu_preflight_config_map_response(
    actual: dict, package: LoraCpuPreflightPackage
) -> None:
    """Bind a server dry-run/create response to the exact packaged ConfigMap bytes."""

    expected = package.config_map
    try:
        actual_meta = actual["metadata"]
        expected_meta = expected["metadata"]
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed preflight ConfigMap response") from exc
    if (
        actual.get("apiVersion") != "v1"
        or actual.get("kind") != "ConfigMap"
        or actual.get("immutable") is not True
        or actual.get("data") != expected["data"]
        or actual_meta.get("name") != expected_meta["name"]
        or actual_meta.get("namespace") != expected_meta["namespace"]
        or any(
            actual_meta.get("annotations", {}).get(name) != value
            for name, value in expected_meta["annotations"].items()
        )
    ):
        raise ValueError("server ConfigMap response differs from exact preflight package")
