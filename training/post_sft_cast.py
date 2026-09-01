"""Deterministic, bounded-memory FP32 to BF16 conversion for the final SFT export."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import ssl
import sys
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .io import digest_json, file_sha256
from .post_sft_artifacts import (
    _safetensor_layout,
    compare_safetensor_layout,
    compare_safetensor_layout_with_exact_auxiliary_omission,
    full_file_manifest,
    inspect_hf_export,
)
from .post_sft_base_surface import validate_base_inference_artifact_manifest

CAST_INPUT_SCHEMA = "cyber_sft_fp32_to_bf16_cast_input_v2"
CAST_RECEIPT_SCHEMA = "cyber_sft_fp32_to_bf16_cast_receipt_v2"
CAST_POLICY = "deterministic_trained_fp32_to_bf16_plus_frozen_base_mtp_restore_v1"
SOURCE_PATH = Path(
    "/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-v1/global_step_318/policy"
)
DESTINATION_PATH = Path(
    "/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-bf16-v4/global_step_318/policy"
)
BASE_MODEL_PATH = Path(
    "/mnt/sfs/models/Qwen/Qwen3.6-27B/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
BASE_MODEL_REPOSITORY = "Qwen/Qwen3.6-27B"
BASE_MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
BASE_WEIGHTS_MANIFEST_SHA256 = (
    "sha256:14ad10368de9b9e5974ff12a4b70ea7884194b58e670177bbac79daeb81f16b9"
)
BASE_INDEX_SHA256 = "sha256:a8ad2c26fb707ff8c245806315b03e3b4b74595528492423af5dae0ce39b4d9b"
BASE_RUNTIME_SIDECAR_SHA256 = {
    "chat_template.jinja": (
        "sha256:e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
    "config.json": "sha256:69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c",
    "configuration.json": "sha256:2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0",
    "generation_config.json": (
        "sha256:e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "merges.txt": "sha256:a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "preprocessor_config.json": (
        "sha256:27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "tokenizer.json": "sha256:5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": (
        "sha256:5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "video_preprocessor_config.json": (
        "sha256:7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
    "vocab.json": "sha256:ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}
BASE_NON_ARTIFACT_TOP_LEVEL_FILES = {
    ".cyber-post-train-lock.json": "checkpoint download provenance; not loaded by inference",
    ".gitattributes": "Hugging Face repository metadata; not loaded by inference",
    "LICENSE": "license text; not loaded by inference",
    "README.md": "model documentation; not loaded by inference",
    "source-tree.json": "checkpoint materialization provenance; not loaded by inference",
}
BASE_EXCLUDED_DIRECTORY_PREFIXES = {
    ".cache/": "Hugging Face client cache/control metadata; not loaded by inference"
}
BASE_WEIGHT_SHARD_COUNT = 15
EVIDENCE_DIR = Path(
    "/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/evidence/bf16-cast-v4/receipt"
)
ACCEPTANCE_RECEIPT_NAME = ".fleet-bf16-cast-acceptance.json"
DEFAULT_MAX_SHARD_BYTES = 2 * 1024**3
DEFAULT_MAX_SOURCE_TENSOR_BYTES = 8 * 1024**3
HASH_CHUNK_BYTES = 8 * 1024**2
TRAINED_TENSOR_COUNT = 1184
TRAINED_PARAMETER_COUNT = 27_356_728_560
RESTORED_AUXILIARY_TENSOR_COUNT = 15
RESTORED_AUXILIARY_PARAMETER_COUNT = 424_699_392
FINAL_TENSOR_COUNT = 1199
FINAL_PARAMETER_COUNT = 27_781_427_952
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-cyber-qwen36-sft-bf16-cast-v4"
CONFIG_MAP_NAME = JOB_NAME
SERVICE_ACCOUNT_NAME = "chris-cyber-qwen36-sft-bf16-cast-observer-v4"
CONTAINER_NAME = "cast"
IMAGE_ID_MAX_ATTEMPTS = 12
IMAGE_ID_RETRY_SECONDS = 1.0
CAST_IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train:"
    "q36-torchgdn-6db8d0c9@sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
)
CAST_COMMAND = {
    "command": ["python", "-m", "training.post_sft_cast", "execute"],
    "args": ["--input", "/bundle/cast-input.json"],
}
CAST_COMMAND_SHA256 = digest_json(CAST_COMMAND)
BUNDLE_ROOT = Path("/bundle")
CAST_INPUT_MOUNT_PATH = "cast-input.json"
MOUNTED_CODE_FILES = {
    "training__init__.py": "training/__init__.py",
    "training_io.py": "training/io.py",
    "training_post_sft_artifacts.py": "training/post_sft_artifacts.py",
    "training_post_sft_base_surface.py": "training/post_sft_base_surface.py",
    "training_post_sft_cast.py": "training/post_sft_cast.py",
}
LOCAL_CODE_FILES = {
    **MOUNTED_CODE_FILES,
    "training__init__.py": "evals/post_sft/runtime/training__init__.py",
}
IMAGE_ID_RE = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*://)?(?:[^@\s]+@)?(sha256:[0-9a-f]{64})$"
)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _validate_digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return value


def _expected_base_inference_artifact_surface() -> dict[str, Any]:
    return {
        "schema": "cyber_sft_base_inference_artifact_surface_v1",
        "policy": "exact_top_level_inference_artifacts_with_reviewed_control_exclusions_v1",
        "weight_shard_count": BASE_WEIGHT_SHARD_COUNT,
        "weights_manifest_sha256": BASE_WEIGHTS_MANIFEST_SHA256,
        "index": {
            "path": "model.safetensors.index.json",
            "sha256": BASE_INDEX_SHA256,
        },
        "required_runtime_sidecar_sha256": dict(BASE_RUNTIME_SIDECAR_SHA256),
        "allowed_non_artifact_top_level_files": dict(
            BASE_NON_ARTIFACT_TOP_LEVEL_FILES
        ),
        "excluded_non_artifact_directory_prefixes": dict(
            BASE_EXCLUDED_DIRECTORY_PREFIXES
        ),
        "unknown_top_level_entries": "reject",
        "symlinks": "reject",
    }


def _validate_base_inference_artifact_surface(value: Any) -> Mapping[str, Any]:
    surface = _mapping(value, "base inference artifact surface")
    if dict(surface) != _expected_base_inference_artifact_surface():
        raise ValueError("base inference artifact surface differs from the frozen plan")
    return surface


def _base_inference_artifact_manifest(
    root: Path, surface_value: Mapping[str, Any]
) -> dict[str, Any]:
    """Hash only the exact serving surface; reject rather than ignore unknown artifacts."""

    surface = _validate_base_inference_artifact_surface(surface_value)
    if root.is_symlink():
        raise ValueError("frozen base root is not a regular directory")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("frozen base root is not a regular directory")
    index_binding = _mapping(surface.get("index"), "base artifact index")
    index_name = str(index_binding.get("path"))
    index_path = resolved / index_name
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError("frozen base index is missing or not a regular file")
    if file_sha256(index_path) != index_binding.get("sha256"):
        raise ValueError("frozen base index differs from the signed model lock")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen base index is unreadable or invalid") from exc
    weight_map = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("frozen base index has no weight map")
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in weight_map.items()
    ):
        raise ValueError("frozen base index has invalid weight-map entries")
    shard_names = sorted(set(weight_map.values()))
    expected_shards = [
        f"model-{number:05d}-of-{BASE_WEIGHT_SHARD_COUNT:05d}.safetensors"
        for number in range(1, BASE_WEIGHT_SHARD_COUNT + 1)
    ]
    if shard_names != expected_shards:
        raise ValueError("frozen base index names an unexpected shard set")

    sidecars = _mapping(
        surface.get("required_runtime_sidecar_sha256"),
        "base runtime sidecar hashes",
    )
    controls = _mapping(
        surface.get("allowed_non_artifact_top_level_files"),
        "base non-artifact files",
    )
    excluded_directories = _mapping(
        surface.get("excluded_non_artifact_directory_prefixes"),
        "base excluded directory prefixes",
    )
    required_names = set(shard_names) | set(sidecars) | {index_name}
    allowed_names = required_names | set(controls) | {
        prefix.removesuffix("/") for prefix in excluded_directories
    }
    excluded_files = []
    excluded_prefixes = []
    observed_names = set()
    for path in sorted(resolved.iterdir(), key=lambda item: item.name):
        observed_names.add(path.name)
        if path.is_symlink():
            raise ValueError(f"frozen base contains prohibited symlink {path.name}")
        if path.name not in allowed_names:
            raise ValueError(f"frozen base contains unknown top-level entry {path.name}")
        directory_prefix = f"{path.name}/"
        if directory_prefix in excluded_directories:
            if not path.is_dir():
                raise ValueError(f"excluded base prefix is not a directory: {directory_prefix}")
            excluded_prefixes.append(
                {
                    "path_prefix": directory_prefix,
                    "reviewed_reason": excluded_directories[directory_prefix],
                }
            )
        elif path.name in controls:
            if not path.is_file():
                raise ValueError(f"excluded base control is not a file: {path.name}")
            excluded_files.append(
                {"path": path.name, "reviewed_reason": controls[path.name]}
            )
        elif not path.is_file():
            raise ValueError(f"required base artifact is not a file: {path.name}")
    missing = sorted(allowed_names - observed_names)
    if missing:
        raise ValueError(f"frozen base is missing required artifacts: {missing}")

    rows = []
    for name in sorted(required_names):
        path = resolved / name
        digest = file_sha256(path)
        expected = (
            index_binding.get("sha256")
            if name == index_name
            else sidecars.get(name)
        )
        if expected is not None and digest != expected:
            raise ValueError(f"frozen base required artifact hash differs: {name}")
        rows.append(
            {
                "path": name,
                "size": path.stat().st_size,
                "sha256": digest.removeprefix("sha256:"),
            }
        )
    weight_rows = [row for row in rows if row["path"] in set(shard_names)]
    if digest_json(weight_rows) != surface.get("weights_manifest_sha256"):
        raise ValueError("frozen base weight bytes differ from the signed model lock")
    manifest = {
        "schema": "cyber_sft_base_inference_artifact_manifest_v1",
        "root": str(resolved),
        "surface_sha256": digest_json(surface),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
        "weights_manifest_sha256": digest_json(weight_rows),
        "runtime_sidecar_sha256": dict(sidecars),
        "excluded_non_artifact_files": excluded_files,
        "excluded_non_artifact_directory_prefixes": excluded_prefixes,
    }
    return validate_base_inference_artifact_manifest(
        manifest,
        surface,
        expected_root=str(resolved),
    )


def canonical_cast_input_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _kubernetes_get(path: str) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "").strip()
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443").strip()
    if not host:
        raise ValueError("KUBERNETES_SERVICE_HOST is required for cast provenance")
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    token = token_path.read_text(encoding="utf-8").strip()
    request = urllib.request.Request(
        f"https://{host}:{port}{path}", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(
        request, timeout=30, context=ssl.create_default_context(cafile=str(ca_path))
    ) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("Kubernetes API returned a non-object")
    return value


def _pod_with_resolved_image_id(
    pod: dict[str, Any], pod_path: str
) -> tuple[dict[str, Any], str, int]:
    """Wait briefly only for Kubernetes to publish the already-running image identity."""

    expected = CAST_IMAGE.rsplit("@", 1)[1]
    for attempt in range(1, IMAGE_ID_MAX_ATTEMPTS + 1):
        status_value = pod.get("status")
        statuses = (
            status_value.get("containerStatuses")
            if isinstance(status_value, Mapping)
            else None
        )
        status = (
            next(
                (
                    row
                    for row in statuses
                    if isinstance(row, Mapping) and row.get("name") == CONTAINER_NAME
                ),
                None,
            )
            if isinstance(statuses, list)
            else None
        )
        raw_image_id = status.get("imageID") if status is not None else None
        if raw_image_id is None:
            image_id = ""
        elif isinstance(raw_image_id, str):
            image_id = raw_image_id.strip()
        else:
            raise ValueError("resolved cast imageID differs from the frozen digest")
        if image_id:
            match = IMAGE_ID_RE.fullmatch(image_id)
            resolved = match.group(1) if match is not None else None
            if resolved != expected:
                raise ValueError("resolved cast imageID differs from the frozen digest")
            return pod, image_id, attempt
        if attempt == IMAGE_ID_MAX_ATTEMPTS:
            break
        time.sleep(IMAGE_ID_RETRY_SECONDS)
        pod = _kubernetes_get(pod_path)
    raise ValueError("resolved cast imageID remained empty after bounded retry")


def _runtime_execution_provenance(cast_input: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the cast to exact live Job, Pod, imageID, ConfigMap, and mounted bytes."""

    plan = _mapping(cast_input.get("execution"), "cast execution plan")
    namespace = os.environ.get("POD_NAMESPACE", "").strip()
    pod_name = os.environ.get("POD_NAME", "").strip()
    pod_uid = os.environ.get("POD_UID", "").strip()
    job_uid = os.environ.get("JOB_UID", "").strip()
    if not all((namespace, pod_name, pod_uid, job_uid)) or namespace != NAMESPACE:
        raise ValueError("cast downward-API identity is missing or unexpected")
    pod_path = f"/api/v1/namespaces/{namespace}/pods/{pod_name}"
    pod = _kubernetes_get(pod_path)
    job = _kubernetes_get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{JOB_NAME}")
    config_map = _kubernetes_get(
        f"/api/v1/namespaces/{namespace}/configmaps/{CONFIG_MAP_NAME}"
    )
    pod, image_id, image_id_observation_attempts = _pod_with_resolved_image_id(
        pod, pod_path
    )
    pod_meta = _mapping(pod.get("metadata"), "Pod metadata")
    job_meta = _mapping(job.get("metadata"), "Job metadata")
    config_meta = _mapping(config_map.get("metadata"), "ConfigMap metadata")
    if (pod_meta.get("name"), pod_meta.get("uid")) != (pod_name, pod_uid):
        raise ValueError("live cast Pod differs from downward-API identity")
    owners = pod_meta.get("ownerReferences")
    if not isinstance(owners, list) or not any(
        isinstance(owner, Mapping)
        and owner.get("kind") == "Job"
        and owner.get("name") == JOB_NAME
        and owner.get("uid") == job_uid
        for owner in owners
    ):
        raise ValueError("live cast Pod is not owned by the expected Job UID")
    if (job_meta.get("name"), job_meta.get("uid")) != (JOB_NAME, job_uid):
        raise ValueError("live cast Job differs from controller identity")
    if _mapping(job_meta.get("labels"), "Job labels").get(
        "kueue.x-k8s.io/queue-name"
    ) != "training-lq":
        raise ValueError("live cast Job is not attached to training-lq")
    pod_spec = _mapping(pod.get("spec"), "Pod spec")
    if pod_spec.get("nodeSelector") != {"workload": "fleetai-training-ng-cpu"}:
        raise ValueError("live cast Pod is not pinned to the reviewed CPU pool")
    containers = pod_spec.get("containers")
    if pod_spec.get("serviceAccountName") != SERVICE_ACCOUNT_NAME or not isinstance(
        containers, list
    ):
        raise ValueError("live cast Pod service account or containers differ")
    container = next(
        (
            row
            for row in containers
            if isinstance(row, Mapping) and row.get("name") == CONTAINER_NAME
        ),
        None,
    )
    if container is None or container.get("image") != CAST_IMAGE:
        raise ValueError("live cast container image differs from the frozen image")
    if container.get("resources") != {
        "requests": {"cpu": "4", "memory": "32Gi"},
        "limits": {"cpu": "8", "memory": "64Gi"},
    }:
        raise ValueError("live cast container resources differ from the reviewed CPU shape")
    mounts = container.get("volumeMounts")
    if not isinstance(mounts, list) or not all(
        expected in mounts
        for expected in (
            {"name": "bundle", "mountPath": "/bundle", "readOnly": True},
            {"name": "sfs", "mountPath": "/mnt/sfs"},
        )
    ):
        raise ValueError("live cast container mounts differ from the reviewed SFS transaction")
    command = {
        "command": list(container.get("command") or []),
        "args": list(container.get("args") or []),
    }
    if digest_json(command) != CAST_COMMAND_SHA256:
        raise ValueError("live cast command differs from the frozen command")
    expected_image_digest = CAST_IMAGE.rsplit("@", 1)[1]
    if (
        config_meta.get("name") != CONFIG_MAP_NAME
        or config_map.get("immutable") is not True
        or not config_meta.get("uid")
        or not config_meta.get("resourceVersion")
    ):
        raise ValueError("live cast ConfigMap identity is incomplete or mutable")
    data = _mapping(config_map.get("data"), "ConfigMap data")
    if set(data) != set(MOUNTED_CODE_FILES) | {CAST_INPUT_MOUNT_PATH}:
        raise ValueError("live cast ConfigMap keys differ from the reviewed bundle")
    expected_code = _mapping(plan.get("config_map_code_sha256"), "reviewed cast code")
    mounted: dict[str, str] = {}
    for key, relative in MOUNTED_CODE_FILES.items():
        text = data.get(key)
        if not isinstance(text, str):
            raise ValueError(f"live cast ConfigMap is missing {key}")
        digest = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
        if digest != expected_code.get(relative) or file_sha256(BUNDLE_ROOT / relative) != digest:
            raise ValueError(f"live or mounted cast code differs for {relative}")
        mounted[relative] = digest
    expected_input_bytes = canonical_cast_input_bytes(cast_input)
    if data.get(CAST_INPUT_MOUNT_PATH, "").encode() != expected_input_bytes:
        raise ValueError("live ConfigMap cast input differs from executing bytes")
    if (BUNDLE_ROOT / CAST_INPUT_MOUNT_PATH).read_bytes() != expected_input_bytes:
        raise ValueError("mounted cast input differs from executing bytes")
    input_file_sha = "sha256:" + hashlib.sha256(expected_input_bytes).hexdigest()
    mounted[CAST_INPUT_MOUNT_PATH] = input_file_sha
    return {
        "schema": "cyber_sft_fp32_to_bf16_cast_execution_v1",
        "image": CAST_IMAGE,
        "image_id": image_id,
        "image_id_observation_attempts": image_id_observation_attempts,
        "resolved_image_digest": expected_image_digest,
        "command_sha256": CAST_COMMAND_SHA256,
        "service_account_name": SERVICE_ACCOUNT_NAME,
        "container_name": CONTAINER_NAME,
        "job": {
            "namespace": namespace,
            "name": JOB_NAME,
            "uid": job_uid,
            "resource_version": str(job_meta.get("resourceVersion")),
            "spec_sha256": digest_json(_mapping(job.get("spec"), "Job spec")),
        },
        "pod": {
            "namespace": namespace,
            "name": pod_name,
            "uid": pod_uid,
            "resource_version": str(pod_meta.get("resourceVersion")),
            "spec_sha256": digest_json(pod_spec),
        },
        "config_map": {
            "namespace": namespace,
            "name": CONFIG_MAP_NAME,
            "uid": str(config_meta["uid"]),
            "resource_version": str(config_meta["resourceVersion"]),
            "immutable": True,
            "mounted_file_sha256": mounted,
            "cast_input_file_sha256": input_file_sha,
        },
        "cast_input_sha256": cast_input["cast_input_sha256"],
    }


def _tensor_sha256(tensor: Any) -> str:
    """Hash tensor storage without constructing one model-sized Python bytes object."""

    byte_view = tensor.detach().contiguous().view(-1).view(__import__("torch").uint8)
    digest = hashlib.sha256()
    for start in range(0, byte_view.numel(), HASH_CHUNK_BYTES):
        chunk = byte_view[start : start + HASH_CHUNK_BYTES].numpy().tobytes()
        digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_execution_plan(value: Any) -> Mapping[str, Any]:
    plan = _mapping(value, "cast execution plan")
    expected = {
        "schema": "cyber_sft_fp32_to_bf16_cast_execution_plan_v4",
        "namespace": NAMESPACE,
        "job_name": JOB_NAME,
        "config_map_name": CONFIG_MAP_NAME,
        "service_account_name": SERVICE_ACCOUNT_NAME,
        "container_name": CONTAINER_NAME,
        "image": CAST_IMAGE,
        "image_digest": CAST_IMAGE.rsplit("@", 1)[1],
        "command_sha256": CAST_COMMAND_SHA256,
        "source_path": str(SOURCE_PATH),
        "destination_path": str(DESTINATION_PATH),
        "base_model_path": str(BASE_MODEL_PATH),
        "base_model_revision": BASE_MODEL_REVISION,
        "policy": CAST_POLICY,
        "source_dtype": "F32",
        "destination_dtype": "BF16",
        "max_shard_bytes": DEFAULT_MAX_SHARD_BYTES,
        "max_source_tensor_bytes": DEFAULT_MAX_SOURCE_TENSOR_BYTES,
        "trained_tensor_count": TRAINED_TENSOR_COUNT,
        "trained_parameter_count": TRAINED_PARAMETER_COUNT,
        "restored_auxiliary_tensor_count": RESTORED_AUXILIARY_TENSOR_COUNT,
        "restored_auxiliary_parameter_count": RESTORED_AUXILIARY_PARAMETER_COUNT,
        "final_tensor_count": FINAL_TENSOR_COUNT,
        "final_parameter_count": FINAL_PARAMETER_COUNT,
        "image_id_max_attempts": IMAGE_ID_MAX_ATTEMPTS,
        "image_id_retry_seconds": IMAGE_ID_RETRY_SECONDS,
        "base_inference_artifact_surface": _expected_base_inference_artifact_surface(),
    }
    for field, expected_value in expected.items():
        if plan.get(field) != expected_value:
            raise ValueError(f"cast execution plan {field} differs from the reviewed rail")
    code_hashes = _mapping(plan.get("config_map_code_sha256"), "cast code hashes")
    if set(code_hashes) != set(MOUNTED_CODE_FILES.values()):
        raise ValueError("cast execution plan code paths differ from the reviewed bundle")
    for path, digest in code_hashes.items():
        _validate_digest(digest, f"cast code hash for {path}")
    return plan


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically promote a complete directory without replacing any filesystem entry."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        rename = getattr(libc, "renameat2", None)
        if rename is None:
            raise RuntimeError("renameat2 is required for collision-free Linux promotion")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif sys.platform == "darwin":  # local test parity; production is Linux.
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("renamex_np is required for collision-free macOS promotion")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(os.fsencode(source), os.fsencode(destination), 0x00000004)
    else:
        raise RuntimeError(f"atomic no-replace promotion is unsupported on {sys.platform}")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, "cast destination already exists", destination)
    raise OSError(error, os.strerror(error), destination)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _copy_source_sidecars(source: Path, output: Path, shard_names: set[str]) -> dict[str, str]:
    """Preserve raw trainer sidecars as evidence; serving replaces them with frozen base files."""

    copied: dict[str, str] = {}
    excluded = shard_names | {"model.safetensors.index.json"}
    for path in sorted(source.iterdir(), key=lambda item: item.name):
        if path.name in excluded:
            continue
        if path.is_symlink() or not path.is_file() or PurePosixPath(path.name).name != path.name:
            raise ValueError(f"raw export contains unsupported non-regular sidecar {path.name}")
        shutil.copyfile(path, output / path.name)
        copied[path.name] = file_sha256(path)
        if file_sha256(output / path.name) != copied[path.name]:
            raise ValueError(f"copied raw sidecar differs for {path.name}")
        _fsync_file(output / path.name)
    return copied


def _shard_groups(
    layout: Mapping[str, Mapping[str, Any]], max_shard_bytes: int
) -> list[list[str]]:
    if max_shard_bytes < 2:
        raise ValueError("max_shard_bytes is too small")
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for key in sorted(layout):
        row = layout[key]
        tensor_bytes = math.prod(row["shape"]) * 2
        if tensor_bytes > max_shard_bytes:
            raise ValueError(f"BF16 tensor {key} exceeds the fixed shard bound")
        if current and current_bytes + tensor_bytes > max_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(key)
        current_bytes += tensor_bytes
    if current:
        groups.append(current)
    if not groups:
        raise ValueError("raw export contains no tensors")
    return groups


def _load_tensor(root: Path, shard_name: str, key: str) -> Any:
    from safetensors import safe_open

    with safe_open(str(root / shard_name), framework="pt", device="cpu") as shard:
        return shard.get_tensor(key)


def cast_fp32_export(
    source: Path,
    base: Path,
    output: Path,
    omission: Mapping[str, Any],
    *,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    max_source_tensor_bytes: int = DEFAULT_MAX_SOURCE_TENSOR_BYTES,
) -> dict[str, Any]:
    """Cast trained tensors and restore only exact frozen-base auxiliary heads."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = source.resolve(strict=True)
    base = base.resolve(strict=True)
    omission_layout = compare_safetensor_layout_with_exact_auxiliary_omission(
        base, source, omission
    )
    if os.path.lexists(output):
        raise FileExistsError(f"refusing pre-existing cast output: {output}")
    output.mkdir(mode=0o700, parents=False)
    layout, source_shards = _safetensor_layout(source)
    base_layout, _ = _safetensor_layout(base)
    missing_keys = set(base_layout) - set(layout)
    wrong_dtype = sorted(key for key, row in layout.items() if row["dtype"] != "F32")
    if wrong_dtype:
        raise ValueError(f"raw export is not uniformly F32 ({len(wrong_dtype)} tensors differ)")
    for key, row in layout.items():
        source_bytes = math.prod(row["shape"]) * 4
        if source_bytes > max_source_tensor_bytes:
            raise ValueError(f"source tensor {key} exceeds the fixed memory bound")
    for key in missing_keys:
        row = base_layout[key]
        if row["dtype"] != "BF16":
            raise ValueError(f"frozen base auxiliary tensor is not BF16: {key}")
        if math.prod(row["shape"]) * 2 > max_source_tensor_bytes:
            raise ValueError(f"frozen base auxiliary tensor {key} exceeds the memory bound")
    groups = _shard_groups(base_layout, max_shard_bytes)
    shard_count = len(groups)
    weight_map: dict[str, str] = {}
    cast_rows: list[dict[str, Any]] = []
    restoration_rows: list[dict[str, Any]] = []
    total_bf16_bytes = 0
    torch.set_num_threads(1)

    for shard_index, keys in enumerate(groups, start=1):
        output_name = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        tensors: dict[str, Any] = {}
        pending: dict[str, dict[str, Any]] = {}
        for key in keys:
            if key in layout:
                row = layout[key]
                source_tensor = _load_tensor(source, str(row["shard"]), key)
                if (
                    source_tensor.dtype != torch.float32
                    or list(source_tensor.shape) != row["shape"]
                ):
                    raise ValueError(f"materialized source tensor differs from header for {key}")
                if not bool(torch.isfinite(source_tensor).all().item()):
                    raise ValueError(f"source tensor contains non-finite values: {key}")
                destination_tensor = source_tensor.to(dtype=torch.bfloat16).contiguous()
                pending[key] = {
                    "kind": "trained_fp32_to_bf16",
                    "key": key,
                    "shape": list(row["shape"]),
                    "elements": source_tensor.numel(),
                    "source_dtype": "F32",
                    "destination_dtype": "BF16",
                    "source_shard": row["shard"],
                    "destination_shard": output_name,
                    "source_tensor_sha256": _tensor_sha256(source_tensor),
                    "destination_tensor_sha256": _tensor_sha256(destination_tensor),
                }
            else:
                row = base_layout[key]
                source_tensor = _load_tensor(base, str(row["shard"]), key)
                if (
                    source_tensor.dtype != torch.bfloat16
                    or list(source_tensor.shape) != row["shape"]
                ):
                    raise ValueError(f"materialized frozen base tensor differs for {key}")
                destination_tensor = source_tensor.contiguous()
                pending[key] = {
                    "kind": "frozen_base_auxiliary_head_restoration",
                    "key": key,
                    "shape": list(row["shape"]),
                    "elements": source_tensor.numel(),
                    "source_dtype": "BF16",
                    "destination_dtype": "BF16",
                    "base_revision": omission.get("base_revision"),
                    "base_shard": row["shard"],
                    "destination_shard": output_name,
                    "base_tensor_sha256": _tensor_sha256(source_tensor),
                    "destination_tensor_sha256": _tensor_sha256(destination_tensor),
                }
            tensors[key] = destination_tensor
            weight_map[key] = output_name
            total_bf16_bytes += destination_tensor.numel() * destination_tensor.element_size()
        save_file(tensors, str(output / output_name), metadata={"format": "pt"})
        tensors.clear()

        with safe_open(str(output / output_name), framework="pt", device="cpu") as written:
            written_keys = sorted(written.keys())
            if written_keys != sorted(keys):
                raise ValueError(f"written shard key set differs for {output_name}")
            for key in sorted(keys):
                row = base_layout[key]
                if key in layout:
                    source_row = layout[key]
                    source_tensor = _load_tensor(source, str(source_row["shard"]), key)
                    expected = (
                        source_tensor.to(dtype=torch.bfloat16).contiguous().view(torch.uint16)
                    )
                else:
                    source_tensor = _load_tensor(base, str(row["shard"]), key)
                    expected = source_tensor.contiguous().view(torch.uint16)
                actual_tensor = written.get_tensor(key)
                if actual_tensor.dtype != torch.bfloat16:
                    raise ValueError(f"written tensor is not BF16: {key}")
                actual = actual_tensor.contiguous().view(torch.uint16)
                if not torch.equal(actual, expected):
                    raise ValueError(f"written BF16 bits differ from the proven source: {key}")
                actual_digest = _tensor_sha256(actual_tensor)
                if actual_digest != pending[key]["destination_tensor_sha256"]:
                    raise ValueError(f"written tensor hash differs after reopen: {key}")
                if key in layout:
                    pending[key]["exact_cast_bits_verified"] = True
                    cast_rows.append(pending[key])
                else:
                    if actual_digest != pending[key]["base_tensor_sha256"]:
                        raise ValueError(f"restored auxiliary tensor hash differs from base: {key}")
                    pending[key]["exact_base_bits_verified"] = True
                    restoration_rows.append(pending[key])
        _fsync_file(output / output_name)

    index = {
        "metadata": {"total_size": total_bf16_bytes},
        "weight_map": {key: weight_map[key] for key in sorted(weight_map)},
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _fsync_file(output / "model.safetensors.index.json")
    sidecars = _copy_source_sidecars(source, output, set(source_shards))
    normalized_source_layout = {
        key: {"shape": layout[key]["shape"], "dtype": "F32"}
        for key in sorted(layout)
    }
    normalized_final_layout = {
        key: {"shape": base_layout[key]["shape"], "dtype": "BF16"}
        for key in sorted(base_layout)
    }
    return {
        "schema": "cyber_sft_fp32_to_bf16_cast_and_restore_proof_v2",
        "policy": CAST_POLICY,
        "tensor_order": "unicode_codepoint_sorted_key_v1",
        "shard_policy": {
            "maximum_bf16_payload_bytes": max_shard_bytes,
            "maximum_source_tensor_bytes": max_source_tensor_bytes,
            "oversize_tensor_allowed": False,
        },
        "source_dtype": "F32",
        "destination_dtype": "BF16",
        "trained_tensor_count": len(cast_rows),
        "trained_parameter_count": sum(row["elements"] for row in cast_rows),
        "restored_auxiliary_tensor_count": len(restoration_rows),
        "restored_auxiliary_parameter_count": sum(
            row["elements"] for row in restoration_rows
        ),
        "final_tensor_count": len(cast_rows) + len(restoration_rows),
        "final_parameter_count": sum(row["elements"] for row in cast_rows)
        + sum(row["elements"] for row in restoration_rows),
        "source_shard_count": len(source_shards),
        "destination_shard_count": shard_count,
        "destination_weight_bytes": total_bf16_bytes,
        "source_layout_sha256": digest_json(normalized_source_layout),
        "final_layout_sha256": digest_json(normalized_final_layout),
        "exact_omission_evidence_sha256": omission_layout["missing_tensors_sha256"],
        "exact_omission_tensors": omission_layout["missing_tensors"],
        "cast_rows_sha256": digest_json(cast_rows),
        "cast_rows": cast_rows,
        "restoration_rows_sha256": digest_json(restoration_rows),
        "restoration_rows": restoration_rows,
        "raw_sidecar_sha256": sidecars,
        "all_source_values_finite": True,
        "all_destination_bits_equal_direct_bf16_cast": True,
        "all_restored_auxiliary_bits_equal_frozen_base": True,
        "final_layout_exactly_matches_frozen_base": True,
    }


def build_cast_input(
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind immutable raw-export evidence to the one reviewed conversion transaction."""

    cast_plan = _validate_execution_plan(plan.get("cast_execution"))
    if observation.get("schema") != "fleet_sft_sfs_checkpoint_observation_v1":
        raise ValueError("unsupported export observation schema")
    observation_digest = _validate_digest(observation.get("observation_sha256"), "observation")
    if digest_json({k: v for k, v in observation.items() if k != "observation_sha256"}) != (
        observation_digest
    ):
        raise ValueError("export observation digest does not validate")
    inspection = _mapping(observation.get("output_inspection"), "raw output inspection")
    if inspection.get("root") != str(SOURCE_PATH) or str(inspection.get("dtype")).lower() != "f32":
        raise ValueError("export observation does not identify the exact raw F32 source")
    if raw_manifest.get("root") != str(SOURCE_PATH):
        raise ValueError("raw manifest names an unexpected source")
    if raw_manifest.get("manifest_sha256") != observation.get(
        "raw_export_full_manifest_sha256"
    ):
        raise ValueError("raw manifest differs from the export observation")
    if inspection.get("files_manifest_sha256") != raw_manifest.get("manifest_sha256"):
        raise ValueError("raw inspection and full manifest differ")
    omission = _mapping(
        _mapping(plan.get("export"), "plan export").get(
            "raw_export_auxiliary_head_omission"
        ),
        "raw export auxiliary-head omission",
    )
    observed_omission = _mapping(
        observation.get("weight_layout_exact_auxiliary_omission"),
        "observed exact auxiliary-head omission",
    )
    if observed_omission.get("schema") != (
        "cyber_sft_safetensors_exact_auxiliary_omission_v1"
    ) or observed_omission.get("exact_allowlist_match") is not True:
        raise ValueError("export observation lacks exact auxiliary-head omission proof")
    if observed_omission.get("raw_export_parameter_count") != inspection.get(
        "parameter_count"
    ):
        raise ValueError("raw inspection parameter count differs from omission proof")
    if observed_omission.get("missing_tensors_sha256") != inspection.get(
        "exact_auxiliary_omission_sha256"
    ):
        raise ValueError("raw inspection and omission layout evidence differ")
    base_model = _mapping(plan.get("base_model"), "base model")
    artifact_surface = _validate_base_inference_artifact_surface(
        cast_plan.get("base_inference_artifact_surface")
    )
    if (
        artifact_surface.get("weights_manifest_sha256")
        != base_model.get("weights_manifest_sha256")
        or _mapping(artifact_surface.get("index"), "base artifact index").get(
            "sha256"
        )
        != base_model.get("weights_index_sha256")
        or artifact_surface.get("required_runtime_sidecar_sha256")
        != base_model.get("runtime_sidecar_sha256")
    ):
        raise ValueError("base inference artifact surface differs from the frozen base model")
    speculative = _mapping(
        _mapping(plan.get("serving"), "serving").get("speculative_decoding"),
        "serving speculative decoding",
    )
    observed_no_speculative = _mapping(
        observation.get("no_speculative_decoding_proof"),
        "observed no-speculative-decoding proof",
    )
    if (
        speculative.get("enabled") is not False
        or speculative.get("proof")
        != "exact_base_registration_contains_no_speculative_decoding_or_draft_model_argument"
        or speculative.get("registration_sha256")
        != omission.get("serving_registration_sha256")
        or observed_no_speculative.get("schema")
        != "cyber_post_sft_no_speculative_decoding_proof_v1"
        or observed_no_speculative.get("registration_sha256")
        != speculative.get("registration_sha256")
        or observed_no_speculative.get("prohibited_runtime_args")
        != speculative.get("prohibited_runtime_args")
        or observed_no_speculative.get("prohibited_runtime_args_absent") is not True
        or observed_no_speculative.get("no_speculative_or_draft_argument") is not True
    ):
        raise ValueError("matched serving does not prove speculative decoding is disabled")
    result = {
        "schema": CAST_INPUT_SCHEMA,
        "source": {
            "path": str(SOURCE_PATH),
            "observation_sha256": observation_digest,
            "raw_full_manifest": dict(raw_manifest),
            "raw_inspection": dict(inspection),
            "exact_auxiliary_omission": dict(observed_omission),
            "source_checkpoint_full_manifest_sha256": observation.get(
                "full_file_manifest_sha256"
            ),
        },
        "frozen_base_auxiliary_source": {
            "path": str(BASE_MODEL_PATH),
            "repository": base_model.get("repository"),
            "revision": base_model.get("revision"),
            "weights_manifest_sha256": base_model.get("weights_manifest_sha256"),
            "inference_artifact_surface": dict(artifact_surface),
            "omission_policy": dict(omission),
            "speculative_decoding": dict(speculative),
            "no_speculative_decoding_proof": dict(observed_no_speculative),
        },
        "destination": {"path": str(DESTINATION_PATH), "must_be_absent": True},
        "execution": dict(cast_plan),
        "expected_trained_parameter_count": omission.get("raw_export_parameter_count"),
        "expected_final_parameter_count": base_model.get("parameter_count"),
    }
    result["cast_input_sha256"] = digest_json(result)
    return result


def validate_local_cast_bundle(plan: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Require ConfigMap source bytes to match the reviewed plan before creation."""

    execution = _validate_execution_plan(plan.get("cast_execution"))
    expected = _mapping(execution.get("config_map_code_sha256"), "cast code hashes")
    observed: dict[str, str] = {}
    for key, mounted_relative in MOUNTED_CODE_FILES.items():
        digest = file_sha256(root / LOCAL_CODE_FILES[key])
        if digest != expected.get(mounted_relative):
            raise ValueError(
                f"local cast bundle differs from reviewed digest for {mounted_relative}"
            )
        observed[mounted_relative] = digest
    return observed


def _payload_manifest(root: Path) -> dict[str, Any]:
    rows = [
        row
        for row in full_file_manifest(root)["files"]
        if row["path"] != ACCEPTANCE_RECEIPT_NAME
    ]
    return {
        "schema": "cyber_sft_full_file_manifest_v1",
        "root": str(root.resolve()),
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }


def _weights_manifest_sha256(full_manifest: Mapping[str, Any]) -> str:
    """Derive the frozen-lock weight digest from a full-file manifest."""

    files = full_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("full manifest files must be a list")
    rows = [
        {"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
        for row in files
        if isinstance(row, Mapping) and str(row.get("path", "")).endswith(".safetensors")
    ]
    if not rows:
        raise ValueError("frozen base full manifest contains no safetensor shards")
    return digest_json(rows)


def _validate_committed(root: Path, expected_input_sha256: str) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("cast destination is not a regular directory")
    receipt_path = root / ACCEPTANCE_RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("cast destination has no terminal acceptance receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != CAST_RECEIPT_SCHEMA:
        raise ValueError("unsupported cast receipt schema")
    receipt_digest = _validate_digest(receipt.get("cast_receipt_sha256"), "cast receipt")
    if (
        digest_json({k: v for k, v in receipt.items() if k != "cast_receipt_sha256"})
        != receipt_digest
    ):
        raise ValueError("cast receipt digest does not validate")
    if receipt.get("cast_input_sha256") != expected_input_sha256:
        raise ValueError("cast destination belongs to another input")
    payload = _payload_manifest(root)
    if payload["manifest_sha256"] != receipt["destination"]["payload_manifest_sha256"]:
        raise ValueError("committed cast payload differs from its receipt")
    return receipt


def _publish_terminal_evidence(
    destination: Path, receipt: Mapping[str, Any], expected_input_sha256: str
) -> dict[str, Any]:
    """Publish the post-marker full manifest and a digest-bound terminal COMPLETE marker."""

    evidence_dir = EVIDENCE_DIR
    suffix = expected_input_sha256.removeprefix("sha256:")[:12]
    partial = evidence_dir.parent / f".partial-{evidence_dir.name}-{suffix}"

    def validate(root: Path) -> dict[str, Any]:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("cast evidence path is not a regular directory")
        observed_receipt = json.loads((root / "cast-receipt.json").read_text())
        manifest = json.loads((root / "cast-full-manifest.json").read_text())
        complete = json.loads((root / "COMPLETE.json").read_text())
        if observed_receipt != dict(receipt):
            raise ValueError("terminal cast evidence contains a different receipt")
        if manifest != full_file_manifest(destination):
            raise ValueError("terminal cast full manifest differs from durable destination")
        digest = complete.get("complete_sha256")
        if digest_json({k: v for k, v in complete.items() if k != "complete_sha256"}) != digest:
            raise ValueError("terminal cast COMPLETE digest does not validate")
        receipt_file = file_sha256(root / "cast-receipt.json")
        manifest_file = file_sha256(root / "cast-full-manifest.json")
        if complete != {
            "schema": "cyber_sft_fp32_to_bf16_cast_complete_v2",
            "status": "COMPLETE",
            "cast_input_sha256": expected_input_sha256,
            "cast_receipt_sha256": receipt["cast_receipt_sha256"],
            "cast_receipt_file_sha256": receipt_file,
            "full_manifest_sha256": manifest["manifest_sha256"],
            "full_manifest_file_sha256": manifest_file,
            "complete_sha256": digest,
        }:
            raise ValueError("terminal cast COMPLETE marker fields differ")
        return complete

    evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(evidence_dir):
        if os.path.lexists(partial):
            raise ValueError("both final and partial cast evidence paths exist")
        return validate(evidence_dir)
    if os.path.lexists(partial):
        validate(partial)
        _rename_noreplace(partial, evidence_dir)
        _fsync_directory(evidence_dir.parent)
        return validate(evidence_dir)
    partial.mkdir(mode=0o700)
    manifest = full_file_manifest(destination)
    receipt_path = partial / "cast-receipt.json"
    manifest_path = partial / "cast-full-manifest.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    complete: dict[str, Any] = {
        "schema": "cyber_sft_fp32_to_bf16_cast_complete_v2",
        "status": "COMPLETE",
        "cast_input_sha256": expected_input_sha256,
        "cast_receipt_sha256": receipt["cast_receipt_sha256"],
        "cast_receipt_file_sha256": file_sha256(receipt_path),
        "full_manifest_sha256": manifest["manifest_sha256"],
        "full_manifest_file_sha256": file_sha256(manifest_path),
    }
    complete["complete_sha256"] = digest_json(complete)
    (partial / "COMPLETE.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    for path in partial.iterdir():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    _fsync_directory(partial)
    _rename_noreplace(partial, evidence_dir)
    _fsync_directory(evidence_dir.parent)
    return validate(evidence_dir)


def execute_cast(cast_input: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, convert, prove, receipt, and atomically publish one exact BF16 export."""

    if cast_input.get("schema") != CAST_INPUT_SCHEMA:
        raise ValueError("unsupported cast input schema")
    expected_input_sha256 = _validate_digest(cast_input.get("cast_input_sha256"), "cast input")
    if digest_json({k: v for k, v in cast_input.items() if k != "cast_input_sha256"}) != (
        expected_input_sha256
    ):
        raise ValueError("cast input digest does not validate")
    source_binding = _mapping(cast_input.get("source"), "source")
    base_binding = _mapping(
        cast_input.get("frozen_base_auxiliary_source"), "frozen base auxiliary source"
    )
    destination_binding = _mapping(cast_input.get("destination"), "destination")
    execution = _validate_execution_plan(cast_input.get("execution"))
    source = Path(str(source_binding.get("path")))
    base = Path(str(base_binding.get("path")))
    destination = Path(str(destination_binding.get("path")))
    if source != SOURCE_PATH or base != BASE_MODEL_PATH or destination != DESTINATION_PATH:
        raise ValueError("cast paths differ from the reviewed transaction")
    if (
        base_binding.get("repository") != BASE_MODEL_REPOSITORY
        or base_binding.get("revision") != BASE_MODEL_REVISION
        or base_binding.get("weights_manifest_sha256") != BASE_WEIGHTS_MANIFEST_SHA256
    ):
        raise ValueError("frozen base auxiliary source identity differs from the reviewed model")
    artifact_surface = _validate_base_inference_artifact_surface(
        base_binding.get("inference_artifact_surface")
    )
    if artifact_surface != execution.get("base_inference_artifact_surface"):
        raise ValueError("cast input artifact surface differs from the execution plan")
    omission = _mapping(base_binding.get("omission_policy"), "omission policy")
    speculative = _mapping(
        base_binding.get("speculative_decoding"), "speculative decoding proof"
    )
    no_speculative = _mapping(
        base_binding.get("no_speculative_decoding_proof"),
        "runtime no-speculative-decoding proof",
    )
    if (
        speculative.get("enabled") is not False
        or speculative.get("registration_sha256")
        != omission.get("serving_registration_sha256")
        or no_speculative.get("registration_sha256")
        != speculative.get("registration_sha256")
        or no_speculative.get("prohibited_runtime_args_absent") is not True
        or no_speculative.get("no_speculative_or_draft_argument") is not True
    ):
        raise ValueError("cast input does not bind disabled speculative decoding")
    if destination_binding.get("must_be_absent") is not True:
        raise ValueError("cast destination must be declared absent")
    if (
        cast_input.get("expected_trained_parameter_count")
        != execution.get("trained_parameter_count")
        or cast_input.get("expected_final_parameter_count")
        != execution.get("final_parameter_count")
    ):
        raise ValueError("cast input counts differ from the reviewed execution plan")
    suffix = expected_input_sha256.removeprefix("sha256:")[:12]
    partial = destination.parent / f".partial-{destination.name}-{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if os.path.lexists(partial):
            raise ValueError("both final and partial cast destinations exist")
        committed = _validate_committed(destination, expected_input_sha256)
        _publish_terminal_evidence(destination, committed, expected_input_sha256)
        return committed
    if os.path.lexists(partial):
        _validate_committed(partial, expected_input_sha256)
        _rename_noreplace(partial, destination)
        _fsync_directory(destination.parent)
        committed = _validate_committed(destination, expected_input_sha256)
        _publish_terminal_evidence(destination, committed, expected_input_sha256)
        return committed
    expected_manifest = _mapping(source_binding.get("raw_full_manifest"), "raw manifest")
    runtime_execution = _runtime_execution_provenance(cast_input)
    observed_manifest = full_file_manifest(source)
    if observed_manifest["manifest_sha256"] != expected_manifest.get("manifest_sha256"):
        raise ValueError("raw FP32 source bytes differ from the evidence manifest")
    base_manifest_before = _base_inference_artifact_manifest(base, artifact_surface)
    proof = cast_fp32_export(
        source,
        base,
        partial,
        omission,
        max_shard_bytes=int(execution["max_shard_bytes"]),
        max_source_tensor_bytes=int(execution["max_source_tensor_bytes"]),
    )
    if proof["trained_parameter_count"] != cast_input.get(
        "expected_trained_parameter_count"
    ):
        raise ValueError("cast trained parameter count differs from the raw export")
    if proof["final_parameter_count"] != cast_input.get("expected_final_parameter_count"):
        raise ValueError("cast final parameter count differs from the frozen architecture")
    final_layout = compare_safetensor_layout(base, partial)
    source_after = full_file_manifest(source)
    if source_after["manifest_sha256"] != observed_manifest["manifest_sha256"]:
        raise ValueError("raw FP32 source changed during conversion")
    base_manifest_after = _base_inference_artifact_manifest(base, artifact_surface)
    if base_manifest_after["manifest_sha256"] != base_manifest_before["manifest_sha256"]:
        raise ValueError("frozen base bytes changed during auxiliary-head restoration")
    inspection = inspect_hf_export(
        partial,
        expected_tokenizer_manifest_sha256="sha256:" + "0" * 64,
        expected_chat_template_sha256="sha256:" + "0" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        expected_parameter_count=int(cast_input["expected_final_parameter_count"]),
        require_base_sidecars=False,
        expected_dtype="BF16",
    )
    payload = _payload_manifest(partial)
    receipt: dict[str, Any] = {
        "schema": CAST_RECEIPT_SCHEMA,
        "cast_input_sha256": expected_input_sha256,
        "source": {
            "path": str(source),
            "observation_sha256": source_binding.get("observation_sha256"),
            "checkpoint_full_manifest_sha256": source_binding.get(
                "source_checkpoint_full_manifest_sha256"
            ),
            "raw_full_manifest_sha256": observed_manifest["manifest_sha256"],
            "raw_full_manifest_after_sha256": source_after["manifest_sha256"],
            "raw_source_stable_during_cast": True,
            "raw_weights_manifest_sha256": source_binding["raw_inspection"][
                "weights_manifest_sha256"
            ],
            "dtype": "F32",
        },
        "frozen_base_auxiliary_source": {
            "path": str(base),
            "repository": base_binding.get("repository"),
            "revision": base_binding.get("revision"),
            "weights_manifest_sha256": base_binding.get("weights_manifest_sha256"),
            "inference_artifact_surface": dict(artifact_surface),
            "inference_artifact_surface_sha256": digest_json(artifact_surface),
            "inference_artifact_manifest_before": base_manifest_before,
            "inference_artifact_manifest_after": base_manifest_after,
            "manifest_scope": "exact_inference_artifact_surface_v1",
            "omission_policy_sha256": digest_json(omission),
            "exact_auxiliary_omission_policy": dict(omission),
            "full_manifest_before_sha256": base_manifest_before["manifest_sha256"],
            "full_manifest_after_sha256": base_manifest_after["manifest_sha256"],
            "base_source_stable_during_cast": True,
            "exact_omission_evidence_sha256": proof[
                "exact_omission_evidence_sha256"
            ],
            "role": "speculative_draft_heads",
            "serving_inference_effect": "inert_without_speculative_decoding",
            "serving_registration_sha256": omission.get(
                "serving_registration_sha256"
            ),
            "speculative_decoding_enabled": False,
            "restoration_semantics": "frozen_base_auxiliary_head_restoration_not_trained_weights",
        },
        "conversion": proof,
        "execution_plan": dict(execution),
        "execution": runtime_execution,
        "destination": {
            "path": str(destination),
            "dtype": "BF16",
            "inspection": {**inspection, "root": str(destination)},
            "exact_base_layout_equivalence": final_layout,
            "payload_manifest_excludes": [ACCEPTANCE_RECEIPT_NAME],
            "payload_manifest_sha256": payload["manifest_sha256"],
            "payload_file_count": payload["file_count"],
            "payload_total_bytes": payload["total_bytes"],
            "atomic_transaction": "directory_rename_noreplace_v1",
            "atomic_promotion": True,
        },
    }
    receipt["cast_receipt_sha256"] = digest_json(receipt)
    with (partial / ACCEPTANCE_RECEIPT_NAME).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _validate_committed(partial, expected_input_sha256)
    _fsync_directory(partial)
    _rename_noreplace(partial, destination)
    _fsync_directory(destination.parent)
    committed = _validate_committed(destination, expected_input_sha256)
    _publish_terminal_evidence(destination, committed, expected_input_sha256)
    return committed


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-input")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--observation", type=Path, required=True)
    build.add_argument("--raw-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--input", type=Path, required=True)
    validate = subparsers.add_parser("validate-bundle")
    validate.add_argument("--plan", type=Path, required=True)
    validate.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build-input":
        result = build_cast_input(
            _read(args.plan), _read(args.observation), _read(args.raw_manifest)
        )
        if os.path.lexists(args.output):
            raise ValueError(f"refusing pre-existing cast input: {args.output}")
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return
    if args.command == "validate-bundle":
        print(json.dumps(validate_local_cast_bundle(_read(args.plan), args.root), sort_keys=True))
        return
    print(json.dumps(execute_cast(_read(args.input)), sort_keys=True))


if __name__ == "__main__":
    main()
