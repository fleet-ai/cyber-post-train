"""Deterministic, bounded-memory FP32 to BF16 conversion for the final SFT export."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import ssl
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .io import digest_json, file_sha256
from .post_sft_artifacts import _safetensor_layout, full_file_manifest, inspect_hf_export

CAST_INPUT_SCHEMA = "cyber_sft_fp32_to_bf16_cast_input_v1"
CAST_RECEIPT_SCHEMA = "cyber_sft_fp32_to_bf16_cast_receipt_v1"
CAST_POLICY = "deterministic_sorted_tensor_fp32_to_bf16_v1"
SOURCE_PATH = Path(
    "/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-v1/global_step_318/policy"
)
DESTINATION_PATH = Path(
    "/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-bf16-v1/global_step_318/policy"
)
EVIDENCE_DIR = Path("/mnt/sfs/jobs/chris-cyber-qwen36-sft-bf16-cast-v1/receipt")
ACCEPTANCE_RECEIPT_NAME = ".fleet-bf16-cast-acceptance.json"
DEFAULT_MAX_SHARD_BYTES = 2 * 1024**3
DEFAULT_MAX_SOURCE_TENSOR_BYTES = 8 * 1024**3
HASH_CHUNK_BYTES = 8 * 1024**2
NAMESPACE = "fleet-train-jobs"
JOB_NAME = "chris-cyber-qwen36-sft-bf16-cast-v1"
CONFIG_MAP_NAME = JOB_NAME
SERVICE_ACCOUNT_NAME = "chris-cyber-qwen36-sft-bf16-cast-observer-v1"
CONTAINER_NAME = "cast"
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
    "training_post_sft_cast.py": "training/post_sft_cast.py",
}


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


def _runtime_execution_provenance(cast_input: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the cast to exact live Job, Pod, imageID, ConfigMap, and mounted bytes."""

    plan = _mapping(cast_input.get("execution"), "cast execution plan")
    namespace = os.environ.get("POD_NAMESPACE", "").strip()
    pod_name = os.environ.get("POD_NAME", "").strip()
    pod_uid = os.environ.get("POD_UID", "").strip()
    job_uid = os.environ.get("JOB_UID", "").strip()
    if not all((namespace, pod_name, pod_uid, job_uid)) or namespace != NAMESPACE:
        raise ValueError("cast downward-API identity is missing or unexpected")
    pod = _kubernetes_get(f"/api/v1/namespaces/{namespace}/pods/{pod_name}")
    job = _kubernetes_get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{JOB_NAME}")
    config_map = _kubernetes_get(
        f"/api/v1/namespaces/{namespace}/configmaps/{CONFIG_MAP_NAME}"
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
    statuses = _mapping(pod.get("status"), "Pod status").get("containerStatuses")
    if not isinstance(statuses, list):
        raise ValueError("live cast Pod has no container status")
    status = next(
        (row for row in statuses if isinstance(row, Mapping) and row.get("name") == CONTAINER_NAME),
        None,
    )
    image_id = str(status.get("imageID") if status else "")
    expected_image_digest = CAST_IMAGE.rsplit("@", 1)[1]
    if expected_image_digest not in image_id:
        raise ValueError("resolved cast imageID differs from the frozen digest")
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
        "schema": "cyber_sft_fp32_to_bf16_cast_execution_plan_v1",
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
        "policy": CAST_POLICY,
        "source_dtype": "F32",
        "destination_dtype": "BF16",
        "max_shard_bytes": DEFAULT_MAX_SHARD_BYTES,
        "max_source_tensor_bytes": DEFAULT_MAX_SOURCE_TENSOR_BYTES,
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
    output: Path,
    *,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    max_source_tensor_bytes: int = DEFAULT_MAX_SOURCE_TENSOR_BYTES,
) -> dict[str, Any]:
    """Cast one immutable FP32 export using fixed grouping and exact bit verification."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source = source.resolve(strict=True)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing pre-existing cast output: {output}")
    output.mkdir(mode=0o700, parents=False)
    layout, source_shards = _safetensor_layout(source)
    wrong_dtype = sorted(key for key, row in layout.items() if row["dtype"] != "F32")
    if wrong_dtype:
        raise ValueError(f"raw export is not uniformly F32 ({len(wrong_dtype)} tensors differ)")
    for key, row in layout.items():
        source_bytes = math.prod(row["shape"]) * 4
        if source_bytes > max_source_tensor_bytes:
            raise ValueError(f"source tensor {key} exceeds the fixed memory bound")
    groups = _shard_groups(layout, max_shard_bytes)
    shard_count = len(groups)
    weight_map: dict[str, str] = {}
    proof_rows: list[dict[str, Any]] = []
    total_bf16_bytes = 0
    torch.set_num_threads(1)

    for shard_index, keys in enumerate(groups, start=1):
        output_name = f"model-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        tensors: dict[str, Any] = {}
        pending: dict[str, dict[str, Any]] = {}
        for key in keys:
            row = layout[key]
            source_tensor = _load_tensor(source, str(row["shard"]), key)
            if source_tensor.dtype != torch.float32 or list(source_tensor.shape) != row["shape"]:
                raise ValueError(f"materialized source tensor differs from header for {key}")
            if not bool(torch.isfinite(source_tensor).all().item()):
                raise ValueError(f"source tensor contains non-finite values: {key}")
            destination_tensor = source_tensor.to(dtype=torch.bfloat16).contiguous()
            pending[key] = {
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
                row = layout[key]
                source_tensor = _load_tensor(source, str(row["shard"]), key)
                expected = source_tensor.to(dtype=torch.bfloat16).contiguous().view(torch.uint16)
                actual_tensor = written.get_tensor(key)
                if actual_tensor.dtype != torch.bfloat16:
                    raise ValueError(f"written tensor is not BF16: {key}")
                actual = actual_tensor.contiguous().view(torch.uint16)
                if not torch.equal(actual, expected):
                    raise ValueError(f"written BF16 bits differ from direct cast: {key}")
                actual_digest = _tensor_sha256(actual_tensor)
                if actual_digest != pending[key]["destination_tensor_sha256"]:
                    raise ValueError(f"written tensor hash differs after reopen: {key}")
                pending[key]["exact_cast_bits_verified"] = True
                proof_rows.append(pending[key])

    index = {
        "metadata": {"total_size": total_bf16_bytes},
        "weight_map": {key: weight_map[key] for key in sorted(weight_map)},
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecars = _copy_source_sidecars(source, output, set(source_shards))
    normalized_layout = {
        key: {"shape": layout[key]["shape"], "dtype": "F32"}
        for key in sorted(layout)
    }
    return {
        "schema": "cyber_sft_fp32_to_bf16_cast_proof_v1",
        "policy": CAST_POLICY,
        "tensor_order": "unicode_codepoint_sorted_key_v1",
        "shard_policy": {
            "maximum_bf16_payload_bytes": max_shard_bytes,
            "maximum_source_tensor_bytes": max_source_tensor_bytes,
            "oversize_tensor_allowed": False,
        },
        "source_dtype": "F32",
        "destination_dtype": "BF16",
        "tensor_count": len(proof_rows),
        "parameter_count": sum(row["elements"] for row in proof_rows),
        "source_shard_count": len(source_shards),
        "destination_shard_count": shard_count,
        "destination_weight_bytes": total_bf16_bytes,
        "source_layout_sha256": digest_json(normalized_layout),
        "cast_rows_sha256": digest_json(proof_rows),
        "cast_rows": proof_rows,
        "raw_sidecar_sha256": sidecars,
        "all_source_values_finite": True,
        "all_destination_bits_equal_direct_bf16_cast": True,
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
    result = {
        "schema": CAST_INPUT_SCHEMA,
        "source": {
            "path": str(SOURCE_PATH),
            "observation_sha256": observation_digest,
            "raw_full_manifest": dict(raw_manifest),
            "raw_inspection": dict(inspection),
            "source_checkpoint_full_manifest_sha256": observation.get(
                "full_file_manifest_sha256"
            ),
        },
        "destination": {"path": str(DESTINATION_PATH), "must_be_absent": True},
        "execution": dict(cast_plan),
        "expected_parameter_count": plan["base_model"]["parameter_count"],
    }
    result["cast_input_sha256"] = digest_json(result)
    return result


def validate_local_cast_bundle(plan: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Require ConfigMap source bytes to match the reviewed plan before creation."""

    execution = _validate_execution_plan(plan.get("cast_execution"))
    expected = _mapping(execution.get("config_map_code_sha256"), "cast code hashes")
    observed: dict[str, str] = {}
    for relative in MOUNTED_CODE_FILES.values():
        digest = file_sha256(root / relative)
        if digest != expected.get(relative):
            raise ValueError(f"local cast bundle differs from reviewed digest for {relative}")
        observed[relative] = digest
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
            "schema": "cyber_sft_fp32_to_bf16_cast_complete_v1",
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
        "schema": "cyber_sft_fp32_to_bf16_cast_complete_v1",
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
    destination_binding = _mapping(cast_input.get("destination"), "destination")
    execution = _validate_execution_plan(cast_input.get("execution"))
    source = Path(str(source_binding.get("path")))
    destination = Path(str(destination_binding.get("path")))
    if source != SOURCE_PATH or destination != DESTINATION_PATH:
        raise ValueError("cast paths differ from the reviewed transaction")
    if destination_binding.get("must_be_absent") is not True:
        raise ValueError("cast destination must be declared absent")
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
    proof = cast_fp32_export(
        source,
        partial,
        max_shard_bytes=int(execution["max_shard_bytes"]),
        max_source_tensor_bytes=int(execution["max_source_tensor_bytes"]),
    )
    if proof["parameter_count"] != cast_input.get("expected_parameter_count"):
        raise ValueError("cast parameter count differs from the frozen architecture")
    source_after = full_file_manifest(source)
    if source_after["manifest_sha256"] != observed_manifest["manifest_sha256"]:
        raise ValueError("raw FP32 source changed during conversion")
    inspection = inspect_hf_export(
        partial,
        expected_tokenizer_manifest_sha256="sha256:" + "0" * 64,
        expected_chat_template_sha256="sha256:" + "0" * 64,
        expected_config_sha256="sha256:" + "0" * 64,
        expected_parameter_count=int(cast_input["expected_parameter_count"]),
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
        "conversion": proof,
        "execution_plan": dict(execution),
        "execution": runtime_execution,
        "destination": {
            "path": str(destination),
            "dtype": "BF16",
            "inspection": {**inspection, "root": str(destination)},
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
