"""Fail-closed, CPU-only transport and composition of the final SFT inference bundle."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import ssl
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .io import digest_json, file_sha256
from .post_sft_artifacts import inspect_hf_export

STAGE_SCHEMA = "cyber_sft_inference_stage_input_v1"
RECEIPT_SCHEMA = "cyber_sft_inference_stage_receipt_v1"
EXECUTION_SCHEMA = "cyber_sft_inference_stage_execution_v1"
FILEBROWSER_ORIGIN = "http://filebrowser.fleet-train-data-plane.svc.cluster.local"
EXPORT_SOURCE = "/exports/cyber-sft/ft-run-574bd7b3/step-318-v1/global_step_318/policy"
DESTINATION = "/models/cyber-sft/ft-run-574bd7b3/step-318"
BASE_ROOT = "/models/qwen3.6-27b/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
JOB_NAME = "chris-cyber-qwen36-sft-stage-574bd7b3-v1"
CONFIG_MAP_NAME = JOB_NAME
SERVICE_ACCOUNT_NAME = "chris-cyber-qwen36-sft-stage-observer-v1"
CONTAINER_NAME = "stage"
BUNDLE_ROOT = Path("/bundle")
MOUNTED_CODE_FILES = {
    "training__init__.py": "training/__init__.py",
    "training_io.py": "training/io.py",
    "training_post_sft_artifacts.py": "training/post_sft_artifacts.py",
    "training_post_sft_staging.py": "training/post_sft_staging.py",
}
STAGE_INPUT_CONFIG_MAP_KEY = "stage-input.json"
STAGE_INPUT_MOUNT_PATH = "stage-input.json"
ACCEPTANCE_RECEIPT_NAME = ".fleet-acceptance.json"
STAGING_IMAGE = (
    "lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1"
)
STAGING_COMMAND = {
    "command": ["python3", "-m", "training.post_sft_staging", "execute"],
    "args": ["--input", "/bundle/stage-input.json", "--work-root", "/work/transfer"],
}
STAGING_COMMAND_SHA256 = digest_json(STAGING_COMMAND)
IMAGE_ID_RE = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*://)?(?:[^@\s]+@)?(sha256:[0-9a-f]{64})$"
)


def canonical_stage_input_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the one byte encoding mounted as ``stage-input.json``."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _resolved_image_digest(image_id: str) -> str:
    match = IMAGE_ID_RE.fullmatch(image_id)
    if match is None:
        raise ValueError("resolved staging imageID is not an exact digest identity")
    return match.group(1)


def _validated_digest_map(value: Any, field: str, expected_paths: set[str]) -> dict[str, str]:
    mapping = _mapping(value, field)
    if set(mapping) != expected_paths:
        raise ValueError(f"{field} paths differ from the reviewed staging bundle")
    result: dict[str, str] = {}
    for path, digest in mapping.items():
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ValueError(f"{field} has an invalid SHA-256 for {path}")
        result[str(path)] = digest
    return result


def _validate_staging_execution_plan(value: Any) -> dict[str, Any]:
    plan = _mapping(value, "staging execution plan")
    expected_fixed = {
        "schema": "cyber_sft_inference_stage_execution_plan_v1",
        "namespace": "inference",
        "job_name": JOB_NAME,
        "config_map_name": CONFIG_MAP_NAME,
        "service_account_name": SERVICE_ACCOUNT_NAME,
        "container_name": CONTAINER_NAME,
        "image": STAGING_IMAGE,
        "image_digest": STAGING_IMAGE.rsplit("@", 1)[1],
        "command_sha256": STAGING_COMMAND_SHA256,
    }
    for field, expected in expected_fixed.items():
        if plan.get(field) != expected:
            raise ValueError(f"staging execution plan {field} differs from the reviewed rail")
    code_hashes = _validated_digest_map(
        plan.get("config_map_code_sha256"),
        "staging execution plan config_map_code_sha256",
        set(MOUNTED_CODE_FILES.values()),
    )
    return {**expected_fixed, "config_map_code_sha256": code_hashes}


def validate_local_staging_bundle(plan: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Require local ConfigMap source bytes to match the reviewed plan exactly."""

    execution = _validate_staging_execution_plan(plan.get("staging_execution"))
    expected = execution["config_map_code_sha256"]
    observed: dict[str, str] = {}
    for relative in MOUNTED_CODE_FILES.values():
        path = root / relative
        digest = file_sha256(path)
        if digest != expected[relative]:
            raise ValueError(
                f"local staging bundle differs from the reviewed digest for {relative}"
            )
        observed[relative] = digest
    return observed


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing an existing filesystem entry."""

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
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
        result = rename(-100, encoded_source, -100, encoded_destination, 1)
    elif sys.platform == "darwin":  # parity for local tests; production is Linux.
        rename = getattr(libc, "renamex_np", None)
        if rename is None:
            raise RuntimeError("renamex_np is required for collision-free macOS promotion")
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(encoded_source, encoded_destination, 0x00000004)
    else:
        raise RuntimeError(f"atomic no-replace promotion is unsupported on {sys.platform}")
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, "promotion destination already exists", destination)
    raise OSError(error, os.strerror(error), destination)


def _fsync_directory(path: Path) -> None:
    """Make directory-entry changes durable before/after the atomic rename."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _kubernetes_get(path: str) -> dict[str, Any]:
    host = os.environ.get("KUBERNETES_SERVICE_HOST", "").strip()
    port = os.environ.get("KUBERNETES_SERVICE_PORT_HTTPS", "443").strip()
    if not host:
        raise ValueError("KUBERNETES_SERVICE_HOST is required for runtime provenance")
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Kubernetes service-account token is empty")
    request = urllib.request.Request(
        f"https://{host}:{port}{path}", headers={"Authorization": f"Bearer {token}"}
    )
    context = ssl.create_default_context(cafile=str(ca_path))
    with urllib.request.urlopen(request, timeout=30, context=context) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise ValueError("Kubernetes API returned a non-object")
    return value


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required for runtime provenance")
    return value


def _runtime_execution_provenance(stage_input: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the running Pod, owning Job, imageID, ConfigMap, and mounted bytes."""

    stage_input_sha256 = str(stage_input.get("stage_input_sha256") or "")
    execution_plan = _validate_staging_execution_plan(stage_input.get("execution"))
    namespace = _required_environment("POD_NAMESPACE")
    pod_name = _required_environment("POD_NAME")
    pod_uid = _required_environment("POD_UID")
    job_uid = _required_environment("JOB_UID")
    if namespace != execution_plan["namespace"]:
        raise ValueError("runtime namespace differs from the reviewed staging plan")
    pod = _kubernetes_get(f"/api/v1/namespaces/{namespace}/pods/{pod_name}")
    job = _kubernetes_get(f"/apis/batch/v1/namespaces/{namespace}/jobs/{JOB_NAME}")
    config_map = _kubernetes_get(
        f"/api/v1/namespaces/{namespace}/configmaps/{CONFIG_MAP_NAME}"
    )
    pod_metadata = _mapping(pod.get("metadata"), "Pod metadata")
    if (
        pod_metadata.get("namespace") != namespace
        or pod_metadata.get("name") != pod_name
        or pod_metadata.get("uid") != pod_uid
    ):
        raise ValueError("live Pod UID differs from the downward-API identity")
    owners = pod_metadata.get("ownerReferences")
    if not isinstance(owners, list) or not any(
        isinstance(owner, Mapping)
        and owner.get("kind") == "Job"
        and owner.get("name") == JOB_NAME
        and owner.get("uid") == job_uid
        for owner in owners
    ):
        raise ValueError("live Pod is not owned by the expected Job UID")
    job_metadata = _mapping(job.get("metadata"), "Job metadata")
    if (
        job_metadata.get("namespace") != namespace
        or job_metadata.get("name") != JOB_NAME
        or job_metadata.get("uid") != job_uid
    ):
        raise ValueError("live Job UID differs from the Pod controller identity")
    job_resource_version = str(job_metadata.get("resourceVersion") or "").strip()
    pod_resource_version = str(pod_metadata.get("resourceVersion") or "").strip()
    if not job_resource_version or not pod_resource_version:
        raise ValueError("Job and Pod resourceVersion are required")

    pod_spec = _mapping(pod.get("spec"), "Pod spec")
    if pod_spec.get("serviceAccountName") != SERVICE_ACCOUNT_NAME:
        raise ValueError("live Pod uses an unexpected service account")
    containers = pod_spec.get("containers")
    if not isinstance(containers, list):
        raise ValueError("live Pod has no container specification")
    container = next(
        (
            item
            for item in containers
            if isinstance(item, Mapping) and item.get("name") == CONTAINER_NAME
        ),
        None,
    )
    if container is None:
        raise ValueError("live Pod is missing the staging container")
    image = str(container.get("image") or "")
    if image != STAGING_IMAGE:
        raise ValueError("live staging container image differs from the frozen image")
    command = {
        "command": list(container.get("command") or []),
        "args": list(container.get("args") or []),
    }
    command_sha256 = digest_json(command)
    if command_sha256 != STAGING_COMMAND_SHA256:
        raise ValueError("live staging command differs from the frozen command")
    volumes = pod_spec.get("volumes")
    if not isinstance(volumes, list):
        raise ValueError("live Pod has no volume specification")
    matching_volumes = [
        volume
        for volume in volumes
        if (
            isinstance(volume, Mapping)
            and isinstance(volume.get("configMap"), Mapping)
            and volume["configMap"].get("name") == CONFIG_MAP_NAME
        )
    ]
    if len(matching_volumes) != 1:
        raise ValueError("live Pod does not mount the expected ConfigMap")
    config_volume = _mapping(matching_volumes[0].get("configMap"), "Pod ConfigMap volume")
    if config_volume.get("optional") not in {None, False}:
        raise ValueError("live Pod makes the reviewed ConfigMap optional")

    pod_status = _mapping(pod.get("status"), "Pod status")
    statuses = pod_status.get("containerStatuses")
    if not isinstance(statuses, list):
        raise ValueError("live Pod has no container status")
    status = next(
        (
            item
            for item in statuses
            if isinstance(item, Mapping) and item.get("name") == CONTAINER_NAME
        ),
        None,
    )
    if status is None:
        raise ValueError("live Pod has no staging-container status")
    image_id = str(status.get("imageID") or "")
    resolved_image_digest = _resolved_image_digest(image_id)
    expected_digest = execution_plan["image_digest"]
    if resolved_image_digest != expected_digest:
        raise ValueError("resolved staging imageID differs from the frozen image digest")

    config_metadata = _mapping(config_map.get("metadata"), "ConfigMap metadata")
    if (
        config_metadata.get("namespace") != namespace
        or config_metadata.get("name") != CONFIG_MAP_NAME
    ):
        raise ValueError("live ConfigMap identity differs from the reviewed staging bundle")
    config_uid = str(config_metadata.get("uid") or "").strip()
    config_resource_version = str(config_metadata.get("resourceVersion") or "").strip()
    if not config_uid or not config_resource_version:
        raise ValueError("ConfigMap UID and resourceVersion are required")
    if config_map.get("immutable") is not True:
        raise ValueError("staging ConfigMap must be immutable")
    config_data = _mapping(config_map.get("data"), "ConfigMap data")
    expected_keys = set(MOUNTED_CODE_FILES) | {STAGE_INPUT_CONFIG_MAP_KEY}
    if set(config_data) != expected_keys:
        raise ValueError("ConfigMap keys differ from the reviewed staging bundle")
    expected_code_hashes = execution_plan["config_map_code_sha256"]
    mounted_hashes: dict[str, str] = {}
    for key, relative in MOUNTED_CODE_FILES.items():
        value = config_data.get(key)
        if not isinstance(value, str):
            raise ValueError(f"ConfigMap is missing text key {key}")
        expected_hash = expected_code_hashes[relative]
        if _sha256_bytes(value.encode()) != expected_hash:
            raise ValueError(f"live ConfigMap differs from the reviewed digest for {relative}")
        mounted_path = BUNDLE_ROOT / relative
        observed_hash = file_sha256(mounted_path)
        if observed_hash != expected_hash:
            raise ValueError(f"mounted ConfigMap bytes differ for {relative}")
        mounted_hashes[relative] = observed_hash
    stage_input_text = config_data.get(STAGE_INPUT_CONFIG_MAP_KEY)
    if not isinstance(stage_input_text, str):
        raise ValueError("ConfigMap is missing the stage input")
    expected_stage_input_bytes = canonical_stage_input_bytes(stage_input)
    if stage_input_text.encode() != expected_stage_input_bytes:
        raise ValueError("live ConfigMap stage input bytes differ from the executing stage input")
    mounted_stage_input_path = BUNDLE_ROOT / STAGE_INPUT_MOUNT_PATH
    if mounted_stage_input_path.read_bytes() != expected_stage_input_bytes:
        raise ValueError("mounted stage input bytes differ from the reviewed stage input")
    mounted_stage_input = _read(mounted_stage_input_path)
    if mounted_stage_input != dict(stage_input):
        raise ValueError("mounted stage input differs from the executing stage input")
    unsigned_stage_input = {
        key: value for key, value in mounted_stage_input.items() if key != "stage_input_sha256"
    }
    if digest_json(unsigned_stage_input) != stage_input_sha256:
        raise ValueError("mounted stage-input digest does not validate")
    stage_input_file_sha256 = _sha256_bytes(expected_stage_input_bytes)
    mounted_hashes[STAGE_INPUT_MOUNT_PATH] = stage_input_file_sha256

    return {
        "schema": EXECUTION_SCHEMA,
        "image": image,
        "image_id": image_id,
        "resolved_image_digest": resolved_image_digest,
        "command_sha256": command_sha256,
        "service_account_name": SERVICE_ACCOUNT_NAME,
        "container_name": CONTAINER_NAME,
        "job": {
            "namespace": namespace,
            "name": JOB_NAME,
            "uid": job_uid,
            "resource_version": job_resource_version,
            "spec_sha256": digest_json(_mapping(job.get("spec"), "Job spec")),
        },
        "pod": {
            "namespace": namespace,
            "name": pod_name,
            "uid": pod_uid,
            "resource_version": pod_resource_version,
            "spec_sha256": digest_json(pod_spec),
        },
        "config_map": {
            "namespace": namespace,
            "name": CONFIG_MAP_NAME,
            "uid": config_uid,
            "resource_version": config_resource_version,
            "immutable": True,
            "reviewed_code_sha256": expected_code_hashes,
            "mounted_file_sha256": mounted_hashes,
            "stage_input_file_sha256": stage_input_file_sha256,
        },
        "stage_input_sha256": stage_input_sha256,
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _manifest_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") != "cyber_sft_full_file_manifest_v1":
        raise ValueError("raw export manifest has an unsupported schema")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("raw export manifest files must be an array of objects")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        name = row.get("path")
        size = row.get("size")
        digest = row.get("sha256")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("raw export manifest paths must be unique and non-empty")
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("raw export manifest contains an unsafe path")
        if not isinstance(size, int) or size < 0:
            raise ValueError("raw export manifest contains an invalid size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("raw export manifest contains an invalid SHA-256")
        int(digest, 16)
        seen.add(name)
        normalized.append({"path": name, "size": size, "sha256": digest})
    normalized.sort(key=lambda row: row["path"])
    if manifest.get("file_count") != len(normalized):
        raise ValueError("raw export manifest file_count is inconsistent")
    if manifest.get("total_bytes") != sum(row["size"] for row in normalized):
        raise ValueError("raw export manifest total_bytes is inconsistent")
    if manifest.get("manifest_sha256") != digest_json(normalized):
        raise ValueError("raw export manifest digest does not validate")
    return normalized


def verify_full_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read each file once and require exact path, size, and SHA-256 equality."""

    resolved = root.resolve(strict=True)
    expected = _manifest_rows(manifest)
    observed_paths = sorted(
        path.relative_to(resolved).as_posix()
        for path in resolved.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if any(path.is_symlink() for path in resolved.rglob("*")):
        raise ValueError("staged export must not contain symlinks")
    if observed_paths != [row["path"] for row in expected]:
        raise ValueError("staged export paths differ from the source manifest")
    for row in expected:
        path = resolved / row["path"]
        if path.stat().st_size != row["size"] or file_sha256(path) != (
            "sha256:" + row["sha256"]
        ):
            raise ValueError(f"staged export content differs for {row['path']}")
    return {
        "file_count": len(expected),
        "total_bytes": sum(row["size"] for row in expected),
        "manifest_sha256": digest_json(expected),
    }


def _payload_manifest(root: Path) -> dict[str, Any]:
    """Hash the served model payload, excluding only the reserved acceptance receipt."""

    resolved = root.resolve(strict=True)
    all_paths = list(resolved.rglob("*"))
    if any(path.is_symlink() for path in all_paths):
        raise ValueError("staged inference bundle must not contain symlinks")
    files = sorted(
        path
        for path in all_paths
        if path.is_file() and path.relative_to(resolved).as_posix() != ACCEPTANCE_RECEIPT_NAME
    )
    rows = [
        {
            "path": path.relative_to(resolved).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path).removeprefix("sha256:"),
        }
        for path in files
    ]
    return {
        "schema": "cyber_sft_payload_manifest_v1",
        "root": str(resolved),
        "reserved_path_exclusion": ACCEPTANCE_RECEIPT_NAME,
        "file_count": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "files": rows,
        "manifest_sha256": digest_json(rows),
    }


def _validate_committed_directory(
    root: Path, *, final: Path, expected_stage_input_sha256: str
) -> dict[str, Any]:
    """Validate a complete partial/final transaction without changing it."""

    if not root.is_dir() or root.is_symlink():
        raise ValueError("staging transaction path is not a regular directory")
    acceptance = root / ACCEPTANCE_RECEIPT_NAME
    if not acceptance.is_file() or acceptance.is_symlink():
        raise ValueError("staging transaction has no regular embedded acceptance receipt")
    receipt = _read(acceptance)
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("staging transaction acceptance receipt has an unsupported schema")
    receipt_sha256 = receipt.get("staging_receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "staging_receipt_sha256"}
    if digest_json(unsigned) != receipt_sha256:
        raise ValueError("staging transaction acceptance receipt digest does not validate")
    if receipt.get("stage_input_sha256") != expected_stage_input_sha256:
        raise ValueError("staging transaction belongs to a different stage input")
    destination = _mapping(receipt.get("destination"), "staging transaction destination")
    if (
        destination.get("path") != str(final)
        or destination.get("acceptance_receipt_path")
        != str(final / ACCEPTANCE_RECEIPT_NAME)
        or destination.get("payload_manifest_excludes") != [ACCEPTANCE_RECEIPT_NAME]
        or destination.get("atomic_transaction") != "directory_rename_noreplace_v1"
        or destination.get("atomic_promotion") is not True
    ):
        raise ValueError("staging transaction destination contract differs")
    composed = _mapping(
        _mapping(receipt.get("composition"), "staging transaction composition").get(
            "inspection"
        ),
        "staging transaction inspection",
    )
    payload = _payload_manifest(root)
    if (
        payload["manifest_sha256"] != composed.get("files_manifest_sha256")
        or payload["manifest_sha256"] != destination.get("payload_manifest_sha256")
        or payload["file_count"] != destination.get("payload_file_count")
        or payload["total_bytes"] != destination.get("payload_total_bytes")
    ):
        raise ValueError("staging transaction payload differs from its acceptance receipt")
    return receipt


def _safe_extract_zip(archive: Path, output: Path) -> Path:
    output.mkdir(mode=0o700)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if not members:
            raise ValueError("filebrowser returned an empty ZIP archive")
        for member in members:
            pure = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if pure.is_absolute() or ".." in pure.parts or stat.S_ISLNK(mode):
                raise ValueError("filebrowser ZIP contains an unsafe member")
        bundle.extractall(output)
    if (output / "model.safetensors.index.json").is_file():
        return output
    roots = [path for path in output.iterdir() if path.is_dir()]
    files = [path for path in output.iterdir() if path.is_file()]
    if len(roots) == 1 and not files and (roots[0] / "model.safetensors.index.json").is_file():
        return roots[0]
    raise ValueError("filebrowser ZIP does not contain one recognizable HF export root")


def _download_export(url: str, destination: Path, forwarded_user: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"X-Forwarded-User": forwarded_user})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("xb") as stream:
        if response.status != 200:
            raise ValueError(f"filebrowser returned HTTP {response.status}")
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    if size == 0:
        raise ValueError("filebrowser returned an empty response")
    return {"zip_bytes": size, "zip_sha256": "sha256:" + digest.hexdigest()}


def _weight_files(raw_root: Path) -> list[str]:
    index = _read(raw_root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("raw export has no safetensors weight map")
    names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and PurePosixPath(name).name == name for name in names):
        raise ValueError("raw export weight map contains an unsafe shard name")
    return ["model.safetensors.index.json", *names]


def compose_bundle(
    raw_root: Path,
    base_root: Path,
    output: Path,
    runtime_sidecars: Mapping[str, str],
) -> None:
    """Compose post weights/index with exact base runtime sidecars into an empty directory."""

    output.mkdir(mode=0o700)
    names = _weight_files(raw_root)
    if set(names) & set(runtime_sidecars):
        raise ValueError("runtime sidecars overlap the model weights/index")
    for name in names:
        source = raw_root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"raw export is missing regular weight file {name}")
        shutil.copyfile(source, output / name)
    for name, expected_sha256 in sorted(runtime_sidecars.items()):
        if PurePosixPath(name).name != name:
            raise ValueError("runtime sidecar names must be top-level files")
        source = base_root / name
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"base checkpoint is missing regular sidecar {name}")
        if file_sha256(source) != expected_sha256:
            raise ValueError(f"base runtime sidecar hash differs for {name}")
        shutil.copyfile(source, output / name)


def build_stage_input(
    plan: Mapping[str, Any],
    observation: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    *,
    tokenizer_evidence_sha256: str,
) -> dict[str, Any]:
    """Bind the post-export evidence needed by the inference-side transport job."""

    if observation.get("schema") != "fleet_sft_sfs_checkpoint_observation_v1":
        raise ValueError("unsupported post-export observation schema")
    if observation.get("run_name") != "ft-run-574bd7b3" or observation.get("step") != 318:
        raise ValueError("post-export observation names the wrong checkpoint")
    inspection = observation.get("output_inspection")
    if not isinstance(inspection, Mapping):
        raise ValueError("post-export observation has no HF output inspection")
    expected_source = plan["export"]["expected_output_path"]
    if inspection.get("root") != expected_source:
        raise ValueError("HF output inspection names an unexpected raw export path")
    rows = _manifest_rows(raw_manifest)
    if raw_manifest.get("root") != expected_source:
        raise ValueError("raw export manifest names an unexpected source path")
    if inspection.get("files_manifest_sha256") != raw_manifest.get("manifest_sha256"):
        raise ValueError("raw export inspection and full manifest disagree")
    if observation.get("raw_export_full_manifest_sha256") != raw_manifest.get(
        "manifest_sha256"
    ) or observation.get("raw_export_file_count") != raw_manifest.get("file_count"):
        raise ValueError("checkpoint observation does not bind the supplied raw export manifest")
    if observation.get("raw_export_total_bytes") != raw_manifest.get("total_bytes"):
        raise ValueError("checkpoint observation raw export byte count differs")
    evidence_binding = plan["base_model"]["tokenizer_equivalence_evidence"]
    if tokenizer_evidence_sha256 != evidence_binding.get("sha256"):
        raise ValueError("tokenizer equivalence evidence differs from the frozen plan")
    observation_sha256 = observation.get("observation_sha256")
    if not isinstance(observation_sha256, str) or not observation_sha256.startswith("sha256:"):
        raise ValueError("post-export observation must be digest-bound")
    undigested = {key: value for key, value in observation.items() if key != "observation_sha256"}
    if digest_json(undigested) != observation_sha256:
        raise ValueError("post-export observation digest does not validate")
    execution_plan = _validate_staging_execution_plan(plan.get("staging_execution"))
    source_url = FILEBROWSER_ORIGIN + "/api/resources/download?" + urllib.parse.urlencode(
        {"file": EXPORT_SOURCE, "source": "sfs"}
    )
    stage_input = {
        "schema": STAGE_SCHEMA,
        "source": {
            "sfs_path": expected_source,
            "filebrowser_url": source_url,
            "raw_full_manifest": dict(raw_manifest),
            "raw_inspection": dict(inspection),
            "observation_sha256": observation_sha256,
        },
        "composition": {
            "base_root": BASE_ROOT,
            "runtime_sidecar_sha256": plan["base_model"]["runtime_sidecar_sha256"],
            "tokenizer_equivalence_evidence_sha256": tokenizer_evidence_sha256,
            "expected_parameter_count": plan["base_model"]["parameter_count"],
            "expected_tokenizer_manifest_sha256": plan["base_model"][
                "tokenizer_manifest_sha256"
            ],
            "expected_chat_template_sha256": plan["base_model"]["chat_template_sha256"],
            "expected_config_sha256": plan["base_model"]["config_sha256"],
        },
        "execution": execution_plan,
        "destination": {"path": DESTINATION, "must_be_absent": True},
        "raw_file_count": len(rows),
    }
    stage_input["stage_input_sha256"] = digest_json(stage_input)
    return stage_input


def execute_stage(stage_input: Mapping[str, Any], *, work_root: Path, forwarded_user: str) -> dict:
    """Download, verify, compose, re-hash, and atomically promote one exact bundle."""

    if stage_input.get("schema") != STAGE_SCHEMA:
        raise ValueError("unsupported inference stage input schema")
    expected_input_sha256 = stage_input.get("stage_input_sha256")
    undigested = {key: value for key, value in stage_input.items() if key != "stage_input_sha256"}
    if digest_json(undigested) != expected_input_sha256:
        raise ValueError("inference stage input digest does not validate")
    if not forwarded_user.strip():
        raise ValueError("FILEBROWSER_USER is required")
    source = _mapping(stage_input.get("source"), "source")
    composition = _mapping(stage_input.get("composition"), "composition")
    destination = _mapping(stage_input.get("destination"), "destination")
    final = Path(str(destination.get("path")))
    if final != Path(DESTINATION) or destination.get("must_be_absent") is not True:
        raise ValueError("inference destination differs from the frozen path")
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    suffix = expected_input_sha256.removeprefix("sha256:")[:12]
    partial = parent / f".partial-{final.name}-{suffix}"
    if os.path.lexists(final):
        if os.path.lexists(partial):
            raise ValueError("both final and partial staging transaction paths exist")
        return _validate_committed_directory(
            final, final=final, expected_stage_input_sha256=str(expected_input_sha256)
        )
    if os.path.lexists(partial):
        _validate_committed_directory(
            partial, final=final, expected_stage_input_sha256=str(expected_input_sha256)
        )
        _fsync_directory(partial)
        _rename_noreplace(partial, final)
        _fsync_directory(parent)
        return _validate_committed_directory(
            final, final=final, expected_stage_input_sha256=str(expected_input_sha256)
        )
    base_root = Path(str(composition.get("base_root"))).resolve(strict=True)
    if base_root != Path(BASE_ROOT):
        raise ValueError("composition base root differs from the frozen revision")
    execution = _runtime_execution_provenance(stage_input)

    work_root.mkdir(parents=True, exist_ok=False)
    archive = work_root / "raw-export.zip"
    extracted = work_root / "raw-export"
    transport = _download_export(str(source.get("filebrowser_url")), archive, forwarded_user)
    raw_root = _safe_extract_zip(archive, extracted)
    raw_manifest = _mapping(source.get("raw_full_manifest"), "raw full manifest")
    raw_verified = verify_full_manifest(raw_root, raw_manifest)
    if raw_verified["manifest_sha256"] != source.get("raw_inspection", {}).get(
        "files_manifest_sha256"
    ):
        raise ValueError("downloaded raw export differs from its inspection receipt")

    compose_bundle(
        raw_root,
        base_root,
        partial,
        _mapping(composition.get("runtime_sidecar_sha256"), "runtime sidecar hashes"),
    )
    composed = inspect_hf_export(
        partial,
        expected_tokenizer_manifest_sha256=str(
            composition.get("expected_tokenizer_manifest_sha256")
        ),
        expected_chat_template_sha256=str(composition.get("expected_chat_template_sha256")),
        expected_config_sha256=str(composition.get("expected_config_sha256")),
        expected_parameter_count=int(composition.get("expected_parameter_count")),
        expected_sidecar_sha256=dict(composition["runtime_sidecar_sha256"]),
        require_base_sidecars=True,
    )
    # The verified partial is promoted byte-for-byte below. Record the durable destination rather
    # than leaking the intentionally ephemeral pre-promotion directory into downstream receipts.
    composed["root"] = str(final)
    raw_inspection = _mapping(source.get("raw_inspection"), "raw inspection")
    if composed["weights_manifest_sha256"] != raw_inspection.get("weights_manifest_sha256"):
        raise ValueError("composed bundle weights differ from the raw export")
    payload = _payload_manifest(partial)
    if payload["manifest_sha256"] != composed["files_manifest_sha256"]:
        raise ValueError("composed payload manifest differs from the HF inspection")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "stage_input_sha256": expected_input_sha256,
        "source_observation_sha256": source.get("observation_sha256"),
        "source_raw_manifest_sha256": raw_verified["manifest_sha256"],
        "execution": execution,
        "transport": transport,
        "composition": {
            "policy": "raw_post_weights_and_index_plus_exact_base_runtime_sidecars_v1",
            "base_root": str(base_root),
            "tokenizer_equivalence_evidence_sha256": composition.get(
                "tokenizer_equivalence_evidence_sha256"
            ),
            "inspection": composed,
        },
        "destination": {
            "path": str(final),
            "acceptance_receipt_path": str(final / ACCEPTANCE_RECEIPT_NAME),
            "payload_manifest_excludes": [ACCEPTANCE_RECEIPT_NAME],
            "payload_manifest_sha256": payload["manifest_sha256"],
            "payload_file_count": payload["file_count"],
            "payload_total_bytes": payload["total_bytes"],
            "atomic_transaction": "directory_rename_noreplace_v1",
            "atomic_promotion": True,
        },
    }
    receipt["staging_receipt_sha256"] = digest_json(receipt)
    acceptance_partial = partial / ACCEPTANCE_RECEIPT_NAME
    with acceptance_partial.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _validate_committed_directory(
        partial, final=final, expected_stage_input_sha256=str(expected_input_sha256)
    )
    _fsync_directory(partial)
    _rename_noreplace(partial, final)
    _fsync_directory(parent)
    return _validate_committed_directory(
        final, final=final, expected_stage_input_sha256=str(expected_input_sha256)
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-input")
    build.add_argument("--plan", type=Path, required=True)
    build.add_argument("--observation", type=Path, required=True)
    build.add_argument("--raw-manifest", type=Path, required=True)
    build.add_argument("--tokenizer-evidence", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    validate_bundle = subparsers.add_parser("validate-bundle")
    validate_bundle.add_argument("--plan", type=Path, required=True)
    validate_bundle.add_argument("--root", type=Path, required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--input", type=Path, required=True)
    execute.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    if args.command == "build-input":
        result = build_stage_input(
            _read(args.plan),
            _read(args.observation),
            _read(args.raw_manifest),
            tokenizer_evidence_sha256=file_sha256(args.tokenizer_evidence),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_stage_input_bytes(result))
        return
    if args.command == "validate-bundle":
        result = validate_local_staging_bundle(_read(args.plan), args.root)
        print(json.dumps(result, sort_keys=True))
        return
    work_root = args.work_root
    if work_root is None:
        work_root = Path(tempfile.mkdtemp(prefix="post-sft-stage-"))
        work_root.rmdir()
    receipt = execute_stage(
        _read(args.input),
        work_root=work_root,
        forwarded_user=os.environ.get("FILEBROWSER_USER", ""),
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
