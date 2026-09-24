"""Narrow ownership boundary for non-root CPU controls on shared SFS.

The shared ``/mnt/sfs/jobs`` directory is not a per-job workspace.  A pinned
trainer running as UID 1000/GID 100 may write only below an existing run tree
that has already been proven to belong to that identity.  This module binds the
lexical path before submission and rechecks the live directory identity inside
the pinned image before any output is created.
"""

from __future__ import annotations

import os
import re
import stat
from copy import deepcopy
from pathlib import Path, PurePosixPath

SFS_JOBS_ROOT = PurePosixPath("/mnt/sfs/jobs")
LORA_CONTROL_ROOT = SFS_JOBS_ROOT / "chris-q38-study-corpora-v1" / "launch-controls"
TRAINER_UID = 1000
TRAINER_GID = 100
DEV_GPU_RELOAD_OPERATION_SUFFIX = "-dev-gpu-reload"
DEV_GPU_RUN_NAME_LABEL = "fleet.ai/run-name"
DEV_GPU_OWNER_PREFIXES = ("chris-q38-",)
DEV_GPU_NODE_SELECTOR = {
    "kubernetes.io/arch": "amd64",
    "workload": "fleetai-training-ng-gpu",
}
DEV_GPU_TOLERATIONS = [
    {
        "effect": "NoSchedule",
        "key": "workload",
        "operator": "Equal",
        "value": "fleetai-training-ng-gpu",
    }
]
DEV_GPU_ADMISSION_TOLERATIONS = [
    {
        "effect": "NoExecute",
        "key": "node.kubernetes.io/not-ready",
        "operator": "Exists",
        "tolerationSeconds": 300,
    },
    {
        "effect": "NoExecute",
        "key": "node.kubernetes.io/unreachable",
        "operator": "Exists",
        "tolerationSeconds": 300,
    },
    {
        "effect": "NoSchedule",
        "key": "nvidia.com/gpu",
        "operator": "Exists",
    },
]
DEV_GPU_CONTROL_MOUNT = PurePosixPath("/controls")
DEV_GPU_CONTROL_SUBPATH = str(LORA_CONTROL_ROOT.relative_to(PurePosixPath("/mnt/sfs")))
_CONTROL = re.compile(r"\.preflight-control-[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
_LABEL_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")


def validate_owned_output_binding(owned_root: str, output_root: str) -> None:
    """Require one hidden control transaction directly below one owned run tree."""

    owned = PurePosixPath(owned_root)
    output = PurePosixPath(output_root)
    if not owned.is_absolute() or ".." in owned.parts or owned != LORA_CONTROL_ROOT:
        raise ValueError("SFS control parent must be the reviewed owned launch-controls tree")
    if (
        not output.is_absolute()
        or ".." in output.parts
        or output.parent != owned
        or _CONTROL.fullmatch(output.name) is None
    ):
        raise ValueError("SFS control output must be one direct hidden child of the owned tree")


def is_direct_dev_gpu_reload_pod(manifest: dict) -> bool:
    """Identify the narrowly named direct dev GPU reload Pod surface."""

    metadata = manifest.get("metadata")
    labels = metadata.get("labels") if isinstance(metadata, dict) else None
    operation = labels.get("operation") if isinstance(labels, dict) else None
    return (
        manifest.get("apiVersion") == "v1"
        and manifest.get("kind") == "Pod"
        and isinstance(operation, str)
        and operation.endswith(DEV_GPU_RELOAD_OPERATION_SUFFIX)
    )


def render_direct_dev_gpu_reload_pod(template: dict, *, run_name: str) -> dict:
    """Render the Pod name and canonical capacity-owner label from one authority."""

    if (
        not isinstance(run_name, str)
        or _LABEL_NAME.fullmatch(run_name) is None
        or not run_name.startswith(DEV_GPU_OWNER_PREFIXES)
    ):
        raise ValueError("direct dev GPU reload run name is not a canonical project identity")
    manifest = deepcopy(template)
    metadata = manifest.get("metadata")
    labels = metadata.get("labels") if isinstance(metadata, dict) else None
    if not isinstance(labels, dict):
        raise ValueError("direct dev GPU reload template metadata is malformed")
    metadata["name"] = run_name
    labels[DEV_GPU_RUN_NAME_LABEL] = run_name
    validate_direct_dev_gpu_reload_output(manifest)
    return manifest


def validate_direct_dev_gpu_reload_output(manifest: dict) -> None:
    """Bind one direct dev reload to the proven non-root SFS output shape."""

    if not is_direct_dev_gpu_reload_pod(manifest):
        raise ValueError("direct dev GPU reload must be one labeled v1 Pod")
    metadata = manifest["metadata"]
    labels = metadata["labels"]
    name = metadata.get("name")
    if (
        not isinstance(name, str)
        or _LABEL_NAME.fullmatch(name) is None
        or not name.startswith(DEV_GPU_OWNER_PREFIXES)
        or labels.get(DEV_GPU_RUN_NAME_LABEL) != name
    ):
        raise ValueError(
            "direct dev GPU reload must use its canonical fleet.ai/run-name ownership label"
        )
    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("direct dev GPU reload Pod spec is malformed")
    containers = spec.get("containers")
    if (
        not isinstance(containers, list)
        or len(containers) != 1
        or not isinstance(containers[0], dict)
        or spec.get("initContainers") not in (None, [])
    ):
        raise ValueError("direct dev GPU reload must use one container and no initializer")
    if spec.get("securityContext") != {
        "runAsNonRoot": True,
        "runAsUser": TRAINER_UID,
        "runAsGroup": TRAINER_GID,
        "fsGroup": TRAINER_GID,
    }:
        raise ValueError("direct dev GPU reload must use the proven UID 1000 SFS identity")
    container = containers[0]
    resources = container.get("resources")
    requests = resources.get("requests") if isinstance(resources, dict) else None
    limits = resources.get("limits") if isinstance(resources, dict) else None
    requested_gpu = requests.get("nvidia.com/gpu") if isinstance(requests, dict) else None
    limited_gpu = limits.get("nvidia.com/gpu") if isinstance(limits, dict) else None
    if (
        isinstance(requested_gpu, bool)
        or isinstance(limited_gpu, bool)
        or str(requested_gpu) != "1"
        or str(limited_gpu) != "1"
    ):
        raise ValueError("direct dev GPU reload must request and limit exactly one GPU")
    if container.get("securityContext") != {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    }:
        raise ValueError("direct dev GPU reload must not override or elevate its SFS identity")
    if container.get("envFrom") not in (None, []):
        raise ValueError("direct dev GPU reload environment must not be imported")
    environment = {}
    for entry in container.get("env", []):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"name", "value"}
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("value"), str)
            or entry["name"] in environment
        ):
            raise ValueError("direct dev GPU reload environment must contain literal unique values")
        environment[entry["name"]] = entry["value"]
    run_dir = PurePosixPath(environment.get("RUN_DIR", ""))
    if ".." in run_dir.parts or run_dir.parent != DEV_GPU_CONTROL_MOUNT:
        raise ValueError("direct dev GPU reload RUN_DIR must be one writable /controls leaf")
    try:
        validate_owned_output_binding(str(LORA_CONTROL_ROOT), str(LORA_CONTROL_ROOT / run_dir.name))
    except ValueError as exc:
        raise ValueError(str(exc)) from None
    if container.get("volumeMounts") != [
        {"name": "sfs-readonly", "mountPath": "/mnt/sfs", "readOnly": True},
        {
            "name": "sfs-control",
            "mountPath": str(DEV_GPU_CONTROL_MOUNT),
            "subPath": DEV_GPU_CONTROL_SUBPATH,
        },
    ]:
        raise ValueError(
            "direct dev GPU reload must expose read-only SFS plus "
            "the exact writable control subpath"
        )
    if spec.get("volumes") != [
        {"name": "sfs-readonly", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
        {"name": "sfs-control", "persistentVolumeClaim": {"claimName": "sfs-shared"}},
    ]:
        raise ValueError("direct dev GPU reload must use only the shared SFS claim")
    if spec.get("nodeSelector") != DEV_GPU_NODE_SELECTOR:
        raise ValueError("direct dev GPU reload must select the reviewed GPU pool")
    tolerations = spec.get("tolerations")
    if not isinstance(tolerations, list) or DEV_GPU_TOLERATIONS[0] not in tolerations:
        raise ValueError("direct dev GPU reload must tolerate the reviewed GPU-pool taint")
    allowed_tolerations = DEV_GPU_TOLERATIONS + DEV_GPU_ADMISSION_TOLERATIONS
    if len(tolerations) != len({repr(sorted(item.items())) for item in tolerations}) or any(
        item not in allowed_tolerations for item in tolerations
    ):
        raise ValueError("direct dev GPU reload tolerations contain unreviewed drift")


def verify_owned_output_runtime(
    owned_root: Path,
    output_root: Path,
    *,
    writable_mount: Path | None = None,
    uid: int = TRAINER_UID,
    gid: int = TRAINER_GID,
) -> dict[str, int | str]:
    """Recheck the exact non-root identity and live parent without changing it."""

    validate_owned_output_binding(str(owned_root), str(output_root))
    if (os.geteuid(), os.getegid()) != (uid, gid):
        raise ValueError("SFS control must run as the pinned trainer identity")
    identity = owned_root.lstat()
    mode = stat.S_IMODE(identity.st_mode)
    if (
        not stat.S_ISDIR(identity.st_mode)
        or owned_root.is_symlink()
        or (identity.st_uid, identity.st_gid) != (uid, gid)
        or mode & stat.S_IWUSR == 0
        or mode & stat.S_IXUSR == 0
    ):
        raise ValueError("SFS control parent is not a writable directory owned by the trainer")
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError("SFS control output already exists")
    receipt: dict[str, int | str] = {
        "owned_root": str(owned_root),
        "output_root": str(output_root),
        "uid": identity.st_uid,
        "gid": identity.st_gid,
        "mode": mode,
    }
    if writable_mount is not None:
        mounted = writable_mount.lstat()
        if (
            writable_mount.is_symlink()
            or not stat.S_ISDIR(mounted.st_mode)
            or (mounted.st_dev, mounted.st_ino) != (identity.st_dev, identity.st_ino)
            or (mounted.st_uid, mounted.st_gid) != (uid, gid)
        ):
            raise ValueError("writable SFS mount is not the exact reviewed control parent")
        receipt["writable_mount"] = str(writable_mount)
    return receipt
