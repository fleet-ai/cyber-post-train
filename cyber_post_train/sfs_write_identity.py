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
from pathlib import Path, PurePosixPath

SFS_JOBS_ROOT = PurePosixPath("/mnt/sfs/jobs")
LORA_CONTROL_ROOT = SFS_JOBS_ROOT / "chris-q38-study-corpora-v1" / "launch-controls"
TRAINER_UID = 1000
TRAINER_GID = 100
DEV_GPU_RELOAD_OPERATION_SUFFIX = "-dev-gpu-reload"
DEV_GPU_CONTROL_MOUNT = PurePosixPath("/controls")
DEV_GPU_CONTROL_SUBPATH = str(LORA_CONTROL_ROOT.relative_to(PurePosixPath("/mnt/sfs")))
_CONTROL = re.compile(r"\.preflight-control-[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")


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


def validate_direct_dev_gpu_reload_output(manifest: dict) -> None:
    """Bind one direct dev reload to the proven non-root SFS output shape."""

    if not is_direct_dev_gpu_reload_pod(manifest):
        raise ValueError("direct dev GPU reload must be one labeled v1 Pod")
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
