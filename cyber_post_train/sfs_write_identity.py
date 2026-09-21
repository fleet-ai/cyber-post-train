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
