"""Atomic score-blind endpoint stream leases on the shared campaign PVC."""

from __future__ import annotations

import fcntl
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

LEASE_KEY = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")


@dataclass
class EndpointLease:
    """A held advisory lock; closing it atomically releases the stream slot."""

    endpoint_key: str
    slot: int
    path: Path
    handle: BinaryIO

    def close(self) -> None:
        if not self.handle.closed:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def __enter__(self) -> EndpointLease:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def acquire_endpoint_lease(
    *, lease_root: Path, endpoint_key: str, maximum_streams: int
) -> EndpointLease:
    """Acquire one of a fixed number of endpoint slots without a check/create race."""

    if not LEASE_KEY.fullmatch(endpoint_key):
        raise ValueError("endpoint lease key is invalid")
    if type(maximum_streams) is not int or maximum_streams not in {1, 2}:
        raise ValueError("endpoint stream limit must be one or two")
    if lease_root.exists() and (lease_root.is_symlink() or not lease_root.is_dir()):
        raise RuntimeError("endpoint lease root is unsafe")
    lease_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    endpoint_root = lease_root / endpoint_key
    if endpoint_root.exists() and (endpoint_root.is_symlink() or not endpoint_root.is_dir()):
        raise RuntimeError("endpoint lease directory is unsafe")
    endpoint_root.mkdir(mode=0o700, exist_ok=True)

    for slot in range(1, maximum_streams + 1):
        path = endpoint_root / f"slot-{slot}.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags, 0o600)
        except OSError:
            continue
        handle = os.fdopen(fd, "a+b")
        try:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise RuntimeError("endpoint lease slot is not a regular file")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, RuntimeError):
            handle.close()
            continue
        return EndpointLease(endpoint_key, slot, path, handle)
    raise RuntimeError("endpoint stream cap is already full")
