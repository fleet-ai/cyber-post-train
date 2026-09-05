"""Safely materialize a projected Kubernetes runtime plan as a regular file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def materialize(source: Path, destination: Path, *, schema: str) -> dict[str, Any]:
    """Validate a projected plan and copy it once to a non-symlink destination."""
    if not source.is_file():
        raise ValueError("projected runtime plan is absent")
    raw = source.read_bytes()
    value = json.loads(raw, object_pairs_hook=_strict_object)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError("projected runtime plan schema drifted")
    stated = value.get("plan_sha256")
    body = {key: item for key, item in value.items() if key != "plan_sha256"}
    actual = "sha256:" + hashlib.sha256(_canonical(body)).hexdigest()
    if stated != actual:
        raise ValueError("projected runtime plan self-digest drifted")

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("materialized runtime plan is not a regular file")
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(_canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError("materialized runtime plan target is unsafe")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--schema", required=True)
    args = parser.parse_args()
    materialize(args.source, args.destination, schema=args.schema)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

