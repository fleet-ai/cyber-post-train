"""Stage immutable JSON evidence from projected/SFS inputs without shell races."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_receipt(
    path: Path, *, file_sha256: str | None = None, projected_root: Path | None = None
) -> dict[str, Any]:
    if path.is_symlink() and projected_root is not None:
        resolved = path.resolve(strict=True)
        root = projected_root.resolve(strict=True)
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"projected evidence escapes its mount: {path}")
        path = resolved
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"evidence is not a regular file: {path}")
    data = path.read_bytes()
    if not data:
        raise ValueError(f"evidence is empty: {path}")
    if file_sha256 is not None and _sha256(data) != file_sha256:
        raise ValueError(f"evidence file digest drifted: {path}")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"evidence is not an object: {path}")
    claimed = value.get("receipt_sha256")
    if claimed is not None:
        body = dict(value)
        body.pop("receipt_sha256")
        if claimed != _sha256(_canonical(body)):
            raise ValueError(f"evidence self digest drifted: {path}")
    return value


def write_regular_once(path: Path, value: dict[str, Any]) -> None:
    """Publish validated evidence atomically; never leave an empty target."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    data = _canonical(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stage(
    *,
    projected_root: Path,
    sfs_release: Path,
    predecessor: Path,
    output_root: Path,
    release_file_sha256: str,
) -> dict[str, Any]:
    parity = load_receipt(projected_root / "parity.json", projected_root=projected_root)
    binding = load_receipt(projected_root / "binding.json", projected_root=projected_root)
    release = load_receipt(sfs_release, file_sha256=release_file_sha256)
    predecessor_receipt = load_receipt(predecessor)
    if release.get("predecessor_acceptance_receipt_sha256") != predecessor_receipt.get(
        "receipt_sha256"
    ):
        raise ValueError("release predecessor authority drifted")
    for name, value in (
        ("parity.json", parity),
        ("binding.json", binding),
        ("release.json", release),
    ):
        write_regular_once(output_root / name, value)
    return {
        "release_receipt_sha256": release.get("receipt_sha256"),
        "predecessor_receipt_sha256": predecessor_receipt.get("receipt_sha256"),
    }
