"""Executable P3/P4 guards for the non-root LoRA CPU preflight.

This file is projected into an immutable ConfigMap as ``preflight_driver.py``
alongside the two small guard modules it invokes.  Keep it standard-library
only so the pinned trainer image does not need this repository installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

if __package__:
    from .sfs_write_identity import verify_owned_output_runtime
    from .source_bundle import verify_source_archive_commit
else:  # pragma: no cover - exercised from the projected ConfigMap in production
    from sfs_write_identity import verify_owned_output_runtime
    from source_bundle import verify_source_archive_commit

BUNDLE_FILES = (
    "preflight_driver.py",
    "sfs_write_identity.py",
    "source_bundle.py",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")

ENV_BUNDLE_SHA256 = "CYBER_PREFLIGHT_BUNDLE_SHA256"
ENV_SOURCE_ARCHIVE = "CYBER_SOURCE_ARCHIVE"
ENV_SOURCE_ARCHIVE_SHA256 = "CYBER_SOURCE_ARCHIVE_SHA256"
ENV_SOURCE_COMMIT = "CYBER_SOURCE_COMMIT"
ENV_SFS_OWNED_ROOT = "CYBER_SFS_OWNED_ROOT"
ENV_SFS_OUTPUT_ROOT = "CYBER_SFS_OUTPUT_ROOT"
ENV_SFS_CONTROL_MOUNT = "CYBER_SFS_CONTROL_MOUNT"
REQUIRED_ENV = (
    ENV_BUNDLE_SHA256,
    ENV_SOURCE_ARCHIVE,
    ENV_SOURCE_ARCHIVE_SHA256,
    ENV_SOURCE_COMMIT,
    ENV_SFS_OWNED_ROOT,
    ENV_SFS_OUTPUT_ROOT,
    ENV_SFS_CONTROL_MOUNT,
)


def bundle_sha256(files: Mapping[str, bytes]) -> str:
    """Hash the exact named source bytes projected into the ConfigMap."""

    if set(files) != set(BUNDLE_FILES) or any(
        not isinstance(value, bytes) for value in files.values()
    ):
        raise ValueError("preflight bundle files differ from the reviewed source set")
    file_hashes = {name: hashlib.sha256(files[name]).hexdigest() for name in BUNDLE_FILES}
    encoded = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_mounted_bundle(root: Path, expected_sha256: str) -> str:
    """Require the running driver and both imported guards to match packaging."""

    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("preflight bundle digest is invalid")
    actual = bundle_sha256({name: (root / name).read_bytes() for name in BUNDLE_FILES})
    if actual != expected_sha256:
        raise ValueError("mounted preflight bundle differs from the reviewed package")
    return actual


def run_preflight(
    environ: Mapping[str, str] | None = None, *, bundle_root: Path | None = None
) -> dict:
    """Run both source and live SFS identity guards before any output exists."""

    values = os.environ if environ is None else environ
    missing = [
        name for name in REQUIRED_ENV if not isinstance(values.get(name), str) or not values[name]
    ]
    if missing:
        raise ValueError("LoRA CPU preflight environment is incomplete")

    mounted_bundle_sha256 = verify_mounted_bundle(
        bundle_root or Path(__file__).resolve().parent,
        values[ENV_BUNDLE_SHA256],
    )
    expected_archive_sha256 = values[ENV_SOURCE_ARCHIVE_SHA256]
    if _SHA256.fullmatch(expected_archive_sha256) is None:
        raise ValueError("source archive digest is invalid")
    source = verify_source_archive_commit(
        Path(values[ENV_SOURCE_ARCHIVE]),
        values[ENV_SOURCE_COMMIT],
    )
    if source["source_archive_sha256"] != expected_archive_sha256:
        raise ValueError("source archive bytes differ from the reviewed package")

    output_identity = verify_owned_output_runtime(
        Path(values[ENV_SFS_OWNED_ROOT]),
        Path(values[ENV_SFS_OUTPUT_ROOT]),
        writable_mount=Path(values[ENV_SFS_CONTROL_MOUNT]),
    )
    return {
        "schema": "cyber_qwen38_lora_cpu_preflight_boundary_v1",
        "preflight_bundle_sha256": mounted_bundle_sha256,
        "source": source,
        "output_identity": output_identity,
    }


def main() -> None:
    print(json.dumps(run_preflight(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
