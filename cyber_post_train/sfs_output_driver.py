"""Standard-library worker for a read-only SFS output-absence observation."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

RECEIPT_SCHEMA = "cyber_sft_output_absence_v1"
LOG_PREFIX = "CYBER_SFT_OUTPUT_ABSENCE="
ENV_DRIVER_SHA256 = "CYBER_SFS_DRIVER_SHA256"
ENV_DRIVER_SOURCE = "CYBER_SFS_DRIVER_SOURCE_B64"
ENV_PLAN_SHA256 = "CYBER_SFS_PLAN_SHA256"
ENV_REQUEST_SHA256 = "CYBER_SFS_REQUEST_SHA256"
ENV_RUN_NAME = "CYBER_SFS_RUN_NAME"
ENV_RUN_DIR = "CYBER_SFS_RUN_DIR"
REQUIRED_ENV = (
    ENV_DRIVER_SHA256,
    ENV_DRIVER_SOURCE,
    ENV_PLAN_SHA256,
    ENV_REQUEST_SHA256,
    ENV_RUN_NAME,
    ENV_RUN_DIR,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RUN_NAME = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?")


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def bound_driver_sha256(encoded: str) -> str:
    try:
        source = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("output-check driver source is not canonical base64") from exc
    return hashlib.sha256(source).hexdigest()


def exact_run_dir(value: str, run_name: str) -> Path:
    path = PurePosixPath(value)
    if (
        _RUN_NAME.fullmatch(run_name) is None
        or path.parts != ("/", "mnt", "sfs", "jobs", run_name)
        or str(path) != value
    ):
        raise ValueError("run output is not the exact owned /mnt/sfs/jobs path")
    return Path(value)


def observe(environ: Mapping[str, str] | None = None) -> dict:
    values = os.environ if environ is None else environ
    if set(values).issuperset(REQUIRED_ENV) is False or any(
        not isinstance(values.get(name), str) or not values[name] for name in REQUIRED_ENV
    ):
        raise ValueError("output-check environment is incomplete")
    if any(
        _SHA256.fullmatch(values[name]) is None
        for name in (ENV_DRIVER_SHA256, ENV_PLAN_SHA256, ENV_REQUEST_SHA256)
    ):
        raise ValueError("output-check digest binding is invalid")
    if values[ENV_DRIVER_SHA256] != bound_driver_sha256(values[ENV_DRIVER_SOURCE]):
        raise ValueError("executing output-check driver differs from its bound bytes")
    if os.getuid() != 1000 or os.getgid() != 100:
        raise ValueError("output check must run as the reviewed non-root identity")
    if values.get("CUDA_VISIBLE_DEVICES") != "" or values.get("NVIDIA_VISIBLE_DEVICES") != "none":
        raise ValueError("output check must have no visible GPU")

    run_name = values[ENV_RUN_NAME]
    run_dir = exact_run_dir(values[ENV_RUN_DIR], run_name)
    parent_status = os.lstat(run_dir.parent)
    if not stat.S_ISDIR(parent_status.st_mode) or stat.S_ISLNK(parent_status.st_mode):
        raise ValueError("shared SFS jobs root is not one real directory")
    try:
        os.lstat(run_dir)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("bound training output already exists")
    proof = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "checked_at_epoch": int(time.time()),
        "plan_sha256": values[ENV_PLAN_SHA256],
        "request_sha256": values[ENV_REQUEST_SHA256],
        "run_name": run_name,
        "run_dir": str(run_dir),
        "sfs_jobs_root": "/mnt/sfs/jobs",
        "output_absent": True,
    }
    return {**proof, "sha256": digest(proof)}


def main() -> None:
    os.umask(0o077)
    print(LOG_PREFIX + canonical_json(observe()).decode(), flush=True)


if __name__ == "__main__":
    main()
