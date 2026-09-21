"""Create-once SFS output checks shared by normal and direct submission."""

from __future__ import annotations

import time
from pathlib import Path

from .jobs import digest

SFS_JOBS_ROOT = Path("/mnt/sfs/jobs")
OUTPUT_ABSENCE_SCHEMA = "cyber_sft_output_absence_v1"
OUTPUT_ABSENCE_MAX_AGE_SECONDS = 300


def require_output_absent(request: dict, *, jobs_root: Path = SFS_JOBS_ROOT) -> None:
    """Prove the create-once output is absent from a host that can see SFS."""
    if not jobs_root.is_dir():
        raise ValueError("the shared /mnt/sfs/jobs mount is unavailable for output checks")
    output = Path(request["run_dir"])
    try:
        relative_output = output.relative_to(SFS_JOBS_ROOT)
    except ValueError:
        try:
            relative_output = output.relative_to(jobs_root)
        except ValueError:
            raise ValueError("training output is outside the shared /mnt/sfs/jobs mount") from None
    observed_output = jobs_root / relative_output
    if observed_output.exists() or observed_output.is_symlink():
        raise ValueError("training output already exists; use a new reviewed run identity")


def build_output_absence_receipt(
    plan: dict,
    request: dict,
    *,
    jobs_root: Path = SFS_JOBS_ROOT,
    now: float | None = None,
) -> dict:
    """Record a source-bound SFS observation for a remote submitter."""
    require_output_absent(request, jobs_root=jobs_root)
    checked_at_epoch = int(time.time() if now is None else now)
    if checked_at_epoch < 0:
        raise ValueError("output-absence receipt time is invalid")
    proof = {
        "schema": OUTPUT_ABSENCE_SCHEMA,
        "status": "passed",
        "checked_at_epoch": checked_at_epoch,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "run_name": request["name"],
        "run_dir": request["run_dir"],
        "sfs_jobs_root": str(SFS_JOBS_ROOT),
        "output_absent": True,
    }
    return {**proof, "sha256": digest(proof)}


def validate_output_absence_receipt(
    receipt: dict,
    plan: dict,
    request: dict,
    *,
    now: float | None = None,
) -> dict:
    """Require an exact, self-digested receipt no more than five minutes old."""
    if not isinstance(receipt, dict):
        raise ValueError("SFS output-absence receipt is not an object")
    checked_at_epoch = receipt.get("checked_at_epoch")
    if type(checked_at_epoch) is not int:
        raise ValueError("SFS output-absence receipt time is invalid")
    expected = {
        "schema": OUTPUT_ABSENCE_SCHEMA,
        "status": "passed",
        "checked_at_epoch": checked_at_epoch,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "run_name": request["name"],
        "run_dir": request["run_dir"],
        "sfs_jobs_root": str(SFS_JOBS_ROOT),
        "output_absent": True,
    }
    if set(receipt) != {*expected, "sha256"} or any(
        receipt.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("SFS output-absence receipt is not bound to this prepared run")
    if receipt["sha256"] != digest(expected):
        raise ValueError("SFS output-absence receipt digest is invalid")
    age = (time.time() if now is None else now) - checked_at_epoch
    if not 0 <= age <= OUTPUT_ABSENCE_MAX_AGE_SECONDS:
        raise ValueError("SFS output-absence receipt is stale")
    return receipt


def prove_output_absent(
    plan: dict,
    request: dict,
    *,
    jobs_root: Path = SFS_JOBS_ROOT,
    receipt: dict | None = None,
    now: float | None = None,
) -> dict:
    """Use the live mount when available, otherwise require a fresh receipt."""
    if jobs_root.is_dir():
        return build_output_absence_receipt(plan, request, jobs_root=jobs_root, now=now)
    if receipt is None:
        raise ValueError(
            "the shared SFS mount is unavailable; provide a fresh source-bound "
            "output-absence receipt"
        )
    return validate_output_absence_receipt(receipt, plan, request, now=now)
