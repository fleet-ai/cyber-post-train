"""Validate a held-out ledger before publishing terminal-observer evidence.

This module deliberately owns no Kubernetes or PostgreSQL mutation.  It binds
score-blind ledger identity rows to the plan compiled from one sealed launch
packet, then publishes caller-supplied evidence only after every validation has
finished.  A copied harness identifier therefore cannot leave a partial final
output directory behind.
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from evals.fleet import heldout_launch

PLAN_ROW_FIELDS = (
    "experiment_id",
    "task_key",
    "task_version_id",
    "model_id",
    "model_revision",
    "serving_block",
    "endpoint_model_id",
    "harness_id",
    "attempt",
    "max_retries",
)


class HeldoutObserverError(ValueError):
    """A sealed-plan validation or create-once publication failed."""


@dataclass(frozen=True)
class PlanBinding:
    """The score-blind identity proven from one sealed launch packet."""

    evaluation_identity_sha256: str
    evaluation_plan_sha256: str
    harness_id: str
    row_count: int


def _identity_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise HeldoutObserverError(f"{label} rows are missing")
    normalized: list[dict[str, Any]] = []
    expected_fields = set(PLAN_ROW_FIELDS)
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise HeldoutObserverError(f"{label} row fields are not score-blind plan identity")
        if any(
            not isinstance(row[field], str) or not row[field]
            for field in PLAN_ROW_FIELDS
            if field not in {"attempt", "max_retries"}
        ) or any(
            type(row[field]) is not int or row[field] < 0 for field in ("attempt", "max_retries")
        ):
            raise HeldoutObserverError(f"{label} row identity values are invalid")
        normalized.append({field: row[field] for field in PLAN_ROW_FIELDS})
    identities = {tuple(row[field] for field in PLAN_ROW_FIELDS) for row in normalized}
    if len(identities) != len(normalized):
        raise HeldoutObserverError(f"{label} rows contain duplicate identities")
    return sorted(
        normalized,
        key=lambda row: tuple(str(row[field]) for field in PLAN_ROW_FIELDS),
    )


def validate_plan_rows(
    packet_path: Path,
    observed_rows: Sequence[Mapping[str, Any]],
) -> PlanBinding:
    """Require database identity to equal the packet's freshly compiled plan.

    ``harness_id`` is never supplied independently.  The evaluator defines it
    as ``protocol-<compiled-plan-sha256>``; recomputing that plan from the
    packet prevents a stale value copied from another arm or campaign.
    """

    try:
        package = heldout_launch.build_package(packet_path)
        sealed = heldout_launch.sealed_evaluation(package)
        plan = sealed.plan
        expected_rows = _identity_rows(sealed.rows, label="sealed plan")
        actual_rows = _identity_rows(observed_rows, label="observed database")
    except HeldoutObserverError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise HeldoutObserverError(
            "sealed launch packet cannot reproduce its evaluator plan"
        ) from exc
    if actual_rows != expected_rows:
        raise HeldoutObserverError("observed database identity differs from the sealed plan")
    expected_harness = "protocol-" + plan["sha256"]
    harnesses = {row["harness_id"] for row in expected_rows}
    if harnesses != {expected_harness}:
        raise HeldoutObserverError("compiled plan does not have one exact harness identity")
    return PlanBinding(
        evaluation_identity_sha256=package.packet.identity_sha256,
        evaluation_plan_sha256=plan["sha256"],
        harness_id=expected_harness,
        row_count=len(expected_rows),
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing another claimant."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_raw = os.fsencode(source)
    destination_raw = os.fsencode(destination)
    if hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_raw, -100, destination_raw, 1)
    elif hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_raw, destination_raw, 0x00000004)
    else:
        raise HeldoutObserverError("atomic create-once publication is unavailable")
    if result == 0:
        return
    code = ctypes.get_errno()
    if code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise HeldoutObserverError("observer output was claimed concurrently")
    raise HeldoutObserverError(f"atomic observer publication failed with errno {code}")


def _artifact_payloads(value: Mapping[str, bytes]) -> dict[str, bytes]:
    if not isinstance(value, Mapping) or not value:
        raise HeldoutObserverError("observer artifacts are missing")
    artifacts: dict[str, bytes] = {}
    for name, payload in value.items():
        path = PurePosixPath(name) if isinstance(name, str) else None
        if (
            path is None
            or path.is_absolute()
            or len(path.parts) != 1
            or path.name in {"", ".", ".."}
            or "\x00" in path.name
            or not isinstance(payload, bytes)
            or path.name in artifacts
        ):
            raise HeldoutObserverError("observer artifact name or payload is invalid")
        artifacts[path.name] = payload
    return artifacts


def validate_then_publish(
    packet_path: Path,
    observed_rows: Sequence[Mapping[str, Any]],
    output_root: Path,
    build_artifacts: Callable[[PlanBinding], Mapping[str, bytes]],
) -> PlanBinding:
    """Validate all inputs, then atomically publish one private evidence root.

    ``build_artifacts`` runs before a staging or final directory is created, so
    callers can finish terminal-receipt and audit validation there.  Every
    artifact is mode 0600 and the final directory is an atomic no-replace
    rename on the same filesystem.
    """

    if output_root.is_symlink() or output_root.exists():
        raise HeldoutObserverError("observer output already exists")
    if output_root.parent.is_symlink() or not output_root.parent.is_dir():
        raise HeldoutObserverError("observer output parent is not an exact directory")
    binding = validate_plan_rows(packet_path, observed_rows)
    artifacts = _artifact_payloads(build_artifacts(binding))
    if output_root.is_symlink() or output_root.exists():
        raise HeldoutObserverError("observer output appeared during validation")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    published = False
    try:
        os.chmod(staging, 0o700)
        for name in sorted(artifacts):
            _write_exclusive(staging / name, artifacts[name])
        _fsync_directory(staging)
        _rename_noreplace(staging, output_root)
        published = True
        _fsync_directory(output_root.parent)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
    return binding
