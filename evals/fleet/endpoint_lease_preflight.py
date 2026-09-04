"""Bounded cross-Pod flock proof for the shared endpoint-lease PVC."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = "fleet-endpoint-lease-cross-pod-preflight-v1"
WAIT_ATTEMPTS = 600


def _digest(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("receipt_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_once(path: Path, value: dict[str, Any]) -> None:
    value["receipt_sha256"] = _digest(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")


def _read_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("endpoint lease preflight receipt is unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("receipt_sha256") != _digest(value):
        raise RuntimeError("endpoint lease preflight receipt drifted")
    return value


def _wait_receipt(path: Path) -> dict[str, Any]:
    for _ in range(WAIT_ATTEMPTS):
        if path.exists():
            return _read_receipt(path)
        time.sleep(1)
    raise RuntimeError("endpoint lease preflight timed out")


def _identity() -> tuple[str, str]:
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    uuid.UUID(job_uid)
    uuid.UUID(pod_uid)
    return job_uid, pod_uid


def hold(root: Path) -> None:
    job_uid, pod_uid = _identity()
    if root.exists():
        raise FileExistsError("endpoint lease preflight root already exists")
    root.mkdir(mode=0o700, parents=True)
    lock_path = root / "slot-1.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ready = {
            "schema_version": SCHEMA,
            "phase": "holder_ready",
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "lock_held": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        _write_once(root / "HOLDER-READY.json", ready)
        contended = _wait_receipt(root / "CONTENTION.json")
        if (
            contended.get("schema_version") != SCHEMA
            or contended.get("phase") != "cross_pod_contention_observed"
            or contended.get("holder_ready_sha256") != ready["receipt_sha256"]
            or contended.get("lock_acquired_while_holder_active") is not False
        ):
            raise RuntimeError("endpoint lease contention evidence drifted")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        released = {
            "schema_version": SCHEMA,
            "phase": "holder_released",
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "holder_ready_sha256": ready["receipt_sha256"],
            "contention_receipt_sha256": contended["receipt_sha256"],
            "lock_released": True,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        _write_once(root / "HOLDER-RELEASED.json", released)
        terminal = _wait_receipt(root / "TERMINAL.json")
        if (
            terminal.get("schema_version") != SCHEMA
            or terminal.get("phase") != "cross_pod_flock_passed"
            or terminal.get("holder_released_sha256") != released["receipt_sha256"]
        ):
            raise RuntimeError("endpoint lease terminal evidence drifted")


def probe(root: Path) -> None:
    job_uid, pod_uid = _identity()
    ready = _wait_receipt(root / "HOLDER-READY.json")
    if (
        ready.get("schema_version") != SCHEMA
        or ready.get("phase") != "holder_ready"
        or ready.get("lock_held") is not True
    ):
        raise RuntimeError("endpoint lease holder evidence drifted")
    lock_path = root / "slot-1.lock"
    with lock_path.open("a+b") as lock:
        acquired_while_held = True
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            acquired_while_held = False
        if acquired_while_held:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            raise RuntimeError("shared PVC flock did not exclude a second Pod")
        contended = {
            "schema_version": SCHEMA,
            "phase": "cross_pod_contention_observed",
            "job_uid": job_uid,
            "pod_uid": pod_uid,
            "holder_ready_sha256": ready["receipt_sha256"],
            "lock_acquired_while_holder_active": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        _write_once(root / "CONTENTION.json", contended)
        released = _wait_receipt(root / "HOLDER-RELEASED.json")
        if (
            released.get("schema_version") != SCHEMA
            or released.get("phase") != "holder_released"
            or released.get("contention_receipt_sha256") != contended["receipt_sha256"]
            or released.get("lock_released") is not True
        ):
            raise RuntimeError("endpoint lease release evidence drifted")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    terminal = {
        "schema_version": SCHEMA,
        "phase": "cross_pod_flock_passed",
        "holder_job_uid": ready["job_uid"],
        "holder_pod_uid": ready["pod_uid"],
        "prober_job_uid": job_uid,
        "prober_pod_uid": pod_uid,
        "holder_ready_sha256": ready["receipt_sha256"],
        "contention_receipt_sha256": contended["receipt_sha256"],
        "holder_released_sha256": released["receipt_sha256"],
        "contention_excluded_second_pod": True,
        "slot_reacquired_after_release": True,
        "scores_included": False,
        "prompts_or_traces_included": False,
    }
    _write_once(root / "TERMINAL.json", terminal)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("hold", "probe"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (hold if args.role == "hold" else probe)(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
