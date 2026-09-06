"""Score-blind live release observer for rank 30 beside active rank 29."""

from __future__ import annotations

import fcntl
import os
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v2 as kube
from evals.fleet import hosted_glm_rank30_single_slot_v1 as successor
from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import hosted_glm_whole_task_successor_v1 as whole
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-exact100-hosted-r030-single-slot-release-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_live_peer() -> dict[str, Any]:
    status, job = kube._kube_get("jobs", successor.PEER_JOB)  # noqa: SLF001
    pod_status, pod = kube._kube_get("pods", successor.PEER_POD)  # noqa: SLF001
    statuses = pod.get("status", {}).get("containerStatuses", [])
    evaluator = next(
        (row for row in statuses if isinstance(row, dict) and row.get("name") == "evaluator"),
        None,
    )
    if (
        status != 200
        or job.get("metadata", {}).get("uid") != successor.PEER_JOB_UID
        or job.get("status", {}).get("active") != 1
        or bool(job.get("status", {}).get("failed"))
        or bool(job.get("status", {}).get("succeeded"))
        or pod_status != 200
        or pod.get("metadata", {}).get("uid") != successor.PEER_POD_UID
        or pod.get("status", {}).get("phase") != "Running"
        or not isinstance(evaluator, dict)
        or evaluator.get("ready") is not True
        or evaluator.get("restartCount") != 0
    ):
        raise RuntimeError("rank-29 live peer identity or health drifted")
    stream = Path(successor.PEER_STREAM)
    if stream.is_symlink() or not stream.is_file():
        raise RuntimeError("rank-29 live peer stream is absent or unsafe")
    stream_stat = stream.stat()
    stream_mtime = int(stream_stat.st_mtime)
    if (
        not stat.S_ISREG(stream_stat.st_mode)
        or stream_stat.st_size <= 0
        or datetime.now(UTC).timestamp() - stream_mtime
        > successor.PEER_STREAM_MAX_AGE_SECONDS
    ):
        raise RuntimeError("rank-29 live peer stream is stale")
    return {
        "job_name": successor.PEER_JOB,
        "job_uid": successor.PEER_JOB_UID,
        "job_active": 1,
        "pod_name": successor.PEER_POD,
        "pod_uid": successor.PEER_POD_UID,
        "pod_phase": "Running",
        "pod_ready": True,
        "pod_restarts": 0,
        "endpoint_slot_held": True,
        "run_id": successor.PEER_RUN_ID,
        "stream_path": str(successor.PEER_STREAM),
        "stream_bytes": stream_stat.st_size,
        "stream_mtime_epoch": stream_mtime,
    }


def _probe_slots() -> dict[str, Any]:
    root = Path(successor.LEASE_ROOT) / successor.LEASE_ENDPOINT_KEY
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("rank-30 endpoint lease root is unsafe")
    available: list[dict[str, Any]] = []
    held_slots: list[int] = []
    handles: list[Any] = []
    held = 0
    try:
        for slot in (1, 2):
            path = root / f"slot-{slot}.lock"
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("rank-30 endpoint lease slot is unsafe")
            flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            handle = os.fdopen(os.open(path, flags), "a+b")
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 0:
                handle.close()
                raise RuntimeError("rank-30 endpoint lease slot drifted")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held += 1
                held_slots.append(slot)
                handle.close()
                continue
            handles.append(handle)
            available.append(
                {
                    "slot": slot,
                    "path": str(path),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "size": metadata.st_size,
                }
            )
        if held_slots != [1] or [row["slot"] for row in available] != [2]:
            raise RuntimeError("rank-30 release requires slot 1 held and slot 2 free")
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    return {
        "lease_root": str(successor.LEASE_ROOT),
        "endpoint_key": successor.LEASE_ENDPOINT_KEY,
        "maximum_streams": 2,
        "held_slots": held,
        "held_slot_numbers": held_slots,
        "available_slots": len(available),
        "available_slot_bindings": available,
        "probe_released": True,
    }


def _collisions(plan: dict[str, Any], key: str) -> dict[str, Any]:
    identities = {
        value
        for item in plan["attempts"]
        for value in (item["cell_id"], item["execution_id"], item["run_id"])
    }
    claims = 0
    claim_collisions = 0
    for path in sorted(Path(successor.CLAIM_ROOT).glob("*.json")):
        claims += 1
        value = kube._safe_receipt(path)  # noqa: SLF001
        path_execution = "sha256:" + path.stem
        claim_collisions += int(
            path_execution in identities
            or bool(kube._identity_values(value).intersection(identities))  # noqa: SLF001
        )
    accepted = 0
    accepted_collisions = 0
    for path in sorted(Path(successor.JOBS_ROOT).glob("*/accepted/*.json")):
        accepted += 1
        value = kube._safe_receipt(path)  # noqa: SLF001
        accepted_collisions += int(
            path.stem in identities
            or bool(kube._identity_values(value).intersection(identities))  # noqa: SLF001
        )
    output_collisions = sum(
        int(path.exists() or path.is_symlink())
        for item in plan["attempts"]
        for path in Path(successor.JOBS_ROOT).glob(f"*/attempts/{item['run_id']}")
    )
    task = engine._task_for_item(plan, plan["attempts"][0])
    with engine.base._client(key) as client:  # noqa: SLF001
        sessions = self_hosted._task_sessions(client, task["task"]["key"])
    session_collisions = 0
    for item in plan["attempts"]:
        config = engine._attempt_config(plan, task, item)
        session_collisions += sum(
            int(kube._session_collides(row, config, item))  # noqa: SLF001
            for row in sessions
            if isinstance(row, dict)
        )
    job_status, _ = kube._kube_get(  # noqa: SLF001
        "jobs", successor.CONTROLLERS[successor.CONTROLLER]["job_name"]
    )
    cm_status, _ = kube._kube_get(  # noqa: SLF001
        "configmaps", successor.CONTROLLERS[successor.CONTROLLER]["configmap_name"]
    )
    if job_status not in {200, 404} or cm_status not in {200, 404}:
        raise RuntimeError("rank-30 target object absence could not be proven")
    return {
        "checked_immediately_before_create": True,
        "observer_job_uid": os.environ["JOB_UID"],
        "observer_pod_uid": os.environ["POD_UID"],
        "observed_cells": 4,
        "claim_files_examined": claims,
        "accepted_files_examined": accepted,
        "session_rows_examined": len(sessions),
        "canonical_claim_collisions": claim_collisions,
        "authoritative_session_collisions": session_collisions,
        "accepted_evidence_collisions": accepted_collisions,
        "output_root_collisions": output_collisions,
        "new_job_collisions": int(job_status != 404),
        "new_configmap_collisions": int(cm_status != 404),
        "api_mutations": 0,
    }


def build(root: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    source_sha = os.environ.get("GLM_HOSTED_R30_SOURCE_SHA256", "")
    if (
        not key
        or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid))
        or successor.SHA256_RE.fullmatch(source_sha) is None
    ):
        raise RuntimeError("rank-30 release observer bindings are invalid")
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    peer = _validate_live_peer()
    lease = _probe_slots()
    collision = _collisions(plan, key)
    if any(
        collision[field]
        for field in (
            "canonical_claim_collisions", "authoritative_session_collisions",
            "accepted_evidence_collisions", "output_root_collisions",
            "new_job_collisions", "new_configmap_collisions", "api_mutations",
        )
    ):
        raise RuntimeError("rank-30 single-slot collision gate is not clear")
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": _now(),
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 1,
        "controller": successor.release_projection(plan),
        "source_package_sha256": source_sha,
        "ledger_authority": whole.LEDGER_AUTHORITY,
        "selection_authority": whole.SELECTION_AUTHORITY,
        "live_rank29_peer": peer,
        "endpoint_lease_observer": lease,
        "fresh_collision_reconciliation": collision,
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    receipt = {
        **body,
        "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256"),
    }
    successor.validate_release(receipt, plan, source_sha)
    return receipt


def main() -> int:
    receipt = build(Path.cwd())
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
