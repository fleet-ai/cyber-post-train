"""Score-blind create-once release for the rank-2 hosted GLM successor."""

from __future__ import annotations

import fcntl
import os
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import hosted_glm_s1_r2_c2_successor_v1 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-release-v1"
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-release-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
S2_JOB = "chris-glm53-exact100-hosted-s2-c2-bulk-v1"
S2_JOB_UID = "faf01255-0696-43e6-a47f-67802184986e"
S2_POD_UID = "72ce8163-fb2d-40ac-92da-da0dd578658d"
BLOCKED_ABORT = Path("/mnt/sfs/jobs/chris-glm53-exact100-hosted-s1-v1/ABORT.json")
BLOCKED_ABORT_SHA = "sha256:59283ed31cddf54045e7083689b088ec44a72aadd4b9cd4a037609ab1cd8d0e2"


def _kube_get(path: str) -> tuple[int, dict[str, Any]]:
    return ledger._kube_get(path)  # noqa: SLF001


def _validate_s2_active() -> None:
    status, job = _kube_get(f"/apis/batch/v1/namespaces/{ledger.NAMESPACE}/jobs/{S2_JOB}")
    if status != 200 or job.get("metadata", {}).get("uid") != S2_JOB_UID or job.get("status", {}).get("active") != 1:
        raise RuntimeError("exact s2 hosted controller is not active")
    status, pods = _kube_get(f"/api/v1/namespaces/{ledger.NAMESPACE}/pods?labelSelector=job-name%3D{S2_JOB}")
    items = pods.get("items", []) if status == 200 else []
    if len(items) != 1 or items[0].get("metadata", {}).get("uid") != S2_POD_UID or items[0].get("status", {}).get("phase") != "Running":
        raise RuntimeError("exact s2 hosted Pod is not active")


def _active_lease_slots() -> int:
    root = Path(successor.LEASE_ROOT) / successor.LEASE_ENDPOINT_KEY
    held = 0
    for path in sorted(root.glob("slot-*.lock")):
        if path.is_symlink() or not path.is_file():
            continue
        handle = path.open("a+b")
        try:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                continue
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held += 1
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
    return held


def build(root: Path) -> dict[str, Any]:
    key, job_uid, pod_uid = os.environ.get("FLEET_API_KEY", ""), os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("rank-2 release requires key and observer UIDs")
    _validate_s2_active()
    if _active_lease_slots() != 1:
        raise RuntimeError("hosted endpoint must have exactly one active scored lease")
    abort = successor.load(BLOCKED_ABORT)
    if abort.get("receipt_sha256") != BLOCKED_ABORT_SHA or abort.get("receipt_sha256") != self_hosted.digest_without(abort, "receipt_sha256") or abort.get("automatic_tail_continuation") is not False:
        raise RuntimeError("rank1/a4 blocked authority drifted")
    inventory = successor.load(bulk_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(inventory, root)
    if successor.SFS_ROOT.exists() or successor.SFS_ROOT.is_symlink():
        raise RuntimeError("rank-2 successor output collision")
    for kind, name in (("jobs", successor.JOB_NAME), ("configmaps", successor.CONFIGMAP_NAME)):
        prefix = "/apis/batch/v1" if kind == "jobs" else "/api/v1"
        status, _ = _kube_get(f"{prefix}/namespaces/{ledger.NAMESPACE}/{kind}/{name}")
        if status != 404:
            raise RuntimeError("rank-2 successor Kubernetes collision")
    engine._fresh_route_check(plan, key)  # noqa: SLF001
    task = plan["tasks"][0]
    claim_collisions = 0
    session_collisions = 0
    first_config = engine._attempt_config(plan, task, plan["attempts"][0])  # noqa: SLF001
    with engine._client(key) as client:  # noqa: SLF001
        sessions = self_hosted._task_sessions(client, first_config["task"]["key"])
    for item in plan["attempts"]:
        claim = Path(successor.CLAIM_ROOT) / engine.claim_filename(item["execution_id"])
        claim_collisions += int(claim.exists() or claim.is_symlink())
        config = engine._attempt_config(plan, task, item)  # noqa: SLF001
        session_collisions += sum(ledger._session_collides(row, config, item) for row in sessions)  # noqa: SLF001
    if claim_collisions or session_collisions:
        raise RuntimeError("rank-2 successor duplicate ledger is not clear")
    body = {
        "schema_version": SCHEMA,
        "status": "CLEAR",
        "successor_job": successor.JOB_NAME,
        "successor_configmap": successor.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "cell_ids": [row["cell_id"] for row in plan["attempts"]],
        "execution_ids": [row["execution_id"] for row in plan["attempts"]],
        "selection_rank": 2,
        "attempts": [1, 2, 3, 4],
        "whole_task_boundary": True,
        "blocked_rank1_a4_abort_receipt_sha256": BLOCKED_ABORT_SHA,
        "s2_job_uid": S2_JOB_UID,
        "s2_pod_uid": S2_POD_UID,
        "active_scored_lease_slots_before_create": 1,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": session_collisions,
        "global_claim_collisions": claim_collisions,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "checked_immediately_before_create": True,
        "mutation_calls": 0,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "observed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body
