"""Score-blind create-once gate for untouched rank-29 attempts 3 and 4."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-exact100-hosted-r029-a3a4-release-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
SOURCE_JOB = "chris-glm53-exact100-hosted-s2-c2-bulk-v1"
PEER_JOB = "chris-glm53-exact100-hosted-r003-a2a4-successor-v2"
PEER_JOB_UID = runtime.PEER_JOB_UID
PEER_POD_UID = "0da05643-b6a5-4cf0-8498-d715e2cb3422"
RANK29_A1_ACCEPTED = Path(
    "/mnt/sfs/jobs/chris-glm53-exact100-hosted-s2-c2-bulk-v1/accepted/"
    "chris-glm53-ac-bulk-b-r029-a1-g1-b51782f9.json"
)
RANK29_A1_ACCEPTED_SHA = (
    "sha256:8240130bdfb7367a4977bab918c4f793341f39e00c237eb16b8d3eca3dba95b6"
)
RANK29_A2_CLAIM = Path(
    "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/"
    "103a5fff50b4de20ed0d36c135cb36b9d99dd237f5df7ecb14007cf4cdc45f4e.json"
)
RANK29_A2_EXECUTION = (
    "sha256:103a5fff50b4de20ed0d36c135cb36b9d99dd237f5df7ecb14007cf4cdc45f4e"
)
RANK29_A2_CELL = (
    "sha256:5a82e16e0892f64f9bc2f751b7fc54bea96ed856c4782947f24ab6e2da3858eb"
)
RANK29_A2_RUN = "chris-glm53-ac-bulk-b-r029-a2-g1-b51782f9"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _kube_get(kind: str, name: str) -> tuple[int, dict[str, Any]]:
    if kind == "jobs":
        path = f"/apis/batch/v1/namespaces/{ledger.NAMESPACE}/jobs/{name}"
    else:
        path = f"/api/v1/namespaces/{ledger.NAMESPACE}/{kind}/{name}"
    return ledger._kube_get(path)  # noqa: SLF001


def _validate_predecessors() -> None:
    status, source_job = _kube_get("jobs", SOURCE_JOB)
    conditions = source_job.get("status", {}).get("conditions", [])
    deadline_failed = any(
        row.get("type") == "Failed"
        and row.get("status") == "True"
        and row.get("reason") == "DeadlineExceeded"
        for row in conditions
        if isinstance(row, dict)
    )
    if (
        status != 200
        or source_job.get("metadata", {}).get("uid") != successor.SOURCE_FAILED_JOB_UID
        or not deadline_failed
        or bool(source_job.get("status", {}).get("active"))
    ):
        raise RuntimeError("rank-29 source deadline failure identity drifted")
    source_pod_status, _ = _kube_get(
        "pods", "chris-glm53-exact100-hosted-s2-c2-bulk-v1-622nw"
    )
    if source_pod_status != 404:
        raise RuntimeError("rank-29 source Pod is not absent")

    status, peer = _kube_get("jobs", PEER_JOB)
    if (
        status != 200
        or peer.get("metadata", {}).get("uid") != PEER_JOB_UID
        or peer.get("status", {}).get("active") != 1
        or bool(peer.get("status", {}).get("failed"))
        or bool(peer.get("status", {}).get("succeeded"))
    ):
        raise RuntimeError("rank-29 successor peer stream is not exact and active")
    pod_status, peer_pod = _kube_get(
        "pods", "chris-glm53-exact100-hosted-r003-a2a4-successor-v2-9g56k"
    )
    statuses = peer_pod.get("status", {}).get("containerStatuses", [])
    if (
        pod_status != 200
        or peer_pod.get("metadata", {}).get("uid") != PEER_POD_UID
        or peer_pod.get("status", {}).get("phase") != "Running"
        or not statuses
        or any(row.get("ready") is not True or row.get("restartCount") != 0 for row in statuses)
    ):
        raise RuntimeError("rank-29 successor peer Pod is not healthy")


def _validate_rank29_history() -> None:
    accepted = successor.load(RANK29_A1_ACCEPTED)
    claim = successor.load(RANK29_A2_CLAIM)
    if (
        accepted.get("receipt_sha256") != RANK29_A1_ACCEPTED_SHA
        or accepted.get("receipt_sha256")
        != self_hosted.digest_without(accepted, "receipt_sha256")
        or accepted.get("selection_rank") != 29
        or accepted.get("attempt") != 1
        or accepted.get("accepted") is not True
        or accepted.get("credited") is not True
        or claim.get("receipt_sha256") != successor.BLOCKED_A2_CLAIM_SHA
        or claim.get("receipt_sha256")
        != self_hosted.digest_without(claim, "receipt_sha256")
        or claim.get("cell_id") != RANK29_A2_CELL
        or claim.get("execution_id") != RANK29_A2_EXECUTION
        or claim.get("run_id") != RANK29_A2_RUN
        or claim.get("selection_rank") != 29
        or claim.get("attempt") != 2
        or claim.get("job_uid") != successor.SOURCE_FAILED_JOB_UID
        or claim.get("pod_uid") != successor.SOURCE_FAILED_POD_UID
        or claim.get("automatic_retry") is not False
    ):
        raise RuntimeError("rank-29 accepted/blocked history drifted")


def _session_collides(
    row: dict[str, Any], config: dict[str, Any], item: dict[str, Any]
) -> bool:
    metadata = row.get("metadata") or {}
    return any(
        metadata.get(field) == expected
        for field, expected in {
            "run_id": config["run_id"],
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
        }.items()
    )


def build(root: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("rank-29 release requires key and nonzero UIDs")
    _validate_predecessors()
    _validate_rank29_history()
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    output_collisions = int(successor.SFS_ROOT.exists() or successor.SFS_ROOT.is_symlink())
    kubernetes_collisions = sum(
        _kube_get(kind, name)[0] == 200
        for kind, name in (
            ("jobs", successor.JOB_NAME),
            ("configmaps", successor.CONFIGMAP_NAME),
        )
    )
    engine._fresh_route_check(plan, key)  # noqa: SLF001
    task = engine._task_for_item(plan, plan["attempts"][0])  # noqa: SLF001
    with engine._client(key) as client:  # noqa: SLF001
        sessions = self_hosted._task_sessions(client, task["task"]["key"])

    blocked_config = {"run_id": RANK29_A2_RUN}
    blocked_item = {"cell_id": RANK29_A2_CELL, "execution_id": RANK29_A2_EXECUTION}
    blocked_session_collisions = sum(
        _session_collides(row, blocked_config, blocked_item) for row in sessions
    )
    claim_collisions = 0
    session_collisions = 0
    for item in plan["attempts"]:
        claim = Path(successor.CLAIM_ROOT) / engine.claim_filename(item["execution_id"])
        claim_collisions += int(claim.exists() or claim.is_symlink())
        config = engine._attempt_config(plan, task, item)  # noqa: SLF001
        session_collisions += sum(_session_collides(row, config, item) for row in sessions)
    if any(
        (
            blocked_session_collisions,
            output_collisions,
            kubernetes_collisions,
            claim_collisions,
            session_collisions,
        )
    ):
        raise RuntimeError("rank-29 successor duplicate ledger is not clear")
    body = {
        "schema_version": runtime.RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": successor.JOB_NAME,
        "successor_configmap": successor.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "planned_cells": len(plan["attempts"]),
        "source_failed_job_uid": successor.SOURCE_FAILED_JOB_UID,
        "source_failed_pod_uid": successor.SOURCE_FAILED_POD_UID,
        "blocked_a2_claim_sha256": successor.BLOCKED_A2_CLAIM_SHA,
        "blocked_a2_session_collisions": blocked_session_collisions,
        "peer_job_uid": PEER_JOB_UID,
        "peer_pod_uid": PEER_POD_UID,
        "peer_active": True,
        "fleet_session_collisions": session_collisions,
        "global_claim_collisions": claim_collisions,
        "kubernetes_object_collisions": kubernetes_collisions,
        "sfs_output_collisions": output_collisions,
        "serving_load_block": successor.SERVING_LOAD_BLOCK,
        "maximum_scored_streams": 2,
        "job_active_deadline_seconds": runtime.ACTIVE_DEADLINE_SECONDS,
        "preclaim_guard_seconds": runtime.CLAIM_GUARD_SECONDS,
        "checked_immediately_before_create": True,
        "mutation_calls": 0,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "observed_at_utc": _now(),
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def main() -> int:
    receipt = build(Path.cwd())
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
