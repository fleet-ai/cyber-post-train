"""Score-blind create-once release for hosted GLM rank-3 attempts 2-4."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank3_a2a4_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-exact100-hosted-r003-a2a4-release-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
S2_JOB = "chris-glm53-exact100-hosted-s2-c2-bulk-v1"
S2_JOB_UID = "faf01255-0696-43e6-a47f-67802184986e"
S2_POD_UID = "72ce8163-fb2d-40ac-92da-da0dd578658d"
CANARY_ACCEPTED_PATH = Path("/mnt/sfs/jobs/chris-glm53-exact100-hosted-r003-a1-canary-v2/accepted/chris-glm53-ac-bulk-b-r003-a1-g1-33d37078.json")
CANARY_VALIDATION_PATH = Path("/workspace/cyber-post-train/.runtime/canary-validation.json")


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _kube_job(name: str) -> tuple[int, dict[str, Any]]:
    return ledger._kube_get(f"/apis/batch/v1/namespaces/{ledger.NAMESPACE}/jobs/{name}")  # noqa: SLF001


def _kube_absent(name: str, kind: str) -> None:
    prefix = "/apis/batch/v1" if kind == "jobs" else "/api/v1"
    status, _ = ledger._kube_get(f"{prefix}/namespaces/{ledger.NAMESPACE}/{kind}/{name}")  # noqa: SLF001
    if status != 404:
        raise RuntimeError("rank-3 successor Kubernetes collision")


def _validate_predecessors(target: Any = successor) -> None:
    status, canary = _kube_job(target.CANARY_JOB)
    if any(
        (
            status != 200,
            canary.get("metadata", {}).get("uid") != target.CANARY_JOB_UID,
            canary.get("status", {}).get("succeeded") != 1,
            bool(canary.get("status", {}).get("active")),
            bool(canary.get("status", {}).get("failed")),
        )
    ):
        raise RuntimeError("rank-3/a1 canary is not exclusively succeeded")
    accepted = successor.load(CANARY_ACCEPTED_PATH)
    validation = successor.load(CANARY_VALIDATION_PATH)
    if any(
        (
            accepted.get("receipt_sha256") != target.CANARY_ACCEPTED_SHA,
            accepted.get("receipt_sha256") != self_hosted.digest_without(accepted, "receipt_sha256"),
            accepted.get("selection_rank") != 3,
            accepted.get("attempt") != 1,
            accepted.get("accepted") is not True,
            accepted.get("credited") is not True,
            validation.get("receipt_sha256") != target.CANARY_VALIDATION_SHA,
            validation.get("receipt_sha256") != self_hosted.digest_without(validation, "receipt_sha256"),
            validation.get("status") != "ACCEPTED_VALIDATED",
            validation.get("accepted", {}).get("receipt_sha256") != target.CANARY_ACCEPTED_SHA,
        )
    ):
        raise RuntimeError("rank-3/a1 acceptance authority drifted")
    status, s2 = _kube_job(S2_JOB)
    if any(
        (
            status != 200,
            s2.get("metadata", {}).get("uid") != S2_JOB_UID,
            s2.get("status", {}).get("active") != 1,
            bool(s2.get("status", {}).get("failed")),
            bool(s2.get("status", {}).get("succeeded")),
        )
    ):
        raise RuntimeError("exact hosted GLM s2 stream is not active")


def build(
    root: Path,
    *,
    target: Any = successor,
    runtime_module: Any = runtime,
    failed_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("rank-3 successor release requires key and nonzero UIDs")
    _validate_predecessors(target)
    inventory = target.load(source_runtime.INVENTORY_PATH)
    plan = target.build_runtime_plan(target.CONTROLLER, inventory, root)
    output_collisions = int(target.SFS_ROOT.exists() or target.SFS_ROOT.is_symlink())
    _kube_absent(target.JOB_NAME, "jobs")
    _kube_absent(target.CONFIGMAP_NAME, "configmaps")
    engine._fresh_route_check(plan, key)  # noqa: SLF001
    claim_collisions = 0
    session_collisions = 0
    sessions_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in plan["attempts"]:
        claim = Path(target.CLAIM_ROOT) / engine.claim_filename(item["execution_id"])
        claim_collisions += int(claim.exists() or claim.is_symlink())
        task = engine._task_for_item(plan, item)  # noqa: SLF001
        config = engine._attempt_config(plan, task, item)  # noqa: SLF001
        task_key = config["task"]["key"]
        if task_key not in sessions_by_task:
            with engine._client(key) as client:  # noqa: SLF001
                sessions_by_task[task_key] = self_hosted._task_sessions(client, task_key)
        session_collisions += sum(
            ledger._session_collides(row, config, item)  # noqa: SLF001
            for row in sessions_by_task[task_key]
        )
    if output_collisions or claim_collisions or session_collisions:
        raise RuntimeError("rank-3 successor duplicate ledger is not clear")
    body = {
        "schema_version": runtime_module.RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": target.JOB_NAME,
        "successor_configmap": target.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "planned_cells": len(plan["attempts"]),
        "canary_job_uid": target.CANARY_JOB_UID,
        "canary_accepted_receipt_sha256": target.CANARY_ACCEPTED_SHA,
        "canary_validation_receipt_sha256": target.CANARY_VALIDATION_SHA,
        "s2_job_uid": S2_JOB_UID,
        "s2_pod_uid": S2_POD_UID,
        "s2_active": True,
        "fleet_session_collisions": session_collisions,
        "global_claim_collisions": claim_collisions,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": output_collisions,
        "serving_load_block": target.SERVING_LOAD_BLOCK,
        "maximum_scored_streams": 2,
        "checked_immediately_before_create": True,
        "mutation_calls": 0,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "observed_at_utc": _now(),
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    if failed_identity is not None:
        body["failed_v1"] = failed_identity
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def main() -> int:
    receipt = build(Path.cwd())
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
