"""Retry-safe release for the fresh rank-2 hosted GLM successor."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import hosted_glm_s1_r2_c2_release_v1 as prior
from evals.fleet import hosted_glm_s1_r2_c2_successor_v2 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-release-v2"
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-release-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
FAILED_V1_JOB = "chris-glm53-exact100-hosted-s1-r002-c2-successor-v1"
FAILED_V1_JOB_UID = "967d4229-f953-4809-932a-75bfca242356"
FAILED_V1_POD_UID = "ff477d84-8148-4dc3-ad84-8dd54954a1e7"


def _validate_failed_v1() -> None:
    status, job = prior._kube_get(  # noqa: SLF001
        f"/apis/batch/v1/namespaces/{ledger.NAMESPACE}/jobs/{FAILED_V1_JOB}"
    )
    conditions = job.get("status", {}).get("conditions", []) if status == 200 else []
    failed = any(row.get("type") == "Failed" and row.get("status") == "True" for row in conditions)
    if job.get("metadata", {}).get("uid") != FAILED_V1_JOB_UID or not failed:
        raise RuntimeError("rank-2 v1 failure tombstone drifted")
    status, pods = prior._kube_get(  # noqa: SLF001
        f"/api/v1/namespaces/{ledger.NAMESPACE}/pods?labelSelector=job-name%3D{FAILED_V1_JOB}"
    )
    items = pods.get("items", []) if status == 200 else []
    if len(items) != 1 or items[0].get("metadata", {}).get("uid") != FAILED_V1_POD_UID:
        raise RuntimeError("rank-2 v1 Pod tombstone drifted")
    terminated = items[0].get("status", {}).get("containerStatuses", [{}])[0].get("state", {}).get("terminated", {})
    if terminated.get("exitCode") != 1 or terminated.get("reason") != "Error":
        raise RuntimeError("rank-2 v1 terminal status drifted")
    if Path("/mnt/sfs/jobs") .joinpath(FAILED_V1_JOB).exists():
        raise RuntimeError("rank-2 v1 unexpectedly created output")


def build(root: Path, *, target: Any = successor) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("rank-2 v2 release requires key and observer UIDs")
    _validate_failed_v1()
    prior._validate_s2_active()  # noqa: SLF001
    occupied = prior._active_lease_slots()  # noqa: SLF001
    if occupied not in (0, 1):
        raise RuntimeError("rank-2 v2 has no free cap-two endpoint slot")
    inventory = target.load(bulk_runtime.INVENTORY_PATH)
    if getattr(target, "ENGINE_ADAPTER_V1", False):
        plan = target.build_runtime_plan(target.CONTROLLER, inventory, root)
    else:
        plan = target.build_runtime_plan(inventory, root)
    if target.SFS_ROOT.exists() or target.SFS_ROOT.is_symlink():
        raise RuntimeError("rank-2 v2 output collision")
    for kind, name in (("jobs", target.JOB_NAME), ("configmaps", target.CONFIGMAP_NAME)):
        prefix = "/apis/batch/v1" if kind == "jobs" else "/api/v1"
        status, _ = prior._kube_get(f"{prefix}/namespaces/{ledger.NAMESPACE}/{kind}/{name}")  # noqa: SLF001
        if status != 404:
            raise RuntimeError("rank-2 v2 Kubernetes collision")
    engine._fresh_route_check(plan, key)  # noqa: SLF001
    task = plan["tasks"][0]
    first = engine._attempt_config(plan, task, plan["attempts"][0])  # noqa: SLF001
    with engine._client(key) as client:  # noqa: SLF001
        sessions = self_hosted._task_sessions(client, first["task"]["key"])
    claim_collisions = session_collisions = 0
    for item in plan["attempts"]:
        claim = Path(target.CLAIM_ROOT) / engine.claim_filename(item["execution_id"])
        claim_collisions += int(claim.exists() or claim.is_symlink())
        config = engine._attempt_config(plan, task, item)  # noqa: SLF001
        session_collisions += sum(ledger._session_collides(row, config, item) for row in sessions)  # noqa: SLF001
    if claim_collisions or session_collisions:
        raise RuntimeError("rank-2 v2 duplicate ledger is not clear")
    body = {
        "schema_version": SCHEMA,
        "status": "CLEAR",
        "successor_job": target.JOB_NAME,
        "successor_configmap": target.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "cell_ids": [row["cell_id"] for row in plan["attempts"]],
        "execution_ids": [row["execution_id"] for row in plan["attempts"]],
        "selection_rank": 2,
        "attempts": [1, 2, 3, 4],
        "whole_task_boundary": True,
        "failed_v1_job_uid": FAILED_V1_JOB_UID,
        "failed_v1_pod_uid": FAILED_V1_POD_UID,
        "failed_v1_claims": 0,
        "failed_v1_model_requests": 0,
        "failed_v1_task_instance_session_verifier_scoring_calls": 0,
        "s2_job_uid": prior.S2_JOB_UID,
        "s2_pod_uid": prior.S2_POD_UID,
        "active_scored_lease_slots_before_create": occupied,
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
