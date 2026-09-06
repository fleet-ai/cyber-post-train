"""Run untouched rank-29 attempts with a pre-claim deadline guard."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

RELEASE_SCHEMA = "fleet-hosted-glm-rank29-a3a4-c2-release-v1"
RELEASE_PATH = Path("/workspace/cyber-post-train/.runtime/release.json")
ACTIVE_DEADLINE_SECONDS = 86_400
CLAIM_GUARD_SECONDS = 32_400
NAMESPACE = "fleet-train-jobs"
PEER_JOB_UID = "a921e359-2ed4-4e90-83f2-05aa5812f454"


def validate_release(plan: dict[str, Any]) -> None:
    receipt = successor.load(RELEASE_PATH)
    expected = {
        "schema_version": RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": successor.JOB_NAME,
        "successor_configmap": successor.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "planned_cells": 2,
        "source_failed_job_uid": successor.SOURCE_FAILED_JOB_UID,
        "source_failed_pod_uid": successor.SOURCE_FAILED_POD_UID,
        "blocked_a2_claim_sha256": successor.BLOCKED_A2_CLAIM_SHA,
        "blocked_a2_session_collisions": 0,
        "peer_job_uid": PEER_JOB_UID,
        "peer_pod_uid": "0da05643-b6a5-4cf0-8498-d715e2cb3422",
        "peer_active": True,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "serving_load_block": successor.SERVING_LOAD_BLOCK,
        "job_active_deadline_seconds": ACTIVE_DEADLINE_SECONDS,
        "preclaim_guard_seconds": CLAIM_GUARD_SECONDS,
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()) or receipt.get(
        "receipt_sha256"
    ) != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise RuntimeError("rank-29 hosted successor release drifted")


def _job_timing() -> tuple[float, int]:
    job_name = os.environ.get("JOB_NAME", "")
    job_uid = os.environ.get("JOB_UID", "")
    if job_name != successor.JOB_NAME or not job_uid:
        raise RuntimeError("rank-29 deadline guard identity absent")
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    url = (
        "https://kubernetes.default.svc/apis/batch/v1/namespaces/"
        f"{NAMESPACE}/jobs/{job_name}"
    )
    with httpx.Client(
        headers={"Authorization": f"Bearer {token}"}, verify=ca, timeout=30
    ) as client:
        response = client.get(url)
    value = response.json() if response.content else {}
    start = value.get("status", {}).get("startTime")
    deadline = value.get("spec", {}).get("activeDeadlineSeconds")
    if (
        response.status_code != 200
        or value.get("metadata", {}).get("uid") != job_uid
        or not isinstance(start, str)
        or deadline != ACTIVE_DEADLINE_SECONDS
    ):
        raise RuntimeError("rank-29 deadline guard Job binding drifted")
    try:
        start_epoch = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise RuntimeError("rank-29 deadline guard start time invalid") from exc
    return start_epoch, deadline


def _deadline_observer(out: Path):
    start_epoch, deadline = _job_timing()

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        if stage != "06-route-valid" or item is None:
            return
        now = datetime.now(UTC).timestamp()
        remaining = start_epoch + deadline - now
        if remaining >= CLAIM_GUARD_SECONDS:
            return
        receipt = {
            "schema_version": "fleet-hosted-glm-preclaim-deadline-stop-v1",
            "status": "STOPPED_BEFORE_NEW_CELL_CLAIM",
            "job_uid": os.environ.get("JOB_UID"),
            "pod_uid": os.environ.get("POD_UID"),
            "plan_sha256": successor.load(out / "PLAN.json")["plan_sha256"],
            "next_cell_id": item["cell_id"],
            "next_execution_id": item["execution_id"],
            "active_deadline_seconds": deadline,
            "claim_guard_seconds": CLAIM_GUARD_SECONDS,
            "remaining_seconds": int(remaining),
            "claim_written": False,
            "model_request_started": False,
            "scores_included": False,
            "prompts_or_traces_included": False,
        }
        receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
        engine._write_once(out / "STOP_BEFORE_DEADLINE.json", receipt)  # noqa: SLF001
        raise RuntimeError("rank-29 successor stopped before unsafe new claim")

    return observe


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    engine.bulk = successor
    out = Path(plan["sfs_root"])
    return engine.run_controller(
        plan,
        out=out,
        proxy=proxy,
        runtime_gate_check=validate_release,
        stage_observer=_deadline_observer(out),
    )
