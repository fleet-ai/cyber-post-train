"""Run untouched rank-29 attempts with a pre-claim deadline guard."""

from __future__ import annotations

import os
import uuid
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
RELEASE_MAX_AGE_SECONDS = 600
PEER_ACCEPTED_ROOT = (
    "/mnt/sfs/jobs/chris-glm53-exact100-hosted-r003-a2a4-successor-v2/accepted"
)
PEER_ACCEPTED_RECEIPTS = [
    {
        "attempt": 2,
        "path": PEER_ACCEPTED_ROOT + "/chris-glm53-ac-bulk-b-r003-a2-g1-33d37078.json",
        "receipt_sha256": "sha256:b9d644bbbe844c9840a789880893629becc0953ea607fc92a98f886beeaa9d55",
        "file_sha256": "sha256:81e3eea0ece47e0ea30aefdeb0e27bd8e150096f3f7dff6c77de723c6b456baa",
        "cell_id": "sha256:57f0ec78f93760d980f31df4dac3fc6daf38f2849ee2949d943e3a41f4a61e2d",
        "execution_id": "sha256:6e93f24d70b8d76f936db9a910492a2479bca86c3a12cbc8c2a5592f16335cea",
        "run_id": "chris-glm53-ac-bulk-b-r003-a2-g1-33d37078",
    },
    {
        "attempt": 3,
        "path": PEER_ACCEPTED_ROOT + "/chris-glm53-ac-bulk-b-r003-a3-g1-33d37078.json",
        "receipt_sha256": "sha256:399aac0f15220d9f4f6ccfab857a02eeddabda496dad2f628cf072df8080f692",
        "file_sha256": "sha256:be0542174dd14a28cd1d5191c2d97a73d5bd457335b4f105506b032c9fac1a3c",
        "cell_id": "sha256:85d8da4373e2dbd30100aca88b08b884f3ef65febf71f77060ae67246df37c8b",
        "execution_id": "sha256:53eb697586931da3dd195636c3395ae58af60fcae1d383d5eb5f3aee960357d7",
        "run_id": "chris-glm53-ac-bulk-b-r003-a3-g1-33d37078",
    },
    {
        "attempt": 4,
        "path": PEER_ACCEPTED_ROOT + "/chris-glm53-ac-bulk-b-r003-a4-g1-33d37078.json",
        "receipt_sha256": "sha256:182d5ddd1f7f1632f296ab695127184d5d73d116425dc2cc0f73d226e0f02f27",
        "file_sha256": "sha256:c88c18a508c8cdf0a44d703520befb999dee4fbe0e757ff767c8bb3371d5fd86",
        "cell_id": "sha256:f5b26b9037dd0f02ad2bb22401c9d0be8b9ab4d98641fdb091cc5f1bf6cddb44",
        "execution_id": "sha256:0ca50d2ac1458b5335e17d76efb2a244b92bcdd84df0d21dac1164bb0f392ade",
        "run_id": "chris-glm53-ac-bulk-b-r003-a4-g1-33d37078",
    },
]


def validate_release_receipt(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
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
        "peer_active": False,
        "peer_succeeded": True,
        "peer_pod_phase": "Succeeded",
        "peer_pod_restarts": 0,
        "peer_accepted_receipts": PEER_ACCEPTED_RECEIPTS,
        "endpoint_lease_root": str(successor.LEASE_ROOT),
        "endpoint_lease_key": successor.LEASE_ENDPOINT_KEY,
        "endpoint_lease_slots_available": 2,
        "endpoint_lease_probe_released": True,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "fresh_generation_claim_collisions": 0,
        "retired_generation_claim_collisions": 0,
        "global_cell_claim_collisions": 0,
        "global_accepted_evidence_collisions": 0,
        "global_output_evidence_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "serving_load_block": successor.SERVING_LOAD_BLOCK,
        "job_active_deadline_seconds": ACTIVE_DEADLINE_SECONDS,
        "preclaim_guard_seconds": CLAIM_GUARD_SECONDS,
        "checked_immediately_before_create": True,
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    binding_fields = ("scored_source_sha256", "scored_package_template_sha256")
    dynamic_fields = {
        "global_accepted_files_examined",
        "global_claim_files_examined",
        "observed_at_utc",
        "observer_job_uid",
        "observer_pod_uid",
        "receipt_sha256",
        "endpoint_lease_slot_bindings",
        *binding_fields,
    }
    lease_bindings = receipt.get("endpoint_lease_slot_bindings")
    lease_bindings_valid = (
        isinstance(lease_bindings, list)
        and len(lease_bindings) == 2
        and [row.get("slot") for row in lease_bindings if isinstance(row, dict)]
        == [1, 2]
        and len(
            {
                (row.get("device"), row.get("inode"))
                for row in lease_bindings
                if isinstance(row, dict)
            }
        )
        == 2
        and all(
            set(row) == {"slot", "path", "device", "inode", "size"}
            and type(row["slot"]) is int
            and row["path"]
            == str(
                Path(successor.LEASE_ROOT)
                / successor.LEASE_ENDPOINT_KEY
                / f"slot-{row['slot']}.lock"
            )
            and type(row["device"]) is int
            and row["device"] > 0
            and type(row["inode"]) is int
            and row["inode"] > 0
            and type(row["size"]) is int
            and row["size"] == 0
            for row in lease_bindings
            if isinstance(row, dict)
        )
    )
    try:
        observed = datetime.fromisoformat(
            str(receipt.get("observed_at_utc", "")).replace("Z", "+00:00")
        )
        age = (datetime.now(UTC) - observed).total_seconds()
        observer_uids_valid = all(
            uuid.UUID(str(receipt.get(key))).int != 0
            for key in ("observer_job_uid", "observer_pod_uid")
        )
    except (TypeError, ValueError):
        age, observer_uids_valid = RELEASE_MAX_AGE_SECONDS + 1, False
    if (
        set(receipt) != set(expected) | dynamic_fields
        or any(receipt.get(key) != value for key, value in expected.items())
        or any(
            successor.SHA256_RE.fullmatch(str(receipt.get(key))) is None
            for key in binding_fields
        )
        or not lease_bindings_valid
        or type(receipt.get("global_claim_files_examined")) is not int
        or receipt.get("global_claim_files_examined", -1) < 0
        or type(receipt.get("global_accepted_files_examined")) is not int
        or receipt.get("global_accepted_files_examined", -1) < 0
        or not observer_uids_valid
        or age < -60
        or age > RELEASE_MAX_AGE_SECONDS
        or receipt.get("receipt_sha256")
        != self_hosted.digest_without(receipt, "receipt_sha256")
    ):
        raise RuntimeError("rank-29 hosted successor release drifted")


def validate_release(plan: dict[str, Any]) -> None:
    validate_release_receipt(plan, successor.load(RELEASE_PATH))


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
