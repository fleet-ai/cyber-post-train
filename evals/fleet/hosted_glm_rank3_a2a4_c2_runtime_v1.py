"""Run the released rank-3 attempts 2-4 hosted GLM successor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

RELEASE_SCHEMA = "fleet-hosted-glm-rank3-a2a4-c2-release-v1"
RELEASE_PATH = Path("/workspace/cyber-post-train/.runtime/release.json")


def validate_release(plan: dict[str, Any]) -> None:
    receipt = successor.load(RELEASE_PATH)
    required = {
        "schema_version": RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": successor.JOB_NAME,
        "successor_configmap": successor.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "planned_cells": 3,
        "canary_job_uid": successor.CANARY_JOB_UID,
        "canary_accepted_receipt_sha256": successor.CANARY_ACCEPTED_SHA,
        "canary_validation_receipt_sha256": successor.CANARY_VALIDATION_SHA,
        "s2_job_uid": "faf01255-0696-43e6-a47f-67802184986e",
        "s2_active": True,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    if any(receipt.get(k) != v for k, v in required.items()) or receipt.get(
        "receipt_sha256"
    ) != self_hosted.digest_without(receipt, "receipt_sha256"):
        raise RuntimeError("rank-3 hosted successor release drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    engine.bulk = successor
    return engine.run_controller(
        plan,
        out=Path(plan["sfs_root"]),
        proxy=proxy,
        runtime_gate_check=validate_release,
    )
