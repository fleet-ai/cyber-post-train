"""Run the one-cell hosted rank-3 GLM canary after an exact live release gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank3_a1_c2_canary_v1 as canary
from evals.fleet import self_hosted

RELEASE_SCHEMA = "fleet-hosted-glm-rank3-a1-c2-canary-release-v1"
RELEASE_PATH = Path("/bootstrap/release.json")


def validate_release(plan: dict[str, Any]) -> None:
    receipt = canary.load(RELEASE_PATH)
    required = {
        "schema_version": RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": canary.JOB_NAME,
        "successor_configmap": canary.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "selection_rank": 3,
        "attempt": 1,
        "whole_task_boundary_clear": True,
        "base_configmap_uid": canary.KNOWN_GOOD_BASE_CONFIGMAP["uid"],
        "active_scored_lease_slots_before_create": 1,
        "maximum_scored_streams": 2,
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("rank-3 hosted canary release drifted")
    if receipt.get("cell_id") != plan["attempts"][0]["cell_id"] or receipt.get(
        "execution_id"
    ) != plan["attempts"][0]["execution_id"]:
        raise RuntimeError("rank-3 hosted canary release cell drifted")
    if receipt.get("receipt_sha256") != self_hosted.digest_without(
        receipt, "receipt_sha256"
    ):
        raise RuntimeError("rank-3 hosted canary release digest drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = canary.load(source_runtime.INVENTORY_PATH)
    plan = canary.build_runtime_plan(canary.CONTROLLER, inventory, root)
    engine.bulk = canary
    return engine.run_controller(
        plan,
        out=Path(plan["sfs_root"]),
        proxy=proxy,
        runtime_gate_check=validate_release,
    )
