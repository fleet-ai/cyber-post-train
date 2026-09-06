"""Run the released complete rank-2 hosted GLM successor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import hosted_glm_s1_r2_c2_release_v1 as release_module
from evals.fleet import hosted_glm_s1_r2_c2_successor_v1 as successor
from evals.fleet import self_hosted

RELEASE_PATH = release_module.OUTPUT_PATH


def validate_release(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
    if any(
        (
            receipt.get("schema_version") != release_module.SCHEMA,
            receipt.get("status") != "CLEAR",
            receipt.get("successor_job") != successor.JOB_NAME,
            receipt.get("successor_configmap") != successor.CONFIGMAP_NAME,
            receipt.get("plan_sha256") != plan["plan_sha256"],
            receipt.get("cell_ids") != [row["cell_id"] for row in plan["attempts"]],
            receipt.get("execution_ids") != [row["execution_id"] for row in plan["attempts"]],
            receipt.get("selection_rank") != 2,
            receipt.get("attempts") != [1, 2, 3, 4],
            receipt.get("whole_task_boundary") is not True,
            receipt.get("s2_job_uid") != release_module.S2_JOB_UID,
            receipt.get("s2_pod_uid") != release_module.S2_POD_UID,
            receipt.get("active_scored_lease_slots_before_create") != 1,
            receipt.get("maximum_scored_streams") != 2,
            receipt.get("fleet_session_collisions") != 0,
            receipt.get("global_claim_collisions") != 0,
            receipt.get("kubernetes_object_collisions") != 0,
            receipt.get("sfs_output_collisions") != 0,
            receipt.get("checked_immediately_before_create") is not True,
            receipt.get("scores_read") is not False,
            receipt.get("prompts_traces_flags_read") is not False,
            receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("rank-2 hosted successor release drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(bulk_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(inventory, root)
    receipt = successor.load(RELEASE_PATH)
    validate_release(plan, receipt)
    release_module._validate_s2_active()  # noqa: SLF001
    if release_module._active_lease_slots() != 1:  # noqa: SLF001
        raise RuntimeError("rank-2 successor no longer has exactly one free cap-two slot")
    engine.bulk = successor
    return engine.run_controller(
        plan,
        out=successor.SFS_ROOT,
        proxy=proxy,
        runtime_gate_check=lambda _: validate_release(plan, receipt),
    )

