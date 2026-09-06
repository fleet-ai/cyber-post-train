"""Execute the fresh rank-2 hosted GLM successor under the max-two lease."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import hosted_glm_s1_r2_c2_release_v1 as prior_release
from evals.fleet import hosted_glm_s1_r2_c2_release_v3 as release
from evals.fleet import hosted_glm_s1_r2_c2_successor_v2 as successor
from evals.fleet import self_hosted


def validate_release(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
    if any(
        (
            receipt.get("schema_version") != release.SCHEMA,
            receipt.get("status") != "CLEAR",
            receipt.get("successor_job") != successor.JOB_NAME,
            receipt.get("successor_configmap") != successor.CONFIGMAP_NAME,
            receipt.get("plan_sha256") != plan["plan_sha256"],
            receipt.get("cell_ids") != [row["cell_id"] for row in plan["attempts"]],
            receipt.get("execution_ids") != [row["execution_id"] for row in plan["attempts"]],
            receipt.get("failed_v1_job_uid") != release.FAILED_V1_JOB_UID,
            receipt.get("failed_v1_pod_uid") != release.FAILED_V1_POD_UID,
            receipt.get("failed_v1_claims") != 0,
            receipt.get("failed_v1_model_requests") != 0,
            receipt.get("failed_v1_task_instance_session_verifier_scoring_calls") != 0,
            receipt.get("active_scored_lease_slots_before_create") not in (0, 1),
            receipt.get("maximum_scored_streams") != 2,
            receipt.get("fleet_session_collisions") != 0,
            receipt.get("global_claim_collisions") != 0,
            receipt.get("kubernetes_object_collisions") != 0,
            receipt.get("sfs_output_collisions") != 0,
            receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("rank-2 hosted v2 release drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(bulk_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(inventory, root)
    receipt = successor.load(release.OUTPUT_PATH)
    validate_release(plan, receipt)
    prior_release._validate_s2_active()  # noqa: SLF001
    if prior_release._active_lease_slots() > 1:  # noqa: SLF001
        raise RuntimeError("rank-2 hosted v2 has no free cap-two slot")
    engine.bulk = successor
    return engine.run_controller(
        plan,
        out=successor.SFS_ROOT,
        proxy=proxy,
        runtime_gate_check=lambda _: validate_release(plan, receipt),
    )
