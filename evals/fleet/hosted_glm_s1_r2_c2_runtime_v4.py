"""Execute the engine-complete rank-2 hosted GLM v4 controller."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import hosted_glm_s1_r2_c2_release_v1 as lease_state
from evals.fleet import hosted_glm_s1_r2_c2_release_v5 as release
from evals.fleet import hosted_glm_s1_r2_c2_successor_v4 as successor
from evals.fleet import self_hosted


def validate_release(plan: dict[str, Any], receipt: dict[str, Any]) -> None:
    if any((
        receipt.get("schema_version") != release.SCHEMA,
        receipt.get("status") != "CLEAR",
        receipt.get("successor_job") != successor.JOB_NAME,
        receipt.get("successor_configmap") != successor.CONFIGMAP_NAME,
        receipt.get("plan_sha256") != plan["plan_sha256"],
        receipt.get("cell_ids") != [row["cell_id"] for row in plan["attempts"]],
        receipt.get("execution_ids") != [row["execution_id"] for row in plan["attempts"]],
        receipt.get("failed_controller_v3_effects") != {"claims": 0, "model_requests": 0, "task_instance_session_verifier_scoring_calls": 0},
        receipt.get("active_scored_lease_slots_before_create") not in (0, 1),
        receipt.get("maximum_scored_streams") != 2,
        receipt.get("fleet_session_collisions") != 0,
        receipt.get("global_claim_collisions") != 0,
        receipt.get("kubernetes_object_collisions") != 0,
        receipt.get("sfs_output_collisions") != 0,
        receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
    )):
        raise RuntimeError("rank-2 hosted v4 release drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(bulk_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    receipt = successor.load(release.OUTPUT_PATH)
    validate_release(plan, receipt)
    lease_state._validate_s2_active()  # noqa: SLF001
    if lease_state._active_lease_slots() > 1:  # noqa: SLF001
        raise RuntimeError("rank-2 hosted v4 has no free cap-two slot")
    engine.bulk = successor
    engine.validate_bulk_adapter(successor)
    if successor.build_runtime_plan(successor.CONTROLLER, inventory, root) != plan:
        raise RuntimeError("rank-2 hosted v4 engine rebuild drifted")
    if os.environ.get("HOSTED_BOOTSTRAP_ONLY") == "1":
        body = {
            "schema_version": "fleet-hosted-glm-rank2-controller-bootstrap-v2",
            "status": "PASSED_ENGINE_PRECLAIM",
            "controller_job": successor.JOB_NAME,
            "runtime_plan_sha256": plan["plan_sha256"],
            "release_receipt_sha256": receipt["receipt_sha256"],
            "bulk_adapter_members_valid": True,
            "engine_runtime_rebuild_equal": True,
            "maximum_scored_streams": 2,
            "claims": 0,
            "model_requests": 0,
            "task_instance_session_verifier_scoring_calls": 0,
            "prompts_traces_flags_or_scores_read": False,
        }
        body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
        output = Path(os.environ["HOSTED_BOOTSTRAP_RECEIPT"])
        output.parent.mkdir(parents=True, mode=0o700, exist_ok=False)
        output.write_bytes(self_hosted.canonical_json(body) + b"\n")
        return body
    return engine.run_controller(plan, out=successor.SFS_ROOT, proxy=proxy, runtime_gate_check=lambda _: validate_release(plan, receipt))
