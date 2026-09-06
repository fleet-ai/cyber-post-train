"""Fresh rank-3 tail identity after the v1 preclaim adapter failure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank3_a2a4_c2_successor_v1 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank3-a2a4-c2-successor-plan-v2"
CONTROLLER = prior.CONTROLLER
JOB_NAME = "chris-glm53-exact100-hosted-r003-a2a4-successor-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = prior.CLAIM_ROOT
LEASE_ROOT = prior.LEASE_ROOT
LEASE_ENDPOINT_KEY = prior.LEASE_ENDPOINT_KEY
SERVING_LOAD_BLOCK = prior.SERVING_LOAD_BLOCK
CANARY_JOB = prior.CANARY_JOB
CANARY_JOB_UID = prior.CANARY_JOB_UID
CANARY_ACCEPTED_SHA = prior.CANARY_ACCEPTED_SHA
CANARY_VALIDATION_SHA = prior.CANARY_VALIDATION_SHA
SHA256_RE = prior.SHA256_RE
COMMIT_RE = prior.COMMIT_RE
CANARY_GATE_SCHEMA = prior.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = prior.RECONCILIATION_GATE_SCHEMA
load = prior.load
validate_inventory_gate = prior.validate_inventory_gate
CONTROLLERS = {
    CONTROLLER: {
        **prior.CONTROLLERS[CONTROLLER],
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
    }
}


def _fresh(plan: dict[str, Any]) -> dict[str, Any]:
    plan.update(
        schema_version=SCHEMA,
        campaign_id=JOB_NAME,
        source_job_id=JOB_NAME,
        job_name=JOB_NAME,
        configmap_name=CONFIGMAP_NAME,
        sfs_root=str(SFS_ROOT),
    )
    plan["partition"]["failed_v1_job_uid"] = "b77981f3-cdc1-4e16-9825-40c664c262a5"
    plan["partition"]["failed_v1_pod_uid"] = "f843b87a-132f-4449-8e67-0d415c8ad6ce"
    plan["partition"]["failed_v1_effects"] = {
        "claims": 0,
        "model_requests": 0,
        "task_instance_session_verifier_scoring_calls": 0,
    }
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def build_plan(root: Path) -> dict[str, Any]:
    return _fresh(prior.build_plan(root))


def build_runtime_plan(controller: str, inventory: dict[str, Any], root: Path) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown rank-3 hosted GLM v2 successor")
    return _fresh(prior.build_runtime_plan(prior.CONTROLLER, inventory, root))


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = build_plan(root)
    if any(
        (
            plan["launch_authorized"] is not False,
            [row["attempt"] for row in plan["attempts"]] != [2, 3, 4],
            plan["execution"]["endpoint_lease"]["maximum_streams"] != 2,
            plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("rank-3 hosted GLM v2 successor drifted")
    return {CONTROLLER: plan}
