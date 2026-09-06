"""Fresh identity for the engine-interface-complete rank-3/a1 GLM canary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_rank3_a1_c2_canary_v1 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank3-a1-c2-canary-plan-v2"
CONTROLLER = prior.CONTROLLER
JOB_NAME = "chris-glm53-exact100-hosted-r003-a1-canary-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = prior.CLAIM_ROOT
LEASE_ROOT = prior.LEASE_ROOT
LEASE_ENDPOINT_KEY = prior.LEASE_ENDPOINT_KEY
SERVING_LOAD_BLOCK = prior.SERVING_LOAD_BLOCK
EXPECTED_RANK = prior.EXPECTED_RANK
EXPECTED_ATTEMPT = prior.EXPECTED_ATTEMPT
EXPECTED_TASK_CELLS = prior.EXPECTED_TASK_CELLS
KNOWN_GOOD_BASE_CONFIGMAP = prior.KNOWN_GOOD_BASE_CONFIGMAP
FAILURE_AUTHORITY = prior.FAILURE_AUTHORITY
SHA256_RE = prior.SHA256_RE
COMMIT_RE = prior.COMMIT_RE
CANARY_GATE_SCHEMA = prior.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = prior.RECONCILIATION_GATE_SCHEMA
CONTROLLERS = {
    CONTROLLER: {
        **prior.CONTROLLERS[CONTROLLER],
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
    }
}
load = prior.load
validate_inventory_gate = prior.validate_inventory_gate


def _fresh(plan: dict[str, Any]) -> dict[str, Any]:
    plan.update(
        schema_version=SCHEMA,
        campaign_id=JOB_NAME,
        source_job_id=JOB_NAME,
        job_name=JOB_NAME,
        configmap_name=CONFIGMAP_NAME,
        sfs_root=str(SFS_ROOT),
    )
    plan["partition"]["failed_v1_job_uid"] = "0add81cd-68d7-4c85-a2ac-cf08fa60b1ec"
    plan["partition"]["failed_v1_pod_uid"] = "98af8398-9ebe-4c97-86f7-d87294a08201"
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


def build_runtime_plan(
    controller: str, inventory: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown rank-3 hosted GLM v2 canary controller")
    return _fresh(prior.build_runtime_plan(prior.CONTROLLER, inventory, root))


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = build_plan(root)
    if plan["launch_authorized"] is not False or plan["plan_sha256"] != self_hosted.digest_without(
        plan, "plan_sha256"
    ):
        raise ValueError("rank-3 hosted GLM v2 canary plan drifted")
    return {CONTROLLER: plan}
