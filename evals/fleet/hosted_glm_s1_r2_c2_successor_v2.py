"""Fresh rank-2 hosted GLM successor after the v1 lease-boundary failure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_successor_v1 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-plan-v2"
CONTROLLER = prior.CONTROLLER
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-successor-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = prior.CLAIM_ROOT
LEASE_ROOT = prior.LEASE_ROOT
LEASE_ENDPOINT_KEY = prior.LEASE_ENDPOINT_KEY
SERVING_LOAD_BLOCK = prior.SERVING_LOAD_BLOCK
EXPECTED_RANK = prior.EXPECTED_RANK
EXPECTED_CELLS = prior.EXPECTED_CELLS
load = prior.load
validate_inventory_gate = prior.validate_inventory_gate


def _fresh(plan: dict[str, Any]) -> dict[str, Any]:
    plan.update(
        {
            "schema_version": SCHEMA,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "sfs_root": str(SFS_ROOT),
        }
    )
    plan["partition"]["v1_preclaim_failure_excluded"] = True
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def build_plan(root: Path) -> dict[str, Any]:
    return _fresh(prior.build_plan(root))


def build_runtime_plan(inventory: dict[str, Any], root: Path) -> dict[str, Any]:
    return _fresh(prior.build_runtime_plan(inventory, root))


def validate_all(root: Path) -> dict[str, Any]:
    plan = build_plan(root)
    if any(
        (
            plan["launch_authorized"] is not False,
            plan["job_name"] != JOB_NAME,
            plan["sfs_root"] != str(SFS_ROOT),
            plan["execution"]["endpoint_lease"]["maximum_streams"] != 2,
            plan["new_session_count"] != 4,
            plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("fresh rank-2 successor drifted")
    return plan
