"""Fresh rank-2 controller identity after the v2 package-closure failure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_s1_r2_c2_successor_v2 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-plan-v3"
CONTROLLER = prior.CONTROLLER
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-successor-v3"
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
    plan.update({"schema_version": SCHEMA, "job_name": JOB_NAME, "configmap_name": CONFIGMAP_NAME, "sfs_root": str(SFS_ROOT)})
    plan["partition"]["v2_package_closure_failure_excluded"] = True
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def build_plan(root: Path) -> dict[str, Any]:
    return _fresh(prior.build_plan(root))


def build_runtime_plan(inventory: dict[str, Any], root: Path) -> dict[str, Any]:
    return _fresh(prior.build_runtime_plan(inventory, root))


def validate_all(root: Path) -> dict[str, Any]:
    value = build_plan(root)
    if value["launch_authorized"] is not False or value["execution"]["endpoint_lease"]["maximum_streams"] != 2:
        raise ValueError("rank-2 v3 held plan drifted")
    if value["plan_sha256"] != self_hosted.digest_without(value, "plan_sha256"):
        raise ValueError("rank-2 v3 plan digest drifted")
    return value
