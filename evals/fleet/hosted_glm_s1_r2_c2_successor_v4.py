"""Engine-complete fresh adapter for the rank-2 hosted GLM successor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_v1 as base
from evals.fleet import hosted_glm_s1_r2_c2_successor_v3 as prior
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-plan-v4"
CONTROLLER = prior.CONTROLLER
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-successor-v4"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = prior.CLAIM_ROOT
LEASE_ROOT = prior.LEASE_ROOT
LEASE_ENDPOINT_KEY = prior.LEASE_ENDPOINT_KEY
SERVING_LOAD_BLOCK = prior.SERVING_LOAD_BLOCK
EXPECTED_RANK = prior.EXPECTED_RANK
EXPECTED_CELLS = prior.EXPECTED_CELLS
CANARY_GATE_SCHEMA = base.predecessor.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = base.predecessor.RECONCILIATION_GATE_SCHEMA
COMMIT_RE = base.COMMIT_RE
SHA256_RE = base.SHA256_RE
CONTROLLERS = {CONTROLLER: base.CONTROLLERS[CONTROLLER]}
load = prior.load
validate_inventory_gate = prior.validate_inventory_gate


def _fresh(plan: dict[str, Any]) -> dict[str, Any]:
    plan.update({"schema_version": SCHEMA, "job_name": JOB_NAME, "configmap_name": CONFIGMAP_NAME, "sfs_root": str(SFS_ROOT)})
    plan["partition"]["v3_adapter_failure_excluded"] = True
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = self_hosted.digest_without(plan, "plan_sha256")
    return plan


def build_plan(root: Path) -> dict[str, Any]:
    return _fresh(prior.build_plan(root))


def build_runtime_plan(controller: str, inventory: dict[str, Any], root: Path) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown rank-2 hosted GLM controller")
    return _fresh(prior.build_runtime_plan(inventory, root))


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = build_plan(root)
    if plan["launch_authorized"] is not False or plan["execution"]["endpoint_lease"]["maximum_streams"] != 2:
        raise ValueError("rank-2 v4 held plan drifted")
    if plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"):
        raise ValueError("rank-2 v4 plan digest drifted")
    return {CONTROLLER: plan}
