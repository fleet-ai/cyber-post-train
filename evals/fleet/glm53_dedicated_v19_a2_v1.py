"""Fresh rank-51 attempt-2 controller bound to the GLM v19 server."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as base
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-glm53-dedicated-v19-a2-plan-v1"
CONTROLLER = base.CONTROLLER
JOB_NAME = "chris-glm53-dedicated-v19-r051-a2-canary-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = base.CLAIM_ROOT
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-dedicated-v19-v1")
SELECTION_RANK = 51
CANARY_ATTEMPT = 2
SERVING_BLOCK = "glm-dedicated-v19-r051-v1"
CONTROLLERS = base.CONTROLLERS
SHA256_RE = base.SHA256_RE
COMMIT_RE = base.COMMIT_RE
CANARY_GATE_SCHEMA = base.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = base.RECONCILIATION_GATE_SCHEMA

load = base.load
validate_inventory_gate = base.validate_inventory_gate
validate_all = base.validate_all
_validate_evidence = base._validate_evidence


def _transform(
    plan: dict[str, Any], *, parity: dict[str, Any], binding: dict[str, Any],
    service_origin: str, runtime: bool,
) -> dict[str, Any]:
    body = base._transform(
        plan, parity=parity, binding=binding, service_origin=service_origin, runtime=runtime
    )
    rank = [row for row in plan["attempts"] if row["selection_rank"] == SELECTION_RANK]
    if [row["attempt"] for row in rank] != [1, 2, 3, 4]:
        raise ValueError("v19 rank-51 authority drifted")
    item = copy.deepcopy(rank[1])
    item["ordinal"] = 1
    body.update({
        "schema_version": SCHEMA,
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
        "sfs_root": str(SFS_ROOT),
        "attempts": [item],
        "serving_load_block": SERVING_BLOCK,
    })
    body["whole_task_reservation"].update({
        "all_four_unstarted_required_at_release": False,
        "accepted_predecessor_attempts_required": [1],
        "remaining_attempts_require_post_canary_release": [3, 4],
    })
    body["execution"]["endpoint_lease"] = {
        "lease_root": str(LEASE_ROOT),
        "endpoint_key": binding["api_run_id"],
        "maximum_streams": 1,
    }
    body["execution"]["traffic_heartbeat_path"] = None
    body["serving"]["serving_block"] = SERVING_BLOCK
    body.pop("plan_sha256", None)
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body


def build_plan(root: Path, *, service_origin: str, parity_path: Path, binding_path: Path) -> dict[str, Any]:
    return _transform(
        source.validate_all(root)[CONTROLLER],
        parity=load(parity_path), binding=load(binding_path),
        service_origin=service_origin, runtime=False,
    )


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path, *,
    service_origin: str | None = None, parity_path: Path | None = None,
    binding_path: Path | None = None,
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown v19 controller")
    origin = service_origin or os.environ.get("DEDICATED_SERVICE_ORIGIN")
    if origin is None:
        raise ValueError("v19 service origin absent")
    parity_path = parity_path or Path(os.environ.get("DEDICATED_PARITY_PATH", "/bootstrap/parity.json"))
    binding_path = binding_path or Path(os.environ.get("DEDICATED_BINDING_PATH", "/bootstrap/binding.json"))
    return _transform(
        source.build_runtime_plan(CONTROLLER, inventory_receipt, root),
        parity=load(parity_path), binding=load(binding_path),
        service_origin=origin, runtime=True,
    )
