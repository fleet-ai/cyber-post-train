"""Fresh rank-51 attempt-2 controller bound to GLM v22."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v20_a2_v1 as evidence_base
from evals.fleet import glm53_dedicated_v21_a2_v1 as prior
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-glm53-dedicated-v22-a2-plan-v1"
CONTROLLER = prior.CONTROLLER
JOB_NAME = "chris-glm53-dedicated-v22-r051-a2-canary-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = prior.CLAIM_ROOT
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-dedicated-v22-v1")
SERVING_BLOCK = "glm-dedicated-v22-r051-v1"
CONTROLLERS = prior.CONTROLLERS
load = prior.load
validate_inventory_gate = prior.validate_inventory_gate
validate_all = prior.validate_all
_validate_evidence = evidence_base._validate_evidence


def _transform(
    plan: dict[str, Any],
    *,
    parity: dict[str, Any],
    binding: dict[str, Any],
    service_origin: str,
    runtime: bool,
) -> dict[str, Any]:
    body = prior._transform(
        plan,
        parity=parity,
        binding=binding,
        service_origin=service_origin,
        runtime=runtime,
    )
    body.update(
        {
            "schema_version": SCHEMA,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "sfs_root": str(SFS_ROOT),
            "serving_load_block": SERVING_BLOCK,
        }
    )
    body["execution"]["endpoint_lease"] = {
        "lease_root": str(LEASE_ROOT),
        "endpoint_key": binding["api_run_id"],
        "maximum_streams": 1,
    }
    body["serving"]["serving_block"] = SERVING_BLOCK
    body.pop("plan_sha256", None)
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body


def build_plan(
    root: Path, *, service_origin: str, parity_path: Path, binding_path: Path
) -> dict[str, Any]:
    return _transform(
        source.validate_all(root)[CONTROLLER],
        parity=load(parity_path),
        binding=load(binding_path),
        service_origin=service_origin,
        runtime=False,
    )


def build_runtime_plan(
    controller: str,
    inventory_receipt: dict[str, Any],
    root: Path,
    *,
    service_origin: str | None = None,
    parity_path: Path | None = None,
    binding_path: Path | None = None,
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown v22 controller")
    origin = service_origin or os.environ.get("DEDICATED_SERVICE_ORIGIN")
    parity_path = parity_path or Path(
        os.environ.get("DEDICATED_PARITY_PATH", "/bootstrap/parity.json")
    )
    binding_path = binding_path or Path(
        os.environ.get("DEDICATED_BINDING_PATH", "/bootstrap/binding.json")
    )
    if origin is None:
        raise ValueError("v22 origin absent")
    return _transform(
        source.build_runtime_plan(CONTROLLER, inventory_receipt, root),
        parity=load(parity_path),
        binding=load(binding_path),
        service_origin=origin,
        runtime=True,
    )
