"""Execute the released GLM v19 rank-51 attempt-2 canary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import glm53_dedicated_v14_scored_canary_runtime_v1 as base
from evals.fleet import glm53_dedicated_v19_a2_v1 as canary
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as inventory
from evals.fleet import self_hosted

RELEASE_SCHEMA = "fleet-glm53-dedicated-v19-a2-release-v1"
base.canary = canary


def _runtime_gate(plan: dict[str, Any]) -> None:
    release = canary.load(Path(os.environ.get("DEDICATED_RELEASE_PATH", "/bootstrap/release.json")))
    item = plan["attempts"][0]
    required = {
        "schema_version": RELEASE_SCHEMA,
        "status": "CLEAR",
        "plan_sha256": plan["plan_sha256"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "selection_rank": 51,
        "attempt": 2,
        "accepted_predecessor_attempts": [1],
        "fleet_session_collisions": 0,
        "global_claim_collisions": 0,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "checked_immediately_before_create": True,
    }
    if any(release.get(k) != v for k, v in required.items()):
        raise RuntimeError("v19 a2 release gate drifted")
    if release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256"):
        raise RuntimeError("v19 a2 release self digest drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    plan = canary.build_runtime_plan(canary.CONTROLLER, canary.load(inventory.INVENTORY_PATH), root)
    engine.bulk = canary
    return engine.run_controller(
        plan, out=canary.SFS_ROOT, proxy=proxy,
        route_check=base._route_check, runtime_gate_check=_runtime_gate,
    )


def bootstrap_only(root: Path) -> dict[str, Any]:
    plan = canary.build_runtime_plan(
        canary.CONTROLLER, canary.load(inventory.INVENTORY_PATH), root
    )
    _runtime_gate(plan)
    receipt = base._bootstrap_only(root)
    receipt["server_bound_runtime_release_gate_valid"] = True
    receipt["runtime_plan_sha256"] = plan["plan_sha256"]
    receipt["receipt_sha256"] = self_hosted.digest_without(receipt, "receipt_sha256")
    output = Path(os.environ["DEDICATED_BOOTSTRAP_RECEIPT"])
    output.write_bytes(self_hosted.canonical_json(receipt) + b"\n")
    return receipt
