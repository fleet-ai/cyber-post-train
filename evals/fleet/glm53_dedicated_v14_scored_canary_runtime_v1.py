"""Execute the released one-cell GLM dedicated canary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import httpx

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as inventory
from evals.fleet import self_hosted

RELEASE_SCHEMA = "fleet-glm53-dedicated-v14-canary-release-v1"


def _route_check(plan: dict[str, Any], key: str) -> None:
    origin = plan["model"]["endpoint_origin"].rstrip("/")
    parity_path = Path(os.environ.get("DEDICATED_PARITY_PATH", "/bootstrap/parity.json"))
    canary._validate_evidence(  # noqa: SLF001
        canary.load(parity_path), plan["dedicated_server_binding"], origin
    )
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=180,
    ) as client:
        account = self_hosted._request(client, "GET", "/v1/account")
    if (
        account.get("team_name") != "fleet"
        or account.get("team_id") != self_hosted.FLEET_TEAM_ID
    ):
        raise RuntimeError("dedicated canary Fleet team identity drifted")
    with urlopen(origin + "/health", timeout=30) as response:
        if response.status != 200:
            raise RuntimeError("dedicated GLM service is unhealthy")
    with urlopen(origin + "/v1/models", timeout=30) as response:
        roster = json.load(response)
    if [row.get("id") for row in roster.get("data", [])] != ["glm-5.3"]:
        raise RuntimeError("dedicated GLM served model drifted")


def _runtime_gate(plan: dict[str, Any]) -> None:
    release_path = Path(os.environ.get("DEDICATED_RELEASE_PATH", "/bootstrap/release.json"))
    release = canary.load(release_path)
    item = plan["attempts"][0]
    if (
        release.get("schema_version") != RELEASE_SCHEMA
        or release.get("status") != "CLEAR"
        or release.get("plan_sha256") != plan["plan_sha256"]
        or release.get("cell_id") != item["cell_id"]
        or release.get("execution_id") != item["execution_id"]
        or release.get("all_four_rank_cells_unstarted") is not True
        or release.get("fleet_session_collisions") != 0
        or release.get("global_claim_collisions") != 0
        or release.get("kubernetes_object_collisions") != 0
        or release.get("checked_immediately_before_create") is not True
        or release.get("receipt_sha256")
        != self_hosted.digest_without(release, "receipt_sha256")
    ):
        raise RuntimeError("dedicated canary release gate drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory_receipt = canary.load(inventory.INVENTORY_PATH)
    plan = canary.build_runtime_plan(canary.CONTROLLER, inventory_receipt, root)
    engine.bulk = canary
    return engine.run_controller(
        plan,
        out=canary.SFS_ROOT,
        proxy=proxy,
        route_check=_route_check,
        runtime_gate_check=_runtime_gate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
