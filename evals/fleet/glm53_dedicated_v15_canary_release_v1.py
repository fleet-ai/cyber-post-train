"""Fresh score-blind duplicate release for the v15 rank-51 GLM canary."""

from __future__ import annotations

import json
import os
import re
import uuid
import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import glm53_dedicated_v14_scored_canary_runtime_v1 as runtime
from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as inventory
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = runtime.RELEASE_SCHEMA
JOB_NAME = "chris-glm53-dedicated-v18-r051-release-v5"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _kube_collision(name: str, kind: str) -> int:
    if kind == "jobs":
        route = f"/apis/batch/v1/namespaces/{ledger.NAMESPACE}/jobs/{name}"
    else:
        route = f"/api/v1/namespaces/{ledger.NAMESPACE}/{kind}/{name}"
    status, _ = ledger._kube_get(route)  # noqa: SLF001
    if status not in {200, 404}:
        raise RuntimeError("dedicated v15 Kubernetes collision authority unavailable")
    return int(status == 200)


def build(root: Path, package_path: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key:
        raise RuntimeError("dedicated v15 release requires Fleet credential")
    for value in (job_uid, pod_uid):
        if uuid.UUID(value).int == 0:
            raise RuntimeError("dedicated v15 release requires nonzero observer UIDs")
    package = json.loads(package_path.read_text())
    if (
        package.get("status") != "READY_HELD"
        or package.get("launch_authorized") is not False
        or package.get("package_sha256")
        != self_hosted.digest_without(package, "package_sha256")
    ):
        raise RuntimeError("dedicated v15 controller package drifted")
    inventory_receipt = source.load(inventory.INVENTORY_PATH)
    plan = canary.build_runtime_plan(canary.CONTROLLER, inventory_receipt, root)
    all_rank = [
        row
        for row in source.build_runtime_plan(canary.CONTROLLER, inventory_receipt, root)[
            "attempts"
        ]
        if row["selection_rank"] == canary.SELECTION_RANK
    ]
    if [row["attempt"] for row in all_rank] != [1, 2, 3, 4]:
        raise RuntimeError("dedicated v15 whole-task authority drifted")
    if (
        package.get("reserved_cell_ids") != [row["cell_id"] for row in all_rank]
        or package.get("reserved_execution_ids")
        != [row["execution_id"] for row in all_rank]
        or not isinstance(package.get("held_plan_sha256"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", package["held_plan_sha256"]) is None
    ):
        raise RuntimeError("dedicated v15 reservation package drifted")
    if OUTPUT.parent.exists() or OUTPUT.parent.is_symlink():
        raise RuntimeError("dedicated v15 release output collision")
    output_collision = int(canary.SFS_ROOT.exists() or canary.SFS_ROOT.is_symlink())
    claim_collision = sum(
        (
            canary.CLAIM_ROOT / engine.claim_filename(row["execution_id"])
        ).exists()
        for row in all_rank
    )
    kube_collision = sum(
        _kube_collision(name, kind)
        for name, kind in (
            (canary.JOB_NAME, "jobs"),
            (canary.CONFIGMAP_NAME, "configmaps"),
            (canary.JOB_NAME + "-source", "configmaps"),
            (canary.JOB_NAME + "-evidence", "configmaps"),
        )
    )
    status, pods = ledger._kube_get(  # noqa: SLF001
        f"/api/v1/namespaces/{ledger.NAMESPACE}/pods?labelSelector="
        f"cyber-post-train.fleet.ai%2Fexperiment%3D{canary.JOB_NAME}"
    )
    if status != 200 or not isinstance(pods.get("items"), list):
        raise RuntimeError("dedicated v15 Pod collision authority unavailable")
    kube_collision += len(pods["items"])

    sessions: dict[str, list[dict[str, Any]]] = {}
    session_collision = 0
    base_plan = source.build_runtime_plan(canary.CONTROLLER, inventory_receipt, root)
    for item in all_rank:
        task = engine._task_for_item(base_plan, item)  # noqa: SLF001
        config = engine._attempt_config(base_plan, task, item)  # noqa: SLF001
        task_key = config["task"]["key"]
        if task_key not in sessions:
            with engine._client(key) as client:  # noqa: SLF001
                sessions[task_key] = self_hosted._task_sessions(client, task_key)
        session_collision += sum(
            ledger._session_collides(row, config, item) for row in sessions[task_key]  # noqa: SLF001
        )
    runtime._route_check(plan, key)  # noqa: SLF001
    if claim_collision or session_collision or output_collision or kube_collision:
        raise RuntimeError("dedicated v15 immediate duplicate release is not clear")
    item = plan["attempts"][0]
    body = {
        "schema_version": SCHEMA,
        "status": "CLEAR",
        "plan_sha256": plan["plan_sha256"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "selection_rank": canary.SELECTION_RANK,
        "attempt": canary.CANARY_ATTEMPT,
        "all_four_rank_cells_unstarted": True,
        "fleet_session_collisions": session_collision,
        "global_claim_collisions": claim_collision,
        "sfs_output_collisions": output_collision,
        "kubernetes_object_collisions": kube_collision,
        "checked_immediately_before_create": True,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "observed_at_utc": _now(),
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-package", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(Path.cwd(), args.controller_package)
    OUTPUT.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
