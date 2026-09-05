"""Score-blind duplicate/reconciliation preflight for the dedicated GLM canary."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import glm53_dedicated_v14_scored_canary_v1 as canary
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as inventory
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-glm53-dedicated-v14-scored-canary-preflight-v1"
JOB_NAME = "chris-glm53-dedicated-v14-r051-preflight-v5"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "PREPARED.json"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _kube_absent(name: str, kind: str) -> None:
    prefix = "/apis/batch/v1" if kind == "jobs" else "/api/v1"
    status, _ = ledger._kube_get(  # noqa: SLF001
        f"{prefix}/namespaces/{ledger.NAMESPACE}/{kind}/{name}"
    )
    if status != 404:
        raise RuntimeError("dedicated GLM canary Kubernetes collision")


def build(root: Path, package_path: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("dedicated GLM preflight requires key and nonzero UIDs")
    package = json.loads(package_path.read_text())
    if (
        package.get("schema_version")
        != "fleet-glm53-dedicated-v14-scored-canary-package-v1"
        or package.get("status") != "READY_HELD"
        or package.get("launch_authorized") is not False
        or package.get("package_sha256")
        != self_hosted.digest_without(package, "package_sha256")
    ):
        raise RuntimeError("dedicated GLM controller package preview drifted")
    inventory_receipt = source.load(inventory.INVENTORY_PATH)
    full = source.build_runtime_plan(canary.CONTROLLER, inventory_receipt, root)
    rank = [row for row in full["attempts"] if row["selection_rank"] == canary.SELECTION_RANK]
    if [row["attempt"] for row in rank] != [1, 2, 3, 4]:
        raise RuntimeError("dedicated GLM whole-task selection drifted")
    if package.get("reserved_cell_ids") != [row["cell_id"] for row in rank]:
        raise RuntimeError("dedicated GLM package cell reservation drifted")
    if package.get("reserved_execution_ids") != [row["execution_id"] for row in rank]:
        raise RuntimeError("dedicated GLM package execution reservation drifted")
    if canary.SFS_ROOT.exists() or canary.SFS_ROOT.is_symlink():
        raise RuntimeError("dedicated GLM canary SFS output collision")
    _kube_absent(canary.JOB_NAME, "jobs")
    _kube_absent(canary.JOB_NAME + "-source", "configmaps")
    _kube_absent(canary.JOB_NAME + "-evidence", "configmaps")
    claim_collisions = 0
    session_collisions = 0
    sessions: list[dict[str, Any]] | None = None
    for item in rank:
        claim = canary.CLAIM_ROOT / engine.claim_filename(item["execution_id"])
        claim_collisions += int(claim.exists() or claim.is_symlink())
        task = engine._task_for_item(full, item)  # noqa: SLF001
        config = engine._attempt_config(full, task, item)  # noqa: SLF001
        if sessions is None:
            with engine._client(key) as client:  # noqa: SLF001
                sessions = self_hosted._task_sessions(client, config["task"]["key"])
        session_collisions += sum(
            ledger._session_collides(row, config, item) for row in sessions  # noqa: SLF001
        )
    if claim_collisions or session_collisions:
        raise RuntimeError("dedicated GLM canary duplicate ledger is not clear")
    body = {
        "schema_version": SCHEMA,
        "status": "CLEAR_HELD",
        "launch_authorized": False,
        "controller_package_sha256": package["package_sha256"],
        "controller_objects_sha256": package["objects_sha256"],
        "held_plan_sha256": package["held_plan_sha256"],
        "selection_rank": canary.SELECTION_RANK,
        "reserved_cell_ids": package["reserved_cell_ids"],
        "reserved_execution_ids": package["reserved_execution_ids"],
        "all_four_rank_cells_unstarted": True,
        "fleet_session_collisions": session_collisions,
        "global_claim_collisions": claim_collisions,
        "kubernetes_object_collisions": 0,
        "sfs_output_collisions": 0,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "checked_at_utc": _now(),
        "mutation_calls": 0,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "fresh_server_binding_required_after_admission": True,
        "fresh_non_scored_parity_required_after_admission": True,
        "fresh_release_required_immediately_before_canary_create": True,
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def main() -> int:
    receipt = build(Path.cwd(), Path("/bootstrap/controller-package.json"))
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
