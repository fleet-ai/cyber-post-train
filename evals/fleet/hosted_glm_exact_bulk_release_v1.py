"""Create the score-blind immediate release gate for hosted GLM bulk."""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as runtime
from evals.fleet import hosted_glm_exact_bulk_v1 as bulk
from evals.fleet import self_hosted

OUTPUT_ROOT = runtime.RELEASE_PATH.parent
TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
NAMESPACE = "fleet-train-jobs"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _kube_get(path: str) -> tuple[int, dict[str, Any]]:
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    ca = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    with httpx.Client(
        base_url="https://kubernetes.default.svc",
        headers={"Authorization": f"Bearer {token}"},
        verify=ca,
        timeout=30,
    ) as client:
        response = client.get(path)
    value = response.json() if response.content else {}
    return response.status_code, value


def _kubernetes_collisions() -> int:
    collisions = 0
    for source in bulk.CONTROLLERS.values():
        for kind, name in (("jobs", source["job_name"]), ("configmaps", source["configmap_name"])):
            status, _ = (
                _kube_get(f"/apis/batch/v1/namespaces/{NAMESPACE}/{kind}/{name}")
                if kind == "jobs"
                else _kube_get(f"/api/v1/namespaces/{NAMESPACE}/{kind}/{name}")
            )
            if status == 200:
                collisions += 1
            elif status != 404:
                raise RuntimeError("Kubernetes collision authority unavailable")
        status, value = _kube_get(
            f"/api/v1/namespaces/{NAMESPACE}/pods?labelSelector="
            f"cyber-post-train.fleet.ai%2Fexperiment%3D{source['job_name']}"
        )
        if status != 200 or not isinstance(value.get("items"), list):
            raise RuntimeError("Kubernetes Pod collision authority unavailable")
        collisions += len(value["items"])
    return collisions


def build(root: Path) -> dict[str, Any]:
    if not os.environ.get("FLEET_API_KEY"):
        raise RuntimeError("FLEET_API_KEY is required")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    for value in (job_uid, pod_uid):
        if uuid.UUID(value).int == 0:
            raise RuntimeError("release observer requires nonzero runtime UIDs")
    _, terminal = runtime._canary_gate()  # noqa: SLF001 - exact prerequisite gate
    inventory = bulk.load(runtime.INVENTORY_PATH)
    plans = {name: bulk.build_runtime_plan(name, inventory, root) for name in bulk.CONTROLLERS}
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise RuntimeError("hosted GLM bulk release root already exists")
    claims = [
        Path(bulk.CLAIM_ROOT) / engine.claim_filename(row["execution_id"])
        for plan in plans.values()
        for row in plan["attempts"]
    ]
    claim_collisions = sum(path.exists() or path.is_symlink() for path in claims)
    output_collisions = sum(
        Path(plan["sfs_root"]).exists() or Path(plan["sfs_root"]).is_symlink()
        for plan in plans.values()
    )
    key = os.environ["FLEET_API_KEY"]
    session_collisions = 0
    task_cache: dict[str, list[dict[str, Any]]] = {}
    for plan in plans.values():
        runtime.engine._fresh_route_check(plan, key)  # noqa: SLF001
        tasks = {row["rank"]: row for row in plan["tasks"]}
        for item in plan["attempts"]:
            config = runtime.engine._attempt_config(  # noqa: SLF001
                plan, tasks[item["selection_rank"]], item
            )
            task_key = config["task"]["key"]
            if task_key not in task_cache:
                with runtime.engine._client(key) as client:  # noqa: SLF001
                    task_cache[task_key] = self_hosted._task_sessions(client, task_key)
            expected_model = self_hosted.persisted_session_model_identity(config)
            for row in task_cache[task_key]:
                metadata = row.get("metadata") or {}
                if row.get("model") == expected_model or any(
                    metadata.get(field) == expected
                    for field, expected in {
                        "run_id": config["run_id"],
                        "cell_id": item["cell_id"],
                        "execution_id": item["execution_id"],
                    }.items()
                ):
                    session_collisions += 1
    kubernetes_collisions = _kubernetes_collisions()
    if claim_collisions or output_collisions or session_collisions or kubernetes_collisions:
        raise RuntimeError("hosted GLM bulk duplicate gate is not clear")
    plan_shas = {name: value["plan_sha256"] for name, value in bulk.validate_all(root).items()}
    execution_ids = sorted(
        row["execution_id"] for plan in plans.values() for row in plan["attempts"]
    )
    body = {
        "schema_version": runtime.RELEASE_SCHEMA,
        "status": "CLEAR",
        "canary_terminal_receipt_sha256": terminal["receipt_sha256"],
        "controller_plan_sha256s": plan_shas,
        "execution_ids_sha256": self_hosted.sha256(self_hosted.canonical_json(execution_ids)),
        "cell_count": len(execution_ids),
        "checked_immediately_before_create": True,
        "fleet_session_collisions": session_collisions,
        "global_claim_collisions": claim_collisions,
        "sfs_output_collisions": output_collisions,
        "kubernetes_object_collisions": kubernetes_collisions,
        "active_or_accepted_collisions": 0,
        "mutation_calls": 0,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "observed_at_utc": _now(),
        "scores_read": False,
        "prompts_traces_flags_read": False,
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    receipt = build(args.repo.resolve())
    OUTPUT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(runtime.RELEASE_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
