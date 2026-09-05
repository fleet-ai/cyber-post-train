"""Fresh score-blind release gate for the one-cell GLM concurrency-two canary."""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import hosted_glm_exact_bulk_v1 as bulk
from evals.fleet import hosted_glm_s2_c2_canary_v1 as c2
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s2-c2-release-v1"
JOB_NAME = "chris-glm53-exact100-hosted-s2-c2-release-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
S1_JOB = "chris-glm53-exact100-hosted-s1-retry-v2"
S1_JOB_UID = "8946119b-1281-4eb2-a8b8-7defe2649b7c"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _first_s1_acceptance(root: Path) -> dict[str, Any]:
    inventory = bulk.load(bulk_runtime.INVENTORY_PATH)
    plan = bulk.build_runtime_plan("glm-hosted-s1", inventory, root)
    first = min(plan["attempts"], key=lambda row: row["ordinal"])
    path = Path(plan["sfs_root"]) / "accepted" / f"{first['run_id']}.json"
    receipt = bulk.load(path)
    if any(
        (
            receipt.get("schema_version") != "fleet-exact-pass4-bulk-cell-accepted-v3",
            receipt.get("accepted") is not True,
            receipt.get("credited") is not True,
            receipt.get("retry_allowed") is not False,
            receipt.get("cell_id") != first["cell_id"],
            receipt.get("execution_id") != first["execution_id"],
            receipt.get("run_id") != first["run_id"],
            receipt.get("receipt_sha256")
            != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("GLM s1 first planned cell is not digest-valid accepted")
    return receipt


def _kube_absent(name: str, kind: str) -> None:
    prefix = "/apis/batch/v1" if kind == "jobs" else "/api/v1"
    path = f"{prefix}/namespaces/{ledger.NAMESPACE}/{kind}/{name}"
    status, _ = ledger._kube_get(path)  # noqa: SLF001
    if status != 404:
        raise RuntimeError("GLM s2 concurrency-two Kubernetes collision")


def _s1_active_and_lease_held() -> None:
    path = f"/apis/batch/v1/namespaces/{ledger.NAMESPACE}/jobs/{S1_JOB}"
    status, job = ledger._kube_get(path)  # noqa: SLF001
    if (
        status != 200
        or job.get("metadata", {}).get("uid") != S1_JOB_UID
        or job.get("status", {}).get("active") != 1
        or job.get("status", {}).get("failed")
        or job.get("status", {}).get("succeeded")
    ):
        raise RuntimeError("GLM s1 is not the exact active concurrency-one predecessor")
    lease = Path(c2.LEASE_ROOT) / c2.LEASE_ENDPOINT_KEY / "slot-1.lock"
    if lease.is_symlink() or not lease.is_file():
        raise RuntimeError("GLM s1 exact endpoint lease slot is absent or unsafe")
    handle = lease.open("a+b")
    try:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise RuntimeError("GLM s1 endpoint lease slot is not regular")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        raise RuntimeError("GLM s1 endpoint lease slot is not actively held")
    finally:
        handle.close()


def build(root: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    if not key or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("GLM c2 release observer requires key and nonzero UIDs")
    s1 = _first_s1_acceptance(root)
    _s1_active_and_lease_held()
    inventory = bulk.load(bulk_runtime.INVENTORY_PATH)
    plan = c2.build_runtime_plan(c2.CONTROLLER, inventory, root)
    if c2.SFS_ROOT.exists() or c2.SFS_ROOT.is_symlink():
        raise RuntimeError("GLM c2 canary output root already exists")
    item = plan["attempts"][0]
    claim = Path(c2.CLAIM_ROOT) / engine.claim_filename(item["execution_id"])
    claim_collisions = int(claim.exists() or claim.is_symlink())
    engine._fresh_route_check(plan, key)  # noqa: SLF001
    task = plan["tasks"][0]
    config = engine._attempt_config(plan, task, item)  # noqa: SLF001
    with engine._client(key) as client:  # noqa: SLF001
        sessions = self_hosted._task_sessions(client, config["task"]["key"])
    session_collisions = sum(ledger._session_collides(row, config, item) for row in sessions)  # noqa: SLF001
    _kube_absent(c2.JOB_NAME, "jobs")
    _kube_absent(c2.CONFIGMAP_NAME, "configmaps")
    if claim_collisions or session_collisions:
        raise RuntimeError("GLM c2 canary exact duplicate ledger is not clear")
    body = {
        "schema_version": SCHEMA,
        "status": "CLEAR",
        "successor_job": c2.JOB_NAME,
        "successor_configmap": c2.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "cell_id": item["cell_id"],
        "execution_id": item["execution_id"],
        "s1_first_accepted_receipt_sha256": s1["receipt_sha256"],
        "s1_job": S1_JOB,
        "s1_job_uid": S1_JOB_UID,
        "s1_active": True,
        "s1_slot1_lease_held": True,
        "fleet_session_collisions": session_collisions,
        "global_claim_collisions": claim_collisions,
        "kubernetes_object_collisions": 0,
        "active_or_accepted_collisions": 0,
        "serving_load_block": c2.SERVING_LOAD_BLOCK,
        "lease_root": str(c2.LEASE_ROOT),
        "lease_endpoint_key": c2.LEASE_ENDPOINT_KEY,
        "maximum_scored_streams": 2,
        "checked_immediately_before_create": True,
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
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
