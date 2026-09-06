"""Score-blind rank-30 release observer bound to live rank-29 attempt 4."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_single_slot_release_v1 as prior
from evals.fleet import hosted_glm_rank30_single_slot_v3 as successor
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-exact100-hosted-r030-single-slot-release-v3"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
engine = prior.engine
whole = prior.whole


def _file_sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_live_peer() -> dict[str, Any]:
    status, job = prior.kube._kube_get("jobs", successor.PEER_JOB)  # noqa: SLF001
    pod_status, pod = prior.kube._kube_get(  # noqa: SLF001
        "pods", successor.PEER_POD
    )
    statuses = pod.get("status", {}).get("containerStatuses", [])
    evaluator = next(
        (row for row in statuses if isinstance(row, dict) and row.get("name") == "evaluator"),
        None,
    )
    if (
        status != 200
        or job.get("metadata", {}).get("uid") != successor.PEER_JOB_UID
        or job.get("status", {}).get("active") != 1
        or bool(job.get("status", {}).get("failed"))
        or bool(job.get("status", {}).get("succeeded"))
        or pod_status != 200
        or pod.get("metadata", {}).get("uid") != successor.PEER_POD_UID
        or pod.get("status", {}).get("phase") != "Running"
        or not isinstance(evaluator, dict)
        or evaluator.get("ready") is not True
        or evaluator.get("restartCount") != 0
    ):
        raise RuntimeError("rank-29 attempt-4 peer identity or health drifted")

    accepted = prior.kube._safe_receipt(  # noqa: SLF001
        successor.PEER_A3_ACCEPTED_PATH
    )
    claim = prior.kube._safe_receipt(successor.PEER_A4_CLAIM_PATH)  # noqa: SLF001
    if any(
        (
            _file_sha(successor.PEER_A3_ACCEPTED_PATH)
            != successor.PEER_A3_ACCEPTED_FILE,
            accepted.get("receipt_sha256") != successor.PEER_A3_ACCEPTED_SELF,
            accepted.get("accepted") is not True,
            accepted.get("credited") is not True,
            accepted.get("selection_rank") != 29,
            accepted.get("attempt") != 3,
            accepted.get("cell_id") != successor.PEER_A3_CELL,
            accepted.get("execution_id") != successor.PEER_A3_EXECUTION,
            accepted.get("run_id") != successor.PEER_A3_RUN,
            _file_sha(successor.PEER_A4_CLAIM_PATH) != successor.PEER_A4_CLAIM_FILE,
            claim.get("receipt_sha256") != successor.PEER_A4_CLAIM_SELF,
            claim.get("selection_rank") != 29,
            claim.get("attempt") != 4,
            claim.get("cell_id") != successor.PEER_A4_CELL,
            claim.get("execution_id") != successor.PEER_A4_EXECUTION,
            claim.get("run_id") != successor.PEER_A4_RUN,
            claim.get("job_uid") != successor.PEER_JOB_UID,
            claim.get("pod_uid") != successor.PEER_POD_UID,
        )
    ):
        raise RuntimeError("rank-29 a3-to-a4 evidence drifted")
    stream = successor.PEER_STREAM
    if stream.is_symlink() or not stream.is_file():
        raise RuntimeError("rank-29 attempt-4 stream is absent or unsafe")
    metadata = stream.stat()
    stream_mtime = int(metadata.st_mtime)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or datetime.now(UTC).timestamp() - stream_mtime
        > successor.PEER_STREAM_MAX_AGE_SECONDS
    ):
        raise RuntimeError("rank-29 attempt-4 stream is stale")
    return {
        "job_name": successor.PEER_JOB,
        "job_uid": successor.PEER_JOB_UID,
        "job_active": 1,
        "pod_name": successor.PEER_POD,
        "pod_uid": successor.PEER_POD_UID,
        "pod_phase": "Running",
        "pod_ready": True,
        "pod_restarts": 0,
        "accepted_attempt": 3,
        "accepted_receipt_path": str(successor.PEER_A3_ACCEPTED_PATH),
        "accepted_receipt_sha256": successor.PEER_A3_ACCEPTED_SELF,
        "accepted_file_sha256": successor.PEER_A3_ACCEPTED_FILE,
        "active_attempt": 4,
        "active_claim_path": str(successor.PEER_A4_CLAIM_PATH),
        "active_claim_sha256": successor.PEER_A4_CLAIM_SELF,
        "active_claim_file_sha256": successor.PEER_A4_CLAIM_FILE,
        "active_run_id": successor.PEER_A4_RUN,
        "stream_path": str(stream),
        "stream_bytes": metadata.st_size,
        "stream_mtime_epoch": stream_mtime,
        "endpoint_slot_held": 1,
    }


def build(root: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    source_sha = os.environ.get("GLM_HOSTED_R30_SOURCE_SHA256", "")
    if (
        not key
        or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid))
        or successor.SHA256_RE.fullmatch(source_sha) is None
    ):
        raise RuntimeError("rank-30 v3 observer bindings are invalid")
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    peer = _validate_live_peer()
    lease = prior._probe_slots()  # noqa: SLF001
    collision = prior._collisions(plan, key)  # noqa: SLF001
    if any(
        collision[field]
        for field in (
            "canonical_claim_collisions",
            "authoritative_session_collisions",
            "accepted_evidence_collisions",
            "output_root_collisions",
            "new_job_collisions",
            "new_configmap_collisions",
            "api_mutations",
        )
    ):
        raise RuntimeError("rank-30 v3 collision gate is not clear")
    body = {
        "schema_version": successor.RELEASE_SCHEMA,
        "status": "CLEAR",
        "checked_at_utc": prior._now(),  # noqa: SLF001
        "launch_authorized": True,
        "scoring_authorized": True,
        "controller_cap": 1,
        "controller": successor.release_projection(plan),
        "source_package_sha256": source_sha,
        "ledger_authority": whole.LEDGER_AUTHORITY,
        "selection_authority": whole.SELECTION_AUTHORITY,
        "live_rank29_peer": peer,
        "endpoint_lease_observer": lease,
        "fresh_collision_reconciliation": collision,
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    receipt = {
        **body,
        "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256"),
    }
    successor.validate_release(receipt, plan, source_sha)
    return receipt


def main() -> int:
    receipt = build(Path.cwd())
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
