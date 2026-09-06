"""Rank-30 single-slot authority bound to live rank-29 attempt 4."""

from __future__ import annotations

import hashlib
import json
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import hosted_glm_rank30_single_slot_v1 as prior
from evals.fleet import self_hosted

CONTROLLER = prior.CONTROLLER
CONTROLLERS = prior.CONTROLLERS
RELEASE_SCHEMA = "fleet-hosted-glm-rank30-single-slot-release-v3"
RELEASE_PATH = prior.RELEASE_PATH
RELEASE_MAX_AGE_SECONDS = 600
PEER_STREAM_MAX_AGE_SECONDS = 300
PEER_JOB = prior.PEER_JOB
PEER_JOB_UID = prior.PEER_JOB_UID
PEER_POD = prior.PEER_POD
PEER_POD_UID = prior.PEER_POD_UID
PEER_A3_ACCEPTED_PATH = Path(
    "/mnt/sfs/jobs/chris-glm53-exact100-hosted-r029-a3a4-successor-v1/accepted/"
    "chris-glm53-ac-bulk-b-r029-a3-g2-b51782f9.json"
)
PEER_A3_ACCEPTED_SELF = (
    "sha256:81a090a15e1847646591e701c350639a31780e3dc6a83f96c9b60b5159f2af3d"
)
PEER_A3_ACCEPTED_FILE = (
    "sha256:ad6b16377e5b73e9db102ce7bae46594534a4ecb67a1ad5a6fe5990960a9b27d"
)
PEER_A3_CELL = (
    "sha256:bd6f3cc1524132f17b48bb6b655ad047b82858deaf58fdbfa6888658a45c7a3b"
)
PEER_A3_EXECUTION = (
    "sha256:6df2e9eb387b1d6cf2cbd5c1e82d955f5a90b07d44f0cfd4ba4d83139e1cf993"
)
PEER_A3_RUN = "chris-glm53-ac-bulk-b-r029-a3-g2-b51782f9"
PEER_A4_CLAIM_PATH = Path(
    "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/"
    "708601826b6acefa693d49318475fceca0541095af3f5422d10e7c7b15e1a72b.json"
)
PEER_A4_CLAIM_SELF = (
    "sha256:cb297199e45e8192dd3cb1308f91cc83d5c9ea9cec1c9fb32325ab16dc52e091"
)
PEER_A4_CLAIM_FILE = (
    "sha256:570335c5388d60fa5382f7d50dce1e74b61d96a93b395ba264fe4226b3d1955e"
)
PEER_A4_CELL = (
    "sha256:414d1bc62c3e9af45b63cea021bf4be8e8584905c0298225f209f2b7ab58e821"
)
PEER_A4_EXECUTION = (
    "sha256:708601826b6acefa693d49318475fceca0541095af3f5422d10e7c7b15e1a72b"
)
PEER_A4_RUN = "chris-glm53-ac-bulk-b-r029-a4-g2-b51782f9"
PEER_STREAM = (
    prior.JOBS_ROOT
    / PEER_JOB
    / "attempts"
    / PEER_A4_RUN
    / "agent-output/opencode-stream.jsonl"
)

SHA256_RE = prior.SHA256_RE
COMMIT_RE = prior.COMMIT_RE
CANARY_GATE_SCHEMA = prior.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = prior.RECONCILIATION_GATE_SCHEMA
validate_inventory_gate = prior.validate_inventory_gate
load = prior.load
AtomicWholeTaskClaims = prior.AtomicWholeTaskClaims
CLAIM_ROOT = prior.CLAIM_ROOT
JOBS_ROOT = prior.JOBS_ROOT
LEASE_ROOT = prior.LEASE_ROOT
LEASE_ENDPOINT_KEY = prior.LEASE_ENDPOINT_KEY
build_plan = prior.build_plan
build_runtime_plan = prior.build_runtime_plan
validate_all = prior.validate_all
release_projection = prior.release_projection


def _safe_receipt(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 262_144:
        raise RuntimeError("rank-29 transition receipt is unsafe")
    payload = path.read_bytes()
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("receipt_sha256")
        != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise RuntimeError("rank-29 transition receipt digest drifted")
    return value, "sha256:" + hashlib.sha256(payload).hexdigest()


def _kube_get(path: str) -> tuple[int, dict[str, Any]]:
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text().strip()
    with httpx.Client(
        base_url="https://kubernetes.default.svc",
        headers={"Authorization": f"Bearer {token}"},
        verify="/var/run/secrets/kubernetes.io/serviceaccount/ca.crt",
        timeout=30,
    ) as client:
        response = client.get(path)
    value = response.json() if response.content else {}
    if not isinstance(value, dict):
        raise RuntimeError("rank-29 Kubernetes response is invalid")
    return response.status_code, value


def validate_current_peer() -> dict[str, Any]:
    status, job = _kube_get(
        f"/apis/batch/v1/namespaces/fleet-train-jobs/jobs/{PEER_JOB}"
    )
    pod_status, pod = _kube_get(
        f"/api/v1/namespaces/fleet-train-jobs/pods/{PEER_POD}"
    )
    statuses = pod.get("status", {}).get("containerStatuses", [])
    evaluator = next(
        (row for row in statuses if isinstance(row, dict) and row.get("name") == "evaluator"),
        None,
    )
    if (
        status != 200
        or job.get("metadata", {}).get("uid") != PEER_JOB_UID
        or job.get("status", {}).get("active") != 1
        or bool(job.get("status", {}).get("failed"))
        or bool(job.get("status", {}).get("succeeded"))
        or pod_status != 200
        or pod.get("metadata", {}).get("uid") != PEER_POD_UID
        or pod.get("status", {}).get("phase") != "Running"
        or not isinstance(evaluator, dict)
        or evaluator.get("ready") is not True
        or evaluator.get("restartCount") != 0
    ):
        raise RuntimeError("rank-29 attempt-4 peer identity or health drifted")
    accepted, accepted_file = _safe_receipt(PEER_A3_ACCEPTED_PATH)
    claim, claim_file = _safe_receipt(PEER_A4_CLAIM_PATH)
    if any(
        (
            accepted_file != PEER_A3_ACCEPTED_FILE,
            accepted.get("receipt_sha256") != PEER_A3_ACCEPTED_SELF,
            accepted.get("accepted") is not True,
            accepted.get("credited") is not True,
            accepted.get("selection_rank") != 29,
            accepted.get("attempt") != 3,
            accepted.get("cell_id") != PEER_A3_CELL,
            accepted.get("execution_id") != PEER_A3_EXECUTION,
            accepted.get("run_id") != PEER_A3_RUN,
            claim_file != PEER_A4_CLAIM_FILE,
            claim.get("receipt_sha256") != PEER_A4_CLAIM_SELF,
            claim.get("selection_rank") != 29,
            claim.get("attempt") != 4,
            claim.get("cell_id") != PEER_A4_CELL,
            claim.get("execution_id") != PEER_A4_EXECUTION,
            claim.get("run_id") != PEER_A4_RUN,
            claim.get("job_uid") != PEER_JOB_UID,
            claim.get("pod_uid") != PEER_POD_UID,
        )
    ):
        raise RuntimeError("rank-29 a3-to-a4 evidence drifted")
    if PEER_STREAM.is_symlink() or not PEER_STREAM.is_file():
        raise RuntimeError("rank-29 attempt-4 stream is absent or unsafe")
    metadata = PEER_STREAM.stat()
    stream_mtime = int(metadata.st_mtime)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size <= 0
        or datetime.now(UTC).timestamp() - stream_mtime > PEER_STREAM_MAX_AGE_SECONDS
    ):
        raise RuntimeError("rank-29 attempt-4 stream is stale")
    return {
        "job_name": PEER_JOB,
        "job_uid": PEER_JOB_UID,
        "job_active": 1,
        "pod_name": PEER_POD,
        "pod_uid": PEER_POD_UID,
        "pod_phase": "Running",
        "pod_ready": True,
        "pod_restarts": 0,
        "accepted_attempt": 3,
        "accepted_receipt_path": str(PEER_A3_ACCEPTED_PATH),
        "accepted_receipt_sha256": PEER_A3_ACCEPTED_SELF,
        "accepted_file_sha256": PEER_A3_ACCEPTED_FILE,
        "active_attempt": 4,
        "active_claim_path": str(PEER_A4_CLAIM_PATH),
        "active_claim_sha256": PEER_A4_CLAIM_SELF,
        "active_claim_file_sha256": PEER_A4_CLAIM_FILE,
        "active_run_id": PEER_A4_RUN,
        "stream_path": str(PEER_STREAM),
        "stream_bytes": metadata.st_size,
        "stream_mtime_epoch": stream_mtime,
        "endpoint_slot_held": 1,
    }


def validate_release(
    receipt: dict[str, Any], plan: dict[str, Any], source_package_sha256: str
) -> None:
    peer = receipt.get("live_rank29_peer") or {}
    lease = receipt.get("endpoint_lease_observer") or {}
    collision = receipt.get("fresh_collision_reconciliation") or {}
    zero_fields = {
        "canonical_claim_collisions",
        "authoritative_session_collisions",
        "accepted_evidence_collisions",
        "output_root_collisions",
        "new_job_collisions",
        "new_configmap_collisions",
        "api_mutations",
    }
    expected_keys = {
        "schema_version",
        "status",
        "checked_at_utc",
        "launch_authorized",
        "scoring_authorized",
        "controller_cap",
        "controller",
        "source_package_sha256",
        "ledger_authority",
        "selection_authority",
        "live_rank29_peer",
        "endpoint_lease_observer",
        "fresh_collision_reconciliation",
        "privacy",
        "receipt_sha256",
    }
    expected_peer = {
        "job_name": PEER_JOB,
        "job_uid": PEER_JOB_UID,
        "job_active": 1,
        "pod_name": PEER_POD,
        "pod_uid": PEER_POD_UID,
        "pod_phase": "Running",
        "pod_ready": True,
        "pod_restarts": 0,
        "accepted_attempt": 3,
        "accepted_receipt_path": str(PEER_A3_ACCEPTED_PATH),
        "accepted_receipt_sha256": PEER_A3_ACCEPTED_SELF,
        "accepted_file_sha256": PEER_A3_ACCEPTED_FILE,
        "active_attempt": 4,
        "active_claim_path": str(PEER_A4_CLAIM_PATH),
        "active_claim_sha256": PEER_A4_CLAIM_SELF,
        "active_claim_file_sha256": PEER_A4_CLAIM_FILE,
        "active_run_id": PEER_A4_RUN,
        "stream_path": str(PEER_STREAM),
        "stream_bytes": 1,
        "stream_mtime_epoch": 1,
        "endpoint_slot_held": 1,
    }
    if any(
        (
            set(receipt) != expected_keys,
            receipt.get("schema_version") != RELEASE_SCHEMA,
            receipt.get("status") != "CLEAR",
            receipt.get("launch_authorized") is not True,
            receipt.get("scoring_authorized") is not True,
            receipt.get("controller_cap") != 1,
            receipt.get("controller") != release_projection(plan),
            receipt.get("source_package_sha256") != source_package_sha256,
            SHA256_RE.fullmatch(str(source_package_sha256)) is None,
            receipt.get("ledger_authority") != prior.whole.LEDGER_AUTHORITY,
            receipt.get("selection_authority") != prior.whole.SELECTION_AUTHORITY,
            set(peer) != set(expected_peer),
            any(
                peer.get(key) != value
                for key, value in expected_peer.items()
                if key not in {"stream_bytes", "stream_mtime_epoch"}
            ),
            type(peer.get("stream_bytes")) is not int,
            peer.get("stream_bytes", 0) <= 0,
            type(peer.get("stream_mtime_epoch")) is not int,
            set(lease)
            != {
                "lease_root",
                "endpoint_key",
                "maximum_streams",
                "held_slots",
                "held_slot_numbers",
                "available_slots",
                "available_slot_bindings",
                "probe_released",
            },
            lease.get("lease_root") != str(LEASE_ROOT),
            lease.get("endpoint_key") != LEASE_ENDPOINT_KEY,
            lease.get("maximum_streams") != 2,
            lease.get("held_slots") != 1,
            lease.get("held_slot_numbers") != [1],
            lease.get("available_slots") != 1,
            lease.get("probe_released") is not True,
            not prior._valid_available_binding(lease.get("available_slot_bindings")),  # noqa: SLF001
            set(collision)
            != {
                "checked_immediately_before_create",
                "observer_job_uid",
                "observer_pod_uid",
                "observed_cells",
                "claim_files_examined",
                "accepted_files_examined",
                "session_rows_examined",
                *zero_fields,
            },
            collision.get("checked_immediately_before_create") is not True,
            prior.engine.UUID_RE.fullmatch(str(collision.get("observer_job_uid")))
            is None,
            prior.engine.UUID_RE.fullmatch(str(collision.get("observer_pod_uid")))
            is None,
            collision.get("observed_cells") != 4,
            any(
                type(collision.get(key)) is not int or collision.get(key, -1) < 0
                for key in (
                    "claim_files_examined",
                    "accepted_files_examined",
                    "session_rows_examined",
                )
            ),
            any(collision.get(key) != 0 for key in zero_fields),
            receipt.get("privacy")
            != {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            receipt.get("receipt_sha256")
            != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("rank-30 current-peer release drifted")
    try:
        checked_at = datetime.fromisoformat(
            str(receipt["checked_at_utc"]).replace("Z", "+00:00")
        )
        now = datetime.now(UTC)
        age = (now - checked_at).total_seconds()
        stream_age = now.timestamp() - peer["stream_mtime_epoch"]
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("rank-30 current-peer release timestamp drifted") from None
    if age < -60 or age > RELEASE_MAX_AGE_SECONDS:
        raise RuntimeError("rank-30 current-peer release is stale")
    if stream_age < -60 or stream_age > PEER_STREAM_MAX_AGE_SECONDS:
        raise RuntimeError("rank-29 attempt-4 stream is stale")
