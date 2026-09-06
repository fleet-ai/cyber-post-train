"""Rank-30-only projection of the reviewed atomic whole-task successor."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import hosted_glm_whole_task_successor_v1 as whole
from evals.fleet import self_hosted

CONTROLLER = "glm-hosted-r30-whole-task"
CONTROLLERS = {CONTROLLER: whole.CONTROLLERS[CONTROLLER]}
RELEASE_SCHEMA = "fleet-hosted-glm-rank30-single-slot-release-v1"
RELEASE_PATH = Path("/bootstrap/release.json")
RELEASE_MAX_AGE_SECONDS = 600
PEER_JOB = "chris-glm53-exact100-hosted-r029-a3a4-successor-v1"
PEER_JOB_UID = "74cdb483-65fb-4331-9e3e-567a54b53315"
PEER_POD = "chris-glm53-exact100-hosted-r029-a3a4-successor-v1-9mnx9"
PEER_POD_UID = "dc891f43-917f-4cdb-9769-1639f6bfcaab"
PEER_RUN_ID = "chris-glm53-ac-bulk-b-r029-a3-g2-b51782f9"
PEER_STREAM = (
    whole.JOBS_ROOT
    / PEER_JOB
    / "attempts"
    / PEER_RUN_ID
    / "agent-output/opencode-stream.jsonl"
)
PEER_STREAM_MAX_AGE_SECONDS = 300

SHA256_RE = whole.SHA256_RE
COMMIT_RE = whole.COMMIT_RE
CANARY_GATE_SCHEMA = whole.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = whole.RECONCILIATION_GATE_SCHEMA
validate_inventory_gate = whole.validate_inventory_gate
load = whole.load
AtomicWholeTaskClaims = whole.AtomicWholeTaskClaims
CLAIM_ROOT = whole.CLAIM_ROOT
JOBS_ROOT = whole.JOBS_ROOT
LEASE_ROOT = whole.LEASE_ROOT
LEASE_ENDPOINT_KEY = whole.LEASE_ENDPOINT_KEY


def build_plan(controller: str, root: Path) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("rank-30 single-slot controller drifted")
    return whole.build_plan(controller, root)


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("rank-30 single-slot controller drifted")
    return whole.build_runtime_plan(controller, inventory_receipt, root)


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = build_plan(CONTROLLER, root)
    if (
        plan["partition"]["whole_task_rank"] != 30
        or [row["attempt"] for row in plan["attempts"]] != [1, 2, 3, 4]
        or any(row["selection_rank"] != 30 for row in plan["attempts"])
    ):
        raise ValueError("rank-30 single-slot partition drifted")
    return {CONTROLLER: plan}


def release_projection(plan: dict[str, Any]) -> dict[str, Any]:
    return whole.release_projection({CONTROLLER: plan})[0]


def validate_release(
    release: dict[str, Any], plan: dict[str, Any], source_package_sha256: str
) -> None:
    collision = release.get("fresh_collision_reconciliation") or {}
    peer = release.get("live_rank29_peer") or {}
    lease = release.get("endpoint_lease_observer") or {}
    expected_keys = {
        "schema_version", "status", "checked_at_utc", "launch_authorized",
        "scoring_authorized", "controller_cap", "controller",
        "source_package_sha256", "ledger_authority", "selection_authority",
        "live_rank29_peer", "endpoint_lease_observer",
        "fresh_collision_reconciliation", "privacy", "receipt_sha256",
    }
    zero_collision_fields = {
        "canonical_claim_collisions", "authoritative_session_collisions",
        "accepted_evidence_collisions", "output_root_collisions",
        "new_job_collisions", "new_configmap_collisions", "api_mutations",
    }
    if any(
        (
            set(release) != expected_keys,
            release.get("schema_version") != RELEASE_SCHEMA,
            release.get("status") != "CLEAR",
            release.get("launch_authorized") is not True,
            release.get("scoring_authorized") is not True,
            release.get("controller_cap") != 1,
            release.get("controller") != release_projection(plan),
            release.get("source_package_sha256") != source_package_sha256,
            SHA256_RE.fullmatch(str(source_package_sha256)) is None,
            release.get("ledger_authority") != whole.LEDGER_AUTHORITY,
            release.get("selection_authority") != whole.SELECTION_AUTHORITY,
            set(peer)
            != {
                "job_name", "job_uid", "job_active", "pod_name", "pod_uid",
                "pod_phase", "pod_ready", "pod_restarts", "endpoint_slot_held",
                "run_id", "stream_path", "stream_bytes", "stream_mtime_epoch",
            },
            any(
                peer.get(key) != value
                for key, value in {
                    "job_name": PEER_JOB,
                    "job_uid": PEER_JOB_UID,
                    "job_active": 1,
                    "pod_name": PEER_POD,
                    "pod_uid": PEER_POD_UID,
                    "pod_phase": "Running",
                    "pod_ready": True,
                    "pod_restarts": 0,
                    "endpoint_slot_held": True,
                    "run_id": PEER_RUN_ID,
                    "stream_path": str(PEER_STREAM),
                }.items()
            ),
            type(peer.get("stream_bytes")) is not int,
            peer.get("stream_bytes", 0) <= 0,
            type(peer.get("stream_mtime_epoch")) is not int,
            set(lease)
            != {
                "lease_root", "endpoint_key", "maximum_streams",
                "held_slots", "held_slot_numbers", "available_slots",
                "available_slot_bindings", "probe_released",
            },
            lease.get("lease_root") != str(LEASE_ROOT),
            lease.get("endpoint_key") != LEASE_ENDPOINT_KEY,
            lease.get("maximum_streams") != 2,
            lease.get("held_slots") != 1,
            lease.get("held_slot_numbers") != [1],
            lease.get("available_slots") != 1,
            lease.get("probe_released") is not True,
            not _valid_available_binding(lease.get("available_slot_bindings")),
            set(collision)
            != {
                "checked_immediately_before_create", "observer_job_uid",
                "observer_pod_uid", "observed_cells", "claim_files_examined",
                "accepted_files_examined", "session_rows_examined",
                *zero_collision_fields,
            },
            collision.get("checked_immediately_before_create") is not True,
            engine.UUID_RE.fullmatch(str(collision.get("observer_job_uid"))) is None,
            engine.UUID_RE.fullmatch(str(collision.get("observer_pod_uid"))) is None,
            collision.get("observed_cells") != 4,
            any(
                type(collision.get(key)) is not int or collision.get(key, -1) < 0
                for key in (
                    "claim_files_examined", "accepted_files_examined",
                    "session_rows_examined",
                )
            ),
            any(collision.get(key) != 0 for key in zero_collision_fields),
            release.get("privacy")
            != {
                "scores_read": False,
                "prompts_traces_flags_read": False,
                "credentials_included": False,
            },
            release.get("receipt_sha256")
            != self_hosted.digest_without(release, "receipt_sha256"),
        )
    ):
        raise RuntimeError("rank-30 single-slot release drifted")
    try:
        checked_at = datetime.fromisoformat(
            str(release["checked_at_utc"]).replace("Z", "+00:00")
        )
        age = (datetime.now(UTC) - checked_at).total_seconds()
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("rank-30 single-slot release timestamp drifted") from None
    if age < -60 or age > RELEASE_MAX_AGE_SECONDS:
        raise RuntimeError("rank-30 single-slot release is stale")
    stream_age = datetime.now(UTC).timestamp() - peer["stream_mtime_epoch"]
    if stream_age < -60 or stream_age > PEER_STREAM_MAX_AGE_SECONDS:
        raise RuntimeError("rank-29 live peer stream is stale")


def _valid_available_binding(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], dict)
        and set(value[0]) == {"slot", "path", "device", "inode", "size"}
        and value[0]["slot"] == 2
        and value[0]["path"]
        == str(LEASE_ROOT / LEASE_ENDPOINT_KEY / f"slot-{value[0]['slot']}.lock")
        and type(value[0]["device"]) is int
        and value[0]["device"] > 0
        and type(value[0]["inode"]) is int
        and value[0]["inode"] > 0
        and value[0]["size"] == 0
    )
