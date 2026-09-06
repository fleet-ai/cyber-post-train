"""Fresh score-blind rank-29 release gate with Path-normalized global roots."""

from __future__ import annotations

import fcntl
import os
import stat
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_release_v1 as ledger
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_runtime_v1 as runtime
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

JOB_NAME = "chris-glm53-exact100-hosted-r029-a3a4-release-v2"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "RELEASE.json"
JOBS_ROOT = Path("/mnt/sfs/jobs")
MAX_SAFE_RECEIPT_BYTES = 262_144
SOURCE_JOB = "chris-glm53-exact100-hosted-s2-c2-bulk-v1"
PEER_JOB = "chris-glm53-exact100-hosted-r003-a2a4-successor-v2"
PEER_JOB_UID = runtime.PEER_JOB_UID
PEER_POD_NAME = "chris-glm53-exact100-hosted-r003-a2a4-successor-v2-9g56k"
PEER_POD_UID = "0da05643-b6a5-4cf0-8498-d715e2cb3422"
PEER_ACCEPTED_ROOT = Path(
    "/mnt/sfs/jobs/chris-glm53-exact100-hosted-r003-a2a4-successor-v2/accepted"
)
PEER_ACCEPTED = (
    {
        "attempt": 2,
        "path": PEER_ACCEPTED_ROOT / "chris-glm53-ac-bulk-b-r003-a2-g1-33d37078.json",
        "receipt_sha256": "sha256:b9d644bbbe844c9840a789880893629becc0953ea607fc92a98f886beeaa9d55",
        "file_sha256": "sha256:81e3eea0ece47e0ea30aefdeb0e27bd8e150096f3f7dff6c77de723c6b456baa",
        "cell_id": "sha256:57f0ec78f93760d980f31df4dac3fc6daf38f2849ee2949d943e3a41f4a61e2d",
        "execution_id": "sha256:6e93f24d70b8d76f936db9a910492a2479bca86c3a12cbc8c2a5592f16335cea",
        "run_id": "chris-glm53-ac-bulk-b-r003-a2-g1-33d37078",
    },
    {
        "attempt": 3,
        "path": PEER_ACCEPTED_ROOT / "chris-glm53-ac-bulk-b-r003-a3-g1-33d37078.json",
        "receipt_sha256": "sha256:399aac0f15220d9f4f6ccfab857a02eeddabda496dad2f628cf072df8080f692",
        "file_sha256": "sha256:be0542174dd14a28cd1d5191c2d97a73d5bd457335b4f105506b032c9fac1a3c",
        "cell_id": "sha256:85d8da4373e2dbd30100aca88b08b884f3ef65febf71f77060ae67246df37c8b",
        "execution_id": "sha256:53eb697586931da3dd195636c3395ae58af60fcae1d383d5eb5f3aee960357d7",
        "run_id": "chris-glm53-ac-bulk-b-r003-a3-g1-33d37078",
    },
    {
        "attempt": 4,
        "path": PEER_ACCEPTED_ROOT / "chris-glm53-ac-bulk-b-r003-a4-g1-33d37078.json",
        "receipt_sha256": "sha256:182d5ddd1f7f1632f296ab695127184d5d73d116425dc2cc0f73d226e0f02f27",
        "file_sha256": "sha256:c88c18a508c8cdf0a44d703520befb999dee4fbe0e757ff767c8bb3371d5fd86",
        "cell_id": "sha256:f5b26b9037dd0f02ad2bb22401c9d0be8b9ab4d98641fdb091cc5f1bf6cddb44",
        "execution_id": "sha256:0ca50d2ac1458b5335e17d76efb2a244b92bcdd84df0d21dac1164bb0f392ade",
        "run_id": "chris-glm53-ac-bulk-b-r003-a4-g1-33d37078",
    },
)
RANK29_A1_ACCEPTED = Path(
    "/mnt/sfs/jobs/chris-glm53-exact100-hosted-s2-c2-bulk-v1/accepted/"
    "chris-glm53-ac-bulk-b-r029-a1-g1-b51782f9.json"
)
RANK29_A1_ACCEPTED_SHA = (
    "sha256:8240130bdfb7367a4977bab918c4f793341f39e00c237eb16b8d3eca3dba95b6"
)
RANK29_A2_CLAIM = Path(
    "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/"
    "103a5fff50b4de20ed0d36c135cb36b9d99dd237f5df7ecb14007cf4cdc45f4e.json"
)
RANK29_A2_EXECUTION = (
    "sha256:103a5fff50b4de20ed0d36c135cb36b9d99dd237f5df7ecb14007cf4cdc45f4e"
)
RANK29_A2_CELL = (
    "sha256:5a82e16e0892f64f9bc2f751b7fc54bea96ed856c4782947f24ab6e2da3858eb"
)
RANK29_A2_RUN = "chris-glm53-ac-bulk-b-r029-a2-g1-b51782f9"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _kube_get(kind: str, name: str) -> tuple[int, dict[str, Any]]:
    if kind == "jobs":
        path = f"/apis/batch/v1/namespaces/{ledger.NAMESPACE}/jobs/{name}"
    else:
        path = f"/api/v1/namespaces/{ledger.NAMESPACE}/{kind}/{name}"
    return ledger._kube_get(path)  # noqa: SLF001


def _validate_predecessors() -> None:
    status, source_job = _kube_get("jobs", SOURCE_JOB)
    conditions = source_job.get("status", {}).get("conditions", [])
    deadline_failed = any(
        row.get("type") == "Failed"
        and row.get("status") == "True"
        and row.get("reason") == "DeadlineExceeded"
        for row in conditions
        if isinstance(row, dict)
    )
    if (
        status != 200
        or source_job.get("metadata", {}).get("uid") != successor.SOURCE_FAILED_JOB_UID
        or not deadline_failed
        or bool(source_job.get("status", {}).get("active"))
    ):
        raise RuntimeError("rank-29 source deadline failure identity drifted")
    source_pod_status, _ = _kube_get(
        "pods", "chris-glm53-exact100-hosted-s2-c2-bulk-v1-622nw"
    )
    if source_pod_status != 404:
        raise RuntimeError("rank-29 source Pod is not absent")

    status, peer = _kube_get("jobs", PEER_JOB)
    peer_conditions = peer.get("status", {}).get("conditions", [])
    peer_complete = any(
        row.get("type") == "Complete" and row.get("status") == "True"
        for row in peer_conditions
        if isinstance(row, dict)
    )
    peer_failed = any(
        row.get("type") == "Failed" and row.get("status") == "True"
        for row in peer_conditions
        if isinstance(row, dict)
    )
    if (
        status != 200
        or peer.get("metadata", {}).get("uid") != PEER_JOB_UID
        or bool(peer.get("status", {}).get("active"))
        or bool(peer.get("status", {}).get("failed"))
        or peer.get("status", {}).get("succeeded") != 1
        or not peer_complete
        or peer_failed
    ):
        raise RuntimeError("rank-29 successor peer is not exact and succeeded")
    pod_status, peer_pod = _kube_get("pods", PEER_POD_NAME)
    statuses = peer_pod.get("status", {}).get("containerStatuses", [])
    if (
        pod_status != 200
        or peer_pod.get("metadata", {}).get("uid") != PEER_POD_UID
        or peer_pod.get("status", {}).get("phase") != "Succeeded"
        or not statuses
        or any(
            row.get("restartCount") != 0
            or row.get("state", {}).get("terminated", {}).get("exitCode") != 0
            for row in statuses
        )
    ):
        raise RuntimeError("rank-29 successor peer Pod is not exact and succeeded")


def _validate_peer_acceptances() -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for expected in PEER_ACCEPTED:
        path = expected["path"]
        receipt = _safe_receipt(path)
        file_sha256 = self_hosted.sha256(path.read_bytes())
        if (
            receipt.get("schema_version") != "fleet-exact-pass4-bulk-cell-accepted-v3"
            or receipt.get("receipt_sha256") != expected["receipt_sha256"]
            or receipt.get("receipt_sha256")
            != self_hosted.digest_without(receipt, "receipt_sha256")
            or file_sha256 != expected["file_sha256"]
            or receipt.get("selection_rank") != 3
            or receipt.get("attempt") != expected["attempt"]
            or receipt.get("cell_id") != expected["cell_id"]
            or receipt.get("execution_id") != expected["execution_id"]
            or receipt.get("run_id") != expected["run_id"]
            or receipt.get("accepted") is not True
            or receipt.get("credited") is not True
            or receipt.get("retry_allowed") is not False
        ):
            raise RuntimeError("rank-3 peer acceptance evidence drifted")
        evidence.append(
            {
                "attempt": expected["attempt"],
                "path": str(path),
                "receipt_sha256": expected["receipt_sha256"],
                "file_sha256": file_sha256,
                "cell_id": expected["cell_id"],
                "execution_id": expected["execution_id"],
                "run_id": expected["run_id"],
            }
        )
    return evidence


def _probe_endpoint_capacity_free() -> list[dict[str, Any]]:
    endpoint_root = Path(successor.LEASE_ROOT) / successor.LEASE_ENDPOINT_KEY
    if endpoint_root.is_symlink() or not endpoint_root.is_dir():
        raise RuntimeError("rank-29 endpoint lease root is unsafe")
    handles: list[Any] = []
    bindings: list[dict[str, Any]] = []
    try:
        for slot in range(1, 3):
            path = endpoint_root / f"slot-{slot}.lock"
            if path.is_symlink() or not path.is_file():
                raise RuntimeError("rank-29 endpoint lease slot is unsafe")
            flags = os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            handle = os.fdopen(os.open(path, flags), "a+b")
            try:
                metadata = os.fstat(handle.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 0:
                    raise RuntimeError("rank-29 endpoint lease slot drifted")
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                handle.close()
                raise
            handles.append(handle)
            bindings.append(
                {
                    "slot": slot,
                    "path": str(path),
                    "device": metadata.st_dev,
                    "inode": metadata.st_ino,
                    "size": metadata.st_size,
                }
            )
    except Exception as exc:
        raise RuntimeError("rank-29 endpoint lease capacity is not fully free") from exc
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    return bindings


def _validate_rank29_history() -> None:
    accepted = successor.load(RANK29_A1_ACCEPTED)
    claim = successor.load(RANK29_A2_CLAIM)
    if (
        accepted.get("receipt_sha256") != RANK29_A1_ACCEPTED_SHA
        or accepted.get("receipt_sha256")
        != self_hosted.digest_without(accepted, "receipt_sha256")
        or accepted.get("selection_rank") != 29
        or accepted.get("attempt") != 1
        or accepted.get("accepted") is not True
        or accepted.get("credited") is not True
        or claim.get("receipt_sha256") != successor.BLOCKED_A2_CLAIM_SHA
        or claim.get("receipt_sha256")
        != self_hosted.digest_without(claim, "receipt_sha256")
        or claim.get("cell_id") != RANK29_A2_CELL
        or claim.get("execution_id") != RANK29_A2_EXECUTION
        or claim.get("run_id") != RANK29_A2_RUN
        or claim.get("selection_rank") != 29
        or claim.get("attempt") != 2
        or claim.get("job_uid") != successor.SOURCE_FAILED_JOB_UID
        or claim.get("pod_uid") != successor.SOURCE_FAILED_POD_UID
        or claim.get("automatic_retry") is not False
    ):
        raise RuntimeError("rank-29 accepted/blocked history drifted")


def _session_collides(
    row: dict[str, Any], config: dict[str, Any], item: dict[str, Any]
) -> bool:
    metadata = row.get("metadata") or {}
    return any(
        metadata.get(field) == expected
        for field, expected in {
            "run_id": config["run_id"],
            "cell_id": item["cell_id"],
            "execution_id": item["execution_id"],
        }.items()
    )


def _reject_protected_keys(value: Any) -> None:
    forbidden = {
        "flag",
        "flags",
        "messages",
        "prompt",
        "prompts",
        "score",
        "scores",
        "solution",
        "solutions",
        "trace",
        "traces",
        "transcript",
        "transcripts",
    }
    if isinstance(value, dict):
        if forbidden.intersection(value):
            raise RuntimeError("protected key in duplicate evidence")
        for item in value.values():
            _reject_protected_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_protected_keys(item)


def _identity_values(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"cell_id", "execution_id", "run_id"} and isinstance(item, str):
                found.add(item)
            else:
                found.update(_identity_values(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_identity_values(item))
    return found


def _safe_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("unsafe duplicate evidence path")
    if path.stat().st_size > MAX_SAFE_RECEIPT_BYTES:
        raise RuntimeError("duplicate evidence exceeds size limit")
    value = successor.load(path)
    _reject_protected_keys(value)
    return value


def _global_evidence(
    plan: dict[str, Any],
    *,
    jobs_root: Path | str = JOBS_ROOT,
    claim_root: Path | str = successor.CLAIM_ROOT,
) -> dict[str, int]:
    jobs_root = Path(jobs_root)
    claim_root = Path(claim_root)
    if (
        jobs_root.is_symlink()
        or not jobs_root.is_dir()
        or claim_root.is_symlink()
        or not claim_root.is_dir()
    ):
        raise RuntimeError("rank-29 global evidence roots are unsafe")
    current = plan["attempts"]
    retired = plan["partition"]["source_generation_1_retired_unclaimed"]
    cells = {row["cell_id"] for row in current}
    fresh_executions = {row["execution_id"] for row in current}
    retired_executions = {row["execution_id"] for row in retired}
    run_ids = {row["run_id"] for row in current} | {row["run_id"] for row in retired}
    target_identities = cells | fresh_executions | retired_executions | run_ids
    result = {
        "global_claim_files_examined": 0,
        "global_accepted_files_examined": 0,
        "fresh_generation_claim_collisions": 0,
        "retired_generation_claim_collisions": 0,
        "global_cell_claim_collisions": 0,
        "global_accepted_evidence_collisions": 0,
        "global_output_evidence_collisions": 0,
    }
    for path in sorted(claim_root.glob("*.json")):
        result["global_claim_files_examined"] += 1
        identities = _identity_values(_safe_receipt(path))
        path_execution = "sha256:" + path.stem
        result["fresh_generation_claim_collisions"] += int(
            path_execution in fresh_executions
            or bool(identities.intersection(fresh_executions))
        )
        result["retired_generation_claim_collisions"] += int(
            path_execution in retired_executions
            or bool(identities.intersection(retired_executions))
        )
        result["global_cell_claim_collisions"] += int(bool(identities.intersection(cells)))
    for path in sorted(jobs_root.glob("*/accepted/*.json")):
        result["global_accepted_files_examined"] += 1
        identities = _identity_values(_safe_receipt(path))
        result["global_accepted_evidence_collisions"] += int(
            path.stem in run_ids or bool(identities.intersection(target_identities))
        )
    for run_id in run_ids:
        result["global_output_evidence_collisions"] += sum(
            int(path.exists() or path.is_symlink())
            for path in jobs_root.glob(f"*/attempts/{run_id}")
        )
    return result


def build(root: Path) -> dict[str, Any]:
    key = os.environ.get("FLEET_API_KEY", "")
    job_uid, pod_uid = os.environ.get("JOB_UID", ""), os.environ.get("POD_UID", "")
    scored_bindings = {
        "scored_source_sha256": os.environ.get("SCORED_SOURCE_SHA256", ""),
        "scored_package_template_sha256": os.environ.get(
            "SCORED_PACKAGE_TEMPLATE_SHA256", ""
        ),
    }
    if (
        not key
        or any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid))
        or any(successor.SHA256_RE.fullmatch(value) is None for value in scored_bindings.values())
    ):
        raise RuntimeError("rank-29 release requires key and nonzero UIDs")
    _validate_predecessors()
    peer_acceptances = _validate_peer_acceptances()
    endpoint_lease_bindings = _probe_endpoint_capacity_free()
    _validate_rank29_history()
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    global_evidence = _global_evidence(plan)
    output_collisions = int(successor.SFS_ROOT.exists() or successor.SFS_ROOT.is_symlink())
    kubernetes_collisions = sum(
        _kube_get(kind, name)[0] == 200
        for kind, name in (
            ("jobs", successor.JOB_NAME),
            ("configmaps", successor.CONFIGMAP_NAME),
        )
    )
    engine._fresh_route_check(plan, key)  # noqa: SLF001
    task = engine._task_for_item(plan, plan["attempts"][0])  # noqa: SLF001
    with engine._client(key) as client:  # noqa: SLF001
        sessions = self_hosted._task_sessions(client, task["task"]["key"])

    blocked_config = {"run_id": RANK29_A2_RUN}
    blocked_item = {"cell_id": RANK29_A2_CELL, "execution_id": RANK29_A2_EXECUTION}
    blocked_session_collisions = sum(
        _session_collides(row, blocked_config, blocked_item) for row in sessions
    )
    session_collisions = 0
    for item in plan["attempts"]:
        config = engine._attempt_config(plan, task, item)  # noqa: SLF001
        session_collisions += sum(_session_collides(row, config, item) for row in sessions)
    if any(
        (
            blocked_session_collisions,
            output_collisions,
            kubernetes_collisions,
            session_collisions,
            *(global_evidence[key] for key in global_evidence if key.endswith("collisions")),
        )
    ):
        raise RuntimeError("rank-29 successor duplicate ledger is not clear")
    body = {
        "schema_version": runtime.RELEASE_SCHEMA,
        "status": "CLEAR",
        "successor_job": successor.JOB_NAME,
        "successor_configmap": successor.CONFIGMAP_NAME,
        "plan_sha256": plan["plan_sha256"],
        "planned_cells": len(plan["attempts"]),
        "source_failed_job_uid": successor.SOURCE_FAILED_JOB_UID,
        "source_failed_pod_uid": successor.SOURCE_FAILED_POD_UID,
        "blocked_a2_claim_sha256": successor.BLOCKED_A2_CLAIM_SHA,
        "blocked_a2_session_collisions": blocked_session_collisions,
        "peer_job_uid": PEER_JOB_UID,
        "peer_pod_uid": PEER_POD_UID,
        "peer_active": False,
        "peer_succeeded": True,
        "peer_pod_phase": "Succeeded",
        "peer_pod_restarts": 0,
        "peer_accepted_receipts": peer_acceptances,
        "endpoint_lease_root": str(successor.LEASE_ROOT),
        "endpoint_lease_key": successor.LEASE_ENDPOINT_KEY,
        "endpoint_lease_slots_available": len(endpoint_lease_bindings),
        "endpoint_lease_slot_bindings": endpoint_lease_bindings,
        "endpoint_lease_probe_released": True,
        "fleet_session_collisions": session_collisions,
        "global_claim_collisions": (
            global_evidence["fresh_generation_claim_collisions"]
            + global_evidence["retired_generation_claim_collisions"]
            + global_evidence["global_cell_claim_collisions"]
        ),
        **global_evidence,
        "kubernetes_object_collisions": kubernetes_collisions,
        "sfs_output_collisions": output_collisions,
        "serving_load_block": successor.SERVING_LOAD_BLOCK,
        "maximum_scored_streams": 2,
        "job_active_deadline_seconds": runtime.ACTIVE_DEADLINE_SECONDS,
        "preclaim_guard_seconds": runtime.CLAIM_GUARD_SECONDS,
        "checked_immediately_before_create": True,
        "mutation_calls": 0,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "observed_at_utc": _now(),
        "scores_read": False,
        "prompts_traces_flags_read": False,
        **scored_bindings,
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def main() -> int:
    receipt = build(Path.cwd())
    OUTPUT_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(OUTPUT_PATH, receipt)  # noqa: SLF001
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
