"""Fresh score-free phased diagnostic for the failed rank-30 v3 observer."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_single_slot_release_v3 as release
from evals.fleet import hosted_glm_rank30_single_slot_v3 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank30-release-phase-diagnostic-v4"
JOB_NAME = "chris-glm53-exact100-hosted-r030-single-slot-release-diagnostic-v4"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
FAILED_RELEASE_JOB = release.JOB_NAME
FAILED_RELEASE_JOB_UID = "9c3235a5-7919-4eae-a099-2886adad2fad"
FAILED_RELEASE_POD = release.JOB_NAME + "-2qzjb"
FAILED_RELEASE_POD_UID = "c1b44935-4eb6-40fc-88c0-d0ee7e4e4928"
FAILED_RELEASE_PACKAGE_SHA256 = (
    "sha256:24179796e2c34f379680a4c41dff8da973d5895fe2563f808aa059f5210ae352"
)
ERROR_CLASSES = {
    AttributeError: "attribute-error",
    json.JSONDecodeError: "json-decode-error",
    KeyError: "key-error",
    TypeError: "type-error",
    UnicodeDecodeError: "unicode-decode-error",
    ValueError: "value-error",
    RuntimeError: "runtime-error",
    OSError: "filesystem-error",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_materialized_sources(root: Path, state: dict[str, Any]) -> None:
    required = {
        "evals/fleet/hosted_glm_rank30_single_slot_release_v3.py": "release",
        "evals/fleet/hosted_glm_rank30_single_slot_v3.py": "successor",
        "evals/fleet/hosted_glm_rank30_single_slot_release_v1.py": "release-v1",
        "evals/fleet/hosted_glm_rank30_single_slot_v1.py": "successor-v1",
    }
    digests: dict[str, str] = {}
    for relative, category in required.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("materialized release source is absent or unsafe")
        digests[category] = self_hosted.sha256(path.read_bytes())
    state["source_files_validated"] = len(digests)
    state["source_digest_sha256"] = self_hosted.sha256(
        self_hosted.canonical_json(digests)
    )


def _validate_failed_v3(_root: Path, state: dict[str, Any]) -> None:
    status, job = release.prior.kube._kube_get(  # noqa: SLF001
        "jobs", FAILED_RELEASE_JOB
    )
    pod_status, pod = release.prior.kube._kube_get(  # noqa: SLF001
        "pods", FAILED_RELEASE_POD
    )
    conditions = job.get("status", {}).get("conditions", [])
    containers = pod.get("status", {}).get("containerStatuses", [])
    failed = any(
        isinstance(row, dict)
        and row.get("type") == "Failed"
        and row.get("status") == "True"
        and row.get("reason") == "BackoffLimitExceeded"
        for row in conditions
    )
    if any(
        (
            status != 200,
            job.get("metadata", {}).get("uid") != FAILED_RELEASE_JOB_UID,
            job.get("status", {}).get("failed") != 1,
            not failed,
            pod_status != 200,
            pod.get("metadata", {}).get("uid") != FAILED_RELEASE_POD_UID,
            pod.get("status", {}).get("phase") != "Failed",
            not containers,
            any(
                row.get("restartCount") != 0
                or row.get("state", {}).get("terminated", {}).get("exitCode") != 1
                for row in containers
            ),
        )
    ):
        raise RuntimeError("failed v3 observer terminal identity drifted")
    failed_root = Path("/mnt/sfs/jobs") / FAILED_RELEASE_JOB
    if failed_root.exists() or failed_root.is_symlink():
        raise RuntimeError("failed v3 observer unexpectedly has SFS output")
    state["failed_observer_validated"] = True


def _validate_observer_bindings(_root: Path, state: dict[str, Any]) -> None:
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    if any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("v4 diagnostic observer UIDs are invalid")
    state["observer_bindings_validated"] = True


def _load_inventory(_root: Path, state: dict[str, Any]) -> None:
    state["inventory"] = successor.load(source_runtime.INVENTORY_PATH)
    state["inventory_loaded"] = True


def _build_plan(root: Path, state: dict[str, Any]) -> None:
    plan = successor.build_runtime_plan(
        successor.CONTROLLER, state["inventory"], root
    )
    state["plan"] = plan
    state["plan_sha256"] = plan["plan_sha256"]


def _validate_current_peer(_root: Path, state: dict[str, Any]) -> None:
    peer = release._validate_live_peer()  # noqa: SLF001
    state["peer"] = {
        "job_uid_validated": peer.get("job_uid") == successor.PEER_JOB_UID,
        "pod_uid_validated": peer.get("pod_uid") == successor.PEER_POD_UID,
        "accepted_attempt": peer.get("accepted_attempt"),
        "active_attempt": peer.get("active_attempt"),
        "stream_bytes": peer.get("stream_bytes"),
        "stream_mtime_epoch": peer.get("stream_mtime_epoch"),
    }


def _probe_endpoint_slot(_root: Path, state: dict[str, Any]) -> None:
    lease = release.prior._probe_slots()  # noqa: SLF001
    state["lease"] = {
        "held_slots": lease.get("held_slots"),
        "held_slot_numbers": lease.get("held_slot_numbers"),
        "available_slots": lease.get("available_slots"),
        "probe_released": lease.get("probe_released"),
    }


def _local_collision_scan(_root: Path, state: dict[str, Any]) -> None:
    plan = state["plan"]
    identities = {
        value
        for item in plan["attempts"]
        for value in (item["cell_id"], item["execution_id"], item["run_id"])
    }
    counts = {
        "claim_files_examined": 0,
        "accepted_files_examined": 0,
        "output_run_ids_examined": 0,
        "claim_collisions": 0,
        "accepted_collisions": 0,
        "output_collisions": 0,
        "target_object_collisions": 0,
    }
    for path in sorted(Path(successor.CLAIM_ROOT).glob("*.json")):
        counts["claim_files_examined"] += 1
        value = release.prior.kube._safe_receipt(path)  # noqa: SLF001
        path_execution = "sha256:" + path.stem
        counts["claim_collisions"] += int(
            path_execution in identities
            or bool(
                release.prior.kube._identity_values(value).intersection(identities)  # noqa: SLF001
            )
        )
    for path in sorted(Path(successor.JOBS_ROOT).glob("*/accepted/*.json")):
        counts["accepted_files_examined"] += 1
        value = release.prior.kube._safe_receipt(path)  # noqa: SLF001
        counts["accepted_collisions"] += int(
            path.stem in identities
            or bool(
                release.prior.kube._identity_values(value).intersection(identities)  # noqa: SLF001
            )
        )
    for item in plan["attempts"]:
        counts["output_run_ids_examined"] += 1
        counts["output_collisions"] += sum(
            int(path.exists() or path.is_symlink())
            for path in Path(successor.JOBS_ROOT).glob(
                f"*/attempts/{item['run_id']}"
            )
        )
    for kind, name in (
        ("jobs", successor.CONTROLLERS[successor.CONTROLLER]["job_name"]),
        ("configmaps", successor.CONTROLLERS[successor.CONTROLLER]["configmap_name"]),
    ):
        status, _value = release.prior.kube._kube_get(kind, name)  # noqa: SLF001
        if status not in {200, 404}:
            raise RuntimeError("target object absence could not be proven")
        counts["target_object_collisions"] += int(status == 200)
    if any(counts[key] for key in counts if key.endswith("collisions")):
        raise RuntimeError("local rank-30 collision evidence is not clear")
    state["local_collision_counts"] = counts


def _validate_projection(_root: Path, state: dict[str, Any]) -> None:
    projection = successor.release_projection(state["plan"])
    if projection.get("selection_rank") != 30 or len(projection["attempts"]) != 4:
        raise RuntimeError("rank-30 release projection drifted")
    state["projection_validated"] = True


Phase = tuple[str, Callable[[Path, dict[str, Any]], None]]
PHASES: tuple[Phase, ...] = (
    ("01-materialized-sources", _validate_materialized_sources),
    ("02-failed-v3-terminal", _validate_failed_v3),
    ("03-observer-bindings", _validate_observer_bindings),
    ("04-inventory-load", _load_inventory),
    ("05-runtime-plan", _build_plan),
    ("06-current-a4-peer", _validate_current_peer),
    ("07-endpoint-slot", _probe_endpoint_slot),
    ("08-local-collisions", _local_collision_scan),
    ("09-release-projection", _validate_projection),
)


def _classification(error: Exception | None) -> str | None:
    if error is None:
        return None
    for error_type, value in ERROR_CLASSES.items():
        if isinstance(error, error_type):
            return value
    return "unclassified-exception"


def _receipt(
    *, completed: list[str], failed_phase: str | None, error: Exception | None,
    state: dict[str, Any]
) -> dict[str, Any]:
    body = {
        "schema_version": SCHEMA,
        "status": "FAILED" if error is not None else "PASSED_TO_SESSION_BOUNDARY",
        "diagnostic_job": JOB_NAME,
        "diagnostic_configmap": CONFIGMAP_NAME,
        "observer_job_uid": os.environ.get("JOB_UID"),
        "observer_pod_uid": os.environ.get("POD_UID"),
        "failed_release_job_uid": FAILED_RELEASE_JOB_UID,
        "failed_release_pod_uid": FAILED_RELEASE_POD_UID,
        "failed_release_package_sha256": FAILED_RELEASE_PACKAGE_SHA256,
        "phase_order": [name for name, _function in PHASES],
        "completed_phases": completed,
        "last_completed_phase": completed[-1] if completed else None,
        "failed_phase": failed_phase,
        "error_type_class": _classification(error),
        "error_sha256": (
            self_hosted.sha256(
                self_hosted.canonical_json(
                    {
                        "phase": failed_phase,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
            )
            if error is not None else None
        ),
        "safe_counts": {
            "source_files_validated": state.get("source_files_validated", 0),
            **state.get("local_collision_counts", {}),
        },
        "peer_state": state.get("peer", {}),
        "lease_state": state.get("lease", {}),
        "plan_sha256": state.get("plan_sha256"),
        "session_boundary_executed": False,
        "model_calls": 0,
        "task_calls": 0,
        "session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "api_mutation_calls": 0,
        "diagnostic_receipt_writes": 1,
        "scores_read": False,
        "prompts_traces_flags_read": False,
        "observed_at_utc": _now(),
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def run(
    root: Path, *, output_path: Path = OUTPUT_PATH,
    phases: Sequence[Phase] = PHASES,
) -> int:
    if [name for name, _function in phases] != [name for name, _function in PHASES]:
        raise RuntimeError("diagnostic phase order drifted")
    completed: list[str] = []
    failed_phase: str | None = None
    error: Exception | None = None
    state: dict[str, Any] = {}
    for name, function in phases:
        try:
            function(root, state)
        except Exception as exc:  # terminal receipt is the purpose of this job
            failed_phase = name
            error = exc
            break
        completed.append(name)
    receipt = _receipt(
        completed=completed, failed_phase=failed_phase, error=error, state=state
    )
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(output_path, receipt)  # noqa: SLF001
    return int(error is not None)


def main() -> int:
    return run(Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
