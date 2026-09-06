"""One-shot score-free phase diagnostic for the failed rank-29 release observer."""

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
from evals.fleet import hosted_glm_rank29_a3a4_c2_release_v1 as release
from evals.fleet import hosted_glm_rank29_a3a4_c2_successor_v1 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank29-release-phase-diagnostic-v1"
SOURCE_COMMIT = "193f660ec80420cd26f49757e67169a13ed63d12"
JOB_NAME = "chris-glm53-exact100-hosted-r029-a3a4-release-diagnostic-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
FAILED_RELEASE_JOB_UID = "d64db41b-a5a5-467c-8211-87a4235cec89"
FAILED_RELEASE_POD_NAME = "chris-glm53-exact100-hosted-r029-a3a4-release-v1-4kbw5"
FAILED_RELEASE_POD_UID = "0b71de2d-fbfa-4eee-8d0a-8b226cf98881"
PLAN_SHA256 = "sha256:8ca09b5e050c040c254ba407d6055dd7f9d0c5f7cdea4c1e0c715a05ab8acd8c"
SOURCE_SHA256S = {
    "evals/fleet/hosted_glm_rank29_a3a4_c2_release_v1.py": (
        "sha256:55d90c700fde74706c69a80700eabee7ebbe1d24620159dffb313934f2eb09d5"
    ),
    "evals/fleet/hosted_glm_rank29_a3a4_c2_runtime_v1.py": (
        "sha256:a944c8f829f203d6f12f2815f75dee89e62161f81c71fa22f9c42db41361d458"
    ),
    "evals/fleet/hosted_glm_rank29_a3a4_c2_successor_v1.py": (
        "sha256:7b6e5d7360280db7f1341144ad8612d7f00f5c345e728744636421ebb28f0333"
    ),
    "evals/fleet/hosted_glm_rank29_a3a4_c2_package_v1.py": (
        "sha256:ee6b62143036b90e400ee669a47b7dfcf7a03f35ef96a23616116199e1123fc6"
    ),
}
ERROR_CLASSIFICATIONS = {
    json.JSONDecodeError: "invalid-json",
    FileNotFoundError: "missing-file",
    PermissionError: "permission-denied",
    BlockingIOError: "lease-unavailable",
    ValueError: "validation-value-error",
    RuntimeError: "gate-runtime-error",
    OSError: "filesystem-error",
}


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _validate_source_bindings(root: Path, state: dict[str, Any]) -> None:
    observed = {
        relative: self_hosted.sha256((root / relative).read_bytes())
        for relative in SOURCE_SHA256S
    }
    if observed != SOURCE_SHA256S:
        raise RuntimeError("merged release source binding drifted")
    state["source_sha256s"] = observed


def _validate_failed_release_terminal(_root: Path, state: dict[str, Any]) -> None:
    status, job = release._kube_get("jobs", release.JOB_NAME)  # noqa: SLF001
    conditions = job.get("status", {}).get("conditions", [])
    failed = any(
        row.get("type") == "Failed"
        and row.get("status") == "True"
        and row.get("reason") == "BackoffLimitExceeded"
        for row in conditions
        if isinstance(row, dict)
    )
    if (
        status != 200
        or job.get("metadata", {}).get("uid") != FAILED_RELEASE_JOB_UID
        or bool(job.get("status", {}).get("active"))
        or bool(job.get("status", {}).get("succeeded"))
        or job.get("status", {}).get("failed") != 1
        or not failed
    ):
        raise RuntimeError("failed release Job terminal identity drifted")
    pod_status, pod = release._kube_get("pods", FAILED_RELEASE_POD_NAME)  # noqa: SLF001
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if (
        pod_status != 200
        or pod.get("metadata", {}).get("uid") != FAILED_RELEASE_POD_UID
        or pod.get("status", {}).get("phase") != "Failed"
        or not statuses
        or any(
            row.get("restartCount") != 0
            or row.get("state", {}).get("terminated", {}).get("exitCode") != 1
            for row in statuses
        )
    ):
        raise RuntimeError("failed release Pod terminal identity drifted")
    failed_root = Path("/mnt/sfs/jobs") / release.JOB_NAME
    if failed_root.exists() or failed_root.is_symlink():
        raise RuntimeError("failed release unexpectedly has SFS output")
    state["failed_release_terminal"] = {
        "job_uid": FAILED_RELEASE_JOB_UID,
        "pod_uid": FAILED_RELEASE_POD_UID,
        "pod_restarts": 0,
        "sfs_output_absent": True,
    }


def _validate_predecessors(_root: Path, state: dict[str, Any]) -> None:
    release._validate_predecessors()  # noqa: SLF001
    state["predecessors_valid"] = True


def _validate_peer_acceptances(_root: Path, state: dict[str, Any]) -> None:
    state["peer_accepted_receipts"] = release._validate_peer_acceptances()  # noqa: SLF001


def _validate_endpoint_capacity(_root: Path, state: dict[str, Any]) -> None:
    bindings = release._probe_endpoint_capacity_free()  # noqa: SLF001
    if len(bindings) != 2:
        raise RuntimeError("endpoint lease capacity proof is incomplete")
    state["endpoint_lease_slot_bindings"] = bindings
    state["endpoint_lease_probe_released"] = True


def _validate_rank29_history(_root: Path, state: dict[str, Any]) -> None:
    release._validate_rank29_history()  # noqa: SLF001
    state["rank29_history_valid"] = True


def _build_plan(root: Path, state: dict[str, Any]) -> None:
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    if plan.get("plan_sha256") != PLAN_SHA256:
        raise RuntimeError("rank-29 successor plan binding drifted")
    state["plan"] = plan


def _validate_global_evidence(_root: Path, state: dict[str, Any]) -> None:
    evidence = release._global_evidence(state["plan"])  # noqa: SLF001
    if any(value for key, value in evidence.items() if key.endswith("collisions")):
        raise RuntimeError("rank-29 global collision evidence is not clear")
    state["global_evidence"] = evidence


def _validate_scored_collisions(_root: Path, state: dict[str, Any]) -> None:
    kubernetes_collisions = sum(
        release._kube_get(kind, name)[0] == 200  # noqa: SLF001
        for kind, name in (
            ("jobs", successor.JOB_NAME),
            ("configmaps", successor.CONFIGMAP_NAME),
        )
    )
    sfs_collision = int(successor.SFS_ROOT.exists() or successor.SFS_ROOT.is_symlink())
    if kubernetes_collisions or sfs_collision:
        raise RuntimeError("rank-29 scored successor collision is present")
    state["kubernetes_object_collisions"] = 0
    state["sfs_output_collisions"] = 0


Phase = tuple[str, Callable[[Path, dict[str, Any]], None]]
PHASES: tuple[Phase, ...] = (
    ("01-source-bindings", _validate_source_bindings),
    ("02-failed-release-terminal", _validate_failed_release_terminal),
    ("03-predecessors-terminal", _validate_predecessors),
    ("04-peer-acceptances", _validate_peer_acceptances),
    ("05-endpoint-lease-capacity", _validate_endpoint_capacity),
    ("06-rank29-history", _validate_rank29_history),
    ("07-plan", _build_plan),
    ("08-global-evidence", _validate_global_evidence),
    ("09-scored-collisions", _validate_scored_collisions),
)


def _classify_error(error: Exception) -> str:
    for error_type, classification in ERROR_CLASSIFICATIONS.items():
        if isinstance(error, error_type):
            return classification
    return "unclassified-exception"


def _receipt(
    *,
    completed: list[str],
    failed_phase: str | None,
    error: Exception | None,
    state: dict[str, Any],
) -> dict[str, Any]:
    job_uid = os.environ.get("JOB_UID", "")
    pod_uid = os.environ.get("POD_UID", "")
    if any(uuid.UUID(value).int == 0 for value in (job_uid, pod_uid)):
        raise RuntimeError("diagnostic observer UIDs are absent")
    classification = _classify_error(error) if error is not None else None
    error_sha256 = (
        self_hosted.sha256(
            self_hosted.canonical_json(
                {
                    "phase": failed_phase,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
        )
        if error is not None
        else None
    )
    body = {
        "schema_version": SCHEMA,
        "status": "FAILED" if error is not None else "PASSED_TO_NETWORK_BOUNDARY",
        "source_commit": SOURCE_COMMIT,
        "diagnostic_job": JOB_NAME,
        "diagnostic_configmap": CONFIGMAP_NAME,
        "observer_job_uid": job_uid,
        "observer_pod_uid": pod_uid,
        "failed_release_job_uid": FAILED_RELEASE_JOB_UID,
        "failed_release_pod_uid": FAILED_RELEASE_POD_UID,
        "phase_order": [name for name, _function in PHASES],
        "completed_phases": completed,
        "last_completed_phase": completed[-1] if completed else None,
        "failed_phase": failed_phase,
        "error_classification": classification,
        "error_sha256": error_sha256,
        "source_sha256s": state.get("source_sha256s", {}),
        "peer_accepted_receipts": state.get("peer_accepted_receipts", []),
        "endpoint_lease_slot_bindings": state.get("endpoint_lease_slot_bindings", []),
        "endpoint_lease_probe_released": state.get(
            "endpoint_lease_probe_released", False
        ),
        "global_evidence": state.get("global_evidence", {}),
        "kubernetes_object_collisions": state.get("kubernetes_object_collisions"),
        "sfs_output_collisions": state.get("sfs_output_collisions"),
        "network_bound_phases_executed": False,
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
    root: Path,
    *,
    output_path: Path = OUTPUT_PATH,
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
        except Exception as exc:  # terminal evidence must survive a phase failure
            failed_phase = name
            error = exc
            break
        completed.append(name)
    receipt = _receipt(
        completed=completed,
        failed_phase=failed_phase,
        error=error,
        state=state,
    )
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    engine._write_once(output_path, receipt)  # noqa: SLF001
    return int(error is not None)


def main() -> int:
    return run(Path.cwd())


if __name__ == "__main__":
    raise SystemExit(main())
