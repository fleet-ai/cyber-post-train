"""Score-blind phase observer for the failed rank-30 preclaim boundary."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_single_slot_v5 as successor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank30-preclaim-phase-observer-v1"
JOB_NAME = "chris-glm53-r030-preclaim-phase-observer-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
OUTPUT_PATH = Path("/mnt/sfs/jobs") / JOB_NAME / "DIAGNOSTIC.json"
MANIFEST_PATH = Path("/bootstrap/frozen-source-manifest.json")
SOURCE_PACKAGE_SHA256 = (
    "sha256:f94237a58819eba51def73e3f9645ad75f3abf5b4de9b2df96a9292f4994c052"
)
RUNTIME_PLAN_SHA256 = (
    "sha256:e0bef037fe7853d3e4d8a0762841ceb6467bdde846a386d6dc5b15d5aecb42c6"
)
RELEASE_PATH = Path(
    "/mnt/sfs/jobs/chris-glm53-r030-single-slot-release-v4/RELEASE.json"
)
RELEASE_FILE_SHA256 = (
    "sha256:3429623bef47d7b790c5c1a9d7694f61b411e1df697da34f2c2d398e67a2d4c0"
)
RELEASE_RECEIPT_SHA256 = (
    "sha256:ba2d5185afb7d122e41953cc640bab9360d18edb7cdbd0feca86635dcee14020"
)
V6_DIAGNOSTIC_FILE_SHA256 = successor.DIAGNOSTIC_FILE_SHA256
V6_DIAGNOSTIC_RECEIPT_SHA256 = successor.DIAGNOSTIC_SELF_SHA256
FAILED_JOB = successor.CONTROLLERS[successor.CONTROLLER]["job_name"]
FAILED_JOB_UID = "dcb44603-ef9b-4563-ac70-9d785555c11e"
FAILED_POD = FAILED_JOB + "-9bwlq"
FAILED_POD_UID = "1f59963a-d282-454e-a576-e3fabd1e517b"
FAILED_STARTED_AT = datetime(2026, 9, 6, 15, 34, 6, tzinfo=UTC)
TARGET_SFS_ROOT = Path("/mnt/sfs/jobs/chris-glm53-exact100-hosted-r030-whole-task-g1-v1")
NAMESPACE = "fleet-train-jobs"
SERVICE_ACCOUNT_ROOT = Path("/var/run/secrets/kubernetes.io/serviceaccount")


def _file_sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _input_digests(state: dict[str, Any]) -> dict[str, str]:
    return {
        "source_package_sha256": SOURCE_PACKAGE_SHA256,
        "source_manifest_sha256": state.get("source_manifest_sha256", ""),
        "runtime_plan_sha256": state.get("runtime_plan_sha256", RUNTIME_PLAN_SHA256),
        "release_file_sha256": RELEASE_FILE_SHA256,
        "release_receipt_sha256": RELEASE_RECEIPT_SHA256,
        "v6_diagnostic_file_sha256": V6_DIAGNOSTIC_FILE_SHA256,
        "v6_diagnostic_receipt_sha256": V6_DIAGNOSTIC_RECEIPT_SHA256,
    }


def _validate_sources(root: Path, state: dict[str, Any]) -> None:
    raw = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (
        set(manifest) != {"schema_version", "source_package_sha256", "files"}
        or manifest.get("schema_version")
        != "fleet-hosted-glm-rank30-preclaim-source-manifest-v1"
        or manifest.get("source_package_sha256") != SOURCE_PACKAGE_SHA256
        or not isinstance(files, dict)
        or not files
    ):
        raise RuntimeError("rank-30 frozen source manifest drifted")
    source_data: dict[str, str] = {}
    for key, binding in files.items():
        if not isinstance(binding, dict) or set(binding) != {"relative_path", "sha256"}:
            raise RuntimeError("rank-30 frozen source binding drifted")
        projected = Path("/bootstrap") / ("source__" + key)
        installed = root / str(binding["relative_path"])
        if any(
            path.is_symlink() or not path.is_file() for path in (projected, installed)
        ):
            raise RuntimeError("rank-30 frozen source file is unsafe")
        payload = projected.read_bytes()
        if _file_sha256(payload) != binding["sha256"] or installed.read_bytes() != payload:
            raise RuntimeError("rank-30 frozen source digest drifted")
        source_data[key] = payload.decode()
    if self_hosted.sha256(self_hosted.canonical_json(source_data)) != SOURCE_PACKAGE_SHA256:
        raise RuntimeError("rank-30 frozen package digest drifted")
    state["source_manifest_sha256"] = _file_sha256(raw)


def _kube_get(kind: str, name: str) -> dict[str, Any]:
    token = (SERVICE_ACCOUNT_ROOT / "token").read_text().strip()
    ca = str(SERVICE_ACCOUNT_ROOT / "ca.crt")
    route = "apis/batch/v1" if kind == "jobs" else "api/v1"
    response = httpx.get(
        f"https://kubernetes.default.svc/{route}/namespaces/{NAMESPACE}/{kind}/{name}",
        headers={"Authorization": "Bearer " + token},
        verify=ca,
        timeout=20,
    )
    if response.status_code != 200:
        raise RuntimeError("rank-30 failed controller identity is unavailable")
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("rank-30 failed controller identity is invalid")
    return value


def _validate_failed_controller(_root: Path, _state: dict[str, Any]) -> None:
    job = _kube_get("jobs", FAILED_JOB)
    pod = _kube_get("pods", FAILED_POD)
    failed = any(
        row.get("type") == "Failed"
        and row.get("status") == "True"
        and row.get("reason") == "BackoffLimitExceeded"
        for row in job.get("status", {}).get("conditions") or []
        if isinstance(row, dict)
    )
    statuses = pod.get("status", {}).get("containerStatuses") or []
    evaluator = [row for row in statuses if row.get("name") == "evaluator"]
    if any(
        (
            job.get("metadata", {}).get("uid") != FAILED_JOB_UID,
            job.get("status", {}).get("failed") != 1,
            not failed,
            pod.get("metadata", {}).get("uid") != FAILED_POD_UID,
            pod.get("status", {}).get("phase") != "Failed",
            len(evaluator) != 1,
            evaluator[0].get("restartCount") != 0 if evaluator else True,
            evaluator[0].get("state", {}).get("terminated", {}).get("exitCode") != 1
            if evaluator
            else True,
            TARGET_SFS_ROOT.exists() or TARGET_SFS_ROOT.is_symlink(),
        )
    ):
        raise RuntimeError("rank-30 failed controller terminal evidence drifted")


def _validate_observer_identity(_root: Path, _state: dict[str, Any]) -> None:
    for name in ("JOB_UID", "POD_UID"):
        try:
            if uuid.UUID(os.environ.get(name, "")).int == 0:
                raise ValueError
        except ValueError:
            raise RuntimeError("rank-30 observer identity is invalid") from None


def _load_inventory(_root: Path, state: dict[str, Any]) -> None:
    state["inventory"] = successor.load(source_runtime.INVENTORY_PATH)


def _build_runtime_plan(root: Path, state: dict[str, Any]) -> None:
    plan = successor.build_runtime_plan(successor.CONTROLLER, state["inventory"], root)
    if plan.get("plan_sha256") != RUNTIME_PLAN_SHA256:
        raise RuntimeError("rank-30 runtime plan digest drifted")
    state["plan"] = plan
    state["runtime_plan_sha256"] = plan["plan_sha256"]


@contextmanager
def _failed_execution_clock() -> Iterator[None]:
    rank30_v3 = successor.prior.prior
    rank30_v1 = rank30_v3.prior
    original_v3 = rank30_v3.datetime
    original_v1 = rank30_v1.datetime

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return FAILED_STARTED_AT if tz is None else FAILED_STARTED_AT.astimezone(tz)

    try:
        rank30_v3.datetime = FrozenDateTime
        rank30_v1.datetime = FrozenDateTime
        yield
    finally:
        rank30_v3.datetime = original_v3
        rank30_v1.datetime = original_v1


def _validate_frozen_release(_root: Path, state: dict[str, Any]) -> None:
    if RELEASE_PATH.is_symlink() or not RELEASE_PATH.is_file():
        raise RuntimeError("rank-30 frozen release is absent or unsafe")
    raw = RELEASE_PATH.read_bytes()
    value = json.loads(raw)
    if (
        _file_sha256(raw) != RELEASE_FILE_SHA256
        or not isinstance(value, dict)
        or value.get("receipt_sha256") != RELEASE_RECEIPT_SHA256
        or value.get("receipt_sha256")
        != self_hosted.digest_without(value, "receipt_sha256")
    ):
        raise RuntimeError("rank-30 frozen release digest drifted")
    with _failed_execution_clock():
        successor.validate_release(value, state["plan"], SOURCE_PACKAGE_SHA256)


def _validate_strict_current_peer(_root: Path, _state: dict[str, Any]) -> None:
    successor.validate_current_peer()


Phase = tuple[str, Callable[[Path, dict[str, Any]], None]]
PHASES: tuple[Phase, ...] = (
    ("01-frozen-source-package", _validate_sources),
    ("02-failed-controller-zero-effect", _validate_failed_controller),
    ("03-observer-identity", _validate_observer_identity),
    ("04-inventory-load", _load_inventory),
    ("05-runtime-plan", _build_runtime_plan),
    ("06-frozen-release-replay", _validate_frozen_release),
    ("07-strict-current-peer", _validate_strict_current_peer),
)
_CANONICAL_PHASES = PHASES
FAILURE_CODES = {
    "01-frozen-source-package": "FROZEN_SOURCE_INVARIANT_FAILED",
    "02-failed-controller-zero-effect": "FAILED_CONTROLLER_INVARIANT_FAILED",
    "03-observer-identity": "OBSERVER_IDENTITY_INVARIANT_FAILED",
    "04-inventory-load": "INVENTORY_INVARIANT_FAILED",
    "05-runtime-plan": "RUNTIME_PLAN_INVARIANT_FAILED",
    "06-frozen-release-replay": "FROZEN_RELEASE_INVARIANT_FAILED",
    "07-strict-current-peer": "CURRENT_PEER_INVARIANT_FAILED",
}


def _validate_phase_bindings() -> None:
    if PHASES is not _CANONICAL_PHASES or len(PHASES) != len(_CANONICAL_PHASES):
        raise RuntimeError("rank-30 observer phase binding drifted")
    for actual, expected in zip(PHASES, _CANONICAL_PHASES, strict=True):
        if actual[0] != expected[0] or actual[1] is not expected[1]:
            raise RuntimeError("rank-30 observer phase binding drifted")


def _receipt(
    completed: list[str],
    failed_phase: str | None,
    failed: bool,
    state: dict[str, Any],
) -> dict[str, Any]:
    if failed != (failed_phase is not None):
        raise RuntimeError("rank-30 observer failure classification is invalid")
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "FAILED" if failed else "PASSED_TO_PROVIDER_BOUNDARY",
        "observer_identity": {
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "job_uid": os.environ.get("JOB_UID"),
            "pod_uid": os.environ.get("POD_UID"),
        },
        "failed_controller_identity": {
            "job_name": FAILED_JOB,
            "job_uid": FAILED_JOB_UID,
            "pod_name": FAILED_POD,
            "pod_uid": FAILED_POD_UID,
        },
        "phase_order": [name for name, _function in PHASES],
        "completed_phases": completed,
        "last_completed_phase": completed[-1] if completed else None,
        "failed_phase": failed_phase,
        "failure_code": FAILURE_CODES[failed_phase] if failed_phase else None,
        "input_digests": _input_digests(state),
        "zero_call_counters": {
            "model_calls": 0,
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutation_calls": 0,
        },
        "privacy": {
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "protected_content_included": False,
        },
        "provider_constructed": False,
        "session_model_boundary_crossed": False,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body


def _write_receipt_once(output_path: Path, receipt: dict[str, Any]) -> None:
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(self_hosted.canonical_json(receipt) + b"\n")


def run(root: Path, *, output_path: Path = OUTPUT_PATH) -> int:
    _validate_phase_bindings()
    completed: list[str] = []
    failed_phase: str | None = None
    failed = False
    state: dict[str, Any] = {}
    for name, function in PHASES:
        try:
            function(root, state)
        except Exception:  # exact errors are deliberately discarded before receipt
            failed_phase, failed = name, True
            break
        completed.append(name)
    receipt = _receipt(completed, failed_phase, failed, state)
    _write_receipt_once(output_path, receipt)
    return int(failed)


def main() -> int:
    return run(Path(os.environ.get("REPO_ROOT", "/workspace/cyber-post-train")))


if __name__ == "__main__":
    raise SystemExit(main())
