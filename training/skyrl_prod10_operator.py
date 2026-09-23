"""Bounded in-cluster stage/preflight coordinator for the prod10 RL canary.

This is intentionally not a general controller.  One sealed packet selects one
of two fixed phases and all Kubernetes names, SFS paths, manifests, and source
identities are re-derived by the existing prod9 rail.  The process runs the
existing observer in a same-process thread, performs one exact create, waits for
UID-bound release, and exits.  It never launches a GPU workload.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from cyber_post_train.jobs import JobsError, digest

from . import dev_cleanup_observer as cleanup
from . import skyrl_prod9_direct as direct
from . import skyrl_prod9_hardening as hardening
from . import skyrl_prod9_training as training
from . import skyrl_reward_rayjob as historical
from .incluster_kubernetes import InClusterKubernetesRunner

PACKET_SCHEMA = "cyber_skyrl_prod10_operator_packet_v1"
RESULT_SCHEMA = "cyber_skyrl_prod10_operator_result_v1"
TERMINATION_SCHEMA = "cyber_skyrl_prod10_operator_termination_v1"
FAILURE_TERMINATION_SCHEMA = "cyber_skyrl_prod10_operator_failure_v1"
OPERATOR_NAMES = {
    "stage": "chris-q38-prod10-stage-operator-v5",
    "preflight": "chris-q38-prod10-preflight-operator-v1",
}
_STAGE_V4_RECOVERY = {
    "schema": "cyber_skyrl_prod10_stage_precreate_recovery_v1",
    "status": "authorized_precreate_recovery",
    "previous_operator_name": "chris-q38-prod10-stage-operator-v4",
    "previous_operator_job_uid": "a9a2c37d-2003-442a-aea8-d589ba9df2d8",
    "previous_packet_sha256": (
        "sha256:7e354a7a441fdbc9f8faec162fc5e5cc3ccdf2a83b47cbf68c10055b9e327e52"
    ),
    "previous_failure_receipt_sha256": (
        "sha256:d7b6e0ccebd728848544f01630b708ae57b89f306a33468046d1ee7ee8f85f02"
    ),
    "previous_release_sha256": (
        "sha256:3a1a9c8c89ed28310afcec312f7cea3db6d117f16405d9720b65a6356a34eacb"
    ),
    "previous_error_code": "operator_unclassified",
    "gpus": 0,
}
_TERMINATION_PATH = Path("/dev/termination-log")
_POLL_SECONDS = 0.25


class OperatorFailure(ValueError):
    """Sanitized fixed-code refusal suitable for a termination receipt."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema or value != _seal(value):
        raise ValueError("prod10 operator evidence is invalid")
    return value


def stage_recovery_binding() -> dict[str, Any]:
    """Return the one reviewed pre-create recovery binding for stage v5."""
    return _seal(_STAGE_V4_RECOVERY)


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _write_termination(*, phase: str, result_path: Path, result: dict[str, Any]) -> None:
    value = _seal(
        {
            "schema": TERMINATION_SCHEMA,
            "status": "passed",
            "phase": phase,
            "result_path": str(result_path),
            "result_sha256": result["sha256"],
            "gpus": 0,
        }
    )
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > 3500:
        raise ValueError("prod10 operator termination receipt is too large")
    descriptor = os.open(_TERMINATION_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_failure_termination(*, phase: str, error: BaseException) -> None:
    value = _seal(
        {
            "schema": FAILURE_TERMINATION_SCHEMA,
            "status": "failed",
            "phase": phase,
            "error_class": type(error).__name__,
            "error_code": getattr(error, "code", "operator_unclassified"),
            "gpus": 0,
        }
    )
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > 1000:
        raise ValueError("prod10 operator failure receipt is too large")
    descriptor = os.open(_TERMINATION_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _identity(value: object) -> historical.RailIdentity:
    if not isinstance(value, dict):
        raise ValueError("prod10 operator identity is invalid")
    identity = historical.identity_from_mapping(value)
    if identity.run_name != "chris-q38-rlreward-prod10":
        raise ValueError("prod10 operator identity changed")
    return identity


def _packet(value: object, phase: str) -> dict[str, Any]:
    packet = _validate_seal(value, PACKET_SCHEMA)
    if phase not in OPERATOR_NAMES or packet.get("phase") != phase:
        raise ValueError("prod10 operator phase changed")
    if packet.get("operator_name") != OPERATOR_NAMES[phase]:
        raise ValueError("prod10 operator name changed")
    _identity(packet.get("identity"))
    direct._validate_seal(packet.get("dev_preview"), direct.CPU_PREVIEW_SCHEMA)
    duplicate = direct._validate_seal(
        packet.get("dev_duplicate_proof"), direct.CPU_DUPLICATE_PROOF_SCHEMA
    )
    if duplicate.get("context") != direct.DEV_CONTEXT:
        raise ValueError("prod10 operator development proof changed")
    if phase == "stage" and packet.get("precreate_recovery") != stage_recovery_binding():
        raise ValueError("prod10 stage pre-create recovery binding changed")
    return packet


def _validate_runtime(
    packet: dict[str, Any], runner: InClusterKubernetesRunner
) -> str:
    if (os.geteuid(), os.getegid()) != (direct.RUNTIME_UID, direct.RUNTIME_GID):
        raise OperatorFailure("runtime_identity_rejected")
    name = os.environ.get("OPERATOR_JOB_NAME", "")
    pod_name = os.environ.get("OPERATOR_POD_NAME", "")
    pod_uid = os.environ.get("OPERATOR_POD_UID", "")
    packet_sha256 = os.environ.get("OPERATOR_PACKET_SHA256", "")
    source_sha256 = os.environ.get("OPERATOR_SOURCE_SHA256", "")
    if name != packet["operator_name"]:
        raise OperatorFailure("runtime_job_name_rejected")
    if packet_sha256 != packet["sha256"]:
        raise OperatorFailure("runtime_packet_digest_rejected")
    if not source_sha256.startswith("sha256:") or len(source_sha256) != 71:
        raise OperatorFailure("runtime_source_digest_rejected")
    try:
        UUID(pod_uid)
    except ValueError as exc:
        raise OperatorFailure("runtime_pod_uid_rejected") from exc
    if not pod_name.startswith(name + "-"):
        raise OperatorFailure("runtime_pod_name_rejected")

    def get(resource: str, target: str, code: str) -> dict[str, Any]:
        result = direct._kubectl(
            runner, direct.PROD_CONTEXT, "get", resource, target, "--output=json"
        )
        if result.returncode:
            raise OperatorFailure(code + "_read_failed")
        try:
            value = json.loads(result.stdout)
        except ValueError as exc:
            raise OperatorFailure(code + "_read_invalid") from exc
        if not isinstance(value, dict):
            raise OperatorFailure(code + "_read_invalid")
        return value

    pod = get("pod", pod_name, "runtime_pod")
    pod_metadata = pod.get("metadata", {})
    if (
        not isinstance(pod_metadata, dict)
        or pod_metadata.get("name") != pod_name
        or pod_metadata.get("uid") != pod_uid
    ):
        raise OperatorFailure("runtime_pod_binding_rejected")
    owners = pod_metadata.get("ownerReferences", [])
    owner_matches = [
        owner
        for owner in owners
        if isinstance(owner, dict)
        and owner.get("apiVersion") == "batch/v1"
        and owner.get("kind") == "Job"
        and owner.get("name") == name
        and owner.get("controller") is True
    ]
    if len(owner_matches) != 1:
        raise OperatorFailure("runtime_job_owner_rejected")
    job_uid = owner_matches[0].get("uid")
    try:
        UUID(job_uid)
    except (TypeError, ValueError) as exc:
        raise OperatorFailure("runtime_job_uid_rejected") from exc
    job = get("job", name, "runtime_job")
    metadata = job.get("metadata", {})
    annotations = metadata.get("annotations", {}) if isinstance(metadata, dict) else {}
    labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
    if (
        metadata.get("name") != name
        or metadata.get("uid") != job_uid
        or annotations.get("fleet.ai/failure-alerts") != "off"
        or annotations.get("cyber-post-train.fleet.ai/operator-packet-sha256")
        != packet_sha256
        or annotations.get("cyber-post-train.fleet.ai/operator-source-sha256")
        != source_sha256
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or job.get("spec", {}).get("template", {}).get("spec", {}).get(
            "priorityClassName"
        )
        != "c1"
    ):
        raise OperatorFailure("runtime_job_binding_rejected")
    return job_uid


def _canonical_directory(path: Path, *, owner: bool = True, code: str) -> None:
    try:
        identity = path.lstat()
    except OSError as exc:
        raise OperatorFailure(code + "_stat_failed") from exc
    mode = stat.S_IMODE(identity.st_mode)
    if (
        path.is_symlink()
        or not stat.S_ISDIR(identity.st_mode)
        or path.resolve() != path
        or (
            owner
            and (identity.st_uid, identity.st_gid)
            != (direct.RUNTIME_UID, direct.RUNTIME_GID)
        )
        or not mode & stat.S_IRUSR
        or not mode & stat.S_IWUSR
        or not mode & stat.S_IXUSR
        or not os.access(path, os.R_OK | os.W_OK | os.X_OK)
    ):
        raise OperatorFailure(code + "_rejected")


def _create_root_for_stage() -> None:
    parent = hardening.CREATE_ONCE_ROOT.parent
    _canonical_directory(parent, owner=False, code="sfs_control_parent")
    if hardening.CREATE_ONCE_ROOT.exists() or hardening.CREATE_ONCE_ROOT.is_symlink():
        raise OperatorFailure("sfs_create_once_root_exists")
    try:
        hardening.CREATE_ONCE_ROOT.mkdir(mode=0o700)
    except OSError as exc:
        raise OperatorFailure("sfs_create_once_root_mkdir_failed") from exc
    _canonical_directory(
        hardening.CREATE_ONCE_ROOT, code="sfs_create_once_root_postcondition"
    )
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_root() -> None:
    _canonical_directory(hardening.CREATE_ONCE_ROOT, code="sfs_create_once_root")
    if not direct.live_create_is_available():
        raise ValueError("prod10 operator live create root is unavailable")


def _create_operation_root(path: Path) -> None:
    if path.parent != hardening.CREATE_ONCE_ROOT:
        raise ValueError("prod10 operator operation root changed")
    if path.exists() or path.is_symlink():
        raise ValueError("prod10 operator operation root already exists")
    path.mkdir(mode=0o700)
    _canonical_directory(path, code="sfs_operation_root_postcondition")


def _read_recovery_file(path: Path, schema: str) -> dict[str, Any]:
    try:
        identity = path.lstat()
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise OperatorFailure("stage_v4_recovery_evidence_unreadable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(identity.st_mode)
        or (identity.st_uid, identity.st_gid)
        != (direct.RUNTIME_UID, direct.RUNTIME_GID)
        or stat.S_IMODE(identity.st_mode) != 0o600
    ):
        raise OperatorFailure("stage_v4_recovery_evidence_rejected")
    try:
        return _validate_seal(value, schema)
    except ValueError as exc:
        raise OperatorFailure("stage_v4_recovery_evidence_rejected") from exc


def _job_absent(
    runner: InClusterKubernetesRunner, name: str, *, code: str
) -> None:
    result = direct._kubectl(
        runner,
        direct.PROD_CONTEXT,
        "get",
        "job",
        name,
        "--ignore-not-found",
        "--output=json",
    )
    if result.returncode or result.stdout.strip():
        raise OperatorFailure(code)


def _reconcile_stage_v4_precreate(
    packet: dict[str, Any],
    *,
    stage: dict[str, Any],
    expected: dict[str, Any],
    identity: historical.RailIdentity,
    runner: InClusterKubernetesRunner,
) -> Path:
    """Preserve and replace only v4 evidence proven to predate a create intent."""
    if packet.get("precreate_recovery") != stage_recovery_binding():
        raise OperatorFailure("stage_v4_recovery_binding_rejected")
    _existing_root()
    operation_root = hardening.stage_operation_root(stage)
    _canonical_directory(operation_root, code="stage_v4_operation_root")
    names = {entry.name for entry in operation_root.iterdir()}
    if names != {"STAGE_OPERATOR_INTENT.json", "STAGE_OBSERVER_ARMED.json"}:
        raise OperatorFailure("stage_v4_recovery_inventory_rejected")
    recovery = stage_recovery_binding()
    intent_path = operation_root / "STAGE_OPERATOR_INTENT.json"
    armed_path = operation_root / "STAGE_OBSERVER_ARMED.json"
    intent = _read_recovery_file(
        intent_path, "cyber_skyrl_prod10_operator_intent_v1"
    )
    armed = _read_recovery_file(
        armed_path, "cyber_direct_cleanup_observer_armed_v1"
    )
    if (
        set(intent)
        != {
            "schema",
            "phase",
            "packet_sha256",
            "operator_job_uid",
            "created_at",
            "sha256",
        }
        or intent.get("phase") != "stage"
        or intent.get("packet_sha256") != recovery["previous_packet_sha256"]
        or intent.get("operator_job_uid") != recovery["previous_operator_job_uid"]
    ):
        raise OperatorFailure("stage_v4_recovery_intent_rejected")
    try:
        direct._timestamp(intent.get("created_at"))
    except JobsError as exc:
        raise OperatorFailure("stage_v4_recovery_intent_rejected") from exc
    expected_binding = hardening.creator_binding_path(operation_root, "stage")
    if (
        armed.get("status") != "armed"
        or armed.get("context") != direct.PROD_CONTEXT
        or armed.get("namespace") != direct.NAMESPACE
        or armed.get("kind") != "job"
        or armed.get("name") != identity.stage_name
        or armed.get("maximum_seconds") != direct.CPU_MAXIMUM_SECONDS
        or armed.get("expected_gpus") != 0
        or armed.get("plan_sha256") != stage["sha256"]
        or armed.get("manifest_sha256") != "sha256:" + digest(expected)
        or armed.get("creator_binding_path") != str(expected_binding)
        or type(armed.get("observer_pid")) is not int
        or armed["observer_pid"] < 1
    ):
        raise OperatorFailure("stage_v4_recovery_observer_rejected")
    try:
        direct._timestamp(armed.get("armed_at"))
    except JobsError as exc:
        raise OperatorFailure("stage_v4_recovery_observer_rejected") from exc
    _job_absent(
        runner,
        recovery["previous_operator_name"],
        code="stage_v4_operator_still_present",
    )
    _job_absent(runner, identity.stage_name, code="stage_target_already_present")
    archived_intent = operation_root / "STAGE_OPERATOR_INTENT.v4.failed.json"
    archived_armed = operation_root / "STAGE_OBSERVER_ARMED.v4.failed.json"
    if any(
        path.exists() or path.is_symlink()
        for path in (archived_intent, archived_armed, expected_binding)
    ):
        raise OperatorFailure("stage_v4_recovery_destination_exists")
    os.rename(intent_path, archived_intent)
    os.rename(armed_path, archived_armed)
    descriptor = os.open(operation_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _write_once(
        operation_root / "STAGE_OPERATOR_RECOVERY_V5.json",
        _seal(
            {
                "schema": "cyber_skyrl_prod10_stage_precreate_recovery_receipt_v1",
                "status": "v4_precreate_evidence_preserved",
                "binding_sha256": recovery["sha256"],
                "previous_intent_sha256": intent["sha256"],
                "previous_observer_sha256": armed["sha256"],
                "archived_files": [archived_armed.name, archived_intent.name],
                "gpus": 0,
            }
        ),
    )
    return operation_root


def _observer_thread(observer: cleanup.Observer) -> tuple[threading.Thread, dict[str, Any]]:
    state: dict[str, Any] = {}

    def target() -> None:
        try:
            state["result"] = observer.run()
        except BaseException as exc:  # propagated after exact cleanup attempt
            state["error"] = exc

    thread = threading.Thread(target=target, name="exact-uid-observer", daemon=False)
    thread.start()
    deadline = time.monotonic() + 30
    while not observer.armed_path.is_file():
        if not thread.is_alive():
            error = state.get("error")
            if isinstance(error, BaseException):
                raise error
            raise RuntimeError("prod10 observer exited before arming")
        if time.monotonic() >= deadline:
            raise RuntimeError("prod10 observer did not arm")
        time.sleep(_POLL_SECONDS)
    try:
        armed = json.loads(observer.armed_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RuntimeError("prod10 observer armed receipt is unreadable") from exc
    state["armed"] = armed
    return thread, state


def _join_observer(thread: threading.Thread, state: dict[str, Any]) -> dict[str, Any]:
    thread.join(direct.CPU_MAXIMUM_SECONDS + 360)
    if thread.is_alive():
        raise RuntimeError("prod10 exact-UID observer exceeded its bound")
    error = state.get("error")
    if isinstance(error, BaseException):
        raise error
    result = state.get("result")
    if not isinstance(result, dict) or result.get("status") != "released":
        raise RuntimeError("prod10 exact-UID observer did not accept and release the Job")
    return result


def _new_observer(
    *, operation_root: Path, purpose: str, name: str, plan_sha256: str, manifest_sha256: str,
    runner: InClusterKubernetesRunner,
) -> cleanup.Observer:
    prefix = purpose.upper()
    return cleanup.Observer(
        context=direct.PROD_CONTEXT,
        namespace=direct.NAMESPACE,
        kind="job",
        name=name,
        maximum_seconds=direct.CPU_MAXIMUM_SECONDS,
        expected_gpus=0,
        plan_sha256=plan_sha256,
        manifest_sha256=manifest_sha256,
        armed_path=operation_root / f"{prefix}_OBSERVER_ARMED.json",
        result_path=operation_root / f"{prefix}_OBSERVER_RESULT.json",
        profile="production-cpu",
        poll_seconds=2,
        run=runner,
    )


def run_stage(packet: dict[str, Any], *, runner: InClusterKubernetesRunner) -> dict[str, Any]:
    identity = _identity(packet["identity"])
    stage, _ = training._stage_identity(packet.get("stage"))
    expected = direct.stage_job_manifest(stage, identity=identity)
    if packet.get("manifest_sha256") != "sha256:" + digest(expected):
        raise ValueError("prod10 stage operator manifest changed")
    try:
        direct.validate_cpu_preview_proof(
            expected,
            packet["dev_preview"],
            purpose="stage",
            context=direct.DEV_CONTEXT,
            fresh=True,
        )
    except JobsError as exc:
        raise OperatorFailure("stage_development_preview_rejected") from exc
    operation_root = _reconcile_stage_v4_precreate(
        packet,
        stage=stage,
        expected=expected,
        identity=identity,
        runner=runner,
    )
    _write_once(
        operation_root / "STAGE_OPERATOR_INTENT.json",
        _seal(
            {
                "schema": "cyber_skyrl_prod10_operator_intent_v1",
                "phase": "stage",
                "packet_sha256": packet["sha256"],
                "operator_job_uid": os.environ["OPERATOR_JOB_UID"],
                "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ),
    )
    observer = _new_observer(
        operation_root=operation_root,
        purpose="stage",
        name=identity.stage_name,
        plan_sha256=stage["sha256"],
        manifest_sha256="sha256:" + digest(expected),
        runner=runner,
    )
    thread, state = _observer_thread(observer)
    try:
        prod_rendered = direct.server_dry_run(
            expected, context=direct.PROD_CONTEXT, runner=runner
        )
        prod_preview = direct.validate_cpu_preview(
            expected, prod_rendered, context=direct.PROD_CONTEXT, purpose="stage"
        )
    except JobsError as exc:
        raise OperatorFailure("stage_production_preview_rejected") from exc
    try:
        authorization = direct.authorize_stage(
            stage,
            expected,
            dev_preview=packet["dev_preview"],
            prod_preview=prod_preview,
            observer=state["armed"],
            identity=identity,
        )
    except JobsError as exc:
        raise OperatorFailure("stage_authorization_rejected") from exc
    try:
        created = direct.create_stage_once(
            operation_root,
            stage,
            expected,
            authorization,
            identity=identity,
            runner=runner,
            dev_duplicate_proof=packet["dev_duplicate_proof"],
        )
    except JobsError as exc:
        raise OperatorFailure("stage_create_gate_rejected") from exc
    release = _join_observer(thread, state)
    receipt = release.get("receipt")
    direct._stage_receipt(stage, receipt, identity=identity)
    return _seal(
        {
            "schema": RESULT_SCHEMA,
            "status": "stage_ready",
            "phase": "stage",
            "packet_sha256": packet["sha256"],
            "stage": stage,
            "authorization": authorization,
            "created": created,
            "receipt": receipt,
            "release": release,
            "gpus": 0,
        }
    )


def run_preflight(
    packet: dict[str, Any], *, runner: InClusterKubernetesRunner
) -> dict[str, Any]:
    identity = _identity(packet["identity"])
    plan = packet.get("plan")
    request = packet.get("request")
    stage_result = _validate_seal(packet.get("stage_result"), RESULT_SCHEMA)
    if not isinstance(plan, dict) or not isinstance(request, dict):
        raise ValueError("prod10 preflight operator plan/request is invalid")
    direct._identity(plan, identity)
    if training.job_request(plan) != request:
        raise ValueError("prod10 preflight operator request changed")
    stage, _ = training._stage_identity(stage_result.get("stage"))
    expected = direct.preflight_job_manifest(plan, identity=identity)
    if packet.get("manifest_sha256") != "sha256:" + digest(expected):
        raise ValueError("prod10 preflight operator manifest changed")
    try:
        direct.validate_cpu_preview_proof(
            expected,
            packet["dev_preview"],
            purpose="preflight",
            context=direct.DEV_CONTEXT,
            fresh=True,
        )
    except JobsError as exc:
        raise OperatorFailure("preflight_development_preview_rejected") from exc
    _existing_root()
    operation_root = hardening.training_operation_root(plan)
    _create_operation_root(operation_root)
    _write_once(
        operation_root / "PREFLIGHT_OPERATOR_INTENT.json",
        _seal(
            {
                "schema": "cyber_skyrl_prod10_operator_intent_v1",
                "phase": "preflight",
                "packet_sha256": packet["sha256"],
                "operator_job_uid": os.environ["OPERATOR_JOB_UID"],
                "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ),
    )
    observer = _new_observer(
        operation_root=operation_root,
        purpose="preflight",
        name=identity.preflight_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(expected),
        runner=runner,
    )
    thread, state = _observer_thread(observer)
    prod_rendered = direct.server_dry_run(expected, context=direct.PROD_CONTEXT, runner=runner)
    prod_preview = direct.validate_cpu_preview(
        expected, prod_rendered, context=direct.PROD_CONTEXT, purpose="preflight"
    )
    authorization = direct.authorize_preflight(
        plan,
        request,
        stage,
        stage_result["authorization"],
        stage_result["created"],
        stage_result["receipt"],
        stage_result["release"],
        expected,
        dev_preview=packet["dev_preview"],
        prod_preview=prod_preview,
        observer=state["armed"],
        identity=identity,
    )
    created = direct.create_preflight_once(
        operation_root,
        plan,
        request,
        stage,
        authorization,
        identity=identity,
        runner=runner,
        dev_duplicate_proof=packet["dev_duplicate_proof"],
    )
    release = _join_observer(thread, state)
    receipt = release.get("receipt")
    direct._preflight_receipt(plan, request, receipt, identity=identity)
    return _seal(
        {
            "schema": RESULT_SCHEMA,
            "status": "preflight_ready",
            "phase": "preflight",
            "packet_sha256": packet["sha256"],
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "stage_result_sha256": stage_result["sha256"],
            "authorization": authorization,
            "created": created,
            "receipt": receipt,
            "release": release,
            "gpus": 0,
        }
    )


def run(packet_path: Path, phase: str) -> dict[str, Any]:
    try:
        value = json.loads(packet_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ValueError("prod10 operator packet is unreadable") from exc
    packet = _packet(value, phase)
    runner = InClusterKubernetesRunner()
    os.environ["OPERATOR_JOB_UID"] = _validate_runtime(packet, runner)
    result = run_stage(packet, runner=runner) if phase == "stage" else run_preflight(
        packet, runner=runner
    )
    root = (
        hardening.stage_operation_root(result["stage"])
        if phase == "stage"
        else hardening.training_operation_root(packet["plan"])
    )
    result_path = root / ("STAGE_OPERATOR_RESULT.json" if phase == "stage" else "PREFLIGHT_OPERATOR_RESULT.json")
    _write_once(result_path, result)
    _write_termination(phase=phase, result_path=result_path, result=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(OPERATOR_NAMES), required=True)
    args = parser.parse_args()
    try:
        value = run(args.packet, args.phase)
    except BaseException as exc:
        try:
            _write_failure_termination(phase=args.phase, error=exc)
        except BaseException:
            pass
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({"status": value["status"], "sha256": value["sha256"]}))


if __name__ == "__main__":
    main()
