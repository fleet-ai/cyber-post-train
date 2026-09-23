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
OPERATOR_NAMES = {
    "stage": "chris-q38-prod10-stage-operator-v2",
    "preflight": "chris-q38-prod10-preflight-operator-v1",
}
_TERMINATION_PATH = Path("/dev/termination-log")
_POLL_SECONDS = 0.25


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != schema or value != _seal(value):
        raise ValueError("prod10 operator evidence is invalid")
    return value


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
    return packet


def _validate_runtime(packet: dict[str, Any]) -> str:
    if (os.geteuid(), os.getegid()) != (direct.RUNTIME_UID, direct.RUNTIME_GID):
        raise ValueError("prod10 operator must run as image user 1000:100")
    name = os.environ.get("OPERATOR_JOB_NAME", "")
    uid = os.environ.get("OPERATOR_JOB_UID", "")
    packet_sha256 = os.environ.get("OPERATOR_PACKET_SHA256", "")
    source_sha256 = os.environ.get("OPERATOR_SOURCE_SHA256", "")
    try:
        UUID(uid)
    except ValueError as exc:
        raise ValueError("prod10 operator Job UID is invalid") from exc
    if (
        name != packet["operator_name"]
        or packet_sha256 != packet["sha256"]
        or not source_sha256.startswith("sha256:")
        or len(source_sha256) != 71
    ):
        raise ValueError("prod10 operator runtime binding changed")
    return uid


def _canonical_directory(path: Path, *, owner: bool = True) -> None:
    identity = path.lstat()
    mode = stat.S_IMODE(identity.st_mode)
    if (
        path.is_symlink()
        or not stat.S_ISDIR(identity.st_mode)
        or path.resolve() != path
        or (owner and (identity.st_uid, identity.st_gid) != (1000, 100))
        or not mode & stat.S_IRUSR
        or not mode & stat.S_IWUSR
        or not mode & stat.S_IXUSR
        or not os.access(path, os.R_OK | os.W_OK | os.X_OK)
    ):
        raise ValueError("prod10 operator SFS directory is not canonical and writable")


def _create_root_for_stage() -> None:
    parent = hardening.CREATE_ONCE_ROOT.parent
    _canonical_directory(parent, owner=False)
    if hardening.CREATE_ONCE_ROOT.exists() or hardening.CREATE_ONCE_ROOT.is_symlink():
        raise ValueError("prod10 operator create-once root already exists")
    hardening.CREATE_ONCE_ROOT.mkdir(mode=0o700)
    _canonical_directory(hardening.CREATE_ONCE_ROOT)
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _existing_root() -> None:
    _canonical_directory(hardening.CREATE_ONCE_ROOT)
    if not direct.live_create_is_available():
        raise ValueError("prod10 operator live create root is unavailable")


def _create_operation_root(path: Path) -> None:
    if path.parent != hardening.CREATE_ONCE_ROOT:
        raise ValueError("prod10 operator operation root changed")
    if path.exists() or path.is_symlink():
        raise ValueError("prod10 operator operation root already exists")
    path.mkdir(mode=0o700)
    _canonical_directory(path)


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
    direct.validate_cpu_preview_proof(
        expected,
        packet["dev_preview"],
        purpose="stage",
        context=direct.DEV_CONTEXT,
        fresh=True,
    )
    _create_root_for_stage()
    _existing_root()
    operation_root = hardening.stage_operation_root(stage)
    _create_operation_root(operation_root)
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
    prod_rendered = direct.server_dry_run(expected, context=direct.PROD_CONTEXT, runner=runner)
    prod_preview = direct.validate_cpu_preview(
        expected, prod_rendered, context=direct.PROD_CONTEXT, purpose="stage"
    )
    authorization = direct.authorize_stage(
        stage,
        expected,
        dev_preview=packet["dev_preview"],
        prod_preview=prod_preview,
        observer=state["armed"],
        identity=identity,
    )
    created = direct.create_stage_once(
        operation_root,
        stage,
        expected,
        authorization,
        identity=identity,
        runner=runner,
        dev_duplicate_proof=packet["dev_duplicate_proof"],
    )
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
    direct.validate_cpu_preview_proof(
        expected,
        packet["dev_preview"],
        purpose="preflight",
        context=direct.DEV_CONTEXT,
        fresh=True,
    )
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
    _validate_runtime(packet)
    runner = InClusterKubernetesRunner()
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
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}))
        raise SystemExit(1) from None
    print(json.dumps({"status": value["status"], "sha256": value["sha256"]}))


if __name__ == "__main__":
    main()
