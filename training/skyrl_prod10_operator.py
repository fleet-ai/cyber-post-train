"""Bounded in-cluster stage/preflight coordinator for the prod10 RL canary.

This is intentionally not a general controller.  One sealed packet selects one
of two fixed phases and all Kubernetes names, SFS paths, manifests, and source
identities are re-derived by the existing prod9 rail.  The stage phase performs
the existing create-once rebind inside this exact-image root Job; it does not
create a second schedulable Job.  It never launches a GPU workload.
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
DIRECT_STAGE_RESULT_SCHEMA = "cyber_skyrl_prod10_operator_direct_stage_result_v2"
TERMINATION_SCHEMA = "cyber_skyrl_prod10_operator_termination_v1"
FAILURE_TERMINATION_SCHEMA = "cyber_skyrl_prod10_operator_failure_v1"
OPERATOR_NAMES = {
    "stage": "chris-q38-prod10-stage-operator-v7",
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
_STAGE_V5_RECOVERY = {
    "schema": "cyber_skyrl_prod10_stage_precreate_recovery_v1",
    "status": "authorized_precreate_recovery",
    "previous_operator_name": "chris-q38-prod10-stage-operator-v5",
    "previous_operator_job_uid": "4ba8eb38-c3e1-400d-9488-1e281d2a3aa9",
    "previous_packet_sha256": (
        "sha256:2a10bba1d58dd004a03dfb03afd3630d742cf553f3ec98b6bb032b535a9ea851"
    ),
    "previous_failure_receipt_sha256": (
        "sha256:e01d4fda69e75213434eb328287b9966f020f2758ff373342a2b6d2708d7ba9e"
    ),
    "previous_release_sha256": (
        "sha256:07e44b411667adaa154e96694523a54f06a4ffbbabf6344e8a95ed221aadd2bf"
    ),
    "previous_error_code": "stage_production_preview_rejected",
    "root_cause": "kubectl_hidden_managed_fields",
    "gpus": 0,
}
_STAGE_V6_RECOVERY = {
    "schema": "cyber_skyrl_prod10_stage_precreate_recovery_v1",
    "status": "authorized_failed_child_recovery",
    "previous_operator_name": "chris-q38-prod10-stage-operator-v6",
    "previous_operator_job_uid": "06185d4f-8162-4de8-9525-252fa6bcb004",
    "previous_operator_pod_name": "chris-q38-prod10-stage-operator-v6-wmxsq",
    "previous_operator_pod_uid": "481004e3-9608-4e28-b2be-d8d497d6f03c",
    "previous_operator_workload_name": "job-chris-q38-prod10-stage-operator-v6-561eb",
    "previous_operator_workload_uid": "e641382b-58a0-4225-86f4-171f65e96713",
    "previous_source_config_map_name": "chris-q38-prod10-stage-operator-v6-source",
    "previous_source_config_map_uid": "fcf3aa67-12bd-40c8-b48b-69843cd08263",
    "previous_packet_config_map_name": "chris-q38-prod10-stage-operator-v6-packet",
    "previous_packet_config_map_uid": "2e60298f-724a-4098-abaf-1f26a836d4e7",
    "previous_config_maps_absent": True,
    "previous_packet_sha256": (
        "sha256:e9419752694db09ebdd72148650f5a5a5e11a04087d050c31bea3def34431fcb"
    ),
    "previous_source_sha256": (
        "sha256:7e0e180d27c1c00dc0cb79cc86272ac5d40aed1b641cc79727c1aa8f6cb4f0ef"
    ),
    "previous_job_manifest_sha256": (
        "sha256:66dafc3aa175d1a474708edaa116e0f81660cef72b03770de0a54b7a70e743e1"
    ),
    "previous_failure_receipt_sha256": (
        "sha256:1312af57876bbc0bb9b87a6ebd05669310bbce44aef215075e73f91bb843d276"
    ),
    "previous_release_sha256": (
        "sha256:ccb0cd0aaca4b0b6d21a3acaa73fa5ddf1f21e86171ce1fe1f38a23fea07c721"
    ),
    "previous_error_code": "operator_unclassified",
    "previous_target_name": "chris-q38-prod10-data-v1",
    "previous_target_job_uid": "5309b021-b876-43c5-aa25-11a8a947379d",
    "previous_target_workload_name": "job-chris-q38-prod10-data-v1-4c7d4",
    "previous_target_workload_uid": "57fb7be7-0952-41a0-a91c-30ce94459385",
    "previous_target_manifest_sha256": (
        "sha256:d62735bd266415d84051f9fc7e374cebf358205f7d2e51a53a41918df10799e7"
    ),
    "previous_kubernetes_resources_absent": True,
    "previous_destination_absent": True,
    "root_cause": "nested_cpu_head_admission_dependency",
    "gpus": 0,
}
_TERMINATION_PATH = Path("/dev/termination-log")
_POLL_SECONDS = 0.25
RUNTIME_UID = 1000
RUNTIME_GID = 100


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
    """Return the one reviewed failed-child recovery binding for stage v7."""
    return _seal(_STAGE_V6_RECOVERY)


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
    if phase == "stage":
        if packet.get("precreate_recovery") != stage_recovery_binding():
            raise ValueError("prod10 stage pre-create recovery binding changed")
    else:
        direct._validate_seal(packet.get("dev_preview"), direct.CPU_PREVIEW_SCHEMA)
        duplicate = direct._validate_seal(
            packet.get("dev_duplicate_proof"), direct.CPU_DUPLICATE_PROOF_SCHEMA
        )
        if duplicate.get("context") != direct.DEV_CONTEXT:
            raise ValueError("prod10 operator development proof changed")
    return packet


def _validate_runtime(
    packet: dict[str, Any], runner: InClusterKubernetesRunner
) -> str:
    if (os.geteuid(), os.getegid()) != (RUNTIME_UID, RUNTIME_GID):
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
            != (RUNTIME_UID, RUNTIME_GID)
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
        != (RUNTIME_UID, RUNTIME_GID)
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
    _resource_absent(runner, "job", name, code=code)


def _resource_absent(
    runner: InClusterKubernetesRunner, resource: str, name: str, *, code: str
) -> None:
    result = direct._kubectl(
        runner,
        direct.PROD_CONTEXT,
        "get",
        resource,
        name,
        "--ignore-not-found",
        "--output=json",
    )
    if result.returncode or result.stdout.strip():
        raise OperatorFailure(code)


def _read_recovery_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        identity = path.lstat()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    except (OSError, ValueError) as exc:
        raise OperatorFailure("stage_v6_create_journal_unreadable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(identity.st_mode)
        or (identity.st_uid, identity.st_gid)
        != (RUNTIME_UID, RUNTIME_GID)
        or stat.S_IMODE(identity.st_mode) != 0o600
        or len(rows) != 2
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise OperatorFailure("stage_v6_create_journal_rejected")
    return rows


def _reconcile_stage_v6_failed(
    packet: dict[str, Any],
    *,
    stage: dict[str, Any],
    expected: dict[str, Any],
    identity: historical.RailIdentity,
    runner: InClusterKubernetesRunner,
) -> Path:
    """Preserve the released v6 child and replace its nested scheduler step."""
    if packet.get("precreate_recovery") != stage_recovery_binding():
        raise OperatorFailure("stage_v6_recovery_binding_rejected")
    _existing_root()
    operation_root = hardening.stage_operation_root(stage)
    _canonical_directory(operation_root, code="stage_v6_operation_root")
    names = {entry.name for entry in operation_root.iterdir()}
    if names != {
        "PROD9_STAGE_CREATE.jsonl",
        "STAGE_OPERATOR_INTENT.json",
        "STAGE_OBSERVER_ARMED.json",
        "STAGE_OBSERVER_ARMED.json.created.json",
        "STAGE_OBSERVER_RESULT.json",
        "STAGE_OPERATOR_INTENT.v4.failed.json",
        "STAGE_OBSERVER_ARMED.v4.failed.json",
        "STAGE_OPERATOR_RECOVERY_V5.json",
        "STAGE_OPERATOR_INTENT.v5.failed.json",
        "STAGE_OBSERVER_ARMED.v5.failed.json",
        "STAGE_OPERATOR_RECOVERY_V6.json",
    }:
        raise OperatorFailure("stage_v6_recovery_inventory_rejected")
    recovery = stage_recovery_binding()
    intent_path = operation_root / "STAGE_OPERATOR_INTENT.json"
    armed_path = operation_root / "STAGE_OBSERVER_ARMED.json"
    v4_intent = _read_recovery_file(
        operation_root / "STAGE_OPERATOR_INTENT.v4.failed.json",
        "cyber_skyrl_prod10_operator_intent_v1",
    )
    v4_armed = _read_recovery_file(
        operation_root / "STAGE_OBSERVER_ARMED.v4.failed.json",
        "cyber_direct_cleanup_observer_armed_v1",
    )
    v5_receipt = _read_recovery_file(
        operation_root / "STAGE_OPERATOR_RECOVERY_V5.json",
        "cyber_skyrl_prod10_stage_precreate_recovery_receipt_v1",
    )
    if (
        v4_intent.get("packet_sha256")
        != _STAGE_V4_RECOVERY["previous_packet_sha256"]
        or v4_intent.get("operator_job_uid")
        != _STAGE_V4_RECOVERY["previous_operator_job_uid"]
        or v5_receipt.get("status") != "v4_precreate_evidence_preserved"
        or v5_receipt.get("binding_sha256") != _seal(_STAGE_V4_RECOVERY)["sha256"]
        or v5_receipt.get("previous_intent_sha256") != v4_intent.get("sha256")
        or v5_receipt.get("previous_observer_sha256") != v4_armed.get("sha256")
        or v5_receipt.get("archived_files")
        != ["STAGE_OBSERVER_ARMED.v4.failed.json", "STAGE_OPERATOR_INTENT.v4.failed.json"]
        or v5_receipt.get("gpus") != 0
    ):
        raise OperatorFailure("stage_v4_recovery_chain_rejected")
    v5_intent = _read_recovery_file(
        operation_root / "STAGE_OPERATOR_INTENT.v5.failed.json",
        "cyber_skyrl_prod10_operator_intent_v1",
    )
    v5_armed = _read_recovery_file(
        operation_root / "STAGE_OBSERVER_ARMED.v5.failed.json",
        "cyber_direct_cleanup_observer_armed_v1",
    )
    v6_recovery = _read_recovery_file(
        operation_root / "STAGE_OPERATOR_RECOVERY_V6.json",
        "cyber_skyrl_prod10_stage_precreate_recovery_receipt_v1",
    )
    if (
        v5_intent.get("packet_sha256")
        != _STAGE_V5_RECOVERY["previous_packet_sha256"]
        or v5_intent.get("operator_job_uid")
        != _STAGE_V5_RECOVERY["previous_operator_job_uid"]
        or v6_recovery.get("status") != "v5_precreate_evidence_preserved"
        or v6_recovery.get("binding_sha256")
        != _seal(_STAGE_V5_RECOVERY)["sha256"]
        or v6_recovery.get("previous_intent_sha256") != v5_intent.get("sha256")
        or v6_recovery.get("previous_observer_sha256") != v5_armed.get("sha256")
        or v6_recovery.get("archived_files")
        != ["STAGE_OBSERVER_ARMED.v5.failed.json", "STAGE_OPERATOR_INTENT.v5.failed.json"]
        or v6_recovery.get("gpus") != 0
    ):
        raise OperatorFailure("stage_v5_recovery_chain_rejected")
    intent = _read_recovery_file(
        intent_path, "cyber_skyrl_prod10_operator_intent_v1"
    )
    armed = _read_recovery_file(
        armed_path, "cyber_direct_cleanup_observer_armed_v1"
    )
    creator = _read_recovery_file(
        operation_root / "STAGE_OBSERVER_ARMED.json.created.json",
        cleanup.CREATOR_BINDING_SCHEMA,
    )
    observed = _read_recovery_file(
        operation_root / "STAGE_OBSERVER_RESULT.json",
        cleanup.DIRECT_RESULT_SCHEMA,
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
        raise OperatorFailure("stage_v6_recovery_intent_rejected")
    try:
        direct._timestamp(intent.get("created_at"))
    except JobsError as exc:
        raise OperatorFailure("stage_v6_recovery_intent_rejected") from exc
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
        raise OperatorFailure("stage_v6_recovery_observer_rejected")
    try:
        direct._timestamp(armed.get("armed_at"))
    except JobsError as exc:
        raise OperatorFailure("stage_v6_recovery_observer_rejected") from exc
    manifest_sha256 = "sha256:" + digest(expected)
    if manifest_sha256 != recovery["previous_target_manifest_sha256"]:
        raise OperatorFailure("stage_v6_target_manifest_rejected")
    if (
        creator.get("status") != "created_once"
        or creator.get("context") != direct.PROD_CONTEXT
        or creator.get("namespace") != direct.NAMESPACE
        or creator.get("kind") != "job"
        or creator.get("name") != identity.stage_name
        or creator.get("plan_sha256") != stage["sha256"]
        or creator.get("manifest_sha256") != manifest_sha256
        or creator.get("uid") != recovery["previous_target_job_uid"]
    ):
        raise OperatorFailure("stage_v6_creator_binding_rejected")
    journal_intent, created = _read_recovery_jsonl(
        operation_root / "PROD9_STAGE_CREATE.jsonl"
    )
    if (
        journal_intent.get("state") != "CREATE_INTENT_DO_NOT_RETRY"
        or journal_intent.get("purpose") != "stage"
        or journal_intent.get("plan_sha256") != stage["sha256"]
        or journal_intent.get("manifest_sha256") != manifest_sha256
        or set(journal_intent.get("duplicate_checks", {}))
        != {"kubernetes_inventories_checked", "development_duplicate_proof_sha256"}
        or journal_intent["duplicate_checks"].get("kubernetes_inventories_checked") != 10
    ):
        raise OperatorFailure("stage_v6_create_intent_rejected")
    try:
        direct._cpu_created(
            created,
            purpose="stage",
            name=identity.stage_name,
            plan_sha256=stage["sha256"],
            manifest_sha256=manifest_sha256,
            authorization_sha256=journal_intent.get("authorization_sha256"),
        )
    except JobsError as exc:
        raise OperatorFailure("stage_v6_created_receipt_rejected") from exc
    if (
        created.get("job_uid") != recovery["previous_target_job_uid"]
        or creator.get("sha256") != _seal(creator).get("sha256")
        or observed.get("status") != "released_without_accepted_execution"
        or observed.get("context") != direct.PROD_CONTEXT
        or observed.get("namespace") != direct.NAMESPACE
        or observed.get("kind") != "job"
        or observed.get("name") != identity.stage_name
        or observed.get("plan_sha256") != stage["sha256"]
        or observed.get("manifest_sha256") != manifest_sha256
        or observed.get("expected_gpus") != 0
        or observed.get("peak_gpus") != 0
        or observed.get("active_gpus") != 0
        or observed.get("uid") != recovery["previous_target_job_uid"]
        or observed.get("terminal_status") != "Deleted"
        or observed.get("workload_name") != recovery["previous_target_workload_name"]
        or observed.get("workload_uid") != recovery["previous_target_workload_uid"]
        or observed.get("pod_names") != []
        or observed.get("pod_uids") != []
        or observed.get("image_ids") != []
        or observed.get("exit_codes") != []
        or observed.get("receipt") is not None
        or observed.get("restarts") != 0
        or observed.get("deletion_reason") != "target_absent"
        or any(
            observed.get(key) is not False
            for key in (
                "target_present",
                "pods_present",
                "rayjob_present",
                "workload_present",
                "raycluster_present",
            )
        )
    ):
        raise OperatorFailure("stage_v6_release_evidence_rejected")
    for resource, name, code in (
        ("job", recovery["previous_operator_name"], "stage_v6_operator_still_present"),
        (
            "pod",
            recovery["previous_operator_pod_name"],
            "stage_v6_operator_pod_still_present",
        ),
        (
            "workload",
            recovery["previous_operator_workload_name"],
            "stage_v6_operator_workload_still_present",
        ),
        ("job", identity.stage_name, "stage_v6_target_still_present"),
        (
            "workload",
            recovery["previous_target_workload_name"],
            "stage_v6_target_workload_still_present",
        ),
    ):
        _resource_absent(runner, resource, name, code=code)
    destination = Path(identity.data_root)
    if destination.exists() or destination.is_symlink():
        raise OperatorFailure("stage_v6_destination_already_present")
    archive = {
        intent_path: operation_root / "STAGE_OPERATOR_INTENT.v6.failed.json",
        armed_path: operation_root / "STAGE_OBSERVER_ARMED.v6.failed.json",
        expected_binding: operation_root / "STAGE_OBSERVER_ARMED.v6.failed.json.created.json",
        operation_root / "STAGE_OBSERVER_RESULT.json": (
            operation_root / "STAGE_OBSERVER_RESULT.v6.failed.json"
        ),
        operation_root / "PROD9_STAGE_CREATE.jsonl": (
            operation_root / "PROD9_STAGE_CREATE.v6.failed.jsonl"
        ),
    }
    if any(path.exists() or path.is_symlink() for path in archive.values()):
        raise OperatorFailure("stage_v6_recovery_destination_exists")
    for source, target in archive.items():
        os.rename(source, target)
    descriptor = os.open(operation_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _write_once(
        operation_root / "STAGE_OPERATOR_RECOVERY_V7.json",
        _seal(
            {
                "schema": "cyber_skyrl_prod10_stage_precreate_recovery_receipt_v1",
                "status": "v6_released_child_evidence_preserved",
                "binding_sha256": recovery["sha256"],
                "previous_intent_sha256": intent["sha256"],
                "previous_observer_sha256": armed["sha256"],
                "previous_creator_sha256": creator["sha256"],
                "previous_release_sha256": observed["sha256"],
                "previous_create_receipt_sha256": created["sha256"],
                "archived_files": sorted(path.name for path in archive.values()),
                "destination_absent": True,
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
    # The child manifest is historical v6 evidence only.  V7 executes the
    # already-reviewed pure rebind directly in this exact-image root Job.
    expected = direct.stage_job_manifest(stage, identity=identity)
    operation_root = _reconcile_stage_v6_failed(
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
    try:
        raw_receipt = training.stage_rebind(stage, identity=identity)
        receipt = {**raw_receipt, "receipt_sha256": digest(raw_receipt)}
        direct._stage_receipt(stage, receipt, identity=identity)
    except (JobsError, ValueError, OSError) as exc:
        raise OperatorFailure("stage_rebind_rejected") from exc
    recovery = _read_recovery_file(
        operation_root / "STAGE_OPERATOR_RECOVERY_V7.json",
        "cyber_skyrl_prod10_stage_precreate_recovery_receipt_v1",
    )
    return _seal(
        {
            "schema": DIRECT_STAGE_RESULT_SCHEMA,
            "status": "stage_ready",
            "phase": "stage",
            "packet_sha256": packet["sha256"],
            "stage": stage,
            "receipt": receipt,
            "recovery_sha256": recovery["sha256"],
            "execution": {
                "kind": "job",
                "name": packet["operator_name"],
                "uid": os.environ["OPERATOR_JOB_UID"],
                "image": stage["image"],
                "source_sha256": os.environ["OPERATOR_SOURCE_SHA256"],
                "sfs_output": stage["destination"],
                "receipt_sha256": receipt["receipt_sha256"],
                "nested_jobs_created": 0,
            },
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
