"""Bounded in-cluster stage/preflight/launch coordinator for prod10 RL.

This is intentionally not a general controller.  One sealed packet selects one
fixed phase and all Kubernetes names, SFS paths, manifests, and source identities
are re-derived by the existing prod9 rail.  Stage and preflight remain zero-GPU;
launch performs one Jobs API create and then observes that exact run through
terminal status, receipt capture, and confirmed resource release.
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

from cyber_post_train.gpu_capacity import build_capacity_census
from cyber_post_train.jobs import Jobs, JobsError, digest

from . import dev_cleanup_observer as cleanup
from . import skyrl_prod9_direct as direct
from . import skyrl_prod9_hardening as hardening
from . import skyrl_prod9_training as training
from . import skyrl_prod10_direct as launch_direct
from . import skyrl_reward_rayjob as historical
from .incluster_kubernetes import InClusterKubernetesError, InClusterKubernetesRunner

PACKET_SCHEMA = "cyber_skyrl_prod10_operator_packet_v1"
RESULT_SCHEMA = "cyber_skyrl_prod10_operator_result_v1"
LAUNCH_RESULT_SCHEMA = "cyber_skyrl_prod10_launch_result_v1"
DIRECT_STAGE_RESULT_SCHEMA = "cyber_skyrl_prod10_operator_direct_stage_result_v2"
MANIFEST_RESULT_SCHEMA = "cyber_skyrl_prod10_rebound_manifest_result_v1"
TERMINATION_SCHEMA = "cyber_skyrl_prod10_operator_termination_v1"
FAILURE_TERMINATION_SCHEMA = "cyber_skyrl_prod10_operator_failure_v1"
OPERATOR_NAMES = {
    "stage": "chris-q38-prod10-stage-operator-v7",
    "manifest": "chris-q38-prod10-manifest-operator-v1",
    "preflight": "chris-q38-prod10-preflight-operator-v2",
    "launch": "chris-q38-prod10-launch-operator-v2",
    "inspect": "chris-q38-prod10-launch-inspect-v3",
    "probe": "chris-q38-prod10-launch-probe-v7",
}
_LAUNCH_V1_FAILURE = {
    "schema": "cyber_skyrl_prod10_launch_failure_binding_v1",
    "status": "failed_before_gpu_create_and_released",
    "operator_name": "chris-q38-prod10-launch-operator-v1",
    "operator_job_uid": "a8b8373c-926f-4aa3-a475-d49a7457f591",
    "operator_pod_uid": "a112c349-acb0-4134-a1e6-cf0425279cb7",
    "operator_workload_uid": "2dc0b188-1e9b-401e-bb48-4744a27e2c81",
    "failure_receipt_sha256": (
        "sha256:7c45bc3c4356b0ca94ecd0fdd312ed068dc890144dc91e5d26c6c2793a53d421"
    ),
    "release_sha256": (
        "sha256:c9adf0f0b995906813668bba82180c84384f9de5f0185270d3f428ae404c0292"
    ),
    "inner_gpu_run_created": False,
    "gpus": 0,
}
_LAUNCH_V2_FAILURE = {
    "schema": "cyber_skyrl_prod10_launch_failure_binding_v1",
    "status": "failed_before_gpu_create_and_released",
    "operator_name": "chris-q38-prod10-launch-operator-v2",
    "operator_job_uid": "1be3b0e6-3825-4ea3-aac4-1a2c92c19165",
    "operator_pod_uid": "909bea1d-fa24-4525-8cc2-068cd6d616af",
    "operator_workload_uid": "e01df31f-f821-4661-9b65-d35159480a7a",
    "failure_receipt_sha256": (
        "sha256:6f7d724ae7a0a61179616871895c0e2cbcb496ef351279527c6d6fb4b6f145c1"
    ),
    "release_sha256": (
        "sha256:2cb973f61095c77641602fdf2ebb469c31f0a7b0aa907b053b01119dd0bc3992"
    ),
    "launch_packet_sha256": (
        "sha256:f0903dc2ce7768dbf81d0c91838d0bbe238c162ff2dce96c48061c124961336a"
    ),
    "inner_gpu_run_created": False,
    "gpus": 0,
}
_INSPECT_V2_SUCCESS = {
    "schema": "cyber_skyrl_prod10_launch_inspection_binding_v1",
    "status": "succeeded_and_released",
    "operator_name": "chris-q38-prod10-launch-inspect-v2",
    "operator_job_uid": "f92a3770-de2d-4fb0-97aa-70b09bd1cde5",
    "operator_pod_uid": "a0c637db-0b75-4af3-9ec2-4d7e31abf566",
    "operator_workload_uid": "7fb710ec-e24e-45c7-9301-53c08c2e0f6a",
    "receipt_sha256": "sha256:73f751235514f1b05a6bde2074dc3bdb62995f1c84ebb096d33ad0e461725c21",
    "observer_sha256": "sha256:62a7cc889963bba3f561b7d3fbd00a5e1d61e328e22c59092b509638a537ef59",
    "result_sha256": "sha256:0ec4bcbe9aa48f9b96a7bb83a8c24aee7ee7c0961d5cca00afa63a04d7dfadbb",
    "launch_boundary": "before_guard_or_guard_write",
    "gpus": 0,
}
_INSPECT_V3_SUCCESS = {
    "schema": "cyber_skyrl_prod10_launch_inspection_binding_v1",
    "status": "succeeded_and_released",
    "operator_name": "chris-q38-prod10-launch-inspect-v3",
    "operator_job_uid": "64437ed3-a2e6-4aae-9da7-b9b927c1a694",
    "operator_pod_uid": "6edc20ad-51f8-4f90-bb2e-e9a46bfd2455",
    "operator_workload_uid": "53713f63-eeff-42a7-ab92-1795316c2912",
    "receipt_sha256": "sha256:20ce05dfc0ad401a8bb3137dcd7cd3ae2db2f933360f3decbf12243209e102b9",
    "observer_sha256": "sha256:d48742b1d2fdfe49db98fe08b98155f7d98f08931c461c66562940c2afc77703",
    "result_sha256": "sha256:ed30f976916f023e9e5a2533040a6d76fe79fa4aec910de5d7e274a044a403e1",
    "launch_boundary": "before_guard_or_guard_write",
    "gpus": 0,
}
_PROBE_V5_SUCCESS = {
    "schema": "cyber_skyrl_prod10_launch_probe_success_binding_v1",
    "status": "diagnostic_succeeded_and_released",
    "operator_name": "chris-q38-prod10-launch-probe-v5",
    "operator_job_uid": "dc6c830e-b2bf-42db-b7cb-c9759e433fab",
    "operator_pod_uid": "4a6dc651-a21e-4a80-a870-82b9cd2dacf1",
    "operator_workload_uid": "fa5b12ad-80d8-498c-ab79-e3d8596addb3",
    "receipt_sha256": "sha256:6b11ed8825e77391f2f52e858627d92e2a5044239314f2d879799664466d6fe4",
    "observer_sha256": "sha256:8ea93460d8a39fe344ee92941b7aebb4c85a892ead4ed4c23920dd6cfee1e3ba",
    "result_sha256": "sha256:620f41c5a4d5ef68bf09d83a0901b7bc2586bbb8ae5904f4bdb08203e9449e0e",
    "diagnosis": "dataset_cache_oserror",
    "gpus": 0,
}
_PROBE_V6_SUCCESS = {
    "schema": "cyber_skyrl_prod10_launch_probe_success_binding_v1",
    "status": "diagnostic_succeeded_and_released",
    "operator_name": "chris-q38-prod10-launch-probe-v6",
    "operator_job_uid": "6a7b3779-b806-48b2-b264-10cebb7aff62",
    "operator_pod_uid": "e6ee1488-51fb-4883-80c9-dea44f6a8741",
    "operator_workload_uid": "46d568b9-c7a5-4ba7-ab1d-1d8b0a1d0249",
    "receipt_sha256": "sha256:2e818e9e7e37880c5b60975f521fc9430b2314a18ad0b0eec2c19964a076fa69",
    "observer_sha256": "sha256:0cf33da636e39511af94f6a22600c62908b25609ca9aa1ab3f5587dc79c93ed0",
    "result_sha256": "sha256:c88a69c013f63dbd5565611cfa04d660909630dcb98f33649d520dbd0d2066ab",
    "diagnosis": "fresh_training_preflight_passed",
    "gpus": 0,
}
_PREFLIGHT_V1_FAILURE = {
    "schema": "cyber_skyrl_prod10_preflight_v1_failure_recovery_v1",
    "status": "failed_closed_released",
    "operator_name": "chris-q38-prod10-preflight-operator-v1",
    "operator_job_uid": "049ccc35-c347-4c4b-8f99-353258e5638f",
    "operator_pod_name": "chris-q38-prod10-preflight-operator-v1-phz8d",
    "operator_pod_uid": "ac8619ac-bf92-423e-9667-87d5247ae448",
    "operator_workload_name": "job-chris-q38-prod10-preflight-operator-v1-41c3d",
    "operator_workload_uid": "91c553f7-e08a-47a7-8d97-df535b28b136",
    "source_config_map_name": "chris-q38-prod10-preflight-operator-v1-source",
    "source_config_map_uid": "40901d30-ec06-4dd9-a6e9-026d183e0c25",
    "packet_config_map_name": "chris-q38-prod10-preflight-operator-v1-packet",
    "packet_config_map_uid": "22f51bd0-710a-47d4-ad7e-cbc141803e8c",
    "failure_receipt_sha256": (
        "sha256:edf26867c2a26fdf476852e7e83cb5df8ebd0f89b06002795cb898e128253086"
    ),
    "release_sha256": ("sha256:29c67ae6990e3401f92268cc758b17a9ec3b07aae6596ea1462b38c6d06e2ecc"),
    "child_name": "chris-q38-prod10-preflight-v1",
    "child_created": False,
    "operation_root": (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/"
        "prod9-create-once-v1/training-b14834bef3d4e12dd9d0b07fce88aa96bee1cf51befd892e9c86f847e60cb8b1"
    ),
    "kubernetes_resources_absent": True,
    "gpus": 0,
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
_LAUNCH_STAGE = "not_started"


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


def preflight_v1_failure_binding() -> dict[str, Any]:
    """Bind the one released outer preflight failure; this identity is never retried."""
    return _seal(_PREFLIGHT_V1_FAILURE)


def launch_v1_failure_binding() -> dict[str, Any]:
    """Bind the released launch failure inspected by the zero-GPU successor."""
    return _seal(_LAUNCH_V1_FAILURE)


def launch_v2_failure_binding() -> dict[str, Any]:
    """Bind the released launch-v2 failure for its read-only inspector."""
    return _seal(_LAUNCH_V2_FAILURE)


def inspect_v2_success_binding() -> dict[str, Any]:
    """Bind the exact released inspector used by the phase-coded probe."""
    return _seal(_INSPECT_V2_SUCCESS)


def inspect_v3_success_binding() -> dict[str, Any]:
    """Bind the exact released launch-v2 boundary inspection."""
    return _seal(_INSPECT_V3_SUCCESS)


def probe_v5_success_binding() -> dict[str, Any]:
    """Bind the released v5 diagnostic before testing writable cache roots."""
    return _seal(_PROBE_V5_SUCCESS)


def probe_v6_success_binding() -> dict[str, Any]:
    """Bind the released v6 proof that the one-variable cache repair passed."""
    return _seal(_PROBE_V6_SUCCESS)


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


def _write_manifest_termination(result: dict[str, Any]) -> None:
    value = _validate_seal(result, MANIFEST_RESULT_SCHEMA)
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(encoded) > 3900:
        raise ValueError("prod10 sanitized manifest receipt is too large")
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
    elif phase == "manifest":
        if packet.get("preflight_v1_failure") != preflight_v1_failure_binding():
            raise ValueError("prod10 preflight v1 recovery binding changed")
    elif phase == "preflight":
        direct._validate_seal(packet.get("dev_preview"), direct.CPU_PREVIEW_SCHEMA)
        direct._validate_seal(
            packet.get("manifest_launch_result"),
            direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA,
        )
        duplicate = direct._validate_seal(
            packet.get("dev_duplicate_proof"), direct.CPU_DUPLICATE_PROOF_SCHEMA
        )
        if duplicate.get("context") != direct.DEV_CONTEXT:
            raise ValueError("prod10 operator development proof changed")
    elif phase == "inspect":
        if packet.get("launch_v2_failure") != launch_v2_failure_binding():
            raise ValueError("prod10 launch inspection predecessor changed")
        plan = packet.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("prod10 launch inspection plan changed")
        direct._identity(plan, _identity(packet.get("identity")))
        launch_direct._preflight_launch(
            packet.get("preflight_launch_result"),
            plan,
            identity=_identity(packet.get("identity")),
            operator_name=OPERATOR_NAMES["preflight"],
        )
    elif phase == "probe":
        if packet.get("launch_v2_failure") != launch_v2_failure_binding():
            raise ValueError("prod10 launch probe failure predecessor changed")
        if packet.get("inspect_v3_success") != inspect_v3_success_binding():
            raise ValueError("prod10 launch probe inspection predecessor changed")
        launch_packet = _packet(packet.get("launch_packet"), "launch")
        if launch_packet.get("sha256") != _LAUNCH_V2_FAILURE["launch_packet_sha256"]:
            raise ValueError("prod10 launch probe packet predecessor changed")
    else:
        if packet.get("launch_v1_failure") != launch_v1_failure_binding():
            raise ValueError("prod10 launch failure predecessor changed")
        if packet.get("probe_v6_success") != probe_v6_success_binding():
            raise ValueError("prod10 launch repair proof changed")
        direct._validate_seal(
            packet.get("preflight_launch_result"),
            direct.STAGE_OPERATOR_LAUNCH_RESULT_SCHEMA,
        )
        direct._validate_seal(packet.get("dev_preview"), direct.PREVIEW_SCHEMA)
        direct._validate_seal(packet.get("duplicate_proof"), launch_direct.DUPLICATE_SCHEMA)
        census = packet.get("capacity_census")
        request = packet.get("request")
        planned = (
            {
                "nodes": request.get("workers"),
                "gpus": request.get("workers", 0) * request.get("gpus_per_worker", 0),
            }
            if isinstance(request, dict)
            else None
        )
        body = (
            {key: item for key, item in census.items() if key != "sha256"}
            if isinstance(census, dict)
            else {}
        )
        if (
            not isinstance(census, dict)
            or census.get("schema") != "cyber_project_gpu_capacity_census_v1"
            or census.get("sha256") != digest(body)
            or census.get("limits") != {"nodes": 10, "gpus": 80}
            or census.get("planned") != planned
        ):
            raise ValueError("prod10 launch capacity proof changed")
    return packet


def _validate_runtime(packet: dict[str, Any], runner: InClusterKubernetesRunner) -> str:
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
        or annotations.get("cyber-post-train.fleet.ai/operator-packet-sha256") != packet_sha256
        or annotations.get("cyber-post-train.fleet.ai/operator-source-sha256") != source_sha256
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or job.get("spec", {}).get("template", {}).get("spec", {}).get("priorityClassName") != "c1"
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
        or (owner and (identity.st_uid, identity.st_gid) != (RUNTIME_UID, RUNTIME_GID))
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
    _canonical_directory(hardening.CREATE_ONCE_ROOT, code="sfs_create_once_root_postcondition")
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
        or (identity.st_uid, identity.st_gid) != (RUNTIME_UID, RUNTIME_GID)
        or stat.S_IMODE(identity.st_mode) != 0o600
    ):
        raise OperatorFailure("stage_v4_recovery_evidence_rejected")
    try:
        return _validate_seal(value, schema)
    except ValueError as exc:
        raise OperatorFailure("stage_v4_recovery_evidence_rejected") from exc


def _job_absent(runner: InClusterKubernetesRunner, name: str, *, code: str) -> None:
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
        or (identity.st_uid, identity.st_gid) != (RUNTIME_UID, RUNTIME_GID)
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
        v4_intent.get("packet_sha256") != _STAGE_V4_RECOVERY["previous_packet_sha256"]
        or v4_intent.get("operator_job_uid") != _STAGE_V4_RECOVERY["previous_operator_job_uid"]
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
        v5_intent.get("packet_sha256") != _STAGE_V5_RECOVERY["previous_packet_sha256"]
        or v5_intent.get("operator_job_uid") != _STAGE_V5_RECOVERY["previous_operator_job_uid"]
        or v6_recovery.get("status") != "v5_precreate_evidence_preserved"
        or v6_recovery.get("binding_sha256") != _seal(_STAGE_V5_RECOVERY)["sha256"]
        or v6_recovery.get("previous_intent_sha256") != v5_intent.get("sha256")
        or v6_recovery.get("previous_observer_sha256") != v5_armed.get("sha256")
        or v6_recovery.get("archived_files")
        != ["STAGE_OBSERVER_ARMED.v5.failed.json", "STAGE_OPERATOR_INTENT.v5.failed.json"]
        or v6_recovery.get("gpus") != 0
    ):
        raise OperatorFailure("stage_v5_recovery_chain_rejected")
    intent = _read_recovery_file(intent_path, "cyber_skyrl_prod10_operator_intent_v1")
    armed = _read_recovery_file(armed_path, "cyber_direct_cleanup_observer_armed_v1")
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
    journal_intent, created = _read_recovery_jsonl(operation_root / "PROD9_STAGE_CREATE.jsonl")
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
    *,
    operation_root: Path,
    purpose: str,
    name: str,
    plan_sha256: str,
    manifest_sha256: str,
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


def run_preflight(packet: dict[str, Any], *, runner: InClusterKubernetesRunner) -> dict[str, Any]:
    identity = _identity(packet["identity"])
    plan = packet.get("plan")
    request = packet.get("request")
    if not isinstance(plan, dict) or not isinstance(request, dict):
        raise ValueError("prod10 preflight operator plan/request is invalid")
    direct._identity(plan, identity)
    if training.job_request(plan) != request:
        raise ValueError("prod10 preflight operator request changed")
    stage, stage_identity = training._stage_identity(packet.get("stage"))
    if stage_identity != identity:
        raise ValueError("prod10 preflight stage identity changed")
    stage_launch = direct._direct_stage_launch(
        packet.get("stage_launch_result"),
        stage,
        identity=identity,
        operator_name=OPERATOR_NAMES["stage"],
        fresh=False,
    )
    result_path = hardening.stage_operation_root(stage) / "STAGE_OPERATOR_RESULT.json"
    if stage_launch["observer"]["receipt"].get("result_path") != str(result_path):
        raise OperatorFailure("direct_stage_result_path_rejected")
    stage_result = _read_recovery_file(result_path, DIRECT_STAGE_RESULT_SCHEMA)
    if stage_result.get("sha256") != stage_launch["observer"]["receipt"].get("result_sha256"):
        raise OperatorFailure("direct_stage_result_digest_rejected")
    direct._direct_stage_evidence(
        stage,
        stage_result,
        stage_launch,
        plan=plan,
        identity=identity,
        operator_name=OPERATOR_NAMES["stage"],
        fresh_release=False,
    )
    manifest_launch = direct._direct_manifest_launch(
        packet.get("manifest_launch_result"),
        plan,
        stage_result,
        stage_launch,
        operator_name=OPERATOR_NAMES["manifest"],
        fresh=True,
    )
    for resource, name in (
        ("job", OPERATOR_NAMES["stage"]),
        ("workload", stage_launch["observer"]["workload_name"]),
        ("configmap", stage_launch["created"]["source_config_map"]["name"]),
        ("configmap", stage_launch["created"]["packet_config_map"]["name"]),
        *(("pod", name) for name in stage_launch["observer"]["pod_names"]),
        ("job", OPERATOR_NAMES["manifest"]),
        ("workload", manifest_launch["observer"]["workload_name"]),
        ("configmap", manifest_launch["created"]["source_config_map"]["name"]),
        ("configmap", manifest_launch["created"]["packet_config_map"]["name"]),
        *(("pod", name) for name in manifest_launch["observer"]["pod_names"]),
    ):
        _resource_absent(runner, resource, name, code="direct_stage_resource_still_present")
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
    authorization = direct.authorize_preflight_direct_manifest(
        plan,
        request,
        stage,
        stage_result,
        stage_launch,
        manifest_launch,
        expected,
        stage_operator_name=OPERATOR_NAMES["stage"],
        manifest_operator_name=OPERATOR_NAMES["manifest"],
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
            "stage_launch_result_sha256": stage_launch["sha256"],
            "manifest_launch_result_sha256": manifest_launch["sha256"],
            "authorization": authorization,
            "created": created,
            "receipt": receipt,
            "release": release,
            "gpus": 0,
        }
    )


def run_manifest(packet: dict[str, Any], *, runner: InClusterKubernetesRunner) -> dict[str, Any]:
    identity = _identity(packet["identity"])
    stage, stage_identity = training._stage_identity(packet.get("stage"))
    if stage_identity != identity:
        raise ValueError("prod10 manifest stage identity changed")
    stage_launch = direct._direct_stage_launch(
        packet.get("stage_launch_result"),
        stage,
        identity=identity,
        operator_name=OPERATOR_NAMES["stage"],
        fresh=False,
    )
    result_path = hardening.stage_operation_root(stage) / "STAGE_OPERATOR_RESULT.json"
    if stage_launch["observer"]["receipt"].get("result_path") != str(result_path):
        raise OperatorFailure("manifest_stage_result_path_rejected")
    stage_result = _read_recovery_file(result_path, DIRECT_STAGE_RESULT_SCHEMA)
    if stage_result.get("sha256") != stage_launch["observer"]["receipt"].get("result_sha256"):
        raise OperatorFailure("manifest_stage_result_digest_rejected")
    stage_result, stage_launch, successor = direct._direct_stage_rebound_evidence(
        stage,
        stage_result,
        stage_launch,
        identity=identity,
        operator_name=OPERATOR_NAMES["stage"],
        fresh_release=False,
    )
    recovery = preflight_v1_failure_binding()
    resources = [
        ("job", OPERATOR_NAMES["stage"]),
        ("workload", stage_launch["observer"]["workload_name"]),
        ("configmap", stage_launch["created"]["source_config_map"]["name"]),
        ("configmap", stage_launch["created"]["packet_config_map"]["name"]),
        *(("pod", name) for name in stage_launch["observer"]["pod_names"]),
        ("job", recovery["operator_name"]),
        ("pod", recovery["operator_pod_name"]),
        ("workload", recovery["operator_workload_name"]),
        ("configmap", recovery["source_config_map_name"]),
        ("configmap", recovery["packet_config_map_name"]),
        ("job", recovery["child_name"]),
    ]
    for resource, name in resources:
        _resource_absent(runner, resource, name, code="manifest_predecessor_resource_still_present")
    operation_root = Path(recovery["operation_root"])
    if operation_root.exists() or operation_root.is_symlink():
        raise OperatorFailure("manifest_failed_preflight_root_exists")
    output_root = Path(identity.output_root)
    if output_root.exists() or output_root.is_symlink():
        raise OperatorFailure("manifest_gpu_output_exists")
    return _seal(
        {
            "schema": MANIFEST_RESULT_SCHEMA,
            "status": "passed",
            "phase": "manifest",
            "stage_result_sha256": stage_result["sha256"],
            "stage_launch_result_sha256": stage_launch["sha256"],
            "preflight_v1_failure_sha256": recovery["sha256"],
            "successor_manifest": successor,
            "successor_manifest_sha256": successor["sha256"],
            "private_rows_exported": False,
            "nested_jobs_created": 0,
            "gpus": 0,
        }
    )


def _observe_created_run(
    operation_root: Path,
    plan: dict[str, Any],
    created: dict[str, Any],
    *,
    runner: InClusterKubernetesRunner,
) -> dict[str, Any]:
    """Retain the launch operator until its exact created RayJob is released."""
    binding_path = hardening.creator_binding_path(operation_root, "training")
    try:
        binding = json.loads(binding_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise OperatorFailure("launch_exact_binding_unavailable") from exc
    binding = direct._validate_seal(binding, cleanup.JOBS_API_EXACT_BINDING_SCHEMA)
    release_contract = direct._seal(
        {
            "schema": cleanup.JOBS_API_RELEASE_CONTRACT_SCHEMA,
            "status": "creator_authorized_exact_uid_release",
            "binding_sha256": binding["sha256"],
            "context": binding["context"],
            "namespace": binding["namespace"],
            "jobs_api_run_name": binding["jobs_api_run_name"],
            "jobs_api_run_id": binding["jobs_api_run_id"],
            "rayjob_name": binding["rayjob_name"],
            "rayjob_uid": binding["rayjob_uid"],
            "authorized_at": binding["bound_at"],
            "release_route": "raw_rayjob_uid_precondition_v1",
        }
    )
    release_path = operation_root / "PROD10_EXACT_RELEASE_CONTRACT.json"
    observer_path = operation_root / "PROD10_EXACT_OBSERVER_RESULT.json"
    _write_once(release_path, release_contract)
    observed = _Prod10ExactUidObserver(
        binding_path=binding_path,
        result_path=observer_path,
        release_contract_path=release_path,
        run=runner,
    ).run()
    receipt = observed.get("receipt")
    args = plan.get("arguments", {})
    steps, interval = args.get("steps"), args.get("eval_interval")
    expected_batches = (
        len(
            {("train", step) for step in range(1, steps + 1)}
            | {("eval", 0), ("eval", steps)}
            | {("eval", step) for step in range(1, steps + 1) if step % interval == 0}
        )
        if type(steps) is int and steps > 0 and type(interval) is int and interval > 0
        else -1
    )
    receipt_body = (
        {key: value for key, value in receipt.items() if key != "sha256"}
        if isinstance(receipt, dict)
        else {}
    )
    if (
        observed.get("status") != "released_after_terminal"
        or observed.get("release_confirmed") is not True
        or observed.get("binding_sha256") != binding["sha256"]
        or observed.get("rayjob_uid") != created["rayjob_uid"]
        or observed.get("peak_gpus") != created["gpus"]
        or observed.get("runtime_image_identity_complete") is not True
        or observed.get("restarts") != 0
        or not isinstance(observed.get("exit_codes"), list)
        or not observed["exit_codes"]
        or any(type(code) is not int or code != 0 for code in observed["exit_codes"])
        or observed.get("terminal_status") != "Succeeded"
        or not isinstance(receipt, dict)
        or set(receipt)
        != {
            "status",
            "plan_sha256",
            "checkpoint_global_step",
            "completed_batches",
            "completed_at",
            "optimizer_update_independently_verified",
            "checkpoint_reload_verified",
            "sha256",
        }
        or receipt.get("sha256") != digest(receipt_body)
        or receipt.get("status") != "native_loop_returned"
        or receipt.get("plan_sha256") != digest(plan)
        or receipt.get("checkpoint_global_step") != steps
        or receipt.get("completed_batches") != expected_batches
        or receipt.get("optimizer_update_independently_verified") is not False
        or receipt.get("checkpoint_reload_verified") is not False
    ):
        raise OperatorFailure("launch_exact_observer_rejected")
    return observed


class _Prod10ExactUidObserver(cleanup.JobsApiExactUidObserver):
    """Wait briefly for a terminal Pod receipt before exact-UID release."""

    def _request_exact_uid_cleanup(self) -> None:
        if self.terminal_status and self.receipt is None:
            remaining = (
                max(0.0, (self.deadline_at - cleanup._now()).total_seconds())
                if self.deadline_at is not None
                else cleanup.TERMINAL_RECEIPT_GRACE_SECONDS
            )
            deadline = time.monotonic() + min(cleanup.TERMINAL_RECEIPT_GRACE_SECONDS, remaining)
            while self.receipt is None and time.monotonic() < deadline:
                time.sleep(min(self.poll_seconds, deadline - time.monotonic()))
                root = self._get("rayjob", self.root_name)
                if root is None:
                    return
                cluster_name = self._validate_root(root)
                self._observe_owned_children(cluster_name)
        super()._request_exact_uid_cleanup()


def _fresh_capacity_census(
    request: dict[str, Any], runner: InClusterKubernetesRunner
) -> dict[str, Any]:
    inventories = runner.capacity_inventory()
    census = build_capacity_census(
        inventories["pods"],
        inventories["inference_models"],
        inventories["rayjobs"],
        inventories["workloads"],
        owner_prefixes=tuple(hardening.PROJECT_OWNER_PREFIXES),
        max_nodes=launch_direct.MAX_NODES,
        max_gpus=launch_direct.MAX_GPUS,
        planned_nodes=request["workers"],
        planned_gpus=request["workers"] * request["gpus_per_worker"],
    )
    if census.get("qualified") is not True:
        raise OperatorFailure("launch_capacity_rejected")
    return census


def _inspect_path(path: Path) -> dict[str, Any]:
    """Return bounded lstat/access metadata without reading file contents."""
    try:
        identity = path.lstat()
    except FileNotFoundError:
        return {"state": "absent"}
    except OSError:
        return {"state": "lstat_error"}
    kind = (
        "regular"
        if stat.S_ISREG(identity.st_mode)
        else "directory"
        if stat.S_ISDIR(identity.st_mode)
        else "symlink"
        if stat.S_ISLNK(identity.st_mode)
        else "other"
    )
    return {
        "state": "present",
        "kind": kind,
        "uid": identity.st_uid,
        "gid": identity.st_gid,
        "mode": f"{stat.S_IMODE(identity.st_mode):04o}",
        "readable": os.access(path, os.R_OK),
        "traversable": os.access(path, os.X_OK) if kind == "directory" else False,
    }


def _inspect_group(paths: list[Path]) -> dict[str, Any]:
    probes = [_inspect_path(path) for path in paths]
    return {
        "expected": len(probes),
        "present": sum(value.get("state") == "present" for value in probes),
        "regular": sum(value.get("kind") == "regular" for value in probes),
        "readable": sum(value.get("readable") is True for value in probes),
        "lstat_errors": sum(value.get("state") == "lstat_error" for value in probes),
    }


def _inspection_boundary(probes: dict[str, dict[str, Any]]) -> str:
    states = {
        name: probes[name].get("state")
        for name in ("guard", "create_journal", "creator_binding")
    }
    if "lstat_error" in states.values():
        return "indeterminate_or_inconsistent"
    if states == {
        "guard": "absent",
        "create_journal": "absent",
        "creator_binding": "absent",
    }:
        return "before_guard_or_guard_write"
    if states == {
        "guard": "present",
        "create_journal": "absent",
        "creator_binding": "absent",
    }:
        return "after_guard_before_intent"
    if states == {
        "guard": "present",
        "create_journal": "present",
        "creator_binding": "absent",
    }:
        return "intent_crossed_never_retry"
    if states == {
        "guard": "present",
        "create_journal": "present",
        "creator_binding": "present",
    }:
        return "binding_present"
    return "indeterminate_or_inconsistent"


def run_inspect(packet: dict[str, Any]) -> dict[str, Any]:
    """Inspect only allowlisted public path metadata after the released v2 failure."""
    identity = _identity(packet["identity"])
    plan = packet.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("prod10 launch inspection plan changed")
    direct._identity(plan, identity)
    launch = launch_direct._preflight_launch(
        packet.get("preflight_launch_result"),
        plan,
        identity=identity,
        operator_name=OPERATOR_NAMES["preflight"],
    )
    operation_root = hardening.training_operation_root(plan)
    data_root = Path(plan["arguments"]["data_manifest"]).parent
    model_root = Path(plan["model"]["root"])
    tokenizer_names = [item["path"] for item in plan["data"]["tokenizer"]["files"]]
    model_names = [item["path"] for item in plan["model"]["files"]]
    shard_names = [name for name in model_names if name.endswith(".safetensors")]
    sidecar_names = [name for name in model_names if name not in shard_names]
    result_path = Path(launch["observer"]["receipt"]["result_path"])
    probes = {
        "control_root": _inspect_path(hardening.CREATE_ONCE_ROOT),
        "operation_root": _inspect_path(operation_root),
        "guard": _inspect_path(direct.jobs_api_guard_path(operation_root, "training")),
        "create_journal": _inspect_path(operation_root / "PROD10_DIRECT_V3_CREATE.jsonl"),
        "creator_binding": _inspect_path(
            hardening.creator_binding_path(operation_root, "training")
        ),
        "preflight_result": _inspect_path(result_path),
        "data_root": _inspect_path(data_root),
        "output_root": _inspect_path(Path(identity.output_root)),
    }
    data = {
        name: _inspect_path(data_root / name)
        for name in ("manifest.json", "split.json", "task-set.json", "train.jsonl", "dev.jsonl")
    }
    return _seal(
        {
            # Reuse the existing sanitized, observer-accepted CPU receipt
            # envelope so this diagnostic does not alter the frozen prod9
            # observer allowlist.  ``phase`` distinguishes the payload.
            "schema": MANIFEST_RESULT_SCHEMA,
            "status": "passed",
            "phase": "inspect",
            "launch_v2_failure_sha256": launch_v2_failure_binding()["sha256"],
            "preflight_launch_sha256": launch["sha256"],
            "paths": probes,
            "launch_boundary": _inspection_boundary(probes),
            "data_files": data,
            "tokenizer_files": _inspect_group([model_root / name for name in tokenizer_names]),
            "model_sidecars": _inspect_group([model_root / name for name in sidecar_names]),
            "model_shards": _inspect_group([model_root / name for name in shard_names]),
            "contents_read": False,
            "nested_jobs_created": 0,
            "gpus": 0,
        }
    )


def _pre_guard_launch(
    packet: dict[str, Any], runner: InClusterKubernetesRunner
) -> dict[str, Any]:
    """Run the exact launch sequence that precedes guard construction."""
    global _LAUNCH_STAGE
    _LAUNCH_STAGE = "plan_identity"
    identity = _identity(packet["identity"])
    plan = packet.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("prod10 launch plan changed")
    direct._identity(plan, identity)
    _LAUNCH_STAGE = "request_identity"
    request = packet.get("request")
    if not isinstance(request, dict) or training.job_request(plan) != request:
        raise ValueError("prod10 launch request changed")
    _LAUNCH_STAGE = "preflight_launch_binding"
    preflight_launch = launch_direct._preflight_launch(
        packet.get("preflight_launch_result"),
        plan,
        identity=identity,
        operator_name=OPERATOR_NAMES["preflight"],
    )
    _LAUNCH_STAGE = "preflight_result_read"
    receipt = preflight_launch["observer"]["receipt"]
    preflight = _read_recovery_file(
        Path(receipt["result_path"]), launch_direct.PREFLIGHT_RESULT_SCHEMA
    )
    _LAUNCH_STAGE = "preflight_result_digest"
    if preflight.get("sha256") != receipt["result_sha256"]:
        raise OperatorFailure("launch_preflight_result_digest_rejected")
    _LAUNCH_STAGE = "sealed_preflight_validate"
    preflight = launch_direct.preflight_result(
        plan, request, preflight, identity=identity
    )
    _LAUNCH_STAGE = "fresh_preflight_revalidate"
    revalidation = launch_direct.revalidate_preflight(
        plan, request, preflight, identity=identity
    )
    _LAUNCH_STAGE = "image_identity"
    image_identity = launch_direct.image_identity(request, preflight)
    _LAUNCH_STAGE = "source_preview_shape"
    source_preview = packet.get("source_preview")
    if not isinstance(source_preview, dict):
        raise ValueError("prod10 launch Jobs preview is invalid")
    _LAUNCH_STAGE = "manifest_rebuild"
    expected = direct.manifest(
        plan,
        request,
        source_preview,
        identity=identity,
        image_identity_receipt=image_identity,
    )
    _LAUNCH_STAGE = "manifest_digest"
    if packet.get("manifest_sha256") != "sha256:" + digest(expected):
        raise ValueError("prod10 launch GPU manifest changed")
    _LAUNCH_STAGE = "operation_root_validate"
    operation_root = hardening.training_operation_root(plan)
    _canonical_directory(operation_root, code="launch_operation_root")
    _LAUNCH_STAGE = "output_absence"
    output_root = Path(identity.output_root)
    if output_root.exists() or output_root.is_symlink():
        raise OperatorFailure("launch_output_exists")
    _LAUNCH_STAGE = "server_dry_run"
    prod_dry_run = direct.server_dry_run(
        expected, context=direct.PROD_CONTEXT, runner=runner
    )
    _LAUNCH_STAGE = "server_preview_validate"
    direct.validate_preview(
        plan,
        request,
        source_preview,
        expected,
        prod_dry_run,
        context=direct.PROD_CONTEXT,
        identity=identity,
        image_identity_receipt=image_identity,
    )
    _LAUNCH_STAGE = "before_guard_passed"
    return {
        "identity": identity,
        "plan": plan,
        "request": request,
        "preflight": preflight,
        "revalidation": revalidation,
        "image_identity": image_identity,
        "source_preview": source_preview,
        "expected": expected,
        "operation_root": operation_root,
    }


def _assert_probe_markers_absent(operation_root: Path) -> None:
    """Prove the diagnostic neither inherits nor creates a durable intent."""
    paths = (
        direct.jobs_api_guard_path(operation_root, "training"),
        operation_root / "PROD10_DIRECT_V3_CREATE.jsonl",
        hardening.creator_binding_path(operation_root, "training"),
    )
    if any(path.exists() or path.is_symlink() for path in paths):
        raise OperatorFailure("launch_probe_marker_present")


def run_probe(
    packet: dict[str, Any], *, runner: InClusterKubernetesRunner
) -> dict[str, Any]:
    """Localize the pre-guard path without exposing exception details."""
    global _LAUNCH_STAGE
    launch_packet = packet.get("launch_packet")
    if not isinstance(launch_packet, dict):
        raise ValueError("prod10 launch probe source packet changed")
    try:
        _LAUNCH_STAGE = "plan_identity"
        plan = launch_packet.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("prod10 launch probe plan changed")
        operation_root = hardening.training_operation_root(plan)
        _assert_probe_markers_absent(operation_root)
        pre_guard = _pre_guard_launch(launch_packet, runner)
        if pre_guard["operation_root"] != operation_root:
            raise OperatorFailure("launch_probe_operation_root_changed")
        _assert_probe_markers_absent(operation_root)
    except Exception as error:
        error_class, error_code = (
            ("OSError", "oserror")
            if isinstance(error, OSError)
            else ("AssertionError", "assertionerror")
            if isinstance(error, AssertionError)
            else ("JobsError", "jobserror")
            if isinstance(error, JobsError)
            else ("InClusterKubernetesError", "inclusterkuberneteserror")
            if isinstance(error, InClusterKubernetesError)
            else ("OtherException", "other_exception")
        )
        return _seal(
            {
                "schema": MANIFEST_RESULT_SCHEMA,
                "status": "passed",
                "phase": "probe",
                "diagnosis": "exception_localized",
                "error_class": error_class,
                "error_code": f"launch_{_LAUNCH_STAGE}_{error_code}",
                "launch_stage": _LAUNCH_STAGE,
                "preflight_stage": training._PREFLIGHT_STAGE,
                "launch_packet_sha256": launch_packet["sha256"],
                "launch_v2_failure_sha256": launch_v2_failure_binding()["sha256"],
                "inspect_v3_success_sha256": inspect_v3_success_binding()["sha256"],
                "error_path_exported": False,
                "error_errno_exported": False,
                "error_message_exported": False,
                "nested_jobs_created": 0,
                "gpus": 0,
            }
        )
    return _seal(
        {
            "schema": MANIFEST_RESULT_SCHEMA,
            "status": "passed",
            "phase": "probe",
            "diagnosis": "before_guard_passed",
            "launch_stage": _LAUNCH_STAGE,
            "preflight_stage": training._PREFLIGHT_STAGE,
            "launch_packet_sha256": launch_packet["sha256"],
            "launch_v2_failure_sha256": launch_v2_failure_binding()["sha256"],
            "inspect_v3_success_sha256": inspect_v3_success_binding()["sha256"],
            "nested_jobs_created": 0,
            "gpus": 0,
        }
    )


def run_launch(packet: dict[str, Any], *, runner: InClusterKubernetesRunner) -> dict[str, Any]:
    """Consume the sealed direct-v3 preflight and perform the sole GPU POST."""
    pre_guard = _pre_guard_launch(packet, runner)
    identity = pre_guard["identity"]
    plan, request = pre_guard["plan"], pre_guard["request"]
    preflight, revalidation = pre_guard["preflight"], pre_guard["revalidation"]
    image_identity = pre_guard["image_identity"]
    source_preview, expected = pre_guard["source_preview"], pre_guard["expected"]
    operation_root = pre_guard["operation_root"]
    guard = cleanup.JobsApiPrefixGuard(
        context=direct.PROD_CONTEXT,
        namespace=direct.NAMESPACE,
        run_name_prefix=request["name"],
        run_dir=request["run_dir"],
        image=request["image"],
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(expected),
        maximum_seconds=direct.MAXIMUM_SECONDS,
        expected_gpus=request["workers"] * request["gpus_per_worker"],
        armed_path=direct.jobs_api_guard_path(operation_root, "training"),
        binding_path=hardening.creator_binding_path(operation_root, "training"),
        run=runner,
    )
    armed = guard.arm()
    prod_preview = direct.validate_preview(
        plan,
        request,
        source_preview,
        expected,
        direct.server_dry_run(expected, context=direct.PROD_CONTEXT, runner=runner),
        context=direct.PROD_CONTEXT,
        identity=identity,
        image_identity_receipt=image_identity,
    )
    authorization = launch_direct.authorize(
        plan,
        request,
        source_preview,
        expected,
        preflight,
        revalidation,
        dev_preview=packet["dev_preview"],
        prod_preview=prod_preview,
        observer=armed,
        identity=identity,
    )
    token = os.environ.get("FLEET_API_KEY", "")
    if not token or not os.environ.get("WANDB_API_KEY"):
        raise OperatorFailure("launch_credentials_unavailable")
    fresh_capacity = _fresh_capacity_census(request, runner)
    created = launch_direct.create_once(
        operation_root,
        plan,
        request,
        source_preview,
        expected,
        authorization,
        token=token,
        identity=identity,
        runner=runner,
        jobs_factory=Jobs,
        census=fresh_capacity,
        duplicate=packet["duplicate_proof"],
    )
    observed = _observe_created_run(operation_root, plan, created, runner=runner)
    return _seal(
        {
            "schema": LAUNCH_RESULT_SCHEMA,
            "status": "gpu_run_succeeded_and_released",
            "phase": "launch",
            "packet_sha256": packet["sha256"],
            "preflight_result_sha256": preflight["sha256"],
            "preflight_revalidation_sha256": revalidation["sha256"],
            "authorization_sha256": authorization["sha256"],
            "created": created,
            "exact_observer": observed,
            "host_capacity_census_sha256": packet["capacity_census"]["sha256"],
            "fresh_capacity_gate_sha256": created["capacity_gate_sha256"],
            "operator_gpus": 0,
            "created_gpus": created["gpus"],
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
    if phase == "stage":
        result = run_stage(packet, runner=runner)
    elif phase == "manifest":
        result = run_manifest(packet, runner=runner)
    elif phase == "launch":
        result = run_launch(packet, runner=runner)
    elif phase == "inspect":
        result = run_inspect(packet)
    elif phase == "probe":
        result = run_probe(packet, runner=runner)
    else:
        result = run_preflight(packet, runner=runner)
    if phase in {"manifest", "inspect", "probe"}:
        _write_manifest_termination(result)
        return result
    root = (
        hardening.stage_operation_root(result["stage"])
        if phase == "stage"
        else hardening.training_operation_root(packet["plan"])
    )
    result_name = {
        "stage": "STAGE_OPERATOR_RESULT.json",
        "preflight": "PREFLIGHT_OPERATOR_RESULT.json",
        "launch": "LAUNCH_OPERATOR_RESULT.json",
    }[phase]
    result_path = root / result_name
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
