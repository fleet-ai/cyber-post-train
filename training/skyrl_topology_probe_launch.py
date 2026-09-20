"""Evidence-bound authorization for the development SkyRL topology probe."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from cyber_post_train.jobs import digest

from . import skyrl_topology_probe as probe

AUTHORIZATION_SCHEMA = "cyber_skyrl_topology_probe_launch_authorization_v1"
OBSERVER_ARMED_SCHEMA = "cyber_dev_cleanup_observer_armed_v1"
OBSERVER_RESULT_SCHEMA = "cyber_dev_cleanup_observer_result_v1"
CPU_RECEIPT_SCHEMA = "cyber_skyrl_topology_probe_cpu_preflight_v1"


def _seal(value: dict) -> dict:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: dict, schema: str) -> None:
    if value.get("schema") != schema or value != _seal(value):
        raise ValueError("topology probe launch evidence digest changed")


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("topology probe launch evidence timestamp is invalid")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _uuid(value: object) -> UUID:
    if not isinstance(value, str):
        raise ValueError("topology probe launch evidence identity is invalid")
    return UUID(value)


def authorize(
    plan: dict,
    *,
    cpu_result: dict,
    cpu_preview: dict,
    receipt_verify_preview: dict,
    fleetjob_preview: dict,
    fleetjob_observer: dict,
) -> dict:
    """Seal the exact evidence that permits one development GPU probe create."""
    probe._validate(plan)
    plan_sha256 = digest(plan)
    cpu_manifest_sha256 = digest(probe.preflight_job_manifest(plan))
    fleetjob_manifest_sha256 = digest(probe.fleetjob_manifest(plan))

    _validate_seal(cpu_result, OBSERVER_RESULT_SCHEMA)
    receipt = cpu_result.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("topology probe CPU receipt is absent")
    _validate_seal(receipt, CPU_RECEIPT_SCHEMA)
    pod_uids = cpu_result.get("pod_uids")
    if not isinstance(pod_uids, list) or len(pod_uids) != 1:
        raise ValueError("topology probe CPU evidence has the wrong Pod count")
    created = _timestamp(cpu_result.get("created_at"))
    deleted = _timestamp(cpu_result.get("deletion_requested_at"))
    released = _timestamp(cpu_result.get("release_observed_at"))
    _uuid(cpu_result.get("uid"))
    _uuid(pod_uids[0])
    if (
        cpu_result.get("status") != "released"
        or cpu_result.get("kind") != "job"
        or cpu_result.get("name") != probe.PREFLIGHT_NAME
        or cpu_result.get("context") != plan["execution"]["kubernetes_context"]
        or cpu_result.get("namespace") != plan["execution"]["namespace"]
        or cpu_result.get("plan_sha256") != "sha256:" + plan_sha256
        or cpu_result.get("manifest_sha256") != "sha256:" + cpu_manifest_sha256
        or cpu_result.get("expected_gpus") != 0
        or cpu_result.get("peak_gpus") != 0
        or cpu_result.get("active_gpus") != 0
        or cpu_result.get("terminal_status") != "Succeeded"
        or cpu_result.get("exit_codes") != [0]
        or cpu_result.get("restarts") != 0
        or cpu_result.get("image_ids") != [probe.IMAGE]
        or cpu_result.get("target_present") is not False
        or cpu_result.get("pods_present") is not False
        or not created <= deleted <= released
        or receipt.get("status") != "passed"
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("gpus") != 0
        or receipt.get("runtime_user") != {"uid": 1000, "gid": 100}
        or receipt.get("request_sha256")
        != digest(probe.request(plan, fleetjob_transport=True, cpu_preflight=True))
        or receipt.get("fleetjob_manifest_sha256") != fleetjob_manifest_sha256
        or receipt.get("preflight_job_manifest_sha256") != cpu_manifest_sha256
        or receipt.get("task_rows_read") != 0
        or receipt.get("rollout_episodes") != 0
        or receipt.get("optimizer_steps") != 0
        or receipt.get("create_once_output_absent") is not True
    ):
        raise ValueError("topology probe CPU gate was not accepted and released")

    _validate_seal(cpu_preview, probe.PREFLIGHT_PREVIEW_SCHEMA)
    if (
        cpu_preview.get("status") != "passed"
        or cpu_preview.get("plan_sha256") != plan_sha256
        or cpu_preview.get("manifest_sha256") != cpu_manifest_sha256
        or cpu_preview.get("name") != probe.PREFLIGHT_NAME
        or cpu_preview.get("gpus") != 0
        or cpu_preview.get("runtime_user") != {"uid": 1000, "gid": 100}
        or cpu_preview.get("submitted") is not False
    ):
        raise ValueError("topology probe CPU preview was not accepted")

    _validate_seal(fleetjob_preview, probe.FLEETJOB_PREVIEW_SCHEMA)
    if (
        fleetjob_preview.get("status") != "passed"
        or fleetjob_preview.get("plan_sha256") != plan_sha256
        or fleetjob_preview.get("manifest_sha256") != fleetjob_manifest_sha256
        or fleetjob_preview.get("name") != plan["run_name"]
        or fleetjob_preview.get("gpu_nodes") != 1
        or fleetjob_preview.get("gpus") != 8
        or fleetjob_preview.get("runtime_user") != {"uid": 1000, "gid": 100}
        or fleetjob_preview.get("submitted") is not False
    ):
        raise ValueError("topology probe FleetJob preview was not accepted")

    receipt_manifest_sha256 = digest(probe.receipt_verify_job_manifest(plan))
    _validate_seal(receipt_verify_preview, probe.RECEIPT_VERIFY_PREVIEW_SCHEMA)
    if (
        receipt_verify_preview.get("status") != "passed"
        or receipt_verify_preview.get("plan_sha256") != plan_sha256
        or receipt_verify_preview.get("manifest_sha256") != receipt_manifest_sha256
        or receipt_verify_preview.get("name") != probe.RECEIPT_VERIFY_NAME
        or receipt_verify_preview.get("gpu_nodes") != 0
        or receipt_verify_preview.get("gpus") != 0
        or receipt_verify_preview.get("runtime_user") != {"uid": 1000, "gid": 100}
        or receipt_verify_preview.get("submitted") is not False
    ):
        raise ValueError("topology probe receipt-verifier preview was not accepted")

    _validate_seal(fleetjob_observer, OBSERVER_ARMED_SCHEMA)
    observer_pid = fleetjob_observer.get("observer_pid")
    if (
        fleetjob_observer.get("status") != "armed"
        or fleetjob_observer.get("context") != plan["execution"]["kubernetes_context"]
        or fleetjob_observer.get("namespace") != plan["execution"]["namespace"]
        or fleetjob_observer.get("kind") != "fleetjob"
        or fleetjob_observer.get("name") != plan["run_name"]
        or fleetjob_observer.get("maximum_seconds") != 1800
        or fleetjob_observer.get("expected_gpus") != 8
        or fleetjob_observer.get("plan_sha256") != "sha256:" + plan_sha256
        or fleetjob_observer.get("manifest_sha256") != "sha256:" + fleetjob_manifest_sha256
        or type(observer_pid) is not int
        or observer_pid < 1
    ):
        raise ValueError("topology probe cleanup observer is not exactly armed")
    _timestamp(fleetjob_observer.get("armed_at"))

    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "plan_sha256": plan_sha256,
            "fleetjob_manifest_sha256": fleetjob_manifest_sha256,
            "cpu_result": cpu_result,
            "cpu_preview": cpu_preview,
            "receipt_verify_preview": receipt_verify_preview,
            "fleetjob_preview": fleetjob_preview,
            "fleetjob_observer": fleetjob_observer,
        }
    )


def validate(plan: dict, value: dict) -> None:
    """Rebuild the authorization so nested evidence cannot be removed or swapped."""
    if value != authorize(
        plan,
        cpu_result=value.get("cpu_result", {}),
        cpu_preview=value.get("cpu_preview", {}),
        receipt_verify_preview=value.get("receipt_verify_preview", {}),
        fleetjob_preview=value.get("fleetjob_preview", {}),
        fleetjob_observer=value.get("fleetjob_observer", {}),
    ):
        raise ValueError("topology probe launch authorization changed")
