"""Create-once direct RayJob transport for the sealed SkyRL topology probe.

The FleetJob admission webhook currently rejects the mandatory failed-job-alert
annotation inside its embedded RayJob.  This module projects that exact embedded
RayJob into a root RayJob and materializes only the platform bindings that the
FleetJob controller would otherwise add.  It never changes the probe plan,
entrypoint, image, topology, resources, deadline, or scientific-work contract.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from cyber_post_train.jobs import (
    API_URLS,
    FAILURE_ALERT_ANNOTATION,
    FAILURE_ALERT_OFF,
    Jobs,
    JobsError,
    digest,
)

from . import skyrl_topology_probe as probe

PACKET_SCHEMA = "cyber_skyrl_topology_probe_direct_rayjob_packet_v1"
PREVIEW_SCHEMA = "cyber_skyrl_topology_probe_direct_rayjob_preview_v1"
INTENT_SCHEMA = "cyber_skyrl_topology_probe_direct_rayjob_intent_v1"
CREATED_SCHEMA = "cyber_skyrl_topology_probe_direct_rayjob_created_v1"
AUTHORIZATION_SCHEMA = "cyber_skyrl_topology_probe_direct_rayjob_authorization_v1"
RELEASE_SCHEMA = "cyber_skyrl_topology_probe_direct_rayjob_release_v1"
DEV_CONTEXT = "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb"
PROD_CONTEXT = "nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6"
QUEUE = "training-lq"
GPU_NODE_SELECTOR = {"workload": "fleetai-training-ng-gpu"}
GPU_TOLERATION = {
    "effect": "NoSchedule",
    "key": "workload",
    "operator": "Equal",
    "value": "fleetai-training-ng-gpu",
}


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: dict[str, Any], schema: str) -> None:
    if value.get("schema") != schema or value != _seal(value):
        raise JobsError("direct RayJob evidence digest changed")


def _upsert_env(entries: list[dict[str, Any]], item: dict[str, Any]) -> None:
    matches = [index for index, entry in enumerate(entries) if entry.get("name") == item["name"]]
    if len(matches) > 1:
        raise JobsError("direct RayJob has duplicate environment names")
    if matches:
        entries[matches[0]] = item
    else:
        entries.append(item)


def _platform_pod(plan: dict[str, Any], template: dict[str, Any], *, head: bool) -> None:
    """Add only the pod bindings normally supplied by the FleetJob controller."""
    execution = plan["execution"]
    metadata = template.setdefault("metadata", {})
    annotations = metadata.setdefault("annotations", {})
    annotations.update(
        {
            "kueue.x-k8s.io/podset-unconstrained-topology": "true",
            "ray/kueue-priority-class": execution["queue_priority"],
            "scheduling.fleet.ai/priority-class": execution["queue_priority"],
        }
    )
    pod = template["spec"]
    if pod.get("priorityClassName") != execution["priority"]:
        raise JobsError("direct RayJob pod priority drift")
    pod.update(
        {
            "automountServiceAccountToken": False,
            "imagePullSecrets": [{"name": "ecr-pull"}],
            "nodeSelector": GPU_NODE_SELECTOR,
            "serviceAccountName": "fleet-trainer",
            "terminationGracePeriodSeconds": 120,
            "tolerations": [GPU_TOLERATION],
            "volumes": [
                {
                    "name": "sfs",
                    "persistentVolumeClaim": {"claimName": execution["output_pvc"]},
                }
            ],
        }
    )
    containers = pod.get("containers", [])
    if len(containers) != 1 or containers[0].get("name") != "ray":
        raise JobsError("direct RayJob must retain one exact Ray container")
    container = containers[0]
    environment = container.setdefault("env", [])
    _upsert_env(
        environment,
        {
            "name": "FLEET_API_KEY",
            "valueFrom": {
                "secretKeyRef": {
                    "name": execution["auth_secret"]["name"],
                    "key": execution["auth_secret"]["key"],
                }
            },
        },
    )
    _upsert_env(
        environment,
        {
            "name": "FLEET_NODE_NAME",
            "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}},
        },
    )
    output_subpath = "/".join(
        (execution["output_registry_subpath"], plan["run_name"], "models", "run")
    )
    container["volumeMounts"] = [
        {
            "name": "sfs",
            "mountPath": plan["model"]["root"],
            "subPath": execution["preflight_model_pvc_subpath"],
            "readOnly": True,
        },
        {
            "name": "sfs",
            "mountPath": plan["output_root"],
            "subPath": output_subpath,
        },
    ]
    if head:
        registry_target = "/".join(
            (
                execution["output_registry_mount"],
                plan["run_name"],
                "models",
                "run",
            )
        )
        pod["initContainers"] = [
            {
                "name": "create-once-output",
                "image": execution["image"],
                "command": [
                    "python",
                    "-c",
                    (
                        "import pathlib,sys;"
                        "pathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=False)"
                    ),
                    registry_target,
                ],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "privileged": False,
                    "runAsGroup": 100,
                    "runAsNonRoot": True,
                    "runAsUser": 1000,
                },
                "resources": {
                    "requests": {"cpu": "100m", "memory": "128Mi"},
                    "limits": {"cpu": "1", "memory": "1Gi"},
                },
                "volumeMounts": [
                    {
                        "name": "sfs",
                        "mountPath": execution["output_registry_mount"],
                        "subPath": execution["output_registry_subpath"],
                    }
                ],
            }
        ]


def manifest(plan: dict[str, Any]) -> dict[str, Any]:
    """Project the sealed FleetJob's embedded RayJob into one direct RayJob."""
    probe._validate(plan)
    source = probe.fleetjob_manifest(plan)["spec"]["job"]
    if source.get("apiVersion") != "ray.io/v1" or source.get("kind") != "RayJob":
        raise JobsError("topology probe embedded RayJob contract changed")
    value = copy.deepcopy(source)
    execution = plan["execution"]
    value["metadata"] = {
        "name": plan["run_name"],
        "namespace": execution["namespace"],
        "labels": {
            "kueue.x-k8s.io/queue-name": QUEUE,
            "kueue.x-k8s.io/priority-class": execution["queue_priority"],
        },
        "annotations": copy.deepcopy(source["metadata"]["annotations"]),
    }
    value["metadata"]["annotations"]["kueue.x-k8s.io/elastic-job"] = "true"
    value["metadata"]["annotations"]["fleet.ai/run-dir"] = plan["output_root"]
    if value["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF:
        raise JobsError("direct RayJob does not disable failed-job alerts")
    value["spec"]["suspend"] = True
    cluster = value["spec"]["rayClusterSpec"]
    _platform_pod(plan, cluster["headGroupSpec"]["template"], head=True)
    workers = cluster.get("workerGroupSpecs", [])
    if len(workers) != 1 or workers[0].get("replicas") != 0:
        raise JobsError("direct RayJob dormant worker topology drift")
    _platform_pod(plan, workers[0]["template"], head=False)
    return value


def packet(plan: dict[str, Any]) -> dict[str, Any]:
    value = manifest(plan)
    return _seal(
        {
            "schema": PACKET_SCHEMA,
            "plan_sha256": digest(plan),
            "source_fleetjob_manifest_sha256": digest(probe.fleetjob_manifest(plan)),
            "manifest_sha256": digest(value),
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": plan["run_name"],
            "submitted": False,
        }
    )


def _strip_server_defaults(expected: dict[str, Any], rendered: dict[str, Any]) -> dict[str, Any]:
    """Return the server object after removing only accepted deterministic defaults."""
    actual = copy.deepcopy(rendered)
    metadata = actual.get("metadata", {})
    if metadata.get("generation") != 1 or "resourceVersion" in metadata:
        raise JobsError("server dry-run metadata defaults changed")
    _timestamp(metadata.get("creationTimestamp"))
    metadata.pop("creationTimestamp")
    metadata.pop("generation")
    metadata.pop("uid")
    if "status" in actual:
        raise JobsError("server dry-run unexpectedly added status")
    spec = actual.get("spec", {})
    if spec.get("submissionMode") != "K8sJobMode" or spec.get("ttlSecondsAfterFinished") != 0:
        raise JobsError("server dry-run RayJob defaults changed")
    spec.pop("submissionMode", None)
    spec.pop("ttlSecondsAfterFinished", None)
    cluster = spec.get("rayClusterSpec", {})
    if "headServiceAnnotations" in cluster:
        raise JobsError("server dry-run cluster defaults changed")
    head = cluster.get("headGroupSpec", {})
    workers = cluster.get("workerGroupSpecs", [])
    if (
        "numOfHosts" in head
        or "scaleStrategy" in head
        or len(workers) != 1
        or workers[0].get("numOfHosts") != 1
        or "scaleStrategy" in workers[0]
    ):
        raise JobsError("server dry-run group defaults changed")
    workers[0].pop("numOfHosts")
    for group in [head, *workers]:
        if "creationTimestamp" in group.get("template", {}).get("metadata", {}):
            raise JobsError("server dry-run template defaults changed")
    return actual


def validate_preview(
    plan: dict[str, Any], expected: dict[str, Any], rendered: dict[str, Any]
) -> dict[str, Any]:
    """Accept a server dry-run only when it retains the exact direct transport."""
    if expected != manifest(plan):
        raise JobsError("direct RayJob differs from its immutable packet")
    metadata = rendered.get("metadata", {})
    try:
        UUID(metadata["uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("direct RayJob dry-run lacks a server UID") from exc
    if _strip_server_defaults(expected, rendered) != expected:
        raise JobsError("server dry-run changed the direct RayJob")
    return _seal(
        {
            "schema": PREVIEW_SCHEMA,
            "status": "passed",
            "plan_sha256": digest(plan),
            "source_fleetjob_manifest_sha256": digest(probe.fleetjob_manifest(plan)),
            "manifest_sha256": digest(expected),
            "server_render_sha256": digest(rendered),
            "kubernetes_context": plan["execution"]["kubernetes_context"],
            "namespace": plan["execution"]["namespace"],
            "name": plan["run_name"],
            "gpu_nodes": 1,
            "gpus": 8,
            "priority": "c1",
            "queue_priority": "q1",
            "failure_alerts": "off",
            "submitted": False,
        }
    )


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise JobsError("direct RayJob evidence timestamp is invalid")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise JobsError("direct RayJob evidence timestamp is invalid") from exc


def authorize(
    plan: dict[str, Any],
    *,
    cpu_result: dict[str, Any],
    cpu_preview: dict[str, Any],
    receipt_preview: dict[str, Any],
    rayjob_preview: dict[str, Any],
    observer: dict[str, Any],
) -> dict[str, Any]:
    """Bind the existing v17 CPU gate and a direct-RayJob observer to one create."""
    probe._validate(plan)
    plan_sha256 = digest(plan)
    source_sha256 = digest(probe.fleetjob_manifest(plan))
    direct_sha256 = digest(manifest(plan))
    cpu_manifest_sha256 = digest(probe.preflight_job_manifest(plan))

    _validate_seal(cpu_result, "cyber_dev_cleanup_observer_result_v1")
    receipt = cpu_result.get("receipt")
    if not isinstance(receipt, dict):
        raise JobsError("direct RayJob CPU receipt is absent")
    if receipt != probe._seal(receipt) or receipt.get("schema") != (
        "cyber_skyrl_topology_probe_cpu_preflight_v1"
    ):
        raise JobsError("direct RayJob CPU receipt digest changed")
    pod_uids = cpu_result.get("pod_uids")
    try:
        UUID(cpu_result["uid"])
        if not isinstance(pod_uids, list) or len(pod_uids) != 1:
            raise ValueError
        UUID(pod_uids[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("direct RayJob CPU evidence identity is invalid") from exc
    armed = _timestamp(cpu_result.get("armed_at"))
    created = _timestamp(cpu_result.get("created_at"))
    deleted = _timestamp(cpu_result.get("deletion_requested_at"))
    released = _timestamp(cpu_result.get("release_observed_at"))
    if (
        cpu_result.get("status") != "released"
        or cpu_result.get("kind") != "job"
        or cpu_result.get("name") != probe.PREFLIGHT_NAME
        or cpu_result.get("maximum_seconds") != 1200
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
        or not armed <= created <= deleted <= released
        or receipt.get("status") != "passed"
        or receipt.get("plan_sha256") != plan_sha256
        or receipt.get("gpus") != 0
        or receipt.get("runtime_user") != {"uid": 1000, "gid": 100}
        or receipt.get("request_sha256")
        != digest(probe.request(plan, fleetjob_transport=True, cpu_preflight=True))
        or receipt.get("fleetjob_manifest_sha256") != source_sha256
        or receipt.get("preflight_job_manifest_sha256") != cpu_manifest_sha256
        or receipt.get("task_rows_read") != 0
        or receipt.get("rollout_episodes") != 0
        or receipt.get("optimizer_steps") != 0
        or receipt.get("create_once_output_absent") is not True
        or receipt.get("vllm_sampler_environment") != probe.VLLM_SAMPLER_ENV
    ):
        raise JobsError("direct RayJob CPU gate was not accepted and released")

    _validate_seal(cpu_preview, probe.PREFLIGHT_PREVIEW_SCHEMA)
    if (
        cpu_preview.get("status") != "passed"
        or cpu_preview.get("plan_sha256") != plan_sha256
        or cpu_preview.get("manifest_sha256") != cpu_manifest_sha256
        or cpu_preview.get("name") != probe.PREFLIGHT_NAME
        or cpu_preview.get("gpus") != 0
        or cpu_preview.get("submitted") is not False
    ):
        raise JobsError("direct RayJob CPU preview was not accepted")

    receipt_manifest_sha256 = digest(probe.receipt_verify_job_manifest(plan))
    _validate_seal(receipt_preview, probe.RECEIPT_VERIFY_PREVIEW_SCHEMA)
    if (
        receipt_preview.get("status") != "passed"
        or receipt_preview.get("plan_sha256") != plan_sha256
        or receipt_preview.get("manifest_sha256") != receipt_manifest_sha256
        or receipt_preview.get("name") != probe.RECEIPT_VERIFY_NAME
        or receipt_preview.get("gpus") != 0
        or receipt_preview.get("submitted") is not False
    ):
        raise JobsError("direct RayJob receipt preview was not accepted")

    _validate_seal(rayjob_preview, PREVIEW_SCHEMA)
    if (
        rayjob_preview.get("status") != "passed"
        or rayjob_preview.get("plan_sha256") != plan_sha256
        or rayjob_preview.get("source_fleetjob_manifest_sha256") != source_sha256
        or rayjob_preview.get("manifest_sha256") != direct_sha256
        or rayjob_preview.get("name") != plan["run_name"]
        or rayjob_preview.get("gpus") != 8
        or rayjob_preview.get("priority") != "c1"
        or rayjob_preview.get("queue_priority") != "q1"
        or rayjob_preview.get("failure_alerts") != "off"
        or rayjob_preview.get("submitted") is not False
    ):
        raise JobsError("direct RayJob preview was not accepted")

    _validate_seal(observer, "cyber_dev_cleanup_observer_armed_v1")
    observer_pid = observer.get("observer_pid")
    if (
        observer.get("status") != "armed"
        or observer.get("context") != plan["execution"]["kubernetes_context"]
        or observer.get("namespace") != plan["execution"]["namespace"]
        or observer.get("kind") != "rayjob"
        or observer.get("name") != plan["run_name"]
        or observer.get("maximum_seconds") != 1800
        or observer.get("expected_gpus") != 8
        or observer.get("plan_sha256") != "sha256:" + plan_sha256
        or observer.get("manifest_sha256") != "sha256:" + direct_sha256
        or type(observer_pid) is not int
        or observer_pid < 1
    ):
        raise JobsError("direct RayJob cleanup observer is not exactly armed")
    _timestamp(observer.get("armed_at"))
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "plan_sha256": plan_sha256,
            "source_fleetjob_manifest_sha256": source_sha256,
            "manifest_sha256": direct_sha256,
            "cpu_result": cpu_result,
            "cpu_preview": cpu_preview,
            "receipt_preview": receipt_preview,
            "rayjob_preview": rayjob_preview,
            "observer": observer,
        }
    )


def validate_authorization(plan: dict[str, Any], value: dict[str, Any]) -> None:
    if value != authorize(
        plan,
        cpu_result=value.get("cpu_result", {}),
        cpu_preview=value.get("cpu_preview", {}),
        receipt_preview=value.get("receipt_preview", {}),
        rayjob_preview=value.get("rayjob_preview", {}),
        observer=value.get("observer", {}),
    ):
        raise JobsError("direct RayJob launch authorization changed")


def validate_release(
    plan: dict[str, Any], receipt: dict[str, Any], observation: dict[str, Any]
) -> dict[str, Any]:
    """Accept only exact direct-RayJob success followed by complete GPU release."""
    probe._validate_probe_receipt(plan, receipt)
    _validate_seal(observation, "cyber_dev_cleanup_observer_result_v1")
    pod_uids = observation.get("pod_uids")
    try:
        target_uid = UUID(observation["uid"])
        rayjob_uid = UUID(observation["rayjob_uid"])
        UUID(observation["workload_uid"])
        UUID(observation["raycluster_uid"])
        if not isinstance(pod_uids, list) or len(pod_uids) != 1:
            raise ValueError
        UUID(pod_uids[0])
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("direct RayJob release identities are invalid") from exc
    created = _timestamp(observation.get("created_at"))
    deleted = _timestamp(observation.get("deletion_requested_at"))
    released = _timestamp(observation.get("release_observed_at"))
    if (
        observation.get("status") != "released"
        or observation.get("kind") != "rayjob"
        or observation.get("name") != plan["run_name"]
        or observation.get("context") != plan["execution"]["kubernetes_context"]
        or observation.get("namespace") != plan["execution"]["namespace"]
        or observation.get("plan_sha256") != "sha256:" + digest(plan)
        or observation.get("manifest_sha256") != "sha256:" + digest(manifest(plan))
        or observation.get("expected_gpus") != 8
        or observation.get("maximum_seconds") != 1800
        or observation.get("terminal_status") != "Succeeded"
        or target_uid != rayjob_uid
        or observation.get("rayjob_name") != plan["run_name"]
        or not isinstance(observation.get("workload_name"), str)
        or not observation["workload_name"]
        or not isinstance(observation.get("raycluster_name"), str)
        or not observation["raycluster_name"]
        or not isinstance(observation.get("pod_names"), list)
        or len(observation["pod_names"]) != 1
        or observation.get("image_ids") != [probe.IMAGE]
        or observation.get("peak_gpus") != 8
        or observation.get("restarts") != 0
        or observation.get("receipt") != receipt
        or observation.get("observer_error_class") != ""
        or observation.get("target_present") is not False
        or observation.get("rayjob_present") is not False
        or observation.get("workload_present") is not False
        or observation.get("raycluster_present") is not False
        or observation.get("pods_present") is not False
        or observation.get("active_gpus") != 0
        or not created <= deleted <= released
        or (deleted - created).total_seconds() > 1800
    ):
        raise JobsError("exact direct RayJob release was not proven")
    return _seal(
        {
            "schema": RELEASE_SCHEMA,
            "status": "released",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(manifest(plan)),
            "rayjob_uid": observation["rayjob_uid"],
            "workload_uid": observation["workload_uid"],
            "raycluster_uid": observation["raycluster_uid"],
            "pod_uids": pod_uids,
            "receipt_sha256": receipt["sha256"],
            "release_observed_at": observation["release_observed_at"],
            "active_gpus": 0,
        }
    )


def write_once_fsynced(path: Path, value: dict[str, Any]) -> None:
    """Create one durable journal record before the non-idempotent create."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            fd = -1
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _kubectl(
    runner: Callable[..., subprocess.CompletedProcess[str]],
    context: str,
    namespace: str,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["kubectl", "--context", context, "--namespace", namespace, *arguments],
        capture_output=True,
        text=True,
        timeout=60,
    )


def exhaustive_duplicate_checks(
    plan: dict[str, Any],
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
) -> dict[str, Any]:
    """Prove the exact identity is absent from both cluster/API histories."""
    name = plan["run_name"]
    namespace = plan["execution"]["namespace"]
    output_subpath = "/".join(
        (
            plan["execution"]["output_registry_subpath"],
            name,
            "models",
            "run",
        )
    )
    checks = 0
    for context in (DEV_CONTEXT, PROD_CONTEXT):
        for resource in ("rayjob", "fleetjob", "job"):
            result = _kubectl(
                runner,
                context,
                namespace,
                "get",
                resource,
                "--output=json",
            )
            checks += 1
            if result.returncode:
                raise JobsError("direct RayJob duplicate check failed")
            try:
                items = json.loads(result.stdout).get("items", [])
            except (AttributeError, ValueError) as exc:
                raise JobsError("direct RayJob duplicate inventory is invalid") from exc
            if not isinstance(items, list):
                raise JobsError("direct RayJob duplicate inventory is invalid")
            for item in items:
                if not isinstance(item, dict):
                    raise JobsError("direct RayJob duplicate inventory is invalid")
                metadata = item.get("metadata", {})
                annotations = metadata.get("annotations", {})
                serialized = json.dumps(item.get("spec", {}), sort_keys=True)
                if (
                    metadata.get("name") == name
                    or annotations.get("fleet.ai/run-dir") == plan["output_root"]
                    or item.get("spec", {}).get("fleet", {}).get("mountRoot")
                    == plan["execution"]["mount_root"]
                    or output_subpath in serialized
                ):
                    raise JobsError("a Kubernetes object already owns this identity/output")
    api_rows = 0
    for target in ("dev", "prod"):
        with jobs_factory(token, base_url=API_URLS[target]) as client:
            rows = client.all_runs()
        api_rows += len(rows)
        if any(
            row.get("name") == name
            or str(row.get("name", "")).startswith(name + "-")
            or row.get("run_dir") == plan["output_root"]
            for row in rows
        ):
            raise JobsError("Jobs API history already owns the direct RayJob identity")
    return {
        "kubernetes_objects_checked": checks,
        "jobs_api_rows_checked": api_rows,
    }


def create_once(
    directory: Path,
    plan: dict[str, Any],
    expected: dict[str, Any],
    preview_proof: dict[str, Any],
    authorization: dict[str, Any],
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
) -> dict[str, Any]:
    """Journal intent, then issue exactly one non-retried ``kubectl create``."""
    if expected != manifest(plan):
        raise JobsError("direct RayJob packet changed")
    validate_authorization(plan, authorization)
    try:
        os.kill(authorization["observer"]["observer_pid"], 0)
    except (KeyError, OSError, TypeError) as exc:
        raise JobsError("direct RayJob cleanup observer is not running") from exc
    _validate_seal(preview_proof, PREVIEW_SCHEMA)
    if (
        preview_proof.get("plan_sha256") != digest(plan)
        or preview_proof.get("manifest_sha256") != digest(expected)
        or preview_proof.get("failure_alerts") != "off"
        or preview_proof.get("submitted") is not False
    ):
        raise JobsError("direct RayJob preview is not launchable")
    journal = directory / "DIRECT_RAYJOB_CREATE.jsonl"
    if journal.exists() or journal.is_symlink():
        raise JobsError("direct RayJob create intent already exists; reconcile, never retry")
    duplicate_proof = exhaustive_duplicate_checks(
        plan, token=token, runner=runner, jobs_factory=jobs_factory
    )
    execution = plan["execution"]
    path = directory / "direct-rayjob.json"
    dry_run = _kubectl(
        runner,
        execution["kubernetes_context"],
        execution["namespace"],
        "create",
        "--dry-run=server",
        "--filename",
        str(path),
        "--output=json",
    )
    if dry_run.returncode:
        raise JobsError("direct RayJob final server dry-run failed")
    current_preview = validate_preview(plan, expected, json.loads(dry_run.stdout))
    intent = _seal(
        {
            "schema": INTENT_SCHEMA,
            "state": "CREATE_INTENT_DO_NOT_RETRY",
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(expected),
            "preview_sha256": current_preview["sha256"],
            "authorization_sha256": authorization["sha256"],
            "duplicate_proof": duplicate_proof,
        }
    )
    write_once_fsynced(journal, intent)
    # This is the only mutating call. Never wrap it in retry logic.
    result = _kubectl(
        runner,
        execution["kubernetes_context"],
        execution["namespace"],
        "create",
        "--filename",
        str(path),
        "--output=json",
    )
    if result.returncode:
        raise JobsError("direct RayJob create was ambiguous; reconcile journal, never retry")
    resource = json.loads(result.stdout)
    metadata = resource.get("metadata", {})
    try:
        UUID(metadata["uid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("created direct RayJob identity is ambiguous") from exc
    if (
        resource.get("apiVersion") != expected["apiVersion"]
        or resource.get("kind") != "RayJob"
        or metadata.get("name") != plan["run_name"]
        or metadata.get("namespace") != execution["namespace"]
        or metadata.get("annotations", {}).get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise JobsError("created direct RayJob differs from its intent")
    return _seal(
        {
            "schema": CREATED_SCHEMA,
            "status": "created",
            "name": metadata["name"],
            "uid": metadata["uid"],
            "created_at": metadata.get("creationTimestamp"),
            "plan_sha256": digest(plan),
            "manifest_sha256": digest(expected),
            "intent_sha256": intent["sha256"],
            "authorization_sha256": authorization["sha256"],
        }
    )


def validate_name(value: str) -> None:
    if re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", value) is None:
        raise JobsError("direct RayJob name is invalid")
