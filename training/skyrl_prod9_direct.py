"""Fresh prod9-only RayJob renderer and bounded one-create rail.

The historical prod8 direct rail deliberately rejects prod9's fresh runtime
schema.  This module owns the replacement *rendering* boundary: it makes the
exact-image CPU preflight Job and the one-node RayJob available for review and
server preview, with the root failed-job-alert opt-out already present.  It has
the exact zero-GPU stage/preflight receipts, fresh preview/absence evidence,
root alert-off proof, and project capacity proof before it can perform one
non-retry create.  Nothing in this module invokes that create by itself.
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
from uuid import NAMESPACE_URL, UUID, uuid5

import yaml

from cyber_post_train.jobs import (
    API_URLS,
    FAILURE_ALERT_ANNOTATION,
    FAILURE_ALERT_OFF,
    Jobs,
    JobsError,
    digest,
)

from . import skyrl_prod9_hardening as hardening
from . import skyrl_prod9_training as training
from . import skyrl_reward_rayjob as historical

PACKET_SCHEMA = "cyber_skyrl_prod9_direct_rayjob_packet_v1"
PREVIEW_SCHEMA = "cyber_skyrl_prod9_direct_rayjob_preview_v1"
CPU_JOB_SCHEMA = "cyber_skyrl_prod9_cpu_preflight_job_v1"
CPU_PREVIEW_SCHEMA = "cyber_skyrl_prod9_cpu_job_preview_v1"
STAGE_AUTHORIZATION_SCHEMA = "cyber_skyrl_prod9_stage_authorization_v1"
PREFLIGHT_AUTHORIZATION_SCHEMA = "cyber_skyrl_prod9_preflight_authorization_v1"
AUTHORIZATION_SCHEMA = "cyber_skyrl_prod9_direct_authorization_v1"
CPU_CREATED_SCHEMA = "cyber_skyrl_prod9_cpu_created_v1"
CREATED_SCHEMA = "cyber_skyrl_prod9_direct_created_v1"
RUNTIME_UID = 1000
RUNTIME_GID = 100
NAMESPACE = historical.NAMESPACE
DEV_CONTEXT = historical.DEV_CONTEXT
PROD_CONTEXT = historical.PROD_CONTEXT
PVC = "sfs-shared"
PREFLIGHT_RECEIPT = "/dev/termination-log"
MAXIMUM_SECONDS = historical.MAXIMUM_SECONDS
CPU_MAXIMUM_SECONDS = 1800
EVIDENCE_MAX_AGE_SECONDS = 300


def _identity(plan: dict[str, Any], identity: historical.RailIdentity) -> historical.RailIdentity:
    """Bind public prod9 fields to the reviewed, fresh create-once identity."""
    if plan.get("schema") != training.SCHEMA:
        raise JobsError("prod9 direct rail requires the fresh runtime schema")
    return historical._identity_for_plan(plan, identity)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": "sha256:" + digest(body)}


def _validate_seal(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobsError("prod9 direct evidence is not an object")
    if value.get("schema") != schema or value != _seal(value):
        raise JobsError("prod9 direct evidence digest changed")
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise JobsError("prod9 evidence timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JobsError("prod9 evidence timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise JobsError("prod9 evidence timestamp has no timezone")
    return parsed.astimezone(UTC)


def _fresh_at(value: object, *, maximum_age: int = EVIDENCE_MAX_AGE_SECONDS) -> None:
    age = (datetime.now(UTC) - _timestamp(value)).total_seconds()
    if not 0 <= age <= maximum_age:
        raise JobsError("prod9 evidence is stale or future-dated")


def _observer_pid(
    observer: object,
    *,
    kind: str,
    name: str,
    plan_sha256: str,
    manifest_sha256: str,
    gpus: int,
    maximum_seconds: int,
) -> dict[str, Any]:
    """Accept only a live, exact-name production observer before one create."""
    value = _validate_seal(observer, "cyber_direct_cleanup_observer_armed_v1")
    pid = value.get("observer_pid")
    creator_binding = value.get("creator_binding_path")
    if (
        value.get("status") != "armed"
        or value.get("context") != PROD_CONTEXT
        or value.get("namespace") != NAMESPACE
        or value.get("kind") != kind
        or value.get("name") != name
        or value.get("maximum_seconds") != maximum_seconds
        or value.get("expected_gpus") != gpus
        or value.get("plan_sha256") != plan_sha256
        or value.get("manifest_sha256") != manifest_sha256
        or type(pid) is not int
        or pid < 1
        or not isinstance(creator_binding, str)
        or not Path(creator_binding).is_absolute()
        or Path(creator_binding).exists()
        or Path(creator_binding).is_symlink()
    ):
        raise JobsError("prod9 exact cleanup observer binding changed")
    _fresh_at(value.get("armed_at"))
    try:
        os.kill(pid, 0)
    except OSError as exc:
        raise JobsError("prod9 cleanup observer is not running") from exc
    return value


def _operation_root(observer: dict[str, Any], *, expected_root: Path, purpose: str) -> Path:
    """Require the observer handoff inside the identity-derived global root."""
    binding = observer.get("creator_binding_path")
    if not isinstance(binding, str):
        raise JobsError("prod9 cleanup observer lacks its create handoff path")
    path = Path(binding)
    parent = path.parent
    expected_binding = hardening.creator_binding_path(expected_root, purpose)
    if (
        not path.is_absolute()
        or path != expected_binding
        or expected_root.parent != hardening.CREATE_ONCE_ROOT
        or hardening.CREATE_ONCE_ROOT.is_symlink()
        or not hardening.CREATE_ONCE_ROOT.is_dir()
        or hardening.CREATE_ONCE_ROOT.resolve() != hardening.CREATE_ONCE_ROOT
        or parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve() != parent
    ):
        raise JobsError("prod9 operation root is not one canonical durable directory")
    return parent


def _publish_creator_binding(
    observer: dict[str, Any],
    *,
    kind: str,
    name: str,
    plan_sha256: str,
    manifest_sha256: str,
    uid: str,
) -> dict[str, Any]:
    value = _seal(
        {
            "schema": "cyber_direct_cleanup_creator_binding_v1",
            "status": "created_once",
            "context": PROD_CONTEXT,
            "namespace": NAMESPACE,
            "kind": kind,
            "name": name,
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "uid": uid,
        }
    )
    _write_once_fsynced(Path(observer["creator_binding_path"]), value)
    return value


def _run_id(plan: dict[str, Any], identity: historical.RailIdentity) -> str:
    _identity(plan, identity)
    return str(uuid5(NAMESPACE_URL, f"fleet-direct-rayjob:{identity.run_name}:{digest(plan)}"))


def _source(preview: dict[str, Any]) -> dict[str, Any]:
    if preview.get("warnings") or set(preview) != {"name", "warnings", "manifest_yaml"}:
        raise JobsError("prod9 Jobs preview reported warnings or changed shape")
    try:
        value = yaml.safe_load(preview["manifest_yaml"])
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise JobsError("prod9 Jobs preview is malformed") from exc
    if not isinstance(value, dict):
        raise JobsError("prod9 Jobs preview is not an object")
    return value


def _replace_env(container: dict[str, Any], name: str, value: str) -> None:
    entries = container.get("env", [])
    matches = [entry for entry in entries if entry.get("name") == name]
    if len(matches) != 1 or set(matches[0]) != {"name", "value"}:
        raise JobsError("prod9 Jobs preview environment changed")
    matches[0]["value"] = value


def _runtime_context() -> dict[str, Any]:
    return {
        "allowPrivilegeEscalation": False,
        "privileged": False,
        "runAsGroup": RUNTIME_GID,
        "runAsNonRoot": True,
        "runAsUser": RUNTIME_UID,
    }


def manifest(
    plan: dict[str, Any],
    request: dict[str, Any],
    preview: dict[str, Any],
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Project a fresh Jobs preview into one root-annotated prod9 RayJob."""
    bound = _identity(plan, identity)
    if training.job_request(plan) != request:
        raise JobsError("prod9 plan/request identity changed")
    source = _source(preview)
    placeholder = bound.run_name + "-00000000"
    metadata = source.get("metadata", {})
    labels, annotations = metadata.get("labels", {}), metadata.get("annotations", {})
    if (
        preview.get("name") != placeholder
        or source.get("apiVersion") != "ray.io/v1"
        or source.get("kind") != "RayJob"
        or metadata.get("name") != placeholder
        or metadata.get("namespace") != NAMESPACE
        or labels.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or labels.get("fleet.ai/run-name") != bound.run_name
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or annotations.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or annotations.get("fleet.ai/run-dir") != plan["output_root"]
        or annotations.get("fleet.ai/job-image") != request["image"]
        or FAILURE_ALERT_ANNOTATION in annotations
    ):
        raise JobsError("prod9 Jobs preview identity or admission changed")
    result = copy.deepcopy(source)
    run_id = _run_id(plan, bound)
    result["metadata"]["name"] = bound.run_name
    result["metadata"]["labels"]["fleet.ai/run-id"] = run_id
    result["metadata"]["annotations"]["fleet.ai/run-id"] = run_id
    result["metadata"]["annotations"][FAILURE_ALERT_ANNOTATION] = FAILURE_ALERT_OFF
    spec = result["spec"]
    if (
        spec.get("entrypoint") != request["command"]
        or spec.get("suspend") is not True
        or spec.get("shutdownAfterJobFinishes") is not True
        or spec.get("submissionMode") != "HTTPMode"
    ):
        raise JobsError("prod9 Jobs preview execution changed")
    spec["backoffLimit"] = 0
    spec["activeDeadlineSeconds"] = MAXIMUM_SECONDS
    cluster = spec.get("rayClusterSpec", {})
    if cluster.get("workerGroupSpecs") not in (None, []):
        raise JobsError("prod9 must remain one physical GPU node")
    head = cluster.get("headGroupSpec", {}).get("template", {}).get("spec", {})
    containers = head.get("containers", [])
    if len(containers) != 1:
        raise JobsError("prod9 preview must contain one head container")
    container = containers[0]
    _replace_env(container, "FLEET_RUN_ID", run_id)
    _replace_env(container, "FLEET_RUN_NAME", bound.run_name)
    generated_secret = placeholder + "-fleet-key"
    secret_names = [row.get("secretRef", {}).get("name") for row in container.get("envFrom", [])]
    if secret_names != ["fleet-api", "wandb-api", generated_secret]:
        raise JobsError("prod9 Jobs preview Secret bindings changed")
    container["envFrom"] = [
        row
        for row in container["envFrom"]
        if row.get("secretRef", {}).get("name") != generated_secret
    ]
    container["securityContext"] = _runtime_context()
    container["terminationMessagePath"] = plan["output_root"] + "/NATIVE_TRAINING_COMPLETE.json"
    container["terminationMessagePolicy"] = "File"
    init = head.get("initContainers", [])
    sfs = [item for item in init if item.get("name") == "sfs-init"]
    if len(sfs) != 1 or sfs[0].get("command") != [
        "sh",
        "-c",
        f"mkdir -p {plan['output_root']} && chown 1000:100 {plan['output_root']}",
    ]:
        raise JobsError("prod9 output initialization changed")
    sfs[0]["command"] = [
        "sh",
        "-ec",
        f"mkdir {plan['output_root']}; chown 1000:100 {plan['output_root']}",
    ]
    training.validate_preview(
        plan,
        request,
        {"manifest_yaml": yaml.safe_dump(result, sort_keys=True), "warnings": []},
    )
    return result


def packet(
    plan: dict[str, Any],
    request: dict[str, Any],
    preview: dict[str, Any],
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Seal a source-previewed, non-submitting prod9 direct-RayJob packet."""
    bound = _identity(plan, identity)
    value = manifest(plan, request, preview, identity=bound)
    body = {
        "schema": PACKET_SCHEMA,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "jobs_preview_sha256": digest(preview),
        "manifest_sha256": digest(value),
        "direct_run_id": _run_id(plan, bound),
        "name": bound.run_name,
        "namespace": NAMESPACE,
        "failure_alerts": "off",
        "submitted": False,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def _cpu_job(name: str, image: str, command: str, env: dict[str, str]) -> dict[str, Any]:
    """Build the root-annotated zero-GPU exact-image preflight Job."""
    if not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name):
        raise JobsError("prod9 CPU preflight name changed")
    container = {
        "name": "preflight",
        "image": image,
        "command": ["/bin/sh", "-lc", "exec " + command],
        "env": [{"name": key, "value": value} for key, value in sorted(env.items())],
        "securityContext": _runtime_context(),
        "resources": {
            "requests": {"cpu": "4", "memory": "32Gi"},
            "limits": {"cpu": "8", "memory": "48Gi"},
        },
        "terminationMessagePath": PREFLIGHT_RECEIPT,
        "terminationMessagePolicy": "File",
        "volumeMounts": [{"name": "sfs", "mountPath": "/mnt/sfs"}],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "annotations": {FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF},
        },
        "spec": {
            "activeDeadlineSeconds": 1800,
            "backoffLimit": 0,
            "template": {
                "metadata": {},
                "spec": {
                    "automountServiceAccountToken": False,
                    "containers": [container],
                    "priorityClassName": "c1",
                    "restartPolicy": "Never",
                    "securityContext": {"supplementalGroups": [2000]},
                    "volumes": [{"name": "sfs", "persistentVolumeClaim": {"claimName": PVC}}],
                },
            },
        },
    }


def stage_job_manifest(
    stage: dict[str, Any], *, identity: historical.RailIdentity
) -> dict[str, Any]:
    """Render the prod9-only zero-GPU SFS rebind Job, never submit it."""
    checked, bound = training._stage_identity(stage)
    if bound != identity:
        raise JobsError("prod9 stage identity differs from the sealed stage specification")
    request = training.stage_request(checked)
    if (
        request.get("name") != bound.stage_name
        or request.get("image") != checked["image"]
        or request.get("failureAlerts") is not False
        or request.get("priority_class") != "c1"
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 1
        or request.get("secrets") != []
    ):
        raise JobsError("prod9 rebind request changed")
    job = _cpu_job(
        bound.stage_name,
        request["image"],
        request["command"],
        {**request["env"], "RUN_DIR": "/tmp"},
    )
    if "nvidia.com/gpu" in json.dumps(job, sort_keys=True) or (
        job["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise JobsError("prod9 stage root resource/alert contract changed")
    return job


def preflight_job_manifest(
    plan: dict[str, Any], *, identity: historical.RailIdentity
) -> dict[str, Any]:
    """Render only the fresh prod9 CPU preflight Job; never submit it."""
    bound = _identity(plan, identity)
    request = training.preflight_request(plan, receipt=PREFLIGHT_RECEIPT)
    if (
        request.get("failureAlerts") is not False
        or request.get("priority_class") != "c1"
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
    ):
        raise JobsError("prod9 CPU preflight request changed")
    job = _cpu_job(
        bound.preflight_name,
        request["image"],
        request["command"],
        {**request["env"], "RUN_DIR": "/tmp"},
    )
    if "nvidia.com/gpu" in json.dumps(job, sort_keys=True) or (
        job["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise JobsError("prod9 CPU preflight root resource/alert contract changed")
    return job


def _strip_server_defaults(expected: dict[str, Any], rendered: dict[str, Any]) -> dict[str, Any]:
    actual = copy.deepcopy(rendered)
    metadata = actual.get("metadata", {})
    try:
        UUID(metadata.pop("uid"))
        _timestamp(metadata.pop("creationTimestamp"))
    except (KeyError, TypeError, ValueError) as exc:
        raise JobsError("prod9 server dry-run lacks identity defaults") from exc
    if metadata.pop("generation", None) != 1 or "resourceVersion" in metadata:
        raise JobsError("prod9 server dry-run metadata defaults changed")
    if actual.pop("status", None) not in (None, {}):
        raise JobsError("prod9 server dry-run added status")
    spec = actual.get("spec", {})
    if spec.pop("ttlSecondsAfterFinished", None) not in (None, 0):
        raise JobsError("prod9 RayJob TTL default changed")
    cluster = spec.get("rayClusterSpec", {})
    if cluster.pop("headServiceAnnotations", None) not in (None, {}):
        raise JobsError("prod9 head-service defaults changed")
    head = cluster.get("headGroupSpec", {})
    if head.pop("numOfHosts", None) not in (None, 1):
        raise JobsError("prod9 head host default changed")
    if head.pop("scaleStrategy", None) not in (None, {}):
        raise JobsError("prod9 head scale default changed")
    return actual


def validate_preview(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    rendered: dict[str, Any],
    *,
    context: str,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Prove server rendering preserves the fresh root annotation and runtime."""
    bound = _identity(plan, identity)
    if expected != manifest(plan, request, source_preview, identity=bound):
        raise JobsError("prod9 direct packet changed")
    if (
        context not in {DEV_CONTEXT, PROD_CONTEXT}
        or _strip_server_defaults(expected, rendered) != expected
        or expected["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise JobsError("prod9 server dry-run changed the direct RayJob")
    body = {
        "schema": PREVIEW_SCHEMA,
        "status": "passed",
        "context": context,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "manifest_sha256": digest(expected),
        "server_render_sha256": digest(rendered),
        "name": bound.run_name,
        "nodes": 1,
        "gpus": 8,
        "priority": "c1",
        "queue_priority": "q1",
        "failure_alerts": "off",
        "submitted": False,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def server_dry_run(
    value: dict[str, Any],
    *,
    context: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Ask Kubernetes to render one manifest without persisting it."""
    if context not in {DEV_CONTEXT, PROD_CONTEXT}:
        raise JobsError("prod9 server preview context is invalid")
    result = runner(
        [
            "kubectl",
            "--context",
            context,
            "--namespace",
            NAMESPACE,
            "create",
            "--dry-run=server",
            "-f",
            "-",
            "-o",
            "json",
        ],
        input=json.dumps(value),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise JobsError("prod9 Kubernetes server dry-run failed")
    try:
        rendered = json.loads(result.stdout)
    except ValueError as exc:
        raise JobsError("prod9 Kubernetes server dry-run returned invalid JSON") from exc
    if not isinstance(rendered, dict):
        raise JobsError("prod9 Kubernetes server dry-run returned a non-object")
    return rendered


def _strip_cpu_server_defaults(
    expected: dict[str, Any], rendered: dict[str, Any]
) -> dict[str, Any]:
    actual = copy.deepcopy(rendered)
    status = actual.pop("status", None)
    metadata = actual.get("metadata", {})
    timestamp = metadata.pop("creationTimestamp", None)
    generation = metadata.pop("generation", None)
    uid = metadata.pop("uid", None)
    labels = metadata.pop("labels", None)
    name = expected["metadata"]["name"]
    generated_labels = {
        "batch.kubernetes.io/controller-uid": uid,
        "batch.kubernetes.io/job-name": name,
        "controller-uid": uid,
        "job-name": name,
    }
    spec = actual.get("spec", {})
    defaults = {
        key: spec.pop(key, None)
        for key in (
            "completionMode",
            "completions",
            "manualSelector",
            "parallelism",
            "podReplacementPolicy",
            "selector",
            "suspend",
        )
    }
    template = spec.get("template", {})
    template_labels = template.get("metadata", {}).pop("labels", None)
    pod = template.get("spec", {})
    pod_defaults = {
        key: pod.pop(key, None)
        for key in ("dnsPolicy", "schedulerName", "terminationGracePeriodSeconds")
        if key not in expected["spec"]["template"]["spec"]
    }
    containers = pod.get("containers", [])
    pull = containers[0].pop("imagePullPolicy", None) if len(containers) == 1 else None
    try:
        UUID(uid)
        _timestamp(timestamp)
    except (TypeError, ValueError, JobsError) as exc:
        raise JobsError("prod9 CPU server dry-run lacks identity defaults") from exc
    if (
        status != {}
        or generation != 1
        or labels is not None
        or template_labels != generated_labels
        or defaults
        != {
            "completionMode": "NonIndexed",
            "completions": 1,
            "manualSelector": False,
            "parallelism": 1,
            "podReplacementPolicy": "TerminatingOrFailed",
            "selector": {"matchLabels": {"batch.kubernetes.io/controller-uid": uid}},
            "suspend": False,
        }
        or pod_defaults
        != {
            "dnsPolicy": "ClusterFirst",
            "schedulerName": "default-scheduler",
            "terminationGracePeriodSeconds": 30,
        }
        or pull != "IfNotPresent"
    ):
        raise JobsError("prod9 CPU server dry-run changed Kubernetes defaults")
    return actual


def validate_cpu_preview(
    expected: dict[str, Any],
    rendered: dict[str, Any],
    *,
    context: str,
    purpose: str,
) -> dict[str, Any]:
    """Prove a zero-GPU Job retains its root alert-off annotation."""
    if context not in {DEV_CONTEXT, PROD_CONTEXT} or purpose not in {"stage", "preflight"}:
        raise JobsError("prod9 CPU preview binding is invalid")
    if (
        expected.get("apiVersion") != "batch/v1"
        or expected.get("kind") != "Job"
        or expected.get("metadata", {}).get("namespace") != NAMESPACE
        or expected.get("metadata", {}).get("annotations", {}).get(FAILURE_ALERT_ANNOTATION)
        != FAILURE_ALERT_OFF
        or expected.get("spec", {}).get("backoffLimit") != 0
        or expected.get("spec", {}).get("activeDeadlineSeconds") != CPU_MAXIMUM_SECONDS
        or expected.get("spec", {}).get("template", {}).get("spec", {}).get("priorityClassName")
        != "c1"
        or "nvidia.com/gpu" in json.dumps(expected, sort_keys=True)
        or _strip_cpu_server_defaults(expected, rendered) != expected
    ):
        raise JobsError("prod9 CPU server dry-run changed the exact Job")
    return _seal(
        {
            "schema": CPU_PREVIEW_SCHEMA,
            "status": "passed",
            "purpose": purpose,
            "context": context,
            "name": expected["metadata"]["name"],
            "manifest_sha256": "sha256:" + digest(expected),
            "server_render_sha256": "sha256:" + digest(rendered),
            "gpus": 0,
            "priority": "c1",
            "failure_alerts": "off",
            "submitted": False,
            "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )


def _cpu_previews(
    expected: dict[str, Any],
    proofs: list[dict[str, Any]],
    *,
    purpose: str,
    fresh: bool,
) -> list[dict[str, Any]]:
    observed = []
    for proof in proofs:
        value = _validate_seal(proof, CPU_PREVIEW_SCHEMA)
        if (
            value.get("status") != "passed"
            or value.get("purpose") != purpose
            or value.get("context") not in {DEV_CONTEXT, PROD_CONTEXT}
            or value.get("name") != expected["metadata"]["name"]
            or value.get("manifest_sha256") != "sha256:" + digest(expected)
            or value.get("gpus") != 0
            or value.get("priority") != "c1"
            or value.get("failure_alerts") != "off"
            or value.get("submitted") is not False
        ):
            raise JobsError("prod9 CPU server preview was not accepted")
        if fresh:
            _fresh_at(value.get("checked_at"))
        observed.append(value["context"])
    if sorted(observed) != sorted({DEV_CONTEXT, PROD_CONTEXT}) or len(proofs) != 2:
        raise JobsError("prod9 CPU server preview set is incomplete")
    return proofs


def _receipt(value: object, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise JobsError("prod9 CPU receipt is not an object")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("schema") != schema or value.get("receipt_sha256") != digest(body):
        raise JobsError("prod9 CPU receipt digest/schema changed")
    return value


def _cpu_release(
    observer: object,
    receipt: dict[str, Any],
    *,
    name: str,
    plan_sha256: str,
    manifest_sha256: str,
    fresh: bool,
) -> dict[str, Any]:
    value = _validate_seal(observer, "cyber_direct_cleanup_observer_result_v1")
    absent = {
        "target_present",
        "pods_present",
        "rayjob_present",
        "workload_present",
        "raycluster_present",
    }
    if (
        value.get("status") != "released"
        or value.get("context") != PROD_CONTEXT
        or value.get("namespace") != NAMESPACE
        or value.get("kind") != "job"
        or value.get("name") != name
        or value.get("plan_sha256") != plan_sha256
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("expected_gpus") != 0
        or value.get("peak_gpus") != 0
        or value.get("active_gpus") != 0
        or value.get("terminal_status") != "Succeeded"
        or value.get("restarts") != 0
        or value.get("observer_error_class") != ""
        or value.get("exit_codes") != [0]
        or value.get("receipt") != receipt
        or any(value.get(key) is not False for key in absent)
    ):
        raise JobsError("prod9 zero-GPU release evidence is incomplete")
    if fresh:
        _fresh_at(value.get("release_observed_at"))
    return value


def _stage_receipt(
    stage: dict[str, Any], receipt: object, *, identity: historical.RailIdentity
) -> dict[str, Any]:
    checked, bound = training._stage_identity(stage)
    if bound != identity:
        raise JobsError("prod9 rebind identity differs from its sealed stage")
    value = _receipt(receipt, training.STAGE_RECEIPT_SCHEMA)
    successor = value.get("successor_manifest")
    successor_body = (
        {key: item for key, item in successor.items() if key != "sha256"}
        if isinstance(successor, dict)
        else {}
    )
    files = value.get("files")
    staged_files = (
        {item.get("path"): item for item in files}
        if isinstance(files, list) and all(isinstance(item, dict) for item in files)
        else {}
    )
    expected_names = {
        "manifest.json",
        "split.json",
        "task-set.json",
        "train.jsonl",
        "dev.jsonl",
    }
    successor_files = successor.get("files") if isinstance(successor, dict) else None
    if (
        value.get("status") != "published"
        or value.get("stage_spec_sha256") != checked["sha256"]
        or value.get("identity_sha256") != identity.sealed_mapping()["sha256"]
        or value.get("source") != identity.predecessor_data_root
        or value.get("destination") != identity.data_root
        or value.get("predecessor_manifest_sha256") != checked["predecessor_manifest_sha256"]
        or not isinstance(successor, dict)
        or successor.get("schema") != "cyber_skyrl_data_v1"
        or successor.get("name") != identity.run_name
        or successor.get("sha256") != "sha256:" + digest(successor_body)
        or value.get("successor_manifest_sha256") != successor.get("sha256")
        or not isinstance(successor_files, dict)
        or set(successor_files) != {"train", "dev"}
        or any(
            not isinstance(successor_files[split], dict)
            or successor_files[split].get("path") != split + ".jsonl"
            or not isinstance(successor_files[split].get("sha256"), str)
            for split in ("train", "dev")
        )
        or set(staged_files) != expected_names
        or len(staged_files) != len(files)
        or any(
            set(item) != {"path", "bytes", "sha256"}
            or type(item["bytes"]) is not int
            or item["bytes"] < 1
            or not isinstance(item["sha256"], str)
            or not item["sha256"].startswith("sha256:")
            for item in staged_files.values()
        )
        or any(
            staged_files[successor_files[split]["path"]]["sha256"]
            != successor_files[split]["sha256"]
            for split in ("train", "dev")
        )
        or value.get("gpus") != 0
        or value.get("runtime_user") != {"uid": 1000, "gid": 100}
        or any(value.get(key) != 0 for key in checked["scientific_work"])
    ):
        raise JobsError("prod9 rebind stage receipt is incomplete")
    return value


def _preflight_receipt(
    plan: dict[str, Any],
    request: dict[str, Any],
    receipt: object,
    *,
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    bound = _identity(plan, identity)
    value = _receipt(receipt, "cyber_skyrl_prod9_training_cpu_preflight_v1")
    required_true = {
        "native_parser_checked",
        "ordered_multi_tool_parser_checked",
        "chunk_continuation_checked",
        "compaction_checked",
        "stepwise_prompt_checked",
        "ordered_multi_tool_execution_checked",
        "output_limit_gradeable_checked",
        "output_limit_partial_tool_blocked_checked",
        "fresh_recorder_checked",
        "tool_result_token_safe",
    }
    if (
        value.get("status") != "passed"
        or value.get("gpus") != 0
        or value.get("runtime_user") != {"uid": 1000, "gid": 100}
        or value.get("plan_sha256") != digest(plan)
        or value.get("request_sha256") != digest(request)
        or value.get("prod9_runtime") != training._binding()
        or value.get("counts") != {"train": 1, "dev": 1}
        or value.get("planned_steps") != plan["arguments"]["steps"]
        or value.get("output_absent") is not True
        or value.get("wandb_create_once")
        != {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "run_id": bound.wandb_run_id,
            "resume": "never",
        }
        or value.get("recorder_implementation") != "training.skyrl_prod9_hardening.Recorder"
        or any(value.get(key) is not True for key in required_true)
    ):
        raise JobsError("prod9 CPU preflight receipt is incomplete")
    return value


def _stage_authorization(
    stage: dict[str, Any],
    expected: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
    identity: historical.RailIdentity,
    require_live_observer: bool,
    fresh_previews: bool,
) -> dict[str, Any]:
    checked, bound = training._stage_identity(stage)
    if bound != identity or expected != stage_job_manifest(checked, identity=bound):
        raise JobsError("prod9 stage manifest changed")
    previews = _cpu_previews(
        expected,
        [dev_preview, prod_preview],
        purpose="stage",
        fresh=fresh_previews,
    )
    plan_sha256, manifest_sha256 = checked["sha256"], "sha256:" + digest(expected)
    if require_live_observer:
        armed = _observer_pid(
            observer,
            kind="job",
            name=bound.stage_name,
            plan_sha256=plan_sha256,
            manifest_sha256=manifest_sha256,
            gpus=0,
            maximum_seconds=CPU_MAXIMUM_SECONDS,
        )
    else:
        armed = _validate_seal(observer, "cyber_direct_cleanup_observer_armed_v1")
        if (
            armed.get("context") != PROD_CONTEXT
            or armed.get("namespace") != NAMESPACE
            or armed.get("kind") != "job"
            or armed.get("name") != bound.stage_name
            or armed.get("plan_sha256") != plan_sha256
            or armed.get("manifest_sha256") != manifest_sha256
            or armed.get("expected_gpus") != 0
            or armed.get("maximum_seconds") != CPU_MAXIMUM_SECONDS
        ):
            raise JobsError("prod9 stage observer binding changed")
    operation_root = hardening.stage_operation_root(checked)
    _operation_root(armed, expected_root=operation_root, purpose="stage")
    return _seal(
        {
            "schema": STAGE_AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "stage_spec_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "dev_preview": previews[0],
            "prod_preview": previews[1],
            "observer": armed,
            "operation_root": str(operation_root),
        }
    )


def authorize_stage(
    stage: dict[str, Any],
    expected: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Authorize exactly one prod9 SFS rebind Job after root-preview proof."""
    return _stage_authorization(
        stage,
        expected,
        dev_preview=dev_preview,
        prod_preview=prod_preview,
        observer=observer,
        identity=identity,
        require_live_observer=True,
        fresh_previews=True,
    )


def _preflight_authorization(
    plan: dict[str, Any],
    request: dict[str, Any],
    stage: dict[str, Any],
    stage_authorization: dict[str, Any],
    stage_created: dict[str, Any],
    stage_receipt: dict[str, Any],
    stage_release: dict[str, Any],
    expected: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
    identity: historical.RailIdentity,
    require_live_observer: bool,
    fresh_stage_release: bool,
    fresh_previews: bool,
) -> dict[str, Any]:
    bound = _identity(plan, identity)
    checked_stage, _ = training._stage_identity(stage)
    stage_expected = stage_job_manifest(checked_stage, identity=bound)
    stage_auth = _validate_seal(stage_authorization, STAGE_AUTHORIZATION_SCHEMA)
    if stage_auth != _stage_authorization(
        checked_stage,
        stage_expected,
        dev_preview=stage_auth["dev_preview"],
        prod_preview=stage_auth["prod_preview"],
        observer=stage_auth["observer"],
        identity=bound,
        require_live_observer=False,
        fresh_previews=False,
    ):
        raise JobsError("prod9 stage authorization changed")
    stage_created_value = _cpu_created(
        stage_created,
        purpose="stage",
        name=bound.stage_name,
        plan_sha256=checked_stage["sha256"],
        manifest_sha256="sha256:" + digest(stage_expected),
        authorization_sha256=stage_auth["sha256"],
    )
    staged = _stage_receipt(checked_stage, stage_receipt, identity=bound)
    if staged["successor_manifest"] != plan["data"]:
        raise JobsError("prod9 rebind output differs from the immutable training plan")
    stage_observer = _cpu_release(
        stage_release,
        staged,
        name=bound.stage_name,
        plan_sha256=checked_stage["sha256"],
        manifest_sha256="sha256:" + digest(stage_expected),
        fresh=fresh_stage_release,
    )
    if stage_observer.get("uid") != stage_created_value["job_uid"]:
        raise JobsError("prod9 stage release UID differs from its one create")
    if expected != preflight_job_manifest(plan, identity=bound):
        raise JobsError("prod9 CPU preflight manifest changed")
    previews = _cpu_previews(
        expected,
        [dev_preview, prod_preview],
        purpose="preflight",
        fresh=fresh_previews,
    )
    plan_sha256, manifest_sha256 = "sha256:" + digest(plan), "sha256:" + digest(expected)
    if require_live_observer:
        armed = _observer_pid(
            observer,
            kind="job",
            name=bound.preflight_name,
            plan_sha256=plan_sha256,
            manifest_sha256=manifest_sha256,
            gpus=0,
            maximum_seconds=CPU_MAXIMUM_SECONDS,
        )
    else:
        armed = _validate_seal(observer, "cyber_direct_cleanup_observer_armed_v1")
        if (
            armed.get("context") != PROD_CONTEXT
            or armed.get("namespace") != NAMESPACE
            or armed.get("kind") != "job"
            or armed.get("name") != bound.preflight_name
            or armed.get("plan_sha256") != plan_sha256
            or armed.get("manifest_sha256") != manifest_sha256
            or armed.get("expected_gpus") != 0
            or armed.get("maximum_seconds") != CPU_MAXIMUM_SECONDS
        ):
            raise JobsError("prod9 CPU preflight observer binding changed")
    operation_root = hardening.training_operation_root(plan)
    _operation_root(armed, expected_root=operation_root, purpose="preflight")
    return _seal(
        {
            "schema": PREFLIGHT_AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "plan_sha256": plan_sha256,
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": manifest_sha256,
            "stage_authorization": stage_auth,
            "stage_created": stage_created_value,
            "stage_receipt": staged,
            "stage_release": stage_observer,
            "dev_preview": previews[0],
            "prod_preview": previews[1],
            "observer": armed,
            "operation_root": str(operation_root),
        }
    )


def authorize_preflight(
    plan: dict[str, Any],
    request: dict[str, Any],
    stage: dict[str, Any],
    stage_authorization: dict[str, Any],
    stage_created: dict[str, Any],
    stage_receipt: dict[str, Any],
    stage_release: dict[str, Any],
    expected: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Authorize the CPU preflight only after the exact rebind has released."""
    return _preflight_authorization(
        plan,
        request,
        stage,
        stage_authorization,
        stage_created,
        stage_receipt,
        stage_release,
        expected,
        dev_preview=dev_preview,
        prod_preview=prod_preview,
        observer=observer,
        identity=identity,
        require_live_observer=True,
        fresh_stage_release=True,
        fresh_previews=True,
    )


def _cpu_created(
    value: object,
    *,
    purpose: str,
    name: str,
    plan_sha256: str,
    manifest_sha256: str,
    authorization_sha256: str,
) -> dict[str, Any]:
    result = _validate_seal(value, CPU_CREATED_SCHEMA)
    try:
        UUID(result.get("job_uid"))
        _timestamp(result.get("created_at"))
    except (TypeError, ValueError, JobsError) as exc:
        raise JobsError("prod9 CPU create receipt identity is invalid") from exc
    if (
        result.get("status") != "created_once"
        or result.get("purpose") != purpose
        or result.get("name") != name
        or result.get("plan_sha256") != plan_sha256
        or result.get("manifest_sha256") != manifest_sha256
        or result.get("authorization_sha256") != authorization_sha256
    ):
        raise JobsError("prod9 CPU create receipt changed")
    return result


def _kubectl(
    runner: Callable[..., subprocess.CompletedProcess[str]], context: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["kubectl", "--context", context, "--namespace", NAMESPACE, *arguments],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _cpu_duplicate_checks(
    name: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, int]:
    checked = 0
    for context in (DEV_CONTEXT, PROD_CONTEXT):
        for resource in ("job", "rayjob", "raycluster", "workload", "pod"):
            result = _kubectl(runner, context, "get", resource, "--output=json")
            if result.returncode:
                raise JobsError("prod9 CPU Kubernetes duplicate inventory failed")
            checked += 1
            try:
                items = json.loads(result.stdout).get("items", [])
            except (AttributeError, ValueError) as exc:
                raise JobsError("prod9 CPU Kubernetes duplicate inventory is invalid") from exc
            for item in items:
                metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
                values = [metadata.get("name", "")]
                values.extend(metadata.get("labels", {}).values())
                values.extend(metadata.get("annotations", {}).values())
                if any(value == name or str(value).startswith(name + "-") for value in values):
                    raise JobsError("prod9 CPU Kubernetes identity already exists")
    return {"kubernetes_inventories_checked": checked}


def _write_once_fsynced(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _create_cpu_once(
    directory: Path,
    expected: dict[str, Any],
    authorization: dict[str, Any],
    *,
    purpose: str,
    name: str,
    plan_sha256: str,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    journal_name = (
        "PROD9_STAGE_CREATE.jsonl" if purpose == "stage" else "PROD9_PREFLIGHT_CREATE.jsonl"
    )
    journal = directory / journal_name
    if journal.exists() or journal.is_symlink():
        raise JobsError("prod9 CPU create intent exists; reconcile, never retry")
    manifest_sha256 = "sha256:" + digest(expected)
    auth = _validate_seal(
        authorization,
        STAGE_AUTHORIZATION_SCHEMA if purpose == "stage" else PREFLIGHT_AUTHORIZATION_SCHEMA,
    )
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or directory.resolve() != Path(auth.get("operation_root", ""))
    ):
        raise JobsError("prod9 CPU create directory differs from its sealed operation root")
    if (
        auth.get("status") != "authorized_for_one_create"
        or auth.get("manifest_sha256") != manifest_sha256
        or auth.get("stage_spec_sha256", auth.get("plan_sha256")) != plan_sha256
    ):
        raise JobsError("prod9 CPU authorization changed")
    observer = auth["observer"]
    _observer_pid(
        observer,
        kind="job",
        name=name,
        plan_sha256=plan_sha256,
        manifest_sha256=manifest_sha256,
        gpus=0,
        maximum_seconds=CPU_MAXIMUM_SECONDS,
    )
    duplicate = _cpu_duplicate_checks(name, runner=runner)
    rendered = server_dry_run(expected, context=PROD_CONTEXT, runner=runner)
    validate_cpu_preview(expected, rendered, context=PROD_CONTEXT, purpose=purpose)
    _observer_pid(
        observer,
        kind="job",
        name=name,
        plan_sha256=plan_sha256,
        manifest_sha256=manifest_sha256,
        gpus=0,
        maximum_seconds=CPU_MAXIMUM_SECONDS,
    )
    for preview in (auth["dev_preview"], auth["prod_preview"]):
        _fresh_at(preview.get("checked_at"))
    if purpose == "preflight":
        _fresh_at(auth["stage_release"].get("release_observed_at"))
    _write_once_fsynced(
        journal,
        {
            "state": "CREATE_INTENT_DO_NOT_RETRY",
            "purpose": purpose,
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "authorization_sha256": auth["sha256"],
            "duplicate_checks": duplicate,
        },
    )
    # This is the sole mutating call.  Do not add a retry or a second create.
    # `_kubectl` is deliberately read-only because it cannot carry stdin.
    result = runner(
        [
            "kubectl",
            "--context",
            PROD_CONTEXT,
            "--namespace",
            NAMESPACE,
            "create",
            "-f",
            "-",
            "-o",
            "json",
        ],
        input=json.dumps(expected),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise JobsError("prod9 CPU create returned failure; reconcile intent, never retry")
    try:
        created = json.loads(result.stdout)
        uid = str(UUID(created["metadata"]["uid"]))
        created_at = created["metadata"]["creationTimestamp"]
        _timestamp(created_at)
    except (KeyError, TypeError, ValueError, JobsError) as exc:
        raise JobsError(
            "prod9 CPU create response is ambiguous; reconcile intent, never retry"
        ) from exc
    if created.get("metadata", {}).get("name") != name:
        raise JobsError("prod9 CPU create returned another identity; reconcile intent, never retry")
    _publish_creator_binding(
        observer,
        kind="job",
        name=name,
        plan_sha256=plan_sha256,
        manifest_sha256=manifest_sha256,
        uid=uid,
    )
    proof = _seal(
        {
            "schema": CPU_CREATED_SCHEMA,
            "status": "created_once",
            "purpose": purpose,
            "name": name,
            "plan_sha256": plan_sha256,
            "manifest_sha256": manifest_sha256,
            "authorization_sha256": auth["sha256"],
            "job_uid": uid,
            "created_at": created_at,
        }
    )
    with journal.open("a") as stream:
        stream.write(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return proof


def create_stage_once(
    directory: Path,
    stage: dict[str, Any],
    expected: dict[str, Any],
    authorization: dict[str, Any],
    *,
    identity: historical.RailIdentity,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create the one permitted zero-GPU prod9 SFS rebind Job."""
    auth = _validate_seal(authorization, STAGE_AUTHORIZATION_SCHEMA)
    canonical = hardening.stage_operation_root(stage)
    if directory.resolve() != canonical or Path(auth.get("operation_root", "")) != canonical:
        raise JobsError("prod9 CPU create directory differs from its sealed operation root")
    if auth != authorize_stage(
        stage,
        expected,
        dev_preview=auth["dev_preview"],
        prod_preview=auth["prod_preview"],
        observer=auth["observer"],
        identity=identity,
    ):
        raise JobsError("prod9 stage authorization changed")
    checked, _ = training._stage_identity(stage)
    return _create_cpu_once(
        directory,
        expected,
        auth,
        purpose="stage",
        name=identity.stage_name,
        plan_sha256=checked["sha256"],
        runner=runner,
    )


def create_preflight_once(
    directory: Path,
    plan: dict[str, Any],
    request: dict[str, Any],
    stage: dict[str, Any],
    authorization: dict[str, Any],
    *,
    identity: historical.RailIdentity,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Create the one permitted zero-GPU exact-image prod9 preflight Job."""
    auth = _validate_seal(authorization, PREFLIGHT_AUTHORIZATION_SCHEMA)
    canonical = hardening.training_operation_root(plan)
    if directory.resolve() != canonical or Path(auth.get("operation_root", "")) != canonical:
        raise JobsError("prod9 CPU create directory differs from its sealed operation root")
    expected = preflight_job_manifest(plan, identity=identity)
    if auth != _preflight_authorization(
        plan,
        request,
        stage,
        auth["stage_authorization"],
        auth["stage_created"],
        auth["stage_receipt"],
        auth["stage_release"],
        expected,
        dev_preview=auth["dev_preview"],
        prod_preview=auth["prod_preview"],
        observer=auth["observer"],
        identity=identity,
        require_live_observer=True,
        fresh_stage_release=True,
        fresh_previews=True,
    ):
        raise JobsError("prod9 CPU preflight authorization changed")
    return _create_cpu_once(
        directory,
        expected,
        auth,
        purpose="preflight",
        name=identity.preflight_name,
        plan_sha256="sha256:" + digest(plan),
        runner=runner,
    )


def _direct_previews(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    proofs: list[dict[str, Any]],
    *,
    identity: historical.RailIdentity,
) -> list[dict[str, Any]]:
    observed = []
    for proof in proofs:
        value = _validate_seal(proof, PREVIEW_SCHEMA)
        if (
            value.get("status") != "passed"
            or value.get("context") not in {DEV_CONTEXT, PROD_CONTEXT}
            or value.get("plan_sha256") != digest(plan)
            or value.get("request_sha256") != digest(request)
            or value.get("manifest_sha256") != digest(expected)
            or value.get("name") != identity.run_name
            or value.get("nodes") != 1
            or value.get("gpus") != 8
            or value.get("priority") != "c1"
            or value.get("queue_priority") != "q1"
            or value.get("failure_alerts") != "off"
            or value.get("submitted") is not False
        ):
            raise JobsError("prod9 direct server preview was not accepted")
        _fresh_at(value.get("checked_at"))
        observed.append(value["context"])
    if sorted(observed) != sorted({DEV_CONTEXT, PROD_CONTEXT}) or len(proofs) != 2:
        raise JobsError("prod9 direct server preview set is incomplete")
    return proofs


def _wandb_exists_default(entity: str, project: str, run_id: str) -> bool:
    """Fail closed unless W&B proves the fresh run ID is absent.

    The credential stays in the caller environment and is never placed in a
    plan, receipt, manifest, or command line.  A 404 is the only acceptable
    absence result; subscription and transport errors are deliberately not
    treated as absence.
    """
    try:
        import wandb
        from wandb.errors import CommError
    except ImportError as exc:  # pragma: no cover - depends on operator image
        raise JobsError("W&B client is required for the fresh run-ID check") from exc
    if not os.environ.get("WANDB_API_KEY"):
        raise JobsError("W&B read credential is required for the fresh run-ID check")
    try:
        run = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
    except CommError as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 404:
            return False
        raise JobsError("fresh W&B run-ID check failed") from None
    return run is not None


def _direct_duplicate_checks(
    identity: historical.RailIdentity,
    *,
    token: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    jobs_factory: Callable[..., Jobs],
) -> dict[str, int]:
    """Prove all prod9 create-once names and the output root are unclaimed.

    The data root is intentionally excluded: it is the already accepted
    output of the zero-GPU rebind stage.  The stage/preflight *objects* must
    nevertheless be gone, so an incomplete predecessor cannot be confused
    with an eligible GPU arm.
    """
    if not isinstance(token, str) or not token:
        raise JobsError("prod9 Jobs API credential is required for duplicate checks")
    names = {identity.run_name, identity.stage_name, identity.preflight_name}
    output = identity.output_root
    inventories = 0
    for context in (DEV_CONTEXT, PROD_CONTEXT):
        for resource in ("rayjob", "raycluster", "job", "workload", "pod"):
            result = _kubectl(runner, context, "get", resource, "--output=json")
            if result.returncode:
                raise JobsError("prod9 Kubernetes duplicate inventory failed")
            inventories += 1
            try:
                items = json.loads(result.stdout).get("items", [])
            except (AttributeError, ValueError) as exc:
                raise JobsError("prod9 Kubernetes duplicate inventory is invalid") from exc
            if not isinstance(items, list):
                raise JobsError("prod9 Kubernetes duplicate inventory is invalid")
            for item in items:
                metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
                labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
                annotations = metadata.get("annotations", {}) if isinstance(metadata, dict) else {}
                values = [metadata.get("name", "")]
                if isinstance(labels, dict):
                    values.extend(labels.values())
                if isinstance(annotations, dict):
                    values.extend(annotations.values())
                serialized = json.dumps(item.get("spec", {}), sort_keys=True)
                if (
                    any(
                        value in names or any(str(value).startswith(name + "-") for name in names)
                        for value in values
                    )
                    or output in values
                    or output in serialized
                ):
                    raise JobsError("prod9 Kubernetes identity/output already exists")
    api_rows = 0
    for target in ("dev", "prod"):
        with jobs_factory(token, base_url=API_URLS[target]) as client:
            rows = client.all_runs()
        api_rows += len(rows)
        for row in rows:
            name = str(row.get("name", ""))
            if (
                row.get("run_dir") == output
                or name in names
                or any(name.startswith(value + "-") for value in names)
            ):
                raise JobsError("prod9 Jobs API history already owns this identity/output")
    return {"kubernetes_inventories_checked": inventories, "jobs_api_rows_checked": api_rows}


def _direct_authorization(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    stage: dict[str, Any],
    stage_authorization: dict[str, Any],
    stage_created: dict[str, Any],
    stage_receipt: dict[str, Any],
    stage_release: dict[str, Any],
    preflight_authorization: dict[str, Any],
    preflight_created: dict[str, Any],
    preflight_receipt: dict[str, Any],
    preflight_release: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
    identity: historical.RailIdentity,
    require_live_observer: bool,
) -> dict[str, Any]:
    """Bind every completed zero-GPU gate to one fresh GPU-create authority."""
    bound = _identity(plan, identity)
    if expected != manifest(plan, request, source_preview, identity=bound):
        raise JobsError("prod9 direct manifest changed")
    checked_stage, stage_identity = training._stage_identity(stage)
    if stage_identity != bound:
        raise JobsError("prod9 stage identity differs from GPU plan identity")
    stage_expected = stage_job_manifest(checked_stage, identity=bound)
    stage_auth = _validate_seal(stage_authorization, STAGE_AUTHORIZATION_SCHEMA)
    if stage_auth != _stage_authorization(
        checked_stage,
        stage_expected,
        dev_preview=stage_auth["dev_preview"],
        prod_preview=stage_auth["prod_preview"],
        observer=stage_auth["observer"],
        identity=bound,
        require_live_observer=False,
        fresh_previews=False,
    ):
        raise JobsError("prod9 stage authorization changed")
    stage_created_value = _cpu_created(
        stage_created,
        purpose="stage",
        name=bound.stage_name,
        plan_sha256=checked_stage["sha256"],
        manifest_sha256="sha256:" + digest(stage_expected),
        authorization_sha256=stage_auth["sha256"],
    )
    staged = _stage_receipt(checked_stage, stage_receipt, identity=bound)
    if staged["successor_manifest"] != plan["data"]:
        raise JobsError("prod9 staged manifest differs from the GPU training plan")
    stage_release_value = _cpu_release(
        stage_release,
        staged,
        name=bound.stage_name,
        plan_sha256=checked_stage["sha256"],
        manifest_sha256="sha256:" + digest(stage_expected),
        fresh=False,
    )
    if stage_release_value.get("uid") != stage_created_value["job_uid"]:
        raise JobsError("prod9 stage release UID differs from its one create")

    preflight_expected = preflight_job_manifest(plan, identity=bound)
    preflight_auth = _validate_seal(preflight_authorization, PREFLIGHT_AUTHORIZATION_SCHEMA)
    if preflight_auth != _preflight_authorization(
        plan,
        request,
        checked_stage,
        stage_auth,
        stage_created_value,
        staged,
        stage_release_value,
        preflight_expected,
        dev_preview=preflight_auth["dev_preview"],
        prod_preview=preflight_auth["prod_preview"],
        observer=preflight_auth["observer"],
        identity=bound,
        require_live_observer=False,
        fresh_stage_release=False,
        fresh_previews=False,
    ):
        raise JobsError("prod9 CPU preflight authorization changed")
    preflight_created_value = _cpu_created(
        preflight_created,
        purpose="preflight",
        name=bound.preflight_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(preflight_expected),
        authorization_sha256=preflight_auth["sha256"],
    )
    checked_preflight = _preflight_receipt(plan, request, preflight_receipt, identity=bound)
    preflight_release_value = _cpu_release(
        preflight_release,
        checked_preflight,
        name=bound.preflight_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(preflight_expected),
        fresh=True,
    )
    if preflight_release_value.get("uid") != preflight_created_value["job_uid"]:
        raise JobsError("prod9 CPU preflight release UID differs from its one create")

    previews = _direct_previews(
        plan, request, source_preview, expected, [dev_preview, prod_preview], identity=bound
    )
    plan_sha256, manifest_sha256 = "sha256:" + digest(plan), "sha256:" + digest(expected)
    if require_live_observer:
        armed = _observer_pid(
            observer,
            kind="rayjob",
            name=bound.run_name,
            plan_sha256=plan_sha256,
            manifest_sha256=manifest_sha256,
            gpus=8,
            maximum_seconds=MAXIMUM_SECONDS,
        )
    else:
        armed = _validate_seal(observer, "cyber_direct_cleanup_observer_armed_v1")
        if (
            armed.get("status") != "armed"
            or armed.get("context") != PROD_CONTEXT
            or armed.get("namespace") != NAMESPACE
            or armed.get("kind") != "rayjob"
            or armed.get("name") != bound.run_name
            or armed.get("plan_sha256") != plan_sha256
            or armed.get("manifest_sha256") != manifest_sha256
            or armed.get("expected_gpus") != 8
            or armed.get("maximum_seconds") != MAXIMUM_SECONDS
        ):
            raise JobsError("prod9 direct observer binding changed")
    stage_root = hardening.stage_operation_root(checked_stage)
    operation_root = hardening.training_operation_root(plan)
    _operation_root(stage_auth["observer"], expected_root=stage_root, purpose="stage")
    _operation_root(preflight_auth["observer"], expected_root=operation_root, purpose="preflight")
    _operation_root(armed, expected_root=operation_root, purpose="training")
    return _seal(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized_for_one_create",
            "plan_sha256": plan_sha256,
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": manifest_sha256,
            "stage": checked_stage,
            "stage_authorization": stage_auth,
            "stage_created": stage_created_value,
            "stage_receipt": staged,
            "stage_release": stage_release_value,
            "preflight_authorization": preflight_auth,
            "preflight_created": preflight_created_value,
            "preflight_receipt": checked_preflight,
            "preflight_release": preflight_release_value,
            "dev_preview": previews[0],
            "prod_preview": previews[1],
            "observer": armed,
            "operation_root": str(operation_root),
            "output_absence_enforcement": "fresh_cpu_preflight_plus_non_idempotent_gpu_init",
        }
    )


def authorize(
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    stage: dict[str, Any],
    stage_authorization: dict[str, Any],
    stage_created: dict[str, Any],
    stage_receipt: dict[str, Any],
    stage_release: dict[str, Any],
    preflight_authorization: dict[str, Any],
    preflight_created: dict[str, Any],
    preflight_receipt: dict[str, Any],
    preflight_release: dict[str, Any],
    *,
    dev_preview: dict[str, Any],
    prod_preview: dict[str, Any],
    observer: dict[str, Any],
    identity: historical.RailIdentity,
) -> dict[str, Any]:
    """Authorize one prod9 GPU create after the rebind and preflight release."""
    return _direct_authorization(
        plan,
        request,
        source_preview,
        expected,
        stage,
        stage_authorization,
        stage_created,
        stage_receipt,
        stage_release,
        preflight_authorization,
        preflight_created,
        preflight_receipt,
        preflight_release,
        dev_preview=dev_preview,
        prod_preview=prod_preview,
        observer=observer,
        identity=identity,
        require_live_observer=True,
    )


def create_once(
    directory: Path,
    plan: dict[str, Any],
    request: dict[str, Any],
    source_preview: dict[str, Any],
    expected: dict[str, Any],
    authorization: dict[str, Any],
    *,
    token: str,
    identity: historical.RailIdentity,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    jobs_factory: Callable[..., Jobs] = Jobs,
    wandb_exists: Callable[[str, str, str], bool] = _wandb_exists_default,
    capacity_reader: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Perform the single prod9 GPU create after re-checking every live gate.

    This is deliberately the only mutating operation in the fresh rail.  An
    uncertain return leaves a durable intent and prohibits retry; operators
    must reconcile the exact identity instead of replaying the request.
    """
    bound = _identity(plan, identity)
    canonical = hardening.training_operation_root(plan)
    journal = canonical / "PROD9_DIRECT_RAYJOB_CREATE.jsonl"
    if journal.exists() or journal.is_symlink():
        raise JobsError("prod9 create intent exists; reconcile, never retry")
    auth = _validate_seal(authorization, AUTHORIZATION_SCHEMA)
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or directory.resolve() != canonical
        or Path(auth.get("operation_root", "")) != canonical
    ):
        raise JobsError("prod9 create directory differs from its sealed operation root")
    stage = auth.get("stage")
    if not isinstance(stage, dict):
        raise JobsError("prod9 direct authorization lacks its sealed stage specification")
    if auth != _direct_authorization(
        plan,
        request,
        source_preview,
        expected,
        stage,
        auth["stage_authorization"],
        auth["stage_created"],
        auth["stage_receipt"],
        auth["stage_release"],
        auth["preflight_authorization"],
        auth["preflight_created"],
        auth["preflight_receipt"],
        auth["preflight_release"],
        dev_preview=auth["dev_preview"],
        prod_preview=auth["prod_preview"],
        observer=auth["observer"],
        identity=bound,
        require_live_observer=True,
    ):
        raise JobsError("prod9 direct authorization changed")
    if stage.get("sha256") != auth["stage_authorization"].get("stage_spec_sha256"):
        raise JobsError("prod9 direct stage specification digest changed")
    duplicate = _direct_duplicate_checks(
        bound, token=token, runner=runner, jobs_factory=jobs_factory
    )
    arguments = plan.get("arguments", {})
    if not isinstance(arguments, dict) or wandb_exists(
        arguments.get("wandb_entity", ""),
        arguments.get("wandb_project", ""),
        arguments.get("wandb_run_id", ""),
    ):
        raise JobsError("W&B run ID already exists")
    rendered = server_dry_run(expected, context=PROD_CONTEXT, runner=runner)
    validate_preview(
        plan,
        request,
        source_preview,
        expected,
        rendered,
        context=PROD_CONTEXT,
        identity=bound,
    )
    # Capacity is the final external-state read.  Everything after it is a
    # local liveness/freshness check, durable intent, and the sole create.
    capacity = hardening.capacity_gate(
        plan, request, expected, identity=bound, reader=capacity_reader
    )
    _observer_pid(
        auth["observer"],
        kind="rayjob",
        name=bound.run_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(expected),
        gpus=8,
        maximum_seconds=MAXIMUM_SECONDS,
    )
    _fresh_at(capacity.get("observed_at"), maximum_age=hardening.CAPACITY_MAX_AGE_SECONDS)
    _fresh_at(auth["preflight_release"].get("release_observed_at"))
    for preview in (auth["dev_preview"], auth["prod_preview"]):
        _fresh_at(preview.get("checked_at"))
    _write_once_fsynced(
        journal,
        {
            "state": "CREATE_INTENT_DO_NOT_RETRY",
            "plan_sha256": "sha256:" + digest(plan),
            "request_sha256": "sha256:" + digest(request),
            "manifest_sha256": "sha256:" + digest(expected),
            "authorization_sha256": auth["sha256"],
            "capacity_gate": capacity,
            "duplicate_checks": duplicate,
            "wandb_run_id_absent": True,
        },
    )
    # This is the sole mutating command.  There is intentionally no retry.
    result = runner(
        [
            "kubectl",
            "--context",
            PROD_CONTEXT,
            "--namespace",
            NAMESPACE,
            "create",
            "-f",
            "-",
            "-o",
            "json",
        ],
        input=json.dumps(expected),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise JobsError("prod9 create returned failure; reconcile intent, never retry")
    try:
        created = json.loads(result.stdout)
        uid = str(UUID(created["metadata"]["uid"]))
        created_at = created["metadata"]["creationTimestamp"]
        _timestamp(created_at)
    except (KeyError, TypeError, ValueError, JobsError) as exc:
        raise JobsError(
            "prod9 create response is ambiguous; reconcile intent, never retry"
        ) from exc
    if created.get("metadata", {}).get("name") != bound.run_name:
        raise JobsError("prod9 create returned another identity; reconcile intent, never retry")
    _publish_creator_binding(
        auth["observer"],
        kind="rayjob",
        name=bound.run_name,
        plan_sha256="sha256:" + digest(plan),
        manifest_sha256="sha256:" + digest(expected),
        uid=uid,
    )
    proof = _seal(
        {
            "schema": CREATED_SCHEMA,
            "status": "created_once",
            "plan_sha256": "sha256:" + digest(plan),
            "manifest_sha256": "sha256:" + digest(expected),
            "authorization_sha256": auth["sha256"],
            "capacity_gate_sha256": capacity["sha256"],
            "rayjob_name": bound.run_name,
            "rayjob_uid": uid,
            "created_at": created_at,
        }
    )
    with journal.open("a") as stream:
        stream.write(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return proof


def live_create_is_available() -> bool:
    """The fresh prod9 one-create rail is present; callers still need live gates."""
    return True
