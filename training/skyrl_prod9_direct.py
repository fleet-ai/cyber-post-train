"""Fresh, non-submitting direct-RayJob renderer for the prod9 SkyRL canary.

The historical prod8 direct rail deliberately rejects prod9's fresh runtime
schema.  This module owns the replacement *rendering* boundary: it makes the
exact-image CPU preflight Job and the one-node RayJob available for review and
server preview, with the root failed-job-alert opt-out already present.  It has
no create or submit function.  A later, separately reviewed live rail must
accept the resulting fresh preview/absence/preflight evidence before it can
perform its single create.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

import yaml

from cyber_post_train.jobs import FAILURE_ALERT_ANNOTATION, FAILURE_ALERT_OFF, JobsError, digest

from . import skyrl_prod9_training as training
from . import skyrl_reward_rayjob as historical

PACKET_SCHEMA = "cyber_skyrl_prod9_direct_rayjob_packet_v1"
PREVIEW_SCHEMA = "cyber_skyrl_prod9_direct_rayjob_preview_v1"
CPU_JOB_SCHEMA = "cyber_skyrl_prod9_cpu_preflight_job_v1"
RUNTIME_UID = 1000
RUNTIME_GID = 100
NAMESPACE = historical.NAMESPACE
DEV_CONTEXT = historical.DEV_CONTEXT
PROD_CONTEXT = historical.PROD_CONTEXT
PVC = "sfs-shared"
PREFLIGHT_RECEIPT = "/dev/termination-log"
MAXIMUM_SECONDS = historical.MAXIMUM_SECONDS


def _identity(plan: dict[str, Any], identity: historical.RailIdentity) -> historical.RailIdentity:
    """Bind public prod9 fields to the reviewed, fresh create-once identity."""
    if plan.get("schema") != training.SCHEMA:
        raise JobsError("prod9 direct rail requires the fresh runtime schema")
    return historical._identity_for_plan(plan, identity)


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


def _timestamp(value: object) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise JobsError("prod9 server dry-run timestamp is invalid")


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
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def live_create_is_available() -> bool:
    """Make the fail-closed boundary explicit for callers and dashboard code."""
    return False
