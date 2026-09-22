"""Non-submitting, current-API launch gates for the lane2 SkyRL canary."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyber_post_train.jobs import (
    FAILURE_ALERT_ANNOTATION,
    FAILURE_ALERT_OFF,
    JobsError,
    bundled_request,
    digest,
)

from . import skyrl_lane2_training as training
from . import skyrl_prod9_direct as shared

PACKET_SCHEMA = "cyber_skyrl_lane2_direct_rayjob_packet_v1"
PREVIEW_SCHEMA = "cyber_skyrl_lane2_direct_rayjob_preview_v1"
DATA_PREVIEW_SCHEMA = "cyber_skyrl_lane2_data_job_preview_v1"
NAMESPACE = shared.NAMESPACE
DEV_CONTEXT = shared.DEV_CONTEXT
PROD_CONTEXT = shared.PROD_CONTEXT
PREFLIGHT_RECEIPT = shared.PREFLIGHT_RECEIPT
MAXIMUM_SECONDS = shared.MAXIMUM_SECONDS


def _identity(plan: dict[str, Any]) -> tuple[str, str]:
    if plan.get("schema") != training.SCHEMA:
        raise JobsError("lane2 direct rail requires the current lane2 runtime")
    binding = plan.get("qualification", {})
    name = plan.get("run_name")
    preflight = binding.get("preflight_name")
    if (
        name != "chris-q38-rlreward-lane2-v1"
        or plan.get("output_root") != "/mnt/sfs/jobs/chris-q38-rlreward-lane2-v1"
        or plan.get("arguments", {}).get("wandb_run_id") != name
        or preflight != "chris-q38-lane2-preflight-v1"
        or binding.get("submission_gate", {}).get("submission_authorized") is not False
    ):
        raise JobsError("lane2 fresh identity changed")
    return name, preflight


def manifest(
    plan: dict[str, Any],
    request: dict[str, Any],
    preview: dict[str, Any],
    *,
    image_identity_receipt: dict[str, Any] | None = None,
) -> dict:
    """Accept the exact current Jobs API preview without transforming it."""
    name, _ = _identity(plan)
    if training.job_request(plan) != request:
        raise JobsError("lane2 plan/request identity changed")
    bound_preview, runtime_identity_source = shared._runtime_bound_preview(
        request, preview, image_identity_receipt
    )
    source = shared._source(preview)
    placeholder = name + "-00000000"
    metadata = source.get("metadata", {})
    labels, annotations = metadata.get("labels", {}), metadata.get("annotations", {})
    if (
        preview.get("name") != placeholder
        or source.get("apiVersion") != "ray.io/v1"
        or source.get("kind") != "RayJob"
        or metadata.get("name") != placeholder
        or metadata.get("namespace") != NAMESPACE
        or labels.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or labels.get("fleet.ai/run-name") != name
        or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
        or labels.get("kueue.x-k8s.io/priority-class") != "q1"
        or annotations.get("fleet.ai/run-id") != "00000000-0000-0000-0000-000000000000"
        or annotations.get("fleet.ai/run-dir") != plan["output_root"]
        or annotations.get("fleet.ai/job-image") != request["image"]
        or annotations.get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise JobsError(
            "lane2 Jobs API preview lacks the exact root alert-off or admission contract"
        )
    training.validate_preview(plan, request, bound_preview)
    spec = source["spec"]
    if (
        spec.get("entrypoint") != request["command"]
        or spec.get("suspend") is not True
        or spec.get("shutdownAfterJobFinishes") is not True
        or spec.get("submissionMode") != "HTTPMode"
        or spec.get("backoffLimit") not in (None, 0)
    ):
        raise JobsError("lane2 Jobs preview execution changed")
    cluster = spec.get("rayClusterSpec", {})
    if cluster.get("workerGroupSpecs") not in (None, []):
        raise JobsError("lane2 must remain one physical GPU node")
    head = cluster.get("headGroupSpec", {}).get("template", {}).get("spec", {})
    containers = head.get("containers", [])
    if len(containers) != 1:
        raise JobsError("lane2 preview must contain one head container")
    container = containers[0]
    generated_secret = placeholder + "-fleet-key"
    secret_names = [row.get("secretRef", {}).get("name") for row in container.get("envFrom", [])]
    if secret_names != ["fleet-api", "wandb-api", generated_secret]:
        raise JobsError("lane2 Jobs preview secret bindings changed")
    if (
        container.get("securityContext") != shared._runtime_context()
        and runtime_identity_source != "exact_image_default_receipt"
    ):
        raise JobsError("lane2 Jobs preview runtime security context changed")
    if container.get("terminationMessagePath", "/dev/termination-log") != ("/dev/termination-log"):
        raise JobsError("lane2 Jobs preview termination receipt path changed")
    init = head.get("initContainers", [])
    sfs = [item for item in init if item.get("name") == "sfs-init"]
    if len(sfs) != 1 or sfs[0].get("command") != [
        "sh",
        "-c",
        f"mkdir -p {plan['output_root']} && chown 1000:100 {plan['output_root']}",
    ]:
        raise JobsError("lane2 output initialization changed")
    return source


def packet(
    plan: dict,
    request: dict,
    preview: dict,
    *,
    image_identity_receipt: dict[str, Any] | None = None,
) -> dict:
    value = manifest(plan, request, preview, image_identity_receipt=image_identity_receipt)
    body = {
        "schema": PACKET_SCHEMA,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "jobs_preview_sha256": digest(preview),
        "manifest_sha256": digest(value),
        "name": preview["name"],
        "run_name_prefix": plan["run_name"],
        "namespace": NAMESPACE,
        "failure_alerts": "off",
        "submitted": False,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def preflight_job_manifest(plan: dict[str, Any]) -> dict[str, Any]:
    """Render only the exact-image CPU preflight Job; never submit it."""
    _, preflight_name = _identity(plan)
    request = training.preflight_request(plan, receipt=PREFLIGHT_RECEIPT)
    if (
        request.get("failureAlerts") is not False
        or request.get("priority_class") != "c1"
        or request.get("workers") != 1
        or request.get("gpus_per_worker") != 8
        or request.get("image") != training.authority.IMAGE
    ):
        raise JobsError("lane2 CPU preflight request changed")
    job = shared._cpu_job(
        preflight_name,
        request["image"],
        request["command"],
        {**request["env"], "RUN_DIR": "/tmp"},
        wandb=True,
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    if (
        request.get("secrets") != ["fleet-api", "wandb-api"]
        or "nvidia.com/gpu" in json.dumps(job, sort_keys=True)
        or job["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
        or container.get("envFrom") != [{"secretRef": {"name": "wandb-api"}}]
    ):
        raise JobsError("lane2 CPU preflight root resource/alert contract changed")
    return job


def data_job_manifest() -> dict[str, Any]:
    """Render the GET-only private-data constructor; never submit it."""
    root = Path(__file__).resolve().parents[1]
    authority_value = training.authority.validate_authority(
        root / "configs/qualification/qwen38-skyrl-lane2-authority-v1.json"
    )
    name = authority_value["execution"]["data_stage_name"]
    paths = (
        "configs/qualification/qwen38-skyrl-lane2-authority-v1.json",
        "configs/qualification/qwen38-skyrl-lane2-data-v1.json",
        "configs/data/qwen38-skyrl-production-task-set-v1.json",
        "configs/data/qwen38-skyrl-production-split-v1.json",
        "configs/data/qwen38-skyrl-lane2-task-set-v1.json",
        "configs/data/qwen38-skyrl-lane2-split-v1.json",
        "configs/data/qwen-blackbox-eligible-v1.json",
        "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json",
        "configs/models/qwen38-27b-1d4bf0f2.lock.json",
        "evals/fleet/configs/opencode-easiest-train100-selection-v2.json",
    )
    files = training._runtime()
    files.update(
        {
            package + "/__init__.py": ""
            for package in ("training", "evals", "evals/fleet", "cyber_post_train")
        }
    )
    files.update({path: (root / path).read_text() for path in paths})
    bundled = bundled_request(
        {
            "name": name,
            "title": name + " zero-GPU GET-only data construction",
            "run_dir": "/mnt/sfs/jobs/" + name,
            "image": training.authority.IMAGE,
            "workers": 1,
            "gpus_per_worker": 1,
            "resources": {
                "cpu_request": "4",
                "cpu_limit": "8",
                "memory_request": "32Gi",
                "memory_limit": "48Gi",
            },
            "priority_class": "c1",
            "requeueIfPreempted": False,
            "failureAlerts": False,
            "secrets": ["fleet-api"],
            "env": {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            },
        },
        files,
        "training.skyrl_lane2_data",
        [
            "--config",
            "configs/qualification/qwen38-skyrl-lane2-data-v1.json",
            "--receipt",
            PREFLIGHT_RECEIPT,
        ],
    )
    job = shared._cpu_job(
        name,
        training.authority.IMAGE,
        bundled["command"],
        {**bundled["env"], "RUN_DIR": "/tmp"},
        wandb=False,
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["envFrom"] = [{"secretRef": {"name": "fleet-api"}}]
    queue_labels = {
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    job["metadata"]["labels"] = copy.deepcopy(queue_labels)
    job["spec"]["suspend"] = True
    job["spec"]["template"]["metadata"]["labels"] = copy.deepcopy(queue_labels)
    if (
        "nvidia.com/gpu" in json.dumps(job, sort_keys=True)
        or job["metadata"]["annotations"] != {FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF}
        or job["metadata"]["labels"] != queue_labels
        or job["spec"].get("suspend") is not True
        or job["spec"]["template"]["spec"]["priorityClassName"] != "c1"
    ):
        raise JobsError("lane2 data Job resource/alert contract changed")
    return job


def validate_server_preview(
    plan: dict,
    request: dict,
    source_preview: dict,
    expected: dict,
    rendered: dict,
    *,
    context: str,
    image_identity_receipt: dict[str, Any] | None = None,
) -> dict:
    if expected != manifest(
        plan,
        request,
        source_preview,
        image_identity_receipt=image_identity_receipt,
    ):
        raise JobsError("lane2 direct packet changed")
    if (
        context not in {DEV_CONTEXT, PROD_CONTEXT}
        or shared._strip_server_defaults(expected, rendered) != expected
        or expected["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
    ):
        raise JobsError("lane2 server dry-run changed the direct RayJob")
    body = {
        "schema": PREVIEW_SCHEMA,
        "status": "passed",
        "context": context,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "manifest_sha256": digest(expected),
        "server_render_sha256": digest(rendered),
        "name": source_preview["name"],
        "run_name_prefix": plan["run_name"],
        "nodes": 1,
        "gpus": 8,
        "priority": "c1",
        "queue_priority": "q1",
        "failure_alerts": "off",
        "submitted": False,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def validate_data_server_preview(
    expected: dict[str, Any], rendered: dict[str, Any], *, context: str
) -> dict[str, Any]:
    """Prove Kueue preserves the exact alert-safe, suspended data Job.

    The shared prod9 validator owns the Kubernetes default surface.  Normalize
    only the two explicit queue labels and explicit suspension before invoking
    it; every other unexpected server mutation remains fatal.
    """
    queue_labels = {
        "kueue.x-k8s.io/queue-name": "training-lq",
        "kueue.x-k8s.io/priority-class": "q1",
    }
    if expected != data_job_manifest():
        raise JobsError("lane2 data Job changed before server preview")
    if (
        context not in {DEV_CONTEXT, PROD_CONTEXT}
        or expected.get("metadata", {}).get("labels") != queue_labels
        or expected.get("spec", {}).get("suspend") is not True
        or expected.get("spec", {}).get("template", {}).get("metadata", {}).get("labels")
        != queue_labels
    ):
        raise JobsError("lane2 data Job queue binding changed")

    normalized_expected = copy.deepcopy(expected)
    normalized_rendered = copy.deepcopy(rendered)
    for index, value in enumerate((normalized_expected, normalized_rendered)):
        root_labels = value.get("metadata", {}).get("labels", {})
        template_labels = (
            value.get("spec", {}).get("template", {}).get("metadata", {}).get("labels", {})
        )
        if any(root_labels.get(key) != item for key, item in queue_labels.items()) or any(
            template_labels.get(key) != item for key, item in queue_labels.items()
        ):
            raise JobsError("lane2 data Job server preview changed queue labels")
        for key in queue_labels:
            root_labels.pop(key)
            template_labels.pop(key)
        # With explicit queue labels the live API server keeps only those
        # labels on the Job object, while still adding the controller labels
        # to the Pod template.  The shared validator models an otherwise
        # identical unlabelled Job, whose root receives that generated set.
        # Copy only the already-server-generated template set into the
        # normalized rendered root; the shared validator still proves every
        # key/value and rejects any additional label.
        if index == 1 and not root_labels and template_labels:
            root_labels.update(template_labels)
        if not root_labels:
            value["metadata"].pop("labels")
        if not template_labels:
            value["spec"]["template"]["metadata"].pop("labels")
    if normalized_rendered.get("spec", {}).get("suspend") is not True:
        raise JobsError("lane2 data Job server preview bypassed Kueue suspension")
    normalized_expected["spec"].pop("suspend")
    normalized_rendered["spec"]["suspend"] = False
    shared.validate_cpu_preview(
        normalized_expected,
        normalized_rendered,
        context=context,
        purpose="stage",
    )
    body = {
        "schema": DATA_PREVIEW_SCHEMA,
        "status": "passed",
        "context": context,
        "name": expected["metadata"]["name"],
        "manifest_sha256": "sha256:" + digest(expected),
        "server_render_sha256": "sha256:" + digest(rendered),
        "gpus": 0,
        "priority": "c1",
        "queue_priority": "q1",
        "failure_alerts": "off",
        "submitted": False,
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def live_create_is_available() -> bool:
    return False
