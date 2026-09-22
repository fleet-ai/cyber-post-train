"""Non-submitting root-annotated manifests for the lane2 SkyRL canary."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml

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


def _run_id(plan: dict[str, Any]) -> str:
    name, _ = _identity(plan)
    return str(uuid5(NAMESPACE_URL, f"fleet-direct-rayjob:{name}:{digest(plan)}"))


def manifest(plan: dict[str, Any], request: dict[str, Any], preview: dict[str, Any]) -> dict:
    """Project a Jobs preview into one root-annotated one-node RayJob."""
    name, _ = _identity(plan)
    if training.job_request(plan) != request:
        raise JobsError("lane2 plan/request identity changed")
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
        or FAILURE_ALERT_ANNOTATION in annotations
    ):
        raise JobsError("lane2 Jobs preview identity or admission changed")
    result = copy.deepcopy(source)
    run_id = _run_id(plan)
    result["metadata"]["name"] = name
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
        raise JobsError("lane2 Jobs preview execution changed")
    spec["backoffLimit"] = 0
    spec["activeDeadlineSeconds"] = MAXIMUM_SECONDS
    cluster = spec.get("rayClusterSpec", {})
    if cluster.get("workerGroupSpecs") not in (None, []):
        raise JobsError("lane2 must remain one physical GPU node")
    head = cluster.get("headGroupSpec", {}).get("template", {}).get("spec", {})
    containers = head.get("containers", [])
    if len(containers) != 1:
        raise JobsError("lane2 preview must contain one head container")
    container = containers[0]
    shared._replace_env(container, "FLEET_RUN_ID", run_id)
    shared._replace_env(container, "FLEET_RUN_NAME", name)
    generated_secret = placeholder + "-fleet-key"
    secret_names = [row.get("secretRef", {}).get("name") for row in container.get("envFrom", [])]
    if secret_names != ["fleet-api", "wandb-api", generated_secret]:
        raise JobsError("lane2 Jobs preview secret bindings changed")
    container["envFrom"] = [
        row
        for row in container["envFrom"]
        if row.get("secretRef", {}).get("name") != generated_secret
    ]
    container["securityContext"] = shared._runtime_context()
    init = head.get("initContainers", [])
    sfs = [item for item in init if item.get("name") == "sfs-init"]
    if len(sfs) != 1 or sfs[0].get("command") != [
        "sh",
        "-c",
        f"mkdir -p {plan['output_root']} && chown 1000:100 {plan['output_root']}",
    ]:
        raise JobsError("lane2 output initialization changed")
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


def packet(plan: dict, request: dict, preview: dict) -> dict:
    value = manifest(plan, request, preview)
    body = {
        "schema": PACKET_SCHEMA,
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "jobs_preview_sha256": digest(preview),
        "manifest_sha256": digest(value),
        "direct_run_id": _run_id(plan),
        "name": plan["run_name"],
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
    )
    if "nvidia.com/gpu" in json.dumps(job, sort_keys=True) or (
        job["metadata"]["annotations"].get(FAILURE_ALERT_ANNOTATION) != FAILURE_ALERT_OFF
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
        "configs/data/qwen38-skyrl-lane2-task-set-v1.json",
        "configs/data/qwen38-skyrl-lane2-split-v1.json",
        "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json",
        "configs/models/qwen38-27b-1d4bf0f2.lock.json",
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
        ["--config", "configs/qualification/qwen38-skyrl-lane2-data-v1.json"],
    )
    job = shared._cpu_job(
        name,
        training.authority.IMAGE,
        bundled["command"],
        {**bundled["env"], "RUN_DIR": "/tmp"},
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    container["envFrom"] = [{"secretRef": {"name": "fleet-api"}}]
    if (
        "nvidia.com/gpu" in json.dumps(job, sort_keys=True)
        or job["metadata"]["annotations"] != {FAILURE_ALERT_ANNOTATION: FAILURE_ALERT_OFF}
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
) -> dict:
    if expected != manifest(plan, request, source_preview):
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
        "name": plan["run_name"],
        "nodes": 1,
        "gpus": 8,
        "priority": "c1",
        "queue_priority": "q1",
        "failure_alerts": "off",
        "submitted": False,
    }
    return {**body, "sha256": "sha256:" + digest(body)}


def live_create_is_available() -> bool:
    return False
