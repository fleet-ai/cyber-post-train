"""Read-only collector for the terminal zero-step HF export execution."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from .io import digest_json
from .post_sft import normalize_zero_step_stored_config, render_zero_step_sft_command

NAMESPACE = "fleet-train-jobs"
QUEUE_NAME = "training-lq"
DEFAULT_SERVICE_ACCOUNT = "default"
SUBMITTER_IMAGE = "anyscale/ray:2.56.0-slim-py312"
OPTIMIZER_EVENT_PATTERN = re.compile(
    r"Started:\s*['\"]optim_step['\"]|\boptimizer(?:_|\s+)step(?:s)?\b",
    re.IGNORECASE,
)
TERMINAL_SUCCESS_PATTERN = re.compile(
    r"\bJob finished successfully\b|\bjobStatus\s*[=:]\s*SUCCEEDED\b|\brun succeeded\b",
    re.IGNORECASE,
)
STEP_PATTERNS = (
    re.compile(r"\bStep\s+(\d+)\s*:", re.IGNORECASE),
    re.compile(
        r"(?:global_step|global step|current_step|current step|latest_step|latest step)"
        r"\s*[=:]\s*(\d+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"['\"](?:global_step|current_step|latest_step|step)['\"]\s*:\s*(\d+)",
        re.IGNORECASE,
    ),
)
def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Mapping[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{field} must be a non-empty string")
    return result


def _embedded_digest(value: Mapping[str, Any], field: str) -> str:
    expected = _text(value, field)
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    if digest_json(unsigned) != expected:
        raise ValueError(f"{field} digest mismatch")
    return expected


def _image_digest(value: str, field: str) -> str:
    match = re.search(r"sha256:[0-9a-f]{64}$", value)
    if not match:
        raise ValueError(f"{field} does not carry a resolved sha256 image digest")
    return match.group(0)


def _container(spec: Mapping[str, Any], expected_image: str, field: str) -> Mapping[str, Any]:
    containers = spec.get("containers")
    if not isinstance(containers, list):
        raise ValueError(f"{field} containers must be an array")
    matches = [
        row
        for row in containers
        if isinstance(row, Mapping) and row.get("image") == expected_image
    ]
    if len(matches) != 1:
        raise ValueError(f"{field} must contain exactly one pinned trainer container")
    return matches[0]


def _validated_stored_config(
    request: Mapping[str, Any],
    export_run: Mapping[str, Any],
    observed: Mapping[str, Any],
    field: str,
) -> dict[str, Any]:
    expected = normalize_zero_step_stored_config(request, export_run)
    if dict(observed) != expected:
        differing = sorted(
            key
            for key in set(expected) | set(observed)
            if expected.get(key) != observed.get(key)
        )
        raise ValueError(f"{field} differs from exact server-normalized config at {differing}")
    return copy.deepcopy(dict(observed))


def _fleet_run_config(cluster_spec: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    head = _mapping(cluster_spec.get("headGroupSpec"), f"{field} head group")
    pod_spec = _mapping(
        _mapping(head.get("template"), f"{field} head template").get("spec"),
        f"{field} head Pod spec",
    )
    container = _container_by_name(pod_spec, "ray-head", f"{field} head")
    env = container.get("env")
    if not isinstance(env, list):
        raise ValueError(f"{field} head container has no environment")
    matches = [
        row
        for row in env
        if isinstance(row, Mapping) and row.get("name") == "FLEET_RUN_CONFIG"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("value"), str):
        raise ValueError(f"{field} has no exact FLEET_RUN_CONFIG value")
    try:
        value = json.loads(matches[0]["value"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} FLEET_RUN_CONFIG is not valid JSON") from exc
    return _mapping(value, f"{field} FLEET_RUN_CONFIG")


def _owner(
    metadata: Mapping[str, Any], *, kind: str, name: str, uid: str, field: str
) -> None:
    owners = metadata.get("ownerReferences")
    matches = [
        row
        for row in owners or []
        if isinstance(row, Mapping)
        and row.get("kind") == kind
        and row.get("name") == name
        and row.get("uid") == uid
        and row.get("controller") is True
    ]
    if len(matches) != 1:
        raise ValueError(f"{field} does not have the exact controller ownership")


def _effective_service_account(spec: Mapping[str, Any]) -> str:
    value = spec.get("serviceAccountName")
    return DEFAULT_SERVICE_ACCOUNT if value in {None, ""} else str(value)


def _resources(container: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    return _mapping(container.get("resources"), f"{field} resources")


def _exact_resources(
    container: Mapping[str, Any], expected: Mapping[str, Any], field: str
) -> None:
    if dict(_resources(container, field)) != dict(expected):
        raise ValueError(f"{field} resources differ from the reviewed execution policy")


def _container_by_name(spec: Mapping[str, Any], name: str, field: str) -> Mapping[str, Any]:
    containers = spec.get("containers")
    if not isinstance(containers, list):
        raise ValueError(f"{field} containers must be an array")
    matches = [row for row in containers if isinstance(row, Mapping) and row.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"{field} must contain exactly one {name} container")
    if len(containers) != 1:
        raise ValueError(f"{field} contains an unreviewed extra container")
    return matches[0]


def _ray_cluster_projection(
    cluster_spec: Mapping[str, Any], trainer_image: str, request: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and project every execution-relevant, reviewed RayCluster field."""

    expected_head_resources = {
        "requests": {"cpu": "2", "memory": "8Gi"},
        "limits": {"cpu": "4", "memory": "16Gi"},
    }
    expected_worker_resources = {
        "requests": {"cpu": "184", "memory": "2560Gi", "nvidia.com/gpu": "8"},
        "limits": {"cpu": "184", "memory": "2560Gi", "nvidia.com/gpu": "8"},
    }
    head = _mapping(cluster_spec.get("headGroupSpec"), "RayCluster head group")
    head_spec = _mapping(
        _mapping(_mapping(head.get("template"), "RayCluster head template").get("spec"),
                 "RayCluster head Pod spec"),
        "RayCluster head Pod spec",
    )
    head_container = _container_by_name(head_spec, "ray-head", "RayCluster head")
    if head_container.get("image") != trainer_image:
        raise ValueError("RayCluster head image differs from the pinned trainer image")
    if head_container.get("command") not in (None, []) or head_container.get("args") not in (
        None,
        [],
    ):
        raise ValueError("RayCluster head has an unreviewed container command")
    _exact_resources(head_container, expected_head_resources, "RayCluster head")

    workers = cluster_spec.get("workerGroupSpecs")
    if not isinstance(workers, list) or len(workers) != 1:
        raise ValueError("RayCluster must contain exactly one reviewed worker group")
    worker = _mapping(workers[0], "RayCluster worker group")
    expected_replicas = request.get("num_workers")
    if expected_replicas != 1 or any(
        worker.get(field) != expected_replicas
        for field in ("replicas", "minReplicas", "maxReplicas")
    ):
        raise ValueError("RayCluster worker replicas differ from the reviewed request")
    if worker.get("groupName") != "gpu-worker":
        raise ValueError("RayCluster worker group name differs from the reviewed policy")
    worker_spec = _mapping(
        _mapping(worker.get("template"), "RayCluster worker template").get("spec"),
        "RayCluster worker Pod spec",
    )
    worker_container = _container_by_name(worker_spec, "ray-worker", "RayCluster worker")
    if worker_container.get("image") != trainer_image:
        raise ValueError("RayCluster worker image differs from the pinned trainer image")
    if worker_container.get("command") not in (None, []) or worker_container.get("args") not in (
        None,
        [],
    ):
        raise ValueError("RayCluster worker has an unreviewed container command")
    _exact_resources(worker_container, expected_worker_resources, "RayCluster worker")
    if request.get("gpus_per_worker") != 8:
        raise ValueError("export request GPU count differs from the reviewed execution policy")

    projection = {
        "head": {
            "service_account_name": _effective_service_account(head_spec),
            "image": trainer_image,
            "command": head_container.get("command"),
            "args": head_container.get("args"),
            "resources": copy.deepcopy(dict(_resources(head_container, "RayCluster head"))),
        },
        "worker": {
            "group_name": worker.get("groupName"),
            "replicas": worker.get("replicas"),
            "min_replicas": worker.get("minReplicas"),
            "max_replicas": worker.get("maxReplicas"),
            "service_account_name": _effective_service_account(worker_spec),
            "image": trainer_image,
            "command": worker_container.get("command"),
            "args": worker_container.get("args"),
            "resources": copy.deepcopy(dict(_resources(worker_container, "RayCluster worker"))),
        },
    }
    if projection["head"]["service_account_name"] != DEFAULT_SERVICE_ACCOUNT or projection[
        "worker"
    ]["service_account_name"] != DEFAULT_SERVICE_ACCOUNT:
        raise ValueError("RayCluster service account differs from the reviewed policy")
    return projection


def _rayjob_submitter_projection(spec: Mapping[str, Any]) -> dict[str, Any]:
    template = _mapping(spec.get("submitterPodTemplate"), "RayJob submitter template")
    pod_spec = _mapping(template.get("spec"), "RayJob submitter Pod spec")
    container = _container_by_name(pod_spec, "ray-job-submitter", "RayJob submitter")
    expected_resources = {
        "requests": {"cpu": "200m", "ephemeral-storage": "2Gi", "memory": "512Mi"},
        "limits": {"cpu": "1", "memory": "1Gi"},
    }
    if (
        container.get("image") != SUBMITTER_IMAGE
        or _effective_service_account(pod_spec) != DEFAULT_SERVICE_ACCOUNT
        or container.get("command") not in (None, [])
        or container.get("args") not in (None, [])
    ):
        raise ValueError("RayJob submitter template differs from the reviewed execution policy")
    _exact_resources(container, expected_resources, "RayJob submitter")
    return {
        "service_account_name": _effective_service_account(pod_spec),
        "image": container.get("image"),
        "command": container.get("command"),
        "args": container.get("args"),
        "resources": copy.deepcopy(dict(_resources(container, "RayJob submitter"))),
    }


def _api_zero_step_evidence(api_run: Mapping[str, Any], resume_step: int) -> dict[str, Any]:
    """Reject optimizer evidence and any reported progress beyond the resume boundary."""

    checked: dict[str, Any] = {}

    def inspect_value(value: Any, path: str) -> None:
        key = path.rsplit(".", 1)[-1].lower()
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                inspect_value(child, f"{path}.{child_key}" if path else str(child_key))
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                inspect_value(child, f"{path}[{index}]")
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return
        if "optimizer" in key and "step" in key:
            if value != 0:
                raise ValueError(f"Jobs API reports optimizer activity at {path}")
            checked[path] = value
        elif "step" in key and any(
            token in key for token in ("step", "global", "current", "latest")
        ):
            if value > resume_step:
                raise ValueError(f"Jobs API reports a step beyond the resume boundary at {path}")
            checked[path] = value

    for field in (
        "step",
        "steps",
        "step_metrics",
        "metrics",
        "current_step",
        "latest_step",
        "global_step",
        "optimizer_step",
        "optimizer_steps",
        "current_metrics",
        "latest_metrics",
    ):
        if field in api_run:
            inspect_value(api_run[field], field)
    return checked


def _log_zero_step_evidence(driver_log: bytes, entrypoint: str, resume_step: int) -> dict[str, Any]:
    if not driver_log:
        raise ValueError("driver log is empty")
    log_text = driver_log.decode("utf-8", errors="replace")
    if entrypoint not in log_text:
        raise ValueError("driver log does not contain the exact RayJob entrypoint")
    if not TERMINAL_SUCCESS_PATTERN.search(log_text):
        raise ValueError("driver log does not contain a terminal success marker")
    optimizer_events = list(OPTIMIZER_EVENT_PATTERN.finditer(log_text))
    if optimizer_events:
        raise ValueError("driver log records an optimizer event")
    observed_steps = sorted(
        {int(match.group(1)) for pattern in STEP_PATTERNS for match in pattern.finditer(log_text)}
    )
    later_steps = [step for step in observed_steps if step > resume_step]
    if later_steps:
        raise ValueError("driver log records steps after the resume boundary")
    return {
        "sha256": "sha256:" + hashlib.sha256(driver_log).hexdigest(),
        "bytes": len(driver_log),
        "contains_exact_entrypoint": True,
        "contains_terminal_success": True,
        "optimizer_events": 0,
        "observed_steps": observed_steps,
        "steps_after_resume": [],
    }


def collect_zero_step_export_run_observation(
    export_request: Mapping[str, Any],
    terminal_rayjob: Mapping[str, Any],
    runtime_raycluster: Mapping[str, Any],
    runtime_pod: Mapping[str, Any],
    submitter_job: Mapping[str, Any],
    submitter_pod: Mapping[str, Any],
    api_run: Mapping[str, Any],
    driver_log: bytes,
) -> dict[str, Any]:
    """Prove the terminal export ran the exact frozen command in the pinned image.

    ``runtime_pod`` is captured while the Ray cluster exists; KubeRay normally removes the trainer
    Pods after terminal completion. ``terminal_rayjob`` and ``api_run`` are captured only after the
    run succeeds. The driver log is retained only by digest and byte count.
    """

    if export_request.get("schema") != "cyber_sft_zero_step_export_request_v1":
        raise ValueError("unsupported zero-step export request schema")
    request_sha256 = _embedded_digest(export_request, "export_request_receipt_sha256")
    request = _mapping(export_request.get("request"), "export request")
    expected = _mapping(export_request.get("expected_execution"), "expected execution")
    export_run = _mapping(export_request.get("export_run"), "export run")
    run_name = _text(export_run, "name")
    run_id = _text(export_run, "run_id")
    rayjob_uid = _text(export_run, "rayjob_uid")
    trainer_version_id = _text(export_run, "trainer_version_id")
    trainer_image = _text(export_run, "trainer_image")
    expected_image_digest = _image_digest(trainer_image, "trainer image")
    entrypoint = render_zero_step_sft_command(request, run_name)
    command_sha256 = digest_json({"entrypoint": entrypoint})
    expected_identity = {
        "request_config_sha256": digest_json(request),
        "stored_config_sha256": digest_json(
            normalize_zero_step_stored_config(request, export_run)
        ),
        "kind": "sft",
        "model_precision": "bf16",
        "strategy": "fsdp",
        "trainer_version_id": trainer_version_id,
        "command_sha256": command_sha256,
    }
    if dict(expected) != expected_identity:
        raise ValueError("export request expected execution identity is not exact")

    metadata = _mapping(terminal_rayjob.get("metadata"), "RayJob metadata")
    spec = _mapping(terminal_rayjob.get("spec"), "RayJob spec")
    status = _mapping(terminal_rayjob.get("status"), "RayJob status")
    if (
        metadata.get("namespace") != NAMESPACE
        or metadata.get("name") != run_name
        or metadata.get("uid") != rayjob_uid
    ):
        raise ValueError("terminal RayJob identity differs from the frozen export run")
    if status.get("jobStatus") != "SUCCEEDED":
        raise ValueError("export RayJob is not terminally SUCCEEDED")
    if spec.get("entrypoint") != entrypoint:
        raise ValueError("RayJob actual entrypoint differs from the frozen command")
    cluster_name = _text(status, "rayClusterName")
    labels = _mapping(metadata.get("labels"), "RayJob metadata labels")
    if labels.get("kueue.x-k8s.io/queue-name") != QUEUE_NAME:
        raise ValueError("RayJob queue differs from the reviewed queue")
    expected_job_fields = {
        "backoffLimit": 0,
        "shutdownAfterJobFinishes": True,
        "submissionMode": "K8sJobMode",
        "ttlSecondsAfterFinished": 0,
    }
    for field, value in expected_job_fields.items():
        if spec.get(field) != value:
            raise ValueError(f"RayJob {field} differs from the reviewed execution policy")
    cluster_spec = _mapping(spec.get("rayClusterSpec"), "RayJob cluster spec")
    rayjob_stored_config = _validated_stored_config(
        request,
        export_run,
        _fleet_run_config(cluster_spec, "RayJob"),
        "RayJob FLEET_RUN_CONFIG",
    )
    cluster_projection = _ray_cluster_projection(cluster_spec, trainer_image, request)
    cluster_projection["submitter"] = _rayjob_submitter_projection(spec)

    raycluster_metadata = _mapping(runtime_raycluster.get("metadata"), "RayCluster metadata")
    raycluster_spec = _mapping(runtime_raycluster.get("spec"), "RayCluster spec")
    if (
        raycluster_metadata.get("namespace") != NAMESPACE
        or raycluster_metadata.get("name") != cluster_name
    ):
        raise ValueError("runtime RayCluster identity differs from the terminal RayJob")
    raycluster_uid = _text(raycluster_metadata, "uid")
    _owner(
        raycluster_metadata,
        kind="RayJob",
        name=run_name,
        uid=rayjob_uid,
        field="runtime RayCluster",
    )
    runtime_cluster_projection = _ray_cluster_projection(raycluster_spec, trainer_image, request)
    runtime_stored_config = _validated_stored_config(
        request,
        export_run,
        _fleet_run_config(raycluster_spec, "runtime RayCluster"),
        "runtime RayCluster FLEET_RUN_CONFIG",
    )
    if runtime_stored_config != rayjob_stored_config:
        raise ValueError("runtime RayCluster FLEET_RUN_CONFIG differs from the RayJob")
    if runtime_cluster_projection != {
        key: value for key, value in cluster_projection.items() if key != "submitter"
    }:
        raise ValueError("runtime RayCluster execution spec differs from the RayJob template")

    pod_metadata = _mapping(runtime_pod.get("metadata"), "runtime Pod metadata")
    pod_spec = _mapping(runtime_pod.get("spec"), "runtime Pod spec")
    pod_status = _mapping(runtime_pod.get("status"), "runtime Pod status")
    pod_labels = _mapping(pod_metadata.get("labels"), "runtime Pod labels")
    if (
        pod_metadata.get("namespace") != NAMESPACE
        or pod_labels.get("ray.io/cluster") != cluster_name
    ):
        raise ValueError("runtime Pod does not belong to the RayJob cluster")
    if pod_labels.get("ray.io/node-type") != "head":
        raise ValueError("runtime evidence must come from the Ray head Pod")
    _owner(
        pod_metadata,
        kind="RayCluster",
        name=cluster_name,
        uid=raycluster_uid,
        field="runtime head Pod",
    )
    pod_container = _container(pod_spec, trainer_image, "runtime Pod")
    container_name = _text(pod_container, "name")
    statuses = pod_status.get("containerStatuses")
    if not isinstance(statuses, list):
        raise ValueError("runtime Pod containerStatuses must be an array")
    matches = [
        row
        for row in statuses
        if isinstance(row, Mapping) and row.get("name") == container_name
    ]
    if len(matches) != 1:
        raise ValueError("runtime Pod has no unique trainer container status")
    resolved_image_id = _text(matches[0], "imageID")
    resolved_image_digest = _image_digest(resolved_image_id, "runtime Pod imageID")
    if resolved_image_digest != expected_image_digest:
        raise ValueError("runtime Pod resolved a different trainer image digest")
    if pod_status.get("phase") not in {"Running", "Succeeded"}:
        raise ValueError("runtime Pod snapshot was not captured from a running trainer")

    submitter_metadata = _mapping(submitter_job.get("metadata"), "submitter Job metadata")
    submitter_spec = _mapping(submitter_job.get("spec"), "submitter Job spec")
    if (
        submitter_metadata.get("namespace") != NAMESPACE
        or submitter_metadata.get("name") != run_name
    ):
        raise ValueError("submitter Job identity differs from the frozen export run")
    submitter_uid = _text(submitter_metadata, "uid")
    _owner(
        submitter_metadata,
        kind="RayJob",
        name=run_name,
        uid=rayjob_uid,
        field="submitter Job",
    )
    submitter_template_spec = _mapping(
        _mapping(submitter_spec.get("template"), "submitter Job template").get("spec"),
        "submitter Job Pod spec",
    )
    submitter_container = _container_by_name(
        submitter_template_spec, "ray-job-submitter", "submitter Job"
    )
    if (
        submitter_container.get("image") != SUBMITTER_IMAGE
        or _effective_service_account(submitter_template_spec) != DEFAULT_SERVICE_ACCOUNT
    ):
        raise ValueError("submitter Job image or service account differs from reviewed policy")
    expected_submitter_resources = cluster_projection["submitter"]["resources"]
    _exact_resources(submitter_container, expected_submitter_resources, "submitter Job")

    submitter_pod_metadata = _mapping(
        submitter_pod.get("metadata"), "submitter Pod metadata"
    )
    if submitter_pod_metadata.get("namespace") != NAMESPACE:
        raise ValueError("submitter Pod namespace differs from the frozen export run")
    _owner(
        submitter_pod_metadata,
        kind="Job",
        name=run_name,
        uid=submitter_uid,
        field="submitter Pod",
    )
    submitter_pod_spec = _mapping(submitter_pod.get("spec"), "submitter Pod spec")
    live_submitter = _container_by_name(
        submitter_pod_spec, "ray-job-submitter", "submitter Pod"
    )
    if live_submitter.get("image") != SUBMITTER_IMAGE:
        raise ValueError("submitter Pod image differs from the reviewed policy")
    _exact_resources(live_submitter, expected_submitter_resources, "submitter Pod")
    if (
        live_submitter.get("command") != submitter_container.get("command")
        or live_submitter.get("args") != submitter_container.get("args")
    ):
        raise ValueError("submitter Pod command differs from its owning Job")

    stored_config = _mapping(api_run.get("config"), "Jobs API stored config")
    if api_run.get("name") != run_name or api_run.get("id") != run_id:
        raise ValueError("Jobs API run identity differs from the frozen export run")
    if stored_config.get("run_id") != run_id or stored_config.get("name") != run_name:
        raise ValueError("Jobs API stored config run identity differs from the frozen export run")
    if not isinstance(stored_config.get("submitted_by_email"), str) or not stored_config[
        "submitted_by_email"
    ]:
        raise ValueError("Jobs API stored config lacks the reviewed server-owned submitter field")
    if api_run.get("kind") != "sft" or str(api_run.get("status", "")).lower() != "succeeded":
        raise ValueError("Jobs API run is not terminally succeeded SFT")
    if api_run.get("trainer_version_id") != trainer_version_id:
        raise ValueError("Jobs API trainer version differs from the frozen export run")
    request_projection = _validated_stored_config(
        request, export_run, stored_config, "Jobs API stored config"
    )
    if request_projection != rayjob_stored_config:
        raise ValueError("Jobs API stored config differs from the RayJob FLEET_RUN_CONFIG")
    resume_step = export_run.get("num_steps")
    if not isinstance(resume_step, int) or resume_step < 1:
        raise ValueError("export run num_steps is invalid")
    api_step_fields = _api_zero_step_evidence(api_run, resume_step)
    log_evidence = _log_zero_step_evidence(driver_log, entrypoint, resume_step)

    observation: dict[str, Any] = {
        "schema": "cyber_sft_zero_step_export_run_observation_v1",
        "terminal_status": "SUCCEEDED",
        "run": {
            "name": run_name,
            "run_id": run_id,
            "rayjob_uid": rayjob_uid,
            "trainer_version_id": trainer_version_id,
        },
        "trainer_image": trainer_image,
        "request_identity": expected_identity,
        "rayjob": {
            "uid": rayjob_uid,
            "cluster_name": cluster_name,
            "spec_sha256": digest_json(spec),
            "reviewed_execution_projection": cluster_projection,
            "reviewed_execution_projection_sha256": digest_json(cluster_projection),
            "queue": QUEUE_NAME,
            "entrypoint": entrypoint,
            "entrypoint_sha256": command_sha256,
        },
        "raycluster": {
            "name": cluster_name,
            "uid": raycluster_uid,
            "spec_sha256": digest_json(raycluster_spec),
        },
        "pod": {
            "name": _text(pod_metadata, "name"),
            "uid": _text(pod_metadata, "uid"),
            "requested_image": trainer_image,
            "resolved_image_id": resolved_image_id,
            "resolved_image_digest": resolved_image_digest,
        },
        "submitter": {
            "job_name": run_name,
            "job_uid": submitter_uid,
            "pod_name": _text(submitter_pod_metadata, "name"),
            "pod_uid": _text(submitter_pod_metadata, "uid"),
            "container_name": "ray-job-submitter",
            "image": SUBMITTER_IMAGE,
            "job_spec_sha256": digest_json(submitter_spec),
            "command_sha256": digest_json(
                {
                    "command": submitter_container.get("command"),
                    "args": submitter_container.get("args"),
                }
            ),
        },
        "resume_from": export_run.get("resume_from"),
        "num_steps": export_run.get("num_steps"),
        "hf_save_interval": export_run.get("hf_save_interval"),
        "optimizer_steps": 0,
        "command_sha256": command_sha256,
        "output_path": _mapping(
            export_request.get("expected_output"), "export expected output"
        ).get("path"),
        "export_request_receipt_sha256": request_sha256,
        "zero_step_evidence": {
            "resume_global_step": resume_step,
            "configured_final_step": resume_step,
            "optimizer_step_events": 0,
            "logs": log_evidence,
            "metrics": {
                "api_observation_sha256": digest_json(api_run),
                "request_owned_config_sha256": digest_json(request_projection),
                "rayjob_fleet_run_config_sha256": digest_json(rayjob_stored_config),
                "checked_step_fields": api_step_fields,
                "reported_optimizer_steps": 0,
            },
        },
    }
    observation["observation_sha256"] = digest_json(observation)
    return observation
