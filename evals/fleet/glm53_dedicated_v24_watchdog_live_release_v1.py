"""Post-create release rail for the v24 external idle watchdog.

The pre-create package stays held.  This module is the only renderer that may
turn the watchdog launch bit on, and only after a fresh, exact, score-free
snapshot binds every live server object to the v24 Jobs API run.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_server_v1 as server
from evals.fleet import glm53_dedicated_v24_watchdog_package_v1 as package

LIVE_STATE_SCHEMA = "fleet-glm53-dedicated-v24-watchdog-live-state-v1"
RELEASE_SCHEMA = "fleet-glm53-dedicated-v24-watchdog-live-release-v1"
AUTHORIZATION_CONFIGMAP_NAME = package.JOB_NAME + "-live-release"
MAX_OBSERVATION_AGE_SECONDS = 60
MAX_READY_AGE_SECONDS = 120
API_RUN_ID_RE = re.compile(r"ft-run-[0-9a-f]{8}")
NAMESPACE = "fleet-train-jobs"

LIVE_STATE_KEYS = {
    "schema_version",
    "status",
    "observed_at_epoch",
    "ready_at_epoch",
    "server_binding",
    "request_sha256",
    "api_get_http_status",
    "api_run_state",
    "api_title_match_count",
    "api_run_dir_match_count",
    "rayjob_running",
    "rayjob_name",
    "rayjob_uid_match_count",
    "workload_admitted",
    "workload_name",
    "workload_finished",
    "workload_preemption_events",
    "workload_uid_match_count",
    "head_pod_phase",
    "head_pod_name",
    "head_pod_ready",
    "head_pod_restarts",
    "head_pod_uid_match_count",
    "service_present",
    "service_name",
    "service_uid_match_count",
    "metrics_http_status",
    "activity_metric_families",
    "jobs_api_credential_secret_name",
    "jobs_api_credential_secret_uid",
    "jobs_api_credential_owner_rayjob_uid",
    "jobs_api_credential_key",
    "jobs_api_credential_probe_http_status",
    "watchdog_job_match_count",
    "watchdog_configmap_match_count",
    "server_run_dir_exists",
    "watchdog_result_root_absent",
    "fleet_task_instance_calls",
    "fleet_session_calls",
    "verifier_calls",
    "scoring_calls",
    "protected_content_included",
    "receipt_sha256",
}

RELEASE_KEYS = {
    "schema_version",
    "status",
    "server_binding",
    "server_binding_sha256",
    "live_state_receipt_sha256",
    "request_sha256",
    "watchdog_job_name",
    "watchdog_result_root",
    "watchdog_package_commit",
    "watchdog_package_sha256",
    "watchdog_implementation_sha256",
    "ready_at_epoch",
    "idle_release_seconds",
    "create_once",
    "server_launch_authorized",
    "watchdog_launch_authorized",
    "qualification_launch_authorized",
    "scored_launch_authorized",
    "fleet_task_instance_calls",
    "fleet_session_calls",
    "verifier_calls",
    "scoring_calls",
    "protected_content_included",
    "receipt_sha256",
}


class LiveReleaseError(RuntimeError):
    """The live v24 watcher handoff is not safe to release."""


def _request_sha256() -> str:
    value = server.payload()
    server.validate_payload(value)
    return crypto.sha256(crypto.canonical_json(value))


def _number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _integer(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _valid_uuid(value: Any) -> bool:
    try:
        return uuid.UUID(str(value)).int != 0
    except (ValueError, AttributeError):
        return False


def _validate_live_state_at(
    value: dict[str, Any], binding: dict[str, Any], *, observed_now_epoch: float
) -> None:
    """Validate a short-lived snapshot against an internally observed clock."""

    try:
        server.validate_binding(binding)
    except server.ServerPlanError as exc:
        raise LiveReleaseError("v24_live_server_binding_invalid") from exc
    observed = value.get("observed_at_epoch")
    ready = value.get("ready_at_epoch")
    api_run_id = binding.get("api_run_id")
    if (
        set(value) != LIVE_STATE_KEYS
        or value.get("schema_version") != LIVE_STATE_SCHEMA
        or value.get("status") != "READY_POST_CREATE_UID_BOUND"
        or value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
        or not _number(observed_now_epoch)
        or not _number(observed)
        or not _number(ready)
        or observed > observed_now_epoch
        or observed_now_epoch - observed > MAX_OBSERVATION_AGE_SECONDS
        or ready <= 0
        or ready > observed
        or observed - ready > MAX_READY_AGE_SECONDS
        or value.get("server_binding") != binding
        or not isinstance(api_run_id, str)
        or API_RUN_ID_RE.fullmatch(api_run_id) is None
        or value.get("request_sha256") != _request_sha256()
        or value.get("api_get_http_status") != 200
        or value.get("api_run_state") != "RUNNING"
        or not _integer(value.get("api_title_match_count"), 1)
        or not _integer(value.get("api_run_dir_match_count"), 1)
        or value.get("rayjob_running") is not True
        or value.get("rayjob_name") != api_run_id
        or not _integer(value.get("rayjob_uid_match_count"), 1)
        or value.get("workload_admitted") is not True
        or not isinstance(value.get("workload_name"), str)
        or not value["workload_name"]
        or value.get("workload_finished") is not False
        or not _integer(value.get("workload_preemption_events"), 0)
        or not _integer(value.get("workload_uid_match_count"), 1)
        or value.get("head_pod_phase") != "Running"
        or not isinstance(value.get("head_pod_name"), str)
        or not value["head_pod_name"]
        or value.get("head_pod_ready") is not True
        or not _integer(value.get("head_pod_restarts"), 0)
        or not _integer(value.get("head_pod_uid_match_count"), 1)
        or value.get("service_present") is not True
        or not isinstance(value.get("service_name"), str)
        or not value["service_name"]
        or not _integer(value.get("service_uid_match_count"), 1)
        or binding.get("service_origin")
        not in {
            f"http://{value.get('service_name')}.{NAMESPACE}.svc:8000",
            f"http://{value.get('service_name')}.{NAMESPACE}.svc.cluster.local:8000",
        }
        or value.get("metrics_http_status") != 200
        or value.get("activity_metric_families") != list(runtime.ACTIVITY_METRICS)
        or value.get("jobs_api_credential_secret_name") != f"{api_run_id}-fleet-key"
        or not _valid_uuid(value.get("jobs_api_credential_secret_uid"))
        or value.get("jobs_api_credential_owner_rayjob_uid") != binding.get("rayjob_uid")
        or value.get("jobs_api_credential_key") != "FLEET_API_KEY"
        or value.get("jobs_api_credential_probe_http_status") != 200
        or not _integer(value.get("watchdog_job_match_count"), 0)
        or not _integer(value.get("watchdog_configmap_match_count"), 0)
        or value.get("server_run_dir_exists") is not True
        or value.get("watchdog_result_root_absent") is not True
        or any(
            not _integer(value.get(field), 0)
            for field in (
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or value.get("protected_content_included") is not False
    ):
        raise LiveReleaseError("v24_watchdog_live_state_invalid")


def validate_live_state(value: dict[str, Any], binding: dict[str, Any]) -> None:
    """Validate a live snapshot using the process clock, never caller-supplied time."""

    _validate_live_state_at(value, binding, observed_now_epoch=time.time())


def build_release(
    live_state: dict[str, Any],
    binding: dict[str, Any],
    *,
    watchdog_package_commit: str,
    watchdog_package_sha256: str,
    watchdog_implementation_sha256: str,
) -> dict[str, Any]:
    validate_live_state(live_state, binding)
    if (
        re.fullmatch(r"[0-9a-f]{40}", watchdog_package_commit) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", watchdog_package_sha256) is None
        or re.fullmatch(r"sha256:[0-9a-f]{64}", watchdog_implementation_sha256) is None
    ):
        raise LiveReleaseError("v24_watchdog_package_identity_invalid")
    body: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA,
        "status": "AUTHORIZED_EXACT_WATCHDOG_ONLY",
        "server_binding": binding,
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "live_state_receipt_sha256": live_state["receipt_sha256"],
        "request_sha256": _request_sha256(),
        "watchdog_job_name": package.JOB_NAME,
        "watchdog_result_root": package.RESULT_ROOT,
        "watchdog_package_commit": watchdog_package_commit,
        "watchdog_package_sha256": watchdog_package_sha256,
        "watchdog_implementation_sha256": watchdog_implementation_sha256,
        "ready_at_epoch": live_state["ready_at_epoch"],
        "idle_release_seconds": runtime.IDLE_RELEASE_SECONDS,
        "create_once": True,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    validate_release(
        body,
        live_state,
        binding,
        watchdog_package_commit=watchdog_package_commit,
        watchdog_package_sha256=watchdog_package_sha256,
        watchdog_implementation_sha256=watchdog_implementation_sha256,
    )
    return body


def validate_release(
    release: dict[str, Any],
    live_state: dict[str, Any],
    binding: dict[str, Any],
    *,
    watchdog_package_commit: str,
    watchdog_package_sha256: str,
    watchdog_implementation_sha256: str,
) -> None:
    validate_live_state(live_state, binding)
    if (
        set(release) != RELEASE_KEYS
        or release.get("schema_version") != RELEASE_SCHEMA
        or release.get("status") != "AUTHORIZED_EXACT_WATCHDOG_ONLY"
        or release.get("receipt_sha256") != crypto.digest_without(release, "receipt_sha256")
        or release.get("server_binding") != binding
        or release.get("server_binding_sha256")
        != crypto.sha256(crypto.canonical_json(binding))
        or release.get("live_state_receipt_sha256") != live_state.get("receipt_sha256")
        or release.get("request_sha256") != _request_sha256()
        or release.get("watchdog_job_name") != package.JOB_NAME
        or release.get("watchdog_result_root") != package.RESULT_ROOT
        or release.get("watchdog_package_commit") != watchdog_package_commit
        or release.get("watchdog_package_sha256") != watchdog_package_sha256
        or release.get("watchdog_implementation_sha256") != watchdog_implementation_sha256
        or release.get("ready_at_epoch") != live_state.get("ready_at_epoch")
        or release.get("idle_release_seconds") != runtime.IDLE_RELEASE_SECONDS
        or release.get("create_once") is not True
        or release.get("server_launch_authorized") is not False
        or release.get("watchdog_launch_authorized") is not True
        or release.get("qualification_launch_authorized") is not False
        or release.get("scored_launch_authorized") is not False
        or any(
            not _integer(release.get(field), 0)
            for field in (
                "fleet_task_instance_calls",
                "fleet_session_calls",
                "verifier_calls",
                "scoring_calls",
            )
        )
        or release.get("protected_content_included") is not False
    ):
        raise LiveReleaseError("v24_watchdog_live_release_invalid")


def _authorization_configmap(release: dict[str, Any]) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": AUTHORIZATION_CONFIGMAP_NAME,
            "namespace": "fleet-train-jobs",
        },
        "immutable": True,
        "data": {
            "LIVE_RELEASE.json": json.dumps(release, sort_keys=True, separators=(",", ":"))
            + "\n"
        },
    }


def render(
    root: Path,
    commit: str,
    binding: dict[str, Any],
    live_state: dict[str, Any],
    *,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Render the exact create-once watcher and its immutable live release."""

    held = package.render(
        root,
        commit,
        binding,
        ready_at_epoch=live_state["ready_at_epoch"],
        priority_classes=priority_classes,
    )
    if held.get("watchdog_launch_authorized") is not False:
        raise LiveReleaseError("v24_held_package_unexpectedly_authorized")
    configmap, source_job = held["objects"]["items"]
    watchdog_package = json.loads(configmap["data"]["package.json"])
    packaged_runtime_sha256 = watchdog_package.get("files", {}).get(
        "evals/fleet/glm53_dedicated_v23_request_counter_watchdog_v1.py"
    )
    if (
        watchdog_package.get("package_commit") != commit
        or packaged_runtime_sha256 != runtime.source_sha256()
    ):
        raise LiveReleaseError("v24_watchdog_package_source_binding_invalid")
    release = build_release(
        live_state,
        binding,
        watchdog_package_commit=commit,
        watchdog_package_sha256=watchdog_package["package_sha256"],
        watchdog_implementation_sha256=packaged_runtime_sha256,
    )
    authorization = _authorization_configmap(release)
    job = copy.deepcopy(source_job)
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/live-release-receipt-sha256": release[
                "receipt_sha256"
            ],
            "cyber-post-train.fleet.ai/server-binding-sha256": release[
                "server_binding_sha256"
            ],
            "cyber-post-train.fleet.ai/credential-secret-uid": live_state[
                "jobs_api_credential_secret_uid"
            ],
        }
    )
    container = job["spec"]["template"]["spec"]["containers"][0]
    command = container["command"][-1]
    suffix = '--ready-at-epoch "$READY_AT_EPOCH"'
    if not command.endswith(suffix):
        raise LiveReleaseError("v24_watchdog_command_contract_drifted")
    container["command"][-1] = command + " \\\n  --authorization /authorization/LIVE_RELEASE.json"
    container["env"] = [row for row in container["env"] if row.get("name") != "GH_TOKEN"]
    container["env"].extend(
        [
            {
                "name": "WATCHDOG_PACKAGE_COMMIT",
                "value": commit,
            },
            {
                "name": "WATCHDOG_PACKAGE_SHA256",
                "value": watchdog_package["package_sha256"],
            },
            {
                "name": "FLEET_API_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": live_state["jobs_api_credential_secret_name"],
                        "key": "FLEET_API_KEY",
                    }
                },
            },
        ]
    )
    container["volumeMounts"].append(
        {"name": "authorization", "mountPath": "/authorization", "readOnly": True}
    )
    job["spec"]["template"]["spec"]["volumes"].append(
        {
            "name": "authorization",
            "configMap": {"name": AUTHORIZATION_CONFIGMAP_NAME},
        }
    )
    return {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [configmap, authorization, job],
        },
        "server_binding": binding,
        "live_release": release,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }


def _kubectl_json(kind: str, name: str | None = None) -> dict[str, Any]:
    command = ["kubectl", "-n", NAMESPACE, "get", kind]
    if name is not None:
        command.append(name)
    command.extend(["-o", "json"])
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise LiveReleaseError(f"kubectl_get_failed:{kind}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise LiveReleaseError(f"kubectl_json_invalid:{kind}")
    return value


def _kubectl_optional(kind: str, name: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["kubectl", "-n", NAMESPACE, "get", kind, name, "-o", "json"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        if "NotFound" in result.stderr or "not found" in result.stderr:
            return None
        raise LiveReleaseError(f"kubectl_get_failed:{kind}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise LiveReleaseError(f"kubectl_json_invalid:{kind}")
    return value


def _kubectl_secret_metadata(name: str) -> dict[str, Any]:
    """Read only a Secret's identity and key names, never any credential bytes."""

    template = (
        """{{.metadata.uid}}{{"\\n"}}"""
        """{{range .metadata.ownerReferences}}"""
        """{{.kind}}{{"\\t"}}{{.uid}}{{"\\n"}}{{end}}"""
        """{{range $key, $_ := .data}}{{$key}}{{"\\n"}}{{end}}"""
    )
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            "secret",
            name,
            "-o",
            "go-template=" + template,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise LiveReleaseError("kubectl_secret_metadata_failed")
    lines = result.stdout.splitlines()
    if not lines or not _valid_uuid(lines[0]):
        raise LiveReleaseError("kubectl_secret_metadata_invalid")
    owners = []
    keys = []
    for line in lines[1:]:
        if "\t" in line:
            kind, uid = line.split("\t", 1)
            owners.append({"kind": kind, "uid": uid})
        elif line:
            keys.append(line)
    return {"uid": lines[0], "ownerReferences": owners, "keys": sorted(keys)}


def _pod_python(pod_name: str, source: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "exec",
            pod_name,
            "-c",
            "ray-head",
            "--",
            "python",
            "-c",
            source,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise LiveReleaseError("server_bound_probe_failed")
    try:
        value = json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise LiveReleaseError("server_bound_probe_invalid") from exc
    if not isinstance(value, dict):
        raise LiveReleaseError("server_bound_probe_invalid")
    return value


def _condition_true(value: dict[str, Any], kind: str) -> bool:
    return any(
        row.get("type") == kind and row.get("status") == "True"
        for row in value.get("status", {}).get("conditions", [])
        if isinstance(row, dict)
    )


def _owned_by(value: dict[str, Any], *, kind: str, uid: str) -> bool:
    owners = value.get("metadata", {}).get("ownerReferences", [])
    return any(
        row.get("kind") == kind and row.get("uid") == uid
        for row in owners
        if isinstance(row, dict)
    )


def _api_probe_source(api_run_id: str) -> str:
    return f"""
import json, os, urllib.parse, urllib.request
url = 'https://api.ft.flt.build/v1/runs/' + urllib.parse.quote({api_run_id!r}, safe='')
request = urllib.request.Request(
    url,
    method='GET',
    headers={{
        'Authorization': 'Bearer ' + os.environ['FLEET_API_KEY'],
        'Accept': 'application/json',
    }},
)
with urllib.request.urlopen(request, timeout=30) as response:
    body = json.load(response)
candidates = [body]
for key in ('run', 'config', 'request'):
    child = body.get(key) if isinstance(body, dict) else None
    if isinstance(child, dict):
        candidates.append(child)
        nested = child.get('config')
        if isinstance(nested, dict):
            candidates.append(nested)
def first(*keys):
    for candidate in candidates:
        for key in keys:
            if candidate.get(key) is not None:
                return candidate[key]
    return None
print(json.dumps({{
    'http_status': response.status,
    'api_run_id': first('id', 'run_id'),
    'title': first('title', 'name'),
    'run_dir': first('run_dir'),
    'state': first('state', 'status'),
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _metrics_probe_source(origin: str) -> str:
    return f"""
import json, urllib.request
request = urllib.request.Request({(origin.rstrip('/') + '/metrics')!r}, method='GET')
with urllib.request.urlopen(request, timeout=30) as response:
    text = response.read().decode('utf-8')
families = sorted({{
    line.split('{{', 1)[0]
    for line in text.splitlines()
    if line.startswith('sglang:')
}})
required = ['sglang:num_queue_reqs', 'sglang:num_requests_total', 'sglang:num_running_reqs']
print(json.dumps({{
    'http_status': response.status,
    'families': [name for name in required if name in families],
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _sfs_probe_source() -> str:
    return f"""
import json
from pathlib import Path
print(json.dumps({{
    'server_run_dir_exists': Path({server.RUN_DIR!r}).is_dir(),
    'watchdog_result_root_absent': not Path({package.RESULT_ROOT!r}).exists(),
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def observe_live(api_run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the exact post-create Jobs API, Kubernetes, endpoint, and SFS state."""

    if API_RUN_ID_RE.fullmatch(api_run_id) is None:
        raise LiveReleaseError("v24_api_run_id_invalid")
    rayjob = _kubectl_json("rayjobs.ray.io", api_run_id)
    rayjob_uid = rayjob.get("metadata", {}).get("uid")
    if not _valid_uuid(rayjob_uid):
        raise LiveReleaseError("v24_rayjob_uid_invalid")
    workloads = [
        row
        for row in _kubectl_json("workloads.kueue.x-k8s.io").get("items", [])
        if isinstance(row, dict) and _owned_by(row, kind="RayJob", uid=rayjob_uid)
    ]
    if len(workloads) != 1:
        raise LiveReleaseError("v24_workload_identity_ambiguous")
    workload = workloads[0]
    workload_uid = workload.get("metadata", {}).get("uid")
    cluster_name = rayjob.get("status", {}).get("rayClusterName")
    head = rayjob.get("status", {}).get("rayClusterStatus", {}).get("head", {})
    pod_name, service_name = head.get("podName"), head.get("serviceName")
    if not all(isinstance(name, str) and name for name in (cluster_name, pod_name, service_name)):
        raise LiveReleaseError("v24_ray_cluster_head_identity_absent")
    raycluster = _kubectl_json("rayclusters.ray.io", cluster_name)
    raycluster_uid = raycluster.get("metadata", {}).get("uid")
    pod = _kubectl_json("pods", pod_name)
    service = _kubectl_json("services", service_name)
    secret_name = f"{api_run_id}-fleet-key"
    secret = _kubectl_secret_metadata(secret_name)
    if (
        not _valid_uuid(workload_uid)
        or not _valid_uuid(raycluster_uid)
        or not _owned_by(raycluster, kind="RayJob", uid=rayjob_uid)
        or not _owned_by(pod, kind="RayCluster", uid=raycluster_uid)
        or not _owned_by(service, kind="RayCluster", uid=raycluster_uid)
        or not any(
            row.get("kind") == "RayJob" and row.get("uid") == rayjob_uid
            for row in secret.get("ownerReferences", [])
        )
        or secret.get("keys") != ["FLEET_API_KEY"]
    ):
        raise LiveReleaseError("v24_server_object_owner_chain_invalid")
    selector = service.get("spec", {}).get("selector", {})
    pod_labels = pod.get("metadata", {}).get("labels", {})
    if not selector or any(pod_labels.get(key) != item for key, item in selector.items()):
        raise LiveReleaseError("v24_service_selector_does_not_bind_head_pod")
    origin = f"http://{service_name}.{NAMESPACE}.svc:8000"
    api = _pod_python(pod_name, _api_probe_source(api_run_id))
    metrics = _pod_python(pod_name, _metrics_probe_source(origin))
    sfs = _pod_python(pod_name, _sfs_probe_source())
    pod_statuses = pod.get("status", {}).get("containerStatuses", [])
    if not pod_statuses:
        raise LiveReleaseError("v24_head_pod_container_state_absent")
    preemption_events = sum(
        1
        for row in workload.get("status", {}).get("conditions", [])
        if isinstance(row, dict) and "preempt" in str(row.get("reason", "")).lower()
    )
    observed = time.time()
    binding = {
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_run_id": api_run_id,
        "rayjob_uid": rayjob_uid,
        "workload_uid": workload_uid,
        "head_pod_uid": pod.get("metadata", {}).get("uid"),
        "service_uid": service.get("metadata", {}).get("uid"),
        "service_origin": origin,
        "served_id": server.SERVED_ID,
        "model_revision": server.MODEL_REVISION,
        "context_length": server.CONTEXT_LENGTH,
    }
    live: dict[str, Any] = {
        "schema_version": LIVE_STATE_SCHEMA,
        "status": "READY_POST_CREATE_UID_BOUND",
        "observed_at_epoch": observed,
        "ready_at_epoch": observed,
        "server_binding": binding,
        "request_sha256": _request_sha256(),
        "api_get_http_status": api.get("http_status"),
        "api_run_state": str(api.get("state", "")).upper(),
        "api_title_match_count": int(api.get("title") == server.TITLE),
        "api_run_dir_match_count": int(api.get("run_dir") == server.RUN_DIR),
        "rayjob_running": rayjob.get("status", {}).get("jobDeploymentStatus") == "Running",
        "rayjob_name": rayjob.get("metadata", {}).get("name"),
        "rayjob_uid_match_count": 1,
        "workload_admitted": _condition_true(workload, "Admitted"),
        "workload_name": workload.get("metadata", {}).get("name"),
        "workload_finished": _condition_true(workload, "Finished"),
        "workload_preemption_events": preemption_events,
        "workload_uid_match_count": 1,
        "head_pod_phase": pod.get("status", {}).get("phase"),
        "head_pod_name": pod_name,
        "head_pod_ready": all(row.get("ready") is True for row in pod_statuses),
        "head_pod_restarts": sum(row.get("restartCount", 0) for row in pod_statuses),
        "head_pod_uid_match_count": 1,
        "service_present": True,
        "service_name": service_name,
        "service_uid_match_count": 1,
        "metrics_http_status": metrics.get("http_status"),
        "activity_metric_families": metrics.get("families"),
        "jobs_api_credential_secret_name": secret_name,
        "jobs_api_credential_secret_uid": secret.get("uid"),
        "jobs_api_credential_owner_rayjob_uid": rayjob_uid,
        "jobs_api_credential_key": "FLEET_API_KEY",
        "jobs_api_credential_probe_http_status": api.get("http_status"),
        "watchdog_job_match_count": int(
            _kubectl_optional("jobs.batch", package.JOB_NAME) is not None
        ),
        "watchdog_configmap_match_count": sum(
            int(_kubectl_optional("configmaps", name) is not None)
            for name in (package.JOB_NAME + "-package", AUTHORIZATION_CONFIGMAP_NAME)
        ),
        "server_run_dir_exists": sfs.get("server_run_dir_exists"),
        "watchdog_result_root_absent": sfs.get("watchdog_result_root_absent"),
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    live["receipt_sha256"] = crypto.digest_without(live, "receipt_sha256")
    validate_live_state(live, binding)
    return binding, live


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _contains(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _create_or_verify_exact(value: dict[str, Any]) -> str:
    kind = value["kind"]
    name = value["metadata"]["name"]
    existing = _kubectl_optional(kind, name)
    if existing is not None:
        if not _contains(existing, value):
            raise LiveReleaseError(f"create_once_collision:{kind}:{name}")
        return "PREEXISTING_EXACT"
    result = subprocess.run(
        ["kubectl", "-n", NAMESPACE, "create", "-f", "-"],
        input=json.dumps(value, sort_keys=True, separators=(",", ":")),
        text=True,
        capture_output=True,
    )
    if result.returncode == 0:
        return "CREATED"
    existing = _kubectl_optional(kind, name)
    if existing is not None and _contains(existing, value):
        return "PREEXISTING_EXACT_AFTER_RACE"
    raise LiveReleaseError(f"create_once_failed:{kind}:{name}")


def _release_on_handoff_failure(binding: dict[str, Any], pod_name: str) -> None:
    api_run_id = binding["api_run_id"]
    source = f"""
import json, os, time, urllib.error, urllib.parse, urllib.request
url = 'https://api.ft.flt.build/v1/runs/' + urllib.parse.quote({api_run_id!r}, safe='')
headers = {{'Authorization': 'Bearer ' + os.environ['FLEET_API_KEY'], 'Accept': 'application/json'}}
api_absent = False
try:
    request = urllib.request.Request(url, method='DELETE', headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        delete_status = response.status
except urllib.error.HTTPError as error:
    if error.code != 404: raise
    delete_status = 404
    api_absent = True
if not api_absent:
    for _ in range(12):
        try:
            request = urllib.request.Request(url, method='GET', headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200: raise RuntimeError('unexpected GET status')
        except urllib.error.HTTPError as error:
            if error.code != 404: raise
            api_absent = True
            break
        time.sleep(5)
print(json.dumps(
    {{'delete_status': delete_status, 'api_absent': api_absent}},
    sort_keys=True,
    separators=(',', ':'),
))
""".strip()
    outcome = _pod_python(pod_name, source)
    if (
        outcome.get("delete_status") not in {200, 202, 204, 404}
        or outcome.get("api_absent") is not True
    ):
        raise LiveReleaseError("handoff_failure_release_unconfirmed")
    for _ in range(12):
        if _kubectl_optional("rayjobs.ray.io", api_run_id) is None:
            return
        time.sleep(5)
    raise LiveReleaseError("handoff_failure_kubernetes_absence_unconfirmed")


def launch(
    root: Path,
    commit: str,
    api_run_id: str,
    *,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Observe, render, and create the exact watcher; release on handoff failure."""

    binding, live = observe_live(api_run_id)
    pod_name = live["head_pod_name"]
    try:
        rendered = render(
            root,
            commit,
            binding,
            live,
            priority_classes=priority_classes,
        )
        outcomes = [
            {
                "kind": item["kind"],
                "name": item["metadata"]["name"],
                "outcome": _create_or_verify_exact(item),
            }
            for item in rendered["objects"]["items"]
        ]
    except Exception:
        _release_on_handoff_failure(binding, pod_name)
        raise
    receipt: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v24-watchdog-launch-v1",
        "status": "WATCHDOG_CREATE_REQUEST_ACCEPTED",
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "live_release_receipt_sha256": rendered["live_release"]["receipt_sha256"],
        "objects": outcomes,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", nargs="?")
    parser.add_argument("--api-run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit", required=True)
    args = parser.parse_args()
    priorities = _kubectl_json("priorityclasses.scheduling.k8s.io").get("items", [])
    receipt = launch(
        args.repo_root,
        args.package_commit,
        args.api_run_id,
        priority_classes=priorities,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
