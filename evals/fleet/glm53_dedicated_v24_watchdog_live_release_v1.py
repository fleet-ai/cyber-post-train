"""Post-create release rail for the v24 external idle watchdog.

The pre-create package stays held.  This module is the only renderer that may
turn the watchdog launch bit on, and only after a fresh, exact, score-free
snapshot binds every live server object to the v24 Jobs API run.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_request_counter_watchdog_v1 as runtime
from evals.fleet import glm53_dedicated_v24_server_v1 as server
from evals.fleet import glm53_dedicated_v24_watchdog_package_v1 as package

LIVE_STATE_SCHEMA = "fleet-glm53-dedicated-v24-watchdog-live-state-v1"
PREVALIDATION_SCHEMA = "fleet-glm53-dedicated-watchdog-prevalidation-v1"
RELEASE_SCHEMA = "fleet-glm53-dedicated-v24-watchdog-live-release-v1"
LAUNCH_SCHEMA = "fleet-glm53-dedicated-v24-watchdog-launch-v1"
AUTHORIZATION_CONFIGMAP_NAME = package.JOB_NAME + "-live-release"
MAX_OBSERVATION_AGE_SECONDS = 60
MAX_READY_AGE_SECONDS = 120
API_RUN_ID_RE = re.compile(r"ft-run-[0-9a-f]{8}")
SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}")
NAMESPACE = "fleet-train-jobs"
APPLICATION_READY_KEYS = {
    "schema_version",
    "status",
    "server_title",
    "server_run_dir",
    "served_id",
    "model_revision",
    "context_length",
    "ready_at_epoch",
    "ready_at_utc",
    "health_http_status",
    "prompts_traces_flags_scores_or_model_outputs_included",
    "receipt_sha256",
}

LIVE_STATE_KEYS = {
    "schema_version",
    "status",
    "observed_at_epoch",
    "ready_at_epoch",
    "server_binding",
    "request_sha256",
    "api_get_http_status",
    "api_run_id_match_count",
    "api_run_state",
    "api_title_match_count",
    "api_run_dir_match_count",
    "rayjob_running",
    "rayjob_name",
    "rayjob_uid_match_count",
    "raycluster_name",
    "raycluster_uid",
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
    "sfs_pvc_name",
    "sfs_pvc_uid",
    "head_pod_sfs_mount_path",
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
    "application_ready_receipt_sha256",
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
    "raycluster_name",
    "raycluster_uid",
    "sfs_pvc_name",
    "sfs_pvc_uid",
    "head_pod_sfs_mount_path",
    "application_ready_receipt_sha256",
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


def extract_jobs_api_run_id(value: dict[str, Any]) -> str:
    """Extract one exact ft-run identity from create/get response variants."""

    if not isinstance(value, dict):
        raise LiveReleaseError("jobs_api_response_invalid")
    candidates = [value]
    for key in ("run", "config", "request"):
        child = value.get(key)
        if isinstance(child, dict):
            candidates.append(child)
            nested = child.get("config")
            if isinstance(nested, dict):
                candidates.append(nested)
    matches = {
        item
        for candidate in candidates
        for key in ("id", "run_id", "name")
        for item in (candidate.get(key),)
        if isinstance(item, str) and API_RUN_ID_RE.fullmatch(item) is not None
    }
    if len(matches) != 1:
        raise LiveReleaseError("jobs_api_run_identity_ambiguous_or_absent")
    return matches.pop()


def _validate_application_ready(value: dict[str, Any]) -> None:
    try:
        ready_utc = datetime.datetime.fromisoformat(
            str(value.get("ready_at_utc", "")).replace("Z", "+00:00")
        )
        ready_utc_epoch = ready_utc.timestamp()
    except (ValueError, TypeError):
        ready_utc_epoch = -1
    if (
        set(value) != APPLICATION_READY_KEYS
        or value.get("schema_version")
        != server.READY_SCHEMA
        or value.get("status") != "APPLICATION_HEALTH_HTTP_200"
        or value.get("server_title") != server.TITLE
        or value.get("server_run_dir") != server.RUN_DIR
        or value.get("served_id") != server.SERVED_ID
        or value.get("model_revision") != server.MODEL_REVISION
        or value.get("context_length") != server.CONTEXT_LENGTH
        or not _number(value.get("ready_at_epoch"))
        or value.get("ready_at_epoch", 0) <= 0
        or abs(ready_utc_epoch - float(value.get("ready_at_epoch", 0))) > 0.001
        or value.get("health_http_status") != 200
        or value.get("prompts_traces_flags_scores_or_model_outputs_included") is not False
        or value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
    ):
        raise LiveReleaseError("v24_application_ready_receipt_invalid")


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
        or not _integer(value.get("api_run_id_match_count"), 1)
        or value.get("api_run_state") not in {"SUBMITTED", "RUNNING"}
        or not _integer(value.get("api_title_match_count"), 1)
        or not _integer(value.get("api_run_dir_match_count"), 1)
        or value.get("rayjob_running") is not True
        or value.get("rayjob_name") != api_run_id
        or not _integer(value.get("rayjob_uid_match_count"), 1)
        or not isinstance(value.get("raycluster_name"), str)
        or not value["raycluster_name"]
        or not _valid_uuid(value.get("raycluster_uid"))
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
        or not isinstance(value.get("sfs_pvc_name"), str)
        or not value["sfs_pvc_name"]
        or not _valid_uuid(value.get("sfs_pvc_uid"))
        or value.get("head_pod_sfs_mount_path") != "/mnt/sfs"
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
        or type(value.get("watchdog_job_match_count")) is not int
        or value["watchdog_job_match_count"] not in {0, 1}
        or type(value.get("watchdog_configmap_match_count")) is not int
        or value["watchdog_configmap_match_count"] not in {0, 1, 2}
        or value.get("server_run_dir_exists") is not True
        or type(value.get("watchdog_result_root_absent")) is not bool
        or SHA256_RE.fullmatch(str(value.get("application_ready_receipt_sha256")))
        is None
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
        "raycluster_name": live_state["raycluster_name"],
        "raycluster_uid": live_state["raycluster_uid"],
        "sfs_pvc_name": live_state["sfs_pvc_name"],
        "sfs_pvc_uid": live_state["sfs_pvc_uid"],
        "head_pod_sfs_mount_path": live_state["head_pod_sfs_mount_path"],
        "application_ready_receipt_sha256": live_state[
            "application_ready_receipt_sha256"
        ],
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
        or release.get("raycluster_name") != live_state.get("raycluster_name")
        or release.get("raycluster_uid") != live_state.get("raycluster_uid")
        or release.get("sfs_pvc_name") != live_state.get("sfs_pvc_name")
        or release.get("sfs_pvc_uid") != live_state.get("sfs_pvc_uid")
        or release.get("head_pod_sfs_mount_path") != "/mnt/sfs"
        or release.get("application_ready_receipt_sha256")
        != live_state.get("application_ready_receipt_sha256")
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
                "name": "WATCHDOG_API_RUN_ID",
                "value": binding["api_run_id"],
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
import json, os, re, urllib.parse, urllib.request
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
run_ids = {{
    candidate.get(key)
    for candidate in candidates
    for key in ('id', 'run_id', 'name')
    if isinstance(candidate.get(key), str)
    and re.fullmatch(r'ft-run-[0-9a-f]{{8}}', candidate[key])
}}
print(json.dumps({{
    'http_status': response.status,
    'api_run_id': next(iter(run_ids)) if len(run_ids) == 1 else None,
    'title': first('title'),
    'run_dir': first('run_dir'),
    'state': first('state', 'status'),
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _api_absence_probe_source(api_run_id: str) -> str:
    return f"""
import json, os, urllib.error, urllib.parse, urllib.request
url = 'https://api.ft.flt.build/v1/runs/' + urllib.parse.quote({api_run_id!r}, safe='')
request = urllib.request.Request(
    url,
    method='GET',
    headers={{
        'Authorization': 'Bearer ' + os.environ['FLEET_API_KEY'],
        'Accept': 'application/json',
    }},
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.status
except urllib.error.HTTPError as error:
    status = error.code
print(json.dumps({{'http_status': status}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _api_delete_source(api_run_id: str) -> str:
    """Render an owner-Pod DELETE using its mounted Fleet credential."""

    if API_RUN_ID_RE.fullmatch(api_run_id) is None:
        raise LiveReleaseError("v24_api_run_id_invalid")
    return f"""
import json, os, urllib.error, urllib.parse, urllib.request
url = 'https://api.ft.flt.build/v1/runs/' + urllib.parse.quote({api_run_id!r}, safe='')
request = urllib.request.Request(
    url,
    method='DELETE',
    headers={{
        'Authorization': 'Bearer ' + os.environ['FLEET_API_KEY'],
        'Accept': 'application/json',
    }},
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        status = response.status
except urllib.error.HTTPError as error:
    status = error.code
if status not in (200, 202, 204, 404):
    raise SystemExit('jobs_api_delete_status_invalid')
print(json.dumps({{
    'api_run_id': {api_run_id!r},
    'delete_http_status': status,
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
ready = json.loads(Path({server.READY_PATH!r}).read_text())
print(json.dumps({{
    'server_run_dir_exists': Path({server.RUN_DIR!r}).is_dir(),
    'watchdog_result_root_absent': not Path({package.RESULT_ROOT!r}).exists(),
    'application_ready': ready,
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _prevalidation_probe_source(
    live: dict[str, Any], api: dict[str, Any]
) -> str:
    """Render a score-free, create-once diagnostic before strict validation."""

    body: dict[str, Any] = {
        "schema_version": PREVALIDATION_SCHEMA,
        "status": "OBSERVED_BEFORE_STRICT_VALIDATION",
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "api_title_present": isinstance(api.get("title"), str),
        "api_title_exact_match": api.get("title") == server.TITLE,
        "api_run_state": str(api.get("state", "")).upper(),
        "live_state": live,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    path = server.RUN_DIR + "/WATCHDOG-LIVE-PREVALIDATION.json"
    return f"""
import json, os
from pathlib import Path
body = json.loads({encoded!r})
target = Path({path!r})
fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(fd, 'w') as handle:
    json.dump(body, handle, sort_keys=True, separators=(',', ':'))
    handle.write('\\n')
print(json.dumps({{
    'path': str(target),
    'receipt_sha256': body['receipt_sha256'],
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _observe_live(api_run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
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
    head_containers = [
        row
        for row in pod.get("spec", {}).get("containers", [])
        if isinstance(row, dict) and row.get("name") == "ray-head"
    ]
    if len(head_containers) != 1:
        raise LiveReleaseError("v24_head_container_identity_ambiguous")
    sfs_mounts = [
        row
        for row in head_containers[0].get("volumeMounts", [])
        if isinstance(row, dict) and row.get("mountPath") == "/mnt/sfs"
    ]
    if len(sfs_mounts) != 1:
        raise LiveReleaseError("v24_head_sfs_mount_identity_ambiguous")
    sfs_volume_name = sfs_mounts[0].get("name")
    sfs_volumes = [
        row
        for row in pod.get("spec", {}).get("volumes", [])
        if isinstance(row, dict) and row.get("name") == sfs_volume_name
    ]
    if len(sfs_volumes) != 1:
        raise LiveReleaseError("v24_head_sfs_volume_identity_ambiguous")
    pvc_name = sfs_volumes[0].get("persistentVolumeClaim", {}).get("claimName")
    if not isinstance(pvc_name, str) or not pvc_name:
        raise LiveReleaseError("v24_head_sfs_pvc_identity_absent")
    pvc = _kubectl_json("persistentvolumeclaims", pvc_name)
    pvc_uid = pvc.get("metadata", {}).get("uid")
    secret_name = f"{api_run_id}-fleet-key"
    secret = _kubectl_secret_metadata(secret_name)
    if (
        not _valid_uuid(workload_uid)
        or not _valid_uuid(raycluster_uid)
        or not _valid_uuid(pvc_uid)
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
    application_ready = sfs.get("application_ready")
    if not isinstance(application_ready, dict):
        raise LiveReleaseError("v24_application_ready_receipt_absent")
    _validate_application_ready(application_ready)
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
        "ready_at_epoch": application_ready["ready_at_epoch"],
        "server_binding": binding,
        "request_sha256": _request_sha256(),
        "api_get_http_status": api.get("http_status"),
        "api_run_id_match_count": int(api.get("api_run_id") == api_run_id),
        "api_run_state": str(api.get("state", "")).upper(),
        "api_title_match_count": int(api.get("title") in {None, server.TITLE}),
        "api_run_dir_match_count": int(api.get("run_dir") == server.RUN_DIR),
        "rayjob_running": rayjob.get("status", {}).get("jobDeploymentStatus") == "Running",
        "rayjob_name": rayjob.get("metadata", {}).get("name"),
        "rayjob_uid_match_count": 1,
        "raycluster_name": cluster_name,
        "raycluster_uid": raycluster_uid,
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
        "sfs_pvc_name": pvc_name,
        "sfs_pvc_uid": pvc_uid,
        "head_pod_sfs_mount_path": "/mnt/sfs",
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
        "application_ready_receipt_sha256": application_ready["receipt_sha256"],
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "protected_content_included": False,
    }
    live["receipt_sha256"] = crypto.digest_without(live, "receipt_sha256")
    _pod_python(pod_name, _prevalidation_probe_source(live, api))
    validate_live_state(live, binding)
    return binding, live


def _kubernetes_server_remnants(
    api_run_id: str, binding: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "get",
            (
                "rayjobs.ray.io,rayclusters.ray.io,workloads.kueue.x-k8s.io,"
                "pods,services"
            ),
            "-o",
            "json",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise LiveReleaseError("local_kubernetes_inventory_failed")
    try:
        rows = json.loads(result.stdout).get("items", [])
    except (json.JSONDecodeError, AttributeError) as exc:
        raise LiveReleaseError("local_kubernetes_inventory_invalid") from exc
    if not isinstance(rows, list):
        raise LiveReleaseError("local_kubernetes_inventory_invalid")
    expected_uids = {
        str(value)
        for value in (binding or {}).values()
        if _valid_uuid(value)
    }
    tokens = {
        api_run_id,
        str((binding or {}).get("server_title", "")),
        str((binding or {}).get("server_run_dir", "")),
    } - {""}
    remnants = []
    for row in rows:
        if not isinstance(row, dict):
            raise LiveReleaseError("local_kubernetes_inventory_invalid")
        metadata = row.get("metadata", {})
        owners = metadata.get("ownerReferences", []) or []
        strings = [str(metadata.get("name", ""))]
        strings.extend(str(owner.get("name", "")) for owner in owners)
        strings.extend(
            str(value)
            for field in ("labels", "annotations")
            for value in (metadata.get(field, {}) or {}).values()
        )
        if (
            metadata.get("uid") in expected_uids
            or any(owner.get("uid") in expected_uids for owner in owners)
            or any(token in candidate for token in tokens for candidate in strings if candidate)
        ):
            remnants.append(
                {
                    "kind": str(row.get("kind", "")),
                    "name": str(metadata.get("name", "")),
                    "uid": str(metadata.get("uid", "")),
                }
            )
    return remnants


def _release_local(api_run_id: str, binding: dict[str, Any] | None = None) -> None:
    token = os.environ.get("FLEET_API_KEY", "")
    if API_RUN_ID_RE.fullmatch(api_run_id) is None or not token:
        raise LiveReleaseError("local_jobs_api_release_identity_absent")
    url = "https://api.ft.flt.build/v1/runs/" + urllib.parse.quote(api_run_id, safe="")
    headers = {"Authorization": "Bearer " + token, "Accept": "application/json"}
    try:
        request = urllib.request.Request(url, method="DELETE", headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status not in {200, 202, 204}:
                raise LiveReleaseError("local_jobs_api_delete_status_invalid")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise LiveReleaseError("local_jobs_api_delete_failed") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise LiveReleaseError("local_jobs_api_delete_failed") from exc
    for _ in range(12):
        try:
            request = urllib.request.Request(url, method="GET", headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise LiveReleaseError("local_jobs_api_get_status_invalid")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                break
            raise LiveReleaseError("local_jobs_api_absence_check_failed") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise LiveReleaseError("local_jobs_api_absence_check_failed") from exc
        time.sleep(5)
    else:
        raise LiveReleaseError("local_jobs_api_absence_unconfirmed")
    for _ in range(12):
        if not _kubernetes_server_remnants(api_run_id, binding):
            return
        time.sleep(5)
    raise LiveReleaseError("local_kubernetes_remnant_absence_unconfirmed")


def observe_live(api_run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Observe the exact server and release it if any handoff prerequisite fails."""

    try:
        return _observe_live(api_run_id)
    except Exception:
        pod_name = None
        rayjob = _kubectl_optional("rayjobs.ray.io", api_run_id)
        if rayjob is not None:
            candidate = (
                rayjob.get("status", {})
                .get("rayClusterStatus", {})
                .get("head", {})
                .get("podName")
            )
            if isinstance(candidate, str) and candidate:
                pod_name = candidate
        if pod_name is not None:
            _release_on_handoff_failure({"api_run_id": api_run_id}, pod_name)
        else:
            _release_local(api_run_id)
        raise


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


def _delete_exact(value: dict[str, Any]) -> str:
    kind = value["kind"]
    name = value["metadata"]["name"]
    existing = _kubectl_optional(kind, name)
    if existing is None:
        return "ALREADY_ABSENT"
    if not _contains(existing, value):
        raise LiveReleaseError(f"rollback_collision:{kind}:{name}")
    result = subprocess.run(
        [
            "kubectl",
            "-n",
            NAMESPACE,
            "delete",
            kind,
            name,
            "--wait=true",
            "--timeout=60s",
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or _kubectl_optional(kind, name) is not None:
        raise LiveReleaseError(f"rollback_absence_unconfirmed:{kind}:{name}")
    return "DELETED_EXACT"


def _rollback_watcher_objects(rendered: dict[str, Any]) -> list[dict[str, str]]:
    outcomes = []
    for item in reversed(rendered["objects"]["items"]):
        outcomes.append(
            {
                "kind": item["kind"],
                "name": item["metadata"]["name"],
                "outcome": _delete_exact(item),
            }
        )
    return outcomes


def _binding_from_existing_object(value: dict[str, Any]) -> dict[str, Any] | None:
    """Read only the public server binding from one owned watcher object."""

    kind = value.get("kind")
    name = value.get("metadata", {}).get("name")
    if kind == "ConfigMap" and name == package.JOB_NAME + "-package":
        raw = value.get("data", {}).get("binding.json")
        try:
            binding = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LiveReleaseError("stale_package_binding_invalid") from exc
        if not isinstance(binding, dict):
            raise LiveReleaseError("stale_package_binding_invalid")
        return binding
    if kind == "ConfigMap" and name == AUTHORIZATION_CONFIGMAP_NAME:
        raw = value.get("data", {}).get("LIVE_RELEASE.json")
        try:
            release = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LiveReleaseError("stale_authorization_invalid") from exc
        if (
            not isinstance(release, dict)
            or set(release) != RELEASE_KEYS
            or release.get("schema_version") != RELEASE_SCHEMA
            or release.get("status") != "AUTHORIZED_EXACT_WATCHDOG_ONLY"
            or release.get("watchdog_job_name") != package.JOB_NAME
            or release.get("watchdog_result_root") != package.RESULT_ROOT
            or release.get("receipt_sha256")
            != crypto.digest_without(release, "receipt_sha256")
            or any(
                release.get(field) != 0
                for field in (
                    "fleet_task_instance_calls",
                    "fleet_session_calls",
                    "verifier_calls",
                    "scoring_calls",
                )
            )
            or release.get("protected_content_included") is not False
        ):
            raise LiveReleaseError("stale_authorization_invalid")
        binding = release.get("server_binding")
        if not isinstance(binding, dict):
            raise LiveReleaseError("stale_authorization_binding_invalid")
        return binding
    if kind == "Job" and name == package.JOB_NAME:
        labels = value.get("metadata", {}).get("labels", {})
        annotations = value.get("metadata", {}).get("annotations", {})
        if (
            labels.get("cyber-post-train.fleet.ai/owner") != "chris"
            or labels.get("cyber-post-train.fleet.ai/experiment") != package.JOB_NAME
            or labels.get("kueue.x-k8s.io/queue-name") != "training-lq"
            or annotations.get("cyber-post-train.fleet.ai/create-once") != "true"
            or annotations.get("cyber-post-train.fleet.ai/score-free") != "true"
        ):
            raise LiveReleaseError("stale_watchdog_job_identity_invalid")
        return None
    raise LiveReleaseError("unexpected_watcher_object_identity")


def _reconcile_existing_watcher_objects(
    rendered: dict[str, Any],
    binding: dict[str, Any],
    live: dict[str, Any],
    *,
    head_pod_name: str,
) -> list[dict[str, str]]:
    """Adopt byte-equivalent partial state or remove a proven inert old run."""

    expected = rendered["objects"]["items"]
    existing = [
        item
        for item in (
            _kubectl_optional(value["kind"], value["metadata"]["name"])
            for value in expected
        )
        if item is not None
    ]
    if not existing:
        if live.get("watchdog_result_root_absent") is not True:
            raise LiveReleaseError("orphan_watchdog_result_root_present")
        return []

    expected_by_identity = {
        (value["kind"], value["metadata"]["name"]): value for value in expected
    }
    exact = [
        _contains(value, expected_by_identity[(value["kind"], value["metadata"]["name"])])
        for value in existing
    ]
    if all(exact):
        return [
            {
                "kind": value["kind"],
                "name": value["metadata"]["name"],
                "outcome": "ADOPTED_EXACT_PARTIAL_OR_ACTIVE",
            }
            for value in existing
        ]
    if any(exact):
        raise LiveReleaseError("mixed_exact_and_stale_watcher_objects")

    prior_bindings = [
        prior
        for prior in (_binding_from_existing_object(value) for value in existing)
        if prior is not None
    ]
    if not prior_bindings or any(prior != prior_bindings[0] for prior in prior_bindings):
        raise LiveReleaseError("stale_watcher_binding_ambiguous")
    prior = prior_bindings[0]
    try:
        server.validate_binding(prior)
    except server.ServerPlanError as exc:
        raise LiveReleaseError("stale_watcher_binding_invalid") from exc
    if prior == binding or prior.get("api_run_id") == binding.get("api_run_id"):
        raise LiveReleaseError("current_run_watcher_object_drift")
    if live.get("watchdog_result_root_absent") is not True:
        raise LiveReleaseError("stale_watcher_has_durable_runtime_evidence")
    jobs = [value for value in existing if value.get("kind") == "Job"]
    if jobs and any(
        bool(value.get("status", {}).get("active"))
        or not (
            bool(value.get("status", {}).get("succeeded"))
            or bool(value.get("status", {}).get("failed"))
        )
        for value in jobs
    ):
        raise LiveReleaseError("stale_watchdog_job_not_terminal")
    old_run_id = str(prior["api_run_id"])
    if _pod_python(head_pod_name, _api_absence_probe_source(old_run_id)).get(
        "http_status"
    ) != 404 or _kubectl_optional("rayjobs.ray.io", old_run_id) is not None:
        raise LiveReleaseError("stale_server_absence_unproven")

    outcomes = []
    for value in reversed(existing):
        outcomes.append(
            {
                "kind": value["kind"],
                "name": value["metadata"]["name"],
                "outcome": _delete_exact(value),
            }
        )
    if any(
        _kubectl_optional(value["kind"], value["metadata"]["name"]) is not None
        for value in expected
    ):
        raise LiveReleaseError("stale_watcher_cleanup_unconfirmed")
    return outcomes


def _watchdog_receipts_source() -> str:
    active = str(Path(package.RESULT_ROOT) / "ACTIVE.json")
    authorization = str(Path(package.RESULT_ROOT) / "AUTHORIZATION_VALIDATED.json")
    return f"""
import json
from pathlib import Path
print(json.dumps({{
    'active': json.loads(Path({active!r}).read_text()),
    'authorization': json.loads(Path({authorization!r}).read_text()),
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _wait_for_watchdog_active(
    binding: dict[str, Any],
    release: dict[str, Any],
    *,
    head_pod_name: str,
    attempts: int = 24,
) -> dict[str, str]:
    if attempts < 1:
        raise LiveReleaseError("watchdog_wait_policy_invalid")
    last_job: dict[str, Any] | None = None
    for index in range(attempts):
        job = _kubectl_optional("jobs.batch", package.JOB_NAME)
        if job is not None:
            last_job = job
            job_uid = job.get("metadata", {}).get("uid")
            if (
                not _valid_uuid(job_uid)
                or job.get("metadata", {}).get("annotations", {}).get(
                    "cyber-post-train.fleet.ai/live-release-receipt-sha256"
                )
                != release["receipt_sha256"]
                or bool(job.get("status", {}).get("failed"))
            ):
                raise LiveReleaseError("watchdog_job_runtime_identity_invalid")
            pods = [
                row
                for row in _kubectl_json("pods").get("items", [])
                if isinstance(row, dict) and _owned_by(row, kind="Job", uid=job_uid)
            ]
            if len(pods) > 1:
                raise LiveReleaseError("watchdog_pod_identity_ambiguous")
            if len(pods) == 1:
                pod = pods[0]
                pod_uid = pod.get("metadata", {}).get("uid")
                statuses = pod.get("status", {}).get("containerStatuses", [])
                if (
                    not _valid_uuid(pod_uid)
                    or pod.get("status", {}).get("phase") not in {"Pending", "Running"}
                    or any(row.get("restartCount", 0) != 0 for row in statuses)
                ):
                    raise LiveReleaseError("watchdog_pod_runtime_identity_invalid")
                if pod.get("status", {}).get("phase") == "Running" and statuses and all(
                    row.get("ready") is True for row in statuses
                ):
                    receipts = _pod_python(
                        head_pod_name, _watchdog_receipts_source()
                    )
                    active = receipts.get("active")
                    authorization = receipts.get("authorization")
                    if not isinstance(active, dict) or not isinstance(authorization, dict):
                        raise LiveReleaseError("watchdog_runtime_receipts_invalid")
                    runtime.validate_active_receipt(
                        active,
                        binding,
                        watcher_job_uid=job_uid,
                        watcher_pod_uid=pod_uid,
                    )
                    runtime.validate_runtime_authorization_receipt(
                        authorization,
                        release,
                        binding,
                        watcher_job_uid=job_uid,
                        watcher_pod_uid=pod_uid,
                        package_commit=release["watchdog_package_commit"],
                        package_sha256=release["watchdog_package_sha256"],
                        expected_runtime_auth_schema=getattr(
                            package,
                            "RUNTIME_AUTH_SCHEMA",
                            runtime.RUNTIME_AUTH_SCHEMA,
                        ),
                        expected_live_release_schema=RELEASE_SCHEMA,
                        expected_watchdog_job_name=package.JOB_NAME,
                        expected_watchdog_result_root=package.RESULT_ROOT,
                    )
                    return {
                        "watchdog_job_uid": job_uid,
                        "watchdog_pod_uid": pod_uid,
                        "active_receipt_sha256": active["receipt_sha256"],
                        "runtime_authorization_receipt_sha256": authorization[
                            "receipt_sha256"
                        ],
                    }
        if index + 1 < attempts:
            time.sleep(5)
    reason = "absent" if last_job is None else "not_active"
    raise LiveReleaseError(f"watchdog_active_receipt_timeout:{reason}")


def _release_on_handoff_failure(binding: dict[str, Any], pod_name: str) -> None:
    api_run_id = str(binding["api_run_id"])
    try:
        # Preferred path: the CPU handoff controller owns an independent Fleet
        # credential and can confirm both API and Kubernetes absence.
        _release_local(api_run_id, binding)
        return
    except LiveReleaseError as exc:
        if str(exc) != "local_jobs_api_release_identity_absent":
            raise

    # Recovery path for an accidentally uncredentialed launcher.  The exact
    # owner Pod already has the run-scoped Fleet credential.  Its exec stream
    # may disconnect as DELETE tears the Pod down, so the independent local
    # Kubernetes control plane—not that doomed exec—confirms GPU-object absence.
    with suppress(ConnectionError, LiveReleaseError, subprocess.SubprocessError):
        _pod_python(pod_name, _api_delete_source(api_run_id))
    for _ in range(12):
        if not _kubernetes_server_remnants(api_run_id, binding):
            return
        time.sleep(5)
    raise LiveReleaseError("fallback_kubernetes_remnant_absence_unconfirmed")


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
    rendered: dict[str, Any] | None = None
    reconciliation: list[dict[str, str]] = []
    try:
        rendered = render(
            root,
            commit,
            binding,
            live,
            priority_classes=priority_classes,
        )
        reconciliation = _reconcile_existing_watcher_objects(
            rendered,
            binding,
            live,
            head_pod_name=pod_name,
        )
        outcomes = [
            {
                "kind": item["kind"],
                "name": item["metadata"]["name"],
                "outcome": _create_or_verify_exact(item),
            }
            for item in rendered["objects"]["items"]
        ]
        active = _wait_for_watchdog_active(
            binding,
            rendered["live_release"],
            head_pod_name=pod_name,
        )
    except Exception:
        if rendered is not None:
            _rollback_watcher_objects(rendered)
        _release_on_handoff_failure(binding, pod_name)
        raise
    receipt: dict[str, Any] = {
        "schema_version": LAUNCH_SCHEMA,
        "status": "WATCHDOG_CREATE_REQUEST_ACCEPTED",
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "live_release_receipt_sha256": rendered["live_release"]["receipt_sha256"],
        "preexisting_object_reconciliation": reconciliation,
        "objects": outcomes,
        "runtime": active,
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
