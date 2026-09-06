"""Post-create release rail for the v24 external idle watchdog.

The pre-create package stays held.  This module is the only renderer that may
turn the watchdog launch bit on, and only after a fresh, exact, score-free
snapshot binds every live server object to the v24 Jobs API run.
"""

from __future__ import annotations

import copy
import json
import re
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
    "rayjob_uid_match_count",
    "workload_admitted",
    "workload_finished",
    "workload_preemption_events",
    "workload_uid_match_count",
    "head_pod_phase",
    "head_pod_ready",
    "head_pod_restarts",
    "head_pod_uid_match_count",
    "service_present",
    "service_uid_match_count",
    "metrics_http_status",
    "activity_metric_families",
    "watchdog_job_match_count",
    "watchdog_configmap_match_count",
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


def _service_origin(api_run_id: str) -> str:
    return f"http://{api_run_id}-head-svc.fleet-train-jobs.svc:8000"


def validate_live_state(
    value: dict[str, Any], binding: dict[str, Any], *, now_epoch: float
) -> None:
    """Validate a short-lived, post-create snapshot before rendering the Job."""

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
        or not isinstance(now_epoch, (int, float))
        or not isinstance(observed, (int, float))
        or not isinstance(ready, (int, float))
        or observed > now_epoch
        or now_epoch - observed > MAX_OBSERVATION_AGE_SECONDS
        or ready <= 0
        or ready > observed
        or observed - ready > MAX_READY_AGE_SECONDS
        or value.get("server_binding") != binding
        or not isinstance(api_run_id, str)
        or API_RUN_ID_RE.fullmatch(api_run_id) is None
        or binding.get("service_origin") != _service_origin(api_run_id)
        or value.get("request_sha256") != _request_sha256()
        or value.get("api_get_http_status") != 200
        or value.get("api_run_state") != "RUNNING"
        or value.get("api_title_match_count") != 1
        or value.get("api_run_dir_match_count") != 1
        or value.get("rayjob_running") is not True
        or value.get("rayjob_uid_match_count") != 1
        or value.get("workload_admitted") is not True
        or value.get("workload_finished") is not False
        or value.get("workload_preemption_events") != 0
        or value.get("workload_uid_match_count") != 1
        or value.get("head_pod_phase") != "Running"
        or value.get("head_pod_ready") is not True
        or value.get("head_pod_restarts") != 0
        or value.get("head_pod_uid_match_count") != 1
        or value.get("service_present") is not True
        or value.get("service_uid_match_count") != 1
        or value.get("metrics_http_status") != 200
        or value.get("activity_metric_families") != list(runtime.ACTIVITY_METRICS)
        or value.get("watchdog_job_match_count") != 0
        or value.get("watchdog_configmap_match_count") != 0
        or value.get("watchdog_result_root_absent") is not True
        or any(
            value.get(field) != 0
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


def build_release(
    live_state: dict[str, Any], binding: dict[str, Any], *, now_epoch: float
) -> dict[str, Any]:
    validate_live_state(live_state, binding, now_epoch=now_epoch)
    body: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA,
        "status": "AUTHORIZED_EXACT_WATCHDOG_ONLY",
        "server_binding": binding,
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "live_state_receipt_sha256": live_state["receipt_sha256"],
        "request_sha256": _request_sha256(),
        "watchdog_job_name": package.JOB_NAME,
        "watchdog_result_root": package.RESULT_ROOT,
        "watchdog_implementation_sha256": runtime.source_sha256(),
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
    validate_release(body, live_state, binding, now_epoch=now_epoch)
    return body


def validate_release(
    release: dict[str, Any],
    live_state: dict[str, Any],
    binding: dict[str, Any],
    *,
    now_epoch: float,
) -> None:
    validate_live_state(live_state, binding, now_epoch=now_epoch)
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
        or release.get("watchdog_implementation_sha256") != runtime.source_sha256()
        or release.get("ready_at_epoch") != live_state.get("ready_at_epoch")
        or release.get("idle_release_seconds") != runtime.IDLE_RELEASE_SECONDS
        or release.get("create_once") is not True
        or release.get("server_launch_authorized") is not False
        or release.get("watchdog_launch_authorized") is not True
        or release.get("qualification_launch_authorized") is not False
        or release.get("scored_launch_authorized") is not False
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
    now_epoch: float,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Render the exact create-once watcher and its immutable live release."""

    release = build_release(live_state, binding, now_epoch=now_epoch)
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
    job = copy.deepcopy(source_job)
    job["metadata"]["annotations"].update(
        {
            "cyber-post-train.fleet.ai/live-release-receipt-sha256": release[
                "receipt_sha256"
            ],
            "cyber-post-train.fleet.ai/server-binding-sha256": release[
                "server_binding_sha256"
            ],
        }
    )
    return {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [configmap, _authorization_configmap(release), job],
        },
        "server_binding": binding,
        "live_release": release,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
