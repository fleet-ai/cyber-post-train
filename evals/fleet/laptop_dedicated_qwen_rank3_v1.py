"""Held laptop rank-3 lane for the fresh dedicated Qwen3.8 TP1 server.

This module cannot launch a scored rollout.  It binds the exact fresh RayJob,
head Pod, and Service, validates those objects immediately before constructing
an IPv4-only local port-forward, and provides a content-free two-stream
qualification receipt.  The four statistical cells remain sequential and held
until a separate score-release authority reconciles claims and sessions.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import hosted_concurrency4_qualification_v1 as tools
from evals.fleet import laptop_opencode_lane_v1 as laptop

SCHEMA = "fleet-laptop-dedicated-qwen-rank3-held-v1"
STREAM_SCHEMA = "fleet-dedicated-qwen-concurrency2-stream-v1"
QUALIFICATION_SCHEMA = "fleet-dedicated-qwen-concurrency2-qualification-v1"

NAMESPACE = "fleet-train-jobs"
API_RUN_ID = "ft-run-75891e80"
RAYJOB_NAME = API_RUN_ID
RAYJOB_UID = "7c6bcfb9-c4a7-497c-b1d5-9f31f7a838cf"
HEAD_POD_NAME = "ft-run-75891e80-ftkdm-head-d29gz"
HEAD_POD_UID = "9781ee19-cd71-44df-9013-89cf351b4269"
SERVICE_NAME = "ft-run-75891e80-ftkdm-head-svc"
SERVICE_UID = "6c244802-37ca-4607-aacd-95a37bc303b0"
SERVICE_PORT = 8000
SERVED_ID = "qwen3.8-27b"
MODEL_REPOSITORY = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONTEXT_LENGTH = 262_144
OPENCODE_VERSION = "1.18.27"
LOCAL_DEFAULT_PORT = 18_527
REQUEST_TIMEOUT_SECONDS = 300


class DedicatedLaptopError(RuntimeError):
    """Stable, content-free failure for the dedicated laptop lane."""


Runner = Callable[[Sequence[str]], str]
CompletionCaller = Callable[[str, str], float]


def _uuid(value: Any, label: str) -> str:
    try:
        parsed = uuid.UUID(str(value))
    except (ValueError, AttributeError) as exc:
        raise DedicatedLaptopError(f"{label}_invalid") from exc
    if parsed.int == 0:
        raise DedicatedLaptopError(f"{label}_invalid")
    return str(parsed)


def server_binding() -> dict[str, Any]:
    return {
        "api_run_id": API_RUN_ID,
        "rayjob": {"name": RAYJOB_NAME, "uid": RAYJOB_UID},
        "head_pod": {"name": HEAD_POD_NAME, "uid": HEAD_POD_UID},
        "service": {
            "name": SERVICE_NAME,
            "namespace": NAMESPACE,
            "uid": SERVICE_UID,
            "port": SERVICE_PORT,
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "served_id": SERVED_ID,
            "context_length": CONTEXT_LENGTH,
        },
        "runtime": {
            "opencode_version": OPENCODE_VERSION,
            "reasoning_parser": "qwen3",
            "tool_call_parser": "qwen3_coder",
        },
    }


def _condition_true(value: Mapping[str, Any], condition_type: str) -> bool:
    rows = value.get("status", {}).get("conditions", [])
    return any(
        isinstance(row, Mapping)
        and row.get("type") == condition_type
        and row.get("status") == "True"
        for row in rows
    )


def validate_live_objects(
    rayjob: Mapping[str, Any], pod: Mapping[str, Any], service: Mapping[str, Any]
) -> dict[str, Any]:
    """Require the fresh identities and a ready, restart-free route."""
    if rayjob.get("metadata", {}).get("name") != RAYJOB_NAME or _uuid(
        rayjob.get("metadata", {}).get("uid"), "rayjob_uid"
    ) != RAYJOB_UID:
        raise DedicatedLaptopError("rayjob_identity_drift")
    status = rayjob.get("status", {})
    if status.get("jobDeploymentStatus") != "Running" or status.get("jobStatus") not in {
        "RUNNING",
        "SUCCEEDED",
    }:
        raise DedicatedLaptopError("rayjob_not_running")
    cluster = status.get("rayClusterStatus", {})
    if cluster.get("state") != "ready":
        raise DedicatedLaptopError("ray_cluster_not_ready")

    if pod.get("metadata", {}).get("name") != HEAD_POD_NAME or _uuid(
        pod.get("metadata", {}).get("uid"), "head_pod_uid"
    ) != HEAD_POD_UID:
        raise DedicatedLaptopError("head_pod_identity_drift")
    if pod.get("status", {}).get("phase") != "Running" or not _condition_true(pod, "Ready"):
        raise DedicatedLaptopError("head_pod_not_ready")
    restarts = sum(
        int(row.get("restartCount", 0))
        for row in pod.get("status", {}).get("containerStatuses", [])
        if isinstance(row, Mapping)
    )
    if restarts != 0:
        raise DedicatedLaptopError("head_pod_restarted")

    metadata = service.get("metadata", {})
    if (
        metadata.get("name") != SERVICE_NAME
        or metadata.get("namespace") != NAMESPACE
        or _uuid(metadata.get("uid"), "service_uid") != SERVICE_UID
    ):
        raise DedicatedLaptopError("service_identity_drift")
    ports = service.get("spec", {}).get("ports", [])
    serve = [
        row
        for row in ports
        if isinstance(row, Mapping)
        and row.get("name") == "serve"
        and row.get("port") == SERVICE_PORT
        and row.get("targetPort") == SERVICE_PORT
    ]
    if len(serve) != 1:
        raise DedicatedLaptopError("service_port_drift")
    return {
        "rayjob_uid": RAYJOB_UID,
        "head_pod_uid": HEAD_POD_UID,
        "service_uid": SERVICE_UID,
        "head_pod_restarts": 0,
        "ready": True,
    }


def validate_live_with_runner(runner: Runner) -> dict[str, Any]:
    """Read only the three exact Kubernetes objects, then validate them."""
    def get(kind: str, name: str) -> dict[str, Any]:
        raw = runner(["kubectl", "-n", NAMESPACE, "get", kind, name, "-o", "json"])
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise DedicatedLaptopError(f"{kind}_response_invalid")
        return value

    return validate_live_objects(
        get("rayjob.ray.io", RAYJOB_NAME),
        get("pod", HEAD_POD_NAME),
        get("service", SERVICE_NAME),
    )


def port_forward_argv(local_port: int = LOCAL_DEFAULT_PORT) -> list[str]:
    if type(local_port) is not int or not (1024 <= local_port <= 65535):
        raise DedicatedLaptopError("local_port_invalid")
    return [
        "kubectl",
        "-n",
        NAMESPACE,
        "port-forward",
        "--address=127.0.0.1",
        f"service/{SERVICE_NAME}",
        f"{local_port}:{SERVICE_PORT}",
    ]


def held_plan(repo_root: Path, local_port: int = LOCAL_DEFAULT_PORT) -> dict[str, Any]:
    base = laptop.held_plan(repo_root)
    if base["plan_sha256"] != (
        "sha256:7cfbdc3b07df90a6e37ebd43aebdc6564f329bfe35be6c7e1b1e9c8246e72d5d"
    ):
        raise DedicatedLaptopError("rank3_base_plan_drift")
    plan = {
        "schema_version": SCHEMA,
        "base_held_plan_sha256": base["plan_sha256"],
        "campaign_id": base["campaign_id"],
        "selection_rank": base["selection_rank"],
        "task": base["task"],
        "cells": base["cells"],
        "server": server_binding(),
        "route": {
            "kind": "uid_bound_local_port_forward",
            "origin": f"http://127.0.0.1:{local_port}",
            "argv": port_forward_argv(local_port),
            "shared_hosted_route_allowed": False,
        },
        "treatment": laptop._qualification_config(),
        "execution": {
            **base["execution"],
            "scored_launch_authorized": False,
            "required_attempt_order": [1, 2, 3, 4],
            "maximum_concurrent_attempts": 1,
            "automatic_retry": False,
        },
        "required_before_score_release": [
            "fresh_uid_bound_server_parity_receipt",
            "fresh_dedicated_concurrency2_qualification_receipt",
            "immutable_hosted_exclusion_for_all_four_rank3_cells",
            "fresh_global_ledger_and_session_inventory_clear",
            "globally_visible_atomic_execution_claim_before_model_call",
        ],
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_scores_or_model_outputs_included": False,
        },
    }
    plan["plan_sha256"] = crypto.digest_without(plan, "plan_sha256")
    return plan


def _strict_json(raw: bytes, code: str) -> dict[str, Any]:
    if len(raw) > 4 * 1024 * 1024:
        raise DedicatedLaptopError(f"{code}_too_large")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DedicatedLaptopError(f"{code}_invalid") from exc
    if not isinstance(value, dict):
        raise DedicatedLaptopError(f"{code}_invalid")
    return value


def validate_model_roster(origin: str) -> dict[str, Any]:
    request = Request(origin.rstrip("/") + "/v1/models", method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise DedicatedLaptopError("model_roster_http_status")
            roster = _strict_json(response.read(4 * 1024 * 1024 + 1), "model_roster")
    except DedicatedLaptopError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DedicatedLaptopError("model_roster_request_failed") from exc
    rows = roster.get("data")
    if not isinstance(rows, list) or len([row for row in rows if row.get("id") == SERVED_ID]) != 1:
        raise DedicatedLaptopError("served_id_not_exactly_once")
    return {"http_status": 200, "served_id": SERVED_ID, "exactly_once": True}


def post_completion(origin: str, tool_name: str) -> float:
    body = tools.canonical_json(tools._request_payload(SERVED_ID, tool_name))
    request = Request(
        origin.rstrip("/") + "/v1/chat/completions",
        method="POST",
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise DedicatedLaptopError("completion_http_status")
            raw = response.read(4 * 1024 * 1024 + 1)
    except DedicatedLaptopError:
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DedicatedLaptopError("completion_request_failed") from exc
    elapsed = time.monotonic() - started
    value = _strict_json(raw, "completion")
    try:
        tools._validate_completion(value, model=SERVED_ID, tool_name=tool_name)
    except tools.QualificationError as exc:
        raise DedicatedLaptopError(exc.code) from exc
    return elapsed


def run_local_stream(
    origin: str, *, caller: CompletionCaller | None = None
) -> dict[str, Any]:
    """Run two generic non-benchmark tool calls; never persist their content."""
    if caller is None:
        caller = post_completion
    started = time.monotonic()
    latencies = [caller(origin, name) for name in ("bash", "submit_report")]
    elapsed = time.monotonic() - started
    body = {
        "schema_version": STREAM_SCHEMA,
        "lane": "local_port_forward",
        "server_binding": server_binding(),
        "non_scored": True,
        "generic_non_benchmark": True,
        "tools": ["bash", "submit_report"],
        "requests_started": 2,
        "requests_succeeded": 2,
        "protocol_valid_requests": 2,
        "elapsed_seconds": round(elapsed, 6),
        "request_latency_seconds": {
            "min": round(min(latencies), 6),
            "max": round(max(latencies), 6),
        },
        "response_content_persisted": False,
        "task_instance_session_verifier_scoring_mutations": 0,
        "passed": True,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def combine_concurrency2(
    cluster_stream: Mapping[str, Any], local_stream: Mapping[str, Any]
) -> dict[str, Any]:
    streams = [dict(cluster_stream), dict(local_stream)]
    if [row.get("lane") for row in streams] != ["cluster_internal", "local_port_forward"]:
        raise DedicatedLaptopError("qualification_lane_set_drift")
    expected_binding = server_binding()
    for row in streams:
        if (
            row.get("schema_version") != STREAM_SCHEMA
            or row.get("server_binding") != expected_binding
            or row.get("receipt_sha256") != crypto.digest_without(row, "receipt_sha256")
            or row.get("non_scored") is not True
            or row.get("generic_non_benchmark") is not True
            or row.get("tools") != ["bash", "submit_report"]
            or row.get("requests_started") != 2
            or row.get("requests_succeeded") != 2
            or row.get("protocol_valid_requests") != 2
            or row.get("task_instance_session_verifier_scoring_mutations") != 0
            or row.get("response_content_persisted") is not False
            or row.get("passed") is not True
        ):
            raise DedicatedLaptopError("qualification_stream_invalid")
    elapsed = max(float(row["elapsed_seconds"]) for row in streams)
    if not math.isfinite(elapsed) or elapsed <= 0 or elapsed > 600:
        raise DedicatedLaptopError("qualification_elapsed_invalid")
    body = {
        "schema_version": QUALIFICATION_SCHEMA,
        "status": "PASSED_NON_SCORED",
        "server_binding": expected_binding,
        "concurrency": 2,
        "lanes": ["cluster_internal", "local_port_forward"],
        "streams_started": 2,
        "streams_succeeded": 2,
        "requests_succeeded": 4,
        "protocol_valid_requests": 4,
        "wave_elapsed_seconds": round(elapsed, 6),
        "streams_per_minute": round(120.0 / elapsed, 6),
        "source_receipt_sha256": [row["receipt_sha256"] for row in streams],
        "scored_launch_authorized": False,
        "request_policy": {
            "benchmark_requests": 0,
            "task_instance_session_verifier_scoring_mutations": 0,
            "response_content_persisted": False,
        },
        "privacy": {
            "credentials_included": False,
            "prompts_traces_flags_scores_or_model_outputs_included": False,
        },
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body
