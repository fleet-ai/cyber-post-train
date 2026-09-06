"""Score-free, UID-bound Qwen DP6 qualification controller runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import opencode_actual_harness_parity_v1 as parity
from evals.fleet import opencode_staged_image_v1 as staged_image
from evals.fleet import qwen38_dp6_m_live_v1 as server_live
from evals.fleet import qwen38_dp6_m_scorefree_v1 as early
from evals.fleet import qwen38_dp6_metric_observer_v5 as metric_observer
from evals.fleet import qwen38_dp8_post_rank99_launch_v1 as core
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-dp6-m-qualification-plan-v1"
RESULT_SCHEMA = "fleet-qwen38-dp6-m-qualification-result-v1"
DISTRIBUTION_SCHEMA = "fleet-qwen38-dp6-wave-distribution-v1"
COUNTER_SNAPSHOT_SCHEMA = "fleet-qwen38-dp6-wave-counter-snapshot-v1"
MIN_LATENCY_HEADROOM_MILLISECONDS = 120_000
MIN_LATENCY_HEADROOM_FRACTION = 0.20
SUBMISSION_SCHEMA = "fleet-qwen38-dp6-m-submission-v1"
LIVE_GATE_SCHEMA = "fleet-qwen38-dp6-m-live-gate-v1"
BINDING_SCHEMA = "fleet-qwen38-dp6-m-server-binding-v1"
NAMESPACE = "fleet-train-jobs"
EVENT_SCHEMA = metric_observer.EVENT_SCHEMA
LEVELS = (1, 2, 4, 6)
RANKS = 6
SERVER_BINDING_PATH = Path(early.RUN_DIR) / "lifecycle/SERVER-BINDING.json"
COUNTER_STATE_PATH = Path(early.RUN_DIR) / "lifecycle/.request-counters.json"
COUNTER_BASELINE_PATH = Path(early.RUN_DIR) / "lifecycle/REQUEST-COUNTER-BASELINE.json"
EVENT_DIR = Path(early.RUN_DIR) / "lifecycle/real-traffic-events"
COUNTER_SNAPSHOT_DIR = Path(early.QUALIFIER_OUTPUT_ROOT) / "counter-snapshots"
DIND_RESOURCE_SAMPLES_PATH = Path("/workspace/dind-resource-samples.tsv")
DIND_RESOURCE_READY_PATH = Path("/workspace/dind-resource-producer.ready")
DIND_RESOURCE_READY_BYTES = b"fleet-qwen38-dp6-m-dind-resource-producer-v1\n"
DIND_RESOURCE_SAMPLE_MAX_AGE_SECONDS = 5
DIND_RESOURCE_SAMPLE_ADVANCE_WAIT_SECONDS = 5
QUALIFIER_RELEASE_SCHEMA = "fleet-qwen38-dp6-m-qualifier-release-v1"
DRAIN_PATH = Path(early.RUN_DIR) / "lifecycle/DRAIN"
BASELINE_WAIT_SECONDS = 30
RESOURCE_SAMPLE_WAIT_SECONDS = 30
DRAIN_SCHEMA = "fleet-qwen38-dp6-qualifier-drain-request-v1"
FAILURE_SCHEMA = "fleet-qwen38-dp6-qualifier-failure-v1"
FAILURE_CODES = {
    "controller_resource_control_failed",
    "latency_headroom_insufficient",
    "pre_request_resource_sample_unavailable",
    "probe_evidence_invalid",
    "wave_distribution_invalid",
    "wave_observation_unavailable",
}
BINDING_KEYS = {
    "schema_version",
    "status",
    "submission_receipt_sha256",
    "api_run_id",
    "title",
    "run_dir",
    "serving_block",
    "ray_cluster_name",
    "ray_cluster_uid",
    "service_name",
    "service_origin",
    "rayjob_uid",
    "workload_uid",
    "head_pod_name",
    "head_pod_uid",
    "service_uid",
    "image",
    "model_revision",
    "context_length",
    "tensor_parallel_size",
    "data_parallel_size",
    "head_pod_running_ready",
    "head_pod_restarts",
    "kueue_preempted",
    "scoring_authorized",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
COUNTER_SNAPSHOT_KEYS = {
    "schema_version",
    "status",
    "concurrency",
    "global_request_total",
    "server_binding_receipt_sha256",
    "service_origin",
    "sampler_state_used_as_counter_authority",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
BASELINE_KEYS = {
    "schema_version",
    "status",
    "server_run_dir",
    "api_run_id",
    "pod_name",
    "pod_uid",
    "service_uid",
    "server_binding_receipt_sha256",
    "observed_at_epoch",
    metric_observer.STATE_KEY,
    "request_delta_since_prior_sample",
    "traffic_refresh_performed",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
RESOURCE_SAMPLE_KEYS = {"observed_at_epoch", "oom_kill", "nr_throttled", "throttled_usec"}
RESOURCE_CONTROL_KEYS = {
    "schema_version",
    "status",
    "before",
    "after",
    "deltas",
    "dind_oom_kill_delta",
    "dind_nr_throttled_delta",
    "dind_throttled_usec_delta",
    "protocol_errors",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
DISTRIBUTION_KEYS = {
    "schema_version",
    "status",
    "plan_receipt_sha256",
    "server_binding",
    "service_origin",
    "concurrency",
    "stream_receipt_sha256s",
    "counter_before_snapshot",
    "counter_after_snapshot",
    "global_request_total_before",
    "global_request_total_after",
    "global_request_delta",
    "per_rank_request_attribution_claimed",
    "gpu_peak_utilization_percent_by_device",
    "gpu_peak_memory_used_mib_by_device",
    "gpu_device_count",
    "gpu_memory_loaded_count",
    "sampling_seconds",
    "observer_errors",
    "task_instance_session_verifier_scoring_calls",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
FAILED_STREAM_SCHEMA = "fleet-qwen38-dp6-sanitized-failed-stream-v1"
FAILED_STREAM_KEYS = {
    "schema_version",
    "status",
    "failure_code",
    "execution",
    "privacy",
    "receipt_sha256",
}
LEVEL_KEYS = {
    "concurrency",
    "status",
    "completed_stream_count",
    "observed_model_request_count",
    "observed_server_request_delta",
    "elapsed_milliseconds",
    "timeout_budget_milliseconds",
    "latency_headroom_milliseconds",
    "required_latency_headroom_milliseconds",
    "minimum_latency_headroom_fraction",
    "error_count",
    "tool_order_exact",
    "tool_arguments_exact",
    "failure_stage",
    "failure_code",
    "stream_receipts",
    "distribution_receipt",
    "controller_resource_receipt",
    "stable_counter_baseline_receipt",
}
RESULT_KEYS = {
    "schema_version",
    "plan_receipt_sha256",
    "qualification_plan",
    "levels",
    "highest_passing_concurrency",
    "scored_calls",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
RUNTIME_RELEASE_KEYS = {
    "schema_version",
    "status",
    "launch_authorized",
    "scoring_authorized",
    "job_name",
    "configmap_name",
    "output_root",
    "serving_block",
    "submission_receipt_sha256",
    "server_binding_receipt_sha256",
    "package_sha256",
    "harness_runtime_image",
    "fresh_job_matches",
    "fresh_configmap_matches",
    "fresh_output_root_exists",
    "server_running_ready_restart0",
    "api_mutations_before_create",
    "task_instance_session_verifier_scoring_calls",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
EVENT_KEYS = {
    "schema_version",
    "status",
    "server_run_dir",
    "api_run_id",
    "pod_name",
    "pod_uid",
    "service_uid",
    "server_binding_receipt_sha256",
    "observed_at_epoch",
    "global_request_total_before",
    "global_request_total_after",
    "global_request_delta",
    "running_requests",
    "queued_requests",
    "gpu_memory_used_mib_by_device",
    "gpu_utilization_percent_by_device",
    "activity_reasons",
    "per_rank_request_attribution_claimed",
    "prompts_traces_flags_or_scores_included",
    "receipt_sha256",
}
MAX_JSON_BYTES = 16 * 1024 * 1024


class PreRequestResourceSampleError(RuntimeError):
    """The sidecar resource producer did not become readable before a probe."""


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{path} is not a safe bounded JSON input")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in rows:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = item
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return self_hosted.digest_without(dict(value), "receipt_sha256")


def _write_drain_once(reason: str) -> dict[str, Any]:
    if reason not in {"qualification_below_c6", "qualifier_infrastructure_failure"}:
        raise ValueError("DP6 drain reason is not classified")
    body = {
        "schema_version": DRAIN_SCHEMA,
        "status": "DRAIN_REQUESTED",
        "reason": reason,
        "server_run_dir": early.RUN_DIR,
        "serving_block": early.SERVING_BLOCK,
        "scoring_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    body["receipt_sha256"] = _digest(body)
    try:
        self_hosted.write_json_once(DRAIN_PATH, body)
    except FileExistsError:
        if DRAIN_PATH.is_symlink():
            raise RuntimeError("DP6 drain path is a symlink") from None
        existing = _load(DRAIN_PATH)
        if existing != body:
            raise RuntimeError("DP6 drain receipt already exists with different bytes") from None
    return body


def _write_failure_once(path: Path, stage: str) -> dict[str, Any]:
    body = {
        "schema_version": FAILURE_SCHEMA,
        "status": "FAILED_INFRASTRUCTURE_BEFORE_QUALIFICATION_COMPLETION",
        "stage": stage,
        "failure_code": "wave_observation_unavailable",
        "server_run_dir": early.RUN_DIR,
        "serving_block": early.SERVING_BLOCK,
        "retry_authorized": False,
        "scoring_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    body["receipt_sha256"] = _digest(body)
    try:
        self_hosted.write_json_once(path, body)
    except FileExistsError:
        if _load(path) != body:
            raise RuntimeError("DP6 qualifier failure receipt drifted") from None
    return body


def validate_submission(value: Mapping[str, Any], root: Path) -> None:
    source_commit = value.get("source_commit")
    release_sha256 = value.get("server_release_receipt_sha256")
    if not isinstance(source_commit, str) or not isinstance(release_sha256, str):
        raise ValueError("early DP6 submission authority is incomplete")
    server_live.validate_submission(
        value,
        {"receipt_sha256": release_sha256},
        root,
        source_commit,
    )


def validate_binding(
    value: Mapping[str, Any], submission: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    validate_submission(submission, root)
    if (
        set(value) != BINDING_KEYS
        or value.get("receipt_sha256") != _digest(value)
        or (
            value.get("schema_version") != BINDING_SCHEMA
            or value.get("status") != "READY_NON_SCORED"
            or value.get("submission_receipt_sha256") != submission.get("receipt_sha256")
            or value.get("api_run_id") != submission.get("api_run_id")
            or value.get("title") != early.TITLE
            or value.get("run_dir") != early.RUN_DIR
            or value.get("serving_block") != early.SERVING_BLOCK
            or value.get("image") != early.runtime.IMAGE
            or value.get("model_revision") != early.runtime.MODEL_REVISION
            or value.get("context_length") != 262144
            or value.get("tensor_parallel_size") != 1
            or value.get("data_parallel_size") != 6
            or value.get("head_pod_running_ready") is not True
            or value.get("head_pod_restarts") != 0
            or value.get("kueue_preempted") is not False
            or value.get("scoring_authorized") is not False
            or value.get("prompts_traces_flags_or_scores_included") is not False
        )
    ):
        raise ValueError("early DP6 server binding drifted")
    for field in ("workload_uid", "ray_cluster_uid"):
        try:
            parsed_uid = uuid.UUID(str(value.get(field)))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"early DP6 {field} drifted") from exc
        if parsed_uid.int == 0:
            raise ValueError(f"early DP6 {field} drifted")
    if not isinstance(value.get("head_pod_name"), str) or not value.get("head_pod_name"):
        raise ValueError("early DP6 head Pod name drifted")
    service_name = value.get("service_name")
    api_run_id = value.get("api_run_id")
    ray_cluster_name = value.get("ray_cluster_name")
    expected_origin = f"http://{service_name}.{NAMESPACE}.svc.cluster.local:8000"
    expected_service_name = f"{ray_cluster_name}-head-svc"
    expected_head_pod_prefix = f"{ray_cluster_name}-head-"
    head_pod_suffix = str(value["head_pod_name"]).removeprefix(expected_head_pod_prefix)
    if (
        not isinstance(service_name, str)
        or not isinstance(api_run_id, str)
        or not isinstance(ray_cluster_name, str)
        or not ray_cluster_name.startswith(f"{api_run_id}-")
        or service_name != expected_service_name
        or not str(value["head_pod_name"]).startswith(expected_head_pod_prefix)
        or not head_pod_suffix
        or value["head_pod_name"] == service_name
        or value.get("service_origin") != expected_origin
    ):
        raise ValueError("early DP6 UID-bound Service identity drifted")
    origin = value.get("service_origin")
    if (
        not isinstance(origin, str)
        or not origin.startswith("http://")
        or not origin.endswith(":8000")
    ):
        raise ValueError("early DP6 service origin drifted")
    validated = parity.validate_server_binding(_parity_binding(value), "qwen3.8-27b")
    return {
        **validated,
        "workload_uid": value["workload_uid"],
        "ray_cluster_name": ray_cluster_name,
        "ray_cluster_uid": value["ray_cluster_uid"],
        "service_name": service_name,
        "service_origin": origin,
        "head_pod_name": value["head_pod_name"],
        "server_binding_receipt_sha256": value["receipt_sha256"],
    }


def _parity_binding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "api_run_id": value["api_run_id"],
        "rayjob_uid": value["rayjob_uid"],
        "head_pod_uid": value["head_pod_uid"],
        "service_uid": value["service_uid"],
        "served_id": "qwen3.8-27b",
        "model_revision": value["model_revision"],
        "context_length": value["context_length"],
    }


def validate_runtime_release(
    value: Mapping[str, Any],
    submission: Mapping[str, Any],
    binding: Mapping[str, Any],
    package_path: Path,
) -> None:
    package_digest = "sha256:" + hashlib.sha256(package_path.read_bytes()).hexdigest()
    if (
        set(value) != RUNTIME_RELEASE_KEYS
        or value.get("receipt_sha256") != _digest(value)
        or (
            value.get("schema_version") != QUALIFIER_RELEASE_SCHEMA
            or value.get("status") != "RELEASED_FOR_ONE_NON_SCORED_QUALIFIER"
            or value.get("launch_authorized") is not True
            or value.get("scoring_authorized") is not False
            or value.get("job_name") != early.QUALIFIER_JOB
            or value.get("configmap_name") != early.QUALIFIER_JOB
            or value.get("output_root") != early.QUALIFIER_OUTPUT_ROOT
            or value.get("serving_block") != early.SERVING_BLOCK
            or value.get("submission_receipt_sha256") != submission.get("receipt_sha256")
            or value.get("server_binding_receipt_sha256") != binding.get("receipt_sha256")
            or value.get("package_sha256") != package_digest
            or value.get("harness_runtime_image") != staged_image.identity()
            or value.get("task_instance_session_verifier_scoring_calls") != 0
            or value.get("prompts_traces_flags_or_scores_included") is not False
        )
    ):
        raise ValueError("mounted early DP6 qualifier release drifted")


def project_server_binding_once(value: Mapping[str, Any]) -> None:
    payload = self_hosted.canonical_json(dict(value)) + b"\n"
    SERVER_BINDING_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if SERVER_BINDING_PATH.exists() or SERVER_BINDING_PATH.is_symlink():
        if SERVER_BINDING_PATH.is_symlink() or SERVER_BINDING_PATH.read_bytes() != payload:
            raise FileExistsError("server-local binding projection drifted")
        return
    descriptor = os.open(SERVER_BINDING_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def qualification_plan(binding: Mapping[str, Any], origin: str, root: Path) -> dict[str, Any]:
    treatment = early.load_all(root)[1]["qualification"]
    value = {
        "schema_version": SCHEMA,
        "status": "AUTHORIZED_NON_SCORED_CONTROLLER",
        "launch_authorized": True,
        "scoring_authorized": False,
        "serving_block": early.SERVING_BLOCK,
        "server_binding": dict(binding),
        "service_origin": origin,
        "harness_runtime_image": staged_image.identity(),
        "treatment": treatment,
        "waves": [
            {
                "concurrency": level,
                "parallel_streams": level,
                "model_requests_per_stream": "observed_from_validated_stream_receipt",
                "server_request_delta": "observed_from_uid_bound_metrics",
                "tool_order": ["bash", "submit_report"],
                "strict_tool_arguments_required": True,
            }
            for level in LEVELS
        ],
        "acceptance": {
            "levels_must_run_in_order": True,
            "zero_http_model_or_tool_protocol_errors": True,
            "all_tool_names_order_and_arguments_exact": True,
            "stop_at_first_failed_level": True,
            "highest_passing_level_is_scored_concurrency_ceiling": True,
            "server_local_real_traffic_events_required": True,
        },
        "side_effects": {
            "task_calls": 0,
            "instance_calls": 0,
            "session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
        },
        "privacy": {
            "credentials_included": False,
            "request_or_response_bodies_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    value["receipt_sha256"] = _digest(value)
    return value


def validate_plan(value: Mapping[str, Any], root: Path) -> None:
    if value.get("receipt_sha256") != _digest(value):
        raise ValueError("early DP6 qualification plan digest drifted")
    expected = qualification_plan(
        value.get("server_binding") or {}, str(value.get("service_origin") or ""), root
    )
    if dict(value) != expected:
        raise ValueError("early DP6 qualification plan drifted")
    parity.validate_server_binding(_parity_binding(value["server_binding"]), "qwen3.8-27b")


def _event(path: Path, binding_receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = _load(path)
    before = value.get("global_request_total_before")
    after = value.get("global_request_total_after")
    delta = value.get("global_request_delta")
    running = value.get("running_requests")
    queued = value.get("queued_requests")
    memory = value.get("gpu_memory_used_mib_by_device")
    utilization = value.get("gpu_utilization_percent_by_device")
    expected_reasons: list[str] = []
    if type(before) is int and type(after) is int and after > before:
        expected_reasons.append("completed_request_counter_increased")
    if type(running) is int and running > 0:
        expected_reasons.append("running_requests_positive")
    if type(queued) is int and queued > 0:
        expected_reasons.append("queued_requests_positive")
    if isinstance(utilization, list) and any(
        type(item) is int and item > 0 for item in utilization
    ):
        expected_reasons.append("gpu_utilization_positive")
    if (
        set(value) != EVENT_KEYS
        or value.get("receipt_sha256") != _digest(value)
        or (
            value.get("schema_version") != EVENT_SCHEMA
            or value.get("status") != "REAL_REQUEST_OR_GPU_ACTIVITY_OBSERVED"
            or value.get("server_run_dir") != early.RUN_DIR
            or value.get("api_run_id") != binding_receipt.get("api_run_id")
            or value.get("pod_name") != binding_receipt.get("head_pod_name")
            or value.get("pod_uid") != binding_receipt.get("head_pod_uid")
            or value.get("service_uid") != binding_receipt.get("service_uid")
            or value.get("server_binding_receipt_sha256") != binding_receipt.get("receipt_sha256")
            or type(value.get("observed_at_epoch")) is not int
            or type(before) is not int
            or type(after) is not int
            or type(delta) is not int
            or before < 0
            or after < before
            or delta != after - before
            or type(running) is not int
            or running < 0
            or type(queued) is not int
            or queued < 0
            or value.get("activity_reasons") != expected_reasons
            or not expected_reasons
            or value.get("per_rank_request_attribution_claimed") is not False
            or value.get("prompts_traces_flags_or_scores_included") is not False
        )
    ):
        raise ValueError("server-local traffic event drifted")
    for field in (
        "gpu_memory_used_mib_by_device",
        "gpu_utilization_percent_by_device",
    ):
        row = value.get(field)
        if (
            not isinstance(row, list)
            or len(row) != RANKS
            or not all(type(item) is int for item in row)
        ):
            raise ValueError("server-local traffic event shape drifted")
    if not all(item > 0 for item in memory) or not all(0 <= item <= 100 for item in utilization):
        raise ValueError("server-local traffic event value drifted")
    return value


def _new_events(before: set[Path], binding_receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    current = set(EVENT_DIR.glob("*.json")) if EVENT_DIR.is_dir() else set()
    values = [_event(path, binding_receipt) for path in current - before]
    return sorted(values, key=lambda row: (row["observed_at_epoch"], row["receipt_sha256"]))


def _stable_counter_baseline(binding_receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Wait for a post-binding no-delta observer sample before any model probe."""
    deadline = time.monotonic() + BASELINE_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            baseline = _load(COUNTER_BASELINE_PATH)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            time.sleep(1)
            continue
        counter = baseline.get(metric_observer.STATE_KEY)
        if (
            set(baseline) == BASELINE_KEYS
            and baseline.get("receipt_sha256") == _digest(baseline)
            and baseline.get("schema_version") == metric_observer.BASELINE_SCHEMA
            and baseline.get("status") == "STABLE_BOUND_GLOBAL_REQUEST_BASELINE"
            and baseline.get("server_run_dir") == early.RUN_DIR
            and baseline.get("api_run_id") == binding_receipt.get("api_run_id")
            and baseline.get("pod_name") == binding_receipt.get("head_pod_name")
            and baseline.get("pod_uid") == binding_receipt.get("head_pod_uid")
            and baseline.get("service_uid") == binding_receipt.get("service_uid")
            and baseline.get("server_binding_receipt_sha256")
            == binding_receipt.get("receipt_sha256")
            and type(counter) is int
            and counter >= 0
            and baseline.get("request_delta_since_prior_sample") == 0
            and baseline.get("traffic_refresh_performed") is False
            and baseline.get("prompts_traces_flags_or_scores_included") is False
        ):
            return baseline
        time.sleep(1)
    raise RuntimeError("stable post-binding request-counter baseline unavailable")


def _fetch_global_request_total(service_origin: str) -> int:
    """Read the exact completed-request authority without touching sampler state."""
    with urllib.request.urlopen(f"{service_origin}/metrics", timeout=3) as response:
        metrics = response.read().decode("utf-8")
    return metric_observer.metric_snapshot(metrics)[0]


def _counter_snapshot(
    phase: str,
    total: int,
    level: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    if phase not in {"BEFORE_WAVE", "AFTER_WAVE"}:
        raise ValueError("counter snapshot phase drifted")
    if type(total) is not int or total < 0 or level not in LEVELS:
        raise ValueError("counter snapshot value drifted")
    value = {
        "schema_version": COUNTER_SNAPSHOT_SCHEMA,
        "status": phase,
        "concurrency": level,
        "global_request_total": total,
        "server_binding_receipt_sha256": plan["server_binding"]["receipt_sha256"],
        "service_origin": plan["service_origin"],
        "sampler_state_used_as_counter_authority": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = _digest(value)
    return value


def _validate_counter_snapshot(
    value: Mapping[str, Any], phase: str, level: int, plan: Mapping[str, Any]
) -> None:
    if set(value) != COUNTER_SNAPSHOT_KEYS:
        raise ValueError("counter snapshot contains an unreviewed field")
    expected = _counter_snapshot(phase, value.get("global_request_total"), level, plan)
    if dict(value) != expected:
        raise ValueError("counter snapshot drifted")


def _write_counter_snapshot_once(value: Mapping[str, Any]) -> None:
    phase = str(value["status"]).lower().replace("_wave", "")
    path = COUNTER_SNAPSHOT_DIR / f"c{value['concurrency']}-{phase}.json"
    COUNTER_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        self_hosted.write_json_once(path, dict(value))
    except FileExistsError:
        if path.is_symlink() or _load(path) != dict(value):
            raise RuntimeError("immutable wave counter snapshot drifted") from None


def _dind_resource_sample() -> dict[str, int]:
    try:
        if DIND_RESOURCE_READY_PATH.read_bytes() != DIND_RESOURCE_READY_BYTES:
            raise RuntimeError("dind resource producer readiness drifted")
        lines = [line.split() for line in DIND_RESOURCE_SAMPLES_PATH.read_text().splitlines()]
        raw = lines[-1]
        values = [int(item) for item in raw]
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise RuntimeError("dind resource observer unavailable") from exc
    if len(values) != 4 or any(item < 0 for item in values):
        raise RuntimeError("dind resource observer shape drifted")
    value = dict(
        zip(
            ("observed_at_epoch", "oom_kill", "nr_throttled", "throttled_usec"),
            values,
            strict=True,
        )
    )
    age = int(time.time()) - value["observed_at_epoch"]
    if not -1 <= age <= DIND_RESOURCE_SAMPLE_MAX_AGE_SECONDS:
        raise RuntimeError("dind resource observer sample is stale")
    return value


def _wait_dind_resource_sample() -> dict[str, int]:
    deadline = time.monotonic() + RESOURCE_SAMPLE_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            return _dind_resource_sample()
        except RuntimeError:
            time.sleep(1)
    raise PreRequestResourceSampleError(
        "controller resource sample did not become ready before probe"
    )


def _wait_newer_dind_resource_sample(observed_at_epoch: int) -> dict[str, int]:
    deadline = time.monotonic() + DIND_RESOURCE_SAMPLE_ADVANCE_WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            value = _dind_resource_sample()
            if value["observed_at_epoch"] > observed_at_epoch:
                return value
        except RuntimeError:
            pass
        time.sleep(0.2)
    raise RuntimeError("dind resource observer sample did not advance")


def _dind_resource_control(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, Any]:
    deltas = {
        key: after[key] - before[key] for key in ("oom_kill", "nr_throttled", "throttled_usec")
    }
    value = {
        "schema_version": "fleet-qwen38-dp6-qualifier-controller-resource-v1",
        "status": "PASSED_NO_CONTROLLER_RESOURCE_ERROR",
        "before": dict(before),
        "after": dict(after),
        "deltas": deltas,
        "dind_oom_kill_delta": deltas["oom_kill"],
        "dind_nr_throttled_delta": deltas["nr_throttled"],
        "dind_throttled_usec_delta": deltas["throttled_usec"],
        "protocol_errors": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    if after["observed_at_epoch"] <= before["observed_at_epoch"] or any(
        delta != 0 for delta in deltas.values()
    ):
        value["status"] = "FAILED_CONTROLLER_RESOURCE_ERROR"
    value["receipt_sha256"] = _digest(value)
    return value


def _validate_resource_control(value: Mapping[str, Any]) -> None:
    before = value.get("before")
    after = value.get("after")
    deltas = value.get("deltas")
    if (
        set(value) != RESOURCE_CONTROL_KEYS
        or not isinstance(before, dict)
        or set(before) != RESOURCE_SAMPLE_KEYS
        or not isinstance(after, dict)
        or set(after) != RESOURCE_SAMPLE_KEYS
        or not isinstance(deltas, dict)
        or set(deltas) != {"oom_kill", "nr_throttled", "throttled_usec"}
        or not all(type(item) is int and item >= 0 for item in before.values())
        or not all(type(item) is int and item >= 0 for item in after.values())
        or dict(value) != _dind_resource_control(before, after)
    ):
        raise ValueError("controller resource receipt contains an unreviewed field")


def observe_wave(
    level: int,
    execute: Any,
    plan: Mapping[str, Any],
    binding_receipt: Mapping[str, Any],
    evidence_sink: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    # The background sampler baseline is useful provenance, but it is mutable
    # and never acts as the wave counter authority. Exact metrics are captured
    # on both sides of the model calls so a first-request sampler race cannot
    # erase one completed request from the measured delta.
    baseline = _stable_counter_baseline(binding_receipt)
    if evidence_sink is not None:
        evidence_sink["stable_counter_baseline_receipt"] = baseline
    counter_before = _counter_snapshot(
        "BEFORE_WAVE",
        _fetch_global_request_total(str(plan["service_origin"])),
        level,
        plan,
    )
    _write_counter_snapshot_once(counter_before)
    before_paths = set(EVENT_DIR.glob("*.json")) if EVENT_DIR.is_dir() else set()
    resource_before = _wait_dind_resource_sample()
    produced = execute()
    if not isinstance(produced, list):
        raise RuntimeError("wave stream collection unavailable")
    rows = [_project_stream(row, plan) for row in produced]
    if evidence_sink is not None:
        evidence_sink["stream_receipts"] = rows
    counter_after = _counter_snapshot(
        "AFTER_WAVE",
        _fetch_global_request_total(str(plan["service_origin"])),
        level,
        plan,
    )
    _write_counter_snapshot_once(counter_after)
    deadline = time.monotonic() + 30
    events: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        events = _new_events(before_paths, binding_receipt)
        active = {
            device
            for event in events
            for device, utilization in enumerate(event["gpu_utilization_percent_by_device"])
            if utilization > 0
        }
        if events and (level != RANKS or active == set(range(RANKS))):
            break
        time.sleep(1)
    if not events:
        raise RuntimeError("server-local observer produced no real-traffic event")
    first, last = events[0], events[-1]
    request_delta = counter_after["global_request_total"] - counter_before["global_request_total"]
    gpu_peaks = [
        max(event["gpu_utilization_percent_by_device"][rank] for event in events)
        for rank in range(RANKS)
    ]
    gpu_memory = [
        max(event["gpu_memory_used_mib_by_device"][rank] for event in events)
        for rank in range(RANKS)
    ]
    resource_after = _wait_newer_dind_resource_sample(resource_before["observed_at_epoch"])
    resource_control = _dind_resource_control(resource_before, resource_after)
    if evidence_sink is not None:
        evidence_sink["controller_resource_receipt"] = resource_control
    receipt = {
        "schema_version": DISTRIBUTION_SCHEMA,
        "status": "PASSED_NON_SCORED_DISTRIBUTION",
        "plan_receipt_sha256": plan["receipt_sha256"],
        "server_binding": plan["server_binding"],
        "service_origin": plan["service_origin"],
        "concurrency": level,
        "stream_receipt_sha256s": [row.get("receipt_sha256") for row in rows],
        "counter_before_snapshot": counter_before,
        "counter_after_snapshot": counter_after,
        "global_request_total_before": counter_before["global_request_total"],
        "global_request_total_after": counter_after["global_request_total"],
        "global_request_delta": request_delta,
        "per_rank_request_attribution_claimed": False,
        "gpu_peak_utilization_percent_by_device": gpu_peaks,
        "gpu_peak_memory_used_mib_by_device": gpu_memory,
        "gpu_device_count": RANKS,
        "gpu_memory_loaded_count": sum(item > 0 for item in gpu_memory),
        "sampling_seconds": max(0, last["observed_at_epoch"] - first["observed_at_epoch"]),
        "observer_errors": [],
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    if evidence_sink is not None:
        evidence_sink["distribution_receipt"] = receipt
    return rows, receipt, resource_control, baseline


def validate_distribution_receipt(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    level: int,
    rows: list[Mapping[str, Any]],
) -> None:
    """Require UID-bound counter and GPU activity across the six DP ranks."""
    if set(value) != DISTRIBUTION_KEYS:
        raise ValueError("DP6 distribution receipt contains an unreviewed field")
    before = value.get("global_request_total_before")
    after = value.get("global_request_total_after")
    delta = value.get("global_request_delta")
    counters_valid = (
        type(before) is int
        and type(after) is int
        and type(delta) is int
        and before >= 0
        and after >= before
        and delta == after - before
        and delta > 0
    )
    gpu = value.get("gpu_peak_utilization_percent_by_device")
    memory = value.get("gpu_peak_memory_used_mib_by_device")
    active_gpus = (
        {index for index, count in enumerate(gpu) if type(count) is int and count > 0}
        if isinstance(gpu, list) and len(gpu) == RANKS
        else set()
    )
    required = set(range(RANKS)) if level == RANKS else None
    counter_before = value.get("counter_before_snapshot")
    counter_after = value.get("counter_after_snapshot")
    snapshots_valid = False
    if isinstance(counter_before, dict) and isinstance(counter_after, dict):
        try:
            _validate_counter_snapshot(counter_before, "BEFORE_WAVE", level, plan)
            _validate_counter_snapshot(counter_after, "AFTER_WAVE", level, plan)
            snapshots_valid = before == counter_before.get(
                "global_request_total"
            ) and after == counter_after.get("global_request_total")
        except (KeyError, TypeError, ValueError):
            snapshots_valid = False
    if value.get("receipt_sha256") != _digest(value) or (
        value.get("schema_version") != DISTRIBUTION_SCHEMA
        or value.get("status") != "PASSED_NON_SCORED_DISTRIBUTION"
        or value.get("plan_receipt_sha256") != plan.get("receipt_sha256")
        or value.get("server_binding") != plan.get("server_binding")
        or value.get("service_origin") != plan.get("service_origin")
        or value.get("concurrency") != level
        or value.get("stream_receipt_sha256s") != [row.get("receipt_sha256") for row in rows]
        or not snapshots_valid
        or not counters_valid
        or value.get("per_rank_request_attribution_claimed") is not False
        or not isinstance(gpu, list)
        or len(gpu) != RANKS
        or not all(type(item) is int and 0 <= item <= 100 for item in gpu)
        or not isinstance(memory, list)
        or len(memory) != RANKS
        or not all(type(item) is int and item > 0 for item in memory)
        or value.get("gpu_device_count") != RANKS
        or value.get("gpu_memory_loaded_count") != RANKS
        # Five-second GPU samples can miss a short c1/c2/c4 burst. They remain
        # descriptive at those levels. C6 retains exact all-device activity.
        or (level == RANKS and len(active_gpus) < level)
        or (required is not None and active_gpus != required)
        or value.get("observer_errors") != []
        or value.get("task_instance_session_verifier_scoring_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP6 wave did not prove six-rank request/GPU distribution")


def run(plan: Mapping[str, Any], binding_receipt: Mapping[str, Any], root: Path) -> dict[str, Any]:
    validate_plan(plan, root)
    candidate_binding = plan.get("server_binding")
    server_binding = candidate_binding if isinstance(candidate_binding, Mapping) else {}

    def safe_probe() -> dict[str, Any]:
        try:
            raw = parity.run(
                "qwen3.8-27b",
                "",
                upstream_origin=plan["service_origin"],
                server_binding=_parity_binding(plan["server_binding"]),
                cluster_dind=True,
                expected_image_id=staged_image.RUNTIME_IMAGE_ID,
            )
            return _project_stream(raw, plan)
        except Exception:  # content-free fail-closed classification
            return _failed_stream("probe_execution_failed")

    observations: list[dict[str, Any]] = []
    highest = 0
    for level in LEVELS:

        def execute(wave_level: int = level) -> list[dict[str, Any]]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=wave_level) as executor:
                return list(executor.map(lambda _: safe_probe(), range(wave_level)))

        started = time.monotonic()
        failure_stage: str | None = None
        failure_code: str | None = None
        rows: list[dict[str, Any]] = []
        distribution: dict[str, Any] = {}
        resource_control: dict[str, Any] = {}
        baseline: dict[str, Any] = {}
        captured_evidence: dict[str, Any] = {}
        observed_model_requests = 0
        observed_server_requests = 0
        passed = False
        try:
            rows, distribution, resource_control, baseline = observe_wave(
                level, execute, plan, binding_receipt, captured_evidence
            )
            rows = [_project_stream(row, plan) for row in rows]
            distribution = _project_distribution_evidence(distribution, plan, level, rows)
            resource_control = _project_resource_evidence(resource_control)
            baseline = _project_baseline_evidence(baseline, server_binding)
            streams_ok = all(_valid_stream(row, plan) for row in rows)
            observed_model_requests = sum(
                int((row.get("execution") or {}).get("model_requests") or 0) for row in rows
            )
            candidate_server_requests = distribution.get("global_request_delta")
            if type(candidate_server_requests) is int:
                observed_server_requests = candidate_server_requests
            validate_distribution_receipt(distribution, plan, level, rows)
            passed = (
                len(rows) == level
                and streams_ok
                and observed_model_requests > 0
                and observed_server_requests == observed_model_requests
                and resource_control.get("status") == "PASSED_NO_CONTROLLER_RESOURCE_ERROR"
            )
        except Exception as exc:  # evidence is projected below; classification is fixed
            captured_rows = captured_evidence.get("stream_receipts", rows)
            rows = (
                [_project_stream(row, plan) for row in captured_rows]
                if isinstance(captured_rows, list)
                else []
            )
            distribution = _project_distribution_evidence(
                captured_evidence.get("distribution_receipt", distribution), plan, level, rows
            )
            resource_control = _project_resource_evidence(
                captured_evidence.get("controller_resource_receipt", resource_control)
            )
            baseline = _project_baseline_evidence(
                captured_evidence.get("stable_counter_baseline_receipt", baseline),
                server_binding,
            )
            failure_stage = (
                "pre_request_resource_sample"
                if isinstance(exc, PreRequestResourceSampleError)
                else "qualification_wave"
            )
            failure_code = (
                "pre_request_resource_sample_unavailable"
                if isinstance(exc, PreRequestResourceSampleError)
                else "wave_distribution_invalid"
                if isinstance(exc, ValueError)
                else "wave_observation_unavailable"
            )
        elapsed_milliseconds = int((time.monotonic() - started) * 1000)
        timeout_budget_milliseconds = parity.TIMEOUT_SECONDS * 1000
        latency_headroom_milliseconds = timeout_budget_milliseconds - elapsed_milliseconds
        required_latency_headroom_milliseconds = max(
            MIN_LATENCY_HEADROOM_MILLISECONDS,
            int(timeout_budget_milliseconds * MIN_LATENCY_HEADROOM_FRACTION),
        )
        passed = passed and (
            latency_headroom_milliseconds >= required_latency_headroom_milliseconds
        )
        if not passed and failure_stage is None:
            failure_stage = "qualification_wave"
            failure_code = (
                "controller_resource_control_failed"
                if resource_control.get("status") == "FAILED_CONTROLLER_RESOURCE_ERROR"
                else "latency_headroom_insufficient"
                if latency_headroom_milliseconds < required_latency_headroom_milliseconds
                else "probe_evidence_invalid"
            )
        observations.append(
            {
                "concurrency": level,
                "status": "PASSED" if passed else "FAILED",
                "completed_stream_count": len(rows),
                "observed_model_request_count": observed_model_requests,
                "observed_server_request_delta": observed_server_requests,
                "elapsed_milliseconds": elapsed_milliseconds,
                "timeout_budget_milliseconds": timeout_budget_milliseconds,
                "latency_headroom_milliseconds": latency_headroom_milliseconds,
                "required_latency_headroom_milliseconds": (required_latency_headroom_milliseconds),
                "minimum_latency_headroom_fraction": MIN_LATENCY_HEADROOM_FRACTION,
                "error_count": 0 if passed else 1,
                "tool_order_exact": passed,
                "tool_arguments_exact": passed,
                "failure_stage": failure_stage,
                "failure_code": failure_code,
                "stream_receipts": rows,
                "distribution_receipt": distribution,
                "controller_resource_receipt": resource_control,
                "stable_counter_baseline_receipt": baseline,
            }
        )
        if not passed:
            break
        highest = level
    result = {
        "schema_version": RESULT_SCHEMA,
        "plan_receipt_sha256": plan["receipt_sha256"],
        "qualification_plan": dict(plan),
        "levels": observations,
        "highest_passing_concurrency": highest,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    result["receipt_sha256"] = _digest(result)
    validate_result(result, plan, root)
    return result


def _valid_stream(row: Mapping[str, Any], plan: Mapping[str, Any]) -> bool:
    if row.get("schema_version") == FAILED_STREAM_SCHEMA:
        _validate_failed_stream(row)
        return False
    try:
        core.validate_stream_receipt(row, plan)
    except ValueError:
        return False
    return True


def _failed_stream(failure_code: str) -> dict[str, Any]:
    if failure_code not in {"probe_execution_failed", "probe_evidence_invalid"}:
        raise ValueError("failed stream classification drifted")
    value = {
        "schema_version": FAILED_STREAM_SCHEMA,
        "status": "FAILED",
        "failure_code": failure_code,
        "execution": {
            "model_requests": 0,
            "task_instance_session_verifier_scoring_calls": 0,
            "scored_launch_authorized": False,
        },
        "privacy": {
            "request_or_response_bodies_included": False,
            "prompts_traces_flags_or_scores_included": False,
        },
    }
    value["receipt_sha256"] = _digest(value)
    return value


def _validate_failed_stream(value: Mapping[str, Any]) -> None:
    expected = _failed_stream(str(value.get("failure_code")))
    if set(value) != FAILED_STREAM_KEYS or dict(value) != expected:
        raise ValueError("failed stream receipt contains an unreviewed field")


def _project_stream(value: Any, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _failed_stream("probe_evidence_invalid")
    if value.get("schema_version") == FAILED_STREAM_SCHEMA:
        try:
            _validate_failed_stream(value)
        except Exception:
            return _failed_stream("probe_evidence_invalid")
        return dict(value)
    try:
        core.validate_stream_receipt(value, plan)
    except Exception:
        return _failed_stream("probe_evidence_invalid")
    # The upstream validator requires exact keys recursively. Copy only after
    # that validation so invalid mappings can never be embedded in RESULT.
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_baseline_evidence(value: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    counter = value.get(metric_observer.STATE_KEY)
    if (
        set(value) != BASELINE_KEYS
        or value.get("receipt_sha256") != _digest(value)
        or value.get("schema_version") != metric_observer.BASELINE_SCHEMA
        or value.get("status") != "STABLE_BOUND_GLOBAL_REQUEST_BASELINE"
        or value.get("server_run_dir") != early.RUN_DIR
        or value.get("api_run_id") != binding.get("api_run_id")
        or value.get("pod_name") != binding.get("head_pod_name")
        or value.get("pod_uid") != binding.get("head_pod_uid")
        or value.get("service_uid") != binding.get("service_uid")
        or value.get("server_binding_receipt_sha256")
        != binding.get("server_binding_receipt_sha256")
        or type(value.get("observed_at_epoch")) is not int
        or type(counter) is not int
        or counter < 0
        or value.get("request_delta_since_prior_sample") != 0
        or value.get("traffic_refresh_performed") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("baseline receipt contains an unreviewed field")


def _project_baseline_evidence(value: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        _validate_baseline_evidence(value, binding)
    except Exception:
        return {}
    return dict(value)


def _project_resource_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        _validate_resource_control(value)
    except Exception:
        return {}
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _project_distribution_evidence(
    value: Any,
    plan: Mapping[str, Any],
    level: int,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    try:
        validate_distribution_receipt(value, plan, level, rows)
    except Exception:
        return {}
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _validate_optional_level_evidence(row: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    streams = row.get("stream_receipts")
    if not isinstance(streams, list):
        raise ValueError("stream receipts contain an unreviewed field")
    for stream_receipt in streams:
        if not isinstance(stream_receipt, dict):
            raise ValueError("stream receipts contain an unreviewed field")
        if stream_receipt.get("schema_version") == FAILED_STREAM_SCHEMA:
            _validate_failed_stream(stream_receipt)
        else:
            core.validate_stream_receipt(stream_receipt, plan)

    baseline = row.get("stable_counter_baseline_receipt")
    if baseline:
        if not isinstance(baseline, dict):
            raise ValueError("baseline receipt contains an unreviewed field")
        _validate_baseline_evidence(baseline, plan["server_binding"])

    distribution = row.get("distribution_receipt")
    if distribution:
        if not isinstance(distribution, dict):
            raise ValueError("distribution receipt contains an unreviewed field")
        validate_distribution_receipt(distribution, plan, row["concurrency"], streams)

    resource = row.get("controller_resource_receipt")
    if resource:
        if not isinstance(resource, dict):
            raise ValueError("controller resource receipt contains an unreviewed field")
        _validate_resource_control(resource)


def validate_result(value: Mapping[str, Any], plan: Mapping[str, Any], root: Path) -> int:
    validate_plan(plan, root)
    if (
        set(value) != RESULT_KEYS
        or value.get("receipt_sha256") != _digest(value)
        or (
            value.get("schema_version") != RESULT_SCHEMA
            or value.get("plan_receipt_sha256") != plan.get("receipt_sha256")
            or value.get("qualification_plan") != dict(plan)
            or value.get("scored_calls") != 0
            or value.get("prompts_traces_flags_or_scores_included") is not False
        )
    ):
        raise ValueError("early DP6 qualification result drifted")
    levels = value.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("early DP6 qualification result is empty")
    seen_failure = False
    failure_count = 0
    highest = 0
    for index, row in enumerate(levels):
        if (
            index >= len(LEVELS)
            or not isinstance(row, dict)
            or set(row) != LEVEL_KEYS
            or row.get("concurrency") != LEVELS[index]
            or row.get("status") not in {"PASSED", "FAILED"}
            or type(row.get("completed_stream_count")) is not int
            or row["completed_stream_count"] < 0
            or type(row.get("observed_model_request_count")) is not int
            or row["observed_model_request_count"] < 0
            or type(row.get("observed_server_request_delta")) is not int
            or row["observed_server_request_delta"] < 0
            or type(row.get("elapsed_milliseconds")) is not int
            or row["elapsed_milliseconds"] < 0
            or row.get("timeout_budget_milliseconds") != parity.TIMEOUT_SECONDS * 1000
            or type(row.get("latency_headroom_milliseconds")) is not int
            or row["latency_headroom_milliseconds"]
            != row["timeout_budget_milliseconds"] - row["elapsed_milliseconds"]
            or row.get("required_latency_headroom_milliseconds")
            != max(
                MIN_LATENCY_HEADROOM_MILLISECONDS,
                int(row["timeout_budget_milliseconds"] * MIN_LATENCY_HEADROOM_FRACTION),
            )
            or row.get("minimum_latency_headroom_fraction") != MIN_LATENCY_HEADROOM_FRACTION
            or row.get("error_count") != (0 if row.get("status") == "PASSED" else 1)
            or row.get("tool_order_exact") is not (row.get("status") == "PASSED")
            or row.get("tool_arguments_exact") is not (row.get("status") == "PASSED")
        ):
            raise ValueError("early DP6 qualification ladder order drifted")
        _validate_optional_level_evidence(row, plan)
        if row.get("status") == "FAILED" and (
            row.get("failure_stage") not in {"pre_request_resource_sample", "qualification_wave"}
            or row.get("failure_code") not in FAILURE_CODES
        ):
            raise ValueError("early DP6 qualification failure is not sanitized")
        if row.get("status") == "PASSED" and (
            row.get("failure_stage") is not None or row.get("failure_code") is not None
        ):
            raise ValueError("passing DP6 qualification carries failure metadata")
        resource = row.get("controller_resource_receipt")
        if row.get("status") == "PASSED" and (
            not isinstance(resource, dict)
            or resource.get("receipt_sha256") != _digest(resource)
            or resource.get("status") != "PASSED_NO_CONTROLLER_RESOURCE_ERROR"
            or resource.get("dind_oom_kill_delta") != 0
            or resource.get("dind_nr_throttled_delta") != 0
            or resource.get("dind_throttled_usec_delta") != 0
            or resource.get("protocol_errors") != 0
        ):
            raise ValueError("early DP6 controller resource control drifted")
        if row.get("status") == "PASSED":
            baseline = row.get("stable_counter_baseline_receipt")
            distribution = row.get("distribution_receipt")
            streams = row.get("stream_receipts")
            model_requests = (
                sum(
                    int((stream.get("execution") or {}).get("model_requests") or 0)
                    for stream in streams
                    if isinstance(stream, dict)
                )
                if isinstance(streams, list)
                else -1
            )
            server_requests = (
                distribution.get("global_request_delta", -1)
                if isinstance(distribution, dict)
                else -1
            )
            if (
                not isinstance(baseline, dict)
                or row.get("completed_stream_count") != row.get("concurrency")
                or row.get("observed_model_request_count") != model_requests
                or row.get("observed_server_request_delta") != server_requests
                or model_requests <= 0
                or server_requests != model_requests
                or type(row.get("elapsed_milliseconds")) is not int
                or row["elapsed_milliseconds"] < 0
                or row.get("timeout_budget_milliseconds") != parity.TIMEOUT_SECONDS * 1000
                or row.get("latency_headroom_milliseconds")
                != row["timeout_budget_milliseconds"] - row["elapsed_milliseconds"]
                or row.get("required_latency_headroom_milliseconds")
                != max(
                    MIN_LATENCY_HEADROOM_MILLISECONDS,
                    int(row["timeout_budget_milliseconds"] * MIN_LATENCY_HEADROOM_FRACTION),
                )
                or row.get("minimum_latency_headroom_fraction") != MIN_LATENCY_HEADROOM_FRACTION
                or row["latency_headroom_milliseconds"]
                < row["required_latency_headroom_milliseconds"]
            ):
                raise ValueError("early DP6 observed request/latency evidence drifted")
        if row.get("status") == "FAILED":
            seen_failure = True
            failure_count += 1
        elif row.get("status") == "PASSED" and not seen_failure:
            highest = row["concurrency"]
        else:
            raise ValueError("early DP6 qualification violates stop-on-failure")
    if seen_failure and (failure_count != 1 or levels[-1].get("status") != "FAILED"):
        raise ValueError("early DP6 qualification continued after failure")
    if value.get("highest_passing_concurrency") != highest:
        raise ValueError("early DP6 qualification ceiling drifted")
    return highest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--server-binding", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd()
    stage = "load_inputs"
    try:
        submission = _load(args.submission)
        binding_receipt = _load(args.server_binding)
        release = _load(args.release)
        stage = "validate_release_and_binding"
        binding = validate_binding(binding_receipt, submission, root)
        validate_runtime_release(release, submission, binding_receipt, args.package)
        stage = "project_server_binding"
        project_server_binding_once(binding_receipt)
        stage = "run_score_free_ladder"
        plan = qualification_plan(binding, binding_receipt["service_origin"], root)
        result = run(plan, binding_receipt, root)
        stage = "write_result"
        self_hosted.write_json_once(args.output, result)
        if result["highest_passing_concurrency"] < RANKS:
            _write_drain_once("qualification_below_c6")
    except Exception:
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_failure_once(args.output.parent / "FAILED.json", stage)
        _write_drain_once("qualifier_infrastructure_failure")
        raise RuntimeError("DP6-m qualifier failed safely") from None
    print(json.dumps({"highest_passing_concurrency": result["highest_passing_concurrency"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
