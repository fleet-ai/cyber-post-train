"""Score-free, UID-bound Qwen DP6 qualification controller runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import opencode_actual_harness_parity_v1 as parity
from evals.fleet import opencode_staged_image_v1 as staged_image
from evals.fleet import qwen38_dp6_l_live_v1 as server_live
from evals.fleet import qwen38_dp6_l_scorefree_v1 as early
from evals.fleet import qwen38_dp6_metric_observer_v4 as metric_observer
from evals.fleet import qwen38_dp8_post_rank99_launch_v1 as core
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-dp6-l-qualification-plan-v1"
RESULT_SCHEMA = "fleet-qwen38-dp6-l-qualification-result-v1"
DISTRIBUTION_SCHEMA = "fleet-qwen38-dp6-wave-distribution-v1"
MIN_LATENCY_HEADROOM_MILLISECONDS = 120_000
MIN_LATENCY_HEADROOM_FRACTION = 0.20
SUBMISSION_SCHEMA = "fleet-qwen38-dp6-l-submission-v1"
LIVE_GATE_SCHEMA = "fleet-qwen38-dp6-l-live-gate-v1"
BINDING_SCHEMA = "fleet-qwen38-dp6-l-server-binding-v1"
NAMESPACE = "fleet-train-jobs"
EVENT_SCHEMA = metric_observer.EVENT_SCHEMA
LEVELS = (1, 2, 4, 6)
RANKS = 6
SERVER_BINDING_PATH = Path(early.RUN_DIR) / "lifecycle/SERVER-BINDING.json"
COUNTER_STATE_PATH = Path(early.RUN_DIR) / "lifecycle/.request-counters.json"
COUNTER_BASELINE_PATH = Path(early.RUN_DIR) / "lifecycle/REQUEST-COUNTER-BASELINE.json"
EVENT_DIR = Path(early.RUN_DIR) / "lifecycle/real-traffic-events"
DIND_RESOURCE_SAMPLES_PATH = Path("/workspace/dind-resource-samples.tsv")
DIND_RESOURCE_READY_PATH = Path("/workspace/dind-resource-producer.ready")
DIND_RESOURCE_READY_BYTES = b"fleet-qwen38-dp6-l-dind-resource-producer-v1\n"
DIND_RESOURCE_SAMPLE_MAX_AGE_SECONDS = 5
DIND_RESOURCE_SAMPLE_ADVANCE_WAIT_SECONDS = 5
QUALIFIER_RELEASE_SCHEMA = "fleet-qwen38-dp6-l-qualifier-release-v1"
DRAIN_PATH = Path(early.RUN_DIR) / "lifecycle/DRAIN"
BASELINE_WAIT_SECONDS = 30
RESOURCE_SAMPLE_WAIT_SECONDS = 30
DRAIN_SCHEMA = "fleet-qwen38-dp6-qualifier-drain-request-v1"
FAILURE_SCHEMA = "fleet-qwen38-dp6-qualifier-failure-v1"
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


class PreRequestResourceSampleError(RuntimeError):
    """The sidecar resource producer did not become readable before a probe."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
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


def _write_failure_once(path: Path, stage: str, error: Exception) -> dict[str, Any]:
    body = {
        "schema_version": FAILURE_SCHEMA,
        "status": "FAILED_INFRASTRUCTURE_BEFORE_QUALIFICATION_COMPLETION",
        "stage": stage,
        "error_type": type(error).__name__,
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
            raise RuntimeError("DP6 qualifier failure receipt drifted") from error
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
    if value.get("receipt_sha256") != _digest(value) or (
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
    if value.get("receipt_sha256") != _digest(value) or (
        value.get("schema_version") != EVENT_SCHEMA
        or value.get("status") != "REAL_REQUEST_OR_GPU_ACTIVITY_OBSERVED"
        or value.get("server_run_dir") != early.RUN_DIR
        or value.get("api_run_id") != binding_receipt.get("api_run_id")
        or value.get("pod_name") != binding_receipt.get("head_pod_name")
        or value.get("pod_uid") != binding_receipt.get("head_pod_uid")
        or value.get("service_uid") != binding_receipt.get("service_uid")
        or value.get("server_binding_receipt_sha256") != binding_receipt.get("receipt_sha256")
        or value.get("prompts_traces_flags_or_scores_included") is not False
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
            state = _load(COUNTER_STATE_PATH)
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            time.sleep(1)
            continue
        counter = baseline.get(metric_observer.STATE_KEY)
        if (
            baseline.get("receipt_sha256") == _digest(baseline)
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
            and state == {metric_observer.STATE_KEY: counter}
            and baseline.get("request_delta_since_prior_sample") == 0
            and baseline.get("traffic_refresh_performed") is False
            and baseline.get("prompts_traces_flags_or_scores_included") is False
        ):
            return baseline
        time.sleep(1)
    raise RuntimeError("stable post-binding request-counter baseline unavailable")


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


def observe_wave(
    level: int,
    execute: Any,
    plan: Mapping[str, Any],
    binding_receipt: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline = _stable_counter_baseline(binding_receipt)
    before_paths = set(EVENT_DIR.glob("*.json")) if EVENT_DIR.is_dir() else set()
    resource_before = _wait_dind_resource_sample()
    rows = execute()
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
        completed_delta = sum(event["global_request_delta"] for event in events)
        if completed_delta > 0 and len(active) >= level:
            break
        time.sleep(1)
    if not events:
        raise RuntimeError("server-local observer produced no real-traffic event")
    first, last = events[0], events[-1]
    request_delta = last["global_request_total_after"] - first["global_request_total_before"]
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
    if resource_control["status"] != "PASSED_NO_CONTROLLER_RESOURCE_ERROR":
        raise RuntimeError("qualifier controller resource control failed")
    receipt = {
        "schema_version": DISTRIBUTION_SCHEMA,
        "status": "PASSED_NON_SCORED_DISTRIBUTION",
        "plan_receipt_sha256": plan["receipt_sha256"],
        "server_binding": plan["server_binding"],
        "service_origin": plan["service_origin"],
        "concurrency": level,
        "stream_receipt_sha256s": [row.get("receipt_sha256") for row in rows],
        "global_request_total_before": first["global_request_total_before"],
        "global_request_total_after": last["global_request_total_after"],
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
    return rows, receipt, resource_control, baseline


def validate_distribution_receipt(
    value: Mapping[str, Any],
    plan: Mapping[str, Any],
    level: int,
    rows: list[Mapping[str, Any]],
) -> None:
    """Require UID-bound counter and GPU activity across the six DP ranks."""
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
    if value.get("receipt_sha256") != _digest(value) or (
        value.get("schema_version") != DISTRIBUTION_SCHEMA
        or value.get("status") != "PASSED_NON_SCORED_DISTRIBUTION"
        or value.get("plan_receipt_sha256") != plan.get("receipt_sha256")
        or value.get("server_binding") != plan.get("server_binding")
        or value.get("service_origin") != plan.get("service_origin")
        or value.get("concurrency") != level
        or value.get("stream_receipt_sha256s") != [row.get("receipt_sha256") for row in rows]
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
        or len(active_gpus) < level
        or (required is not None and active_gpus != required)
        or value.get("observer_errors") != []
        or value.get("task_instance_session_verifier_scoring_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("DP6 wave did not prove six-rank request/GPU distribution")


def run(plan: Mapping[str, Any], binding_receipt: Mapping[str, Any], root: Path) -> dict[str, Any]:
    validate_plan(plan, root)

    def safe_probe() -> dict[str, Any]:
        try:
            return parity.run(
                "qwen3.8-27b",
                "",
                upstream_origin=plan["service_origin"],
                server_binding=_parity_binding(plan["server_binding"]),
                cluster_dind=True,
                expected_image_id=staged_image.RUNTIME_IMAGE_ID,
            )
        except Exception as exc:  # content-free fail-closed classification
            value = {
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "tool_contract": {},
                "execution": {"task_instance_session_verifier_scoring_calls": 0},
                "privacy": {"responses_or_model_outputs_included": False},
            }
            value["receipt_sha256"] = _digest(value)
            return value

    observations: list[dict[str, Any]] = []
    highest = 0
    for level in LEVELS:

        def execute(wave_level: int = level) -> list[dict[str, Any]]:
            with concurrent.futures.ThreadPoolExecutor(max_workers=wave_level) as executor:
                return list(executor.map(lambda _: safe_probe(), range(wave_level)))

        started = time.monotonic()
        failure_stage: str | None = None
        failure_type: str | None = None
        try:
            rows, distribution, resource_control, baseline = observe_wave(
                level, execute, plan, binding_receipt
            )
            streams_ok = all(_valid_stream(row, plan) for row in rows)
            validate_distribution_receipt(distribution, plan, level, rows)
            observed_model_requests = sum(
                int((row.get("execution") or {}).get("model_requests") or 0) for row in rows
            )
            observed_server_requests = distribution["global_request_delta"]
            passed = (
                len(rows) == level
                and streams_ok
                and observed_model_requests > 0
                and observed_server_requests == observed_model_requests
            )
        except (RuntimeError, ValueError) as exc:
            rows, distribution, resource_control, baseline, passed = [], {}, {}, {}, False
            observed_model_requests = 0
            observed_server_requests = 0
            failure_stage = (
                "pre_request_resource_sample"
                if isinstance(exc, PreRequestResourceSampleError)
                else "qualification_wave"
            )
            failure_type = type(exc).__name__
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
            failure_type = "QualificationWaveValidationError"
        observations.append(
            {
                "concurrency": level,
                "status": "PASSED" if passed else "FAILED",
                "completed_stream_count": len(rows) if passed else 0,
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
                "error_type": failure_type,
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
    try:
        core.validate_stream_receipt(row, plan)
    except ValueError:
        return False
    return True


def validate_result(value: Mapping[str, Any], plan: Mapping[str, Any], root: Path) -> int:
    validate_plan(plan, root)
    if value.get("receipt_sha256") != _digest(value) or (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("plan_receipt_sha256") != plan.get("receipt_sha256")
        or value.get("qualification_plan") != dict(plan)
        or value.get("scored_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP6 qualification result drifted")
    levels = value.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("early DP6 qualification result is empty")
    seen_failure = False
    failure_count = 0
    highest = 0
    for index, row in enumerate(levels):
        if not isinstance(row, dict) or row.get("concurrency") != LEVELS[index]:
            raise ValueError("early DP6 qualification ladder order drifted")
        if row.get("status") == "FAILED" and (
            row.get("failure_stage")
            not in {"pre_request_resource_sample", "qualification_wave"}
            or not isinstance(row.get("error_type"), str)
            or not row["error_type"].isascii()
            or not row["error_type"].isidentifier()
            or len(row["error_type"]) > 64
        ):
            raise ValueError("early DP6 qualification failure is not sanitized")
        if row.get("status") == "PASSED" and (
            row.get("failure_stage") is not None or row.get("error_type") is not None
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
                or baseline.get("receipt_sha256") != _digest(baseline)
                or baseline.get("schema_version") != metric_observer.BASELINE_SCHEMA
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
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _write_failure_once(args.output.parent / "FAILED.json", stage, exc)
        _write_drain_once("qualifier_infrastructure_failure")
        raise
    print(json.dumps({"highest_passing_concurrency": result["highest_passing_concurrency"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
