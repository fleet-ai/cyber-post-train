"""Score-free, UID-bound Qwen DP8 qualification controller runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import opencode_actual_harness_parity_v1 as parity
from evals.fleet import qwen38_dp8_early_qualification_v1 as early
from evals.fleet import qwen38_dp8_metric_observer_v2 as metric_observer
from evals.fleet import qwen38_dp8_post_rank99_launch_v1 as core
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-dp8-early-qualification-plan-v2"
RESULT_SCHEMA = "fleet-qwen38-dp8-early-qualification-result-v2"
SUBMISSION_SCHEMA = "fleet-qwen38-dp8-early-submission-v2"
BINDING_SCHEMA = "fleet-qwen38-dp8-early-server-binding-v2"
EVENT_SCHEMA = "fleet-qwen38-dp8-real-traffic-observation-v2"
LEVELS = (1, 2, 4, 8)
SERVER_BINDING_PATH = Path(early.RUN_DIR) / "lifecycle/SERVER-BINDING.json"
COUNTER_STATE_PATH = Path(early.RUN_DIR) / "lifecycle/.request-counters.json"
COUNTER_BASELINE_PATH = Path(early.RUN_DIR) / "lifecycle/REQUEST-COUNTER-BASELINE.json"
EVENT_DIR = Path(early.RUN_DIR) / "lifecycle/real-traffic-events"
DIND_RESOURCE_SAMPLES_PATH = Path("/workspace/dind-resource-samples.tsv")
QUALIFIER_RELEASE_SCHEMA = "fleet-qwen38-dp8-early-qualifier-release-v2"
BASELINE_WAIT_SECONDS = 30


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return self_hosted.digest_without(dict(value), "receipt_sha256")


def validate_submission(value: Mapping[str, Any], root: Path) -> None:
    config = early.load_all(root)[0]
    gate = value.get("live_gate")
    if not isinstance(gate, dict):
        raise ValueError("early DP8 submission omitted durable live gate")
    if value.get("receipt_sha256") != _digest(value) or (
        value.get("schema_version") != SUBMISSION_SCHEMA
        or value.get("status") != "SUBMITTED_NON_SCORED_SERVER"
        or value.get("title") != early.TITLE
        or value.get("run_dir") != early.RUN_DIR
        or value.get("serving_block") != early.SERVING_BLOCK
        or value.get("config_sha256") != config["config_sha256"]
        or value.get("request_sha256")
        != self_hosted.sha256(self_hosted.canonical_json(early.jobs_payload(root)))
        or value.get("live_gate_receipt_sha256") != gate.get("receipt_sha256")
        or gate.get("receipt_sha256") != _digest(gate)
        or gate.get("status") != "PASSED_IMMEDIATELY_BEFORE_CREATE"
        or gate.get("source_commit") != value.get("source_commit")
        or gate.get("server_release_receipt_sha256") != value.get("server_release_receipt_sha256")
        or gate.get("config_sha256") != config["config_sha256"]
        or gate.get("request_sha256") != value.get("request_sha256")
        or gate.get("jobs_api_title_matches") != 0
        or gate.get("jobs_api_run_dir_matches") != 0
        or gate.get("kubernetes_identity_matches") != 0
        or gate.get("sfs_run_dir_exists") is not False
        or gate.get("api_mutations") != 0
        or gate.get("scoring_authorized") is not False
        or value.get("route") != "POST /v1/runs"
        or value.get("http_status") != 202
        or value.get("server_instances_created") != 1
        or value.get("scored_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP8 submission receipt drifted")
    api_run_id = value.get("api_run_id")
    if not isinstance(api_run_id, str) or not api_run_id.startswith("ft-run-"):
        raise ValueError("early DP8 submission API identity drifted")


def validate_binding(
    value: Mapping[str, Any], submission: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    validate_submission(submission, root)
    if value.get("receipt_sha256") != _digest(value) or (
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
        or value.get("data_parallel_size") != 8
        or value.get("head_pod_running_ready") is not True
        or value.get("head_pod_restarts") != 0
        or value.get("kueue_preempted") is not False
        or value.get("scoring_authorized") is not False
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("early DP8 server binding drifted")
    origin = value.get("service_origin")
    if (
        not isinstance(origin, str)
        or not origin.startswith("http://")
        or not origin.endswith(":8000")
    ):
        raise ValueError("early DP8 service origin drifted")
    return parity.validate_server_binding(
        {
            "api_run_id": value["api_run_id"],
            "rayjob_uid": value["rayjob_uid"],
            "head_pod_uid": value["head_pod_uid"],
            "service_uid": value["service_uid"],
            "served_id": "qwen3.8-27b",
            "model_revision": value["model_revision"],
            "context_length": value["context_length"],
        },
        "qwen3.8-27b",
    )


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
        or value.get("job_name") != "chris-cyber-q38-dp8-c-qualifier-v2"
        or value.get("configmap_name") != "chris-cyber-q38-dp8-c-qualifier-v2"
        or value.get("output_root") != "/mnt/sfs/jobs/chris-cyber-q38-dp8-c-qualifier-v2"
        or value.get("serving_block") != early.SERVING_BLOCK
        or value.get("submission_receipt_sha256") != submission.get("receipt_sha256")
        or value.get("server_binding_receipt_sha256") != binding.get("receipt_sha256")
        or value.get("package_sha256") != package_digest
        or value.get("task_instance_session_verifier_scoring_calls") != 0
        or value.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("mounted early DP8 qualifier release drifted")


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
        raise ValueError("early DP8 qualification plan digest drifted")
    expected = qualification_plan(
        value.get("server_binding") or {}, str(value.get("service_origin") or ""), root
    )
    if dict(value) != expected:
        raise ValueError("early DP8 qualification plan drifted")
    parity.validate_server_binding(value["server_binding"], "qwen3.8-27b")


def _event(path: Path, binding_receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = _load(path)
    if value.get("receipt_sha256") != _digest(value) or (
        value.get("schema_version") != EVENT_SCHEMA
        or value.get("status") != "REAL_REQUEST_COUNTER_INCREASED"
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
        "request_counters_before_by_rank",
        "request_counters_after_by_rank",
        "request_deltas_by_rank",
        "gpu_memory_used_mib_by_rank",
        "gpu_utilization_percent_by_rank",
    ):
        row = value.get(field)
        if not isinstance(row, list) or len(row) != 8 or not all(type(item) is int for item in row):
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
        counters = baseline.get("request_counters_by_rank")
        if (
            baseline.get("receipt_sha256") == _digest(baseline)
            and baseline.get("schema_version") == metric_observer.BASELINE_SCHEMA
            and baseline.get("status") == "STABLE_BOUND_COUNTER_BASELINE"
            and baseline.get("server_run_dir") == early.RUN_DIR
            and baseline.get("api_run_id") == binding_receipt.get("api_run_id")
            and baseline.get("pod_name") == binding_receipt.get("head_pod_name")
            and baseline.get("pod_uid") == binding_receipt.get("head_pod_uid")
            and baseline.get("service_uid") == binding_receipt.get("service_uid")
            and baseline.get("server_binding_receipt_sha256")
            == binding_receipt.get("receipt_sha256")
            and isinstance(counters, list)
            and len(counters) == 8
            and all(type(item) is int and item >= 0 for item in counters)
            and state == {"request_counters_by_rank": counters}
            and baseline.get("request_delta_since_prior_sample") == 0
            and baseline.get("traffic_refresh_performed") is False
            and baseline.get("prompts_traces_flags_or_scores_included") is False
        ):
            return baseline
        time.sleep(1)
    raise RuntimeError("stable post-binding request-counter baseline unavailable")


def _dind_resource_sample() -> dict[str, int]:
    try:
        lines = [line.split() for line in DIND_RESOURCE_SAMPLES_PATH.read_text().splitlines()]
        raw = lines[-1]
        values = [int(item) for item in raw]
    except (FileNotFoundError, IndexError, ValueError) as exc:
        raise RuntimeError("dind resource observer unavailable") from exc
    if len(values) != 4 or any(item < 0 for item in values):
        raise RuntimeError("dind resource observer shape drifted")
    return dict(
        zip(
            ("observed_at_epoch", "oom_kill", "nr_throttled", "throttled_usec"),
            values,
            strict=True,
        )
    )


def _dind_resource_control(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, Any]:
    deltas = {
        key: after[key] - before[key] for key in ("oom_kill", "nr_throttled", "throttled_usec")
    }
    value = {
        "schema_version": "fleet-qwen38-dp8-qualifier-controller-resource-v1",
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
    if after["observed_at_epoch"] < before["observed_at_epoch"] or any(
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
    resource_before = _dind_resource_sample()
    rows = execute()
    deadline = time.monotonic() + 30
    events: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        events = _new_events(before_paths, binding_receipt)
        active = {
            rank
            for event in events
            for rank, delta in enumerate(event["request_deltas_by_rank"])
            if delta > 0
        }
        if len(active) >= level:
            break
        time.sleep(1)
    if not events:
        raise RuntimeError("server-local observer produced no real-traffic event")
    first, last = events[0], events[-1]
    request_delta = [
        last["request_counters_after_by_rank"][rank]
        - first["request_counters_before_by_rank"][rank]
        for rank in range(8)
    ]
    gpu_peaks = [
        max(event["gpu_utilization_percent_by_rank"][rank] for event in events) for rank in range(8)
    ]
    gpu_memory = [
        max(event["gpu_memory_used_mib_by_rank"][rank] for event in events) for rank in range(8)
    ]
    resource_after = _dind_resource_sample()
    resource_control = _dind_resource_control(resource_before, resource_after)
    if resource_control["status"] != "PASSED_NO_CONTROLLER_RESOURCE_ERROR":
        raise RuntimeError("qualifier controller resource control failed")
    receipt = {
        "schema_version": core.DISTRIBUTION_SCHEMA,
        "status": "PASSED_NON_SCORED_DISTRIBUTION",
        "plan_receipt_sha256": plan["receipt_sha256"],
        "server_binding": plan["server_binding"],
        "service_origin": plan["service_origin"],
        "concurrency": level,
        "stream_receipt_sha256s": [row.get("receipt_sha256") for row in rows],
        "request_counters_before_by_rank": first["request_counters_before_by_rank"],
        "request_counters_after_by_rank": last["request_counters_after_by_rank"],
        "request_deltas_by_rank": request_delta,
        "gpu_peak_utilization_percent_by_rank": gpu_peaks,
        "gpu_peak_memory_used_mib_by_rank": gpu_memory,
        "gpu_device_count": 8,
        "gpu_memory_loaded_count": sum(item > 0 for item in gpu_memory),
        "sampling_seconds": max(0, last["observed_at_epoch"] - first["observed_at_epoch"]),
        "observer_errors": [],
        "task_instance_session_verifier_scoring_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    receipt["receipt_sha256"] = _digest(receipt)
    return rows, receipt, resource_control, baseline


def run(plan: Mapping[str, Any], binding_receipt: Mapping[str, Any], root: Path) -> dict[str, Any]:
    validate_plan(plan, root)

    def safe_probe() -> dict[str, Any]:
        try:
            return parity.run(
                "qwen3.8-27b",
                "",
                upstream_origin=plan["service_origin"],
                server_binding=plan["server_binding"],
                cluster_dind=True,
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
        try:
            rows, distribution, resource_control, baseline = observe_wave(
                level, execute, plan, binding_receipt
            )
            streams_ok = all(_valid_stream(row, plan) for row in rows)
            core.validate_distribution_receipt(distribution, plan, level, rows)
            observed_model_requests = sum(
                int((row.get("execution") or {}).get("model_requests") or 0) for row in rows
            )
            observed_server_requests = sum(distribution["request_deltas_by_rank"])
            passed = (
                len(rows) == level
                and streams_ok
                and observed_model_requests > 0
                and observed_server_requests == observed_model_requests
            )
        except (RuntimeError, ValueError):
            rows, distribution, resource_control, baseline, passed = [], {}, {}, {}, False
            observed_model_requests = 0
            observed_server_requests = 0
        elapsed_milliseconds = int((time.monotonic() - started) * 1000)
        timeout_budget_milliseconds = parity.TIMEOUT_SECONDS * 1000
        observations.append(
            {
                "concurrency": level,
                "status": "PASSED" if passed else "FAILED",
                "completed_stream_count": len(rows) if passed else 0,
                "observed_model_request_count": observed_model_requests,
                "observed_server_request_delta": observed_server_requests,
                "elapsed_milliseconds": elapsed_milliseconds,
                "timeout_budget_milliseconds": timeout_budget_milliseconds,
                "latency_headroom_milliseconds": (
                    timeout_budget_milliseconds - elapsed_milliseconds
                ),
                "error_count": 0 if passed else 1,
                "tool_order_exact": passed,
                "tool_arguments_exact": passed,
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
        raise ValueError("early DP8 qualification result drifted")
    levels = value.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("early DP8 qualification result is empty")
    seen_failure = False
    failure_count = 0
    highest = 0
    for index, row in enumerate(levels):
        if not isinstance(row, dict) or row.get("concurrency") != LEVELS[index]:
            raise ValueError("early DP8 qualification ladder order drifted")
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
            raise ValueError("early DP8 controller resource control drifted")
        if row.get("status") == "PASSED":
            baseline = row.get("stable_counter_baseline_receipt")
            distribution = row.get("distribution_receipt")
            streams = row.get("stream_receipts")
            model_requests = sum(
                int((stream.get("execution") or {}).get("model_requests") or 0)
                for stream in streams
                if isinstance(stream, dict)
            ) if isinstance(streams, list) else -1
            server_requests = (
                sum(distribution.get("request_deltas_by_rank") or [])
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
                or row["latency_headroom_milliseconds"] <= 0
            ):
                raise ValueError("early DP8 observed request/latency evidence drifted")
        if row.get("status") == "FAILED":
            seen_failure = True
            failure_count += 1
        elif row.get("status") == "PASSED" and not seen_failure:
            highest = row["concurrency"]
        else:
            raise ValueError("early DP8 qualification violates stop-on-failure")
    if seen_failure and (failure_count != 1 or levels[-1].get("status") != "FAILED"):
        raise ValueError("early DP8 qualification continued after failure")
    if value.get("highest_passing_concurrency") != highest:
        raise ValueError("early DP8 qualification ceiling drifted")
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
    submission = _load(args.submission)
    binding_receipt = _load(args.server_binding)
    release = _load(args.release)
    binding = validate_binding(binding_receipt, submission, root)
    validate_runtime_release(release, submission, binding_receipt, args.package)
    project_server_binding_once(binding_receipt)
    plan = qualification_plan(binding, binding_receipt["service_origin"], root)
    result = run(plan, binding_receipt, root)
    self_hosted.write_json_once(args.output, result)
    print(json.dumps({"highest_passing_concurrency": result["highest_passing_concurrency"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
