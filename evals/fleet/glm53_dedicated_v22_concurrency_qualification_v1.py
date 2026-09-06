"""Held score-free concurrency qualification for the UID-bound GLM v22 server."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from evals.fleet import endpoint_lease, self_hosted
from evals.fleet import opencode_actual_harness_parity_v1 as actual_harness

SCHEMA = "fleet-glm53-dedicated-v22-concurrency-qualification-held-v1"
JOB_NAME = "chris-glm53-dedicated-v22-concurrency-qualification-v1"
RESULT_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
BINDING = Path("docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v22-server-binding.json")
PARITY = Path(
    "docs/evidence/glm53-study/2026-09-06-glm53-dedicated-v22-actual-opencode-parity.json"
)
ORIGIN = "http://ft-run-d2dff491-gfw5m-head-svc.fleet-train-jobs.svc.cluster.local:8000"
LEASE_ROOT = Path("/mnt/sfs/endpoint-leases/opencode11827-dedicated-v22-v1")
METRIC = re.compile(
    r'^sglang:num_requests_total\{model_name="glm-5\.3"\}\s+([0-9]+(?:\.[0-9]+)?)$', re.M
)
CONCURRENCY = (1, 2, 4)
STREAM_IDS = ("synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d")
GPU_OBSERVER_SCHEMA = "fleet-glm53-dedicated-v22-concurrency-gpu-wave-v1"
GPU_OBSERVER_WAIT_SECONDS = 300


class QualificationError(RuntimeError):
    """Stable content-free qualification failure."""


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise QualificationError("evidence_shape_invalid")
    return value


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def read_request_counter(origin: str) -> int:
    with urlopen(origin.rstrip("/") + "/metrics", timeout=15) as response:
        if response.status != 200:
            raise QualificationError("metrics_http_status")
        raw = response.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise QualificationError("metrics_response_too_large")
    match = METRIC.search(raw.decode("utf-8", errors="strict"))
    if match is None:
        raise QualificationError("request_counter_absent")
    return int(float(match.group(1)))


HarnessRunner = Callable[..., dict[str, Any]]


def run_wave(
    concurrency: int,
    *,
    origin: str,
    binding: dict[str, Any],
    runner: HarnessRunner = actual_harness.run,
    counter: Callable[[str], int] = read_request_counter,
) -> dict[str, Any]:
    if concurrency not in CONCURRENCY:
        raise QualificationError("concurrency_not_frozen")
    before = counter(origin)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []

    def one(stream_id: str) -> dict[str, Any]:
        stream_started = time.monotonic()
        receipt = runner(
            "glm-5.3",
            "",
            upstream_origin=origin,
            server_binding=binding,
            docker_add_host_gateway=True,
        )
        elapsed = time.monotonic() - stream_started
        if (
            receipt.get("status") != "PASSED_NON_SCORED"
            or receipt.get("endpoint", {}).get("server_binding") != binding
            or receipt.get("execution", {}).get("task_instance_session_verifier_scoring_calls") != 0
        ):
            raise QualificationError("actual_harness_stream_failed")
        return {
            "stream_id": stream_id,
            "elapsed_seconds": round(elapsed, 6),
            "receipt_sha256": receipt["receipt_sha256"],
            "model_requests": receipt["execution"]["model_requests"],
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, STREAM_IDS[index]) for index in range(concurrency)]
        for future in as_completed(futures):
            rows.append(future.result())
    elapsed = time.monotonic() - started
    after = counter(origin)
    rows.sort(key=lambda row: row["stream_id"])
    model_requests = sum(int(row["model_requests"]) for row in rows)
    if after - before != model_requests:
        raise QualificationError("exclusive_request_counter_delta_mismatch")
    latencies = [float(row["elapsed_seconds"]) for row in rows]
    return {
        "concurrency": concurrency,
        "streams": rows,
        "streams_succeeded": len(rows),
        "errors": 0,
        "retries": 0,
        "timeouts": 0,
        "elapsed_seconds": round(elapsed, 6),
        "throughput_streams_per_second": round(len(rows) / elapsed, 9),
        "stream_latency_seconds": {
            "median": round(statistics.median(latencies), 6),
            "p95": round(_p95(latencies), 6),
        },
        "request_counter_before": before,
        "request_counter_after": after,
        "request_counter_delta": after - before,
        "model_requests_observed": model_requests,
    }


def evaluate(waves: list[dict[str, Any]], gpu_observer: dict[str, Any]) -> dict[str, Any]:
    if [row.get("concurrency") for row in waves] != list(CONCURRENCY):
        raise QualificationError("wave_order_mismatch")
    samples = gpu_observer.get("waves")
    if not isinstance(samples, list) or [row.get("concurrency") for row in samples] != list(
        CONCURRENCY
    ):
        raise QualificationError("gpu_observer_wave_mismatch")
    failures: list[str] = []
    for wave, observed in zip(waves, samples, strict=True):
        concurrency = int(wave["concurrency"])
        if (
            wave.get("streams_succeeded") != concurrency
            or wave.get("errors") != 0
            or wave.get("retries") != 0
            or wave.get("timeouts") != 0
            or wave.get("request_counter_delta") != wave.get("model_requests_observed")
        ):
            failures.append(f"c{concurrency}_protocol_or_counter")
        try:
            validate_gpu_wave(observed, concurrency, gpu_observer.get("server"))
        except QualificationError:
            failures.append(f"c{concurrency}_gpu_or_identity")
    base_p95 = float(waves[0]["stream_latency_seconds"]["p95"])
    if base_p95 <= 0:
        failures.append("c1_latency_invalid")
    else:
        if float(waves[1]["stream_latency_seconds"]["p95"]) / base_p95 > 2.0:
            failures.append("c2_latency_ratio")
        if float(waves[2]["stream_latency_seconds"]["p95"]) / base_p95 > 2.0:
            failures.append("c4_latency_ratio")
    base_throughput = float(waves[0]["throughput_streams_per_second"])
    if base_throughput <= 0 or (
        float(waves[2]["throughput_streams_per_second"]) / base_throughput < 1.5
    ):
        failures.append("c4_throughput_ratio")
    return {
        "status": "PASSED_SCORE_FREE" if not failures else "FAILED",
        "failures": failures,
        "scored_concurrency_change_authorized": not failures,
        "qualified_concurrency_ceiling": max(CONCURRENCY) if not failures else 0,
    }


def validate_gpu_wave(
    observed: dict[str, Any], concurrency: int, expected_server: dict[str, Any] | None
) -> None:
    utilization = observed.get("max_utilization_percent_by_device")
    if (
        observed.get("schema_version") != GPU_OBSERVER_SCHEMA
        or observed.get("status") != "OBSERVED_SCORE_FREE_WAVE"
        or observed.get("concurrency") != concurrency
        or observed.get("server") != expected_server
        or observed.get("devices_seen") != 8
        or not isinstance(observed.get("samples_per_device"), int)
        or observed["samples_per_device"] < 1
        or not isinstance(utilization, list)
        or len(utilization) != 8
        or any(type(value) not in {int, float} or not 0 < value <= 100 for value in utilization)
        or observed.get("server_identity_unchanged") is not True
        or observed.get("receipt_sha256") != self_hosted.digest_without(observed, "receipt_sha256")
    ):
        raise QualificationError("gpu_wave_observer_invalid")


def validate_runtime_ramp(wave: dict[str, Any], *, baseline: dict[str, Any] | None = None) -> None:
    concurrency = int(wave.get("concurrency", 0))
    if (
        concurrency not in CONCURRENCY
        or wave.get("streams_succeeded") != concurrency
        or wave.get("errors") != 0
        or wave.get("retries") != 0
        or wave.get("timeouts") != 0
        or wave.get("request_counter_delta") != wave.get("model_requests_observed")
    ):
        raise QualificationError(f"c{concurrency}_runtime_gate_failed")
    if baseline is not None:
        baseline_p95 = float(baseline["stream_latency_seconds"]["p95"])
        if baseline_p95 <= 0 or float(wave["stream_latency_seconds"]["p95"]) / baseline_p95 > 2.0:
            raise QualificationError(f"c{concurrency}_latency_gate_failed")


def await_gpu_observer(root: Path, concurrency: int, server: dict[str, Any]) -> dict[str, Any]:
    path = root / f"c{concurrency}.json"
    deadline = time.monotonic() + GPU_OBSERVER_WAIT_SECONDS
    while time.monotonic() < deadline:
        if path.is_file() and not path.is_symlink():
            observed = load(path)
            validate_gpu_wave(observed, concurrency, server)
            return observed
        time.sleep(1)
    raise QualificationError(f"c{concurrency}_gpu_observer_timeout")


def write_once(path: Path, body: dict[str, Any]) -> None:
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    self_hosted.write_json_once(path, body)


def execute(
    root: Path,
    authorization_path: Path,
    out: Path,
    *,
    runner: HarnessRunner = actual_harness.run,
    counter: Callable[[str], int] = read_request_counter,
    gpu_observer_root: Path,
    observer: Callable[[Path, int, dict[str, Any]], dict[str, Any]] = await_gpu_observer,
) -> dict[str, Any]:
    plan = render(root)
    authorization = load(authorization_path)
    binding = load(root / BINDING)
    if (
        authorization.get("qualification_launch_authorized") is not True
        or authorization.get("server") != plan["server"]
        or authorization.get("no_active_scored_controller") is not True
        or authorization.get("receipt_sha256")
        != self_hosted.digest_without(authorization, "receipt_sha256")
    ):
        raise QualificationError("post_acceptance_authorization_invalid")
    if out.exists() or out.is_symlink():
        raise QualificationError("qualification_result_collision")
    with endpoint_lease.acquire_endpoint_lease(
        lease_root=LEASE_ROOT,
        endpoint_key=binding["api_run_id"],
        maximum_streams=1,
    ):
        waves: list[dict[str, Any]] = []
        gpu_waves: list[dict[str, Any]] = []
        for concurrency in CONCURRENCY:
            phase = {
                "schema_version": "fleet-glm53-dedicated-v22-concurrency-phase-v1",
                "status": "WAVE_STARTED_SCORE_FREE",
                "concurrency": concurrency,
                "server": plan["server"],
            }
            write_once(out.parent / f"PHASE-c{concurrency}.json", phase)
            wave = run_wave(
                concurrency,
                origin=ORIGIN,
                binding=binding,
                runner=runner,
                counter=counter,
            )
            validate_runtime_ramp(wave, baseline=waves[0] if waves else None)
            observed = observer(gpu_observer_root, concurrency, plan["server"])
            validate_gpu_wave(observed, concurrency, plan["server"])
            waves.append(wave)
            gpu_waves.append(observed)
    result = {
        "schema_version": "fleet-glm53-dedicated-v22-concurrency-raw-v1",
        "status": "COMPLETED_SCORE_FREE_WAVES",
        "plan_package_sha256": plan["package_sha256"],
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "server": plan["server"],
        "waves": waves,
        "gpu_waves": gpu_waves,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "response_or_tool_argument_content_persisted": False,
        "scored_concurrency_change_authorized": False,
    }
    write_once(out, result)
    return result


def render(root: Path) -> dict[str, Any]:
    binding = load(root / BINDING)
    parity = load(root / PARITY)
    actual_harness.validate_server_binding(binding, "glm-5.3")
    if (
        parity.get("receipt_sha256") != self_hosted.digest_without(parity, "receipt_sha256")
        or parity.get("status") != "PASSED_NON_SCORED"
        or parity.get("endpoint", {}).get("kind") != "dedicated_uid_bound_inference"
        or parity.get("endpoint", {}).get("server_binding") != binding
        or parity.get("execution", {}).get("task_instance_session_verifier_scoring_calls") != 0
    ):
        raise QualificationError("dedicated_parity_invalid")
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "READY_HELD",
        "launch_authorized": False,
        "job_name": JOB_NAME,
        "result_root": str(RESULT_ROOT),
        "server": {
            "api_run_id": binding["api_run_id"],
            "rayjob_uid": binding["rayjob_uid"],
            "head_pod_uid": binding["head_pod_uid"],
            "service_uid": binding["service_uid"],
            "origin": ORIGIN,
            "served_id": binding["served_id"],
            "model_revision": binding["model_revision"],
            "context_length": binding["context_length"],
            "parity_receipt_sha256": parity["receipt_sha256"],
        },
        "treatment": {
            "harness": parity["harness"]["name"],
            "harness_version": parity["harness"]["version"],
            "harness_image_id": parity["harness"]["image_id"],
            "provider_adapter": parity["harness"]["provider_adapter"],
            "context_management": parity["harness"]["context_management"],
            "compaction_headroom_tokens": parity["harness"]["compaction_headroom_tokens"],
            "context_window_size": parity["harness"]["context_window_size"],
            "max_output_tokens": parity["harness"]["max_output_tokens"],
            "tools": parity["tool_contract"]["names"],
            "tool_catalog_sha256": parity["tool_contract"]["mcp_catalog_sha256"],
            "automatic_retry": False,
        },
        "score_free_boundary": {
            "synthetic_distinct_workload_ids": [
                "synthetic-a",
                "synthetic-b",
                "synthetic-c",
                "synthetic-d",
            ],
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "persist_response_content": False,
            "persist_prompts_traces_flags_scores": False,
        },
        "phases": [
            {"name": "baseline", "concurrency": 1, "streams": ["synthetic-a"]},
            {"name": "ramp-2", "concurrency": 2, "streams": ["synthetic-a", "synthetic-b"]},
            {
                "name": "ramp-4",
                "concurrency": 4,
                "streams": ["synthetic-a", "synthetic-b", "synthetic-c", "synthetic-d"],
            },
        ],
        "measurements": {
            "per_request_http_success": True,
            "per_request_latency_seconds": True,
            "aggregate_throughput_requests_per_second": True,
            "server_request_counter_delta": True,
            "gpu_utilization_percent_per_device": True,
            "gpu_memory_mib_per_device": True,
            "response_or_tool_argument_content": False,
        },
        "fail_closed_ramp": {
            "ramp_2_requires": {
                "baseline_all_requests_valid": True,
                "http_or_harness_failures": 0,
                "timeout_or_retry_policy_changes": 0,
                "p95_latency_ratio_to_baseline_lte": 2.0,
                "all_eight_gpus_active_in_wave": True,
                "server_identity_unchanged": True,
            },
            "ramp_4_requires": {
                "ramp_2_all_requests_valid": True,
                "http_or_harness_failures": 0,
                "timeout_or_retry_policy_changes": 0,
                "p95_latency_ratio_to_baseline_lte": 2.0,
                "throughput_ratio_to_baseline_gte": 1.5,
                "all_eight_gpus_active_in_wave": True,
                "server_identity_unchanged": True,
            },
            "stop_on_first_failed_requirement": True,
            "gpu_observer_receipt_required_before_next_wave": True,
            "never_overlap_scored_controller": True,
        },
        "launch_prerequisites": {
            "rank51_attempt2_terminal_accepted": True,
            "no_active_scored_controller": True,
            "fresh_server_uid_and_workload_history": True,
            "fresh_result_root_absence": True,
            "fresh_endpoint_lease_exclusive": True,
            "actual_model_request_counter_watchdog_active": True,
        },
        "scored_concurrency_change_authorized": False,
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "validate"), nargs="?", default="plan")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--gpu-observer-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(render(args.root), sort_keys=True))
    elif args.command == "run":
        if args.authorization is None or args.out is None or args.gpu_observer_root is None:
            parser.error("run requires --authorization, --gpu-observer-root, and --out")
        execute(
            args.root,
            args.authorization,
            args.out,
            gpu_observer_root=args.gpu_observer_root,
        )
    else:
        if args.raw is None or args.out is None:
            parser.error("validate requires --raw and --out")
        raw = load(args.raw)
        observed = {"server": raw["server"], "waves": raw["gpu_waves"]}
        verdict = {
            "schema_version": "fleet-glm53-dedicated-v22-concurrency-qualified-v1",
            **evaluate(raw["waves"], observed),
            "raw_receipt_sha256": raw["receipt_sha256"],
            "gpu_observer_receipt_sha256s": [row["receipt_sha256"] for row in raw["gpu_waves"]],
            "privacy": False,
        }
        write_once(args.out, verdict)
        if verdict["status"] != "PASSED_SCORE_FREE":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
