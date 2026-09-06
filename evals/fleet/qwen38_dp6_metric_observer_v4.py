"""Observe one global request counter while retaining DP6 device evidence.

The installed SGLang build exposes ``sglang:num_requests_total`` as a global
counter partitioned by ``is_streaming``.  It does not attach ``dp_rank`` to
that family.  Six ``sglang:max_total_num_tokens`` samples do carry
``dp_rank`` and prove the scheduler topology.  Request count and rank activity
are therefore deliberately separate authorities: the global counter proves
request traffic, while a six-device ``nvidia-smi`` sample proves loaded and
active devices without inventing per-rank request attribution.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dp6_metric_observer_v1 as legacy
from evals.fleet import qwen38_dp6_metric_observer_v2 as prior
from evals.fleet import qwen38_dp6_metric_observer_v3 as schema

RANKS = 6
REQUEST_FAMILY = prior.REQUEST_FAMILY
STARTUP_ANCHOR_FAMILY = prior.STARTUP_ANCHOR_FAMILY
BASELINE_SCHEMA = "fleet-qwen38-dp6-global-request-baseline-v1"
EVENT_SCHEMA = "fleet-qwen38-dp6-global-request-traffic-observation-v1"
STATE_KEY = "global_request_total"
REQUEST_LABEL_KEYS = {"engine_type", "is_streaming", "model_name"}


def _global_request_total(metrics: str) -> int:
    """Return the monotonic aggregate after proving the exact live schema."""
    anchors: dict[int, int] = {}
    request_series: set[tuple[tuple[str, str], ...]] = set()
    request_context: set[tuple[str, str]] = set()
    total = 0
    for family, labels, value in prior._samples(metrics):  # noqa: SLF001
        if family == STARTUP_ANCHOR_FAMILY:
            rank = prior._rank(labels)  # noqa: SLF001
            if rank is None or rank in anchors or value <= 0:
                raise ValueError("startup anchor does not prove one positive sample per rank")
            anchors[rank] = value
            continue
        if family != REQUEST_FAMILY:
            continue
        if set(labels) != REQUEST_LABEL_KEYS:
            raise ValueError("global request-counter label schema drifted")
        if labels["is_streaming"] not in {"true", "false"}:
            raise ValueError("global request-counter streaming partition drifted")
        identity = tuple(sorted(labels.items()))
        if identity in request_series:
            raise ValueError("global request-counter series was duplicated")
        request_series.add(identity)
        request_context.add((labels["engine_type"], labels["model_name"]))
        total += value
    if set(anchors) != set(range(RANKS)):
        raise ValueError("startup anchor did not prove all six scheduler ranks")
    if not request_series or len(request_context) != 1:
        raise ValueError("global request-counter authority is absent or ambiguous")
    return total


def wait_for_global_request_total(
    fetch_metrics: Any,
    schema_path: Path,
    *,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> int:
    """Wait only for metric shape convergence; never refresh traffic state."""
    deadline = monotonic() + schema.METRIC_SCHEMA_WARMUP_SECONDS
    attempt = 0
    while True:
        attempt += 1
        metrics = fetch_metrics()
        try:
            total = _global_request_total(metrics)
        except ValueError as exc:
            schema._write_schema(  # noqa: SLF001 - same immutable observer package
                schema_path,
                metrics,
                attempt=attempt,
                validation_status="INCOMPLETE_DURING_BOUNDED_WARMUP",
            )
            if monotonic() >= deadline:
                raise schema.MetricSchemaWarmupTimeout(
                    "global request metric schema did not converge before the deadline"
                ) from exc
            sleep(schema.METRIC_SCHEMA_POLL_SECONDS)
            continue
        schema._write_schema(  # noqa: SLF001 - same immutable observer package
            schema_path, metrics, attempt=attempt, validation_status="CONVERGED_GLOBAL_COUNTER"
        )
        return total


def _validate_identity(
    *,
    server_run_dir: str,
    pod_name: str,
    pod_uid: str,
    api_run_id: str,
    service_uid: str,
    server_binding_receipt_sha256: str,
) -> None:
    # Reuse the exact route/UID validation without emitting a traffic event.
    if (
        legacy.observation(
            [0] * RANKS,
            [0] * RANKS,
            memory_mib=[0] * RANKS,
            utilization_percent=[0] * RANKS,
            server_run_dir=server_run_dir,
            pod_name=pod_name,
            pod_uid=pod_uid,
            api_run_id=api_run_id,
            service_uid=service_uid,
            server_binding_receipt_sha256=server_binding_receipt_sha256,
            observed_at_epoch=0,
        )
        is not None
    ):
        raise AssertionError("identity validation unexpectedly emitted traffic")


def baseline_observation(
    total: int,
    *,
    server_run_dir: str,
    pod_name: str,
    pod_uid: str,
    api_run_id: str,
    service_uid: str,
    server_binding_receipt_sha256: str,
    observed_at_epoch: int,
) -> dict[str, Any]:
    if type(total) is not int or total < 0:
        raise ValueError("global request baseline drifted")
    _validate_identity(
        server_run_dir=server_run_dir,
        pod_name=pod_name,
        pod_uid=pod_uid,
        api_run_id=api_run_id,
        service_uid=service_uid,
        server_binding_receipt_sha256=server_binding_receipt_sha256,
    )
    value: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA,
        "status": "STABLE_BOUND_GLOBAL_REQUEST_BASELINE",
        "server_run_dir": server_run_dir,
        "api_run_id": api_run_id,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
        "service_uid": service_uid,
        "server_binding_receipt_sha256": server_binding_receipt_sha256,
        "observed_at_epoch": observed_at_epoch,
        STATE_KEY: total,
        "request_delta_since_prior_sample": 0,
        "traffic_refresh_performed": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = schema._digest(value)  # noqa: SLF001
    return value


def traffic_observation(
    before: int,
    after: int,
    *,
    memory_mib: list[int],
    utilization_percent: list[int],
    server_run_dir: str,
    pod_name: str,
    pod_uid: str,
    api_run_id: str,
    service_uid: str,
    server_binding_receipt_sha256: str,
    observed_at_epoch: int,
) -> dict[str, Any] | None:
    if any(type(value) is not int or value < 0 for value in (before, after)):
        raise ValueError("global request counter drifted")
    if after < before:
        raise ValueError("global request counter decreased")
    if after == before:
        return None
    if (
        len(memory_mib) != RANKS
        or len(utilization_percent) != RANKS
        or any(type(value) is not int or value <= 0 for value in memory_mib)
        or any(type(value) is not int or not 0 <= value <= 100 for value in utilization_percent)
    ):
        raise ValueError("six-device GPU evidence drifted")
    _validate_identity(
        server_run_dir=server_run_dir,
        pod_name=pod_name,
        pod_uid=pod_uid,
        api_run_id=api_run_id,
        service_uid=service_uid,
        server_binding_receipt_sha256=server_binding_receipt_sha256,
    )
    value: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA,
        "status": "REAL_GLOBAL_REQUEST_COUNTER_INCREASED",
        "server_run_dir": server_run_dir,
        "api_run_id": api_run_id,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
        "service_uid": service_uid,
        "server_binding_receipt_sha256": server_binding_receipt_sha256,
        "observed_at_epoch": observed_at_epoch,
        "global_request_total_before": before,
        "global_request_total_after": after,
        "global_request_delta": after - before,
        "gpu_memory_used_mib_by_device": memory_mib,
        "gpu_utilization_percent_by_device": utilization_percent,
        "per_rank_request_attribution_claimed": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = schema._digest(value)  # noqa: SLF001
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("--traffic-path", type=Path, required=True)
    parser.add_argument("--binding-path", type=Path, required=True)
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--event-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--server-run-dir", required=True)
    args = parser.parse_args()
    schema_path = Path(args.server_run_dir) / "lifecycle/METRIC-SCHEMA.json"
    phase = "binding"
    try:
        binding = legacy.server_binding(
            args.binding_path, args.server_run_dir, socket.gethostname()
        )
        phase = "metric_schema_warmup"

        def fetch_metrics() -> str:
            with urllib.request.urlopen(args.metrics_url, timeout=3) as response:
                return response.read().decode("utf-8")

        after = wait_for_global_request_total(fetch_metrics, schema_path)
        before: int | None = None
        if args.state_path.is_file() and not args.state_path.is_symlink():
            previous = json.loads(args.state_path.read_text())
            candidate = previous.get(STATE_KEY) if isinstance(previous, dict) else None
            if type(candidate) is int and candidate >= 0:
                before = candidate
        phase = "counter_state"
        if before is not None and after < before:
            raise ValueError("global request counter decreased")
        legacy._atomic_json(args.state_path, {STATE_KEY: after})  # noqa: SLF001
        if before is None:
            prior._write_status(args.status_path, "COUNTER_STATE_INITIALIZED", phase)  # noqa: SLF001
            return
        if after == before:
            phase = "stable_baseline"
            baseline = baseline_observation(
                after,
                server_run_dir=args.server_run_dir,
                pod_name=socket.gethostname(),
                pod_uid=binding["head_pod_uid"],
                api_run_id=binding["api_run_id"],
                service_uid=binding["service_uid"],
                server_binding_receipt_sha256=binding["receipt_sha256"],
                observed_at_epoch=int(time.time()),
            )
            legacy._atomic_json(args.baseline_path, baseline)  # noqa: SLF001
            prior._write_status(  # noqa: SLF001
                args.status_path, "STABLE_BOUND_COUNTER_BASELINE", phase
            )
            return
        phase = "gpu_sample"
        raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        memory, utilization = legacy.gpu_sample(raw)
        phase = "traffic_receipt"
        now = int(time.time())
        receipt = traffic_observation(
            before,
            after,
            memory_mib=memory,
            utilization_percent=utilization,
            server_run_dir=args.server_run_dir,
            pod_name=socket.gethostname(),
            pod_uid=binding["head_pod_uid"],
            api_run_id=binding["api_run_id"],
            service_uid=binding["service_uid"],
            server_binding_receipt_sha256=binding["receipt_sha256"],
            observed_at_epoch=now,
        )
        if receipt is None:
            raise AssertionError("positive global delta did not produce traffic evidence")
        event = args.event_dir / f"{now}-{receipt['receipt_sha256'][7:23]}.json"
        legacy._create_json_once(event, receipt)  # noqa: SLF001
        legacy._atomic_json(args.receipt_path, receipt)  # noqa: SLF001
        args.traffic_path.touch(exist_ok=True)
        os.utime(args.traffic_path, (now, now))
        prior._write_status(args.status_path, "REAL_REQUEST_COUNTER_INCREASED", phase)  # noqa: SLF001
    except Exception as exc:
        prior._write_status(args.status_path, "FAILED", phase, exc)  # noqa: SLF001
        raise


if __name__ == "__main__":
    main()
