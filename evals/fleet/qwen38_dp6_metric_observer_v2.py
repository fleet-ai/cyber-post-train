"""Content-free Qwen DP6 observer with a zero-request startup baseline."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dp6_metric_observer_v1 as legacy

RANKS = 6
BASELINE_SCHEMA = legacy.BASELINE_SCHEMA
REQUEST_FAMILY = "sglang:num_requests_total"
STARTUP_ANCHOR_FAMILY = "sglang:max_total_num_tokens"
STATUS_SCHEMA = "fleet-qwen38-dp6-metric-observer-status-v2"


def _rank(labels: Mapping[str, str]) -> int | None:
    text = labels.get("dp_rank") or labels.get("data_parallel_rank")
    if text is None:
        return None
    if not text.isdigit() or int(text) not in range(RANKS):
        raise ValueError("metric rank is outside the DP6 treatment")
    return int(text)


def _samples(metrics: str) -> list[tuple[str, dict[str, str], int]]:
    rows: list[tuple[str, dict[str, str], int]] = []
    for raw in metrics.splitlines():
        match = legacy.METRIC_RE.match(raw.strip())
        if match is None:
            continue
        head = match.group("head")
        if head not in {REQUEST_FAMILY, STARTUP_ANCHOR_FAMILY}:
            continue
        labels = dict(legacy.LABEL_RE.findall(match.group("labels") or ""))
        value = float(match.group("value"))
        if value < 0 or not value.is_integer():
            raise ValueError("metric counter or anchor is not a non-negative integer")
        rows.append((head, labels, int(value)))
    return rows


def request_counters(metrics: str) -> list[int]:
    """Return per-rank totals, using an eager startup gauge to prove zero ranks.

    Prometheus counters do not expose a labeled series until that label set has
    first been incremented.  A fresh server therefore has no request-counter
    samples.  The exact SGLang startup gauge is emitted once per scheduler; its
    complete DP6 rank set proves that absent request series are genuine zeros.
    Request counters are partitioned by labels such as ``is_streaming`` and
    must be summed rather than overwritten.
    """
    anchors: dict[int, int] = {}
    totals = [0] * RANKS
    seen_request_series: set[tuple[tuple[str, str], ...]] = set()
    for family, labels, value in _samples(metrics):
        rank = _rank(labels)
        if family == STARTUP_ANCHOR_FAMILY:
            if rank is None or rank in anchors or value <= 0:
                raise ValueError("startup anchor does not identify one positive sample per rank")
            anchors[rank] = value
            continue
        if rank is None:
            raise ValueError("request counter omitted the DP rank")
        identity = tuple(sorted(labels.items()))
        if identity in seen_request_series:
            raise ValueError("request counter series was duplicated")
        seen_request_series.add(identity)
        totals[rank] += value
    if set(anchors) != set(range(RANKS)):
        raise ValueError("startup anchor did not prove all six scheduler ranks")
    return totals


def observer_status(
    status: str,
    phase: str,
    *,
    observed_at_epoch: int,
    error_type: str | None = None,
) -> dict[str, Any]:
    allowed = {
        "COUNTER_STATE_INITIALIZED",
        "STABLE_BOUND_COUNTER_BASELINE",
        "REAL_REQUEST_COUNTER_INCREASED",
        "FAILED",
    }
    if status not in allowed or not phase or observed_at_epoch < 0:
        raise ValueError("observer status shape drifted")
    if (status == "FAILED") != (error_type is not None):
        raise ValueError("observer failure classification drifted")
    value: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA,
        "status": status,
        "phase": phase,
        "observed_at_epoch": observed_at_epoch,
        "error_type": error_type,
        "request_or_response_bodies_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = (
        "sha256:"
        + __import__("hashlib")
        .sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        .hexdigest()
    )
    return value


def _write_status(path: Path, status: str, phase: str, error: Exception | None = None) -> None:
    legacy._atomic_json(  # noqa: SLF001 - same immutable observer package
        path,
        observer_status(
            status,
            phase,
            observed_at_epoch=int(time.time()),
            error_type=type(error).__name__ if error is not None else None,
        ),
    )


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
    phase = "binding"
    try:
        binding = legacy.server_binding(
            args.binding_path, args.server_run_dir, socket.gethostname()
        )
        phase = "metrics_fetch"
        with urllib.request.urlopen(args.metrics_url, timeout=3) as response:
            metrics = response.read().decode("utf-8")
        phase = "metric_schema"
        after = request_counters(metrics)
        before: list[int] | None = None
        if args.state_path.is_file() and not args.state_path.is_symlink():
            previous = json.loads(args.state_path.read_text())
            candidate = (
                previous.get("request_counters_by_rank") if isinstance(previous, dict) else None
            )
            if (
                isinstance(candidate, list)
                and len(candidate) == RANKS
                and all(type(value) is int and value >= 0 for value in candidate)
            ):
                before = candidate
        phase = "counter_state"
        if before is not None and any(after[index] < before[index] for index in range(RANKS)):
            raise ValueError("request counters decreased")
        legacy._atomic_json(  # noqa: SLF001 - same immutable observer package
            args.state_path, {"request_counters_by_rank": after}
        )
        if before is None:
            _write_status(args.status_path, "COUNTER_STATE_INITIALIZED", phase)
            return
        if after == before:
            phase = "stable_baseline"
            baseline = legacy.baseline_observation(
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
            _write_status(args.status_path, "STABLE_BOUND_COUNTER_BASELINE", phase)
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
        receipt = legacy.observation(
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
            raise AssertionError("positive counter delta did not produce traffic evidence")
        event = args.event_dir / f"{now}-{receipt['receipt_sha256'][7:23]}.json"
        legacy._create_json_once(event, receipt)  # noqa: SLF001
        legacy._atomic_json(args.receipt_path, receipt)  # noqa: SLF001
        args.traffic_path.touch(exist_ok=True)
        os.utime(args.traffic_path, (now, now))
        _write_status(args.status_path, "REAL_REQUEST_COUNTER_INCREASED", phase)
    except Exception as exc:
        _write_status(args.status_path, "FAILED", phase, exc)
        raise


if __name__ == "__main__":
    main()
