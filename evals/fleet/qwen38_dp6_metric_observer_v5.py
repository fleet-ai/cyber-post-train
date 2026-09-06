"""Observe exact global, active-request, topology, and DP6 device evidence.

The installed SGLang build exposes ``sglang:num_requests_total`` as a global
counter partitioned by ``is_streaming``. It only increments when a request
finishes, so it cannot by itself protect a long in-flight request from the idle
rail. Exact ``num_running_reqs`` and ``num_queue_reqs`` gauges plus six-device
GPU activity therefore protect active work. Six startup anchors prove the
scheduler topology. None of these separate authorities is misrepresented as
per-rank request attribution.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dp6_metric_observer_v1 as legacy
from evals.fleet import qwen38_dp6_metric_observer_v2 as prior
from evals.fleet import qwen38_dp6_metric_observer_v3 as schema

RANKS = 6
REQUEST_FAMILY = prior.REQUEST_FAMILY
STARTUP_ANCHOR_FAMILY = prior.STARTUP_ANCHOR_FAMILY
RUNNING_FAMILY = "sglang:num_running_reqs"
QUEUE_FAMILY = "sglang:num_queue_reqs"
TARGET_FAMILIES = {
    REQUEST_FAMILY,
    STARTUP_ANCHOR_FAMILY,
    RUNNING_FAMILY,
    QUEUE_FAMILY,
}
BASELINE_SCHEMA = "fleet-qwen38-dp6-immutable-global-request-baseline-v2"
EVENT_SCHEMA = "fleet-qwen38-dp6-request-or-gpu-activity-observation-v1"
STATUS_SCHEMA = "fleet-qwen38-dp6-activity-observer-status-v1"
SCHEMA_OBSERVATION = "fleet-qwen38-dp6-activity-metric-schema-observation-v1"
STATE_KEY = "global_request_total"
REQUEST_LABEL_KEYS = {"engine_type", "is_streaming", "model_name"}
RANK_LABEL_KEYS = {
    "dp_rank",
    "engine_type",
    "model_name",
    "moe_ep_rank",
    "pp_rank",
    "tp_rank",
}
EXPECTED_ENGINE = "unified"
EXPECTED_MODEL = "qwen3.8-27b"
LABEL_BLOCK_RE = re.compile(r'^(?:\w+="[^"]*")(?:,\w+="[^"]*")*$')


def _is_target_line(raw: str) -> bool:
    return any(
        raw == family or raw.startswith((family + "{", family + " ")) for family in TARGET_FAMILIES
    )


def _target_samples(metrics: str) -> list[tuple[str, dict[str, str], int]]:
    """Parse every exact target-family line or fail closed on malformed input."""
    rows: list[tuple[str, dict[str, str], int]] = []
    for raw in metrics.splitlines():
        line = raw.strip()
        if not _is_target_line(line):
            continue
        match = legacy.METRIC_RE.match(line)
        if match is None or match.group("head") not in TARGET_FAMILIES:
            raise ValueError("target metric line is malformed")
        label_block = match.group("labels") or ""
        if LABEL_BLOCK_RE.fullmatch(label_block) is None:
            raise ValueError("target metric label block is malformed")
        pairs = legacy.LABEL_RE.findall(label_block)
        labels = dict(pairs)
        if len(pairs) != len(labels):
            raise ValueError("target metric contains a duplicate label")
        value = float(match.group("value"))
        if value < 0 or not value.is_integer():
            raise ValueError("target metric is not a non-negative integer")
        rows.append((match.group("head"), labels, int(value)))
    return rows


def schema_observation(
    metrics: str,
    *,
    observed_at_epoch: int,
    warmup_attempt: int,
    validation_status: str,
) -> dict[str, Any]:
    """Persist only target-family shape, including malformed raw-line counts."""
    families: dict[str, dict[str, Any]] = {}
    for family in sorted(TARGET_FAMILIES):
        raw_lines = [line.strip() for line in metrics.splitlines() if _is_target_line(line.strip())]
        family_raw = [line for line in raw_lines if line.startswith((family + "{", family + " "))]
        parsed: list[dict[str, str]] = []
        for line in family_raw:
            match = legacy.METRIC_RE.match(line)
            if match is None or match.group("head") != family:
                continue
            pairs = legacy.LABEL_RE.findall(match.group("labels") or "")
            if len(pairs) != len(dict(pairs)):
                continue
            parsed.append(dict(pairs))
        families[family] = {
            "raw_family_line_count": len(family_raw),
            "parsed_sample_count": len(parsed),
            "label_key_sets": [
                list(keys) for keys in sorted({tuple(sorted(labels)) for labels in parsed})
            ],
        }
    value: dict[str, Any] = {
        "schema_version": SCHEMA_OBSERVATION,
        "status": validation_status,
        "observed_at_epoch": observed_at_epoch,
        "warmup_attempt": warmup_attempt,
        "expected_data_parallel_ranks": RANKS,
        "target_families": families,
        "metric_values_included": False,
        "request_or_response_bodies_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = schema._digest(value)  # noqa: SLF001
    return value


def _write_schema(path: Path, metrics: str, *, attempt: int, status: str) -> None:
    legacy._atomic_json(  # noqa: SLF001
        path,
        schema_observation(
            metrics,
            observed_at_epoch=int(time.time()),
            warmup_attempt=attempt,
            validation_status=status,
        ),
    )


def observer_status(
    status: str,
    phase: str,
    *,
    observed_at_epoch: int,
    failure_code: str | None = None,
) -> dict[str, Any]:
    allowed = {
        "STABLE_BOUND_COUNTER_BASELINE",
        "REAL_REQUEST_OR_GPU_ACTIVITY_OBSERVED",
        "FAILED",
    }
    if (
        status not in allowed
        or not phase
        or (status == "FAILED") != (failure_code == "observer_failed")
    ):
        raise ValueError("activity observer status drifted")
    value: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA,
        "status": status,
        "phase": phase,
        "observed_at_epoch": observed_at_epoch,
        "failure_code": failure_code,
        "request_or_response_bodies_included": False,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = schema._digest(value)  # noqa: SLF001
    return value


def _write_status(path: Path, status: str, phase: str, failed: bool = False) -> None:
    legacy._atomic_json(  # noqa: SLF001
        path,
        observer_status(
            status,
            phase,
            observed_at_epoch=int(time.time()),
            failure_code="observer_failed" if failed else None,
        ),
    )


def _validate_common(labels: Mapping[str, str], expected: set[str]) -> None:
    if (
        set(labels) != expected
        or labels.get("engine_type") != EXPECTED_ENGINE
        or labels.get("model_name") != EXPECTED_MODEL
    ):
        raise ValueError("target metric engine, model, or label schema drifted")


def metric_snapshot(metrics: str) -> tuple[int, int, int]:
    """Return completed, running, and queued request totals for exact live identity."""
    anchors: dict[int, int] = {}
    active: dict[str, dict[int, int]] = {RUNNING_FAMILY: {}, QUEUE_FAMILY: {}}
    request_series: set[tuple[tuple[str, str], ...]] = set()
    total = 0
    for family, labels, value in _target_samples(metrics):
        if family == REQUEST_FAMILY:
            _validate_common(labels, REQUEST_LABEL_KEYS)
            if labels["is_streaming"] not in {"true", "false"}:
                raise ValueError("global request-counter streaming partition drifted")
            identity = tuple(sorted(labels.items()))
            if identity in request_series:
                raise ValueError("global request-counter series was duplicated")
            request_series.add(identity)
            total += value
            continue
        _validate_common(labels, RANK_LABEL_KEYS)
        if any(labels[key] != "0" for key in ("moe_ep_rank", "pp_rank", "tp_rank")):
            raise ValueError("target metric parallel-rank labels drifted")
        rank = prior._rank(labels)  # noqa: SLF001
        if rank is None:
            raise ValueError("ranked target metric omitted dp_rank")
        destination = anchors if family == STARTUP_ANCHOR_FAMILY else active[family]
        if rank in destination:
            raise ValueError("ranked target metric series was duplicated")
        if family == STARTUP_ANCHOR_FAMILY and value <= 0:
            raise ValueError("startup anchor is not positive")
        destination[rank] = value
    if set(anchors) != set(range(RANKS)):
        raise ValueError("startup anchor did not prove all six scheduler ranks")
    if not request_series:
        raise ValueError("global request-counter authority is absent")
    if any(set(values) != set(range(RANKS)) for values in active.values()):
        raise ValueError("active-request gauges did not prove all six scheduler ranks")
    return total, sum(active[RUNNING_FAMILY].values()), sum(active[QUEUE_FAMILY].values())


def _global_request_total(metrics: str) -> int:
    """Compatibility wrapper around the complete exact activity snapshot."""
    return metric_snapshot(metrics)[0]


def wait_for_metric_snapshot(
    fetch_metrics: Any,
    schema_path: Path,
    *,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> tuple[int, int, int]:
    """Wait only for metric shape convergence; never refresh traffic state."""
    deadline = monotonic() + schema.METRIC_SCHEMA_WARMUP_SECONDS
    attempt = 0
    while True:
        attempt += 1
        metrics = fetch_metrics()
        try:
            snapshot = metric_snapshot(metrics)
        except ValueError as exc:
            _write_schema(
                schema_path, metrics, attempt=attempt, status="INCOMPLETE_DURING_BOUNDED_WARMUP"
            )
            if monotonic() >= deadline:
                raise schema.MetricSchemaWarmupTimeout(
                    "global request metric schema did not converge before the deadline"
                ) from exc
            sleep(schema.METRIC_SCHEMA_POLL_SECONDS)
            continue
        _write_schema(schema_path, metrics, attempt=attempt, status="CONVERGED_ACTIVITY_SCHEMA")
        return snapshot


def wait_for_global_request_total(
    fetch_metrics: Any,
    schema_path: Path,
    *,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> int:
    """Compatibility wrapper for callers that need only the completed total."""
    return wait_for_metric_snapshot(fetch_metrics, schema_path, monotonic=monotonic, sleep=sleep)[0]


def idle_release_eligible(
    before: int,
    after: int,
    *,
    running_requests: int,
    queued_requests: int,
    utilization_percent: list[int],
    activity_age_seconds: int,
    maximum_idle_seconds: int = 600,
) -> bool:
    """Release only after every request and GPU activity authority is idle."""
    if after < before:
        raise ValueError("global request counter decreased")
    if any(type(value) is not int or value < 0 for value in (running_requests, queued_requests)):
        raise ValueError("active request gauge drifted")
    if len(utilization_percent) != RANKS or any(
        type(value) is not int or not 0 <= value <= 100 for value in utilization_percent
    ):
        raise ValueError("GPU utilization evidence drifted")
    return (
        after == before
        and running_requests == 0
        and queued_requests == 0
        and not any(utilization_percent)
        and activity_age_seconds >= maximum_idle_seconds
    )


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


def preserve_initial_baseline(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    """Create the sole baseline once; later idle samples must not rewrite it."""
    try:
        legacy._create_json_once(path, value)  # noqa: SLF001
        return value
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise ValueError("immutable baseline path drifted") from None
        existing = json.loads(path.read_text())
        if not isinstance(existing, dict):
            raise ValueError("immutable baseline receipt drifted") from None
        expected_identity = {
            "schema_version": BASELINE_SCHEMA,
            "status": "STABLE_BOUND_GLOBAL_REQUEST_BASELINE",
            "server_run_dir": value["server_run_dir"],
            "api_run_id": value["api_run_id"],
            "pod_name": value["pod_name"],
            "pod_uid": value["pod_uid"],
            "service_uid": value["service_uid"],
            "server_binding_receipt_sha256": value["server_binding_receipt_sha256"],
            "request_delta_since_prior_sample": 0,
            "traffic_refresh_performed": False,
            "prompts_traces_flags_or_scores_included": False,
        }
        if (
            existing.get("receipt_sha256") != schema._digest(existing)  # noqa: SLF001
            or any(existing.get(key) != item for key, item in expected_identity.items())
            or type(existing.get(STATE_KEY)) is not int
            or existing[STATE_KEY] < 0
            or type(existing.get("observed_at_epoch")) is not int
        ):
            raise ValueError("immutable baseline receipt drifted") from None
        return existing


def traffic_observation(
    before: int,
    after: int,
    *,
    memory_mib: list[int],
    utilization_percent: list[int],
    running_requests: int = 0,
    queued_requests: int = 0,
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
    if any(type(value) is not int or value < 0 for value in (running_requests, queued_requests)):
        raise ValueError("active request gauge drifted")
    if (
        len(memory_mib) != RANKS
        or len(utilization_percent) != RANKS
        or any(type(value) is not int or value <= 0 for value in memory_mib)
        or any(type(value) is not int or not 0 <= value <= 100 for value in utilization_percent)
    ):
        raise ValueError("six-device GPU evidence drifted")
    reasons = []
    if after > before:
        reasons.append("completed_request_counter_increased")
    if running_requests > 0:
        reasons.append("running_requests_positive")
    if queued_requests > 0:
        reasons.append("queued_requests_positive")
    if any(utilization_percent):
        reasons.append("gpu_utilization_positive")
    if not reasons:
        return None
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
        "status": "REAL_REQUEST_OR_GPU_ACTIVITY_OBSERVED",
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
        "running_requests": running_requests,
        "queued_requests": queued_requests,
        "gpu_memory_used_mib_by_device": memory_mib,
        "gpu_utilization_percent_by_device": utilization_percent,
        "activity_reasons": reasons,
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

        after, running_requests, queued_requests = wait_for_metric_snapshot(
            fetch_metrics, schema_path
        )
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
        if before is None:
            before = after
        phase = "traffic_receipt"
        now = int(time.time())
        receipt = traffic_observation(
            before,
            after,
            memory_mib=memory,
            utilization_percent=utilization,
            running_requests=running_requests,
            queued_requests=queued_requests,
            server_run_dir=args.server_run_dir,
            pod_name=socket.gethostname(),
            pod_uid=binding["head_pod_uid"],
            api_run_id=binding["api_run_id"],
            service_uid=binding["service_uid"],
            server_binding_receipt_sha256=binding["receipt_sha256"],
            observed_at_epoch=now,
        )
        if receipt is None:
            phase = "stable_baseline"
            baseline = baseline_observation(
                after,
                server_run_dir=args.server_run_dir,
                pod_name=socket.gethostname(),
                pod_uid=binding["head_pod_uid"],
                api_run_id=binding["api_run_id"],
                service_uid=binding["service_uid"],
                server_binding_receipt_sha256=binding["receipt_sha256"],
                observed_at_epoch=now,
            )
            preserve_initial_baseline(args.baseline_path, baseline)
            _write_status(args.status_path, "STABLE_BOUND_COUNTER_BASELINE", phase)
            return
        event = args.event_dir / f"{now}-{receipt['receipt_sha256'][7:23]}.json"
        legacy._create_json_once(event, receipt)  # noqa: SLF001
        legacy._atomic_json(args.receipt_path, receipt)  # noqa: SLF001
        args.traffic_path.touch(exist_ok=True)
        os.utime(args.traffic_path, (now, now))
        _write_status(args.status_path, "REAL_REQUEST_OR_GPU_ACTIVITY_OBSERVED", phase)
    except Exception:
        _write_status(args.status_path, "FAILED", phase, failed=True)
        raise RuntimeError("DP6 metric observer failed safely") from None


if __name__ == "__main__":
    main()
