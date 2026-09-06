"""Content-free DP6 observer that preserves metric schema before validation."""

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
from evals.fleet import self_hosted

RANKS = 6
STATUS_SCHEMA = prior.STATUS_SCHEMA
SCHEMA_OBSERVATION = "fleet-qwen38-dp6-metric-schema-observation-v1"
TARGET_FAMILIES = (prior.REQUEST_FAMILY, prior.STARTUP_ANCHOR_FAMILY)
METRIC_SCHEMA_WARMUP_SECONDS = 30
METRIC_SCHEMA_POLL_SECONDS = 1


class MetricSchemaWarmupTimeout(ValueError):
    """The target metric families never converged inside the bounded window."""


def schema_observation(
    metrics: str,
    *,
    observed_at_epoch: int,
    warmup_attempt: int = 1,
    validation_status: str = "CAPTURED_BEFORE_METRIC_VALIDATION",
) -> dict[str, Any]:
    """Describe only target-family shape; never persist metric values."""
    families: dict[str, dict[str, Any]] = {}
    for family in TARGET_FAMILIES:
        label_key_sets: set[tuple[str, ...]] = set()
        parsed_samples = 0
        raw_family_lines = 0
        for raw in metrics.splitlines():
            stripped = raw.strip()
            if stripped.startswith(family + "{") or stripped.startswith(family + " "):
                raw_family_lines += 1
            match = legacy.METRIC_RE.match(stripped)
            if match is None or match.group("head") != family:
                continue
            parsed_samples += 1
            labels = dict(legacy.LABEL_RE.findall(match.group("labels") or ""))
            label_key_sets.add(tuple(sorted(labels)))
        families[family] = {
            "raw_family_line_count": raw_family_lines,
            "parsed_sample_count": parsed_samples,
            "label_key_sets": [list(keys) for keys in sorted(label_key_sets)],
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
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _write_schema(
    path: Path, metrics: str, *, attempt: int, validation_status: str
) -> None:
    legacy._atomic_json(  # noqa: SLF001 - same immutable observer package
        path,
        schema_observation(
            metrics,
            observed_at_epoch=int(time.time()),
            warmup_attempt=attempt,
            validation_status=validation_status,
        ),
    )


def wait_for_request_counters(
    fetch_metrics: Any,
    schema_path: Path,
    *,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> list[int]:
    """Wait briefly for all scheduler metric files to reach the shared registry."""
    deadline = monotonic() + METRIC_SCHEMA_WARMUP_SECONDS
    attempt = 0
    while True:
        attempt += 1
        metrics = fetch_metrics()
        try:
            counters = prior.request_counters(metrics)
        except ValueError as exc:
            _write_schema(
                schema_path,
                metrics,
                attempt=attempt,
                validation_status="INCOMPLETE_DURING_BOUNDED_WARMUP",
            )
            if monotonic() >= deadline:
                raise MetricSchemaWarmupTimeout(
                    "metric schema did not converge before the bounded deadline"
                ) from exc
            sleep(METRIC_SCHEMA_POLL_SECONDS)
            continue
        _write_schema(
            schema_path,
            metrics,
            attempt=attempt,
            validation_status="CONVERGED",
        )
        return counters


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

        after = wait_for_request_counters(fetch_metrics, schema_path)
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
            prior._write_status(  # noqa: SLF001 - versioned observer composition
                args.status_path, "COUNTER_STATE_INITIALIZED", phase
            )
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
        prior._write_status(  # noqa: SLF001
            args.status_path, "REAL_REQUEST_COUNTER_INCREASED", phase
        )
    except Exception as exc:
        prior._write_status(args.status_path, "FAILED", phase, exc)  # noqa: SLF001
        raise


if __name__ == "__main__":
    main()
