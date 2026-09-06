"""Reusable pre-qualifier and terminal-observer guards for Qwen DP6."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import qwen38_dp6_metric_observer_v4 as metric_observer
from evals.fleet import self_hosted


def observer_pythonpath(dependency_dir: str) -> str:
    path = Path(dependency_dir)
    if path.name != "fleet" or path.parent.name != "evals" or not path.is_absolute():
        raise ValueError("observer dependency directory is not an absolute evals/fleet package")
    return str(path.parents[1])


def validate_observer_environment(env: Mapping[str, Any], dependency_dir: str) -> None:
    expected = observer_pythonpath(dependency_dir)
    if env.get("PYTHONPATH") != expected:
        raise ValueError("observer package root is absent from PYTHONPATH")


def select_stable_sfs_observer(
    items: list[dict[str, Any]], excluded_pod_uid: str
) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for item in items:
        if item.get("kind") != "Pod" or item.get("status", {}).get("phase") != "Running":
            continue
        metadata = item.get("metadata", {})
        uid = metadata.get("uid")
        statuses = item.get("status", {}).get("containerStatuses") or []
        claims = {
            volume.get("persistentVolumeClaim", {}).get("claimName")
            for volume in item.get("spec", {}).get("volumes") or []
        }
        if (
            uid != excluded_pod_uid
            and isinstance(uid, str)
            and uid
            and "sfs-shared" in claims
            and statuses
            and all(row.get("ready") is True for row in statuses)
            and sum(int(row.get("restartCount") or 0) for row in statuses) == 0
        ):
            name = metadata.get("name")
            if isinstance(name, str) and name:
                candidates.append((name, uid))
    if not candidates:
        raise RuntimeError("no stable non-target SFS observer exists")
    return sorted(candidates)[0]


def validate_counter_producer_ready(
    baseline: Mapping[str, Any],
    state: Mapping[str, Any],
    status: Mapping[str, Any],
    binding: Mapping[str, Any],
    run_dir: str,
) -> dict[str, Any]:
    counter = baseline.get(metric_observer.STATE_KEY)

    def digest(value: Mapping[str, Any]) -> str:
        return self_hosted.digest_without(dict(value), "receipt_sha256")

    if (
        baseline.get("receipt_sha256") != digest(baseline)
        or baseline.get("schema_version") != metric_observer.BASELINE_SCHEMA
        or baseline.get("status") != "STABLE_BOUND_GLOBAL_REQUEST_BASELINE"
        or baseline.get("server_run_dir") != run_dir
        or baseline.get("api_run_id") != binding.get("api_run_id")
        or baseline.get("pod_name") != binding.get("head_pod_name")
        or baseline.get("pod_uid") != binding.get("head_pod_uid")
        or baseline.get("service_uid") != binding.get("service_uid")
        or baseline.get("server_binding_receipt_sha256") != binding.get("receipt_sha256")
        or type(counter) is not int
        or counter < 0
        or state != {metric_observer.STATE_KEY: counter}
        or baseline.get("request_delta_since_prior_sample") != 0
        or baseline.get("traffic_refresh_performed") is not False
        or baseline.get("prompts_traces_flags_or_scores_included") is not False
        or status.get("receipt_sha256") != digest(status)
        or status.get("schema_version") != metric_observer.STATUS_SCHEMA
        or status.get("status") != "STABLE_BOUND_COUNTER_BASELINE"
        or status.get("phase") != "stable_baseline"
        or status.get("error_type") is not None
        or status.get("request_or_response_bodies_included") is not False
        or status.get("prompts_traces_flags_or_scores_included") is not False
    ):
        raise ValueError("bound request-counter producer is not ready")
    value = {
        "schema_version": "fleet-qwen38-dp6-counter-producer-ready-v1",
        "status": "READY_BOUND_STABLE_COUNTER_PRODUCER",
        "server_binding_receipt_sha256": binding.get("receipt_sha256"),
        "baseline_receipt_sha256": baseline.get("receipt_sha256"),
        "observer_status_receipt_sha256": status.get("receipt_sha256"),
        "state_matches_baseline": True,
        "scored_calls": 0,
        "prompts_traces_flags_or_scores_included": False,
    }
    value["receipt_sha256"] = digest(value)
    return value
