"""Fail closed on Pod-policy drift or Kueue workload preemption.

``preemptionPolicy: Never`` is a Kubernetes scheduler property. It prevents a
Pod from preempting lower-priority Pods; it does not make a Kueue Workload
immune to quota preemption. Long-running GPU operators must therefore check
both layers and preserve the exact Workload UID.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

STATIC_PRIORITY_POLICIES = {
    "fleet-infra-quiet": "Never",
    "fleet-serve-low": "Never",
    "fleet-train-high": "PreemptLowerPriority",
    "fleet-infra-loud": "PreemptLowerPriority",
}


def validate_pod_priority_policy(
    pod_spec: dict[str, Any], priority_policies: dict[str, str] | None = None
) -> None:
    priority = pod_spec.get("priorityClassName")
    requested = pod_spec.get("preemptionPolicy")
    policies = priority_policies or STATIC_PRIORITY_POLICIES
    if priority not in policies:
        raise ValueError("rendered Pod uses an unknown PriorityClass")
    actual = policies[priority]
    if requested is not None and requested != actual:
        raise ValueError("rendered Pod preemptionPolicy contradicts its PriorityClass")


def select_highest_nonpreempting(
    priority_classes: list[dict[str, Any]], *, allowed_names: set[str] | None = None
) -> dict[str, Any]:
    """Select the highest live and API-allowed class that cannot preempt peers."""
    eligible = [
        row
        for row in priority_classes
        if isinstance(row, dict)
        and row.get("preemptionPolicy") == "Never"
        and isinstance((row.get("metadata") or {}).get("name"), str)
        and type(row.get("value")) is int
        and (allowed_names is None or row["metadata"]["name"] in allowed_names)
    ]
    if not eligible:
        raise ValueError("no live nonpreempting PriorityClass is available")
    top = max(row["value"] for row in eligible)
    winners = [row for row in eligible if row["value"] == top]
    if len(winners) != 1:
        raise ValueError("highest nonpreempting PriorityClass is ambiguous")
    row = winners[0]
    return {
        "name": row["metadata"]["name"],
        "uid": str((row.get("metadata") or {}).get("uid") or ""),
        "value": row["value"],
        "preemption_policy": "Never",
        "selection_scope": "live_cluster"
        if allowed_names is None
        else "live_cluster_intersect_jobs_api",
    }


def require_no_kueue_preemption(
    workload: dict[str, Any], *, expected_uid: str
) -> None:
    """Reject a UID-bound Workload that Kueue has ever preempted or requeued.

    Kueue retains historical preemption text after a workload is readmitted.
    Checking only the current ``Admitted=True`` condition would therefore let a
    recreated RayCluster masquerade as the original serving endpoint.
    """
    metadata = workload.get("metadata") or {}
    if metadata.get("uid") != expected_uid:
        raise ValueError("Kueue Workload UID changed")
    conditions = (workload.get("status") or {}).get("conditions") or []
    if not isinstance(conditions, list) or not all(
        isinstance(condition, dict) for condition in conditions
    ):
        raise ValueError("Kueue Workload conditions are invalid")
    for condition in conditions:
        condition_type = str(condition.get("type") or "").lower()
        reason = str(condition.get("reason") or "").lower()
        message = str(condition.get("message") or "").lower()
        if "preempt" in reason or "preempt" in message or (
            condition_type in {"preempted", "requeued", "evicted"}
            and condition.get("status") == "True"
        ):
            raise RuntimeError("Kueue Workload was preempted or evicted")


def main() -> int:
    rendered = json.load(sys.stdin)
    raw_live = os.environ.get("PRIORITY_CLASS_POLICIES_JSON")
    live = json.loads(raw_live) if raw_live else None
    if live is not None and (
        not isinstance(live, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in live.items())
    ):
        raise ValueError("live PriorityClass policy inventory is invalid")
    kind = rendered.get("kind")
    if kind == "Job":
        pod_spec = rendered["spec"]["template"]["spec"]
    elif kind == "Pod":
        pod_spec = rendered["spec"]
    else:
        raise ValueError("rendered object must be a Job or Pod")
    validate_pod_priority_policy(pod_spec, live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
