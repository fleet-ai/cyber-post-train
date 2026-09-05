"""Fail closed when a rendered Pod contradicts its PriorityClass policy."""

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
