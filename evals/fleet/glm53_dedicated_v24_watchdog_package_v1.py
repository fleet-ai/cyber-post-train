"""Create-once package for the v24 server's merged external idle watchdog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as prior
from evals.fleet import glm53_dedicated_v24_server_v1 as server

JOB_NAME = "chris-glm53-dedicated-v24-request-watchdog-v1"
RESULT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"


def render(
    root: Path,
    commit: str,
    binding: dict[str, Any],
    *,
    ready_at_epoch: float,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    prior.validate_cpu_priority_inventory(priority_classes)
    configmap = prior.build_watchdog_configmap_for_binding(
        root,
        commit,
        binding,
        ready_at_epoch=ready_at_epoch,
        binding_validator=server.validate_binding,
        job_name=JOB_NAME,
        result_root=RESULT_ROOT,
    )
    job = prior.build_watchdog_job(configmap, job_name=JOB_NAME, result_root=RESULT_ROOT)
    return {
        "objects": {"apiVersion": "v1", "kind": "List", "items": [configmap, job]},
        "server_binding": binding,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
