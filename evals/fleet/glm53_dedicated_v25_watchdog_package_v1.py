"""Create-once external watchdog package for the exact GLM v25 server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as prior
from evals.fleet import glm53_dedicated_v25_create_v1 as server

JOB_NAME = "chris-glm53-dedicated-v25-request-watchdog-v1"
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
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    suffix = '--ready-at-epoch "$READY_AT_EPOCH"'
    if not command.endswith(suffix):
        raise ValueError("v25_watchdog_command_contract_drifted")
    expected = (
        "--expected-runtime-auth-schema "
        "fleet-glm53-dedicated-v25-watchdog-runtime-auth-v1 "
        "--expected-live-release-schema "
        "fleet-glm53-dedicated-v25-watchdog-live-release-v1 "
        f"--expected-watchdog-job-name {JOB_NAME} "
        f"--expected-watchdog-result-root {RESULT_ROOT} "
    )
    job["spec"]["template"]["spec"]["containers"][0]["command"][-1] = (
        command[: -len(suffix)] + expected + suffix
    )
    return {
        "objects": {"apiVersion": "v1", "kind": "List", "items": [configmap, job]},
        "server_binding": binding,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
