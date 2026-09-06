"""Create-once external watchdog package for the held GLM v33 server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as prior
from evals.fleet import glm53_dedicated_v33_create_v1 as server

JOB_NAME = "chris-glm53-dedicated-v33-request-watchdog-v1"
RESULT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
RUNTIME_AUTH_SCHEMA = "fleet-glm53-dedicated-v33-watchdog-runtime-auth-v1"
LIVE_RELEASE_SCHEMA = "fleet-glm53-dedicated-v33-watchdog-live-release-v1"


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
    runtime_path = "evals/fleet/glm53_dedicated_v33_request_counter_watchdog_v1.py"
    raw = prior._source(root, commit, runtime_path)  # noqa: SLF001
    key = prior._key(runtime_path)  # noqa: SLF001
    configmap["data"][key] = raw.decode()
    manifest = json.loads(configmap["data"]["package.json"])
    manifest["files"][runtime_path] = crypto.sha256(raw)
    manifest["package_sha256"] = crypto.digest_without(manifest, "package_sha256")
    configmap["data"]["package.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    job = prior.build_watchdog_job(configmap, job_name=JOB_NAME, result_root=RESULT_ROOT)
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    inherited = "python -m evals.fleet.glm53_dedicated_v23_request_counter_watchdog_v1"
    exact = "python -m evals.fleet.glm53_dedicated_v33_request_counter_watchdog_v1"
    if command.count(inherited) != 1:
        raise ValueError("v33_watchdog_entrypoint_template_drifted")
    command = command.replace(inherited, exact)
    suffix = '--ready-at-epoch "$READY_AT_EPOCH"'
    if not command.endswith(suffix):
        raise ValueError("v33_watchdog_command_contract_drifted")
    expected = (
        f"--expected-runtime-auth-schema {RUNTIME_AUTH_SCHEMA} "
        f"--expected-live-release-schema {LIVE_RELEASE_SCHEMA} "
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
