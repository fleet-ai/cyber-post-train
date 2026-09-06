"""Create-once external watchdog package for the held GLM v34 server."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v33_watchdog_package_v1 as prior
from evals.fleet import glm53_dedicated_v34_create_v1 as server

JOB_NAME = "chris-glm53-dedicated-v34-request-watchdog-v1"
RESULT_ROOT = f"/mnt/sfs/jobs/{JOB_NAME}"
RUNTIME_AUTH_SCHEMA = "fleet-glm53-dedicated-v34-watchdog-runtime-auth-v1"
LIVE_RELEASE_SCHEMA = "fleet-glm53-dedicated-v34-watchdog-live-release-v1"
_LOCK = threading.Lock()


@contextmanager
def bound_prior() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise ValueError("v34_watchdog_package_already_bound")
    values = {
        "JOB_NAME": JOB_NAME,
        "RESULT_ROOT": RESULT_ROOT,
        "RUNTIME_AUTH_SCHEMA": RUNTIME_AUTH_SCHEMA,
        "LIVE_RELEASE_SCHEMA": LIVE_RELEASE_SCHEMA,
        "server": server,
    }
    previous = {name: getattr(prior, name) for name in values}
    try:
        for name, value in values.items():
            setattr(prior, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(prior, name, value)
        _LOCK.release()


def render(
    root: Path,
    commit: str,
    binding: dict[str, Any],
    *,
    ready_at_epoch: float,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    with bound_prior():
        rendered = prior.render(
            root,
            commit,
            binding,
            ready_at_epoch=ready_at_epoch,
            priority_classes=priority_classes,
        )
    configmap, job = rendered["objects"]["items"]
    runtime_path = "evals/fleet/glm53_dedicated_v34_request_counter_watchdog_v1.py"
    raw = prior.prior._source(root, commit, runtime_path)  # noqa: SLF001
    key = prior.prior._key(runtime_path)  # noqa: SLF001
    configmap["data"][key] = raw.decode()
    manifest = json.loads(configmap["data"]["package.json"])
    manifest["files"][runtime_path] = crypto.sha256(raw)
    manifest["package_sha256"] = crypto.digest_without(manifest, "package_sha256")
    configmap["data"]["package.json"] = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    old = "python -m evals.fleet.glm53_dedicated_v33_request_counter_watchdog_v1"
    new = "python -m evals.fleet.glm53_dedicated_v34_request_counter_watchdog_v1"
    if command.count(old) != 1:
        raise ValueError("v34_watchdog_entrypoint_template_drifted")
    command = command.replace(old, new)
    job["spec"]["template"]["spec"]["containers"][0]["command"][-1] = command
    return {
        "objects": {"apiVersion": "v1", "kind": "List", "items": [configmap, job]},
        "server_binding": binding,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
