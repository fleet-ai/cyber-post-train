"""Immutable held package for the GLM v30 create/watchdog/parity controller."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v29_controller_package_v1 as prior
from evals.fleet import glm53_dedicated_v30_controller_v1 as controller
from evals.fleet import glm53_dedicated_v30_create_v1 as server
from evals.fleet import glm53_dedicated_v30_incluster_parity_v1 as parity

JOB_NAME = "chris-glm53-v30-create-watchdog-parity-controller-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = JOB_NAME + "-authorization"
PACKAGE_SCHEMA = "fleet-glm53-dedicated-v30-controller-package-v1"
FILES = tuple(
    sorted(
        set(prior.FILES)
        | set(parity.FILES)
        | {
            "evals/fleet/glm53_dedicated_v30_controller_v1.py",
            "evals/fleet/glm53_dedicated_v30_create_v1.py",
            "evals/fleet/glm53_dedicated_v30_watchdog_live_release_v1.py",
            "evals/fleet/glm53_dedicated_v30_watchdog_package_v1.py",
        }
    )
)

_LOCK = threading.Lock()


class PackageError(RuntimeError):
    """The exact v30 controller package is invalid."""


@contextmanager
def bound_prior() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise PackageError("v30_controller_package_already_bound")
    values = {
        "JOB_NAME": JOB_NAME,
        "CONFIGMAP_NAME": CONFIGMAP_NAME,
        "AUTHORIZATION_CONFIGMAP_NAME": AUTHORIZATION_CONFIGMAP_NAME,
        "FILES": FILES,
        "controller": controller,
        "server": server,
    }
    original = {name: getattr(prior, name) for name in values}
    try:
        for name, value in values.items():
            setattr(prior, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(prior, name, value)
        _LOCK.release()


def build_source_configmap(root: Path, commit: str) -> dict[str, Any]:
    with bound_prior():
        result = prior.build_source_configmap(root, commit)
    package = json.loads(result["data"]["package.json"])
    package["schema_version"] = PACKAGE_SCHEMA
    package["package_sha256"] = crypto.digest_without(package, "package_sha256")
    result["data"]["package.json"] = (
        json.dumps(package, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return result


def build_authorization_configmap(authorization: dict[str, Any]) -> dict[str, Any]:
    server.validate_authorization(authorization)
    with bound_prior():
        return prior.build_authorization_configmap(authorization)


def build_job(package_commit: str) -> dict[str, Any]:
    with bound_prior():
        job = prior.build_job(package_commit)
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    old = "python -m evals.fleet.glm53_dedicated_v29_controller_v1"
    new = "python -m evals.fleet.glm53_dedicated_v30_controller_v1"
    if command.count(old) != 1:
        raise PackageError("v30_controller_entrypoint_template_invalid")
    job["spec"]["template"]["spec"]["containers"][0]["command"][-1] = command.replace(
        old, new
    )
    return job


def render(root: Path, commit: str, authorization: dict[str, Any]) -> dict[str, Any]:
    source = build_source_configmap(root, commit)
    result = {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [
                source,
                build_authorization_configmap(authorization),
                build_job(commit),
            ],
        },
        "server_launch_authorized": False,
        "watchdog_handoff_required": True,
        "incluster_parity_required": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
    package = json.loads(source["data"]["package.json"])
    if (
        package.get("schema_version") != PACKAGE_SCHEMA
        or package.get("server_title") != server.TITLE
        or package.get("server_run_dir") != server.RUN_DIR
        or set(package.get("files", {})) != set(FILES)
    ):
        raise PackageError("v30_controller_package_identity_invalid")
    return result


def build_held(package_commit: str) -> dict[str, Any]:
    source = build_source_configmap(Path.cwd(), package_commit)
    package = json.loads(source["data"]["package.json"])
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v30-lifecycle-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "package_sha256": package["package_sha256"],
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "controller_job_name": JOB_NAME,
        "watchdog_job_name": "chris-glm53-dedicated-v30-request-watchdog-v1",
        "parity_job_name": parity.JOB_NAME,
        "required_active_dedicated_nodes": 0,
        "required_active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
        "server_priority_class": server.payload()["priority_class"],
        "server_preemption_policy": "Never",
        "idle_release_seconds": 600,
        "watchdog_active_before_parity": True,
        "incluster_parity_required": True,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body
