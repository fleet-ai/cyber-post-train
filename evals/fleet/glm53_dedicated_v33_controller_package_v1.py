"""Immutable held package for the GLM v33 custom-name lifecycle."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v32_controller_package_v1 as prior
from evals.fleet import glm53_dedicated_v33_controller_v1 as controller
from evals.fleet import glm53_dedicated_v33_create_v1 as server
from evals.fleet import glm53_dedicated_v33_incluster_parity_v1 as parity
from evals.fleet import glm53_dedicated_v33_live_authorization_v1 as live_authorization

JOB_NAME = "chris-glm53-v33-create-watchdog-parity-controller-v1"
CONFIGMAP_NAME = JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = JOB_NAME + "-authorization"
PACKAGE_SCHEMA = "fleet-glm53-dedicated-v33-controller-package-v1"
FILES = tuple(
    sorted(
        set(prior.FILES)
        | set(parity.FILES)
        | {
            "evals/fleet/glm53_dedicated_v33_controller_v1.py",
            "evals/fleet/glm53_dedicated_v33_create_v1.py",
            "evals/fleet/glm53_dedicated_v33_live_authorization_v1.py",
            "evals/fleet/glm53_dedicated_v33_watchdog_live_release_v1.py",
            "evals/fleet/glm53_dedicated_v33_watchdog_package_v1.py",
        }
    )
)
_LOCK = threading.Lock()


class PackageError(RuntimeError):
    """The exact v33 controller package is invalid."""


@contextmanager
def bound_prior() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise PackageError("v33_controller_package_already_bound")
    values = {
        "JOB_NAME": JOB_NAME,
        "CONFIGMAP_NAME": CONFIGMAP_NAME,
        "AUTHORIZATION_CONFIGMAP_NAME": AUTHORIZATION_CONFIGMAP_NAME,
        "PACKAGE_SCHEMA": PACKAGE_SCHEMA,
        "FILES": FILES,
        "controller": controller,
        "server": server,
        "parity": parity,
        "live_authorization": live_authorization,
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


def build_authorization_configmap(value: dict[str, Any]) -> dict[str, Any]:
    server.validate_authorization(value)
    with bound_prior():
        return prior.build_authorization_configmap(value)


def build_job(package_commit: str) -> dict[str, Any]:
    with bound_prior():
        job = prior.build_job(package_commit)
    command = job["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    old = "python -m evals.fleet.glm53_dedicated_v32_controller_v1"
    new = "python -m evals.fleet.glm53_dedicated_v33_controller_v1"
    if command.count(old) != 1:
        raise PackageError("v33_controller_entrypoint_template_invalid")
    job["spec"]["template"]["spec"]["containers"][0]["command"][-1] = command.replace(old, new)
    return job


def render(root: Path, commit: str, authorization: dict[str, Any]) -> dict[str, Any]:
    source = build_source_configmap(root, commit)
    result = {
        "objects": {
            "apiVersion": "v1",
            "kind": "List",
            "items": [source, build_authorization_configmap(authorization), build_job(commit)],
        },
        "server_launch_authorized": False,
        "watchdog_handoff_required": True,
        "exactly_one_live_observation_required": True,
        "generation_exact_parity_entrypoint_required": True,
        "incluster_parity_required": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
    }
    package = json.loads(source["data"]["package.json"])
    if set(package.get("files", {})) != set(FILES):
        raise PackageError("v33_controller_package_identity_invalid")
    return result


def build_held(root: Path, package_commit: str) -> dict[str, Any]:
    source = build_source_configmap(root, package_commit)
    package = json.loads(source["data"]["package.json"])
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v33-lifecycle-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "package_sha256": package["package_sha256"],
        "server_title": server.TITLE,
        "server_api_name": server.API_NAME,
        "api_run_id_pattern": server.API_RUN_ID_RE.pattern,
        "server_run_dir": server.RUN_DIR,
        "controller_job_name": JOB_NAME,
        "watchdog_job_name": "chris-glm53-dedicated-v33-request-watchdog-v1",
        "parity_job_name": parity.JOB_NAME,
        "parity_entrypoint_module": "evals.fleet.glm53_dedicated_v33_incluster_parity_v1",
        "required_active_dedicated_nodes": 0,
        "required_active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
        "idle_release_seconds": 600,
        "exactly_one_live_observation": True,
        "watchdog_active_before_parity": True,
        "failure_rollback_and_release_required": True,
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
