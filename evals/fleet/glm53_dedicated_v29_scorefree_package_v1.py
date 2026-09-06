"""Fresh immutable package adapter for the exact GLM v29 score-free qualifier."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v23_scorefree_package_v1 as package
from evals.fleet import glm53_dedicated_v29_scorefree_qualifier_v1 as qualifier

SCHEMA = "fleet-glm53-dedicated-v29-scorefree-package-v1"
CONFIGMAP_NAME = qualifier.JOB_NAME + "-package"
AUTHORIZATION_CONFIGMAP_NAME = qualifier.JOB_NAME + "-authorization"
RUN = "evals/fleet/scripts/run_glm53_dedicated_v29_scorefree_qualification_v1.sh"
FILES = tuple(
    sorted(
        set(package.FILES)
        | {
            "evals/fleet/glm53_dedicated_v29_scorefree_qualifier_v1.py",
        }
    )
)
OPERATOR_FILES = (
    "evals/fleet/glm53_dedicated_v29_scorefree_gpu_observer_v1.py",
    "evals/fleet/scripts/observe_glm53_dedicated_v29_scorefree_gpu_v1.sh",
)
_LOCK = threading.Lock()


@contextmanager
def bound_package() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise package.PackageError("v29_scorefree_package_already_bound")
    values = {
        "qualifier": qualifier,
        "SCHEMA": SCHEMA,
        "CONFIGMAP_NAME": CONFIGMAP_NAME,
        "AUTHORIZATION_CONFIGMAP_NAME": AUTHORIZATION_CONFIGMAP_NAME,
        "RUN": RUN,
        "FILES": FILES,
        "OPERATOR_FILES": OPERATOR_FILES,
    }
    prior = {name: getattr(package, name) for name in values}
    try:
        for name, value in values.items():
            setattr(package, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(package, name, value)
        _LOCK.release()


def build_configmap(root: Path, commit: str) -> dict[str, Any]:
    with bound_package():
        return package.build_configmap(root, commit)


def build_authorization_configmap(authorization: dict[str, Any]) -> dict[str, Any]:
    with bound_package():
        return package.build_authorization_configmap(authorization)


def build_job(configmap: dict[str, Any], authorization: dict[str, Any]) -> dict[str, Any]:
    with bound_package():
        return package.build_job(configmap, authorization)


def render(root: Path, commit: str, authorization: dict[str, Any]) -> dict[str, Any]:
    with bound_package():
        return package.render(root, commit, authorization)


def build_held(package_commit: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v29-scorefree-package-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "job_name": qualifier.JOB_NAME,
        "result_root": str(qualifier.RESULT_ROOT),
        "server_title": qualifier.SERVER_TITLE,
        "server_run_dir": qualifier.SERVER_RUN_DIR,
        "concurrency_waves": list(qualifier.CONCURRENCY),
        "actual_opencode_bash_submit_report_required": True,
        "fleet_task_instance_calls": 0,
        "fleet_session_calls": 0,
        "verifier_calls": 0,
        "scoring_calls": 0,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body
