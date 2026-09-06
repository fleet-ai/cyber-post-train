"""Single-observation watchdog handoff for GLM v35 custom run identities."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v33_watchdog_live_release_v1 as engine
from evals.fleet import glm53_dedicated_v35_create_v1 as server
from evals.fleet import glm53_dedicated_v35_watchdog_package_v1 as package

LIVE_STATE_SCHEMA = "fleet-glm53-dedicated-v35-watchdog-live-state-v1"
RELEASE_SCHEMA = package.LIVE_RELEASE_SCHEMA
LAUNCH_SCHEMA = "fleet-glm53-dedicated-v35-watchdog-launch-v1"
ADAPTER_LAUNCH_SCHEMA = "fleet-glm53-dedicated-v35-watchdog-adapter-launch-v1"
AUTHORIZATION_CONFIGMAP_NAME = package.JOB_NAME + "-live-release"
AdapterError = engine.AdapterError
_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise AdapterError("v35_watchdog_engine_already_bound")
    values = {
        "server": server,
        "package": package,
        "LIVE_STATE_SCHEMA": LIVE_STATE_SCHEMA,
        "RELEASE_SCHEMA": RELEASE_SCHEMA,
        "LAUNCH_SCHEMA": LAUNCH_SCHEMA,
        "ADAPTER_LAUNCH_SCHEMA": ADAPTER_LAUNCH_SCHEMA,
        "AUTHORIZATION_CONFIGMAP_NAME": AUTHORIZATION_CONFIGMAP_NAME,
    }
    prior = {name: getattr(engine, name) for name in values}
    try:
        for name, value in values.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in prior.items():
            setattr(engine, name, value)
        _LOCK.release()


def build_held(package_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", package_commit) is None:
        raise AdapterError("v35_watchdog_package_commit_invalid")
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v35-watchdog-adapter-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "server_title": server.TITLE,
        "server_api_name": server.API_NAME,
        "api_run_id_pattern": server.API_RUN_ID_RE.pattern,
        "server_run_dir": server.RUN_DIR,
        "watchdog_job_name": package.JOB_NAME,
        "watchdog_result_root": package.RESULT_ROOT,
        "idle_release_seconds": engine.engine.runtime.IDLE_RELEASE_SECONDS,
        "exactly_one_live_observation": True,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def observe_live(api_run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with bound_engine():
        return engine.observe_live(api_run_id)


def release_local(api_run_id: str, binding: dict[str, Any] | None = None) -> None:
    with bound_engine():
        engine.release_local(api_run_id, binding)


def launch(
    root: Path,
    commit: str,
    api_run_id: str,
    *,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    with bound_engine():
        result = engine.launch(
            root,
            commit,
            api_run_id,
            priority_classes=priority_classes,
        )
    if result.get("schema_version") != ADAPTER_LAUNCH_SCHEMA:
        release_local(api_run_id, result.get("server_binding"))
        raise AdapterError("v35_watchdog_launch_receipt_invalid")
    return result
