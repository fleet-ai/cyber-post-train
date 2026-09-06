"""Single-observation watchdog handoff for GLM v33 custom run identities."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as engine
from evals.fleet import glm53_dedicated_v33_create_v1 as server
from evals.fleet import glm53_dedicated_v33_watchdog_package_v1 as package

LIVE_STATE_SCHEMA = "fleet-glm53-dedicated-v33-watchdog-live-state-v1"
RELEASE_SCHEMA = package.LIVE_RELEASE_SCHEMA
LAUNCH_SCHEMA = "fleet-glm53-dedicated-v33-watchdog-launch-v1"
ADAPTER_LAUNCH_SCHEMA = "fleet-glm53-dedicated-v33-watchdog-adapter-launch-v1"
AUTHORIZATION_CONFIGMAP_NAME = package.JOB_NAME + "-live-release"
_LOCK = threading.Lock()


class AdapterError(RuntimeError):
    """The exact v33 watchdog adapter failed closed."""


def _api_probe_source(api_run_id: str) -> str:
    pattern = server.API_RUN_ID_RE.pattern
    return f"""
import json, os, re, urllib.parse, urllib.request
url = 'https://api.ft.flt.build/v1/runs/' + urllib.parse.quote({api_run_id!r}, safe='')
request = urllib.request.Request(
    url,
    method='GET',
    headers={{
        'Authorization': 'Bearer ' + os.environ['FLEET_API_KEY'],
        'Accept': 'application/json',
    }},
)
with urllib.request.urlopen(request, timeout=30) as response:
    body = json.load(response)
candidates = [body]
for key in ('run', 'config', 'request'):
    child = body.get(key) if isinstance(body, dict) else None
    if isinstance(child, dict):
        candidates.append(child)
        nested = child.get('config')
        if isinstance(nested, dict):
            candidates.append(nested)
def first(*keys):
    for candidate in candidates:
        for key in keys:
            if candidate.get(key) is not None:
                return candidate[key]
    return None
run_ids = {{
    candidate.get(key)
    for candidate in candidates
    for key in ('id', 'run_id', 'name')
    if isinstance(candidate.get(key), str)
    and re.fullmatch({pattern!r}, candidate[key])
}}
print(json.dumps({{
    'http_status': response.status,
    'api_run_id': next(iter(run_ids)) if len(run_ids) == 1 else None,
    'title': first('title'),
    'run_dir': first('run_dir'),
    'state': first('state', 'status'),
}}, sort_keys=True, separators=(',', ':')))
""".strip()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise AdapterError("v33_watchdog_engine_already_bound")
    values = {
        "server": server,
        "package": package,
        "LIVE_STATE_SCHEMA": LIVE_STATE_SCHEMA,
        "RELEASE_SCHEMA": RELEASE_SCHEMA,
        "LAUNCH_SCHEMA": LAUNCH_SCHEMA,
        "AUTHORIZATION_CONFIGMAP_NAME": AUTHORIZATION_CONFIGMAP_NAME,
        "API_RUN_ID_RE": server.API_RUN_ID_RE,
        "_api_probe_source": _api_probe_source,
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
        raise AdapterError("v33_watchdog_package_commit_invalid")
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v33-watchdog-adapter-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "server_title": server.TITLE,
        "server_api_name": server.API_NAME,
        "api_run_id_pattern": server.API_RUN_ID_RE.pattern,
        "server_run_dir": server.RUN_DIR,
        "watchdog_job_name": package.JOB_NAME,
        "watchdog_result_root": package.RESULT_ROOT,
        "idle_release_seconds": engine.runtime.IDLE_RELEASE_SECONDS,
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
        engine._release_local(api_run_id, binding)  # noqa: SLF001


def launch(
    root: Path,
    commit: str,
    api_run_id: str,
    *,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    with bound_engine():
        binding, live = engine.observe_live(api_run_id)
        receipt = engine.launch_from_observation(
            root, commit, binding, live, priority_classes=priority_classes
        )
        if receipt.get("schema_version") != LAUNCH_SCHEMA:
            engine._release_local(api_run_id, binding)  # noqa: SLF001
            raise AdapterError("v33_watchdog_launch_receipt_invalid")
    result: dict[str, Any] = {
        "schema_version": ADAPTER_LAUNCH_SCHEMA,
        "status": "WATCHDOG_ACTIVE_UID_BOUND_SINGLE_OBSERVATION",
        "engine_launch_receipt_sha256": receipt["receipt_sha256"],
        "server_binding": binding,
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "application_ready_receipt_sha256": live["application_ready_receipt_sha256"],
        "runtime": receipt["runtime"],
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    result["receipt_sha256"] = crypto.digest_without(result, "receipt_sha256")
    return result
