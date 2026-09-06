"""Exact v27 adapter for the reviewed live watchdog handoff engine."""

from __future__ import annotations

import argparse
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as engine
from evals.fleet import glm53_dedicated_v27_create_v1 as server
from evals.fleet import glm53_dedicated_v27_watchdog_package_v1 as package

LIVE_STATE_SCHEMA = "fleet-glm53-dedicated-v27-watchdog-live-state-v1"
RELEASE_SCHEMA = package.LIVE_RELEASE_SCHEMA
LAUNCH_SCHEMA = "fleet-glm53-dedicated-v27-watchdog-launch-v1"
AUTHORIZATION_CONFIGMAP_NAME = package.JOB_NAME + "-live-release"

_LOCK = threading.Lock()


class AdapterError(RuntimeError):
    """The exact v27 watchdog adapter failed closed."""


def build_held(package_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", package_commit) is None:
        raise AdapterError("v27_watchdog_package_commit_invalid")
    server.validate_payload(server.payload())
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v27-watchdog-adapter-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "package_commit": package_commit,
        "server_title": server.TITLE,
        "server_run_dir": server.RUN_DIR,
        "server_request_sha256": server.request_sha256(),
        "watchdog_job_name": package.JOB_NAME,
        "watchdog_result_root": package.RESULT_ROOT,
        "runtime_auth_schema": package.RUNTIME_AUTH_SCHEMA,
        "live_release_schema": package.LIVE_RELEASE_SCHEMA,
        "idle_release_seconds": engine.runtime.IDLE_RELEASE_SECONDS,
        "engine_module": "evals.fleet.glm53_dedicated_v24_watchdog_live_release_v1",
        "generic_runtime_module": (
            "evals.fleet.glm53_dedicated_v23_request_counter_watchdog_v1"
        ),
        "exact_v27_contract_bound": True,
        "engine_binding_restored_after_call": True,
        "server_launch_authorized": False,
        "watchdog_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise AdapterError("v27_watchdog_engine_already_bound")
    values = {
        "server": server,
        "package": package,
        "LIVE_STATE_SCHEMA": LIVE_STATE_SCHEMA,
        "RELEASE_SCHEMA": RELEASE_SCHEMA,
        "LAUNCH_SCHEMA": LAUNCH_SCHEMA,
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


def launch(
    root: Path,
    commit: str,
    api_run_id: str,
    *,
    priority_classes: list[dict[str, Any]],
) -> dict[str, Any]:
    with bound_engine():
        receipt = engine.launch(root, commit, api_run_id, priority_classes=priority_classes)
        if (
            receipt.get("schema_version") != LAUNCH_SCHEMA
            or receipt.get("server_launch_authorized") is not False
            or receipt.get("watchdog_launch_authorized") is not True
            or receipt.get("qualification_launch_authorized") is not False
            or receipt.get("scored_launch_authorized") is not False
            or receipt.get("protected_content_included") is not False
        ):
            engine._release_local(api_run_id)
            raise AdapterError("v27_watchdog_launch_receipt_invalid")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", nargs="?")
    parser.add_argument("--api-run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit", required=True)
    args = parser.parse_args(argv)
    priorities = engine._kubectl_json("priorityclasses.scheduling.k8s.io").get("items", [])
    receipt = launch(
        args.repo_root,
        args.package_commit,
        args.api_run_id,
        priority_classes=priorities,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
