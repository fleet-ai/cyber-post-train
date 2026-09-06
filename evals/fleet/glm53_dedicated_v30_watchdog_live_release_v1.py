"""Exact held v30 adapter for the reviewed live watchdog handoff engine."""

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
from evals.fleet import glm53_dedicated_v30_create_v1 as server
from evals.fleet import glm53_dedicated_v30_watchdog_package_v1 as package

LIVE_STATE_SCHEMA = "fleet-glm53-dedicated-v30-watchdog-live-state-v1"
RELEASE_SCHEMA = package.LIVE_RELEASE_SCHEMA
LAUNCH_SCHEMA = "fleet-glm53-dedicated-v30-watchdog-launch-v1"
ADAPTER_LAUNCH_SCHEMA = "fleet-glm53-dedicated-v30-watchdog-adapter-launch-v1"
AUTHORIZATION_CONFIGMAP_NAME = package.JOB_NAME + "-live-release"

_LOCK = threading.Lock()


class AdapterError(RuntimeError):
    """The exact v30 watchdog adapter failed closed."""


def build_held(package_commit: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", package_commit) is None:
        raise AdapterError("v30_watchdog_package_commit_invalid")
    server.validate_payload(server.payload())
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v30-watchdog-adapter-held-v1",
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
        "exact_v30_contract_bound": True,
        "controller_side_fleet_credential_required": True,
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
        raise AdapterError("v30_watchdog_engine_already_bound")
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
        binding, live = engine.observe_live(api_run_id)
        engine_receipt = engine.launch(
            root, commit, api_run_id, priority_classes=priority_classes
        )
        if (
            engine_receipt.get("schema_version") != LAUNCH_SCHEMA
            or engine_receipt.get("server_binding_sha256")
            != crypto.sha256(crypto.canonical_json(binding))
            or engine_receipt.get("server_launch_authorized") is not False
            or engine_receipt.get("watchdog_launch_authorized") is not True
            or engine_receipt.get("qualification_launch_authorized") is not False
            or engine_receipt.get("scored_launch_authorized") is not False
            or engine_receipt.get("protected_content_included") is not False
        ):
            engine._release_local(api_run_id)  # noqa: SLF001
            raise AdapterError("v30_watchdog_launch_receipt_invalid")
    receipt: dict[str, Any] = {
        "schema_version": ADAPTER_LAUNCH_SCHEMA,
        "status": "WATCHDOG_ACTIVE_UID_BOUND",
        "engine_launch_receipt_sha256": engine_receipt["receipt_sha256"],
        "server_binding": binding,
        "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
        "application_ready_receipt_sha256": live[
            "application_ready_receipt_sha256"
        ],
        "runtime": engine_receipt["runtime"],
        "server_launch_authorized": False,
        "watchdog_launch_authorized": True,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launch", nargs="?")
    parser.add_argument("--api-run-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--package-commit", required=True)
    args = parser.parse_args(argv)
    priorities = engine._kubectl_json("priorityclasses.scheduling.k8s.io").get(  # noqa: SLF001
        "items", []
    )
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
