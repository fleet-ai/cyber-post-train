"""Fresh zero-project-server create-once GLM v30 identity over the v29 rail."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v29_create_v1 as engine

SCHEMA = "fleet-glm53-dedicated-v30-create-authorization-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v30-create-result-v1"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v30"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v30"
READY_PATH = RUN_DIR + "/READY.json"
READY_SCHEMA = "fleet-glm53-dedicated-v30-application-ready-v1"
CONTROL_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v30-create-control"
AUTHORIZATION_PATH = CONTROL_DIR + "/CREATE-AUTHORIZED.json"
RESULT_PATH = CONTROL_DIR + "/CREATED.json"

AUTH_KEYS = engine.AUTH_KEYS
API_URL = engine.API_URL
AUTH_MAX_AGE_SECONDS = engine.AUTH_MAX_AGE_SECONDS
SERVED_ID = engine.SERVED_ID
MODEL_REVISION = engine.MODEL_REVISION
CONTEXT_LENGTH = engine.CONTEXT_LENGTH
CreateError = engine.CreateError
ServerPlanError = CreateError

_LOCK = threading.Lock()


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise CreateError("v30_create_engine_already_bound")
    values = {
        "SCHEMA": SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "TITLE": TITLE,
        "RUN_DIR": RUN_DIR,
        "READY_PATH": READY_PATH,
        "READY_SCHEMA": READY_SCHEMA,
        "CONTROL_DIR": CONTROL_DIR,
        "AUTHORIZATION_PATH": AUTHORIZATION_PATH,
        "RESULT_PATH": RESULT_PATH,
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


def _observer_source() -> str:
    with bound_engine():
        return engine._observer_source()  # noqa: SLF001


def payload() -> dict[str, Any]:
    with bound_engine():
        return engine.payload()


def validate_payload(value: dict[str, Any]) -> None:
    with bound_engine():
        engine.validate_payload(value)


def request_sha256() -> str:
    with bound_engine():
        return engine.request_sha256()


def validate_binding(binding: dict[str, Any]) -> None:
    with bound_engine():
        engine.validate_binding(binding)


def validate_authorization(value: dict[str, Any]) -> None:
    with bound_engine():
        engine.validate_authorization(value)
    if (
        value.get("active_dedicated_nodes") != 0
        or value.get("active_dedicated_gpus") != 0
        or value.get("planned_nodes_after_create") != 1
        or value.get("planned_gpus_after_create") != 8
    ):
        raise CreateError("v30_create_requires_zero_project_server")


def create_once(
    authorization: dict[str, Any],
    *,
    result_path: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    validate_authorization(authorization)
    with bound_engine():
        return engine.create_once(authorization, result_path=result_path, opener=opener)


def build_held() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v30-create-wrapper-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "server_title": TITLE,
        "server_run_dir": RUN_DIR,
        "request_sha256": request_sha256(),
        "reviewed_create_engine": "evals.fleet.glm53_dedicated_v29_create_v1",
        "required_active_dedicated_nodes": 0,
        "required_active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
        "priority_class": payload()["priority_class"],
        "preemption_policy": "Never",
        "fresh_live_authorization_required": True,
        "watchdog_handoff_required_immediately": True,
        "server_launch_authorized": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "api_mutation_calls": 0,
        "protected_content_included": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=AUTHORIZATION_PATH)
    parser.add_argument("--result-path", default=RESULT_PATH)
    args = parser.parse_args(argv)
    if args.authorization != AUTHORIZATION_PATH or args.result_path != RESULT_PATH:
        raise CreateError("v30_cli_sfs_identity_invalid")
    authorization_path = Path(args.authorization)
    if not authorization_path.is_file():
        raise CreateError("v30_cli_authorization_absent")
    try:
        authorization = json.loads(authorization_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CreateError("v30_cli_authorization_invalid") from exc
    if not isinstance(authorization, dict):
        raise CreateError("v30_cli_authorization_invalid")
    create_once(authorization, result_path=Path(args.result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
