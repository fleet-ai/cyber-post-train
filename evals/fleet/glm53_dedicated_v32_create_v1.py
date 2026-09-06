"""Fresh zero-project-server create-once GLM v32 identity over the v31 rail."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_release
from evals.fleet import glm53_dedicated_v31_create_v1 as engine
from evals.fleet import glm53_dedicated_v32_live_authorization_v1 as live_authorization

SCHEMA = "fleet-glm53-dedicated-v32-create-authorization-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v32-create-result-v1"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v32"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v32"
READY_PATH = RUN_DIR + "/READY.json"
READY_SCHEMA = "fleet-glm53-dedicated-v32-application-ready-v1"
CONTROL_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v32-create-control"
AUTHORIZATION_PATH = CONTROL_DIR + "/CREATE-AUTHORIZED.json"
RESULT_PATH = CONTROL_DIR + "/CREATED.json"

AUTH_KEYS = engine.AUTH_KEYS | {"live_observation"}
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
        raise CreateError("v32_create_engine_already_bound")
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
        value = engine.payload()
    value["name"] = TITLE
    return value


def validate_payload(value: dict[str, Any]) -> None:
    if value != payload():
        raise CreateError("v32_create_payload_invalid")
    legacy = dict(value)
    legacy.pop("name")
    with bound_engine():
        engine.validate_payload(legacy)


def request_sha256() -> str:
    return crypto.sha256(crypto.canonical_json(payload()))


def validate_binding(binding: dict[str, Any]) -> None:
    with bound_engine():
        engine.validate_binding(binding)


def validate_authorization(value: dict[str, Any]) -> None:
    now = time.time()
    observed = value.get("observed_at_epoch")
    if (
        set(value) != AUTH_KEYS
        or value.get("schema_version") != SCHEMA
        or value.get("status") != "PASSED_LIVE_CREATE_GATES"
        or not isinstance(observed, (int, float))
        or isinstance(observed, bool)
        or observed > now + 5
        or now - observed > AUTH_MAX_AGE_SECONDS
        or value.get("server_title") != TITLE
        or value.get("server_run_dir") != RUN_DIR
        or value.get("request_sha256") != request_sha256()
        or value.get("preview_http_status") != 200
        or value.get("jobs_api_title_matches") != 0
        or value.get("jobs_api_run_dir_matches") != 0
        or value.get("kubernetes_identity_or_remnant_matches") != 0
        or value.get("sfs_run_dir_absent") is not True
        or value.get("control_result_absent") is not True
        or value.get("active_dedicated_nodes") != 0
        or value.get("active_dedicated_gpus") != 0
        or value.get("planned_nodes_after_create") != 1
        or value.get("planned_gpus_after_create") != 8
        or value.get("priority_class") != payload()["priority_class"]
        or value.get("preemption_policy") != "Never"
        or value.get("server_launch_authorized") is not True
        or value.get("watchdog_handoff_required_immediately") is not True
        or value.get("qualification_launch_authorized") is not False
        or value.get("scored_launch_authorized") is not False
        or value.get("api_mutation_calls") != 0
        or value.get("protected_content_included") is not False
        or value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
    ):
        raise CreateError("v32_create_authorization_invalid")
    live = value.get("live_observation")
    if not isinstance(live, dict):
        raise CreateError("v32_create_authorization_invalid")
    try:
        live_authorization.validate_live_observation(live, value)
    except live_authorization.LiveAuthorizationError as exc:
        raise CreateError("v32_create_authorization_invalid") from exc


def create_once(
    authorization: dict[str, Any],
    *,
    result_path: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    validate_authorization(authorization)
    if result_path.exists():
        raise CreateError("v32_result_already_exists_reconcile_do_not_retry")
    token = os.environ.get("FLEET_API_KEY", "")
    if not token:
        raise CreateError("v32_jobs_api_credential_absent")
    request_body = payload()
    request = urllib.request.Request(
        API_URL,
        data=crypto.canonical_json(request_body),
        method="POST",
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
    )
    try:
        with opener(request, timeout=30) as response:
            status = int(response.status)
            raw = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise CreateError("v32_create_response_unknown_reconcile_do_not_retry") from exc
    if status != 202:
        raise CreateError("v32_create_http_status_invalid_reconcile_do_not_retry")
    try:
        response_value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CreateError("v32_create_response_invalid_reconcile_do_not_retry") from exc
    if not isinstance(response_value, dict):
        raise CreateError("v32_create_response_invalid_reconcile_do_not_retry")
    api_run_id = live_release.extract_jobs_api_run_id(response_value)
    if re.fullmatch(r"ft-run-[0-9a-f]{8}", api_run_id) is None:
        raise CreateError("v32_create_identity_invalid_reconcile_do_not_retry")
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "CREATE_ACCEPTED_WATCHDOG_HANDOFF_REQUIRED",
        "api_run_id": api_run_id,
        "http_status": status,
        "server_title": TITLE,
        "server_run_dir": RUN_DIR,
        "request_sha256": request_sha256(),
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "watchdog_handoff_complete": False,
        "qualification_launch_authorized": False,
        "scored_launch_authorized": False,
        "protected_content_included": False,
    }
    result["receipt_sha256"] = crypto.digest_without(result, "receipt_sha256")
    result_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        with result_path.open("x") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
    except FileExistsError as exc:
        raise CreateError("v32_result_race_reconcile_do_not_retry") from exc
    return result


def build_held() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v32-create-wrapper-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "server_title": TITLE,
        "server_run_dir": RUN_DIR,
        "request_sha256": request_sha256(),
        "required_active_dedicated_nodes": 0,
        "required_active_dedicated_gpus": 0,
        "planned_nodes_after_create": 1,
        "planned_gpus_after_create": 8,
        "live_authorization_builder": (
            "evals.fleet.glm53_dedicated_v32_live_authorization_v1"
        ),
        "sfs_observer_pod_name": live_authorization.SFS_OBSERVER_POD_NAME,
        "sfs_observer_pod_uid": live_authorization.SFS_OBSERVER_POD_UID,
        "sfs_observer_mount_path": live_authorization.SFS_OBSERVER_MOUNT_PATH,
        "control_result_path": live_authorization.CONTROL_RESULT_PATH,
        "priority_class": payload()["priority_class"],
        "preemption_policy": "Never",
        "single_live_observation_required": True,
        "generation_exact_parity_entrypoint_required": True,
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
        raise CreateError("v32_cli_sfs_identity_invalid")
    try:
        authorization = json.loads(Path(args.authorization).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CreateError("v32_cli_authorization_invalid") from exc
    if not isinstance(authorization, dict):
        raise CreateError("v32_cli_authorization_invalid")
    create_once(authorization, result_path=Path(args.result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
