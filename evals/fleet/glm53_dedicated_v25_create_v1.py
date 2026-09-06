"""Create-once GLM v25 server request with exact response identity parsing."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v24_server_v1 as v24
from evals.fleet import glm53_dedicated_v24_watchdog_live_release_v1 as live_release

SCHEMA = "fleet-glm53-dedicated-v25-create-authorization-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v25-create-result-v1"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v25"
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v25"
READY_PATH = RUN_DIR + "/READY.json"
READY_SCHEMA = "fleet-glm53-dedicated-v25-application-ready-v1"
API_URL = "https://api.ft.flt.build/v1/runs"
AUTH_MAX_AGE_SECONDS = 60
CONTROL_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v25-create-control"
AUTHORIZATION_PATH = CONTROL_DIR + "/CREATE-AUTHORIZED.json"
RESULT_PATH = CONTROL_DIR + "/CREATED.json"

AUTH_KEYS = {
    "schema_version",
    "status",
    "observed_at_epoch",
    "server_title",
    "server_run_dir",
    "request_sha256",
    "preview_http_status",
    "jobs_api_title_matches",
    "jobs_api_run_dir_matches",
    "kubernetes_identity_or_remnant_matches",
    "sfs_run_dir_absent",
    "control_result_absent",
    "active_dedicated_nodes",
    "active_dedicated_gpus",
    "planned_nodes_after_create",
    "planned_gpus_after_create",
    "priority_class",
    "preemption_policy",
    "server_launch_authorized",
    "watchdog_handoff_required_immediately",
    "qualification_launch_authorized",
    "scored_launch_authorized",
    "api_mutation_calls",
    "protected_content_included",
    "receipt_sha256",
}

SERVED_ID = v24.SERVED_ID
MODEL_REVISION = v24.MODEL_REVISION
CONTEXT_LENGTH = v24.CONTEXT_LENGTH


class CreateError(RuntimeError):
    """The create-once GLM v25 server gate failed closed."""


ServerPlanError = CreateError


def _observer_source() -> str:
    source = v24.READY_OBSERVER
    replacements = (
        (repr(v24.READY_SCHEMA), repr(READY_SCHEMA)),
        (repr(v24.READY_PATH), repr(READY_PATH)),
        (repr(v24.RUN_DIR), repr(RUN_DIR)),
        (repr(v24.TITLE), repr(TITLE)),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise CreateError("v25_observer_template_identity_invalid")
        source = source.replace(old, new)
    if v24.TITLE in source or v24.RUN_DIR in source:
        raise CreateError("v25_observer_stale_identity_present")
    return source


def validate_binding(binding: dict[str, Any]) -> None:
    expected_keys = {
        "server_title",
        "server_run_dir",
        "api_run_id",
        "rayjob_uid",
        "workload_uid",
        "head_pod_uid",
        "service_uid",
        "service_origin",
        "served_id",
        "model_revision",
        "context_length",
    }
    candidate = dict(binding)
    candidate["server_title"] = v24.TITLE
    candidate["server_run_dir"] = v24.RUN_DIR
    try:
        v24.validate_binding(candidate)
    except v24.ServerPlanError as exc:
        raise CreateError("v25_server_binding_invalid") from exc
    if (
        set(binding) != expected_keys
        or binding.get("server_title") != TITLE
        or binding.get("server_run_dir") != RUN_DIR
    ):
        raise CreateError("v25_server_binding_invalid")


def payload() -> dict[str, Any]:
    value = copy.deepcopy(v24.payload())
    value["title"] = TITLE
    value["run_dir"] = RUN_DIR
    value["env"]["GLM53_RUN_DIR"] = RUN_DIR
    value["command"] = v24.render_server_command(observer_source=_observer_source())
    return value


def validate_payload(value: dict[str, Any]) -> None:
    expected = payload()
    encoded = crypto.canonical_json(value).decode("utf-8")
    if (
        value != expected
        or value.get("title") != TITLE
        or value.get("run_dir") != RUN_DIR
        or value.get("env", {}).get("GLM53_RUN_DIR") != RUN_DIR
        or v24.TITLE in encoded
        or v24.RUN_DIR in encoded
        or TITLE not in _observer_source()
        or RUN_DIR not in _observer_source()
        or READY_PATH not in _observer_source()
    ):
        raise CreateError("v25_payload_identity_invalid")


def request_sha256() -> str:
    value = payload()
    validate_payload(value)
    return crypto.sha256(crypto.canonical_json(value))


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
        or not isinstance(value.get("active_dedicated_nodes"), int)
        or isinstance(value.get("active_dedicated_nodes"), bool)
        or not 0 <= value["active_dedicated_nodes"] <= 1
        or not isinstance(value.get("active_dedicated_gpus"), int)
        or isinstance(value.get("active_dedicated_gpus"), bool)
        or not 0 <= value["active_dedicated_gpus"] <= 8
        or value.get("planned_nodes_after_create") != value["active_dedicated_nodes"] + 1
        or value.get("planned_nodes_after_create") > 2
        or value.get("planned_gpus_after_create") != value["active_dedicated_gpus"] + 8
        or value.get("planned_gpus_after_create") > 16
        or value.get("priority_class") != v24.PRIORITY_CLASS
        or value.get("preemption_policy") != v24.PREEMPTION_POLICY
        or value.get("server_launch_authorized") is not True
        or value.get("watchdog_handoff_required_immediately") is not True
        or value.get("qualification_launch_authorized") is not False
        or value.get("scored_launch_authorized") is not False
        or value.get("api_mutation_calls") != 0
        or value.get("protected_content_included") is not False
        or value.get("receipt_sha256") != crypto.digest_without(value, "receipt_sha256")
    ):
        raise CreateError("v25_create_authorization_invalid")


def create_once(
    authorization: dict[str, Any],
    *,
    result_path: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Perform the single POST; an indeterminate response must never be retried."""

    validate_authorization(authorization)
    if result_path.exists():
        raise CreateError("v25_result_already_exists_reconcile_do_not_retry")
    token = os.environ.get("FLEET_API_KEY", "")
    if not token:
        raise CreateError("v25_jobs_api_credential_absent")
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
        raise CreateError("v25_create_response_unknown_reconcile_do_not_retry") from exc
    if status != 202:
        raise CreateError("v25_create_http_status_invalid_reconcile_do_not_retry")
    try:
        response_value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CreateError("v25_create_response_invalid_reconcile_do_not_retry") from exc
    if not isinstance(response_value, dict):
        raise CreateError("v25_create_response_invalid_reconcile_do_not_retry")
    api_run_id = live_release.extract_jobs_api_run_id(response_value)
    if re.fullmatch(r"ft-run-[0-9a-f]{8}", api_run_id) is None:
        raise CreateError("v25_create_identity_invalid_reconcile_do_not_retry")
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
    try:
        result_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "w") as handle:
            json.dump(result, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
    except OSError as exc:
        raise CreateError("v25_result_persist_failed_reconcile_do_not_retry") from exc
    return result


def build_held() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": "fleet-glm53-dedicated-v25-create-wrapper-held-v1",
        "status": "PASSED_HELD_NO_LAUNCH",
        "server_title": TITLE,
        "server_run_dir": RUN_DIR,
        "request_sha256": request_sha256(),
        "response_identity_parser": (
            "evals.fleet.glm53_dedicated_v24_watchdog_live_release_v1."
            "extract_jobs_api_run_id"
        ),
        "create_response_name_field_supported": True,
        "expected_create_http_status": 202,
        "unknown_response_requires_reconciliation_without_retry": True,
        "cli_authorization_path": AUTHORIZATION_PATH,
        "cli_result_path": RESULT_PATH,
        "control_result_absence_required": True,
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
        raise CreateError("v25_cli_sfs_identity_invalid")
    authorization_path = Path(args.authorization)
    if not authorization_path.is_file():
        raise CreateError("v25_cli_authorization_absent")
    try:
        authorization = json.loads(authorization_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CreateError("v25_cli_authorization_invalid") from exc
    if not isinstance(authorization, dict):
        raise CreateError("v25_cli_authorization_invalid")
    create_once(authorization, result_path=Path(args.result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
