"""Create-once GLM v34 server with response-body-independent reconciliation."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v33_create_v1 as engine
from evals.fleet import glm53_dedicated_v34_live_authorization_v1 as live_authorization

SCHEMA = "fleet-glm53-dedicated-v34-create-authorization-v1"
RESULT_SCHEMA = "fleet-glm53-dedicated-v34-create-result-v1"
TITLE = "chris-cyber-evalserve-glm53-tp8-a-v34"
API_NAME = "glm53-tp8-v34"
API_RUN_ID_RE = re.compile(r"glm53-tp8-v34-[0-9a-f]{8}")
RELEASABLE_RUN_ID_RE = re.compile(r"(?:glm53-tp8-v34|ft-run)-[0-9a-f]{8}")
RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v34"
READY_PATH = RUN_DIR + "/READY.json"
READY_SCHEMA = "fleet-glm53-dedicated-v34-application-ready-v1"
CONTROL_DIR = RUN_DIR + "-create-control"
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
POST_CREATE_ATTEMPTS = 60
POST_CREATE_POLL_SECONDS = 1.0
_LOCK = threading.Lock()


class ReconciliationBackend(Protocol):
    """The score-blind Jobs API surface used around the one POST."""

    def list_runs(self) -> tuple[list[dict[str, Any]], int]: ...

    def get_run(self, api_run_id: str) -> dict[str, Any] | None: ...

    def release_run(self, api_run_id: str) -> int: ...


class SystemReconciliationBackend(live_authorization.SystemBackend):
    """Use exhaustive list/exact GET and a bounded exact release."""

    def release_run(self, api_run_id: str) -> int:
        if RELEASABLE_RUN_ID_RE.fullmatch(api_run_id) is None:
            raise CreateError("v34_post_create_release_identity_invalid")
        status, _ = self._request(  # noqa: SLF001
            "DELETE", "/v1/runs/" + urllib.parse.quote(api_run_id, safe="")
        )
        if status not in {200, 202, 204, 404}:
            raise CreateError("v34_post_create_release_failed")
        for _ in range(12):
            if self.get_run(api_run_id) is None:
                return status
            time.sleep(5)
        raise CreateError("v34_post_create_release_absence_unconfirmed")


@contextmanager
def bound_engine() -> Iterator[None]:
    if not _LOCK.acquire(blocking=False):
        raise CreateError("v34_create_engine_already_bound")
    values = {
        "SCHEMA": SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "TITLE": TITLE,
        "API_NAME": API_NAME,
        "RUN_DIR": RUN_DIR,
        "READY_PATH": READY_PATH,
        "READY_SCHEMA": READY_SCHEMA,
        "CONTROL_DIR": CONTROL_DIR,
        "AUTHORIZATION_PATH": AUTHORIZATION_PATH,
        "RESULT_PATH": RESULT_PATH,
        "API_RUN_ID_RE": API_RUN_ID_RE,
        "live_authorization": live_authorization,
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


def payload() -> dict[str, Any]:
    with bound_engine():
        return engine.payload()


def validate_payload(value: dict[str, Any]) -> None:
    with bound_engine():
        engine.validate_payload(value)


def request_sha256() -> str:
    return crypto.sha256(crypto.canonical_json(payload()))


def validate_binding(value: dict[str, Any]) -> None:
    keys = {
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
    try:
        uids_valid = all(
            uuid.UUID(str(value[key])).int != 0
            for key in ("rayjob_uid", "workload_uid", "head_pod_uid", "service_uid")
        )
    except (KeyError, ValueError):
        uids_valid = False
    if (
        set(value) != keys
        or value.get("server_title") != TITLE
        or value.get("server_run_dir") != RUN_DIR
        or API_RUN_ID_RE.fullmatch(str(value.get("api_run_id", ""))) is None
        or not uids_valid
        or not str(value.get("service_origin", "")).startswith("http://")
        or value.get("served_id") != SERVED_ID
        or value.get("model_revision") != MODEL_REVISION
        or value.get("context_length") != CONTEXT_LENGTH
    ):
        raise CreateError("v34_server_binding_invalid")


def validate_authorization(value: dict[str, Any]) -> None:
    with bound_engine():
        try:
            engine.validate_authorization(value)
        except CreateError as exc:
            raise CreateError("v34_create_authorization_invalid") from exc


def _candidate_dicts(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = [value]
    for key in ("run", "config", "request"):
        child = value.get(key)
        if isinstance(child, dict):
            result.append(child)
            nested = child.get("config")
            if isinstance(nested, dict):
                result.append(nested)
    return result


def _values(value: Mapping[str, Any], key: str) -> set[str]:
    return {
        item
        for candidate in _candidate_dicts(value)
        if isinstance((item := candidate.get(key)), str)
    }


def _top_level_run_id(row: Mapping[str, Any]) -> str | None:
    value = row.get("name")
    return value if isinstance(value, str) and RELEASABLE_RUN_ID_RE.fullmatch(value) else None


def _matches_generation_identity(row: Mapping[str, Any]) -> bool:
    api_run_id = _top_level_run_id(row)
    return (
        api_run_id is not None
        and API_RUN_ID_RE.fullmatch(api_run_id) is not None
        or TITLE in _values(row, "title")
        or RUN_DIR in _values(row, "run_dir")
    )


def _safe_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if not _matches_generation_identity(row):
            continue
        titles = sorted(_values(row, "title"))
        run_dirs = sorted(_values(row, "run_dir"))
        result.append(
            {
                "api_run_id": _top_level_run_id(row),
                "title_state": "EXACT"
                if titles == [TITLE]
                else "ABSENT"
                if not titles
                else "OTHER",
                "run_dir_state": (
                    "EXACT" if run_dirs == [RUN_DIR] else "ABSENT" if not run_dirs else "OTHER"
                ),
                "status": str(row.get("status", "")).upper(),
            }
        )
    return sorted(result, key=lambda row: str(row["api_run_id"]))


def _validate_exact_created_row(row: Mapping[str, Any], api_run_id: str) -> None:
    titles = _values(row, "title")
    run_dirs = _values(row, "run_dir")
    identity_values = {
        item
        for candidate in _candidate_dicts(row)
        for key in ("id", "run_id", "name")
        if isinstance((item := candidate.get(key)), str)
    }
    exact_ids = {item for item in identity_values if API_RUN_ID_RE.fullmatch(item)}
    if (
        row.get("name") != api_run_id
        or exact_ids != {api_run_id}
        or identity_values - {api_run_id, API_NAME}
        or run_dirs != {RUN_DIR}
        or titles not in (set(), {TITLE})
        or str(row.get("status", "")).upper() not in {"SUBMITTED", "SUSPENDED", "RUNNING"}
    ):
        raise CreateError("v34_post_create_identity_ambiguous")


def _release_candidates(backend: ReconciliationBackend, candidates: list[dict[str, Any]]) -> None:
    run_ids = {_top_level_run_id(row) for row in candidates}
    if None in run_ids or not run_ids:
        for api_run_id in sorted(item for item in run_ids if item is not None):
            backend.release_run(api_run_id)
        raise CreateError("v34_post_create_ambiguity_not_fully_releasable")
    for api_run_id in sorted(run_ids):
        assert api_run_id is not None
        backend.release_run(api_run_id)
    if any(backend.get_run(api_run_id) is not None for api_run_id in run_ids):
        raise CreateError("v34_post_create_release_absence_unconfirmed")


def _preflight_absent(backend: ReconciliationBackend) -> tuple[int, str]:
    rows, pages = backend.list_runs()
    projection = _safe_projection(rows)
    if not isinstance(pages, int) or isinstance(pages, bool) or pages < 1 or projection:
        raise CreateError("v34_post_create_identity_preflight_collision")
    return pages, crypto.sha256(crypto.canonical_json(projection))


def _reconcile_created(
    backend: ReconciliationBackend,
    *,
    sleep: Callable[[float], None],
) -> tuple[str, int, str]:
    for attempt in range(POST_CREATE_ATTEMPTS):
        rows, pages = backend.list_runs()
        candidates = [row for row in rows if _matches_generation_identity(row)]
        if not candidates:
            if attempt + 1 < POST_CREATE_ATTEMPTS:
                sleep(POST_CREATE_POLL_SECONDS)
            continue
        projection = _safe_projection(rows)
        try:
            if len(candidates) != 1:
                raise CreateError("v34_post_create_identity_ambiguous")
            api_run_id = _top_level_run_id(candidates[0])
            if api_run_id is None or API_RUN_ID_RE.fullmatch(api_run_id) is None:
                raise CreateError("v34_post_create_identity_ambiguous")
            _validate_exact_created_row(candidates[0], api_run_id)
            exact = backend.get_run(api_run_id)
            if exact is None:
                raise CreateError("v34_post_create_identity_ambiguous")
            _validate_exact_created_row(exact, api_run_id)
            return api_run_id, pages, crypto.sha256(crypto.canonical_json(projection))
        except CreateError:
            _release_candidates(backend, candidates)
            raise CreateError("v34_post_create_identity_ambiguous_released") from None
    raise CreateError("v34_post_create_identity_absent_reconcile_do_not_retry")


def create_once(
    authorization: dict[str, Any],
    *,
    result_path: Path,
    opener: Callable[..., Any] = urllib.request.urlopen,
    backend: ReconciliationBackend | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    validate_authorization(authorization)
    if result_path.exists():
        raise CreateError("v34_result_already_exists_reconcile_do_not_retry")
    token = os.environ.get("FLEET_API_KEY", "")
    if not token:
        raise CreateError("v34_jobs_api_credential_absent")
    active_backend = SystemReconciliationBackend() if backend is None else backend
    preflight_pages, preflight_snapshot = _preflight_absent(active_backend)
    request = urllib.request.Request(
        API_URL,
        data=crypto.canonical_json(payload()),
        method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with opener(request, timeout=30) as response:
            status = int(response.status)
    except (OSError, urllib.error.URLError):
        status = 0
    api_run_id, post_pages, post_snapshot = _reconcile_created(active_backend, sleep=sleep)
    if status != 202:
        _release_candidates(
            active_backend,
            [{"name": api_run_id, "title": TITLE, "run_dir": RUN_DIR, "status": "RUNNING"}],
        )
        raise CreateError("v34_create_http_status_invalid_released")
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "CREATE_ACCEPTED_RECONCILED_WATCHDOG_HANDOFF_REQUIRED",
        "api_run_id": api_run_id,
        "http_status": status,
        "server_title": TITLE,
        "server_api_name": API_NAME,
        "server_run_dir": RUN_DIR,
        "request_sha256": request_sha256(),
        "authorization_receipt_sha256": authorization["receipt_sha256"],
        "pre_create_jobs_api_pages": preflight_pages,
        "pre_create_identity_snapshot_sha256": preflight_snapshot,
        "post_create_jobs_api_pages": post_pages,
        "post_create_identity_snapshot_sha256": post_snapshot,
        "create_response_body_authoritative": False,
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
    except (FileExistsError, OSError):
        _release_candidates(
            active_backend,
            [{"name": api_run_id, "title": TITLE, "run_dir": RUN_DIR, "status": "RUNNING"}],
        )
        raise CreateError("v34_result_write_failed_server_released") from None
    return result


def build_held() -> dict[str, Any]:
    with bound_engine():
        value = engine.build_held()
    value.update(
        {
            "schema_version": "fleet-glm53-dedicated-v34-create-wrapper-held-v1",
            "response_body_identity_required": False,
            "exhaustive_pre_post_jobs_api_reconciliation_required": True,
            "ambiguous_post_create_identity_release_required": True,
        }
    )
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", default=AUTHORIZATION_PATH)
    parser.add_argument("--result-path", default=RESULT_PATH)
    args = parser.parse_args(argv)
    if args.authorization != AUTHORIZATION_PATH or args.result_path != RESULT_PATH:
        raise CreateError("v34_cli_sfs_identity_invalid")
    try:
        authorization = json.loads(Path(args.authorization).read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CreateError("v34_cli_authorization_invalid") from exc
    if not isinstance(authorization, dict):
        raise CreateError("v34_cli_authorization_invalid")
    create_once(authorization, result_path=Path(args.result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
