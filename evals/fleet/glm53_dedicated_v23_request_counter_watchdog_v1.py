"""Request-counter-only 600-second idle watchdog for a UID-bound GLM v23 server."""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto

SCHEMA = "fleet-glm53-dedicated-v23-request-counter-watchdog-active-v1"
IMPLEMENTATION_ID = "glm53-v23-sglang-request-counter-watchdog-v1"
METRIC = "sglang_num_requests_total"
IDLE_RELEASE_SECONDS = 600
RELEASE_ROUTE = "DELETE /v1/runs/{api_run_id}"
COUNTER = re.compile(
    r'^sglang:num_requests_total\{model_name="glm-5\.3"\}\s+([0-9]+(?:\.[0-9]+)?)$', re.M
)


class WatchdogError(RuntimeError):
    """The request-counter watchdog failed closed."""


def source_sha256() -> str:
    """Bind an active watcher receipt to the exact loaded implementation bytes."""

    return crypto.sha256(Path(__file__).read_bytes())


def build_active_receipt(
    binding: dict[str, Any], *, watcher_job_uid: str, watcher_pod_uid: str
) -> dict[str, Any]:
    """Describe a running UID-bound watcher without any protected content."""

    required = {
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
    if set(binding) != required or not binding["api_run_id"].startswith("ft-run-"):
        raise WatchdogError("watchdog_server_binding_invalid")
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "ACTIVE_UID_BOUND",
        "server_binding": binding,
        "metric": METRIC,
        "idle_release_seconds": IDLE_RELEASE_SECONDS,
        "health_or_process_liveness_refreshes": False,
        "model_request_counter_growth_refreshes": True,
        "release_via_jobs_api": True,
        "release_route": RELEASE_ROUTE,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_module": ("evals.fleet.glm53_dedicated_v23_request_counter_watchdog_v1"),
        "implementation_sha256": source_sha256(),
        "watcher_job_uid": watcher_job_uid,
        "watcher_pod_uid": watcher_pod_uid,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def parse_counter(metrics: str) -> int:
    match = COUNTER.search(metrics)
    if match is None:
        raise WatchdogError("request_counter_absent")
    value = float(match.group(1))
    if not value.is_integer() or value < 0:
        raise WatchdogError("request_counter_invalid")
    return int(value)


def read_counter_http(
    origin: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> int:
    """Read only the local server metrics surface, never a health/liveness signal."""

    url = origin.rstrip("/") + "/metrics"
    request = urllib.request.Request(url, method="GET")
    try:
        with opener(request, timeout=10) as response:
            if response.status != 200:
                raise WatchdogError("metrics_http_status_invalid")
            payload = response.read().decode("utf-8")
    except (OSError, UnicodeError, urllib.error.URLError) as exc:
        raise WatchdogError("metrics_read_failed") from exc
    return parse_counter(payload)


def release_run_http(
    *,
    api_base: str,
    bearer_token: str,
    api_run_id: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> None:
    """Release the one exact server through the Fleet Jobs API."""

    if not bearer_token or not api_run_id.startswith("ft-run-"):
        raise WatchdogError("jobs_api_release_identity_invalid")
    url = api_base.rstrip("/") + "/v1/runs/" + urllib.parse.quote(api_run_id, safe="")
    request = urllib.request.Request(
        url,
        method="DELETE",
        headers={"Authorization": f"Bearer {bearer_token}", "Accept": "application/json"},
    )
    try:
        with opener(request, timeout=30) as response:
            if response.status not in {200, 202, 204}:
                raise WatchdogError("jobs_api_release_http_status_invalid")
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise WatchdogError("jobs_api_release_failed") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise WatchdogError("jobs_api_release_failed") from exc


def advance(state: dict[str, Any], *, counter: int, now: float) -> dict[str, Any]:
    previous = state.get("counter")
    if not isinstance(previous, int) or previous < 0 or counter < previous:
        raise WatchdogError("request_counter_regressed_or_state_invalid")
    if counter > previous:
        return {"counter": counter, "last_model_request_at": now}
    return dict(state)


def should_release(state: dict[str, Any], *, now: float) -> bool:
    last = state.get("last_model_request_at")
    if not isinstance(last, (int, float)) or last > now:
        raise WatchdogError("last_model_request_time_invalid")
    return now - float(last) >= IDLE_RELEASE_SECONDS


def watch(
    *,
    api_run_id: str,
    initial_counter: int,
    ready_at: float,
    read_counter: Callable[[], int],
    release_via_jobs_api: Callable[[str], None],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 5,
) -> str:
    """Release exactly once after 600 seconds without request-counter growth."""

    if not api_run_id.startswith("ft-run-") or initial_counter < 0 or poll_seconds <= 0:
        raise WatchdogError("watchdog_identity_or_poll_invalid")
    state = {"counter": initial_counter, "last_model_request_at": ready_at}
    while True:
        now = clock()
        state = advance(state, counter=read_counter(), now=now)
        if should_release(state, now=now):
            release_via_jobs_api(api_run_id)
            return "RELEASED_IDLE"
        sleep(poll_seconds)


def watch_http(
    *,
    binding: dict[str, Any],
    bearer_token: str,
    ready_at: float,
    api_base: str = "https://api.ft.flt.build",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 5,
) -> str:
    """Run the concrete metrics-to-Jobs-API release path for one bound server."""

    origin = binding.get("service_origin")
    api_run_id = binding.get("api_run_id")
    if not isinstance(origin, str) or not isinstance(api_run_id, str):
        raise WatchdogError("watchdog_server_binding_invalid")
    initial_counter = read_counter_http(origin)
    return watch(
        api_run_id=api_run_id,
        initial_counter=initial_counter,
        ready_at=ready_at,
        read_counter=lambda: read_counter_http(origin),
        release_via_jobs_api=lambda run_id: release_run_http(
            api_base=api_base,
            bearer_token=bearer_token,
            api_run_id=run_id,
        ),
        clock=clock,
        sleep=sleep,
        poll_seconds=poll_seconds,
    )
