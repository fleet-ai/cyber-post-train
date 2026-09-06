"""Request-counter-only 600-second idle watchdog for a UID-bound GLM v23 server."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
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
METRIC_CONTRACT_COMMIT = "97c6978369ac1e04c91fcc01c98acc25129a6000"
METRIC_CONTRACT_SOURCE = (
    "https://github.com/sgl-project/sglang/blob/"
    f"{METRIC_CONTRACT_COMMIT}/docs/docs/references/production_metrics.mdx"
)
ACTIVITY_METRICS = (
    "sglang:num_requests_total",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
)
ACTIVITY_LINE = re.compile(
    r"^(?P<metric>sglang:(?:num_requests_total|num_running_reqs|num_queue_reqs))"
    r"\{(?P<labels>[^{}]*)\}\s+"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)$"
)
LABEL = re.compile(r'(?P<name>[A-Za-z_][A-Za-z0-9_]*)="(?P<value>(?:[^"\\]|\\.)*)"')


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
        "active_request_or_queue_refreshes": True,
        "active_request_metrics": ["sglang:num_running_reqs", "sglang:num_queue_reqs"],
        "metric_contract_commit": METRIC_CONTRACT_COMMIT,
        "metric_contract_source": METRIC_CONTRACT_SOURCE,
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


def parse_activity(metrics: str) -> dict[str, int]:
    totals = {metric: 0 for metric in ACTIVITY_METRICS}
    matched = {metric: 0 for metric in ACTIVITY_METRICS}
    labelsets: dict[str, set[tuple[tuple[str, str], ...]]] = {
        metric: set() for metric in ACTIVITY_METRICS
    }
    for line in metrics.splitlines():
        stripped = line.strip()
        match = ACTIVITY_LINE.fullmatch(stripped)
        if match is None:
            if stripped.startswith(ACTIVITY_METRICS):
                raise WatchdogError("activity_metric_line_invalid")
            continue
        metric = match.group("metric")
        labels_text = match.group("labels")
        labels: dict[str, str] = {}
        position = 0
        while position < len(labels_text):
            label = LABEL.match(labels_text, position)
            if label is None or label.group("name") in labels:
                raise WatchdogError("request_counter_labels_invalid")
            try:
                labels[label.group("name")] = json.loads(f'"{label.group("value")}"')
            except json.JSONDecodeError as exc:
                raise WatchdogError("request_counter_labels_invalid") from exc
            position = label.end()
            if position == len(labels_text):
                break
            if labels_text[position] != ",":
                raise WatchdogError("request_counter_labels_invalid")
            position += 1
        if labels.get("model_name") != "glm-5.3":
            continue
        canonical = tuple(sorted(labels.items()))
        if canonical in labelsets[metric]:
            raise WatchdogError("activity_metric_series_duplicated")
        labelsets[metric].add(canonical)
        value = float(match.group("value"))
        if not value.is_integer() or value < 0:
            raise WatchdogError("request_counter_invalid")
        totals[metric] += int(value)
        matched[metric] += 1
    if any(matched[metric] == 0 for metric in ACTIVITY_METRICS):
        raise WatchdogError("activity_metric_family_absent")
    return {
        "requests": totals["sglang:num_requests_total"],
        "running": totals["sglang:num_running_reqs"],
        "queued": totals["sglang:num_queue_reqs"],
    }


def parse_counter(metrics: str) -> int:
    return parse_activity(metrics)["requests"]


def read_activity_http(
    origin: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, int]:
    """Read exact request and in-flight gauges, never a health/liveness signal."""

    url = origin.rstrip("/") + "/metrics"
    request = urllib.request.Request(url, method="GET")
    try:
        with opener(request, timeout=10) as response:
            if response.status != 200:
                raise WatchdogError("metrics_http_status_invalid")
            payload = response.read().decode("utf-8")
    except (OSError, UnicodeError, urllib.error.URLError) as exc:
        raise WatchdogError("metrics_read_failed") from exc
    return parse_activity(payload)


def read_counter_http(
    origin: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> int:
    return read_activity_http(origin, opener=opener)["requests"]


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


def run_absent_http(
    *,
    api_base: str,
    bearer_token: str,
    api_run_id: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> bool:
    if not bearer_token or not api_run_id.startswith("ft-run-"):
        raise WatchdogError("jobs_api_absence_identity_invalid")
    url = api_base.rstrip("/") + "/v1/runs/" + urllib.parse.quote(api_run_id, safe="")
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {bearer_token}", "Accept": "application/json"},
    )
    try:
        with opener(request, timeout=30) as response:
            if response.status == 200:
                return False
            raise WatchdogError("jobs_api_absence_http_status_invalid")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return True
        raise WatchdogError("jobs_api_absence_check_failed") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise WatchdogError("jobs_api_absence_check_failed") from exc


def release_with_bounded_confirmation(
    api_run_id: str,
    *,
    release: Callable[[str], None],
    absent: Callable[[str], bool],
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 3,
    polls_per_attempt: int = 12,
    poll_seconds: float = 5,
) -> None:
    """Release only the bound run and require Jobs API absence within a bounded grace."""

    if attempts < 1 or polls_per_attempt < 1 or poll_seconds <= 0:
        raise WatchdogError("release_retry_policy_invalid")
    for attempt in range(attempts):
        try:
            release(api_run_id)
            for _ in range(polls_per_attempt):
                if absent(api_run_id):
                    return
                sleep(poll_seconds)
        except WatchdogError:
            if attempt + 1 == attempts:
                raise
        if attempt + 1 < attempts:
            sleep(poll_seconds)
    raise WatchdogError("jobs_api_release_absence_unconfirmed")


def release_run_and_confirm_http(*, api_base: str, bearer_token: str, api_run_id: str) -> None:
    release_with_bounded_confirmation(
        api_run_id,
        release=lambda run_id: release_run_http(
            api_base=api_base, bearer_token=bearer_token, api_run_id=run_id
        ),
        absent=lambda run_id: run_absent_http(
            api_base=api_base, bearer_token=bearer_token, api_run_id=run_id
        ),
    )


def advance(
    state: dict[str, Any],
    *,
    counter: int,
    running: int,
    queued: int,
    now: float,
) -> dict[str, Any]:
    previous = state.get("counter")
    if (
        not isinstance(previous, int)
        or previous < 0
        or counter < previous
        or running < 0
        or queued < 0
    ):
        raise WatchdogError("request_counter_regressed_or_state_invalid")
    if counter > previous or running > 0 or queued > 0:
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
    read_activity: Callable[[], dict[str, int]],
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
        activity = read_activity()
        if set(activity) != {"requests", "running", "queued"} or any(
            not isinstance(value, int) for value in activity.values()
        ):
            raise WatchdogError("activity_snapshot_invalid")
        state = advance(
            state,
            counter=activity["requests"],
            running=activity["running"],
            queued=activity["queued"],
            now=now,
        )
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
    initial = read_activity_http(origin)
    return watch(
        api_run_id=api_run_id,
        initial_counter=initial["requests"],
        ready_at=ready_at,
        read_activity=lambda: read_activity_http(origin),
        release_via_jobs_api=lambda run_id: release_run_and_confirm_http(
            api_base=api_base,
            bearer_token=bearer_token,
            api_run_id=run_id,
        ),
        clock=clock,
        sleep=sleep,
        poll_seconds=poll_seconds,
    )


def _write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise WatchdogError("watchdog_receipt_collision")
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    temporary.chmod(0o400)
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise WatchdogError("watchdog_receipt_collision") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("watch", nargs="?")
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--active-receipt", type=Path, required=True)
    parser.add_argument("--ready-at-epoch", type=float, required=True)
    args = parser.parse_args()
    binding = json.loads(args.binding.read_text())
    terminal = args.active_receipt.with_name("TERMINAL.json")
    try:
        initial = read_activity_http(binding["service_origin"])
        receipt = build_active_receipt(
            binding,
            watcher_job_uid=os.environ.get("JOB_UID", ""),
            watcher_pod_uid=os.environ.get("POD_UID", ""),
        )
        receipt["initial_request_counter"] = initial["requests"]
        receipt["initial_running_requests"] = initial["running"]
        receipt["initial_queued_requests"] = initial["queued"]
        receipt["ready_at_epoch"] = args.ready_at_epoch
        receipt["terminal_receipt_required"] = True
        receipt["receipt_sha256"] = crypto.digest_without(receipt, "receipt_sha256")
        _write_once(args.active_receipt, receipt)
        status = watch(
            api_run_id=binding["api_run_id"],
            initial_counter=initial["requests"],
            ready_at=args.ready_at_epoch,
            read_activity=lambda: read_activity_http(binding["service_origin"]),
            release_via_jobs_api=lambda run_id: release_run_and_confirm_http(
                api_base="https://api.ft.flt.build",
                bearer_token=os.environ.get("GH_TOKEN", ""),
                api_run_id=run_id,
            ),
            clock=time.time,
        )
        outcome: dict[str, Any] = {
            "schema_version": "fleet-glm53-dedicated-v23-watchdog-terminal-v1",
            "status": status + "_API_ABSENT",
            "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
            "active_receipt_sha256": receipt["receipt_sha256"],
            "release_route": RELEASE_ROUTE,
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "protected_content_included": False,
        }
        outcome["receipt_sha256"] = crypto.digest_without(outcome, "receipt_sha256")
        _write_once(terminal, outcome)
        return 0
    except Exception as exc:
        release_confirmed = False
        try:
            release_run_and_confirm_http(
                api_base="https://api.ft.flt.build",
                bearer_token=os.environ.get("GH_TOKEN", ""),
                api_run_id=str(binding.get("api_run_id", "")),
            )
            release_confirmed = True
        except WatchdogError:
            pass
        failure: dict[str, Any] = {
            "schema_version": "fleet-glm53-dedicated-v23-watchdog-terminal-v1",
            "status": (
                "FAILED_CLOSED_RELEASED_API_ABSENT"
                if release_confirmed
                else "FAILED_CLOSED_RELEASE_UNCONFIRMED"
            ),
            "server_binding_sha256": crypto.sha256(crypto.canonical_json(binding)),
            "reason": type(exc).__name__,
            "release_route": RELEASE_ROUTE,
            "fleet_task_instance_calls": 0,
            "fleet_session_calls": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "protected_content_included": False,
        }
        failure["receipt_sha256"] = crypto.digest_without(failure, "receipt_sha256")
        if not terminal.exists():
            _write_once(terminal, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
