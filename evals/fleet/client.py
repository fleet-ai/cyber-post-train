"""Small authenticated clients for Fleet jobs and the inference gateway."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from .models import DEFAULT_GATEWAY_MODEL, FLEET_TEAM_ID, JobBatch, validate_task_keys

ORCHESTRATOR_BASE_URL = "https://orchestrator.fleetai.com"
ANALYSIS_BASE_URL = "https://api.internal.fleet-platform.fleetai.com"
INFERENCE_BASE_URL = "https://inference.flt.build"


class FleetEvalError(RuntimeError):
    """A safe-to-display Fleet evaluation failure."""


def require_api_key() -> str:
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise FleetEvalError("FLEET_API_KEY is not set")
    return key


@contextmanager
def authenticated_client(
    *,
    api_key: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> Iterator[httpx.Client]:
    """Yield a client without ever putting the credential in an artifact."""

    key = api_key or require_api_key()
    with httpx.Client(
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=120,
        follow_redirects=False,
        transport=transport,
    ) as client:
        yield client


def _response_json(response: httpx.Response, *, operation: str) -> Any:
    try:
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise FleetEvalError(f"{operation} failed with HTTP {exc.response.status_code}") from None
    except ValueError:
        raise FleetEvalError(f"{operation} returned non-JSON data") from None


def verify_fleet_team(client: httpx.Client) -> dict[str, str]:
    response = client.get(f"{ORCHESTRATOR_BASE_URL}/v1/account")
    data = _response_json(response, operation="Fleet account preflight")
    if not isinstance(data, dict):
        raise FleetEvalError("Fleet account preflight returned an unexpected response")
    team_name = str(data.get("team_name") or "")
    team_id = str(data.get("team_id") or data.get("id") or "")
    if team_name != "fleet":
        raise FleetEvalError(f"FLEET_API_KEY belongs to team {team_name or '<unknown>'}, not fleet")
    if team_id and team_id != FLEET_TEAM_ID:
        raise FleetEvalError("Fleet account returned an unexpected team id")
    return {"team_name": team_name, "team_id": team_id or FLEET_TEAM_ID}


def verify_gateway_model(
    client: httpx.Client,
    *,
    gateway_model: str = DEFAULT_GATEWAY_MODEL,
) -> dict[str, str]:
    response = client.get(f"{INFERENCE_BASE_URL}/v1/models")
    data = _response_json(response, operation="inference model-catalog preflight")
    rows = data.get("data", []) if isinstance(data, dict) else []
    model_ids = {
        str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id") is not None
    }
    if gateway_model not in model_ids:
        raise FleetEvalError(f"inference model {gateway_model!r} was not found")
    return {"gateway_model": gateway_model, "status": "available"}


def _task_keys_from_job_detail(data: Any) -> tuple[str, ...]:
    if not isinstance(data, dict):
        raise FleetEvalError("source job detail returned an unexpected response")
    job_input = data.get("input")
    if not isinstance(job_input, dict):
        raise FleetEvalError("source job detail did not include immutable launch input")
    keys = job_input.get("task_keys")
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise FleetEvalError("source job launch input did not contain task_keys")
    try:
        return validate_task_keys(keys)
    except ValueError as exc:
        raise FleetEvalError(str(exc)) from None


def _task_keys_from_session_roster(data: Any) -> tuple[str, ...]:
    """Recover task keys from the server-owned roster of a legacy job."""

    if not isinstance(data, dict) or not isinstance(data.get("tasks"), list):
        raise FleetEvalError("source job session roster returned an unexpected response")
    keys: list[str] = []
    for group in data["tasks"]:
        task = group.get("task") if isinstance(group, dict) else None
        key = task.get("key") if isinstance(task, dict) else None
        if isinstance(key, str):
            keys.append(key)
    try:
        return validate_task_keys(keys)
    except ValueError as exc:
        raise FleetEvalError(str(exc)) from None


def fetch_source_task_keys(
    client: httpx.Client,
    source_job_id: str,
    *,
    expected_count: int = 160,
) -> tuple[str, ...]:
    """Read the immutable launch input rather than scraping session rows."""

    response = client.get(f"{ANALYSIS_BASE_URL}/v2/jobs/{source_job_id}")
    data = _response_json(response, operation="source job detail read")
    try:
        keys = _task_keys_from_job_detail(data)
    except FleetEvalError as detail_error:
        # Some historical jobs predate durable launch-input retention. Their
        # canonical session roster still binds groups to registered tasks.
        if not isinstance(data, dict) or data.get("input") is not None:
            raise detail_error
        roster_response = client.get(f"{ORCHESTRATOR_BASE_URL}/v1/sessions/job/{source_job_id}")
        roster = _response_json(roster_response, operation="source job session roster read")
        keys = _task_keys_from_session_roster(roster)
    if len(keys) != expected_count:
        raise FleetEvalError(f"source job has {len(keys)} tasks; expected {expected_count}")
    return keys


def launch_batch(client: httpx.Client, batch: JobBatch) -> dict[str, str]:
    if batch.planned_sessions > 6:
        raise FleetEvalError("refusing a job above the six-session cyber evaluation cap")
    response = client.post(
        f"{ORCHESTRATOR_BASE_URL}/v1/jobs",
        headers={"Idempotency-Key": batch.idempotency_key},
        json=batch.payload(),
    )
    data = _response_json(response, operation=f"launch {batch.name}")
    if not isinstance(data, dict) or not data.get("job_id"):
        raise FleetEvalError("job launch did not return job_id")
    return {"job_id": str(data["job_id"]), "name": batch.name}


def get_job(client: httpx.Client, job_id: str) -> dict[str, Any]:
    response = client.get(f"{ORCHESTRATOR_BASE_URL}/v1/jobs/{job_id}")
    data = _response_json(response, operation="job status read")
    if not isinstance(data, dict):
        raise FleetEvalError("job status returned an unexpected response")
    return data
