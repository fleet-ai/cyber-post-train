from __future__ import annotations

import httpx
import pytest

from evals.fleet.client import (
    FleetEvalError,
    _task_keys_from_session_roster,
    authenticated_client,
    fetch_source_task_keys,
    launch_batch,
    verify_fleet_team,
    verify_gateway_model,
)
from evals.fleet.models import DEFAULT_GATEWAY_MODEL, FLEET_TEAM_ID, build_plan


def _key(index: int = 1) -> str:
    return f"cysec1-2-demo-gen_blackbox-{index:024x}__blackbox_ctf_v1"


def test_preflights_and_source_job_resolution() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/account":
            return httpx.Response(200, json={"team_name": "fleet", "team_id": FLEET_TEAM_ID})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": DEFAULT_GATEWAY_MODEL}]})
        if request.url.path == "/v2/jobs/job-1":
            return httpx.Response(200, json={"id": "job-1", "input": {"task_keys": [_key()]}})
        raise AssertionError(request.url)

    with authenticated_client(
        api_key="not-a-real-key", transport=httpx.MockTransport(handler)
    ) as client:
        assert verify_fleet_team(client)["team_name"] == "fleet"
        assert verify_gateway_model(client)["status"] == "available"
        assert fetch_source_task_keys(client, "job-1", expected_count=1) == (_key(),)


def test_wrong_team_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"team_name": "other-team"})

    with (
        authenticated_client(
            api_key="not-a-real-key", transport=httpx.MockTransport(handler)
        ) as client,
        pytest.raises(FleetEvalError, match="not fleet"),
    ):
        verify_fleet_team(client)


def test_launch_uses_stable_idempotency_header() -> None:
    batch = build_plan([_key()]).batches[0]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Idempotency-Key"] == batch.idempotency_key
        assert request.url.path == "/v1/jobs"
        return httpx.Response(200, json={"job_id": "new-job"})

    with authenticated_client(
        api_key="not-a-real-key", transport=httpx.MockTransport(handler)
    ) as client:
        assert launch_batch(client, batch)["job_id"] == "new-job"


def test_http_errors_do_not_echo_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="secret-bearing upstream detail")

    with (
        authenticated_client(
            api_key="not-a-real-key", transport=httpx.MockTransport(handler)
        ) as client,
        pytest.raises(FleetEvalError) as caught,
    ):
        verify_fleet_team(client)

    assert "secret-bearing" not in str(caught.value)


def test_task_keys_from_session_roster() -> None:
    roster = {
        "tasks": [
            {"task": {"key": "cysec1-2-current-gen_demo__blackbox_ctf_v1"}},
            {"task": {"key": "cysec1-2-fira-gen_demo__blackbox_ctf_v1"}},
        ]
    }
    assert _task_keys_from_session_roster(roster) == (
        "cysec1-2-current-gen_demo__blackbox_ctf_v1",
        "cysec1-2-fira-gen_demo__blackbox_ctf_v1",
    )
