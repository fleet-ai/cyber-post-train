"""Fresh75 heldout queue tests are offline or use a GET-only mock transport."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evals.fleet import fresh75_holdout_queue as queue_module

QUEUE_PATH = queue_module.REPO_ROOT / "configs/evaluation/qwen38-fresh75-fleet-dev17-queue-v1.json"


def test_checked_in_queue_is_complete_and_inert():
    result = queue_module.validate_queue(QUEUE_PATH)
    assert result == {
        **result,
        "status": "valid_mutation_free_queue",
        "tasks": 17,
        "base_controls": 1,
        "accepted_sft_arms": 7,
        "pending_training_arms": 1,
        "launch_authorized": False,
    }
    queue = json.loads(QUEUE_PATH.read_text())
    assert {arm["arm_id"] for arm in queue["accepted_sft_arms"]} == queue_module.ACCEPTED_ARMS
    assert {arm["arm_id"] for arm in queue["pending_training_arms"]} == {"b8-lr1e5-e4"}
    assert all(arm["model_revision"] is None for arm in queue["accepted_sft_arms"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["failure_budget"].update(remaining=1),
        lambda value: value["accepted_sft_arms"][0]["checkpoint"].update(optimizer_step=999),
        lambda value: value["accepted_sft_arms"][0].update(model_revision="a" * 40),
        lambda value: value["pending_training_arms"][0].update(status="accepted"),
        lambda value: value["retry_policy"].update(automatic_retry=True),
    ],
)
def test_queue_drift_fails_closed(tmp_path, mutate):
    value = json.loads(QUEUE_PATH.read_text())
    mutate(value)
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        queue_module.validate_queue(path)


def test_live_check_uses_only_get(monkeypatch):
    queue = json.loads(QUEUE_PATH.read_text())
    census = json.loads(
        (queue_module.REPO_ROOT / queue["provenance"]["get_only_census"]["path"]).read_text()
    )
    expected_bindings = {
        row["task_version_id"]: row["binding_sha256"] for row in census["task_bindings"]
    }
    base = queue["base_control"]
    observed_methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        observed_methods.append(request.method)
        assert request.method == "GET"
        if request.url.path == "/v1/account":
            return httpx.Response(
                200,
                json={
                    "team_name": "fleet",
                    "team_id": "a1025f0b-ad67-49fc-a023-51800ab43e84",
                },
            )
        if request.url.path.startswith("/v1/tasks/"):
            return httpx.Response(200, json={"read_only_fixture": True})
        if request.url.path == "/v1/sessions":
            return httpx.Response(200, json={"sessions": [], "has_more": False})
        if request.url.path == "/fleet/v1/model-catalog":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            **base["route"]["catalog"],
                            "id": base["route"]["served_id"],
                            "model_revision": base["model"]["revision"],
                            "routed": True,
                            "status": "ready",
                            "ready_replicas": 1,
                            "capabilities": ["tool_calling"],
                        }
                    ]
                },
            )
        if request.url.path == "/model_info":
            return httpx.Response(200, json=base["route"]["model_info"])
        if request.url.path == "/server_info":
            return httpx.Response(
                200,
                json={
                    **base["route"]["server_info"],
                    "served_model_name": base["route"]["served_id"],
                },
            )
        raise AssertionError(f"unexpected GET {request.url}")

    def fake_task_binding(client, task, selected):
        response = client.get(
            f"https://orchestrator.fleetai.com/v1/tasks/{task['task_key']}",
            params={"version_id": task["task_version_id"]},
        )
        response.raise_for_status()
        return (
            {
                "version_id": task["task_version_id"],
                "cyber_contract": {"verifier_contract": "3.0.0"},
            },
            {},
            {},
        )

    monkeypatch.setattr(queue_module.rollout_worker, "_task_binding", fake_task_binding)
    monkeypatch.setattr(
        queue_module,
        "_binding_digest",
        lambda binding: expected_bindings[binding[0]["version_id"]],
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = queue_module.live_check(QUEUE_PATH, client=client)
    assert result["status"] == "get_only_live_check_passed"
    assert result["external_methods_used"] == ["GET"]
    assert result["external_mutation_performed"] is False
    assert result["launch_authorized"] is False
    assert set(observed_methods) == {"GET"}


def test_checker_has_no_http_mutation_callsite():
    source = Path(queue_module.__file__).read_text()
    for forbidden in ('"POST"', '"PUT"', '"PATCH"', '"DELETE"', ".post(", ".put(", ".delete("):
        assert forbidden not in source
