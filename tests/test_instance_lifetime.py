"""Synthetic Fleet HTTP responses; never provision or extend a real instance."""

import copy
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from evals.fleet import opencode_self_hosted as runtime


@pytest.fixture
def instance():
    created = datetime.now(UTC) - timedelta(seconds=30)
    return {
        "instance_id": "synthetic-instance",
        "team_id": runtime.FLEET_TEAM_ID,
        "status": "running",
        "terminated_at": None,
        "env_key": "synthetic-env",
        "version": "v1",
        "data_key": "synthetic-data",
        "data_version": "v2",
        "created_at": created.isoformat(),
        "expires_at": (created + timedelta(seconds=600)).isoformat(),
    }


def config():
    return {
        "run_id": "synthetic-run",
        "environment": {
            "id": "synthetic-env",
            "version": "v1",
            "data_id": "synthetic-data",
            "data_version": "v2",
            "ttl_seconds": 3600,
        },
        "harness": {"timeout_seconds": 1800},
    }


@pytest.mark.parametrize("already_sufficient", [False, True])
def test_lifetime_extends_once_to_original_deadline_and_reads_back(instance, already_sufficient):
    target = datetime.fromisoformat(instance["created_at"]) + timedelta(seconds=3600)
    if already_sufficient:
        instance["expires_at"] = (target + timedelta(seconds=60)).isoformat()
    calls = []
    observed = copy.deepcopy(instance)

    def handler(request):
        calls.append(request.method)
        if request.method == "POST":
            assert str(request.url).endswith("/synthetic-instance/extend_ttl")
            assert json.loads(request.content) == {"absolute_expires_at": target.isoformat()}
            observed["expires_at"] = target.isoformat()
        return httpx.Response(200, json=observed)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        receipt = runtime.ensure_instance_lifetime(client, config(), instance)
    assert calls == (["GET"] if already_sufficient else ["POST", "GET"])
    assert receipt["ttl_updated"] is not already_sufficient
    assert receipt["receipt_sha256"] == runtime.digest_without(receipt, "receipt_sha256")


@pytest.mark.parametrize("defect", ["owner", "version", "expired", "old", "future", "budget"])
def test_lifetime_invalid_identity_or_budget_never_mutates(instance, defect):
    settings = config()
    if defect == "owner":
        instance["team_id"] = "not-fleet"
    elif defect == "version":
        instance["data_version"] = "mutable-latest"
    elif defect == "expired":
        instance["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    elif defect == "old":
        instance["created_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    elif defect == "future":
        instance["created_at"] = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    else:
        settings["environment"]["ttl_seconds"] = 1800
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _: pytest.fail("unexpected HTTP"))) as c,
        pytest.raises(RuntimeError),
    ):
        runtime.ensure_instance_lifetime(c, settings, instance)


@pytest.mark.parametrize("defect", ["failed_post", "identity", "creation", "expiry"])
def test_lifetime_does_not_retry_or_accept_bad_readback(instance, defect):
    calls = []

    def handler(request):
        calls.append(request.method)
        if defect == "failed_post":
            return httpx.Response(503, text="private server body")
        observed = copy.deepcopy(instance)
        observed["expires_at"] = (
            json.loads(request.content)["absolute_expires_at"]
            if (request.method == "POST")
            else (
                datetime.fromisoformat(instance["created_at"]) + timedelta(seconds=3600)
            ).isoformat()
        )
        if request.method == "GET":
            if defect == "identity":
                observed["instance_id"] = "replacement"
            elif defect == "creation":
                observed["created_at"] = datetime.now(UTC).isoformat()
            else:
                observed["expires_at"] = instance["expires_at"]
        return httpx.Response(200, json=observed)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError) as error,
    ):
        runtime.ensure_instance_lifetime(client, config(), instance)
    assert calls.count("POST") == 1
    assert "private server body" not in str(error.value)


def test_versioned_and_legacy_seed_binding_must_agree():
    legacy = {"data_id": "data", "data_version": "v1"}
    task = {"environment_id": "env", **legacy}
    assert runtime.task_data_binding(task) == legacy
    task["seed_config"] = {"seed": {"env_key": "env", "data_key": "data", "data_version": "v1"}}
    assert runtime.task_data_binding(task) == legacy
    del task["data_id"], task["data_version"]
    assert runtime.task_data_binding(task) == legacy
    task["data_id"], task["data_version"] = "data", "v2"
    with pytest.raises(RuntimeError, match="disagree"):
        runtime.task_data_binding(task)
    assert runtime.task_data_binding({}) is None  # requires later exact-instance gate


@pytest.mark.parametrize(
    "task",
    [
        {"data_id": "partial"},
        {"data_id": "", "data_version": "v1"},
        {"seed_config": []},
        {"seed_config": {"a": {}, "b": {}}},
        {"environment_id": "env", "seed_config": {"a": {"env_key": "wrong"}}},
    ],
)
def test_ambiguous_seed_binding_is_rejected(task):
    with pytest.raises(RuntimeError):
        runtime.task_data_binding(task)
