import json
import subprocess

import httpx
import pytest

from evals.fleet import opencode_self_hosted as runner


@pytest.mark.parametrize(
    "detail,reason",
    [
        ("Not Found", "route_not_found"),
        ("Task not found for team_id=private", "task_not_found_for_team"),
        ("Task version not found for task_key=private", "task_version_not_found"),
        (
            "Pinned task-version environment_version_id did not resolve: private",
            "environment_version_unresolved",
        ),
        (
            "1 task version(s) declare a verifier but have no verifier_version_id pin: private",
            "verifier_version_unpinned",
        ),
        (
            "Exact task hydration did not resolve environment and data versions",
            "environment_or_data_version_unresolved",
        ),
        ("private prompt, credential and traceback", "unclassified"),
        ({"private": "content"}, "unclassified"),
        (None, "unclassified"),
    ],
)
def test_diagnostics_do_not_serialize_response_text(detail, reason):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(404, json={"detail": detail})

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(runner.FleetRequestError) as caught,
    ):
        runner._request(client, "POST", "/v1/rollout-rewards/task/versions/version/instances")
    assert len(calls) == 1
    receipt = runner.sanitized_failure_receipt(caught.value, run_id="test", elapsed_seconds=1)
    assert receipt["reason"] == reason
    assert receipt["http_status"] == 404
    assert receipt["response_sha256"].startswith("sha256:")
    assert "private" not in json.dumps(receipt)
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("content", [b"not JSON", b"[]", b"null"])
def test_non_object_response_retains_only_digest(content):
    result = runner.request_diagnostic(httpx.Response(502, content=content))
    assert result == {"response_sha256": runner.sha256(content), "reason": "unclassified"}


@pytest.mark.parametrize("operation", ["rollout", "runtime_preflight"])
@pytest.mark.parametrize("status", [404, 503])
def test_actual_creation_paths_retain_safe_diagnostic_without_retry(
    operation, status, monkeypatch, tmp_path
):
    calls = []
    detail = "Task version not found for task_key=private-body-do-not-persist"
    body = json.dumps({"detail": detail}).encode()

    def handler(request):
        calls.append(request)
        if request.method == "GET" and request.url.path == "/v1/account":
            return httpx.Response(200, json={"team_name": "fleet", "team_id": runner.FLEET_TEAM_ID})
        assert request.method == "POST"
        return httpx.Response(status, content=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(runner.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setenv("FLEET_API_KEY", "fixture-secret")
    monkeypatch.setenv("AGENT_HARNESS_IMAGE", "fixture-image")
    monkeypatch.setenv("FIXED_PROXY_IMAGE", "fixture-proxy")
    monkeypatch.setattr(runner, "load_and_verify_task", lambda *_: {"prompt": "synthetic"})
    monkeypatch.setattr(runner, "assert_authoritative_routes_deployed", lambda *_: {})
    monkeypatch.setattr(runner, "build_instance_payload", lambda *_: {"ttl_seconds": 300})
    monkeypatch.setattr(
        runner, "_docker", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="")
    )
    monkeypatch.setattr(runner.time, "sleep", lambda _: pytest.fail("must not retry POST"))
    config = {
        "schema_version": "fixture",
        "run_id": "test-create",
        "source_job_id": "fixture",
        "task": {"key": "fixture__blackbox_ctf_v1", "version_id": "fixture-version"},
        "environment": {},
        "verifier": {},
        "model": {},
        "authority": {
            "provisioning_route_template": (
                "/v1/rollout-rewards/{task_key}/versions/{task_version_id}/instances"
            )
        },
        "harness": {"name": "qwen_code"},
        "execution": {"network": "test-create-network"},
    }
    output = tmp_path / "episode"
    try:
        with pytest.raises(runner.FleetRequestError) as caught:
            if operation == "rollout":
                runner.run(config, output, tmp_path / "unused-proxy.py")
            else:
                runner.runtime_preflight(client, config, {})
    finally:
        client.close()
    assert len([r for r in calls if r.method == "POST"]) == 1
    receipt = runner.sanitized_failure_receipt(caught.value, run_id="test", elapsed_seconds=1)
    assert receipt["reason"] == "task_version_not_found"
    assert receipt["response_sha256"] == runner.sha256(body)
    assert receipt["http_status"] == status
    if operation == "rollout":
        stored = json.loads((output / "failure.json").read_text())
        assert stored["reason"] == receipt["reason"]
        assert stored["response_sha256"] == receipt["response_sha256"]
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "private-body-do-not-persist" not in path.read_text()
            assert "fixture-secret" not in path.read_text()
