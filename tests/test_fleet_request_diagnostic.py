import json

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
