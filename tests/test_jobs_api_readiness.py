from __future__ import annotations

import httpx
import pytest

from cyber_post_train.jobs_api_readiness import inspect_failure_alert_rollout


def _document(*, runs: bool = True, preview: bool = True) -> dict:
    paths = {}
    for path, enabled in (("/v1/runs", runs), ("/v1/runs/preview", preview)):
        paths[path] = {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/RLJobConfig"}}
                    }
                }
            }
        }
        if not enabled:
            paths[path]["post"]["requestBody"]["content"]["application/json"]["schema"] = {
                "type": "object",
                "properties": {},
            }
    return {
        "openapi": "3.1.0",
        "paths": paths,
        "components": {
            "schemas": {
                "RLJobConfig": {
                    "type": "object",
                    "properties": {"failureAlerts": {"type": "boolean", "default": True}},
                }
            }
        },
    }


def test_readiness_is_one_anonymous_openapi_get_and_never_authorizes_submission() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url.path == "/v1/openapi.json"
        assert "authorization" not in request.headers
        return httpx.Response(200, json=_document())

    result = inspect_failure_alert_rollout(
        "https://api.example.test", transport=httpx.MockTransport(handler)
    )

    assert len(requests) == 1
    assert result["state"] == "advertised_not_submission_qualified"
    assert result["failure_alert_opt_out_advertised"] is True
    assert result["endpoints"] == {"/v1/runs": True, "/v1/runs/preview": True}
    assert result["read_only"] is True
    assert result["submission_authorized"] is False
    assert "fresh authenticated server preview" in result["next_gate"]


def test_missing_field_fails_closed_without_a_preview_request() -> None:
    document = _document()
    document["components"]["schemas"]["RLJobConfig"]["properties"] = {}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=document)

    result = inspect_failure_alert_rollout(
        "https://api.example.test", transport=httpx.MockTransport(handler)
    )

    assert [request.url.path for request in requests] == ["/v1/openapi.json"]
    assert result["state"] == "not_advertised"
    assert result["failure_alert_opt_out_advertised"] is False
    assert result["submission_authorized"] is False


def test_partial_contract_is_not_ready() -> None:
    result = inspect_failure_alert_rollout(
        "https://api.example.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=_document(preview=False))),
    )

    assert result["state"] == "not_advertised"
    assert result["endpoints"] == {"/v1/runs": True, "/v1/runs/preview": False}


def test_unavailable_or_malformed_openapi_fails_closed() -> None:
    unavailable = inspect_failure_alert_rollout(
        "https://api.example.test", transport=httpx.MockTransport(lambda _: httpx.Response(503))
    )
    malformed = inspect_failure_alert_rollout(
        "https://api.example.test",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="no")),
    )

    assert unavailable["state"] == "openapi_unavailable"
    assert malformed["state"] == "openapi_malformed"
    assert unavailable["failure_alert_opt_out_advertised"] is False
    assert malformed["failure_alert_opt_out_advertised"] is False


@pytest.mark.parametrize("value", ["http://api.example.test", "https://api.example.test/v1"])
def test_only_an_https_origin_is_accepted(value: str) -> None:
    with pytest.raises(ValueError, match="HTTPS origin|must not contain a path"):
        inspect_failure_alert_rollout(value)
