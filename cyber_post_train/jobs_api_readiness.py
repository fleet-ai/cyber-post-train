"""Read-only rollout check for Fleet Jobs API failed-job alert opt-out.

The public OpenAPI document can tell an operator whether a deployed API
*advertises* the ``failureAlerts`` request field.  It cannot prove that a
specific job will render correctly, so a positive result is deliberately only
permission to request a fresh authenticated server preview.  It is never
permission to create a workload.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx

OPENAPI_PATH = "/v1/openapi.json"
FAILURE_ALERT_FIELD = "failureAlerts"
RUN_ENDPOINTS = ("/v1/runs", "/v1/runs/preview")
SCHEMA = "cyber_jobs_api_failure_alert_rollout_readiness_v1"
NEXT_GATE = (
    "A fresh authenticated server preview must render the root RayJob annotation "
    "fleet.ai/failure-alerts: off before any create."
)


def _base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("Jobs API readiness target must be an HTTPS origin")
    if parsed.path not in ("", "/"):
        raise ValueError("Jobs API readiness target must not contain a path")
    return f"https://{parsed.netloc}"


def _resolve(document: Mapping[str, Any], schema: Any) -> Mapping[str, Any] | None:
    """Resolve local component references without accepting remote documents."""
    seen: set[str] = set()
    while isinstance(schema, Mapping) and "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/components/schemas/"):
            return None
        if reference in seen:
            return None
        seen.add(reference)
        name = reference.removeprefix("#/components/schemas/")
        components = document.get("components")
        schemas = components.get("schemas") if isinstance(components, Mapping) else None
        schema = schemas.get(name) if isinstance(schemas, Mapping) else None
    return schema if isinstance(schema, Mapping) else None


def _advertises_failure_alerts(document: Mapping[str, Any], path: str) -> bool:
    paths = document.get("paths")
    endpoint = paths.get(path) if isinstance(paths, Mapping) else None
    post = endpoint.get("post") if isinstance(endpoint, Mapping) else None
    body = post.get("requestBody") if isinstance(post, Mapping) else None
    content = body.get("content") if isinstance(body, Mapping) else None
    media = content.get("application/json") if isinstance(content, Mapping) else None
    schema = media.get("schema") if isinstance(media, Mapping) else None
    resolved = _resolve(document, schema)
    properties = resolved.get("properties") if resolved is not None else None
    field = properties.get(FAILURE_ALERT_FIELD) if isinstance(properties, Mapping) else None
    return isinstance(field, Mapping) and field.get("type") == "boolean"


def inspect_failure_alert_rollout(
    base_url: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Inspect only the anonymous OpenAPI contract and fail closed.

    This function intentionally carries no token argument and issues exactly
    one GET.  Its result must be followed by the normal authenticated Jobs API
    preview and root-annotation validation before submission.
    """
    target = _base_url(base_url)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "base_url": target,
        "read_only": True,
        "submission_authorized": False,
        "next_gate": NEXT_GATE,
    }
    try:
        with httpx.Client(
            base_url=target,
            timeout=20,
            follow_redirects=False,
            headers={"Accept": "application/json"},
            transport=transport,
        ) as client:
            response = client.get(OPENAPI_PATH)
    except httpx.HTTPError:
        return {
            **result,
            "state": "openapi_unavailable",
            "failure_alert_opt_out_advertised": False,
        }
    if response.status_code != 200:
        return {
            **result,
            "state": "openapi_unavailable",
            "http_status": response.status_code,
            "failure_alert_opt_out_advertised": False,
        }
    try:
        document = response.json()
    except ValueError:
        document = None
    if not isinstance(document, Mapping):
        return {
            **result,
            "state": "openapi_malformed",
            "failure_alert_opt_out_advertised": False,
        }
    endpoints = {path: _advertises_failure_alerts(document, path) for path in RUN_ENDPOINTS}
    advertised = all(endpoints.values())
    return {
        **result,
        "state": "advertised_not_submission_qualified" if advertised else "not_advertised",
        "failure_alert_opt_out_advertised": advertised,
        "endpoints": endpoints,
    }
