"""Content-free behavioral preflight for shared hosted inference routes.

The preflight never contacts task, instance, session, verifier, or scoring APIs.
It sends two tiny synthetic requests, validates forced structured tool selection,
and persists only status classes, latency, and booleans—not response content.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evals.fleet import hosted_concurrency4_qualification_v1 as qualification
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-behavioral-preflight-v1"
ORIGIN = "https://inference.flt.build"
MODELS_URL = f"{ORIGIN}/v1/models"
COMPLETIONS_URL = f"{ORIGIN}/v1/chat/completions"
ALLOWED_MODELS = {"qwen3.8-27b", "glm-5.3"}
TOOLS = ("bash", "submit_report")
MAX_TOKENS = 64
TIMEOUT_SECONDS = 180
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class PreflightError(RuntimeError):
    """Stable, response-content-free probe failure."""


def _load_response(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_RESPONSE_BYTES:
        raise PreflightError("response_too_large")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError("response_invalid_json") from exc
    if not isinstance(value, dict):
        raise PreflightError("response_invalid_shape")
    return value


def _request(model: str, tool: str, api_key: str) -> dict[str, Any]:
    payload = qualification._request_payload(model, tool)
    payload["max_tokens"] = MAX_TOKENS
    started = time.monotonic()
    request = Request(
        COMPLETIONS_URL,
        method="POST",
        data=self_hosted.canonical_json(payload),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        return {"http_status": exc.code, "result_class": "HTTP_ERROR"}
    except (URLError, TimeoutError, OSError):
        return {"http_status": None, "result_class": "TRANSPORT_ERROR"}
    elapsed_ms = round((time.monotonic() - started) * 1000)
    if status != 200:
        return {"http_status": status, "result_class": "HTTP_ERROR"}
    try:
        qualification._validate_completion(_load_response(raw), model=model, tool_name=tool)
    except (qualification.QualificationError, PreflightError):
        result = "INVALID_TOOL_STRUCTURE"
    else:
        result = "VALID_TOOL_STRUCTURE"
    return {"http_status": status, "latency_ms": elapsed_ms, "result_class": result}


def probe(
    model: str,
    api_key: str,
    *,
    caller: Callable[[str, str, str], dict[str, Any]] = _request,
    advertised_ids: set[str] | None = None,
) -> dict[str, Any]:
    if model not in ALLOWED_MODELS:
        raise ValueError("unsupported exact served model")
    if advertised_ids is None:
        models = qualification.get_json(MODELS_URL, api_key).get("data") or []
        advertised_ids = {
            row.get("id")
            for row in models
            if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
    if model not in advertised_ids:
        raise PreflightError("exact_served_id_absent")
    observations = []
    for tool in TOOLS:
        result = caller(model, tool, api_key)
        observations.append(
            {
                "tool": tool,
                "http_status": result.get("http_status"),
                "latency_ms": result.get("latency_ms"),
                "result_class": result.get("result_class"),
                "structured_tool_available": result.get("result_class")
                == "VALID_TOOL_STRUCTURE",
                "response_content_persisted": False,
            }
        )
    body = {
        "schema_version": SCHEMA,
        "served_id": model,
        "exact_served_id_advertised": True,
        "request": {
            "generic_non_benchmark": True,
            "max_tokens": MAX_TOKENS,
            "prompt_content_persisted": False,
            "response_content_persisted": False,
        },
        "observations": observations,
        "passed": all(row["structured_tool_available"] for row in observations),
        "credentials_included": False,
        "benchmark_content_included": False,
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.error("--out must be unused")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        parser.error("FLEET_API_KEY is required")
    receipt = probe(args.model, key)
    self_hosted.write_json_once(args.out, receipt)
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
