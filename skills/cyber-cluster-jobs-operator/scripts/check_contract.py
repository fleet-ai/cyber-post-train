#!/usr/bin/env python3
"""Fail closed when the deployed Fleet cluster Jobs API changes shape."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.ft.flt.build"
EXPECTED_FIELDS = {
    "image",
    "command",
    "workers",
    "gpus_per_worker",
    "env",
    "secrets",
    "resources",
    "priority_class",
    "privileged",
    "run_dir",
    "title",
}


def _request_schema(document: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        operation = document["paths"]["/v1/runs"]["post"]
        body = operation["requestBody"]["content"]["application/json"]["schema"]
        reference = body["$ref"]
    except (KeyError, TypeError) as exc:
        raise ValueError("live OpenAPI lacks POST /v1/runs JSON request schema") from exc
    prefix = "#/components/schemas/"
    if not isinstance(reference, str) or not reference.startswith(prefix):
        raise ValueError("POST /v1/runs request schema is not a local component reference")
    name = reference.removeprefix(prefix)
    try:
        schema = document["components"]["schemas"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"live OpenAPI lacks referenced schema {name!r}") from exc
    if not isinstance(schema, dict):
        raise ValueError(f"live OpenAPI schema {name!r} is not an object")
    return name, schema


def validate(document: dict[str, Any]) -> dict[str, Any]:
    name, schema = _request_schema(document)
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict):
        raise ValueError(f"{name} has no properties object")
    if not isinstance(required, list) or not {"image", "command"}.issubset(required):
        raise ValueError(f"{name} no longer requires both image and command")
    missing = sorted(EXPECTED_FIELDS - properties.keys())
    if missing:
        raise ValueError(f"{name} is missing general job fields: {missing}")
    gpu = properties["gpus_per_worker"]
    workers = properties["workers"]
    if gpu.get("minimum") != 1 or workers.get("minimum") != 1:
        raise ValueError("worker or GPU minimum changed; re-evaluate resource routing")
    description = str(schema.get("description") or "").lower()
    if "generic" not in description or "image" not in description or "command" not in description:
        raise ValueError(f"{name} no longer declares the general image-and-command contract")
    return {
        "status": "PASSED",
        "base_url": DEFAULT_BASE_URL,
        "route": "POST /v1/runs",
        "schema": name,
        "required": sorted(required),
        "fields": sorted(properties),
        "minimum_workers": workers["minimum"],
        "minimum_gpus_per_worker": gpu["minimum"],
        "maximum_gpus_per_worker": gpu.get("maximum"),
        "mutations": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    url = f"{args.base_url.rstrip('/')}/openapi.json"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as response:  # noqa: S310
            if response.status != 200:
                raise ValueError(f"GET {url} returned HTTP {response.status}")
            document = json.load(response)
        result = validate(document)
        result["base_url"] = args.base_url.rstrip("/")
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED", "reason": str(exc), "mutations": 0}), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
