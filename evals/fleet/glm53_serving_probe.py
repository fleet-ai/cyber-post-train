"""Content-blind GLM5.3 serving readiness and structured-tool probe."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def request_json(
    url: str, *, method: str = "GET", payload: dict[str, Any] | None = None
) -> Any:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        if response.status != 200:
            raise RuntimeError(f"serving probe HTTP status {response.status}")
        return json.load(response)


def request_status(url: str) -> None:
    with urllib.request.urlopen(url, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"serving probe HTTP status {response.status}")


def probe(origin: str) -> dict[str, Any]:
    request_status(f"{origin}/health")
    models = request_json(f"{origin}/v1/models")
    model_ids = [row.get("id") for row in models.get("data", [])]
    if "glm-5.3" not in model_ids:
        raise RuntimeError("served model identity mismatch")
    model_info = request_json(f"{origin}/get_model_info")
    server_info = request_json(f"{origin}/get_server_info")
    response = request_json(
        f"{origin}/v1/chat/completions",
        method="POST",
        payload={
            "model": "glm-5.3",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Call submit_report exactly once with a brief benign status. "
                        "Do not call bash."
                    ),
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "Run a shell command.",
                        "parameters": {
                            "type": "object",
                            "properties": {"command": {"type": "string"}},
                            "required": ["command"],
                            "additionalProperties": False,
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "submit_report",
                        "description": "Submit the final report.",
                        "parameters": {
                            "type": "object",
                            "properties": {"report": {"type": "string"}},
                            "required": ["report"],
                            "additionalProperties": False,
                        },
                    },
                },
            ],
            "tool_choice": "required",
            "temperature": 0,
            "max_tokens": 128,
        },
    )
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise RuntimeError("structured-tool probe choice count mismatch")
    choice = choices[0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    names = [(call.get("function") or {}).get("name") for call in calls]
    if names != ["submit_report"]:
        raise RuntimeError("structured-tool probe did not select submit_report exactly once")
    for call in calls:
        arguments = (call.get("function") or {}).get("arguments")
        if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
            raise RuntimeError("structured-tool probe arguments are not valid JSON")

    model_path = model_info.get("model_path")
    if model_path != "/mnt/sfs/models/glm-5.3-30333038":
        raise RuntimeError("live model path mismatch")
    context_length = server_info.get("context_length")
    if context_length != 262144:
        raise RuntimeError("live context length mismatch")
    return {
        "schema": "fleet_cyber_glm53_serving_probe_v1",
        "health": True,
        "served_model": response.get("model"),
        "model_path": model_path,
        "context_length": context_length,
        "finish_reason": choice.get("finish_reason"),
        "tool_names": names,
        "tool_arguments_valid_json": True,
        "response_content_included": False,
        "tool_arguments_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    print(json.dumps(probe(args.origin.rstrip("/")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
