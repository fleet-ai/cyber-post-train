"""Small exact OpenCode settings and durable-receipt helpers for parity rails."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto

OPENCODE_CONTEXT_MANAGEMENT = "opencode_1.18.27_native_compaction_autocontinue_v1"
OPENCODE_NO_AUTOCONTINUE_CONTEXT_MANAGEMENT = "opencode_1.18.27_native_compaction_no_autocontinue"


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Write one canonical, durable, create-once JSON receipt."""

    payload = crypto.canonical_json(value) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def opencode_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Render the exact OpenCode 1.18.27 compaction treatment."""

    harness = config["harness"]
    if harness.get("name") != "opencode" or harness.get("version") != "1.18.27":
        raise ValueError("OpenCode requires the exact supported harness version")
    policy = harness.get("context_management")
    if policy not in {
        OPENCODE_CONTEXT_MANAGEMENT,
        OPENCODE_NO_AUTOCONTINUE_CONTEXT_MANAGEMENT,
    }:
        raise ValueError("OpenCode requires an explicitly supported context policy")
    fields = ("context_window_size", "max_output_tokens")
    if any(type(harness.get(field)) is not int or harness[field] <= 0 for field in fields):
        raise ValueError("OpenCode context and output limits must be positive integers")
    context, output = (harness[field] for field in fields)
    if output >= context:
        raise ValueError("OpenCode output limit must leave room for input")
    headroom = harness.get("compaction_headroom_tokens")
    if policy == OPENCODE_CONTEXT_MANAGEMENT:
        if type(headroom) is not int or headroom <= 0:
            raise ValueError("OpenCode autocontinue headroom must be a positive integer")
        if output + headroom >= context:
            raise ValueError("OpenCode output and compaction headroom must leave room for input")
    elif headroom is not None:
        raise ValueError("OpenCode no-autocontinue treatment forbids compaction headroom")
    model_id = config["model"]["served_id"]
    settings = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "fleet-cluster": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Fleet cluster inference",
                "options": {
                    "baseURL": "http://model-proxy:8877/v1",
                    "apiKey": "local-proxy-only",
                    "timeout": False,
                    "chunkTimeout": 300000,
                },
                "models": {
                    model_id: {
                        "name": model_id,
                        "reasoning": True,
                        "tool_call": True,
                        "interleaved": "reasoning_content",
                        "limit": {"context": context, "output": output},
                    }
                },
            }
        },
        "mcp": {
            "fleet": {
                "type": "remote",
                "url": "http://fleet-mcp-proxy:8090/mcp",
                "enabled": True,
            }
        },
        "permission": {"*": "deny", "fleet_*": "allow"},
        "tools": {
            name: False
            for name in (
                "bash",
                "edit",
                "read",
                "glob",
                "grep",
                "list",
                "task",
                "webfetch",
                "websearch",
                "skill",
            )
        },
    }
    if policy == OPENCODE_CONTEXT_MANAGEMENT:
        settings["compaction"] = {"auto": True, "reserved": headroom}
        settings["provider"]["fleet-cluster"]["models"][model_id]["limit"]["input"] = (
            context - output
        )
    else:
        settings["plugin"] = [
            "file:///home/node/.config/opencode/fleet-disable-compaction-autocontinue.mjs"
        ]
    return settings
