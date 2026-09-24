#!/usr/bin/env python3
"""Reproduce the sanitized Qwen3.8 tool-contract tokenizer A/B receipt.

This is an offline synthetic magnitude probe. It does not read or emit corpus,
benchmark, task, or production prompt text, rendered prompts, or token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
TOKENIZER_JSON_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
BACKEND_SHA256 = "ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0"
TOOL_CATALOG_SHA256 = "85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a"

EXPECTED = {
    "full_trajectory": {
        "arm_a_tokens": 234,
        "arm_a_token_ids_sha256": (
            "f30f30105fc3d38cf729f41b6232fa6a61e0e2fda8975ae46c372236cf543009"
        ),
        "arm_b_tokens": 1111,
        "arm_b_token_ids_sha256": (
            "a069b970b417d43bcdecb61106e68179cd6a297fd0ace6ca2f75b2988e3b9fd9"
        ),
        "token_delta_b_minus_a": 877,
        "first_differing_token_index_zero_based": 41,
        "common_prefix_tokens": 41,
        "common_suffix_tokens": 91,
    },
    "initial_context_with_generation_prompt": {
        "arm_a_tokens": 76,
        "arm_a_token_ids_sha256": (
            "65f6463a355321094483d14683bb127cc4bf80f19575cba24f2c50436d3f4dfe"
        ),
        "arm_b_tokens": 950,
        "arm_b_token_ids_sha256": (
            "29b69df267aed2b8ba598e8a6e0e6b9b35960572890185a90f9b8d8d9a8418d9"
        ),
        "token_delta_b_minus_a": 874,
        "first_differing_token_index_zero_based": 41,
        "common_prefix_tokens": 41,
        "common_suffix_tokens": 36,
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any, *, ensure_ascii: bool = True) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def token_ids_sha256(token_ids: list[int]) -> str:
    # The sequence digest deliberately does not sort: array order is semantic.
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def synthetic_messages() -> list[dict[str, Any]]:
    """Return a benign fixture with one shell call and one report call."""
    return [
        {
            "role": "system",
            "content": "Follow the task instructions and use available tools as needed.",
        },
        {
            "role": "user",
            "content": "Run one harmless local check, then submit the synthetic result.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": {"script": "printf 'ok\\n'"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "content": "exitCode: 0\\n\\n--- stdout ---\\nok",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "submit_report",
                        "arguments": {
                            "flag": "FLAG{00000000-0000-4000-8000-000000000000}",
                            "explanation": "Synthetic benign result.",
                        },
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "content": "Synthetic report accepted.",
        },
    ]


def token_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> list[int]:
    result = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
    )
    if isinstance(result, dict):
        result = result["input_ids"]
    return list(result)


def compare(arm_a: list[int], arm_b: list[int]) -> dict[str, Any]:
    first = next(
        (index for index, pair in enumerate(zip(arm_a, arm_b, strict=False)) if pair[0] != pair[1]),
        min(len(arm_a), len(arm_b)),
    )
    common_suffix = 0
    while (
        common_suffix < min(len(arm_a), len(arm_b))
        and arm_a[-1 - common_suffix] == arm_b[-1 - common_suffix]
    ):
        common_suffix += 1
    return {
        "arm_a_tokens": len(arm_a),
        "arm_a_token_ids_sha256": token_ids_sha256(arm_a),
        "arm_b_tokens": len(arm_b),
        "arm_b_token_ids_sha256": token_ids_sha256(arm_b),
        "token_delta_b_minus_a": len(arm_b) - len(arm_a),
        "first_differing_token_index_zero_based": first,
        "common_prefix_tokens": first,
        "common_suffix_tokens": common_suffix,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_snapshot = (
        Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots" / REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-snapshot",
        type=Path,
        default=Path(os.environ.get("QWEN38_TOKENIZER_SNAPSHOT", default_snapshot)),
    )
    parser.add_argument(
        "--tool-catalog",
        type=Path,
        default=root / "configs/data/qwen38-rl-filtered-canary-tool-catalog-v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = args.tokenizer_snapshot.resolve()
    if snapshot.name != REVISION:
        raise ValueError(f"tokenizer snapshot must be exact revision {REVISION}")
    if sha256_bytes((snapshot / "chat_template.jinja").read_bytes()) != CHAT_TEMPLATE_SHA256:
        raise ValueError("chat template digest mismatch")
    if sha256_bytes((snapshot / "tokenizer.json").read_bytes()) != TOKENIZER_JSON_SHA256:
        raise ValueError("tokenizer.json digest mismatch")

    catalog = json.loads(args.tool_catalog.read_text(encoding="utf-8"))
    if sha256_bytes(canonical_json_bytes(catalog)) != TOOL_CATALOG_SHA256:
        raise ValueError("canonical tool catalog digest mismatch")
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"fleet_{tool['name']}",
                "description": tool["description"],
                "parameters": tool["inputSchema"],
            },
        }
        for tool in catalog
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    backend_digest = sha256_bytes(tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    if backend_digest != BACKEND_SHA256:
        raise ValueError("tokenizer backend digest mismatch")

    arm_a = synthetic_messages()
    arm_b = deepcopy(arm_a)
    for message in arm_b:
        for call in message.get("tool_calls", []):
            call["function"]["name"] = f"fleet_{call['function']['name']}"

    full_a = token_ids(tokenizer, arm_a, tools=[], add_generation_prompt=False)
    full_b = token_ids(tokenizer, arm_b, tools=tools, add_generation_prompt=False)
    initial_a = token_ids(
        tokenizer,
        arm_a[:2],
        tools=[],
        add_generation_prompt=True,
    )
    initial_b = token_ids(
        tokenizer,
        arm_b[:2],
        tools=tools,
        add_generation_prompt=True,
    )
    results = {
        "full_trajectory": compare(full_a, full_b),
        "initial_context_with_generation_prompt": compare(initial_a, initial_b),
    }
    if results != EXPECTED:
        raise AssertionError(
            "synthetic receipt drift:\n"
            + json.dumps({"expected": EXPECTED, "observed": results}, indent=2, sort_keys=True)
        )

    import tokenizers  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    output = {
        "schema": "cyber_qwen38_tool_contract_synthetic_probe_v1",
        "status": "verified",
        "caveat": "Synthetic magnitude probe, not a corpus or benchmark prompt.",
        "identity": {
            "revision": REVISION,
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "results": results,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
