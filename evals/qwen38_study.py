"""Fail-closed controls for the Qwen3.8 external-evaluation track."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from evals.webexploitbench.config import ExperimentConfig
from evals.webexploitbench.gateway import FleetGatewayClient
from evals.webexploitbench.manifest import ProtocolManifest, sha256_file

ROOT = Path(__file__).resolve().parents[1]
Q36_WEB_CONFIG = ROOT / (
    "evals/webexploitbench/configs/qwen36-27b-6a9e13bd-level0-qwen-code-full.json"
)
Q38_WEB_CONFIG = ROOT / (
    "evals/webexploitbench/configs/qwen38-27b-1d4bf0f2-level0-qwen-code-full.json"
)
Q36_WEB_PROTOCOL = ROOT / "evals/webexploitbench/manifests/qwen36-27b-qwen-code-protocol-v3.json"
Q38_WEB_PROTOCOL = ROOT / "evals/webexploitbench/manifests/qwen38-27b-qwen-code-protocol-v1.json"
MODEL_LOCK = ROOT / "configs/models/qwen38-27b-1d4bf0f2.lock.json"
SERVING_LOCK = ROOT / "evals/webexploitbench/serving/qwen38-27b-1d4bf0f2.lock.json"
Q36_EXPLOITGYM = ROOT / "evals/exploitgym/configs/qwen36-27b-v1-pilot.json"
Q38_EXPLOITGYM = ROOT / "evals/exploitgym/configs/qwen38-27b-v1-pilot.json"
CONTROL_IMAGE = ROOT / "docs/evidence/exploitgym/2026-08-31-control-image-v1.json"

MODEL = "qwen3.8-27b"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _changed(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    if set(left) != set(right):
        raise ValueError("paired objects have different field sets")
    return {key for key in left if left[key] != right[key]}


def validate_static_controls() -> dict[str, Any]:
    base_config = ExperimentConfig.load(Q36_WEB_CONFIG)
    candidate_config = ExperimentConfig.load(Q38_WEB_CONFIG)
    config_changes = _changed(base_config.to_dict(), candidate_config.to_dict())
    if config_changes != {"model", "run_id"}:
        raise ValueError(f"WebExploitBench config drift: {sorted(config_changes)}")

    base_protocol = ProtocolManifest.load(Q36_WEB_PROTOCOL)
    candidate_protocol = ProtocolManifest.load(Q38_WEB_PROTOCOL)
    protocol_changes = _changed(base_protocol.to_dict(), candidate_protocol.to_dict())
    allowed_protocol_changes = {
        "model",
        "model_revision",
        "tokenizer_revision",
        "chat_template_revision",
        "serving_contract_sha256",
        "experiment_config_sha256",
    }
    if protocol_changes != allowed_protocol_changes:
        raise ValueError(f"WebExploitBench protocol drift: {sorted(protocol_changes)}")
    if candidate_protocol.experiment_config_sha256 != "sha256:" + sha256_file(Q38_WEB_CONFIG):
        raise ValueError("Qwen3.8 protocol does not bind its exact config")

    model_lock = _load(MODEL_LOCK)
    serving_lock = _load(SERVING_LOCK)
    if (model_lock.get("repo"), model_lock.get("revision")) != (
        "Qwen/Qwen3.8-27B",
        REVISION,
    ):
        raise ValueError("Qwen3.8 model lock identity drifted")
    selection = model_lock.get("selection") or {}
    if selection.get("benchmark_temporal_status") != "released_after_webexploitbench":
        raise ValueError("Qwen3.8 temporal caveat is missing")
    tokenizer = model_lock.get("tokenizer") or {}
    if candidate_protocol.tokenizer_revision != tokenizer.get("manifest_sha256"):
        raise ValueError("Qwen3.8 tokenizer binding drifted")
    template = next(
        row for row in tokenizer.get("files", []) if row.get("path") == "chat_template.jinja"
    )
    if candidate_protocol.chat_template_revision != "sha256:" + template["sha256"]:
        raise ValueError("Qwen3.8 chat-template binding drifted")
    if serving_lock.get("id") != MODEL or serving_lock.get("revision") != REVISION:
        raise ValueError("Qwen3.8 serving lock identity drifted")
    if candidate_protocol.serving_contract_sha256 != serving_lock.get("canonical_spec_sha256"):
        raise ValueError("Qwen3.8 serving-contract binding drifted")
    if serving_lock.get("tool_preflight", {}).get("structured_tool_call") is not True:
        raise ValueError("Qwen3.8 serving lock lacks a structured-tool preflight")

    base_gym = _load(Q36_EXPLOITGYM)
    candidate_gym = _load(Q38_EXPLOITGYM)
    source = candidate_gym.get("selection") or {}
    if source.get("source_protocol_sha256") != _digest(base_gym):
        raise ValueError("ExploitGym source protocol digest drifted")
    if source.get("task_ids") != (base_gym.get("selection") or {}).get("task_ids"):
        raise ValueError("ExploitGym task identities drifted")
    for field in (
        "pass_k",
        "max_workers",
        "timeout_seconds_per_task",
        "firewall_required",
        "allowed_model_domain",
        "web_search",
    ):
        if candidate_gym["execution"].get(field) != base_gym["execution"].get(field):
            raise ValueError(f"ExploitGym execution control drifted: {field}")
    control = _load(CONTROL_IMAGE)
    if candidate_gym["harness"].get("control_image") != control.get("image"):
        raise ValueError("ExploitGym immutable control image drifted")
    if candidate_gym["model"].get("gateway_model") != MODEL:
        raise ValueError("ExploitGym route identity drifted")

    return {
        "schema": "qwen38_external_eval_static_control_v1",
        "model": MODEL,
        "revision": REVISION,
        "webexploitbench_protocol_sha256": "sha256:" + candidate_protocol.digest(),
        "exploitgym_source_protocol_sha256": source["source_protocol_sha256"],
        "control_image": control["image"],
        "temporal_interpretation": "paired_before_after_only_not_temporally_clean_holdout",
        "valid": True,
    }


def live_probe(api_key: str) -> dict[str, Any]:
    config = ExperimentConfig.load(Q38_WEB_CONFIG)
    gateway = FleetGatewayClient(config, api_key=api_key).probe(perform_chat=True)
    binding = gateway.catalog_binding or {}
    if binding.get("model_revision") != REVISION:
        raise ValueError("live catalog revision differs from the frozen Qwen3.8 revision")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Call identity exactly once with value parity."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "identity",
                "description": "Return a value",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "identity"}},
        "temperature": 0,
        "max_tokens": 256,
    }
    request = Request(
        "https://inference.flt.build/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        value = json.load(response)
    message = value["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "identity":
        raise ValueError("Qwen3.8 did not return the forced structured tool call")
    if json.loads(calls[0]["function"]["arguments"]) != {"value": "parity"}:
        raise ValueError("Qwen3.8 forced-tool arguments differed")
    return {
        **validate_static_controls(),
        "schema": "qwen38_external_eval_live_preflight_v1",
        "models_endpoint_ok": gateway.models_endpoint_ok,
        "catalog_endpoint_ok": gateway.catalog_endpoint_ok,
        "chat_endpoint_ok": gateway.chat_endpoint_ok,
        "judge_chat_endpoint_ok": gateway.judge_chat_endpoint_ok,
        "catalog_binding": binding,
        "structured_tool_call": True,
        "returned_model": value.get("model"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "probe"))
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "validate":
        receipt = validate_static_controls()
    else:
        key = os.environ.get("FLEET_API_KEY")
        if not key:
            raise SystemExit("FLEET_API_KEY is required for the live probe")
        receipt = live_probe(key)
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.write_text(encoded, encoding="utf-8")
        output.chmod(0o600)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
