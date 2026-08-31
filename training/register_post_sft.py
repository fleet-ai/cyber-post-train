"""Validate and register the one rendered post-SFT inference contract."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .io import digest_json

TEAM_ID = "a1025f0b-ad67-49fc-a023-51800ab43e84"
MODEL_ID = "ft-run-574bd7b3-step-318"
SOURCE_PATH = "/models/cyber-sft/ft-run-574bd7b3/step-318"
ENGINE_IMAGE = {
    "repository": "lmsysorg/sglang",
    "digest": "sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1",
}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def validate_registration_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("schema") != "cyber_post_sft_serving_registration_v1":
        raise ValueError("unsupported post-SFT serving registration receipt")
    digest = receipt.get("serving_receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "serving_receipt_sha256"}
    if digest_json(unsigned) != digest:
        raise ValueError("post-SFT serving registration receipt digest does not validate")
    registration = dict(_mapping(receipt.get("registration"), "registration"))
    if registration.get("id") != MODEL_ID:
        raise ValueError("post-SFT registration model id differs from the selected checkpoint")
    spec = _mapping(registration.get("spec"), "registration spec")
    model = _mapping(spec.get("model"), "registration model")
    runtime = _mapping(spec.get("runtime"), "registration runtime")
    if model.get("sourcePath") != SOURCE_PATH:
        raise ValueError("post-SFT registration source path differs from the staged bundle")
    if model.get("precision") != "bf16" or model.get("tensorParallelSize") != 1:
        raise ValueError("post-SFT registration precision or tensor parallelism drifted")
    if runtime.get("engine") != "sglang" or runtime.get("image") != ENGINE_IMAGE:
        raise ValueError("post-SFT registration engine identity drifted")
    args = runtime.get("args")
    if not isinstance(args, list):
        raise ValueError("post-SFT registration runtime args must be an array")
    for flag, expected in (
        ("--served-model-name", MODEL_ID),
        ("--context-length", "262144"),
        ("--reasoning-parser", "qwen3"),
        ("--tool-call-parser", "qwen3_coder"),
    ):
        if args.count(flag) != 1 or args.index(flag) + 1 >= len(args):
            raise ValueError(f"post-SFT registration must contain exactly one {flag}")
        if args[args.index(flag) + 1] != expected:
            raise ValueError(f"post-SFT registration {flag} drifted")
    return registration


def register(receipt: Mapping[str, Any], api_key: str) -> dict[str, Any]:
    registration = validate_registration_receipt(receipt)
    if not api_key:
        raise ValueError("FLEET_API_KEY is required")
    headers = {"Authorization": "Bearer " + api_key}
    account_request = urllib.request.Request(
        "https://orchestrator.fleetai.com/v1/account", headers=headers
    )
    with urllib.request.urlopen(account_request, timeout=30) as response:
        account = json.load(response)
    team_id = str(
        account.get("team_id") or account.get("team", {}).get("id") or account.get("id") or ""
    )
    if team_id != TEAM_ID:
        raise ValueError("Fleet account resolved to an unexpected team")
    payload = json.dumps(registration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        "https://inference.flt.build/fleet/v1/models",
        data=payload,
        method="POST",
        headers={
            **headers,
            "Content-Type": "application/json",
            "Idempotency-Key": "post-sft-" + str(receipt["serving_receipt_sha256"])[7:31],
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    result_spec = _mapping(result.get("spec"), "registration API result spec")
    result_model = _mapping(result_spec.get("model"), "registration API result model")
    if result.get("id") != MODEL_ID or result_model.get("revision") != registration["spec"][
        "model"
    ]["revision"]:
        raise ValueError("registration API result differs from the submitted immutable identity")
    return {
        "id": result.get("id"),
        "object": result.get("object"),
        "phase": result.get("status", {}).get("phase"),
        "model_revision": result_model.get("revision"),
        "serving_receipt_sha256": receipt.get("serving_receipt_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    result = register(receipt, os.environ.get("FLEET_API_KEY", ""))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
