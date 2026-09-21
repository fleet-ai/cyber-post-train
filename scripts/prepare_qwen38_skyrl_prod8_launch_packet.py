#!/usr/bin/env python3
"""Validate the frozen, non-reusable historical prod8 launch packet.

Prod8 was created once but its terminal state and resource release cannot be
proved.  Its packet therefore remains immutable historical evidence, rather
than being re-sealed against newer source code.  The fresh prod9 preparation
path owns all current runtime, plan, request, and preflight bindings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyber_post_train.jobs import digest
from scripts import audit_qwen38_skyrl_launch_readiness as readiness
from training import skyrl_reward_rayjob as direct

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs/qualification/qwen38-rl-reward-canary-prod8-launch-packet-v1.json"
PREVIEW_EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-21-skyrl-prod8-read-only-previews-v1.json"
)
EXPECTED = {
    "plan_sha256": "09cabfc727e8b8448bd00a5ea3cee03914844e67cfc80b1f033bdbe2671212de",
    "request_sha256": "7c2df31feceb5c741cb16463b554b91d00b11c6bf50ec85b3203706823c2fb44",
    "runtime_sha256": "acb1d1a1ff5aec9d8c153dfc57e52a13bb0b081fc666a4a0d0db139e8cb6831b",
    "manifest_sha256": "sha256:5561f1a349abbe1a580dd5763368c1e6c1524861c39d744f7dad4f25d9950dd3",
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one object")
    return value


def _seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = "sha256:" + digest(result)
    return result


def _validate_historical_packet(value: dict) -> dict:
    """Accept only the exact committed, non-authorizing prod8 record."""
    if (
        value != _seal({key: item for key, item in value.items() if key != "sha256"})
        or value.get("schema") != "cyber_qwen38_skyrl_prod8_launch_packet_v1"
        or value.get("digests") != EXPECTED
        or value.get("launch_authorized") is not False
        or value.get("external_mutations_by_builder") != 0
        or value.get("identity", {}).get("name") != direct.RUN_NAME
        or value.get("identity", {}).get("output_root") != "/mnt/sfs/jobs/chris-q38-rlreward-prod8"
        or value.get("read_only_preview_observation", {}).get("authorization_reusable") is not False
    ):
        raise ValueError("prod8 historical launch packet changed")
    if OUTPUT.read_bytes() != raw(value):
        raise ValueError("prod8 historical launch packet is not canonical")
    return value


def current_source_status() -> dict:
    """Describe current code without ever refreshing the prod8 packet.

    This intentionally compiles only public, sanitized inputs.  It reports
    whether today's source closure matches the old packet, but prod8 remains
    non-reusable even if it does.  A fresh prod9 receipt is the only path for
    current plan/preflight evidence.
    """
    run = readiness.load(readiness.CANARY_RUN)
    manifest = readiness.prod8_metadata(run, {})
    plan, request = readiness.compile_prod8(run, manifest)
    current = {
        "plan_sha256": digest(plan),
        "request_sha256": digest(request),
        "runtime_sha256": plan["runtime_sha256"],
        "manifest_sha256": manifest["sha256"],
    }
    preflight = direct.preflight_job_manifest(plan)
    return {
        "historical_packet_sha256": _load(OUTPUT)["sha256"],
        "historical_digests": EXPECTED,
        "current_digests": current,
        "current_preflight_manifest_sha256": digest(preflight),
        "matches_historical_source_closure": current == EXPECTED,
        "prod8_reusable": False,
        "reason": "prod8 terminal state and GPU release are unknown; use fresh prod9",
    }


def build() -> dict:
    """Return the frozen prod8 record; never regenerate it from current code."""
    return _validate_historical_packet(_load(OUTPUT))


def raw(value: dict) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    value = build()
    status = current_source_status()
    print(
        json.dumps(
            {
                "packet": str(OUTPUT.relative_to(ROOT)),
                "sha256": value["sha256"],
                "prod8_reusable": status["prod8_reusable"],
                "matches_historical_source_closure": status["matches_historical_source_closure"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
