"""Runtime adapter for one preflight-cleared Qwen G18 hosted canary."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation18 as g18
from evals.fleet import self_hosted


def runtime_gate(plan: dict) -> None:
    g18.validate_plan(plan)
    g18.validate_parity(g18.load(Path(plan["repo_root"]) / g18.PARITY_PATH))
    path = Path(os.environ.get("G18_PREFLIGHT_PATH", ""))
    expected = os.environ.get("G18_PREFLIGHT_SHA256")
    receipt = g18.load(path)
    if any(
        (
            receipt.get("schema_version") != "fleet-qwen-generation18-hosted-canary-preflight-v1",
            receipt.get("status") != "CLEAR",
            receipt.get("plan_sha256") != plan["plan_sha256"],
            receipt.get("cell_id") != plan["attempts"][0]["cell_id"],
            receipt.get("execution_id") != plan["attempts"][0]["execution_id"],
            receipt.get("mutation_calls") != 0,
            receipt.get("receipt_sha256") != expected,
            receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("Generation-18 live preflight receipt drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    plan = g18.load(args.plan)
    prior = engine.bulk
    try:
        engine.bulk = g18
        engine.run_controller(plan, out=args.out, proxy=args.proxy, runtime_gate_check=runtime_gate)
    finally:
        engine.bulk = prior
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
