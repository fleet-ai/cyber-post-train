"""Runtime adapter for preflight-released Qwen G19 hosted bulk shards."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation19_bulk as g19
from evals.fleet import self_hosted


def runtime_gate(plan: dict) -> None:
    path = Path(os.environ.get("G19_PREFLIGHT_PATH", ""))
    expected = os.environ.get("G19_PREFLIGHT_SHA256")
    receipt = g19.g17.load(path)
    plans = g19.validate_all(Path(plan["repo_root"]))
    execution_ids = sorted(
        row["execution_id"] for built in plans.values() for row in built["attempts"]
    )
    if any(
        (
            receipt.get("schema_version") != "fleet-qwen-generation19-hosted-bulk-preflight-v1",
            receipt.get("status") != "CLEAR",
            receipt.get("planned_cells") != 384,
            receipt.get("planned_execution_ids_sha256")
            != self_hosted.sha256(self_hosted.canonical_json(execution_ids)),
            receipt.get("g18_accepted_receipt_sha256")
            != "sha256:083deadd604e090bb43fc4455fed032e763119e8e89ce4bae7632e02ef99eb30",
            receipt.get("mutation_calls") != 0,
            receipt.get("receipt_sha256") != expected,
            receipt.get("receipt_sha256") != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("Generation-19 live release preflight drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    plan = g19.g17.load(args.plan)
    prior = engine.bulk
    try:
        engine.bulk = g19
        engine.run_controller(plan, out=args.out, proxy=args.proxy, runtime_gate_check=runtime_gate)
    finally:
        engine.bulk = prior
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
