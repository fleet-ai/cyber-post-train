"""Instrumented runtime for the retry-safe Qwen G19 v2 controllers."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_generation19_v2 as g19
from evals.fleet import self_hosted


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def _stage_path(root: Path, stage: str, item: dict[str, Any] | None) -> Path:
    prefix = ""
    if item is not None:
        prefix = f"r{item['selection_rank']:03d}-a{item['attempt']}-"
    return root / f"STAGE-{prefix}{stage}.json"


def _write_stage(root: Path, stage: str, item: dict[str, Any] | None) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "schema_version": "fleet-qwen-generation19-runtime-stage-v1",
        "stage": stage,
        "claim_written": stage in {"07-claim-written", "08-model-runner-entered"},
        "model_call_started": stage == "08-model-runner-entered",
        "prompts_traces_flags_or_scores_included": False,
        "credentials_included": False,
    }
    if item is not None:
        body.update(
            cell_id=item["cell_id"],
            execution_id=item["execution_id"],
            selection_rank=item["selection_rank"],
            attempt=item["attempt"],
        )
    self_hosted.write_json_once(_stage_path(root, stage, item), _seal(body))


def runtime_gate(plan: dict[str, Any], bulk_module: Any = g19) -> None:
    path = Path(os.environ.get("G19_PREFLIGHT_PATH", ""))
    expected = os.environ.get("G19_PREFLIGHT_SHA256")
    receipt = g19.prior.g17.load(path)
    plans = bulk_module.validate_all(Path(plan["repo_root"]))
    execution_ids = sorted(
        row["execution_id"] for built in plans.values() for row in built["attempts"]
    )
    if any(
        (
            receipt.get("schema_version")
            != "fleet-qwen-generation19-hosted-bulk-preflight-v1",
            receipt.get("status") != "CLEAR",
            receipt.get("planned_cells") != 384,
            receipt.get("planned_execution_ids_sha256")
            != self_hosted.sha256(self_hosted.canonical_json(execution_ids)),
            receipt.get("mutation_calls") != 0,
            receipt.get("receipt_sha256") != expected,
            receipt.get("receipt_sha256")
            != self_hosted.digest_without(receipt, "receipt_sha256"),
        )
    ):
        raise RuntimeError("Generation-19 v2 live release preflight drifted")


def run(
    plan: dict[str, Any],
    *,
    out: Path,
    proxy: Path,
    diagnostic_root: Path,
    bulk_module: Any = g19,
) -> dict[str, Any]:
    diagnostic_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    state: dict[str, Any] = {"stage": None, "item": None}

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        _write_stage(diagnostic_root, stage, item)
        state.update(stage=stage, item=item)

    prior = engine.bulk
    try:
        engine.bulk = bulk_module
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda value: runtime_gate(value, bulk_module),
            stage_observer=observe,
        )
    except Exception as exc:
        item = state["item"]
        body: dict[str, Any] = {
            "schema_version": "fleet-qwen-generation19-runtime-failure-v1",
            "status": "FAILED",
            "last_completed_stage": state["stage"],
            "error_type": type(exc).__name__,
            "error_sha256": self_hosted.sha256(str(exc).encode()),
            "job_uid": os.environ.get("JOB_UID"),
            "pod_uid": os.environ.get("POD_UID"),
            "claim_written": state["stage"] in {"07-claim-written", "08-model-runner-entered"},
            "model_call_started": state["stage"] == "08-model-runner-entered",
            "prompts_traces_flags_or_scores_included": False,
            "credentials_included": False,
        }
        if item is not None:
            body.update(
                cell_id=item["cell_id"],
                execution_id=item["execution_id"],
                selection_rank=item["selection_rank"],
                attempt=item["attempt"],
            )
        with contextlib.suppress(FileExistsError):
            self_hosted.write_json_once(diagnostic_root / "FAILED.json", _seal(body))
        raise
    finally:
        engine.bulk = prior


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proxy", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    args = parser.parse_args()
    plan = g19.prior.g17.load(args.plan)
    run(plan, out=args.out, proxy=args.proxy, diagnostic_root=args.diagnostic_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
