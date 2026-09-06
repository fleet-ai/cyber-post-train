"""Runtime wrapper for the held rank-15/rank-16 hosted successors."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import qwen_hosted_whole_task_successor_v1 as successor
from evals.fleet import self_hosted


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def run(plan: dict[str, Any], *, out: Path, proxy: Path, diagnostic_root: Path) -> dict[str, Any]:
    if out.exists() or out.is_symlink():
        raise RuntimeError("hosted whole-task create-once output root already exists")
    plans = successor.build_plans(Path(plan["repo_root"]))
    release = successor.load_runtime_release(plans)
    if plan != plans.get(plan.get("controller")):
        raise RuntimeError("hosted whole-task runtime plan drifted")
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    provider = successor.AtomicWholeTaskClaims(plan, out, key=key)
    state: dict[str, Any] = {"stage": None, "item": None}

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        state.update(stage=stage, item=item)

    prior = engine.bulk
    try:
        engine.bulk = successor
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda _plan: successor.validate_release(release, plans),
            claim_provider=provider,
            stage_observer=observe,
        )
    except Exception as exc:
        item = state["item"]
        body: dict[str, Any] = {
            "schema_version": "fleet-qwen38-hosted-whole-task-runtime-failure-v1",
            "status": "FAILED",
            "controller": plan["controller"],
            "plan_sha256": plan["plan_sha256"],
            "last_completed_stage": state["stage"],
            "reservation_committed": out.joinpath("RESERVATION.json").is_file(),
            "error_type": type(exc).__name__,
            "error_sha256": self_hosted.sha256(str(exc).encode()),
            "job_uid": os.environ.get("JOB_UID"),
            "pod_uid": os.environ.get("POD_UID"),
            "scores_included": False,
            "prompts_or_traces_included": False,
            "credentials_included": False,
        }
        if item is not None:
            body.update(
                cell_id=item["cell_id"],
                execution_id=item["execution_id"],
                selection_rank=item["selection_rank"],
                attempt=item["attempt"],
            )
        diagnostic_root.mkdir(mode=0o700, parents=True, exist_ok=True)
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
    run(
        successor.load(args.plan),
        out=args.out,
        proxy=args.proxy,
        diagnostic_root=args.diagnostic_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
