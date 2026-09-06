"""Execute one reviewed atomic hosted GLM whole-task successor."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import hosted_glm_whole_task_successor_v1 as successor
from evals.fleet import self_hosted


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def run(controller: str, root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(controller, inventory, root)
    source_package_sha256 = os.environ.get("GLM_HOSTED_WHOLE_TASK_SOURCE_SHA256", "")
    plans = {
        name: successor.build_runtime_plan(name, inventory, root)
        for name in successor.CONTROLLERS
    }
    release = successor.load_runtime_release(plans, source_package_sha256)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    out = Path(plan["sfs_root"])
    provider = successor.AtomicWholeTaskClaims(plan, out, key=key)
    state: dict[str, Any] = {"stage": None, "item": None}

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        state.update(stage=stage, item=item)
        if stage == "07-claim-written":
            if item is None:
                raise RuntimeError("hosted GLM whole-task model boundary item is missing")
            provider.mark_model_boundary(item)

    prior = engine.bulk
    try:
        engine.bulk = successor
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda _plan: successor.validate_release(
                release,
                plans,
                source_package_sha256,
            ),
            claim_provider=provider,
            stage_observer=observe,
        )
    except Exception as exc:
        item = state["item"]
        body: dict[str, Any] = {
            "schema_version": "fleet-hosted-glm-whole-task-runtime-failure-v1",
            "status": "FAILED",
            "controller": controller,
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
        diagnostic_root = out.parent / f"{out.name}-runtime-diagnostic-v1"
        diagnostic_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):
            self_hosted.write_json_once(diagnostic_root / "FAILED.json", _seal(body))
        raise
    finally:
        engine.bulk = prior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=sorted(successor.CONTROLLERS), required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    run(args.controller, args.repo.resolve(), args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
