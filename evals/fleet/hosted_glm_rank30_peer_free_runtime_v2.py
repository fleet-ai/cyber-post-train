"""Execute the reviewed peer-free rank-30 hosted GLM successor."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_peer_free_successor_v2 as successor
from evals.fleet import hosted_glm_whole_task_engine_v1 as engine
from evals.fleet import self_hosted


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    source_sha = os.environ.get("GLM_HOSTED_R30_SOURCE_SHA256", "")
    release = successor.load_runtime_release(plan, source_sha)
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    out = Path(plan["sfs_root"])
    provider = successor.AtomicWholeTaskClaims(plan, out, key=key)
    state: dict[str, Any] = {"stage": "00-runtime-entry", "item": None}

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        state.update(stage=stage, item=item)
        if stage == "07-claim-written":
            if item is None:
                raise RuntimeError("peer-free rank30 model boundary item is missing")
            provider.mark_model_boundary(item)

    prior_bulk = engine.bulk
    try:
        engine.bulk = successor
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda _plan: successor.validate_release(
                release, plan, source_sha
            ),
            claim_provider=provider,
            stage_observer=observe,
        )
    except Exception:
        body: dict[str, Any] = {
            "schema_version": "fleet-hosted-glm-rank30-peer-free-runtime-failure-v2",
            "status": "FAILED_CLOSED",
            "controller": successor.CONTROLLER,
            "plan_sha256": plan["plan_sha256"],
            "last_completed_stage": state["stage"],
            "reservation_committed": out.joinpath("RESERVATION.json").is_file(),
            "provider_session_model_boundary_crossed": out.joinpath(
                "model-boundaries"
            ).exists(),
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        }
        item = state["item"]
        if item is not None:
            body.update(
                cell_id=item["cell_id"],
                execution_id=item["execution_id"],
                selection_rank=item["selection_rank"],
                attempt=item["attempt"],
            )
        diagnostic = out.parent / f"{out.name}-runtime-diagnostic-v2"
        diagnostic.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(FileExistsError):
            self_hosted.write_json_once(diagnostic / "FAILED.json", _seal(body))
        raise
    finally:
        engine.bulk = prior_bulk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
