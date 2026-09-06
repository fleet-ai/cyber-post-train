"""Execute rank 30 only after the v6-gated release passes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as source_runtime
from evals.fleet import hosted_glm_rank30_single_slot_v5 as successor
from evals.fleet import hosted_glm_whole_task_engine_v1 as engine


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = successor.load(source_runtime.INVENTORY_PATH)
    plan = successor.build_runtime_plan(successor.CONTROLLER, inventory, root)
    source_sha = os.environ.get("GLM_HOSTED_R30_SOURCE_SHA256", "")
    release = successor.load(successor.RELEASE_PATH)

    def current_gate() -> None:
        successor.validate_release(release, plan, source_sha)
        successor.validate_current_peer()

    current_gate()
    key = os.environ.get("FLEET_API_KEY")
    if not key:
        raise RuntimeError("FLEET_API_KEY is required")
    out = Path(plan["sfs_root"])
    provider = successor.AtomicWholeTaskClaims(plan, out, key=key)
    prior = engine.bulk

    def observe(stage: str, item: dict[str, Any] | None) -> None:
        if stage == "07-claim-written":
            if item is None:
                raise RuntimeError("rank-30 model boundary item is missing")
            provider.mark_model_boundary(item)

    try:
        engine.bulk = successor
        return engine.run_controller(
            plan,
            out=out,
            proxy=proxy,
            runtime_gate_check=lambda _plan: current_gate(),
            claim_provider=provider,
            stage_observer=observe,
        )
    finally:
        engine.bulk = prior


def main() -> int:
    root = Path(os.environ.get("REPO_ROOT", "/workspace/cyber-post-train"))
    run(root, root / "evals/fleet/fixed_proxy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
