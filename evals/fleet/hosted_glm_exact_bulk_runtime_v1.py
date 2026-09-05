"""Run one released hosted GLM bulk stream with the reviewed v3 engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_v1 as bulk
from evals.fleet import self_hosted

INVENTORY_PATH = Path("/mnt/sfs/jobs/chris-cyber-exact100-pass4-inventory-v2/TERMINAL.json")
RELEASE_PATH = Path("/mnt/sfs/jobs/chris-glm53-exact100-hosted-bulk-v1-release/RELEASE.json")
RELEASE_SCHEMA = "fleet-hosted-glm-exact-bulk-release-v1"


def _canary_gate() -> tuple[dict[str, Any], dict[str, Any]]:
    accepted = bulk.load(bulk.CANARY_ROOT / "ACCEPTED.json")
    terminal = bulk.load(bulk.CANARY_ROOT / "TERMINAL.json")
    if any(
        (
            accepted.get("schema_version") != "fleet-exact-pass4-bulk-cell-accepted-v3",
            accepted.get("accepted") is not True,
            accepted.get("credited") is not True,
            accepted.get("retry_allowed") is not False,
            accepted.get("selection_rank") != 1,
            accepted.get("attempt") != 1,
            accepted.get("receipt_sha256")
            != self_hosted.digest_without(accepted, "receipt_sha256"),
            terminal.get("schema_version") != "fleet-hosted-glm-exact-canary-terminal-v1",
            terminal.get("status") != "ACCEPTED",
            terminal.get("cell_id") != accepted.get("cell_id"),
            terminal.get("execution_id") != accepted.get("execution_id"),
            terminal.get("run_id") != accepted.get("run_id"),
            terminal.get("accepted_receipt_sha256") != accepted.get("receipt_sha256"),
            terminal.get("session_id") != accepted.get("session_id"),
            terminal.get("verifier_execution_id") != accepted.get("verifier_execution_id"),
            terminal.get("receipt_sha256")
            != self_hosted.digest_without(terminal, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted GLM canary acceptance gate drifted")
    return accepted, terminal


def validate_release(plan: dict[str, Any]) -> None:
    _, terminal = _canary_gate()
    release = bulk.load(RELEASE_PATH)
    all_plans = bulk.validate_all(Path(plan["repo_root"]))
    plan_shas = {name: value["plan_sha256"] for name, value in all_plans.items()}
    execution_ids = sorted(
        row["execution_id"] for value in all_plans.values() for row in value["attempts"]
    )
    if any(
        (
            release.get("schema_version") != RELEASE_SCHEMA,
            release.get("status") != "CLEAR",
            release.get("canary_terminal_receipt_sha256") != terminal["receipt_sha256"],
            release.get("controller_plan_sha256s") != plan_shas,
            release.get("execution_ids_sha256")
            != self_hosted.sha256(self_hosted.canonical_json(execution_ids)),
            release.get("cell_count") != 394,
            release.get("checked_immediately_before_create") is not True,
            release.get("fleet_session_collisions") != 0,
            release.get("global_claim_collisions") != 0,
            release.get("sfs_output_collisions") != 0,
            release.get("kubernetes_object_collisions") != 0,
            release.get("active_or_accepted_collisions") != 0,
            release.get("mutation_calls") != 0,
            release.get("scores_read") is not False,
            release.get("prompts_traces_flags_read") is not False,
            release.get("receipt_sha256") != self_hosted.digest_without(release, "receipt_sha256"),
        )
    ):
        raise RuntimeError("hosted GLM bulk release gate drifted")


def run(controller: str, root: Path, proxy: Path) -> dict[str, Any]:
    inventory = bulk.load(INVENTORY_PATH)
    plan = bulk.build_runtime_plan(controller, inventory, root)
    # The reviewed v3 engine uses its module authority dynamically.  Bind it to
    # this exact four-stream authority before rebuilding or writing a claim.
    engine.bulk = bulk
    return engine.run_controller(
        plan,
        out=Path(plan["sfs_root"]),
        proxy=proxy,
        runtime_gate_check=validate_release,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=sorted(bulk.CONTROLLERS), required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    run(args.controller, args.repo.resolve(), args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
