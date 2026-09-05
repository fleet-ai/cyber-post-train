"""Held whole-task Qwen hosted bulk after G18 canary acceptance.

Ranks 2 and 3 remain reserved for dedicated and laptop blocks. Rank 4 contains
the accepted historical canary but is excluded to keep this successor strictly
whole-task partitioned. Rank 5 is owned by G18 and likewise excluded.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as exact
from evals.fleet import qwen_bulk_generation16 as g17
from evals.fleet import self_hosted

EXECUTION_GENERATION = 19
ALLOWED_RANKS = [1, *range(6, 101)]
CONTROLLERS = {
    "qwen-a": {
        "ranks": ALLOWED_RANKS[::2],
        "job_name": "chris-q38-ac-exact100-g19-a192-v1",
        "configmap_name": "chris-q38-ac-exact100-g19-a192-run-v1",
    },
    "qwen-b": {
        "ranks": ALLOWED_RANKS[1::2],
        "job_name": "chris-q38-ac-exact100-g19-b192-v1",
        "configmap_name": "chris-q38-ac-exact100-g19-b192-run-v1",
    },
}


def build_plans(inventory: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    g17.validate_inventory_gate(inventory, root)
    sources = [
        g17.build_runtime_plan(controller, inventory, root) for controller in g17.CONTROLLERS
    ]
    tasks = {row["rank"]: row for source in sources for row in source["tasks"]}
    template = sources[0]
    campaign = exact.read_object(
        root / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
    )
    universe = exact.build_universe(campaign, root)
    cells = {
        (row["selection_rank"], row["attempt"]): row
        for row in universe["cells"]
        if row["model"] == "qwen3.8-27b"
    }
    plans: dict[str, dict[str, Any]] = {}
    for controller, authority in CONTROLLERS.items():
        attempts = []
        for ordinal, (rank, attempt) in enumerate(
            ((rank, attempt) for rank in authority["ranks"] for attempt in range(1, 5)),
            1,
        ):
            cell = cells[(rank, attempt)]
            execution = exact.execution_for(cell["cell_id"], EXECUTION_GENERATION)
            run_id = (
                f"chris-q38-ac-g19-{controller[-1]}-r{rank:03d}-a{attempt}-"
                f"{execution['execution_id'][7:15]}"
            )
            attempts.append(
                {
                    "ordinal": ordinal,
                    "cell_id": cell["cell_id"],
                    "selection_rank": rank,
                    "attempt": attempt,
                    "task_version_id": cell["task_version_id"],
                    "execution_generation": EXECUTION_GENERATION,
                    "execution_id": execution["execution_id"],
                    "task_key": tasks[rank]["task"]["key"],
                    "environment_version_id": tasks[rank]["environment"]["version_id"],
                    "run_id": run_id,
                    "network": run_id.removeprefix("chris-")[:63],
                }
            )
        execution = copy.deepcopy(template["execution"])
        execution["endpoint_lease"] = {
            "lease_root": g17.LEASE_ROOT,
            "endpoint_key": "qwen-hosted-autocontinue-v1",
            "maximum_streams": 2,
        }
        body = {
            "schema_version": "fleet-qwen-generation19-hosted-bulk-plan-v1",
            "controller": controller,
            "campaign_id": authority["job_name"],
            "source_job_id": authority["job_name"],
            "sfs_root": f"/mnt/sfs/jobs/{authority['job_name']}",
            "repo_root": ".",
            "inventory_receipt": {"receipt_sha256": inventory["receipt_sha256"]},
            "inventory_receipt_sha256": inventory["receipt_sha256"],
            "model": copy.deepcopy(template["model"]),
            "harness": copy.deepcopy(template["harness"]),
            "authority": copy.deepcopy(template["authority"]),
            "serving_block": "qwen-hosted-autocontinue-v1",
            "tasks": [copy.deepcopy(tasks[rank]) for rank in authority["ranks"]],
            "attempts": attempts,
            "execution": execution,
            "treatment": copy.deepcopy(template["treatment"]),
            "release_required": True,
            "launch_authorized": False,
            "release_gates": {
                "g18_validated_acceptance_required": True,
                "fresh_score_blind_global_ledger_required": True,
            },
            "privacy": copy.deepcopy(template["privacy"]),
        }
        plans[controller] = {
            **body,
            "plan_sha256": self_hosted.digest_without(body, "plan_sha256"),
        }
    validate(plans)
    return plans


def validate(plans: dict[str, dict[str, Any]]) -> None:
    identities: set[tuple[int, int]] = set()
    rank_owners: dict[int, str] = {}
    for controller, plan in plans.items():
        if any(
            (
                plan.get("schema_version") != "fleet-qwen-generation19-hosted-bulk-plan-v1",
                plan.get("controller") != controller,
                plan.get("launch_authorized") is not False,
                plan.get("execution", {}).get("endpoint_lease", {}).get("maximum_streams") != 2,
                plan.get("plan_sha256") != self_hosted.digest_without(plan, "plan_sha256"),
                len(plan.get("attempts") or []) != 192,
                len(plan.get("tasks") or []) != 48,
            )
        ):
            raise ValueError("Generation-19 hosted plan drifted")
        for row in plan["attempts"]:
            identity = (row["selection_rank"], row["attempt"])
            if identity in identities:
                raise ValueError("Generation-19 partitions overlap")
            identities.add(identity)
            if rank_owners.setdefault(row["selection_rank"], controller) != controller:
                raise ValueError("Generation-19 split a task across controllers")
    expected = {(rank, attempt) for rank in ALLOWED_RANKS for attempt in range(1, 5)}
    if identities != expected or set(rank_owners) & {2, 3, 4, 5}:
        raise ValueError("Generation-19 whole-task coverage drifted")
