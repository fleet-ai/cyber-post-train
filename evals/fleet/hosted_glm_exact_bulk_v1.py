"""Held four-stream hosted GLM successor after the rank-1 canary.

This changes execution grouping only.  It preserves every available successor
identity from the exact pass@4 authority, excludes the dedicated rank-12 task
boundary, excludes rank-1/attempt-1 (the prerequisite canary), and fails closed
around rank-13/attempt-1 because the predecessor treated that historical canary
as accepted and therefore defines no fresh successor execution for it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_v3 as predecessor
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-exact-bulk-plan-v1"
CANARY_ROOT = Path("/mnt/sfs/jobs/chris-glm53-exact100-hosted-r001-a1-canary-v4")
UNMATERIALIZED_SUCCESSOR_CELL = (13, 1)
CLAIM_ROOT = predecessor.CLAIM_ROOT
LEASE_ROOT = predecessor.LEASE_ROOT
SHA256_RE = predecessor.SHA256_RE
COMMIT_RE = predecessor.COMMIT_RE
CONTROLLERS = {
    f"glm-hosted-s{stream}": {
        "model": "glm-5.3",
        "job_name": f"chris-glm53-exact100-hosted-s{stream}-v1",
        "configmap_name": f"chris-glm53-exact100-hosted-s{stream}-v1-run",
        "serving_block": "glm-hosted-autocontinue-v1",
        "task_ranks": list(range(first, last + 1)),
    }
    for stream, (first, last) in enumerate(((1, 25), (26, 50), (51, 75), (76, 100)), 1)
}


def load(path: Path) -> dict[str, Any]:
    return predecessor.load(path)


def validate_inventory_gate(value: dict[str, Any], root: Path) -> None:
    predecessor.validate_inventory_gate(value, root)


def _source_plans(root: Path) -> dict[str, dict[str, Any]]:
    return {
        key: value
        for key, value in predecessor.validate_all(root).items()
        if value["model"]["served_id"] == "glm-5.3"
    }


def _source_rows(root: Path) -> dict[tuple[int, int], tuple[str, dict[str, Any]]]:
    rows: dict[tuple[int, int], tuple[str, dict[str, Any]]] = {}
    for controller, plan in _source_plans(root).items():
        for row in plan["attempts"]:
            key = (row["selection_rank"], row["attempt"])
            if key in rows:
                raise ValueError("source GLM statistical cell overlap")
            rows[key] = (controller, copy.deepcopy(row))
    if len(rows) != 399 or UNMATERIALIZED_SUCCESSOR_CELL in rows:
        raise ValueError("source GLM successor authority is not the exact 399-cell plan")
    return rows


def selected_keys(controller: str) -> list[tuple[int, int]]:
    source = CONTROLLERS.get(controller)
    if source is None:
        raise ValueError("unknown hosted GLM stream")
    keys = []
    for rank in source["task_ranks"]:
        if rank == 12:
            continue
        attempts = (2, 3, 4) if rank == 1 else (1, 2, 3, 4)
        keys.extend(
            (rank, attempt)
            for attempt in attempts
            if (rank, attempt) != UNMATERIALIZED_SUCCESSOR_CELL
        )
    return keys


def build_plan(controller: str, root: Path) -> dict[str, Any]:
    source = CONTROLLERS.get(controller)
    if source is None:
        raise ValueError("unknown hosted GLM stream")
    source_plans = _source_plans(root)
    rows = _source_rows(root)
    attempts = []
    for ordinal, key in enumerate(selected_keys(controller), 1):
        source_controller, row = rows[key]
        attempts.append({**row, "ordinal": ordinal, "source_controller": source_controller})
    compact_template = source_plans["glm-a"]
    runtime_template = predecessor.load(root / predecessor.RUNTIME_TEMPLATE_PATHS["glm-5.3"])
    harness = copy.deepcopy(runtime_template["harness"])
    harness["compaction_headroom_tokens"] = compact_template["treatment"][
        "compaction_headroom_tokens"
    ]
    execution = copy.deepcopy(compact_template["execution"])
    execution.update(
        {
            "required_task_tools": ["bash", "submit_report"],
            "required_task_tool_catalog_sha256": runtime_template["execution"][
                "required_task_tool_catalog_sha256"
            ],
            "training_data_eligible": False,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "campaign_id": source["job_name"],
        "source_job_id": source["job_name"],
        "controller": controller,
        "job_name": source["job_name"],
        "configmap_name": source["configmap_name"],
        "sfs_root": f"/mnt/sfs/jobs/{source['job_name']}",
        "repo_root": ".",
        "model": copy.deepcopy(runtime_template["model"]),
        "harness": harness,
        "serving_block": source["serving_block"],
        "task_count": len({row["selection_rank"] for row in attempts}),
        "new_session_count": len(attempts),
        "attempts": attempts,
        "execution": {
            **execution,
            "workers": 1,
            "attempts_per_task_sequential": True,
            "same_task_max_inflight": 1,
            "global_execution_claim_before_model_call": True,
            "claim_root": CLAIM_ROOT,
            "endpoint_lease": {
                "lease_root": LEASE_ROOT,
                "endpoint_key": source["serving_block"],
                "maximum_streams": 4,
            },
            "priority_class": "fleet-serve-low",
            "preemption_policy": "Never",
        },
        "treatment": copy.deepcopy(compact_template["treatment"]),
        "prerequisite": {
            "canary_root": str(CANARY_ROOT),
            "selection_rank": 1,
            "attempt": 1,
            "accepted_required": True,
        },
        "partition": {
            "rank12_reserved_for_dedicated": True,
            "rank1_attempt1_reserved_for_canary": True,
            "rank13_attempt1_requires_fresh_reviewed_successor": True,
            "complete_task_boundaries_otherwise": True,
        },
        "launch_authorized": False,
        "privacy": copy.deepcopy(compact_template["privacy"]),
    }
    return {**body, "plan_sha256": self_hosted.digest_without(body, "plan_sha256")}


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plans = {name: build_plan(name, root) for name in CONTROLLERS}
    observed: set[tuple[int, int]] = set()
    for plan in plans.values():
        if (
            plan["schema_version"] != SCHEMA
            or plan["model"]["served_id"] != "glm-5.3"
            or plan["execution"]["endpoint_lease"]["maximum_streams"] != 4
            or plan["execution"]["priority_class"] != "fleet-serve-low"
            or plan["execution"]["preemption_policy"] != "Never"
            or plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256")
        ):
            raise ValueError("hosted GLM bulk plan drifted")
        for row in plan["attempts"]:
            key = (row["selection_rank"], row["attempt"])
            if key in observed:
                raise ValueError("hosted GLM bulk partition overlap")
            observed.add(key)
    expected = {
        (rank, attempt)
        for rank in range(1, 101)
        if rank != 12
        for attempt in ((2, 3, 4) if rank == 1 else (1, 2, 3, 4))
        if (rank, attempt) != UNMATERIALIZED_SUCCESSOR_CELL
    }
    if observed != expected or len(observed) != 394:
        raise ValueError("hosted GLM bulk partition is not the exact reviewed 394-cell tail")
    return plans


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    validate_inventory_gate(inventory_receipt, root)
    plan = validate_all(root)[controller]
    sources = {
        name: predecessor.build_runtime_plan(name, inventory_receipt, root)
        for name in ("glm-a", "glm-b")
    }
    tasks_by_rank = {
        row["rank"]: copy.deepcopy(row) for source in sources.values() for row in source["tasks"]
    }
    runtime = copy.deepcopy(plan)
    selected_ranks = sorted({key[0] for key in selected_keys(controller)})
    runtime["tasks"] = [tasks_by_rank[rank] for rank in selected_ranks]
    runtime["inventory_receipt"] = copy.deepcopy(inventory_receipt)
    runtime["inventory_receipt_sha256"] = inventory_receipt["receipt_sha256"]
    runtime["authority"] = copy.deepcopy(sources["glm-a"]["authority"])
    runtime["launch_authorized"] = True
    runtime.pop("plan_sha256")
    runtime["plan_sha256"] = self_hosted.digest_without(runtime, "plan_sha256")
    return runtime
