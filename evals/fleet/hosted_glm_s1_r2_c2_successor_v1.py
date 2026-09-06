"""Complete rank-2 GLM task successor under the proven hosted cap-two block."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s1-r2-c2-successor-plan-v1"
CONTROLLER = "glm-hosted-s1"
JOB_NAME = "chris-glm53-exact100-hosted-s1-r002-c2-successor-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = source.CLAIM_ROOT
LEASE_ROOT = source.LEASE_ROOT
LEASE_ENDPOINT_KEY = "glm-hosted-autocontinue-v1"
SERVING_LOAD_BLOCK = "glm-hosted-autocontinue-c2-v1"
EXPECTED_RANK = 2
EXPECTED_CELLS = {(EXPECTED_RANK, attempt) for attempt in range(1, 5)}


def load(path: Path) -> dict[str, Any]:
    return source.load(path)


def validate_inventory_gate(value: dict[str, Any], root: Path) -> None:
    source.validate_inventory_gate(value, root)


def _transform(plan: dict[str, Any], *, runtime: bool) -> dict[str, Any]:
    selected = [
        copy.deepcopy(row)
        for row in plan["attempts"]
        if (row["selection_rank"], row["attempt"]) in EXPECTED_CELLS
    ]
    selected.sort(key=lambda row: row["attempt"])
    if {(row["selection_rank"], row["attempt"]) for row in selected} != EXPECTED_CELLS:
        raise ValueError("rank-2 complete-task authority drifted")
    for ordinal, row in enumerate(selected, 1):
        row["ordinal"] = ordinal
    body = copy.deepcopy(plan)
    body.update(
        {
            "schema_version": SCHEMA,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "sfs_root": str(SFS_ROOT),
            "task_count": 1,
            "new_session_count": 4,
            "attempts": selected,
            "serving_load_block": SERVING_LOAD_BLOCK,
            "launch_authorized": runtime,
        }
    )
    if "tasks" in body:
        body["tasks"] = [row for row in body["tasks"] if row["rank"] == EXPECTED_RANK]
        if len(body["tasks"]) != 1:
            raise ValueError("rank-2 runtime task authority drifted")
    body["execution"]["workers"] = 1
    body["execution"]["endpoint_lease"] = {
        "lease_root": str(LEASE_ROOT),
        "endpoint_key": LEASE_ENDPOINT_KEY,
        "maximum_streams": 2,
    }
    body["partition"] = {
        "whole_task_rank": EXPECTED_RANK,
        "all_four_attempts": True,
        "rank1_attempt4_blocked_and_excluded": True,
        "accepted_or_active_cells_excluded": True,
    }
    body["concurrency_treatment"] = {
        "serving_load_block": SERVING_LOAD_BLOCK,
        "maximum_scored_streams": 2,
        "pool_with_other_concurrency_without_review": False,
    }
    body.pop("plan_sha256", None)
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body


def build_plan(root: Path) -> dict[str, Any]:
    return _transform(source.validate_all(root)[CONTROLLER], runtime=False)


def build_runtime_plan(inventory: dict[str, Any], root: Path) -> dict[str, Any]:
    return _transform(source.build_runtime_plan(CONTROLLER, inventory, root), runtime=True)


def validate_all(root: Path) -> dict[str, Any]:
    plan = build_plan(root)
    lease = plan["execution"]["endpoint_lease"]
    if any(
        (
            plan["launch_authorized"] is not False,
            plan["model"]["served_id"] != "glm-5.3",
            plan["harness"]["version"] != "1.18.27",
            plan["execution"]["required_task_tools"] != ["bash", "submit_report"],
            lease["maximum_streams"] != 2,
            plan["new_session_count"] != 4,
            plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("rank-2 hosted successor drifted")
    return plan

