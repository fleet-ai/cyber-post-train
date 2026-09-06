"""Three-cell hosted GLM successor after the validated rank-3/a1 canary."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank3-a2a4-c2-successor-plan-v1"
CONTROLLER = "glm-hosted-r3-successor"
SOURCE_CONTROLLER = "glm-hosted-s1"
JOB_NAME = "chris-glm53-exact100-hosted-r003-a2a4-successor-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = source.CLAIM_ROOT
LEASE_ROOT = source.LEASE_ROOT
LEASE_ENDPOINT_KEY = "glm-hosted-autocontinue-v1"
SERVING_LOAD_BLOCK = "glm-hosted-autocontinue-c2-v1"
EXPECTED_RANK = 3
SELECTED_ATTEMPTS = (2, 3, 4)
CANARY_JOB = "chris-glm53-exact100-hosted-r003-a1-canary-v2"
CANARY_JOB_UID = "c8de84a4-ef6a-4da3-b630-5de8ba0cf0e0"
CANARY_ACCEPTED_SHA = "sha256:e73426d9855ce7bc7d85887ab18f5cacb78226a6a4e475a32afea40913630c26"
CANARY_VALIDATION_SHA = "sha256:76590bed8308ac5ad080317380d71c851319dd2ab1ac07f7c046c25522daacd3"
CONTROLLERS = {
    CONTROLLER: {
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
        "model": "glm-5.3",
        "serving_block": LEASE_ENDPOINT_KEY,
        "task_ranks": [EXPECTED_RANK],
    }
}
load = source.load
validate_inventory_gate = source.validate_inventory_gate


def _transform(plan: dict[str, Any], *, runtime: bool) -> dict[str, Any]:
    rank_rows = [
        copy.deepcopy(row)
        for row in plan["attempts"]
        if row["selection_rank"] == EXPECTED_RANK
    ]
    rank_rows.sort(key=lambda row: row["attempt"])
    if [row["attempt"] for row in rank_rows] != [1, 2, 3, 4]:
        raise ValueError("rank-3 whole-task authority drifted")
    selected = [row for row in rank_rows if row["attempt"] in SELECTED_ATTEMPTS]
    for ordinal, row in enumerate(selected, 1):
        row["ordinal"] = ordinal

    body = copy.deepcopy(plan)
    body.update(
        schema_version=SCHEMA,
        campaign_id=JOB_NAME,
        source_job_id=JOB_NAME,
        controller=CONTROLLER,
        job_name=JOB_NAME,
        configmap_name=CONFIGMAP_NAME,
        sfs_root=str(SFS_ROOT),
        serving_block=LEASE_ENDPOINT_KEY,
        serving_load_block=SERVING_LOAD_BLOCK,
        task_count=1,
        new_session_count=3,
        attempts=selected,
        launch_authorized=runtime,
    )
    if "tasks" in body:
        body["tasks"] = [row for row in body["tasks"] if row["rank"] == EXPECTED_RANK]
        if len(body["tasks"]) != 1:
            raise ValueError("rank-3 runtime task authority drifted")
    body["execution"].update(
        workers=1,
        priority_class="fleet-infra-quiet",
        preemption_policy="Never",
        endpoint_lease={
            "lease_root": str(LEASE_ROOT),
            "endpoint_key": LEASE_ENDPOINT_KEY,
            "maximum_streams": 2,
        },
    )
    body["partition"] = {
        "whole_task_rank": EXPECTED_RANK,
        "canary_attempt_1_accepted_required": True,
        "canary_job_uid": CANARY_JOB_UID,
        "canary_accepted_receipt_sha256": CANARY_ACCEPTED_SHA,
        "canary_validation_receipt_sha256": CANARY_VALIDATION_SHA,
        "selected_attempts": list(SELECTED_ATTEMPTS),
        "accepted_active_or_blocked_cells_excluded": True,
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
    return _transform(source.validate_all(root)[SOURCE_CONTROLLER], runtime=False)


def build_runtime_plan(
    controller: str, inventory: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown rank-3 hosted GLM successor controller")
    return _transform(source.build_runtime_plan(SOURCE_CONTROLLER, inventory, root), runtime=True)


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = build_plan(root)
    if any(
        (
            plan["launch_authorized"] is not False,
            plan["model"]["served_id"] != "glm-5.3",
            plan["harness"]["version"] != "1.18.27",
            plan["execution"]["required_task_tools"] != ["bash", "submit_report"],
            plan["execution"]["endpoint_lease"]["maximum_streams"] != 2,
            plan["execution"]["preemption_policy"] != "Never",
            plan["new_session_count"] != 3,
            [row["attempt"] for row in plan["attempts"]] != [2, 3, 4],
            plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("rank-3 hosted GLM successor plan drifted")
    return {CONTROLLER: plan}
