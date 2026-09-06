"""Held hosted GLM successor for untouched rank-29 attempts 3 and 4."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_universe as universe
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank29-a3a4-c2-successor-plan-v1"
CONTROLLER = "glm-hosted-r29-a3a4-successor"
SOURCE_CONTROLLER = "glm-hosted-s2"
JOB_NAME = "chris-glm53-exact100-hosted-r029-a3a4-successor-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = source.CLAIM_ROOT
LEASE_ROOT = source.LEASE_ROOT
LEASE_ENDPOINT_KEY = "glm-hosted-autocontinue-v1"
SERVING_LOAD_BLOCK = "glm-hosted-autocontinue-c2-v1"
EXPECTED_RANK = 29
SELECTED_ATTEMPTS = (3, 4)
EXECUTION_GENERATION = 2
SOURCE_FAILED_JOB_UID = "faf01255-0696-43e6-a47f-67802184986e"
SOURCE_FAILED_POD_UID = "72ce8163-fb2d-40ac-92da-da0dd578658d"
BLOCKED_A2_CLAIM_SHA = (
    "sha256:bd17483d6e18c033cf19a1277e172a481c2abb8e35cee3840be9ad53b8dffc72"
)
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
SHA256_RE = source.SHA256_RE
COMMIT_RE = source.COMMIT_RE
CANARY_GATE_SCHEMA = source.predecessor.CANARY_GATE_SCHEMA
RECONCILIATION_GATE_SCHEMA = source.predecessor.RECONCILIATION_GATE_SCHEMA


def _transform(plan: dict[str, Any], *, runtime: bool) -> dict[str, Any]:
    rows = sorted(
        (
            copy.deepcopy(row)
            for row in plan["attempts"]
            if row["selection_rank"] == EXPECTED_RANK
        ),
        key=lambda row: row["attempt"],
    )
    if [row["attempt"] for row in rows] != [1, 2, 3, 4]:
        raise ValueError("rank-29 whole-task authority drifted")
    source_generation = [
        {
            "attempt": row["attempt"],
            "cell_id": row["cell_id"],
            "execution_id": row["execution_id"],
            "run_id": row["run_id"],
        }
        for row in rows
        if row["attempt"] in SELECTED_ATTEMPTS
    ]
    selected = [row for row in rows if row["attempt"] in SELECTED_ATTEMPTS]
    for ordinal, row in enumerate(selected, 1):
        execution = universe.execution_for(row["cell_id"], EXECUTION_GENERATION)
        row.update(execution, ordinal=ordinal)
        row["run_id"] = row["run_id"].replace("-g1-", "-g2-")
        row["network"] = row["run_id"].removeprefix("chris-")[:63]

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
        new_session_count=2,
        attempts=selected,
        launch_authorized=runtime,
    )
    if "tasks" in body:
        body["tasks"] = [row for row in body["tasks"] if row["rank"] == EXPECTED_RANK]
        if len(body["tasks"]) != 1:
            raise ValueError("rank-29 runtime task authority drifted")
    body["execution"].update(
        workers=1,
        priority_class="fleet-serve-low",
        preemption_policy="Never",
        endpoint_lease={
            "lease_root": str(LEASE_ROOT),
            "endpoint_key": LEASE_ENDPOINT_KEY,
            "maximum_streams": 2,
        },
    )
    body["partition"] = {
        "whole_task_rank": EXPECTED_RANK,
        "attempt_1_accepted": True,
        "attempt_2_nonrepeatable_blocked": True,
        "attempt_2_claim_sha256": BLOCKED_A2_CLAIM_SHA,
        "attempt_2_source_job_deadline_exceeded": True,
        "source_failed_job_uid": SOURCE_FAILED_JOB_UID,
        "source_failed_pod_uid": SOURCE_FAILED_POD_UID,
        "selected_attempts": list(SELECTED_ATTEMPTS),
        "fresh_execution_generation": EXECUTION_GENERATION,
        "source_generation_1_retired_unclaimed": source_generation,
        "source_generation_1_retirement_reason": (
            "source_controller_deadline_exceeded_before_cell_claim"
        ),
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
        raise ValueError("unknown rank-29 hosted GLM successor controller")
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
            plan["execution"]["priority_class"] != "fleet-serve-low",
            plan["execution"]["preemption_policy"] != "Never",
            plan["new_session_count"] != 2,
            [row["attempt"] for row in plan["attempts"]] != [3, 4],
            any(row["execution_generation"] != 2 for row in plan["attempts"]),
            plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("rank-29 hosted GLM successor plan drifted")
    return {CONTROLLER: plan}
