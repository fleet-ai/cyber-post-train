"""One-cell GLM stream-2 canary for an explicit concurrency-two serving block."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-s2-c2-canary-plan-v1"
CONTROLLER = "glm-hosted-s2"
JOB_NAME = "chris-glm53-exact100-hosted-s2-c2-canary-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
S1_ROOT = Path("/mnt/sfs/jobs/chris-glm53-exact100-hosted-s1-v1")
SERVING_LOAD_BLOCK = "glm-hosted-autocontinue-c2-v1"
LEASE_ENDPOINT_KEY = "glm-hosted-autocontinue-v1"
EXPECTED_CELL = (26, 1)
CLAIM_ROOT = source.CLAIM_ROOT
LEASE_ROOT = source.LEASE_ROOT
SHA256_RE = source.SHA256_RE
COMMIT_RE = source.COMMIT_RE
CONTROLLERS = {
    CONTROLLER: {
        **copy.deepcopy(source.CONTROLLERS[CONTROLLER]),
        # Keep the source campaign name: the runtime engine derives controller
        # identity from it.  Job and output identities are separate fields.
        "job_name": source.CONTROLLERS[CONTROLLER]["job_name"],
    }
}


def load(path: Path) -> dict[str, Any]:
    return source.load(path)


def validate_inventory_gate(value: dict[str, Any], root: Path) -> None:
    source.validate_inventory_gate(value, root)


def _transform(plan: dict[str, Any], *, runtime: bool) -> dict[str, Any]:
    selected = [
        copy.deepcopy(row)
        for row in plan["attempts"]
        if (row["selection_rank"], row["attempt"]) == EXPECTED_CELL
    ]
    if len(selected) != 1:
        raise ValueError("GLM s2 concurrency-two canary cell authority drifted")
    selected[0]["ordinal"] = 1
    body = copy.deepcopy(plan)
    body.update(
        {
            "schema_version": SCHEMA,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "sfs_root": str(SFS_ROOT),
            "task_count": 1,
            "new_session_count": 1,
            "attempts": selected,
            "serving_load_block": SERVING_LOAD_BLOCK,
            "launch_authorized": runtime,
        }
    )
    body["execution"]["workers"] = 1
    body["execution"]["endpoint_lease"] = {
        "lease_root": str(LEASE_ROOT),
        "endpoint_key": LEASE_ENDPOINT_KEY,
        "maximum_streams": 2,
    }
    body["prerequisite"] = {
        "s1_root": str(S1_ROOT),
        "s1_first_planned_cell_accepted_required": True,
        "fresh_exact_ledger_required": True,
        "shared_endpoint_lease_required": True,
    }
    body["concurrency_treatment"] = {
        "serving_load_block": SERVING_LOAD_BLOCK,
        "maximum_scored_streams": 2,
        "pool_with_concurrency_one_without_review": False,
    }
    if "tasks" in body:
        body["tasks"] = [row for row in body["tasks"] if row["rank"] == EXPECTED_CELL[0]]
        if len(body["tasks"]) != 1:
            raise ValueError("GLM s2 concurrency-two runtime task drifted")
    body.pop("plan_sha256", None)
    body["plan_sha256"] = self_hosted.digest_without(body, "plan_sha256")
    return body


def build_plan(controller: str, root: Path) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown GLM concurrency-two canary controller")
    return _transform(source.validate_all(root)[CONTROLLER], runtime=False)


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plan = build_plan(CONTROLLER, root)
    row = plan["attempts"][0]
    lease = plan["execution"]["endpoint_lease"]
    if any(
        (
            plan["schema_version"] != SCHEMA,
            (row["selection_rank"], row["attempt"]) != EXPECTED_CELL,
            plan["model"]["served_id"] != "glm-5.3",
            plan["harness"]["version"] != "1.18.27",
            plan["execution"]["required_task_tools"] != ["bash", "submit_report"],
            lease["lease_root"] != str(LEASE_ROOT),
            lease["endpoint_key"] != LEASE_ENDPOINT_KEY,
            lease["maximum_streams"] != 2,
            plan["serving_load_block"] != SERVING_LOAD_BLOCK,
            plan["launch_authorized"] is not False,
            plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("GLM s2 concurrency-two canary plan drifted")
    return {CONTROLLER: plan}


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    if controller != CONTROLLER:
        raise ValueError("unknown GLM concurrency-two canary controller")
    return _transform(source.build_runtime_plan(CONTROLLER, inventory_receipt, root), runtime=True)
