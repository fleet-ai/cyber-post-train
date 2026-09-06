"""Held one-cell rank-3 GLM canary using the exact live s2 runner treatment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-hosted-glm-rank3-a1-c2-canary-plan-v1"
CONTROLLER = "glm-hosted-r3-a1-canary"
SOURCE_CONTROLLER = "glm-hosted-s1"
JOB_NAME = "chris-glm53-exact100-hosted-r003-a1-canary-v1"
CONFIGMAP_NAME = JOB_NAME + "-run"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
CLAIM_ROOT = source.CLAIM_ROOT
LEASE_ROOT = source.LEASE_ROOT
LEASE_ENDPOINT_KEY = "glm-hosted-autocontinue-v1"
SERVING_LOAD_BLOCK = "glm-hosted-autocontinue-c2-v1"
EXPECTED_RANK = 3
EXPECTED_ATTEMPT = 1
EXPECTED_TASK_CELLS = {(EXPECTED_RANK, attempt) for attempt in range(1, 5)}

# This immutable ConfigMap is the live, successful s2 runtime authority.  The
# canary package may replace only identity/partition modules and self_hosted.py
# from commit 640f430 (which adds sanitized HTTP routing evidence).
KNOWN_GOOD_BASE_CONFIGMAP = {
    "name": "chris-glm53-exact100-hosted-s2-c2-bulk-v1-run",
    "uid": "1b9b171e-5ea8-4f9d-b014-e4e0bd106789",
    "immutable": True,
    "critical_data_sha256": {
        "bulk_runtime.py": "sha256:664dc9780e14e1e0be3045eaf2151cb48b983295b385b12e9a026f8d2bc64d2d",
        "endpoint_lease.py": "sha256:ea4927ec3428c9f2f401e22da89f2f036317360738d6fdfde7dc111c5f498432",
        "engine.py": "sha256:06f9857c99ee1773d505ea60c06c335b373d91b7dbf267d2074bbdc39c4aa20f",
        "fixed_proxy.py": "sha256:a73ef328a17bcc8990c7c941f75be1b3038fc5fbf699ff664a8a8d39dcf1c441",
        "runner.py": "sha256:b1f9c5028f65b0d7772538e3ce075310dc6c7a46b3de58d0196bc474e74e9e9d",
        "run.sh": "sha256:c79c3997a444501917a8e59d1d7f99262b8836deaa05080ecde6f2072c748625",
    },
}
FAILURE_AUTHORITY = {
    "commit": "640f430",
    "receipt_path": "docs/evidence/glm53-study/2026-09-06-glm53-hosted-rank2-a1-request-abort.json",
    "receipt_sha256": "sha256:b1f1f98061343219762b160ee14f76f9209ebe903b3e2dca6471727089d8a67a",
    "file_sha256": "sha256:5348561c6a47065a76ffe72274d141b8b0d6d30fa6fe3deed68d51ad0aa70da2",
}
CONTROLLERS = {
    CONTROLLER: {
        "job_name": JOB_NAME,
        "configmap_name": CONFIGMAP_NAME,
        "model": "glm-5.3",
        "serving_block": LEASE_ENDPOINT_KEY,
        "task_ranks": [EXPECTED_RANK],
    }
}
SHA256_RE = source.SHA256_RE
COMMIT_RE = source.COMMIT_RE


def load(path: Path) -> dict[str, Any]:
    return source.load(path)


def validate_inventory_gate(value: dict[str, Any], root: Path) -> None:
    source.validate_inventory_gate(value, root)


def _transform(plan: dict[str, Any], *, runtime: bool) -> dict[str, Any]:
    task_rows = [
        copy.deepcopy(row)
        for row in plan["attempts"]
        if (row["selection_rank"], row["attempt"]) in EXPECTED_TASK_CELLS
    ]
    task_rows.sort(key=lambda row: row["attempt"])
    if {(row["selection_rank"], row["attempt"]) for row in task_rows} != EXPECTED_TASK_CELLS:
        raise ValueError("rank-3 whole-task authority drifted")
    selected = [row for row in task_rows if row["attempt"] == EXPECTED_ATTEMPT]
    if len(selected) != 1:
        raise ValueError("rank-3 canary cell authority drifted")
    selected[0]["ordinal"] = 1

    body = copy.deepcopy(plan)
    body.update(
        {
            "schema_version": SCHEMA,
            "campaign_id": JOB_NAME,
            "source_job_id": JOB_NAME,
            "controller": CONTROLLER,
            "job_name": JOB_NAME,
            "configmap_name": CONFIGMAP_NAME,
            "sfs_root": str(SFS_ROOT),
            "serving_block": LEASE_ENDPOINT_KEY,
            "serving_load_block": SERVING_LOAD_BLOCK,
            "task_count": 1,
            "new_session_count": 1,
            "attempts": selected,
            "launch_authorized": runtime,
            "known_good_s2_runtime": copy.deepcopy(KNOWN_GOOD_BASE_CONFIGMAP),
            "failure_authority": copy.deepcopy(FAILURE_AUTHORITY),
        }
    )
    if "tasks" in body:
        body["tasks"] = [row for row in body["tasks"] if row["rank"] == EXPECTED_RANK]
        if len(body["tasks"]) != 1:
            raise ValueError("rank-3 runtime task authority drifted")
    body["execution"]["workers"] = 1
    body["execution"]["endpoint_lease"] = {
        "lease_root": str(LEASE_ROOT),
        "endpoint_key": LEASE_ENDPOINT_KEY,
        "maximum_streams": 2,
    }
    body["execution"]["priority_class"] = "fleet-infra-quiet"
    body["execution"]["preemption_policy"] = "Never"
    body["partition"] = {
        "whole_task_rank": EXPECTED_RANK,
        "whole_task_cells": [
            {
                "attempt": row["attempt"],
                "cell_id": row["cell_id"],
                "execution_id": row["execution_id"],
            }
            for row in task_rows
        ],
        "all_four_cells_unstarted_required_at_release": True,
        "canary_attempt": EXPECTED_ATTEMPT,
        "remaining_attempts_held": [2, 3, 4],
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
        raise ValueError("unknown rank-3 hosted GLM canary controller")
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
            plan["new_session_count"] != 1,
            plan["plan_sha256"] != self_hosted.digest_without(plan, "plan_sha256"),
        )
    ):
        raise ValueError("rank-3 hosted GLM canary plan drifted")
    return {CONTROLLER: plan}
