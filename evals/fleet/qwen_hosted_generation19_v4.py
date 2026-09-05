"""Fresh G19 v4 identities after completing the bulk-engine adapter contract."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_generation19_bulk as original
from evals.fleet import qwen_hosted_generation19_v2 as identity
from evals.fleet import qwen_hosted_generation19_v3 as prior
from evals.fleet import self_hosted

CANARY_GATE_SCHEMA = original.g17.base.CANARY_GATE_SCHEMA
COMMIT_RE = original.g17.COMMIT_RE
RECONCILIATION_GATE_SCHEMA = original.g17.base.RECONCILIATION_GATE_SCHEMA
SHA256_RE = original.g17.SHA256_RE
load = original.g17.load
validate_inventory_gate = original.g17.validate_inventory_gate
CONTROLLERS = {
    "qwen-a": {
        "job_name": "chris-q38-ac-exact100-g19-a192-v4",
        "configmap_name": "chris-q38-ac-exact100-g19-a192-run-v4",
    },
    "qwen-b": {
        "job_name": "chris-q38-ac-exact100-g19-b192-v4",
        "configmap_name": "chris-q38-ac-exact100-g19-b192-run-v4",
    },
}
PLAN_PATHS = {
    controller: f"evals/fleet/configs/qwen-hosted-generation19-{controller}-v4.json"
    for controller in CONTROLLERS
}


def build_plans(root: Path) -> dict[str, dict[str, Any]]:
    sources = prior.validate_all(root)
    plans: dict[str, dict[str, Any]] = {}
    for controller, source in sources.items():
        authority = CONTROLLERS[controller]
        body = copy.deepcopy(source)
        body.pop("plan_sha256")
        body["schema_version"] = "fleet-qwen-generation19-hosted-bulk-plan-v4"
        body["campaign_id"] = authority["job_name"]
        body["source_job_id"] = authority["job_name"]
        body["sfs_root"] = f"/mnt/sfs/jobs/{authority['job_name']}"
        for item in body["attempts"]:
            run_id = (
                f"chris-q38-ac-g19v4-{controller[-1]}-r{item['selection_rank']:03d}-"
                f"a{item['attempt']}-{item['execution_id'][7:15]}"
            )
            item["run_id"] = run_id
            item["network"] = run_id.removeprefix("chris-")[:63]
        plans[controller] = {
            **body,
            "plan_sha256": self_hosted.digest_without(body, "plan_sha256"),
        }
    validate(plans)
    return plans


def validate(plans: dict[str, dict[str, Any]]) -> None:
    identities: set[tuple[int, int]] = set()
    for controller, plan in plans.items():
        if any(
            (
                plan.get("schema_version")
                != "fleet-qwen-generation19-hosted-bulk-plan-v4",
                plan.get("campaign_id") != CONTROLLERS[controller]["job_name"],
                plan.get("source_job_id") != CONTROLLERS[controller]["job_name"],
                plan.get("sfs_root")
                != f"/mnt/sfs/jobs/{CONTROLLERS[controller]['job_name']}",
                plan.get("plan_sha256")
                != self_hosted.digest_without(plan, "plan_sha256"),
                len(plan.get("attempts") or []) != 192,
            )
        ):
            raise ValueError("Generation-19 v4 successor plan drifted")
        projection = [
            (
                row["cell_id"],
                row["execution_id"],
                row["selection_rank"],
                row["attempt"],
                row["task_version_id"],
            )
            for row in plan["attempts"]
        ]
        if self_hosted.sha256(self_hosted.canonical_json(projection)) != identity.IDENTITY_DIGESTS[
            controller
        ]:
            raise ValueError("Generation-19 v4 identity projection drifted")
        for row in plan["attempts"]:
            execution = original.exact.execution_for(row["cell_id"], original.EXECUTION_GENERATION)
            if any(
                (
                    SHA256_RE.fullmatch(row["cell_id"]) is None,
                    original.g17.UUID_RE.fullmatch(row["task_version_id"]) is None,
                    row["execution_id"] != execution["execution_id"],
                )
            ):
                raise ValueError("Generation-19 v4 changed statistical cells")
            identities.add((row["selection_rank"], row["attempt"]))
    expected = {
        (rank, attempt) for rank in original.ALLOWED_RANKS for attempt in range(1, 5)
    }
    if identities != expected:
        raise ValueError("Generation-19 v4 coverage drifted")


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plans = {controller: load(root / path) for controller, path in PLAN_PATHS.items()}
    validate(plans)
    return plans


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    plan = validate_all(root)[controller]
    if plan["inventory_receipt"] != inventory_receipt:
        raise ValueError("Generation-19 v4 inventory projection drifted")
    return plan
