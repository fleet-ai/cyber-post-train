"""Fresh retry-safe G19 controller identities after the preclaim v1 failure."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_generation19_bulk as prior
from evals.fleet import self_hosted

CONTROLLERS = {
    "qwen-a": {
        "job_name": "chris-q38-ac-exact100-g19-a192-v2",
        "configmap_name": "chris-q38-ac-exact100-g19-a192-run-v2",
    },
    "qwen-b": {
        "job_name": "chris-q38-ac-exact100-g19-b192-v2",
        "configmap_name": "chris-q38-ac-exact100-g19-b192-run-v2",
    },
}
PLAN_PATHS = {
    controller: f"evals/fleet/configs/qwen-hosted-generation19-{controller}-v2.json"
    for controller in CONTROLLERS
}
IDENTITY_DIGESTS = {
    "qwen-a": "sha256:f422f1c6b57074c4d7a0d9670fcd2b68f27933ea3a69ba59c8d53fa30a57a647",
    "qwen-b": "sha256:c4227ed3db2115a864d58edc29cc7cdf22d861cd767c8c17c07f8d72ef97529c",
}


def build_plans(root: Path) -> dict[str, dict[str, Any]]:
    sources = prior.validate_all(root)
    plans: dict[str, dict[str, Any]] = {}
    for controller, source in sources.items():
        authority = CONTROLLERS[controller]
        body = copy.deepcopy(source)
        body.pop("plan_sha256")
        body["schema_version"] = "fleet-qwen-generation19-hosted-bulk-plan-v2"
        body["campaign_id"] = authority["job_name"]
        body["source_job_id"] = authority["job_name"]
        body["sfs_root"] = f"/mnt/sfs/jobs/{authority['job_name']}"
        for item in body["attempts"]:
            run_id = (
                f"chris-q38-ac-g19v2-{controller[-1]}-r{item['selection_rank']:03d}-"
                f"a{item['attempt']}-{item['execution_id'][7:15]}"
            )
            item["run_id"] = run_id
            item["network"] = run_id.removeprefix("chris-")[:63]
        plans[controller] = {
            **body,
            "plan_sha256": self_hosted.digest_without(body, "plan_sha256"),
        }
    validate(plans, root, sources)
    return plans


def validate(
    plans: dict[str, dict[str, Any]],
    root: Path,
    sources: dict[str, dict[str, Any]] | None = None,
) -> None:
    identities: set[tuple[int, int]] = set()
    for controller, plan in plans.items():
        if any(
            (
                plan.get("schema_version")
                != "fleet-qwen-generation19-hosted-bulk-plan-v2",
                plan.get("campaign_id") != CONTROLLERS[controller]["job_name"],
                plan.get("source_job_id") != CONTROLLERS[controller]["job_name"],
                plan.get("sfs_root")
                != f"/mnt/sfs/jobs/{CONTROLLERS[controller]['job_name']}",
                plan.get("plan_sha256")
                != self_hosted.digest_without(plan, "plan_sha256"),
                len(plan.get("attempts") or []) != 192,
            )
        ):
            raise ValueError("Generation-19 v2 successor plan drifted")
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
        if (
            self_hosted.sha256(self_hosted.canonical_json(projection))
            != IDENTITY_DIGESTS[controller]
        ):
            raise ValueError("Generation-19 v2 identity projection drifted")
        for row in plan["attempts"]:
            execution = prior.exact.execution_for(row["cell_id"], prior.EXECUTION_GENERATION)
            if any(
                (
                    prior.g17.SHA256_RE.fullmatch(row["cell_id"]) is None,
                    prior.g17.UUID_RE.fullmatch(row["task_version_id"]) is None,
                    row["execution_id"] != execution["execution_id"],
                )
            ):
                raise ValueError("Generation-19 v2 changed statistical cells")
        if sources is not None:
            projected = [(row["cell_id"], row["execution_id"]) for row in plan["attempts"]]
            source = [
                (row["cell_id"], row["execution_id"])
                for row in sources[controller]["attempts"]
            ]
            if projected != source:
                raise ValueError("Generation-19 v2 changed predecessor cells")
        identities.update((row["selection_rank"], row["attempt"]) for row in plan["attempts"])
    if identities != {(rank, attempt) for rank in prior.ALLOWED_RANKS for attempt in range(1, 5)}:
        raise ValueError("Generation-19 v2 coverage drifted")


def validate_all(root: Path) -> dict[str, dict[str, Any]]:
    plans = {controller: prior.g17.load(root / path) for controller, path in PLAN_PATHS.items()}
    validate(plans, root)
    return plans


def build_runtime_plan(
    controller: str, inventory_receipt: dict[str, Any], root: Path
) -> dict[str, Any]:
    plan = validate_all(root)[controller]
    if plan["inventory_receipt"] != inventory_receipt:
        raise ValueError("Generation-19 v2 inventory projection drifted")
    return plan
