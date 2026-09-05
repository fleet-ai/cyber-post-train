"""Run the released one-cell GLM concurrency-two canary."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_bulk_runtime_v3 as engine
from evals.fleet import hosted_glm_exact_bulk_runtime_v1 as bulk_runtime
from evals.fleet import hosted_glm_exact_bulk_v1 as bulk
from evals.fleet import hosted_glm_s2_c2_canary_v1 as c2
from evals.fleet import hosted_glm_s2_c2_release_v1 as release_module
from evals.fleet import self_hosted

RELEASE_PATH = release_module.OUTPUT_PATH


def validate_release_receipt(plan: dict[str, Any], release: dict[str, Any]) -> None:
    item = plan["attempts"][0]
    if any(
        (
            release.get("schema_version") != release_module.SCHEMA,
            release.get("status") != "CLEAR",
            release.get("successor_job") != c2.JOB_NAME,
            release.get("successor_configmap") != c2.CONFIGMAP_NAME,
            release.get("plan_sha256") != plan["plan_sha256"],
            release.get("cell_id") != item["cell_id"],
            release.get("execution_id") != item["execution_id"],
            release.get("s1_job") != release_module.S1_JOB,
            release.get("s1_job_uid") != release_module.S1_JOB_UID,
            release.get("s1_active") is not True,
            release.get("s1_slot1_lease_held") is not True,
            release.get("fleet_session_collisions") != 0,
            release.get("global_claim_collisions") != 0,
            release.get("kubernetes_object_collisions") != 0,
            release.get("active_or_accepted_collisions") != 0,
            release.get("serving_load_block") != c2.SERVING_LOAD_BLOCK,
            release.get("lease_root") != str(c2.LEASE_ROOT),
            release.get("lease_endpoint_key") != c2.LEASE_ENDPOINT_KEY,
            release.get("maximum_scored_streams") != 2,
            release.get("checked_immediately_before_create") is not True,
            release.get("mutation_calls") != 0,
            release.get("scores_read") is not False,
            release.get("prompts_traces_flags_read") is not False,
            release.get("receipt_sha256")
            != self_hosted.digest_without(release, "receipt_sha256"),
        )
    ):
        raise RuntimeError("GLM concurrency-two canary release gate drifted")


def validate_release(plan: dict[str, Any]) -> None:
    # Revalidate the live s1 receipt as well as the release observer's binding.
    current_s1 = release_module._first_s1_acceptance(Path(plan["repo_root"]))  # noqa: SLF001
    release_module._s1_active_and_lease_held()  # noqa: SLF001
    release = c2.load(RELEASE_PATH)
    validate_release_receipt(plan, release)
    if release.get("s1_first_accepted_receipt_sha256") != current_s1["receipt_sha256"]:
        raise RuntimeError("GLM concurrency-two s1 acceptance binding drifted")


def run(root: Path, proxy: Path) -> dict[str, Any]:
    inventory = bulk.load(bulk_runtime.INVENTORY_PATH)
    plan = c2.build_runtime_plan(c2.CONTROLLER, inventory, root)
    engine.bulk = c2
    return engine.run_controller(
        plan,
        out=c2.SFS_ROOT,
        proxy=proxy,
        runtime_gate_check=validate_release,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--proxy", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo.resolve(), args.proxy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
