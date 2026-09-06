"""Held whole-task successor for GLM v22 after the rank-51 canary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import glm53_dedicated_v22_a2_package_v1 as serving
from evals.fleet import glm53_dedicated_v22_a2_v1 as evidence
from evals.fleet import hosted_glm_exact_bulk_v1 as source
from evals.fleet import self_hosted

SCHEMA = "fleet-glm53-dedicated-v22-rank100-held-v1"
CONTROLLER = "glm-hosted-s4"
SELECTION_RANK = 100
TASK_VERSION_ID = "2b841931-a8ff-4f5c-b32b-6824ea5b4429"
JOB_NAME = "chris-glm53-dedicated-v22-r100-whole-task-v1"
SFS_ROOT = Path("/mnt/sfs/jobs") / JOB_NAME
EXPECTED_CELL_IDS = [
    "sha256:f2a641a9478f0190b6a2cf86de0f9c2d21cbd006560269f0de1c4c3326727648",
    "sha256:aeb858432cdd81e52bafd799f95da9896e6a4fa459e478e8e2b564f41e26edb3",
    "sha256:04183ed1098f0d260c0aa60768507cfce65ef842850a8f940088450b074830cf",
    "sha256:227ef85a3976364d2361064e2aa83b58cc3270b64b5fc636307a2cb211fb548f",
]


def render(root: Path) -> dict[str, Any]:
    parity = evidence.load(root / serving.PARITY)
    binding = evidence.load(root / serving.BINDING)
    evidence._validate_evidence(parity, binding, serving.ORIGIN)
    source_plan = source.validate_all(root)[CONTROLLER]
    attempts = [row for row in source_plan["attempts"] if row["selection_rank"] == SELECTION_RANK]
    if (
        [row["attempt"] for row in attempts] != [1, 2, 3, 4]
        or [row["cell_id"] for row in attempts] != EXPECTED_CELL_IDS
        or {row["task_version_id"] for row in attempts} != {TASK_VERSION_ID}
    ):
        raise ValueError("rank100 whole-task authority drifted")
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "READY_HELD",
        "launch_authorized": False,
        "job_name": JOB_NAME,
        "sfs_root": str(SFS_ROOT),
        "selection_rank": SELECTION_RANK,
        "task_version_id": TASK_VERSION_ID,
        "cell_ids": [row["cell_id"] for row in attempts],
        "execution_ids": [row["execution_id"] for row in attempts],
        "run_ids": [row["run_id"] for row in attempts],
        "attempts": [1, 2, 3, 4],
        "server_api_run_id": binding["api_run_id"],
        "serving_block": "glm-dedicated-v22-r051-v1",
        "service_origin": serving.ORIGIN,
        "parity_receipt_sha256": parity["receipt_sha256"],
        "required_predecessor": {
            "selection_rank": 51,
            "attempt": 2,
            "accepted": True,
        },
        "launch_time_gates": {
            "fresh_global_ledger_required": True,
            "all_four_cells_unstarted_required": True,
            "authoritative_session_collisions_required": 0,
            "kubernetes_object_collisions_required": 0,
            "sfs_output_collisions_required": 0,
            "canonical_claim_collisions_required": 0,
            "server_uid_and_workload_history_revalidation_required": True,
            "atomic_four_claim_reservation_before_first_model_call_required": True,
            "rollback_all_four_claims_on_partial_pre_model_reservation_failure": True,
            "authoritative_acceptance_before_each_next_attempt": True,
        },
        "scoring_create_permitted": False,
    }
    body["package_sha256"] = self_hosted.digest_without(body, "package_sha256")
    return body
