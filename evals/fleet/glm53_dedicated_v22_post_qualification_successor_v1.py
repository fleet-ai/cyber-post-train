"""Held scored successor boundary after the GLM v22 score-free qualification."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as qualification

SCHEMA = "fleet-glm53-dedicated-v22-post-qualification-successor-held-v1"
QUALIFIED_SCHEMA = "fleet-glm53-dedicated-v22-concurrency-qualified-v1"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}")
JOB_NAME = "chris-glm53-dedicated-v22-r100-whole-task-v1"
SFS_ROOT = "/mnt/sfs/jobs/chris-glm53-dedicated-v22-r100-whole-task-v1"
SELECTION_RANK = 100
TASK_VERSION_ID = "2b841931-a8ff-4f5c-b32b-6824ea5b4429"
ATTEMPTS = [1, 2, 3, 4]
CELL_IDS = [
    "sha256:f2a641a9478f0190b6a2cf86de0f9c2d21cbd006560269f0de1c4c3326727648",
    "sha256:aeb858432cdd81e52bafd799f95da9896e6a4fa459e478e8e2b564f41e26edb3",
    "sha256:04183ed1098f0d260c0aa60768507cfce65ef842850a8f940088450b074830cf",
    "sha256:227ef85a3976364d2361064e2aa83b58cc3270b64b5fc636307a2cb211fb548f",
]
EXECUTION_IDS = [
    "sha256:9e051bb9590d894493b1be717b1ddf71e9e4a793c7a896c7ab2f1ce5f568ec2a",
    "sha256:f45999138185b1401a49849f21beaf268fb53a9e2ffd925ad12304b1de5ccce3",
    "sha256:02eb409679c9fda266a9f3a27c5dd22b12c72f8750ce152b0c51a3fb4be90fb7",
    "sha256:074a7bfeac406be4a8140613255aba8aae3d382c8076bcb163c4b0eab2e5916c",
]
RUN_IDS = [
    "chris-glm53-ac-bulk-b-r100-a1-g1-2b841931",
    "chris-glm53-ac-bulk-b-r100-a2-g1-2b841931",
    "chris-glm53-ac-bulk-b-r100-a3-g1-2b841931",
    "chris-glm53-ac-bulk-b-r100-a4-g1-2b841931",
]


class SuccessorError(RuntimeError):
    """Stable failure at the post-qualification scored boundary."""


def _valid_digest(value: str) -> bool:
    return value.startswith("sha256:") and len(value) == 71


def _validate_qualified(receipt: dict[str, Any]) -> int:
    ceiling = receipt.get("qualified_concurrency_ceiling")
    if (
        receipt.get("schema_version") != QUALIFIED_SCHEMA
        or receipt.get("status") != "PASSED_SCORE_FREE"
        or receipt.get("failures") != []
        or receipt.get("scored_concurrency_change_authorized") is not True
        or ceiling not in qualification.CONCURRENCY
        or ceiling != max(qualification.CONCURRENCY)
        or receipt.get("receipt_sha256") != crypto.digest_without(receipt, "receipt_sha256")
    ):
        raise SuccessorError("score_free_qualification_not_accepted")
    return int(ceiling)


def _validate_live(live: dict[str, Any], expected: dict[str, Any]) -> None:
    if (
        live.get("server_api_run_id") != expected["server_api_run_id"]
        or live.get("serving_block") != expected["serving_block"]
        or live.get("rank51_attempt2_accepted_validated") is not True
        or live.get("qualification_exclusively_succeeded") is not True
        or live.get("qualification_job_name") != qualification.JOB_NAME
        or UUID_RE.fullmatch(live.get("qualification_job_uid", "")) is None
        or UUID_RE.fullmatch(live.get("qualification_pod_uid", "")) is None
        or live.get("active_scored_controller_count") != 0
        or live.get("endpoint_lease_available") is not True
        or live.get("server_uid_and_workload_history_unchanged") is not True
        or live.get("workload_preemption_events") != 0
        or live.get("candidate_job_collisions") != 0
        or live.get("candidate_sfs_root_collisions") != 0
        or live.get("authoritative_session_collisions") != 0
        or live.get("accepted_receipt_collisions") != 0
        or live.get("canonical_claim_collisions") != 0
    ):
        raise SuccessorError("fresh_live_successor_gate_failed")


def _validate_whole_task(candidate: dict[str, Any], expected: dict[str, Any]) -> None:
    cells = candidate.get("cells")
    expected_rows = list(
        zip(
            expected["attempts"],
            expected["cell_ids"],
            expected["execution_ids"],
            expected["run_ids"],
            strict=True,
        )
    )
    if (
        candidate.get("selection_rank") != expected["selection_rank"]
        or candidate.get("task_version_id") != expected["task_version_id"]
        or not isinstance(cells, list)
        or len(cells) != 4
        or [
            (row.get("attempt"), row.get("cell_id"), row.get("execution_id"), row.get("run_id"))
            for row in cells
        ]
        != expected_rows
        or any(
            row.get("ledger_state") != "unstarted"
            or row.get("claim_collisions") != 0
            or row.get("session_collisions") != 0
            or row.get("output_collisions") != 0
            or row.get("accepted_collisions") != 0
            or not _valid_digest(row.get("cell_id", ""))
            or not _valid_digest(row.get("execution_id", ""))
            for row in cells
        )
        or not _valid_digest(candidate.get("global_ledger_receipt_sha256", ""))
        or not _valid_digest(candidate.get("global_ledger_file_sha256", ""))
        or candidate.get("global_ledger_validated") is not True
    ):
        raise SuccessorError("whole_task_candidate_not_fresh")


def expected_whole_task(root: Path) -> dict[str, Any]:
    plan = qualification.render(root)
    return {
        "job_name": JOB_NAME,
        "sfs_root": SFS_ROOT,
        "selection_rank": SELECTION_RANK,
        "task_version_id": TASK_VERSION_ID,
        "cell_ids": CELL_IDS,
        "execution_ids": EXECUTION_IDS,
        "run_ids": RUN_IDS,
        "attempts": ATTEMPTS,
        "server_api_run_id": plan["server"]["api_run_id"],
        "serving_block": "glm-dedicated-v22-r051-v1",
        "service_origin": plan["server"]["origin"],
        "parity_receipt_sha256": plan["server"]["parity_receipt_sha256"],
    }


def build_held(
    root: Path,
    qualified: dict[str, Any],
    live: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Bind one exact whole task while retaining an independent-review stop."""

    ceiling = _validate_qualified(qualified)
    expected = expected_whole_task(root)
    _validate_live(live, expected)
    _validate_whole_task(candidate, expected)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "READY_HELD_FOR_INDEPENDENT_REVIEW",
        "qualification_receipt_sha256": qualified["receipt_sha256"],
        "qualification_job_name": live["qualification_job_name"],
        "qualification_job_uid": live["qualification_job_uid"],
        "qualification_pod_uid": live["qualification_pod_uid"],
        "qualified_distinct_task_concurrency_ceiling": ceiling,
        "prepared_distinct_task_controller_count": 1,
        "server_api_run_id": expected["server_api_run_id"],
        "serving_block": expected["serving_block"],
        "service_origin": expected["service_origin"],
        "controller": {
            "job_name": expected["job_name"],
            "sfs_root": expected["sfs_root"],
            "selection_rank": expected["selection_rank"],
            "task_version_id": expected["task_version_id"],
            "attempts": expected["attempts"],
            "cell_ids": expected["cell_ids"],
            "execution_ids": expected["execution_ids"],
            "run_ids": expected["run_ids"],
            "same_task_max_inflight": 1,
            "attempts_sequential": True,
            "authoritative_acceptance_before_next_attempt": True,
            "create_once_job": True,
            "create_once_sfs_root": True,
            "canonical_claim_receipt_before_model_call": True,
        },
        "shared_endpoint_lease": {
            "lease_root": str(qualification.LEASE_ROOT),
            "endpoint_key": expected["server_api_run_id"],
            "maximum_distinct_task_streams": ceiling,
            "same_task_max_inflight": 1,
        },
        "fresh_launch_receipt_required": True,
        "fresh_global_ledger_receipt_sha256": candidate["global_ledger_receipt_sha256"],
        "fresh_global_ledger_file_sha256": candidate["global_ledger_file_sha256"],
        "parent_independent_review_required": True,
        "launch_authorized": False,
        "scoring_create_permitted": False,
        "privacy": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body
