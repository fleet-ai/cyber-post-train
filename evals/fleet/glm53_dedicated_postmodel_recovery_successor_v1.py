"""Held recovery plan after post-model Fleet scoring-route failures.

This module is deliberately score-free.  It binds the two consumed incidents,
rejects any candidate that is accepted, ingested, claimed, or ambiguous, and
emits a create-once successor plan that still requires independent review.
"""

from __future__ import annotations

from typing import Any

from evals.fleet import exact_pass4_crypto as crypto

SCHEMA = "fleet-glm53-dedicated-postmodel-recovery-successor-held-v1"
COMPARISON_SCHEMA = "fleet-glm53-postmodel-fleet-request-failure-comparison-v1"
SERVER_TITLE = "chris-cyber-evalserve-glm53-tp8-a-v23"
SERVER_RUN_DIR = "/mnt/sfs/jobs/chris-cyber-evalserve-glm53-tp8-a-v23"
CONTROLLER_JOB = "chris-glm53-dedicated-v23-r100-whole-task-v1"
CONTROLLER_ROOT = "/mnt/sfs/jobs/chris-glm53-dedicated-v23-r100-whole-task-v1"
TASK_VERSION_ID = "2b841931-a8ff-4f5c-b32b-6824ea5b4429"
SELECTION_RANK = 100
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
CONSUMED_INCIDENTS = [
    {
        "lane": "hosted",
        "authority_path": (
            "docs/evidence/glm53-study/"
            "2026-09-06-glm53-hosted-s1-rank1-a4-request-abort.json"
        ),
        "authority_receipt_sha256": (
            "sha256:8a0c02065c5bcbdb140eeb5250f37954920b0840502828caed6e92b8d2184996"
        ),
        "authority_file_sha256": (
            "sha256:72c59d169eea74ede8ae5348b2ce7cbc1f1ac7afe9b13e20f74610906c83e8a7"
        ),
        "cell_id": "sha256:f3921b5927bcf73a0df58ff991841db52f9b23f04cae5d482c5f7e8630308250",
        "execution_id": (
            "sha256:b475fceb1fa1875953ab08c7d769962c6b6c57c7a919d56b658140b37adaed78"
        ),
    },
    {
        "lane": "dedicated",
        "authority_path": (
            "docs/evidence/glm53-study/"
            "2026-09-06-glm53-dedicated-v22-r51-a2-terminal-reconciliation-v1.json"
        ),
        "authority_receipt_sha256": (
            "sha256:db94ebd6eef8c4986b7721d5f8c14c1243c9597d5263cccd17c32137c3d6c628"
        ),
        "authority_file_sha256": (
            "sha256:383fa9cbe3c5d9252316b311b48e9d99933e07c0c6c2d2c21a930cacb60073d2"
        ),
        "cell_id": "sha256:2aed501bd3b6db1176ea03687f7bc91c88d205997b00e3a0747dc6a1e1e8b5df",
        "execution_id": (
            "sha256:a9be25eab3efd65f86a832b3c9681dd992d40b51cc1d6d4046212e6797e0ec81"
        ),
    },
]


class RecoveryPlanError(RuntimeError):
    """The score-free recovery plan cannot be sealed."""


def comparison() -> dict[str, Any]:
    """Return the common score-blind failure boundary and discriminating gate."""

    body: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "incidents": CONSUMED_INCIDENTS,
        "shared_observations": {
            "model_interaction_before_failure": True,
            "scoring_intent_present": True,
            "fleet_request_error": True,
            "reward_result_present": False,
            "result_present": False,
            "session_ingest_present": False,
            "matching_verifier_execution_present": False,
            "cleanup_complete": True,
            "controller_exit_code": 1,
            "controller_restarts": 0,
        },
        "smallest_reusable_fix": {
            "late_body_blind_scoring_route_preflight": "GET_EXPECT_405",
            "persist_failure_fields": ["method", "route", "http_status"],
            "preflight_mutation_calls": 0,
            "response_body_read": False,
            "automatic_retry": False,
        },
        "interpretation": (
            "The probe discriminates route or authentication failure from a later "
            "POST backend failure; it does not predict or authorize scoring success."
        ),
        "privacy": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body


def _validate_ledger(ledger: dict[str, Any]) -> None:
    states = ledger.get("state_counts") or {}
    if (
        ledger.get("schema_version") != "fleet-exact-pass4-global-ledger-v1"
        or ledger.get("validated") is not True
        or ledger.get("receipt_sha256") != crypto.digest_without(ledger, "receipt_sha256")
        or not isinstance(ledger.get("file_sha256"), str)
        or not ledger["file_sha256"].startswith("sha256:")
        or len(ledger["file_sha256"]) != 71
        or sum(states.values()) != 400
        or ledger.get("all_accepted_ingested_ambiguous_cells_excluded") is not True
    ):
        raise RecoveryPlanError("global_ledger_not_authoritative")


def _validate_candidate(candidate: dict[str, Any]) -> None:
    expected = list(zip(range(1, 5), CELL_IDS, EXECUTION_IDS, RUN_IDS, strict=True))
    rows = candidate.get("cells")
    consumed_cells = {item["cell_id"] for item in CONSUMED_INCIDENTS}
    consumed_executions = {item["execution_id"] for item in CONSUMED_INCIDENTS}
    if (
        candidate.get("selection_rank") != SELECTION_RANK
        or candidate.get("task_version_id") != TASK_VERSION_ID
        or not isinstance(rows, list)
        or [
            (row.get("attempt"), row.get("cell_id"), row.get("execution_id"), row.get("run_id"))
            for row in rows
        ]
        != expected
        or any(
            row.get("ledger_state") != "unstarted"
            or row.get("latest_generation") != 0
            or row.get("accepted_generations") != 0
            or row.get("session_matches") != 0
            or row.get("verifier_matches") != 0
            or row.get("claim_collisions") != 0
            or row.get("output_collisions") != 0
            or row.get("accepted_receipt_collisions") != 0
            or row.get("cell_id") in consumed_cells
            or row.get("execution_id") in consumed_executions
            for row in rows
        )
    ):
        raise RecoveryPlanError("candidate_is_not_wholly_unstarted")


def build_held(ledger: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Build a no-launch plan for one fresh whole task on a future server."""

    _validate_ledger(ledger)
    _validate_candidate(candidate)
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "READY_HELD_FOR_ROOT_REVIEW",
        "comparison_receipt_sha256": comparison()["receipt_sha256"],
        "consumed_incidents": CONSUMED_INCIDENTS,
        "global_ledger_receipt_sha256": ledger["receipt_sha256"],
        "global_ledger_file_sha256": ledger["file_sha256"],
        "server": {
            "title": SERVER_TITLE,
            "run_dir": SERVER_RUN_DIR,
            "nodes": 1,
            "gpus": 8,
            "priority_class": "fleet-infra-quiet",
            "preemption_policy": "Never",
            "jobs_api_create_once": True,
            "fresh_uid_bound_parity_required": True,
            "actual_request_idle_release_seconds": 600,
        },
        "controller": {
            "job_name": CONTROLLER_JOB,
            "sfs_root": CONTROLLER_ROOT,
            "selection_rank": SELECTION_RANK,
            "task_version_id": TASK_VERSION_ID,
            "attempts": [1, 2, 3, 4],
            "cell_ids": CELL_IDS,
            "execution_ids": EXECUTION_IDS,
            "run_ids": RUN_IDS,
            "same_task_max_inflight": 1,
            "attempts_sequential": True,
            "authoritative_acceptance_before_next_attempt": True,
            "canonical_claim_before_model_call": True,
            "create_once": True,
        },
        "future_live_gates": [
            "fresh_global_ledger_and_authoritative_session_reconciliation",
            "jobs_api_kubernetes_sfs_duplicate_and_capacity_preview",
            "exact_server_uids_and_live_actual_opencode_parity",
            "projected_full_path_cpu_canary",
            "late_body_blind_scoring_route_preflight",
            "root_independent_review",
        ],
        "gpu_server_submit_permitted": False,
        "scoring_create_permitted": False,
        "launch_authorized": False,
        "api_mutation_calls": 0,
        "privacy": False,
    }
    body["receipt_sha256"] = crypto.digest_without(body, "receipt_sha256")
    return body
