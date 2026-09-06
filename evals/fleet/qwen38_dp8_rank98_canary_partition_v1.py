"""Held rank-98 scored-canary partition after a complete DP8 qualification.

This module performs no API, Kubernetes, SFS, claim, model, or scoring writes.
It validates the immutable inputs for a future append-only release whose
controller may reserve all four G19 claims but initially execute only attempt 1.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from evals.fleet import qwen38_dp8_early_qualification_v1 as early
from evals.fleet import qwen38_dp8_post_rank99_launch_v1 as ladder
from evals.fleet import self_hosted

SCHEMA = "fleet-qwen38-dp8-rank98-canary-partition-held-v1"
LIVE_SCAN_SCHEMA = "fleet-qwen38-dp8-rank98-canary-live-scan-v1"
LEDGER_PATH = (
    "docs/evidence/qwen38-study/"
    "2026-09-05-exact-pass4-ledger-evidence-snapshot-v42.json"
)
LEDGER_RECEIPT_SHA256 = (
    "sha256:b6fc97f7d6364d8d669d5642c5bf3406c486d51f45d441b2ef051f77924cf01e"
)
LEDGER_FILE_SHA256 = (
    "sha256:c657d436f1d980f5d30c619686cd1fe0aed763622e388e88bf6b01bb1b8a080f"
)
G19_RUNTIME_PATH = "evals/fleet/exact_pass4_bulk_runtime_v3.py"
G19_RUNTIME_FILE_SHA256 = (
    "sha256:b80b39412906d3cb5163e55a882d0f5ae936fd03417fae61bc0a497db1f18daa"
)
TASK_KEY = "cysec1-2-fentry-gen_blackbox-6157a8d49f4e47208ebbc716__blackbox_ctf_v1"
TASK_VERSION_ID = "390e360e-ad44-43c2-91fd-8e7c9801b0be"
TASK_VERSION = 30
SELECTION_RANK = 98
CELL_IDS = [
    "sha256:fa44c73ce9b606dfeee70245ca9d6732274a53af166a7546f7008efb7dee1e39",
    "sha256:87dfc931108c9cba062e2cf1a170ea8f80a5dede022704c927dd654f82c3f4dc",
    "sha256:ea3afd1d70ddd54bef514c90822914145632384ccd73dd40fcfbb2247c8bfb26",
    "sha256:213208c6b9da1e577afefabee435ca5199f564a79cb4a81beacf0716a2a8e28f",
]
EXECUTION_IDS = [
    "sha256:df3bc0fdaec573b481b2ef215b480ad58edd023776cbba4f2a95216cb35a998b",
    "sha256:41fd786356475817e23755b2841e1914cad2d68485740939cec6ef03f6f7612b",
    "sha256:00cfa13c70b57398717cadf0b474b49eed1d3957a18e0fb9a84f5341f1cec6ba",
    "sha256:729e57e20a14ba2a5605b1cd249bc1b4730f031d06ace8b4a177ddd19dc42d88",
]
RUN_IDS = [
    "chris-q38-ac-g19-b-r098-a1-df3bc0fd",
    "chris-q38-ac-g19-b-r098-a2-41fd7863",
    "chris-q38-ac-g19-b-r098-a3-00cfa13c",
    "chris-q38-ac-g19-b-r098-a4-729e57e2",
]
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


class PartitionError(RuntimeError):
    """The held scored-canary partition is not safe to review."""


def _digest_valid(value: Mapping[str, Any]) -> bool:
    return value.get("receipt_sha256") == self_hosted.digest_without(
        dict(value), "receipt_sha256"
    )


def _validate_qualification(value: Mapping[str, Any]) -> str:
    highest = ladder.validate_result(
        dict(value), {"receipt_sha256": value.get("plan_receipt_sha256")}
    )
    levels = value.get("levels")
    if (
        highest != 8
        or value.get("highest_passing_concurrency") != 8
        or value.get("scored_calls") != 0
        or not isinstance(levels, list)
        or [row.get("concurrency") for row in levels] != [1, 2, 4, 8]
        or any(row.get("status") != "PASSED" for row in levels)
    ):
        raise PartitionError("complete_c1_to_c8_qualification_required")
    return str(value["receipt_sha256"])


def _validate_server(value: Mapping[str, Any]) -> None:
    uid_fields = ("rayjob_uid", "workload_uid", "head_pod_uid", "service_uid")
    if (
        not _digest_valid(value)
        or value.get("status") != "READY_NON_SCORED"
        or value.get("title") != early.TITLE
        or value.get("run_dir") != early.RUN_DIR
        or value.get("serving_block") != early.SERVING_BLOCK
        or not str(value.get("api_run_id", "")).startswith("ft-run-")
        or any(UUID_RE.fullmatch(str(value.get(field, ""))) is None for field in uid_fields)
        or not str(value.get("service_origin", "")).startswith("http://")
        or not str(value.get("service_origin", "")).endswith(":8000")
        or not str(value.get("traffic_path", "")).startswith(str(early.RUN_DIR))
        or value.get("served_id") != "qwen3.8-27b"
        or value.get("context_length") != 262144
        or value.get("tensor_parallel_size") != 1
        or value.get("data_parallel_size") != 8
        or value.get("head_pod_running_ready") is not True
        or value.get("head_pod_restarts") != 0
        or value.get("workload_preempted") is not False
        or not str(value.get("parity_receipt_sha256", "")).startswith("sha256:")
        or not str(value.get("parity_file_sha256", "")).startswith("sha256:")
        or not str(value.get("parity_path", "")).startswith(
            "docs/evidence/qwen38-study/"
        )
    ):
        raise PartitionError("exact_dp8_server_binding_required")


def _validate_live_scan(value: Mapping[str, Any], server: Mapping[str, Any]) -> None:
    rows = value.get("cells")
    expected = list(zip(range(1, 5), CELL_IDS, EXECUTION_IDS, RUN_IDS, strict=True))
    if (
        not _digest_valid(value)
        or value.get("schema_version") != LIVE_SCAN_SCHEMA
        or value.get("ledger_path") != LEDGER_PATH
        or value.get("ledger_receipt_sha256") != LEDGER_RECEIPT_SHA256
        or value.get("ledger_file_sha256") != LEDGER_FILE_SHA256
        or value.get("ledger_state_counts")
        != {"accepted": 12, "active": 2, "blocked": 3, "unstarted": 383}
        or value.get("server_binding_receipt_sha256") != server.get("receipt_sha256")
        or value.get("task_key") != TASK_KEY
        or value.get("task_version_id") != TASK_VERSION_ID
        or value.get("task_version") != TASK_VERSION
        or value.get("selection_rank") != SELECTION_RANK
        or value.get("task_session_rows_examined") != 26
        or value.get("exact_session_matches") != 0
        or value.get("matching_verifier_executions") != 0
        or value.get("controller_job_collisions") != 0
        or value.get("controller_sfs_collisions") != 0
        or value.get("g19_skip_reason") != "global_execution_claim_already_exists"
        or value.get("g19_skips_all_four_canonical_claims") is not True
        or value.get("g19_runtime_path") != G19_RUNTIME_PATH
        or value.get("g19_runtime_file_sha256") != G19_RUNTIME_FILE_SHA256
        or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(value.get("g19_controller_package_sha256", ""))
        )
        or value.get("api_mutation_calls") != 0
        or not isinstance(rows, list)
        or [
            (row.get("attempt"), row.get("cell_id"), row.get("execution_id"), row.get("run_id"))
            for row in rows
        ]
        != expected
        or any(
            row.get("ledger_state") != "unstarted"
            or row.get("latest_generation") != 0
            or row.get("accepted_receipt_matches") != 0
            or row.get("claim_path_exists") is not False
            or row.get("output_root_exists") is not False
            for row in rows
        )
    ):
        raise PartitionError("rank98_live_partition_not_fresh")


def build_held(
    qualification_result: Mapping[str, Any],
    server_binding: Mapping[str, Any],
    live_scan: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the exact future release while retaining the root-review stop."""

    qualification_sha = _validate_qualification(qualification_result)
    _validate_server(server_binding)
    _validate_live_scan(live_scan, server_binding)
    claims = [
        {
            "attempt": attempt,
            "cell_id": cell,
            "execution_id": execution,
            "run_id": run,
            "canonical_path": (
                "/mnt/sfs/cell-execution-claims/opencode11827-autocontinue-v1/"
                f"{execution.removeprefix('sha256:')}.json"
            ),
            "initial_state": "active" if attempt == 1 else "reservation_only",
        }
        for attempt, cell, execution, run in zip(
            range(1, 5), CELL_IDS, EXECUTION_IDS, RUN_IDS, strict=True
        )
    ]
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "HELD_PENDING_ROOT_REVIEW",
        "launch_authorized": False,
        "scoring_authorized": False,
        "qualification_result_sha256": qualification_sha,
        "qualified_concurrency_ceiling": 8,
        "fresh_live_scan_receipt_sha256": live_scan["receipt_sha256"],
        "ledger": {
            "path": LEDGER_PATH,
            "receipt_sha256": LEDGER_RECEIPT_SHA256,
            "file_sha256": LEDGER_FILE_SHA256,
        },
        "server_binding": dict(server_binding),
        "partition": {
            "selection_rank": SELECTION_RANK,
            "task_key": TASK_KEY,
            "task_version_id": TASK_VERSION_ID,
            "task_version": TASK_VERSION,
            "claims": claims,
            "whole_task_reserved": True,
            "g19_skip_reason": "global_execution_claim_already_exists",
            "g19_skips_all_four": True,
            "g19_runtime_path": G19_RUNTIME_PATH,
            "g19_runtime_file_sha256": G19_RUNTIME_FILE_SHA256,
            "g19_controller_package_sha256": live_scan["g19_controller_package_sha256"],
        },
        "atomic_reservation": {
            "preflight_all_four_absent_before_first_create": True,
            "o_excl_all_four_before_any_model_call": True,
            "publish_self_digested_reservation_receipt": True,
            "claims_bind_actual_controller_job_and_pod_uids": True,
            "partial_pre_model_failure_rolls_back_only_new_exact_claims": True,
            "rollback_requires_same_job_and_pod_uids": True,
            "rollback_requires_model_request_count_zero": True,
            "rollback_preserves_append_only_proof": True,
        },
        "canary": {
            "only_authorized_attempt": 1,
            "attempts_2_to_4_model_calls_authorized": False,
            "formal_authoritative_acceptance_required": True,
            "same_task_max_inflight": 1,
        },
        "continuation": {
            "append_only_release_required": True,
            "same_controller_job_and_pod_uids_required": True,
            "same_server_binding_and_serving_treatment_required": True,
            "attempts_sequential": True,
            "formal_acceptance_required_before_each_next_attempt": True,
            "after_any_model_call_attempt1_irrevocable": True,
            "reservation_only_claims_may_not_silently_transfer": True,
            "owner_or_server_identity_loss_blocks_attempts_2_to_4": True,
        },
        "future_release_must_bind": [
            "controller_configmap_name_and_sha256",
            "controller_job_name",
            "controller_package_sha256",
            "fresh_live_scan_receipt_sha256",
            "root_review_receipt_sha256",
        ],
        "held_controller_identity": {
            "configmap_name": "chris-cyber-q38-dp8-c-r098-a1-canary-v1-held",
            "job_name": "chris-cyber-q38-dp8-c-r098-a1-canary-v1-held",
            "active_attempt": 1,
            "held_attempts": [2, 3, 4],
        },
        "controller_create_permitted": False,
        "api_mutation_calls": 0,
        "prompts_traces_flags_scores_or_model_outputs_included": False,
    }
    body["receipt_sha256"] = self_hosted.digest_without(body, "receipt_sha256")
    return body
