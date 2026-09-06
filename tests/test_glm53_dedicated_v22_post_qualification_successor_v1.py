from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_v22_concurrency_qualification_v1 as qualification
from evals.fleet import glm53_dedicated_v22_post_qualification_successor_v1 as successor

ROOT = Path(__file__).resolve().parents[1]


def _qualified() -> dict:
    value = {
        "schema_version": successor.QUALIFIED_SCHEMA,
        "status": "PASSED_SCORE_FREE",
        "failures": [],
        "scored_concurrency_change_authorized": True,
        "qualified_concurrency_ceiling": 4,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def _live(expected: dict) -> dict:
    return {
        "server_api_run_id": expected["server_api_run_id"],
        "serving_block": expected["serving_block"],
        "rank51_attempt2_accepted_validated": True,
        "qualification_exclusively_succeeded": True,
        "qualification_job_name": qualification.JOB_NAME,
        "qualification_job_uid": "11111111-1111-4111-8111-111111111111",
        "qualification_pod_uid": "22222222-2222-4222-8222-222222222222",
        "active_scored_controller_count": 0,
        "endpoint_lease_available": True,
        "server_uid_and_workload_history_unchanged": True,
        "workload_preemption_events": 0,
        "candidate_job_collisions": 0,
        "candidate_sfs_root_collisions": 0,
        "authoritative_session_collisions": 0,
        "accepted_receipt_collisions": 0,
        "canonical_claim_collisions": 0,
    }


def _candidate(expected: dict) -> dict:
    return {
        "selection_rank": expected["selection_rank"],
        "task_version_id": expected["task_version_id"],
        "global_ledger_receipt_sha256": "sha256:" + "9" * 64,
        "global_ledger_file_sha256": "sha256:" + "8" * 64,
        "global_ledger_validated": True,
        "cells": [
            {
                "attempt": attempt,
                "cell_id": cell,
                "execution_id": execution,
                "run_id": run,
                "ledger_state": "unstarted",
                "claim_collisions": 0,
                "session_collisions": 0,
                "output_collisions": 0,
                "accepted_collisions": 0,
            }
            for attempt, cell, execution, run in zip(
                expected["attempts"],
                expected["cell_ids"],
                expected["execution_ids"],
                expected["run_ids"],
                strict=True,
            )
        ],
    }


def test_successor_binds_highest_qualified_ceiling_and_stays_held() -> None:
    expected = successor.expected_whole_task(ROOT)
    value = successor.build_held(ROOT, _qualified(), _live(expected), _candidate(expected))
    assert value["qualified_distinct_task_concurrency_ceiling"] == 4
    assert value["prepared_distinct_task_controller_count"] == 1
    assert value["controller"]["attempts"] == [1, 2, 3, 4]
    assert value["controller"]["same_task_max_inflight"] == 1
    assert value["controller"]["create_once_job"] is True
    assert value["shared_endpoint_lease"] == {
        "lease_root": str(qualification.LEASE_ROOT),
        "endpoint_key": expected["server_api_run_id"],
        "maximum_distinct_task_streams": 4,
        "same_task_max_inflight": 1,
    }
    assert value["parent_independent_review_required"] is True
    assert value["launch_authorized"] is False
    assert value["scoring_create_permitted"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_successor_rejects_failed_qualifier_and_non_unstarted_cell() -> None:
    expected = successor.expected_whole_task(ROOT)
    qualified = _qualified()
    qualified["status"] = "FAILED"
    qualified["receipt_sha256"] = crypto.digest_without(qualified, "receipt_sha256")
    with pytest.raises(successor.SuccessorError, match="score_free_qualification_not_accepted"):
        successor.build_held(ROOT, qualified, _live(expected), _candidate(expected))
    candidate = _candidate(expected)
    candidate["cells"][2]["ledger_state"] = "active"
    with pytest.raises(successor.SuccessorError, match="whole_task_candidate_not_fresh"):
        successor.build_held(ROOT, _qualified(), _live(expected), candidate)


def test_successor_rejects_concurrent_scored_controller_or_stale_server() -> None:
    expected = successor.expected_whole_task(ROOT)
    live = _live(expected)
    live["active_scored_controller_count"] = 1
    with pytest.raises(successor.SuccessorError, match="fresh_live_successor_gate_failed"):
        successor.build_held(ROOT, _qualified(), live, _candidate(expected))
    live = _live(expected)
    live["server_uid_and_workload_history_unchanged"] = False
    with pytest.raises(successor.SuccessorError, match="fresh_live_successor_gate_failed"):
        successor.build_held(ROOT, _qualified(), live, _candidate(expected))
