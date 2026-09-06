from __future__ import annotations

import pytest

from evals.fleet import qwen38_dp8_rank98_canary_partition_v1 as partition
from evals.fleet import self_hosted


def _qualification() -> dict:
    return {
        "plan_receipt_sha256": "sha256:" + "1" * 64,
        "receipt_sha256": "sha256:" + "2" * 64,
        "highest_passing_concurrency": 8,
        "scored_calls": 0,
        "levels": [
            {"concurrency": value, "status": "PASSED"} for value in (1, 2, 4, 8)
        ],
    }


def _server() -> dict:
    value = {
        "status": "READY_NON_SCORED",
        "title": partition.early.TITLE,
        "run_dir": partition.early.RUN_DIR,
        "serving_block": partition.early.SERVING_BLOCK,
        "api_run_id": "ft-run-12345678",
        "rayjob_uid": "11111111-1111-4111-8111-111111111111",
        "workload_uid": "22222222-2222-4222-8222-222222222222",
        "head_pod_uid": "33333333-3333-4333-8333-333333333333",
        "service_uid": "44444444-4444-4444-8444-444444444444",
        "service_origin": "http://ft-run-12345678-head-svc.example:8000",
        "served_id": "qwen3.8-27b",
        "context_length": 262144,
        "tensor_parallel_size": 1,
        "data_parallel_size": 8,
        "head_pod_running_ready": True,
        "head_pod_restarts": 0,
        "workload_preempted": False,
        "parity_receipt_sha256": "sha256:" + "3" * 64,
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def _scan(server: dict) -> dict:
    value = {
        "schema_version": partition.LIVE_SCAN_SCHEMA,
        "ledger_path": partition.LEDGER_PATH,
        "ledger_receipt_sha256": partition.LEDGER_RECEIPT_SHA256,
        "ledger_file_sha256": partition.LEDGER_FILE_SHA256,
        "ledger_state_counts": {
            "accepted": 12,
            "active": 2,
            "blocked": 3,
            "unstarted": 383,
        },
        "server_binding_receipt_sha256": server["receipt_sha256"],
        "task_key": partition.TASK_KEY,
        "task_version_id": partition.TASK_VERSION_ID,
        "task_version": partition.TASK_VERSION,
        "selection_rank": partition.SELECTION_RANK,
        "task_session_rows_examined": 26,
        "exact_session_matches": 0,
        "matching_verifier_executions": 0,
        "controller_job_collisions": 0,
        "controller_sfs_collisions": 0,
        "g19_skip_reason": "global_execution_claim_already_exists",
        "g19_skips_all_four_canonical_claims": True,
        "g19_runtime_path": partition.G19_RUNTIME_PATH,
        "g19_runtime_file_sha256": partition.G19_RUNTIME_FILE_SHA256,
        "g19_controller_package_sha256": "sha256:" + "4" * 64,
        "api_mutation_calls": 0,
        "cells": [
            {
                "attempt": attempt,
                "cell_id": cell,
                "execution_id": execution,
                "run_id": run,
                "ledger_state": "unstarted",
                "latest_generation": 0,
                "accepted_receipt_matches": 0,
                "claim_path_exists": False,
                "output_root_exists": False,
            }
            for attempt, cell, execution, run in zip(
                range(1, 5),
                partition.CELL_IDS,
                partition.EXECUTION_IDS,
                partition.RUN_IDS,
                strict=True,
            )
        ],
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def test_partition_binds_server_ledger_and_all_four_g19_claims(monkeypatch) -> None:
    monkeypatch.setattr(partition.ladder, "validate_result", lambda *_args: 8)
    server = _server()
    value = partition.build_held(_qualification(), server, _scan(server))
    assert value["server_binding"] == server
    assert value["ledger"] == {
        "path": partition.LEDGER_PATH,
        "receipt_sha256": partition.LEDGER_RECEIPT_SHA256,
        "file_sha256": partition.LEDGER_FILE_SHA256,
    }
    claims = value["partition"]["claims"]
    assert len(claims) == 4
    assert [row["initial_state"] for row in claims] == [
        "active",
        "reservation_only",
        "reservation_only",
        "reservation_only",
    ]
    assert value["partition"]["g19_skips_all_four"] is True
    assert value["receipt_sha256"] == self_hosted.digest_without(
        value, "receipt_sha256"
    )


def test_partition_authorizes_only_a1_and_stays_held(monkeypatch) -> None:
    monkeypatch.setattr(partition.ladder, "validate_result", lambda *_args: 8)
    server = _server()
    value = partition.build_held(_qualification(), server, _scan(server))
    assert value["canary"]["only_authorized_attempt"] == 1
    assert value["canary"]["attempts_2_to_4_model_calls_authorized"] is False
    assert value["launch_authorized"] is False
    assert value["scoring_authorized"] is False
    assert value["controller_create_permitted"] is False
    assert value["api_mutation_calls"] == 0


def test_partition_requires_complete_c8_and_exact_fresh_candidate(monkeypatch) -> None:
    monkeypatch.setattr(partition.ladder, "validate_result", lambda *_args: 4)
    server = _server()
    with pytest.raises(partition.PartitionError, match="complete_c1_to_c8"):
        partition.build_held(_qualification(), server, _scan(server))

    monkeypatch.setattr(partition.ladder, "validate_result", lambda *_args: 8)
    scan = _scan(server)
    scan["cells"][2]["claim_path_exists"] = True
    scan["receipt_sha256"] = self_hosted.digest_without(scan, "receipt_sha256")
    with pytest.raises(partition.PartitionError, match="not_fresh"):
        partition.build_held(_qualification(), server, scan)


def test_reservation_recovery_never_silently_transfers(monkeypatch) -> None:
    monkeypatch.setattr(partition.ladder, "validate_result", lambda *_args: 8)
    server = _server()
    value = partition.build_held(_qualification(), server, _scan(server))
    reservation = value["atomic_reservation"]
    continuation = value["continuation"]
    assert reservation["rollback_requires_same_job_and_pod_uids"] is True
    assert reservation["rollback_requires_model_request_count_zero"] is True
    assert continuation["after_any_model_call_attempt1_irrevocable"] is True
    assert continuation["reservation_only_claims_may_not_silently_transfer"] is True
    assert continuation["owner_or_server_identity_loss_blocks_attempts_2_to_4"] is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("workload_preempted", True),
        ("head_pod_running_ready", False),
        ("head_pod_restarts", 1),
        ("data_parallel_size", 4),
    ],
)
def test_partition_rejects_server_identity_or_health_drift(
    monkeypatch, field: str, replacement: object
) -> None:
    monkeypatch.setattr(partition.ladder, "validate_result", lambda *_args: 8)
    server = _server()
    server[field] = replacement
    server["receipt_sha256"] = self_hosted.digest_without(server, "receipt_sha256")
    with pytest.raises(partition.PartitionError, match="server_binding"):
        partition.build_held(_qualification(), server, _scan(server))


def test_module_has_no_submit_or_mutation_rail() -> None:
    assert not hasattr(partition, "submit")
    assert not hasattr(partition, "create_claim")
    assert not hasattr(partition, "write_release")
