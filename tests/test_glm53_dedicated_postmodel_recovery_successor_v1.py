import copy
import json
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_crypto as crypto
from evals.fleet import glm53_dedicated_postmodel_recovery_successor_v1 as recovery

ROOT = Path(__file__).resolve().parents[1]


def _ledger() -> dict:
    value = {
        "schema_version": "fleet-exact-pass4-global-ledger-v1",
        "validated": True,
        "file_sha256": "sha256:" + "7" * 64,
        "state_counts": {"accepted": 10, "active": 0, "blocked": 2, "unstarted": 388},
        "all_accepted_ingested_ambiguous_cells_excluded": True,
    }
    value["receipt_sha256"] = crypto.digest_without(value, "receipt_sha256")
    return value


def _candidate() -> dict:
    return {
        "selection_rank": recovery.SELECTION_RANK,
        "task_version_id": recovery.TASK_VERSION_ID,
        "cells": [
            {
                "attempt": attempt,
                "cell_id": cell,
                "execution_id": execution,
                "run_id": run,
                "ledger_state": "unstarted",
                "latest_generation": 0,
                "accepted_generations": 0,
                "session_matches": 0,
                "verifier_matches": 0,
                "claim_collisions": 0,
                "output_collisions": 0,
                "accepted_receipt_collisions": 0,
            }
            for attempt, cell, execution, run in zip(
                range(1, 5),
                recovery.CELL_IDS,
                recovery.EXECUTION_IDS,
                recovery.RUN_IDS,
                strict=True,
            )
        ],
    }


def test_comparison_binds_shared_post_model_boundary_and_score_free_discriminator() -> None:
    value = recovery.comparison()
    assert value["shared_observations"]["model_interaction_before_failure"] is True
    assert value["shared_observations"]["result_present"] is False
    assert value["shared_observations"]["session_ingest_present"] is False
    assert value["smallest_reusable_fix"] == {
        "late_body_blind_scoring_route_preflight": "GET_EXPECT_405",
        "persist_failure_fields": ["method", "route", "http_status"],
        "preflight_mutation_calls": 0,
        "response_body_read": False,
        "automatic_retry": False,
    }
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_recovery_successor_is_create_once_and_unconditionally_held() -> None:
    value = recovery.build_held(_ledger(), _candidate())
    assert value["server"]["jobs_api_create_once"] is True
    assert value["server"]["preemption_policy"] == "Never"
    assert value["controller"]["same_task_max_inflight"] == 1
    assert value["controller"]["authoritative_acceptance_before_next_attempt"] is True
    assert value["gpu_server_submit_permitted"] is False
    assert value["scoring_create_permitted"] is False
    assert value["launch_authorized"] is False
    assert value["api_mutation_calls"] == 0
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ledger_state", "accepted"),
        ("latest_generation", 1),
        ("accepted_generations", 1),
        ("session_matches", 1),
        ("verifier_matches", 1),
        ("claim_collisions", 1),
        ("output_collisions", 1),
        ("accepted_receipt_collisions", 1),
    ],
)
def test_recovery_successor_rejects_consumed_or_ambiguous_candidate(
    field: str, value: object
) -> None:
    candidate = _candidate()
    candidate["cells"][0][field] = value
    with pytest.raises(recovery.RecoveryPlanError, match="candidate_is_not_wholly_unstarted"):
        recovery.build_held(_ledger(), candidate)


def test_recovery_successor_rejects_consumed_identity_and_incomplete_ledger() -> None:
    candidate = _candidate()
    candidate["cells"][0]["cell_id"] = recovery.CONSUMED_INCIDENTS[0]["cell_id"]
    with pytest.raises(recovery.RecoveryPlanError, match="candidate_is_not_wholly_unstarted"):
        recovery.build_held(_ledger(), candidate)

    ledger = copy.deepcopy(_ledger())
    ledger["all_accepted_ingested_ambiguous_cells_excluded"] = False
    ledger["receipt_sha256"] = crypto.digest_without(ledger, "receipt_sha256")
    with pytest.raises(recovery.RecoveryPlanError, match="global_ledger_not_authoritative"):
        recovery.build_held(ledger, _candidate())


def test_v23_authenticated_preview_is_digest_valid_and_still_held() -> None:
    path = (
        ROOT
        / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-v23-jobs-api-preview-held-v1.json"
    )
    value = json.loads(path.read_bytes())
    assert value["status"] == "PASSED_PREVIEW_ONLY"
    assert value["preview_http_status"] == 200
    assert value["api_mutation_calls"] == 0
    assert value["rendered"]["gpus"] == 8
    assert value["rendered"]["priority"] == "fleet-infra-quiet"
    assert value["rendered"]["preemption_policy"] == "Never"
    assert value["rendered"]["run_dir"] == recovery.SERVER_RUN_DIR
    assert value["api_title_matches"] == 0
    assert value["api_run_dir_matches"] == 0
    assert value["kubernetes_identity_matches"] == 0
    assert value["sfs_absence_not_yet_bound"] is True
    assert value["future_full_capacity_gate_required"] is True
    assert value["launch_authorized"] is False
    assert value["gpu_server_submit_permitted"] is False
    assert value["scoring_create_permitted"] is False
    assert value["receipt_sha256"] == crypto.digest_without(value, "receipt_sha256")


def test_root_review_checklist_keeps_all_live_gates_explicit() -> None:
    text = (
        ROOT
        / "docs/evidence/glm53-study/"
        "2026-09-06-glm53-dedicated-recovery-root-review-checklist-v1.md"
    ).read_text()
    for gate in (
        "self-digested 400-cell global ledger",
        "two-node/16-GPU",
        "preemptionPolicy: Never",
        "complete Kueue Workload condition history",
        "actual-request counter",
        "600-second idle release",
        "CPU full-path",
        "Root independently reviews",
    ):
        assert gate in text
