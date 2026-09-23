from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study"

FILES = {
    "snapshot": (
        "2026-09-23-q38-dev17-seed59-pair-provisional-noartifact.json",
        "5d1a9c70699e375e9bddf82d4e3746be98a6c6b2a25c23199089d47bd0d4b630",
        "sha256:196771c9acd4cd684f46d337c62f34ee987451df52bd9986d3bb6549d3e58386",
        "q38_dev17_s59_pair_provisional_noartifact_snapshot_v1",
    ),
    "pre": (
        "2026-09-23-q38-dev17-seed59-pair-retirement-pre.json",
        "a2b341b186a23701c240d16a57cc54c670fad94b5bdea3a31eeb3c15de174666",
        "sha256:aa58aa7d4934dfa9813340f8234b1a49d6d742679e83cc4684e75b83ec4803e2",
        "q38_dev17_protocol_v2_exact_pair_retirement_pre_v1",
    ),
    "invalid_v1": (
        "2026-09-23-q38-dev17-seed59-pair-retirement-invalid-v1.json",
        "35235bd522ba7b188eaee9a3f0de3c82e7a3327270d4e8ef2ce6c591de5ea11e",
        "sha256:5c4d2812706ffd69b1ed96a9c9bfff1f9b0c336e2c50e680ba82e6fd536dd02a",
        "q38_dev17_protocol_v2_exact_pair_retirement_receipt_v1",
    ),
    "post_v2": (
        "2026-09-23-q38-dev17-seed59-pair-retirement-receipt-v2.json",
        "b1bca260da6d9b5b869b2a576fff33d539e14fdd183b8f8c721e45bda75379f6",
        "sha256:b45f9e2b5e88489d9b43e75c5abd1b2325d75f9fa32839bef31da113545290d6",
        "q38_dev17_protocol_v2_exact_pair_retirement_receipt_v2",
    ),
}

SNAPSHOT_PRIVACY = {
    "score_blind": True,
    "scores_read_or_included": False,
    "task_cell_session_or_verifier_ids_included": False,
    "prompts_responses_flags_rewards_or_trace_content_read_or_included": False,
    "credentials_read_or_included": False,
}
PAIR_PRIVACY = {
    "score_blind": True,
    "scores_read_or_included": False,
    "cell_session_task_or_verifier_ids_included": False,
    "prompts_responses_flags_rewards_or_trace_content_read_or_included": False,
    "credentials_included": False,
}
FORBIDDEN_ID_KEYS = {
    "cell_id",
    "execution_id",
    "ledger_cell_id",
    "run_id",
    "session_id",
    "task_id",
    "task_key",
    "task_version_id",
    "verifier_execution_id",
    "verifier_id",
}
PRIVATE_PAYLOAD_KEY_PARTS = ("credential", "flag", "prompt", "response", "reward", "score", "trace")
SAFE_ZERO_ID_AGGREGATES = {
    "claimed_rows_with_session_id",
    "live_session_ids_attached_to_claims",
}
SAFE_INFRASTRUCTURE_RESPONSE = {
    "response_kind": "Job",
    "response_name_matched": True,
    "response_uid_matched": True,
}


def _canonical(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "sha256"}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def _read(label: str) -> dict[str, Any]:
    filename = FILES[label][0]
    path = EVIDENCE / filename
    assert path.is_file() and not path.is_symlink()
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _assert_score_blind(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = key.lower()
            assert lowered not in FORBIDDEN_ID_KEYS
            if lowered == "score_blind":
                assert nested is True
            elif lowered in SAFE_ZERO_ID_AGGREGATES:
                assert nested == 0
            elif lowered in SAFE_INFRASTRUCTURE_RESPONSE:
                assert nested == SAFE_INFRASTRUCTURE_RESPONSE[lowered]
            elif any(part in lowered for part in PRIVATE_PAYLOAD_KEY_PARTS):
                assert nested is False
            _assert_score_blind(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_score_blind(nested)


def test_exact_seed59_score_blind_evidence_bytes_are_preserved() -> None:
    for label, (filename, file_sha256, self_sha256, schema) in FILES.items():
        path = EVIDENCE / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256, label
        value = _read(label)
        assert value["schema"] == schema, label
        assert value["sha256"] == self_sha256, label
        assert value["privacy"] == (SNAPSHOT_PRIVACY if label == "snapshot" else PAIR_PRIVACY)
        _assert_score_blind(value)


def test_invalid_v1_is_preserved_but_only_canonical_v2_is_authoritative() -> None:
    invalid = _read("invalid_v1")
    post = _read("post_v2")
    canonical_invalid = "sha256:7b740197ad415576804ecdbc74b118f9974a6de33ac557962488ef3c367f44f1"
    assert _canonical(invalid) == canonical_invalid
    assert invalid["sha256"] != canonical_invalid
    assert _canonical(post) == post["sha256"]
    assert post["supersedes_invalid_receipt"] == {
        "path": (
            "/private/tmp/q38-s59-provisional-noartifact.20260923T121234Z/"
            "RETIREMENT_RECEIPT_INVALID_V1.json"
        ),
        "file_sha256": "sha256:" + FILES["invalid_v1"][1],
        "embedded_self_sha256": FILES["invalid_v1"][2],
        "canonical_self_sha256": canonical_invalid,
        "reason": (
            "embedded self digest included a trailing newline and failed canonical verification"
        ),
    }


def test_seed59_is_whole_pair_invalid_and_exact_pair_is_released() -> None:
    snapshot = _read("snapshot")
    pre = _read("pre")
    post = _read("post_v2")
    classification = snapshot["provisional_classification"]
    assert classification["unsupported_no_artifact_cell_present"] is True
    assert classification["unsupported_signature"] == "post_claim.connecterror"
    assert classification["counterpart_sessions_started"] == 0
    assert classification["mutation_authorized"] is False
    assert pre["reason"]["class"] == "exact_valid8_whole_pair_exclusion"
    assert pre["base_database"] == {
        "total": 17,
        "accepted": 5,
        "claimed": 4,
        "pending": 6,
        "retry_review": 2,
        "local_results": 6,
        "stale_active": 0,
        "claimed_rows_with_session_id": 0,
        "max_heartbeat_age_seconds": 31,
        "min_lease_remaining_seconds": 269,
    }
    released = post["post_release"]
    assert released["exact_job_count"] == 0
    assert released["exact_owner_pod_count"] == 0
    assert released["exact_owner_workload_count"] == 0
    assert released["kueue_quota_claims_released"] is True
    assert released["ledger_claims_mutated"] is False
    assert released["live_session_ids_attached_to_claims"] == 0
    assert released["residual_lease_bound_claims_without_session"] == 4
    database = released["base_database"]
    assert {
        key: database[key]
        for key in (
            "total",
            "accepted",
            "claimed",
            "pending",
            "retry_review",
            "local_results",
            "stale_active",
            "claimed_rows_with_session_id",
        )
    } == {
        "total": 17,
        "accepted": 5,
        "claimed": 4,
        "pending": 6,
        "retry_review": 2,
        "local_results": 6,
        "stale_active": 0,
        "claimed_rows_with_session_id": 0,
    }
    unrecoverable = next(
        row
        for row in database["retry_review_signatures"]
        if row["failure_code"] == "post_claim.connecterror"
    )
    assert unrecoverable == {
        "failure_code": "post_claim.connecterror",
        "agent_exit_code": None,
        "agent_termination": None,
        "session_ingest_status": None,
        "count": 1,
        "has_local_result": False,
    }
    assert post["scope"] == {
        "seed": 59,
        "whole_pair_excluded": True,
        "other_jobs_mutated": False,
        "serving_route_mutated": False,
    }
    assert post["budget"] == {
        "reference_receipt_file_sha256": (
            "sha256:e0190275fbb5c80bcbf8b8d3ff2af9e8c25e17977f7165f5016a3b891d24f2cf"
        ),
        "reference_receipt_self_sha256": (
            "sha256:613ee62e980b23bc95cd958400a21e4479c1f28c631fc11340bde1af7ffc5b54"
        ),
        "projected_with_seed64": 486,
        "daily_cap": 500,
        "remaining": 14,
        "retirement_credit": 0,
        "fresh_census_required_before_successor_launch": True,
    }
