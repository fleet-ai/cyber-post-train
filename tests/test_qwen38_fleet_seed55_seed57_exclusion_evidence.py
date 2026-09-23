from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study"

FILES = {
    "seed55_invalidation": (
        "2026-09-23-q38-dev17-seed55-pair-invalidation.json",
        "6ac8ce9b0c9db5337d392e5705ae4f4fbb1ebdedd38b1f6d47a9f28a19bb3f45",
        "sha256:321e750c7a66be292bd724dc28dfcbd8b79f02344da6ce3ddcf3a716392e7080",
    ),
    "seed55_terminal": (
        "2026-09-23-q38-dev17-seed55-base-terminal-observation.json",
        "6d5525309f0d275f5940257fd4ec9a011134a5e9392607760187c9ca1af517d1",
        "sha256:ea7a7258890c1fd583dcf714c239ee84ed97aaec02808d36a72dea30b6a611fb",
    ),
    "seed55_candidate_pre": (
        "2026-09-23-q38-dev17-seed55-candidate-retirement-pre.json",
        "93e68b175f1f526709c10dd5b9582b7d84c9984d455c34ea2b6f1a6c9102d11d",
        "sha256:02d5c71f722d2aecfbe3719b7f57016369ea5d09aa9eae8801ba074d6894e3e7",
    ),
    "seed55_candidate_post": (
        "2026-09-23-q38-dev17-seed55-candidate-retirement-receipt.json",
        "3ad589068ace08483d93cd36eb87de446fc02d227f207f0b3e4a5aa1a441bb81",
        "sha256:6e6807cfddfc94a903320221221a10d79f44ed3f27b8d48ba753b2d3f119b4f9",
    ),
    "seed57_invalidation": (
        "2026-09-23-q38-dev17-seed57-pair-invalidation.json",
        "c26c4a86d580faa69f469f76144a75285bf6ff559b616f9735c0b5818906fb3c",
        "sha256:8334a43d901d1b8a8f8d7d52f0038adc54e98b03a8963e4d45f5a7915bdf88cd",
    ),
    "seed57_pre": (
        "2026-09-23-q38-dev17-seed57-pair-retirement-pre.json",
        "5a74a84acf7acf3f98831cdc5116941e373dbc85dacec306661bb8b9c41f764c",
        "sha256:aff75bc2005d49e77712eb9a586fe66052fde49692efa6bd1ba448767afb8cfa",
    ),
    "seed57_post": (
        "2026-09-23-q38-dev17-seed57-pair-retirement-receipt.json",
        "336ae0293eec925bf7705d03c155a914f0e14a82c060a33e6f2f304ef32ec5b2",
        "sha256:70d268a69069ee21d4a72fd08f61f9bfaded4a79dbbfbaf596c6fd955fcc0944",
    ),
}

SEED55_SUMMARY_PRIVACY = {
    "cell_session_task_ids_included": False,
    "credentials_included": False,
    "prompts_responses_flags_rewards_or_trace_content_read_or_included": False,
    "score_blind": True,
    "scores_read_or_included": False,
}
SEED55_TERMINAL_PRIVACY = {
    "credentials_included": False,
    "prompts_responses_flags_rewards_or_trace_content_included": False,
    "score_values_included": False,
}
SEED55_RETIREMENT_PRIVACY = {
    "credentials_read": False,
    "prompts_read": False,
    "score_blind": True,
    "scores_read": False,
    "traces_read": False,
}
SEED57_PRIVACY = {
    "cell_session_task_ids_included": False,
    "credentials_included": False,
    "prompts_responses_flags_rewards_or_trace_content_read_or_included": False,
    "score_blind": True,
    "scores_read_or_included": False,
}
EXPECTED_SCHEMA_PRIVACY = {
    "seed55_invalidation": (
        "q38_dev17_protocol_v2_whole_pair_invalidation_snapshot_v1",
        SEED55_SUMMARY_PRIVACY,
    ),
    "seed55_terminal": (
        "cyber_fleet_heldout_terminal_observation_v1",
        SEED55_TERMINAL_PRIVACY,
    ),
    "seed55_candidate_pre": (
        "q38_dev17_protocol_v2_exact_job_retirement_pre_v1",
        SEED55_RETIREMENT_PRIVACY,
    ),
    "seed55_candidate_post": (
        "q38_dev17_protocol_v2_exact_job_retirement_receipt_v1",
        SEED55_RETIREMENT_PRIVACY,
    ),
    "seed57_invalidation": (
        "q38_dev17_protocol_v2_whole_pair_invalidation_snapshot_v1",
        SEED57_PRIVACY,
    ),
    "seed57_pre": (
        "q38_dev17_protocol_v2_exact_pair_retirement_pre_v1",
        SEED57_PRIVACY,
    ),
    "seed57_post": (
        "q38_dev17_protocol_v2_exact_pair_retirement_receipt_v1",
        SEED57_PRIVACY,
    ),
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
SAFE_TRUE_PRIVACY_CONTROLS = {"score_blind", "score_blind_reconciliation_required"}


def _canonical(value: dict[str, Any]) -> str:
    copy = dict(value)
    expected = copy.pop("sha256")
    actual = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(copy, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert actual == expected
    return expected


def _read(name: str) -> dict[str, Any]:
    path = EVIDENCE / name
    assert path.is_file() and not path.is_symlink()
    value = json.loads(path.read_text())
    _canonical(value)
    return value


def _assert_score_blind(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = key.lower()
            assert lowered not in FORBIDDEN_ID_KEYS
            if lowered in SAFE_TRUE_PRIVACY_CONTROLS:
                assert nested is True
            elif lowered in SAFE_ZERO_ID_AGGREGATES:
                assert nested == 0
            elif any(part in lowered for part in PRIVATE_PAYLOAD_KEY_PARTS):
                assert nested is False
            _assert_score_blind(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_score_blind(nested)


def test_exact_score_blind_evidence_bytes_are_preserved() -> None:
    for name, (filename, file_sha256, self_sha256) in FILES.items():
        path = EVIDENCE / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256, name
        value = _read(filename)
        assert value["sha256"] == self_sha256, name
        expected_schema, expected_privacy = EXPECTED_SCHEMA_PRIVACY[name]
        assert value["schema"] == expected_schema, name
        assert value["privacy"] == expected_privacy, name
        _assert_score_blind(value)


def test_seed55_is_whole_pair_invalid_without_cell_splicing() -> None:
    summary = _read("2026-09-23-q38-dev17-seed55-pair-invalidation.json")
    terminal = _read(FILES["seed55_terminal"][0])
    pre = _read(FILES["seed55_candidate_pre"][0])
    post = _read(FILES["seed55_candidate_post"][0])
    assert summary["classification"] == (
        "whole_pair_infrastructure_invalid_requires_fresh_paired_successor"
    )
    assert summary["decision"] == {
        "accepted_now": False,
        "cell_level_splicing_allowed": False,
        "existing_session_reconciliation_performed": False,
        "next_gate": "exclude_seed55_pair_and_version_fresh_paired_successor",
        "whole_pair_excluded": True,
    }
    assert terminal["database"]["summary"]["by_state"]["accepted"] == 12
    assert terminal["database"]["summary"]["by_state"]["retry_review"] == 5
    assert terminal["database"]["summary"]["local_results"] == 17
    cohorts = summary["reason"]["cohorts"]
    assert [(row["agent_termination"], row["count"]) for row in cohorts] == [
        ("output_limit", 4),
        ("malformed_trace", 1),
    ]
    assert cohorts[1]["structural_class"] == "unsupported_agent_outcome"
    assert pre["reason"]["seed"] == 55
    assert pre["job"]["started_sessions"] == 0
    assert post["post_release"] == {
        "exact_job_count": 0,
        "exact_owner_pod_count": 0,
        "exact_owner_workload_count": 0,
        "started_sessions": 0,
        "target_database_exists": False,
    }


def test_seed57_is_whole_pair_invalid_and_exact_pair_is_released() -> None:
    invalidation = _read(FILES["seed57_invalidation"][0])
    pre = _read(FILES["seed57_pre"][0])
    post = _read(FILES["seed57_post"][0])
    assert invalidation["classification"] == (
        "whole_pair_infrastructure_invalid_requires_fresh_paired_successor"
    )
    unrecoverable = invalidation["reason"]["cohorts"][1]
    assert unrecoverable == {
        "agent_exit_code": None,
        "agent_termination": None,
        "count": 1,
        "failure_code": "post_claim.remoteprotocolerror",
        "has_local_result": False,
        "has_session": False,
        "session_ingest_status": None,
        "structural_class": "unrecoverable_no_session",
    }
    assert pre["reason"]["seed"] == 57
    assert pre["reason"]["class"] == "exact_valid8_whole_pair_exclusion"
    released = post["post_release"]
    database = released["candidate_database"]
    assert database["accepted"] == 6
    assert database["retry_review"] == 8
    assert database["local_results"] == 13
    assert database["claimed_or_running_or_grading"] == 3
    assert database["claimed_rows_with_session_id"] == 0
    assert released["live_session_ids_attached_to_claims"] == 0
    assert released["residual_lease_bound_claims_without_session"] == 3
    assert released["exact_job_count"] == 0
    assert released["exact_owner_pod_count"] == 0
    assert released["exact_owner_workload_count"] == 0
    assert released["kueue_quota_claims_released"] is True
    assert released["ledger_claims_mutated"] is False
    assert post["scope"] == {
        "other_jobs_mutated": False,
        "seed": 57,
        "serving_route_mutated": False,
        "whole_pair_excluded": True,
    }
