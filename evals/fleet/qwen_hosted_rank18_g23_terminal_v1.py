"""Exact sanitized terminal authority for the rank-18 g23 release observer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

EVIDENCE_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank18-g23-terminal-v1.json"
)
FAILURE_EVIDENCE_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-06-qwen38-hosted-rank18-g23-release-observer-failed-v1.json"
)
JOB_UID = "7e6e3883-63c7-4587-991a-6fce9788a8a3"
POD_UID = "6b70e0f4-1570-43fe-acdc-3ed7525a7143"
FAILURE_RECEIPT_SHA256 = (
    "sha256:115c51cce93f37103d5457576c68b7593d85c92642c5af38ea74c6d776916fbd"
)


def failure_receipt() -> dict[str, Any]:
    return {
        "api_mutations": 0,
        "credentials_included": False,
        "failure_category": "safe_gate_failure",
        "failure_code": "fleet_session_identity_ambiguous",
        "job_uid": JOB_UID,
        "last_stage": "collect",
        "model_calls": 0,
        "pod_uid": POD_UID,
        "prompts_traces_flags_included": False,
        "receipt_sha256": FAILURE_RECEIPT_SHA256,
        "schema_version": "fleet-qwen38-hosted-rank18-g23-release-failure-v1",
        "scores_included": False,
        "scoring_calls": 0,
        "session_mutations": 0,
        "status": "FAILED",
        "task_calls": 0,
        "verifier_calls": 0,
    }


def validate_failure(value: Mapping[str, Any]) -> None:
    if dict(value) != failure_receipt() or value.get("receipt_sha256") != (
        self_hosted.digest_without(dict(value), "receipt_sha256")
    ):
        raise ValueError("rank18 sanitized failure receipt drifted")


def terminal_receipt() -> dict[str, Any]:
    value = {
        "schema_version": "fleet-qwen38-hosted-rank18-g23-terminal-v1",
        "status": "TERMINAL_BLOCKED_NONREPEATABLE",
        "job": {
            "name": "chris-q38-hosted-r018-release-gate-g23-v1",
            "uid": JOB_UID,
            "terminal_condition": "Failed",
        },
        "pod": {"uid": POD_UID, "phase": "Failed", "exit_code": 1, "restarts": 0},
        "observer": {
            "observation_present": False,
            "failure_present": True,
            "failure_receipt_sha256": FAILURE_RECEIPT_SHA256,
            "failure_receipt_digest_valid": True,
            "failure_code": "fleet_session_identity_ambiguous",
            "last_stage": "collect",
        },
        "authority": {
            "rank": 18,
            "execution_generation": 23,
            "retry_authorized": False,
            "scored_successor_authorized": False,
            "rank_walk_authorized": False,
            "metadata_contract_or_bounded_inventory_required": True,
        },
        "tally_before": {
            "accepted": 51,
            "active": 0,
            "blocked_nonrepeatable": 8,
            "unstarted": 341,
        },
        "tally_after": {
            "accepted": 51,
            "active": 0,
            "blocked_nonrepeatable": 9,
            "unstarted": 340,
        },
        "side_effects": {
            "model_calls": 0,
            "task_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "scored_jobs_created": 0,
        },
        "privacy": {
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        },
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate(value: Mapping[str, Any]) -> None:
    if dict(value) != terminal_receipt():
        raise ValueError("rank18 terminal authority drifted")


def load(root: Path) -> dict[str, Any]:
    failure = json.loads((root / FAILURE_EVIDENCE_PATH).read_text())
    if not isinstance(failure, dict):
        raise ValueError("rank18 sanitized failure receipt must be an object")
    validate_failure(failure)
    value = json.loads((root / EVIDENCE_PATH).read_text())
    if not isinstance(value, dict):
        raise ValueError("rank18 terminal authority must be an object")
    validate(value)
    return value
