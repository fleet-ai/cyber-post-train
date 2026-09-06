"""Validate the terminal score-blind Qwen selector-v2 evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.fleet import qwen_hosted_identity_selector_package_v2 as package
from evals.fleet import qwen_hosted_identity_selector_v2 as selector
from evals.fleet import qwen_hosted_rank17_g22_release_observer_v1 as base

OBSERVATION_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-06-qwen38-hosted-identity-selector-v2-observation.json"
)
TERMINAL_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-06-qwen38-hosted-identity-selector-v2-terminal.json"
)
OBSERVATION_FILE_SHA256 = (
    "sha256:7100a8f062d3f02a665e5d98e64b348968f98f6246b3b37c4215c91db5ae6965"
)
TERMINAL_FILE_SHA256 = (
    "sha256:faa19b2cdb1becb27ee4a14e164c4eef7c3a96c7de97b391665502cc6ab78eb9"
)
JOB = {
    "completion_time": "2026-09-06T17:21:21Z",
    "name": "chris-q38-hosted-identity-selector-v2",
    "terminal_condition": "Complete",
    "uid": "6cd4e7cd-6f58-442c-b57a-db1d46ea075a",
}
POD = {
    "exit_code": 0,
    "finished_at": "2026-09-06T17:21:18Z",
    "name": "chris-q38-hosted-identity-selector-v2-w4gqw",
    "phase": "Succeeded",
    "restarts": 0,
    "uid": "7f7d4277-15d5-4d92-a5a4-6677c7dd6e69",
}


def expected_terminal(observation: dict[str, Any]) -> dict[str, Any]:
    return base.seal(
        {
            "schema_version": "fleet-qwen38-hosted-identity-selector-terminal-v2",
            "status": "NO_CLEAR_CANDIDATE_TERMINAL",
            "classification": "score_blind_identity_ambiguous",
            "job": JOB,
            "pod": POD,
            "observation_path": (
                "/mnt/sfs/jobs/chris-q38-hosted-identity-selector-v2/OBSERVATION.json"
            ),
            "observation_file_sha256": OBSERVATION_FILE_SHA256,
            "observation_receipt_sha256": observation["receipt_sha256"],
            "authoritative_tally": observation["authoritative_tally"],
            "classification_counts": observation["classification_counts"],
            "earliest_clear_rank": observation["earliest_clear_rank"],
            "session_pages_read": observation["session_pages_read"],
            "session_rows_examined": observation["session_rows_examined"],
            "retry_v2_authorized": False,
            "scored_successor_authorized": False,
            "model_calls": 0,
            "task_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
            "protected_values_materialized": False,
            "scores_included": False,
            "prompts_traces_flags_included": False,
            "credentials_included": False,
        }
    )


def validate(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    observation_path = root / OBSERVATION_PATH
    observation_raw = observation_path.read_bytes()
    if base.sha256(observation_raw) != OBSERVATION_FILE_SHA256:
        raise ValueError("selector-v2 observation file drifted")
    observation = base.load(observation_path)
    binding = package.build_binding(root)
    selector.validate_observation(observation, binding)
    if (
        observation["status"] != "NO_CLEAR_CANDIDATE"
        or observation["classification_counts"]
        != {"AMBIGUOUS_BLOCK": 100, "EXACT_MODEL_COLLISION": 0, "IDENTITY_CLEAR": 0}
        or observation["earliest_clear_rank"] is not None
        or observation["runtime"]
        != {"job_uid": JOB["uid"], "pod_uid": POD["uid"]}
    ):
        raise ValueError("selector-v2 observation outcome drifted")
    terminal_path = root / TERMINAL_PATH
    terminal_raw = terminal_path.read_bytes()
    if base.sha256(terminal_raw) != TERMINAL_FILE_SHA256:
        raise ValueError("selector-v2 terminal file drifted")
    terminal = base.load(terminal_path)
    if terminal != expected_terminal(observation):
        raise ValueError("selector-v2 terminal receipt drifted")
    return observation, terminal
