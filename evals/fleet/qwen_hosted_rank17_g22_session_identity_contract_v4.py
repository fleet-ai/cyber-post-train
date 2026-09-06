"""Held rank-17 release contract for a future metadata-only session API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

HELD_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-06-qwen38-hosted-rank17-g22-release-held-v4.json"
)


def held_contract() -> dict[str, Any]:
    value = {
        "schema_version": "fleet-qwen38-hosted-rank17-g22-release-contract-v4",
        "status": "HELD_BLOCKED_METADATA_ONLY_SESSION_IDENTITY_API",
        "launch_authorized": False,
        "scoring_authorized": False,
        "rank17_ledger_reclassified": False,
        "terminal_predecessor": {
            "job_uid": "cb0b2df6-48dd-49da-8ec5-b1d722acac46",
            "pod_uid": "c6cc35c9-6ca4-41af-bda8-a5c972981987",
            "status": "TERMINAL_FAILED_NO_RETRY",
            "failure_code": "fleet_session_identity_ambiguous",
            "observation_present": False,
            "model_calls": 0,
            "task_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
        },
        "v3_privacy_caveat": {
            "session_list_route_exposed_verifier_execution": True,
            "v3_json_decoder_materialized_full_session_rows": True,
            "protected_value_persisted_or_emitted": False,
            "protected_value_inspected_by_human": False,
            "v3_score_blind_release_claim_retracted": True,
            "v3_result_not_usable_as_release_authority": True,
        },
        "reviewed_public_source": {
            "repository": "fleet-ai/theseus",
            "commit": "0fff262678a80901fee6aa398e0142039e8a40bc",
            "sessions_source_sha256": (
                "sha256:6bed21b587c8a60b49769ef53b896569d1b684940304419e317040f32ebe5943"
            ),
            "tasks_source_sha256": (
                "sha256:65e35b88d54831154b1c54c77a708bd7cf26d1a714623c3bebc51e39faa16d94"
            ),
            "models_source_sha256": (
                "sha256:392344878c98e4e87853d66a1186a81abc65e76c4755ebd382c07962462c2b79"
            ),
            "session_summary_fields": [
                "session_id",
                "eval_task_id",
                "task_key",
                "model",
                "status",
                "created_at",
                "started_at",
                "ended_at",
                "verifier_execution",
            ],
            "session_task_version_id_exposed": False,
            "verifier_execution_summary_exposed": True,
            "generic_session_detail_route_exposed": False,
            "task_response_exact_eval_task_id_exposed": False,
            "task_response_exact_task_version_id_exposed": False,
            "task_response_contains_protected_task_or_verifier_fields": True,
        },
        "live_route_shape_probe": {
            "observed_at_utc": "2026-09-06T16:30:19Z",
            "fake_session_metadata_status": 404,
            "fake_session_reference_route_status": 401,
            "response_bodies_read": False,
        },
        "required_unblock_contract": {
            "metadata_only": True,
            "team_scoped": True,
            "list_fields": [
                "session_id",
                "eval_task_id",
                "task_key",
                "model",
                "status",
            ],
            "detail_fields": [
                "session_id",
                "eval_task_id",
                "eval_task_version_id",
                "task_key",
                "model",
            ],
            "forbidden_fields": [
                "prompt",
                "transcript",
                "reference_traces",
                "tool_use_workflow",
                "verifier",
                "verifier_execution",
                "score",
                "flag",
            ],
            "list_detail_identity_agreement_required": True,
            "missing_conflicting_or_extra_detail_fails_closed": True,
            "deployed_behavioral_proof_required": True,
            "fresh_observer_identity_required_after_deployment": True,
        },
        "reserved_uncreated_v4_identity": {
            "job_name": "chris-q38-hosted-r017-release-gate-g22-v4",
            "configmap_name": "chris-q38-hosted-r017-release-gate-g22-v4-package",
            "output_root": "/mnt/sfs/jobs/chris-q38-hosted-r017-release-gate-g22-v4",
            "created": False,
        },
        "privacy": {
            "live_session_payloads_read": False,
            "statement_scope": "v4_source_audit_and_route_shape_probe_only",
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate(value: Mapping[str, Any]) -> None:
    if dict(value) != held_contract():
        raise ValueError("rank17 v4 metadata contract drifted")
