"""Held contract for a score-blind, identity-complete session projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evals.fleet import self_hosted


def contract() -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "fleet-qwen38-session-identity-projection-contract-v1",
        "status": "HELD_BLOCKED_UNDEPLOYED_METADATA_CONTRACT",
        "launch_authorized": False,
        "scoring_authorized": False,
        "api_methods": ["GET"],
        "authoritative_tally": {
            "accepted": 51,
            "active": 0,
            "blocked_nonrepeatable": 9,
            "unstarted": 340,
        },
        "selector_v2_terminal": {
            "job_uid": "6cd4e7cd-6f58-442c-b57a-db1d46ea075a",
            "pod_uid": "7f7d4277-15d5-4d92-a5a4-6677c7dd6e69",
            "status": "NO_CLEAR_CANDIDATE",
            "identity_counts": {
                "AMBIGUOUS_BLOCK": 100,
                "EXACT_MODEL_COLLISION": 0,
                "IDENTITY_CLEAR": 0,
            },
            "receipt_sha256": (
                "sha256:e3ba8cc7556075efb857daa648ea56b2b918e81b9823ffee277f49376bb93847"
            ),
            "model_calls": 0,
            "session_mutations": 0,
            "verifier_calls": 0,
            "scoring_calls": 0,
            "api_mutations": 0,
        },
        "reviewed_theseus": {
            "commit": "02be7e987f04e76bcaf970e954ea8ef17fb84423",
            "source_sha256": {
                "orchestrator/public_api/sessions.py": (
                    "sha256:7c0182e859c3adb2c46cb7803328fcd8a02c6f865da211a8ddcfeb0ee79c14b8"
                ),
                "orchestrator/core/db.py": (
                    "sha256:80f5cceccb73cddc7f1ba5ad73b58068e8e1a8b5cb65f260b58912ec11bccaa6"
                ),
                "orchestrator/checkpoint_models.py": (
                    "sha256:d188cff2b057acf55446d3669b868aaa6f0b3fcd802f82834447b48c18911196"
                ),
                "orchestrator/temporal/workflows/session.py": (
                    "sha256:789df0ba71d26552474ea84e2f53ac34451debe52a9eb6b757a3e8ad43433a8f"
                ),
            },
            "source_route_matrix": {
                "scoped_session_list": {
                    "exact_task_version": False,
                    "exact_model_identity": False,
                    "contains_verifier_score_summary": True,
                },
                "reference_traces": {
                    "exact_task_version": True,
                    "exact_model_identity": False,
                    "contains_protected_trace_fields": True,
                },
                "job_detail": {
                    "exact_task_version": False,
                    "exact_model_identity": False,
                    "contains_job_input": False,
                },
                "session_by_job": {
                    "exact_task_version": False,
                    "exact_model_identity": False,
                    "contains_protected_session_fields": True,
                },
            },
            "trace_ingest_identity": {
                "accepts_exact_task_version": True,
                "accepts_provider_model": True,
                "stores_model_base_foreign_key": True,
                "drops_model_on_foreign_key_failure": True,
                "preserves_dropped_model_elsewhere": False,
            },
        },
        "related_draft": {
            "pull_request": 30112,
            "head": "44cd3b2e28544c7a089e02b095ff1eb59c5c590e",
            "adds_exact_task_version": True,
            "adds_write_once_exact_model_identity": True,
            "includes_archived_sessions": True,
            "uses_snapshot_keyset_pagination": True,
            "deployed": False,
        },
        "required_projection": {
            "route": "/v1/sessions/identities",
            "team_scoped": True,
            "requires_exactly_one_task_selector": True,
            "fields": [
                "session_id",
                "eval_task_id",
                "eval_task_version_id",
                "task_key",
                "model_identity",
                "model_identity_status",
                "status",
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
                "metadata",
                "workflow_input_json",
                "cell_id",
                "execution_id",
                "run_id",
            ],
            "model_resolution": {
                "source": "sessions.model_identity write-once server provenance",
                "provider_and_model_required": True,
                "missing_or_invalid_is_ambiguous": True,
                "mutable_provider_catalog_join_forbidden": True,
                "caller_metadata_identity_forbidden": True,
                "historical_dropped_trace_ingest_model_remains_ambiguous": True,
            },
            "pagination": {
                "limit_max": 500,
                "includes_archived_sessions": True,
                "stable_order": ["created_at_desc", "session_id_desc"],
                "cursor": "immutable_created_at_session_id_keyset",
                "snapshot_head_returned": True,
                "has_more_probe": "limit_plus_one",
            },
        },
        "deployment_gate": {
            "draft_shared_pr_only": True,
            "merge_or_deploy_authorized": False,
            "deployed_behavioral_probe_required": True,
            "fresh_selector_identity_required": True,
            "rank_walk_forbidden": True,
        },
        "privacy": {
            "live_session_payloads_read": False,
            "scores_read": False,
            "prompts_traces_flags_read": False,
            "credentials_included": False,
        },
    }
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    return value


def validate(value: Mapping[str, Any]) -> None:
    if dict(value) != contract():
        raise ValueError("session identity projection contract drifted")
