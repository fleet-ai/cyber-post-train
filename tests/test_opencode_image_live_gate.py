"""Immutable, fail-closed checks for the live OpenCode image-gate audit."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
AUDIT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-12-opencode-agent-image-live-gate-v1.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_live_image_gate_audit_is_self_digested_and_binds_predecessors() -> None:
    audit = read(AUDIT)
    assert audit["sha256"] == digest_json(
        {key: value for key, value in audit.items() if key != "sha256"}
    )
    for reference in audit["predecessors"].values():
        artifact = ROOT / reference["path"]
        document = read(artifact)
        assert file_sha256(artifact) == reference["file_sha256"]
        assert document["sha256"] == reference["document_sha256"]


def test_source_and_live_contract_cannot_be_mistaken_for_deployment() -> None:
    audit = read(AUDIT)
    pull_request = audit["upstream_change"]["pull_request"]
    openapi = audit["live_dev"]["image_build_openapi"]
    api = audit["live_dev"]["image_build_api"]
    assert pull_request["state"] == "OPEN"
    assert pull_request["merge_or_deployment_proven"] is False
    assert pull_request["checks"]["pending_or_failed"] == 0
    assert openapi["local_build_path_present"] is True
    assert openapi["rewrite_timestamp_present"] is False
    assert openapi["repository_validator_result"] == (
        "rejected_missing_rewrite_timestamp"
    )
    assert api["pod_ready"] is True
    assert api["pod_restarts"] == 0
    assert api["deployed_source_matches_unmodified_main_contract"] is True
    assert api["deployed_source_contains_dedicated_repository_terraform"] is False


def test_green_source_does_not_open_registry_or_image_acceptance_gates() -> None:
    audit = read(AUDIT)
    registry = audit["live_dev"]["registry_and_iam_readback"]
    decision = audit["decision"]
    assert registry["dedicated_repository_exists"] is None
    assert registry["builder_role_has_dedicated_push_scope"] is None
    assert registry["pr_source_or_kubernetes_annotation_used_as_live_policy_proof"] is False
    assert registry["user_pasted_credentials_used"] is False
    assert decision["collector_or_base_eval_image_accepted"] is False
    assert decision["self_trace_collection_launchable"] is False
    assert decision["base_evaluation_launchable"] is False
    assert decision["image_acceptance_resolves_all_self_trace_bindings"] is False


def test_audit_performed_no_mutation_or_paid_work() -> None:
    audit = read(AUDIT)
    scope = audit["scope"]
    assert scope["read_only_github_observations"] is True
    assert scope["read_only_cluster_observations"] is True
    for key in (
        "prompt_or_completion_requests",
        "task_or_grading_requests",
        "scoring_requests",
        "context_uploads",
        "image_build_jobs",
        "registry_publications",
        "dev_qualification_pods",
        "gpu_requests",
        "cluster_or_api_mutations",
        "github_mutations",
    ):
        assert scope[key] == 0
