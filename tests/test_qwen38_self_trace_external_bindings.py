"""Fail-closed checks for the Qwen3.8 self-trace external bindings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
AUDIT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-12-self-trace-external-binding-audit-v1.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bound(reference: dict) -> tuple[Path, dict]:
    path = ROOT / reference["path"]
    value = read(path)
    assert file_sha256(path) == reference["file_sha256"]
    if "document_sha256" in reference:
        assert value["sha256"] == reference["document_sha256"]
    return path, value


def test_external_binding_audit_is_sealed_and_keeps_v1_fail_closed() -> None:
    audit = read(AUDIT)
    assert audit["sha256"] == digest_json(
        {key: value for key, value in audit.items() if key != "sha256"}
    )
    assert audit["status"] == "blocked_no_authoritative_direct_runtime_binding"
    _, plan = bound(audit["frozen_gate"]["study_plan"])
    _, request = bound(audit["frozen_gate"]["collection_request"])
    assert audit["frozen_gate"]["immutable_v1_rewritten"] is False
    assert request["runtime"]["collector_image"] is None
    assert request["runtime"]["base_route_certificate_sha256"] is None
    assert request["interface"]["system_prompt_sha256"] is None
    assert plan["unresolved"] == {
        "direct_base_route_certificate_sha256": None,
        "direct_collector_image_digest": None,
        "direct_system_prompt_sha256": None,
    }
    assert audit["decision"]["bindings_resolved"] == []
    assert audit["decision"]["collection_or_training_launchable"] is False


def test_local_oci_and_green_pr_are_not_a_pullable_collector_receipt() -> None:
    audit = read(AUDIT)
    collector = audit["collector_image"]
    _, publication = bound(collector["publication_plan"])
    assert collector["status"] == "blocked_no_pullable_immutable_reference"
    assert collector["candidate_is_pullable_registry_identity"] is False
    assert publication["local_qualification"]["pullable_registry_reference_proven"] is False
    assert publication["dev_builder"]["rewrite_timestamp_request_field_present"] is False
    assert publication["dev_builder"]["proposed_repository_builder_iam_authorized"] is False
    assert collector["theseus_change"]["state"] == "open"
    assert collector["theseus_change"]["nonpassing_required_or_reported_checks"] == 0
    assert collector["theseus_change"]["merge_or_deployment_proven"] is False
    assert collector["live_dev_image_api"]["rewrite_timestamp_present"] is False
    assert collector["gpu_or_registry_mutation_performed"] is False


def test_route_component_cannot_be_relabelled_as_direct_certificate() -> None:
    audit = read(AUDIT)
    route = audit["base_route"]
    _, component = bound(route["component"])
    stable = component["registration"]["stable_route"]
    model = component["model_artifact"]
    assert route["status"] == "component_qualified_certificate_absent"
    assert component["status"] == "passed_component_only"
    assert component["launchable"] is False
    assert model["repository"] == stable["model"]["repository"] == "Qwen/Qwen3.8-27B"
    assert (
        model["revision"]
        == stable["model"]["revision"]
        == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    assert model["payload_rehashed"] is True
    assert model["symlinks_absent"] is True
    assert "tool_calling" in component["gateway_projections"]["catalog"]["capabilities"]
    assert component["scope"]["prompt_requests"] == 0
    assert component["scope"]["completion_requests"] == 0


def test_known_prompt_bytes_are_candidate_provenance_not_qwen38_authority() -> None:
    audit = read(AUDIT)
    prompt = audit["direct_system_prompt"]
    candidate = prompt["candidate"]
    path = ROOT / candidate["path"]
    data = path.read_bytes()
    git_blob = hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()
    assert prompt["status"] == "candidate_present_scientific_choice_unmade"
    assert file_sha256(path) == candidate["file_sha256"]
    assert len(data) == candidate["utf8_bytes"]
    assert len(data.decode("utf-8")) == candidate["character_length"]
    assert git_blob == candidate["git_blob"]
    _, prior = bound(prompt["provenance"]["paired_canary_plan"])
    assert prior["model"]["repo"] == "Qwen/Qwen3.6-27B"
    assert prior["conversation_contract"]["system_prompt_path"] == candidate["path"]
    assert prior["conversation_contract"]["system_prompt_sha256"] == candidate[
        "file_sha256"
    ]
    assert prompt["provenance"]["qwen38_direct_treatment_authority"] is False


def test_audit_scope_contains_no_live_or_paid_action() -> None:
    audit = read(AUDIT)
    assert audit["scope"] == {
        "read_only_cluster_or_api_observations": True,
        "prompt_artifact_content_emitted": False,
        "prompt_or_completion_requests": 0,
        "task_or_grading_requests": 0,
        "scoring_requests": 0,
        "evaluation_submissions": 0,
        "gpu_requests": 0,
        "registry_publications": 0,
        "cluster_or_api_mutations": 0,
    }
    assert audit["decision"]["next_executable_action"] == (
        "none_until_external_image_infrastructure_and_scientific_prompt_choice_are_resolved"
    )
