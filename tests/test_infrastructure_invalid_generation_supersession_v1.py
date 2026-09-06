import copy
import json
from pathlib import Path

import pytest

from evals.fleet import infrastructure_invalid_generation_supersession_v1 as gate
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]
EVIDENCE_DIR = Path("docs/evidence/qwen38-study")
MANIFEST = ROOT / EVIDENCE_DIR / (
    "2026-09-06-infrastructure-invalid-generation-supersession-manifest-v1.json"
)


def test_all_four_held_receipts_are_model_blind_and_non_launching():
    manifest = json.loads(MANIFEST.read_text())
    gate.validate_manifest(manifest, ROOT)
    assert manifest["counts"] == {
        "held_cells": 4,
        "no_authoritative_outcome": 2,
        "bound_endpoint_identity_discontinuity": 2,
        "launch_authorized": 0,
    }
    assert len({row["cell_id"] for row in manifest["cells"]}) == 4
    assert len({row["successor_execution_id"] for row in manifest["cells"]}) == 4


def test_successors_preserve_exact_task_attempt_cell_and_increment_generation():
    for value in gate.build_held_receipts(ROOT, observed_at_utc="frozen"):
        assert value["successor_execution"]["execution_generation"] == (
            value["prior_execution"]["execution_generation"] + 1
        )
        assert value["successor_execution"][
            "same_model_task_version_attempt_and_cell_required"
        ] is True
        assert value["statistical_cell"]["maximum_accepted_valid_generations"] == 1
        assert value["authorization"]["launch_authorized"] is False


def test_score_or_trace_dependent_classification_fails_closed():
    value = gate.build_held_receipts(ROOT, observed_at_utc="frozen")[0]
    for field in (
        "classification_uses_scores",
        "classification_uses_prompts_traces_or_flags",
        "classification_uses_model_output",
    ):
        changed = copy.deepcopy(value)
        changed["scientific_basis"][field] = True
        changed["receipt_sha256"] = self_hosted.digest_without(
            changed, "receipt_sha256"
        )
        with pytest.raises(gate.SupersessionError):
            gate.validate_held_receipt(changed, ROOT)


@pytest.mark.parametrize("model", ["model-a", "model-b", "closed-or-open-is-irrelevant"])
def test_classifier_is_model_blind(model):
    no_outcome = {
        "model": model,
        "session_state": {"exact_matches": 0},
        "acceptance_state": "absent",
        "verifier_state": "absent",
        "cleanup_state": "complete",
    }
    assert (
        gate.classify_infrastructure_invalid(
            no_outcome, proof={"authoritative_outcome_absence_proven": True}
        )
        == "no_authoritative_outcome"
    )

    discontinuity = {
        "model": model,
        "session_state": {"exact_matches": 1, "session_ingest_present": True},
        "acceptance_state": "local_receipt_present_but_not_credited",
        "verifier_state": "present_and_reconciled",
        "cleanup_state": "complete",
    }
    assert (
        gate.classify_infrastructure_invalid(
            discontinuity,
            proof={"bound_endpoint_identity_discontinuity_proven": True},
        )
        == "bound_endpoint_identity_discontinuity"
    )


def test_classifier_rejects_ordinary_process_failure_without_authority_proof():
    with pytest.raises(gate.SupersessionError):
        gate.classify_infrastructure_invalid(
            {
                "session_state": {},
                "acceptance_state": "absent",
                "verifier_state": "absent",
                "cleanup_state": "unknown",
                "process": {"exit_code": 1},
            },
            proof={},
        )


def test_any_launch_or_claim_authority_fails_closed():
    value = gate.build_held_receipts(ROOT, observed_at_utc="frozen")[0]
    for field in value["authorization"]:
        changed = copy.deepcopy(value)
        changed["authorization"][field] = True
        changed["receipt_sha256"] = self_hosted.digest_without(
            changed, "receipt_sha256"
        )
        with pytest.raises(gate.SupersessionError):
            gate.validate_held_receipt(changed, ROOT)


def test_cell_task_or_successor_identity_drift_fails_closed():
    value = gate.build_held_receipts(ROOT, observed_at_utc="frozen")[0]
    mutations = (
        ("task", "version_id", "different"),
        ("statistical_cell", "cell_id", "sha256:" + "0" * 64),
        ("successor_execution", "execution_id", "sha256:" + "1" * 64),
        ("successor_execution", "execution_generation", 999),
    )
    for section, field, replacement in mutations:
        changed = copy.deepcopy(value)
        changed[section][field] = replacement
        changed["receipt_sha256"] = self_hosted.digest_without(
            changed, "receipt_sha256"
        )
        with pytest.raises(gate.SupersessionError):
            gate.validate_held_receipt(changed, ROOT)


def test_manifest_rejects_more_than_one_valid_generation_precedence():
    value = json.loads(MANIFEST.read_text())
    changed = copy.deepcopy(value)
    changed["precedence"]["maximum_accepted_valid_generations_per_statistical_cell"] = 2
    changed["receipt_sha256"] = self_hosted.digest_without(
        changed, "receipt_sha256"
    )
    with pytest.raises(gate.SupersessionError):
        gate.validate_manifest(changed, ROOT)


def test_launch_time_clearance_requires_every_score_blind_collision_gate():
    held = gate.build_held_receipts(ROOT, observed_at_utc="frozen")[0]
    observation = {
        "cell_id": held["statistical_cell"]["cell_id"],
        "successor_execution_id": held["successor_execution"]["execution_id"],
        "prior_claim_receipt_sha256": held["prior_execution"]["claim"][
            "receipt_sha256"
        ],
        "prior_matching_authoritative_sessions": held["prior_execution"][
            "matching_authoritative_sessions"
        ],
        "successor_matching_authoritative_sessions": 0,
        "valid_accepted_generation_count": 0,
        "active_valid_generation_count": 0,
        "successor_claim_count": 0,
        "successor_output_root_count": 0,
        "api_mutations": 0,
        "scores_or_protected_content_read": False,
    }
    gate.validate_launch_time_clearance(held, observation)
    for field in observation:
        changed = copy.deepcopy(observation)
        changed[field] = None
        with pytest.raises(gate.SupersessionError):
            gate.validate_launch_time_clearance(held, changed)


def test_precedence_preserves_old_generation_and_allows_only_one_successor():
    held = gate.build_held_receipts(ROOT, observed_at_utc="frozen")[0]
    prior = {
        "execution_id": held["prior_execution"]["execution_id"],
        "execution_generation": held["prior_execution"]["execution_generation"],
        "state": "infrastructure_invalid_preserved",
        "valid_accepted": False,
    }
    successor = {
        "execution_id": held["successor_execution"]["execution_id"],
        "execution_generation": held["successor_execution"]["execution_generation"],
        "state": "accepted_valid",
        "valid_accepted": True,
    }
    gate.validate_generation_precedence(held, [prior])
    gate.validate_generation_precedence(held, [prior, successor])

    duplicate = copy.deepcopy(successor)
    duplicate["execution_id"] = "sha256:" + "f" * 64
    with pytest.raises(gate.SupersessionError):
        gate.validate_generation_precedence(held, [prior, successor, duplicate])
    invalid_rewrite = copy.deepcopy(prior)
    invalid_rewrite["valid_accepted"] = True
    with pytest.raises(gate.SupersessionError):
        gate.validate_generation_precedence(held, [invalid_rewrite])
