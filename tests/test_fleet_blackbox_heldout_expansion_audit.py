"""Contracts for the exact, no-launch Fleet heldout expansion audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from training.task_family_split import canonical_digest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "configs/data/fleet-blackbox-heldout-expansion-audit-20260924-v1.json"


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def _file_sha256(relative: str) -> str:
    return "sha256:" + hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_audit_is_self_sealed_and_binds_immutable_sources() -> None:
    audit = json.loads(AUDIT.read_text())
    unsigned = {key: value for key, value in audit.items() if key != "sha256"}
    assert audit["sha256"] == canonical_digest(unsigned)
    for relative, binding in audit["sources"].items():
        source = _load(relative)
        assert binding["file_sha256"] == _file_sha256(relative)
        if "logical_sha256" in binding:
            assert source["sha256"] == binding["logical_sha256"]
            assert source["sha256"] == canonical_digest(
                {key: value for key, value in source.items() if key != "sha256"}
            )


def test_catalog_partition_and_current_decision_are_exact() -> None:
    audit = json.loads(AUDIT.read_text())
    catalog = _load("configs/data/qwen38-self-sft-blackbox-task-catalog-20260924-v1.json")
    protocol = _load("configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json")
    counts = catalog["counts"]
    partition = audit["catalog_partition"]

    assert partition["exact_task_versions"] == counts["current_blackbox_task_versions"] == 1217
    assert partition["proven_good"]["exact_task_versions"] == 75
    assert partition["known_broken"]["exact_task_versions"] == counts["known_broken"] == 74
    assert partition["unresolved"]["exact_task_versions"] == counts["unanalyzed"] == 1035
    assert partition["qa_needed"]["exact_task_versions"] == counts["qa_needed"] == 33
    assert protocol["classification"]["clean"] == 20
    assert protocol["classification"]["exposed"] == 5
    assert audit["decision"]["additional_heldout_tasks_authorized_now"] == 0
    assert audit["decision"]["conditional_teacher3k_compatible_candidate_pool"] == 16


def test_exact_conditional_pool_is_the_teacher3k_unexposed_singleton_pool() -> None:
    audit = json.loads(AUDIT.read_text())
    census = _load("configs/data/fleet-blackbox-shared-atom-lineage-census-20260924-v1.json")
    actual = audit["conditional_candidates"]
    expected = [
        {
            "application": row["atom_artifact_keys"][0].split("/")[2],
            "atom_artifact_key": row["atom_artifact_keys"][0],
            "component_id": row["component_id"],
            "qa_status": row["qa_status"],
            "task_key": row["task_key"],
            "task_version_id": row["task_version_id"],
        }
        for row in census["candidate_task_versions"]
        if not row["teacher3k_shared_atom_exposed"]
    ]
    expected.sort(key=lambda row: (row["task_key"], row["task_version_id"]))
    assert actual == expected
    assert len(actual) == len({row["component_id"] for row in actual}) == 16
    assert all(len(row["atom_artifact_key"].split("/")) >= 4 for row in actual)
    assert Counter(row["qa_status"] for row in actual) == Counter(
        audit["conditional_candidate_strata"]["qa_status"]
    )
    assert Counter(row["application"] for row in actual) == Counter(
        audit["conditional_candidate_strata"]["application"]
    )


def test_conditional_pool_is_disjoint_from_all_admitted_and_frozen_train_components() -> None:
    audit = json.loads(AUDIT.read_text())
    catalog = _load("configs/data/qwen38-self-sft-blackbox-task-catalog-20260924-v1.json")
    census = _load("configs/data/fleet-blackbox-shared-atom-lineage-census-20260924-v1.json")
    candidate_components = {row["component_id"] for row in audit["conditional_candidates"]}
    admitted_components = {
        component["component_id"]
        for component in census["components"]
        if component["inherited_role"] is not None
    }
    train_components = {
        row["transitive_family"]["component_id"]
        for row in catalog["task_versions"]
        if row["classification"] == "immediately_train_eligible"
    }
    assert len(admitted_components) == 75
    assert len(train_components) == 50
    assert not candidate_components & admitted_components
    assert not candidate_components & train_components


def test_262k_parent_is_a_subset_of_the_bound_teacher3k_lineage() -> None:
    audit = json.loads(AUDIT.read_text())
    teacher = _load("configs/data/qwen38-teacher3k-training-lineage-map-20260924-v1.json")
    parent = _load("configs/qualification/qwen38-teacher3k-262k-v12-parent-plan.json")
    teacher_keys = {row["task_key"] for row in teacher["training_task_keys"]}
    parent_keys = set(parent["datasets"]["train"]["task_keys"])
    assert len(parent_keys) == audit["lineage_checks"]["teacher3k_262k_parent"]["parent_task_keys"]
    assert not parent_keys - teacher_keys
    assert (
        audit["lineage_checks"]["teacher3k_262k_parent"][
            "parent_task_keys_missing_from_teacher3k_map"
        ]
        == 0
    )


def test_no_unqualified_or_unresolved_row_is_promoted() -> None:
    audit = json.loads(AUDIT.read_text())
    assert audit["catalog_partition"]["qa_needed"]["runtime_receipt_qualified"] == 0
    assert audit["safety"] == {
        "external_mutations": 0,
        "launch_authorized": False,
        "split_changed": False,
        "tasks_reserved": False,
    }
    assert len(audit["missing_evidence"]) == 4
