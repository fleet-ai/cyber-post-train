import json
from pathlib import Path

from training import fleet_blackbox_atom_lineage_expansion as expansion
from training import task_family_split

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_checked_in_artifacts_reproduce_exactly() -> None:
    expansion.check(ROOT)
    assert expansion.build(ROOT) == tuple(
        load(path) for path in (expansion.LINEAGE_CENSUS, expansion.SPLIT, expansion.DECISION)
    )


def test_all_33_candidates_are_exactly_bound_but_none_are_runtime_qualified() -> None:
    census, split, decision = expansion.build(ROOT)
    assert (
        census["qualification_authority_git_commit"] == "915b385196842674114a4e9891d4e2cd02613d57"
    )
    assert census["counts"] == {
        "exact_task_versions": 108,
        "receipt_proven_task_versions": 75,
        "qa_candidate_task_versions": 33,
        "atom_artifact_keys": 104,
        "components": 101,
        "candidate_components": 26,
        "components_by_task_version_count": {"1": 99, "3": 1, "6": 1},
        "candidate_components_overlapping_receipt_proven": 0,
    }
    assert split["candidate_decision"] == {
        "exact_lineage_bound": 33,
        "new_runtime_receipt_qualified": 0,
        "excluded_missing_complete_runtime_receipts": 33,
        "shared_atom_components": 26,
    }
    assert decision["counts"] == {
        "candidate_task_versions": 33,
        "exact_live_lineage_bound": 33,
        "runtime_receipt_qualified": 0,
        "teacher3k_exact_version_exposed": 12,
        "teacher3k_shared_atom_exposed": 17,
        "teacher3k_shared_atom_unexposed": 16,
    }


def test_transitive_split_preserves_every_parent_role_without_candidate_admission() -> None:
    _, split, _ = expansion.build(ROOT)
    parent = load(expansion.PARENT_SPLIT)
    expected = {(row["task_key"], row["task_version_id"]): row["split"] for row in parent["tasks"]}
    observed = {(row["task_key"], row["task_version_id"]): row["split"] for row in split["tasks"]}
    assert observed == expected
    assert split["counts"] == {
        "dev": {"components": 17, "task_versions": 17},
        "final_test": {"components": 8, "task_versions": 8},
        "train": {"components": 50, "task_versions": 50},
    }
    assert split["leakage_checks"] == {
        "admitted_component_role_conflicts": 0,
        "candidate_components_overlapping_admitted_components": 0,
        "all_parent_roles_preserved": True,
        "candidate_task_versions_admitted": 0,
    }


def test_artifacts_are_self_digesting_and_no_launch() -> None:
    for value in expansion.build(ROOT):
        assert value["sha256"] == task_family_split.canonical_digest(
            {key: item for key, item in value.items() if key != "sha256"}
        )
        assert value["safety"]["launch_authorized"] is False
        assert value["safety"]["external_mutations"] == 0
