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
    assert expansion.build_ceiling_receipt(ROOT) == load(expansion.CEILINGS)


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
    for value in (*expansion.build(ROOT), expansion.build_ceiling_receipt(ROOT)):
        assert value["sha256"] == task_family_split.canonical_digest(
            {key: item for key, item in value.items() if key != "sha256"}
        )
        assert value["safety"]["launch_authorized"] is False
        assert value["safety"]["external_mutations"] == 0


def test_component_ceiling_corrects_the_distinct_family_assumption() -> None:
    receipt = expansion.build_ceiling_receipt(ROOT)
    assert receipt["catalog_partition"]["mutually_exclusive_counts"] == {
        "receipt_proven_admitted": 75,
        "known_broken_excluded": 74,
        "qa_clean_missing_complete_runtime_receipt": 8,
        "qa_agent_failure_missing_complete_runtime_receipt": 25,
        "not_analyzed_missing_exact_receipt_and_lineage": 1035,
    }
    assert receipt["transitive_candidate_correction"] == {
        "exact_task_versions": 33,
        "stable_atom_components": 26,
        "component_size_histogram": {"1": 24, "3": 1, "6": 1},
        "components_overlapping_admitted": 0,
        "runtime_receipt_qualified_task_versions": 0,
        "teacher3k": {
            "exact_version_exposed": 12,
            "shared_atom_exposed_task_versions": 17,
            "shared_atom_exposed_components": 10,
            "shared_atom_unexposed_task_versions": 16,
            "shared_atom_unexposed_components": 16,
            "all_unexposed_components_are_singletons": True,
            "all_multi_version_components_are_exposed": True,
        },
    }
    conditional = receipt["conditional_after_complete_zero_model_qualification"]
    assert conditional["teacher3k_leakage_only_maximum"]["heldout_components"] == 36
    assert conditional["general_lineage_only_all_candidates_heldout_maximum"] == {
        "current_heldout_components": 25,
        "candidate_component_additions": 26,
        "heldout_components": 51,
        "heldout_exact_task_versions_if_every_alias_is_retained": 58,
        "independent_roster_rule": (
            "predeclare one exact qualified representative per component; keep every other "
            "exact version in the same role and do not count it as an independent family"
        ),
        "representative_distribution_preserved": False,
    }
    assert conditional["representative_anchored_plan"] == {
        "component_roles": {"dev": 23, "final_test": 11, "train": 67},
        "new_component_roles": {"dev": 6, "final_test": 3, "train": 17},
        "general_heldout_components": 34,
        "teacher3k_shared_atom_exposed_candidate_components_forced_to_train": 10,
        "teacher3k_unexposed_candidate_components": {
            "train": 7,
            "dev": 6,
            "final_test": 3,
        },
        "teacher3k_compatible_heldout_components": 29,
        "allocation_rule": (
            "preserve all 75 inherited roles; assign only qualified candidate components; "
            "ignore capability outcomes; balance application, environment, difficulty, and "
            "vulnerability metadata with a frozen seed"
        ),
    }
    assert receipt["correction"]["corrected_proportional_heldout_components"] == 34
    assert receipt["qualified_now"]["additional_task_versions_or_components"] == 0


def test_component_ceiling_receipt_contains_no_private_task_identity() -> None:
    receipt = expansion.build_ceiling_receipt(ROOT)
    encoded = json.dumps(receipt, sort_keys=True)
    assert '"task_key"' not in encoded
    assert '"task_version_id"' not in encoded
    assert '"component_id"' not in encoded
    assert receipt["privacy"] == {
        "task_prompts_or_trajectories_read": False,
        "task_keys_or_version_ids_repeated_in_this_receipt": False,
        "private_content_persisted": False,
    }
