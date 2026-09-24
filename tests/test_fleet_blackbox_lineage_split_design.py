import json
from pathlib import Path

import pytest

from training import fleet_blackbox_lineage_split_design as design
from training import task_family_split

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def test_checked_in_artifacts_reproduce_exactly() -> None:
    design.check(ROOT)
    anchor, split, evidence = design.build(ROOT)
    assert anchor == _load(design.ROLE_ANCHOR)
    assert split == _load(design.SPLIT)
    assert evidence == _load(design.DESIGN)


def test_latest_qualification_cannot_support_heldout_100() -> None:
    _, split, evidence = design.build(ROOT)
    decision = evidence["population_decision"]
    assert decision == {
        "current_inventory_task_versions": 1217,
        "exact_receipt_proven_task_versions": 75,
        "known_broken_task_versions": 74,
        "qa_cleared_but_unproven_task_versions": 33,
        "not_analyzed_without_exact_receipt": 1035,
        "heldout_100_supported_now": False,
        "reason": decision["reason"],
    }
    assert split["counts"] == {
        "dev": {"groups": 17, "task_versions": 17},
        "final_test": {"groups": 8, "task_versions": 8},
        "train": {"groups": 50, "task_versions": 50},
    }
    assert evidence["expansion_gate"]["minimum_additional_distinct_qualified_families"] == 225
    assert evidence["expansion_gate"]["qa_candidate_upper_bound_if_all_distinct_and_qualified"] == {
        "total_families": 108,
        "heldout_families": 36,
        "shortfall_to_100": 64,
    }


def test_existing_roles_are_immutable_and_family_disjoint() -> None:
    anchor, split, _ = design.build(ROOT)
    assert anchor["sha256"] == (
        "sha256:48350b8fc23143abe297db3b2364590c72d564553ebb4e9a349fa06ec246a26b"
    )
    assert split["sha256"] == (
        "sha256:b613abaae0e5b52ef279a46a4ab0145dba4517442d1e4198fd52af7de0161d10"
    )
    assert split["anchor_audit"] == {
        "inherited_role_count": 75,
        "present_inherited_group_count": 75,
        "absent_inherited_group_count": 0,
        "new_group_count": 0,
        "inherited_heldout_group_count": 25,
        "historical_task_key_group_drift": 0,
        "all_current_groups_immutable": True,
    }
    roles: dict[str, set[str]] = {}
    for row in split["tasks"]:
        roles.setdefault(row["group_id"], set()).add(row["split"])
    assert len(roles) == 75
    assert all(len(value) == 1 for value in roles.values())
    assert split["leakage_checks"]["reviewed_family_overlap"] == 0


def test_future_expansion_is_gated_on_shared_atom_closure() -> None:
    _, _, evidence = design.build(ROOT)
    scope = evidence["balance"]["lineage_safety_scope"]
    assert scope["shared_atom_transitive_closure_implemented"] is False
    assert "blocked" in scope["extension_status"]
    requirements = evidence["expansion_gate"]["required_before_extension"]
    assert any(
        "transitive grouping by shared reviewed atom identity" in item for item in requirements
    )


def test_unproven_qa_candidates_never_enter_the_split() -> None:
    _, split, _ = design.build(ROOT)
    selected = {(row["task_key"], row["task_version_id"]) for row in split["tasks"]}
    candidates = _load(design.CANDIDATES)["tasks"]
    candidate_ids = {(row["task_key"], row["task_version_id"]) for row in candidates}
    assert len(candidates) == 33
    assert selected.isdisjoint(candidate_ids)


def test_checkpoint_specific_lineage_is_not_called_global_holdout() -> None:
    _, _, evidence = design.build(ROOT)
    lineage = evidence["known_training_lineage"]
    assert lineage["checkpoint_specific_results"]["teacher3k"] == {
        "protocol_path": design.TEACHER3K_PROTOCOL,
        "protocol_sha256": (
            "sha256:2ac58c3ded7ba5c9bacd27dc3a379c726912a3f0cafc3ee542672de730e1fe0e"
        ),
        "clean_heldout_families": 20,
        "exposed_heldout_families": 5,
        "exposed_training_aliases": 6,
        "status": "20_checkpoint_specific_lineage_clean",
    }
    assert lineage["checkpoint_specific_results"]["sep7_original_teacher"] == {
        "candidate_key_overlap_with_current_heldout": 25,
        "actual_trained_family_identities": "not_retained",
        "status": "not_certified_for_current_heldout",
    }
    assert lineage["global_historical_boundary"]["certified_globally_untouched_families"] == 0
    assert set(lineage["global_historical_boundary"]["unresolved_exact_lineage_attempts"]) == set(
        design.UNRESOLVED_EXACT_LINEAGE
    )


def test_all_known_attempt_ids_are_bound() -> None:
    _, _, evidence = design.build(ROOT)
    source = _load(design.EVIDENCE_INDEX)
    inventory = evidence["known_training_lineage"]["inventory"]
    assert inventory["sft_attempt_ids"] == sorted(row["id"] for row in source["sft_attempts"])
    assert inventory["rl_attempt_ids"] == sorted(row["id"] for row in source["rl_attempts"])


def test_artifacts_are_sealed_prompt_free_and_no_launch() -> None:
    values = design.build(ROOT)
    for value in values:
        assert value["sha256"] == task_family_split.canonical_digest(
            {key: item for key, item in value.items() if key != "sha256"}
        )
    evidence = values[-1]
    assert evidence["safety"] == {
        "launch_authorized": False,
        "external_mutations": 0,
        "private_content_persisted": False,
        "task_prompts_or_trajectories_read": False,
    }
    forbidden = {"prompt", "trajectory", "transcript", "score", "answer", "credential", "flag"}
    for key in _walk_keys(evidence):
        assert key not in forbidden


def _walk_keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_check_rejects_artifact_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    anchor, split, evidence = design.build(ROOT)
    drifted = dict(evidence)
    drifted["status"] = "launchable"
    (tmp_path / "anchor.json").write_text(json.dumps(anchor, indent=2, sort_keys=True) + "\n")
    (tmp_path / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    (tmp_path / "design.json").write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(design, "ROLE_ANCHOR", "anchor.json")
    monkeypatch.setattr(design, "SPLIT", "split.json")
    monkeypatch.setattr(design, "DESIGN", "design.json")
    monkeypatch.setattr(design, "build", lambda _root: (anchor, split, evidence))
    with pytest.raises(ValueError, match="reproduction drift"):
        design.check(tmp_path)


def test_writer_is_create_once_and_idempotent(tmp_path: Path) -> None:
    value = {"schema": "test", "sha256": "sha256:" + "0" * 64}
    path = tmp_path / "artifact.json"
    design._write_once_or_equal(path, value)
    design._write_once_or_equal(path, value)
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="refusing to replace"):
        design._write_once_or_equal(path, value)
