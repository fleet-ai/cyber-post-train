from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from training import task_family_split

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/qualification/fleet-blackbox-heldout16-zero-model-wave-20260924-v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_is_sealed_and_all_sources_are_exact() -> None:
    plan = _read(PLAN_PATH)
    assert plan["schema"] == "fleet_blackbox_heldout16_zero_model_wave_plan_v1"
    assert plan["sha256"] == _digest({key: value for key, value in plan.items() if key != "sha256"})
    for binding in plan["source_bindings"].values():
        path = ROOT / binding["path"]
        assert path.is_file() and not path.is_symlink()
        assert binding["file_sha256"] == _file_digest(path)
        if "logical_sha256" in binding:
            assert binding["logical_sha256"] == _read(path)["sha256"]


def test_roster_is_the_exact_16_candidate_pool() -> None:
    plan = _read(PLAN_PATH)
    audit = _read(ROOT / plan["source_bindings"]["heldout_expansion_audit"]["path"])
    roster = plan["roster"]
    assert roster == audit["conditional_candidates"]
    assert plan["roster_sha256"] == _digest(roster)
    assert len(roster) == 16
    assert len({row["task_version_id"] for row in roster}) == 16
    assert len({row["component_id"] for row in roster}) == 16
    assert Counter(row["qa_status"] for row in roster) == {"agent_failure": 15, "clean": 1}
    assert Counter(row["application"] for row in roster) == {
        "concur": 5,
        "current": 1,
        "fakelook": 1,
        "fanaplan": 2,
        "fcompliance": 2,
        "fira": 1,
        "fubspot": 1,
        "oracle_epm": 3,
    }


def test_roster_matches_live_lineage_and_singleton_components() -> None:
    plan = _read(PLAN_PATH)
    live = _read(ROOT / plan["source_bindings"]["live_lineage"]["path"])
    live_by_version = {row["task_version_id"]: row for row in live["task_versions"]}
    census = _read(ROOT / "configs/data/fleet-blackbox-shared-atom-lineage-census-20260924-v1.json")
    component_by_id = {row["component_id"]: row for row in census["components"]}
    for row in plan["roster"]:
        observed = live_by_version[row["task_version_id"]]
        assert observed["task_key"] == row["task_key"]
        assert observed["qa_status"] == row["qa_status"]
        assert observed["lineage"]["application"] == row["application"]
        assert observed["atom_artifact_keys"] == [row["atom_artifact_key"]]
        component = component_by_id[row["component_id"]]
        assert component["atom_artifact_keys"] == [row["atom_artifact_key"]]
        assert component["inherited_role"] is None
        assert component["source_counts"] == {"qa_candidate_unqualified": 1}
        assert [
            (task["task_key"], task["task_version_id"]) for task in component["task_versions"]
        ] == [(row["task_key"], row["task_version_id"])]


def test_runtime_budget_and_cleanup_contract_fails_closed() -> None:
    plan = _read(PLAN_PATH)
    qualification = plan["qualification_contract"]
    assert qualification["model_calls"] == 0
    assert qualification["training_data_eligible"] is False
    assert qualification["automatic_retry"] is False
    assert qualification["candidate_count"] == 16
    assert set(qualification["required_live_tool_names"]) == {"bash", "submit_report"}
    assert {
        "verifier_completed",
        "finite_authoritative_outcome",
        "environment_cleanup_completed",
    } <= set(qualification["required_checks"])

    execution = plan["wave_execution"]
    assert execution["root_kind"] == "Job"
    assert execution["root_metadata_annotations"] == {"fleet.ai/failure-alerts": "off"}
    assert execution["priority_class_name"] == "c1"
    assert execution["gpu_requests"] == execution["gpu_limits"] == 0
    assert execution["backoff_limit"] == 0
    assert execution["server_preview_count"] == 2
    assert "exact_name_uid_resource_version" in execution["outer_cleanup"]

    budget = plan["budget_contract"]
    assert budget["authorization_cap"] == 500
    assert budget["candidate_cells"] == 16
    assert budget["atomic_wave_reservation_required"] is True
    assert budget["reservation_must_precede_first_environment_create"] is True
    assert budget["pr604_contract"]["pull_request"] == 604
    assert budget["pr604_contract"]["read_only_preflight_is_not_a_reservation"] is True

    gates = plan["launch_gates"]
    assert gates["launch_authorized"] is False
    assert gates["no_external_launch_or_reservation_performed"] is True
    assert plan["no_launch_receipt"]["external_mutations"] == 0


def test_assignment_is_outcome_blind_and_blocked_until_metadata_is_complete() -> None:
    plan = _read(PLAN_PATH)
    assignment = plan["outcome_blind_assignment"]
    assert assignment["assignment_state"] == "blocked_pending_reviewed_environment_metadata"
    assert assignment["freeze_before_first_qualification_mutation"] is True
    assert assignment["no_role_reassignment_after_any_runtime_result"] is True
    assert assignment["algorithm"]["target_component_counts"] == {
        "dev": 6,
        "final": 3,
        "train": 7,
    }
    assert sum(assignment["algorithm"]["target_component_counts"].values()) == 16
    assert "runtime_qualification_status" in assignment["prohibited_assignment_inputs"]
    assert "finite_authoritative_outcome_value" in assignment["prohibited_assignment_inputs"]

    live = _read(ROOT / plan["source_bindings"]["live_lineage"]["path"])
    ids = {row["task_version_id"] for row in plan["roster"]}
    rows = [row for row in live["task_versions"] if row["task_version_id"] in ids]
    assert len(rows) == 16
    with pytest.raises(ValueError, match="reviewed environment metadata is required"):
        task_family_split.build(
            rows,
            inventory_sha256=_digest(
                sorted(
                    (
                        {
                            "lineage": row["lineage"],
                            "task_key": row["task_key"],
                            "task_version_id": row["task_version_id"],
                        }
                        for row in rows
                    ),
                    key=lambda row: (row["task_key"], row["task_version_id"]),
                )
            ),
            seed=assignment["algorithm"]["seed"],
            ratios=assignment["algorithm"]["ratios"],
            dimensions=tuple(assignment["algorithm"]["balanced_dimensions"]),
            max_group_task_version_fraction=assignment["algorithm"][
                "max_group_task_version_fraction"
            ],
        )


def test_training_overlap_gate_cannot_be_weakened() -> None:
    plan = _read(PLAN_PATH)
    gate = plan["training_overlap_gates"]
    assert gate["teacher3k_32k_64k_96k_component_overlap"] == 0
    assert gate["teacher3k_262k_parent_component_overlap"] == 0
    assert gate["prospective_self_sft_admitted75_components_overlap"] == 0
    assert gate["prospective_self_sft_train50_components_overlap"] == 0
    assert gate["unknown_or_unresolved_training_lineage_allowed"] is False
    assert "dev or final" in gate["required_before_any_sft_or_rl_optimizer_update"]
