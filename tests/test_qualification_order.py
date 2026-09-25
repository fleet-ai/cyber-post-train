import copy
import json
from pathlib import Path

import pytest

from training.qualification_order import checked, plan


def row(key, version, atom, app):
    return {"task_key": key, "task_version_id": version, "task_id": key,
            "atom_artifact_keys": [f"cyber/atoms/{app}/{atom}"],
            "applications": [app], "environment_version_id": "exact-env",
            "difficulty": "medium", "source_project": "project", "source_repo": "repo",
            "lifecycle": "production", "verifier_attached": True}


def test_order_is_family_closed_deterministic_and_not_runtime_acceptance():
    rows = [row("a1", "v1", "login-spoof", "a"),
            row("a1", "v2", "login-spoof", "a"),
            row("a2", "v1", "session-spoof", "a"),
            row("b1", "v1", "file-upload", "b"),
            row("b2", "v1", "file-traversal", "b")]
    first = plan(copy.deepcopy(rows), dev_count=2)
    assert checked(first) == first
    assert first == plan(list(reversed(rows)), dev_count=2)
    assert first["counts"] == {"versions": 5, "families": 4,
                                "dev_reserved": 2, "final_reserved": 2,
                                "final_qualified_floor": 100}
    assert any(len(f["versions"]) == 2 for f in first["ordered_families"])
    assert {f["strata"]["app"] for f in first["ordered_families"]
            if f["reserved_role"] == "final_test"} == {"a", "b"}
    assert first["quality_gate"] == {"runtime": "unproven", "positive_control": "unproven",
                                     "model_outcomes_used": False, "launch_authorized": False}


def test_model_outcomes_and_incomplete_bindings_are_rejected():
    good = row("a", "v", "sql-injection", "a")
    with pytest.raises(ValueError, match="unsafe"):
        plan([good | {"score": 1}], dev_count=1)
    with pytest.raises(ValueError, match="unsafe"):
        plan([good | {"environment_version_id": None}], dev_count=1)


def test_frozen_roster_replays_exactly():
    path = Path(__file__).parents[1] / "configs/data/fleet-blackbox-qualification-order-20260925-v1.json"
    frozen = checked(json.loads(path.read_text()))
    repeated = plan(frozen["candidate_versions"])
    assert frozen["ordered_families"] == repeated["ordered_families"]
    assert frozen["role_strata"] == repeated["role_strata"]
