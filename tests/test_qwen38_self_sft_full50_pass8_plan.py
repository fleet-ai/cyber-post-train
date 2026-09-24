"""Contracts for the no-launch 50-component pass@8 successor plan."""

import copy
import json
from pathlib import Path

import pytest

from training import qwen38_self_sft_full50_pass8_plan as plan

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / plan.OUTPUT).read_text())


def test_checked_in_plan_rebuilds_from_exact_reviewed_sources() -> None:
    value = _load()
    assert plan.build(ROOT) == value
    plan.check(ROOT)
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    assert value["sha256"] == plan._digest(unsigned)  # noqa: SLF001


def test_exact_component_roster_is_train_only_and_heldout_safe() -> None:
    value = _load()
    roster = value["train_roster"]["component_task_version_roster"]
    identities = {(row["task_key"], row["task_version_id"]) for row in roster}
    components = {row["shared_atom_component_id"] for row in roster}
    catalog = json.loads((ROOT / plan.CATALOG).read_text())
    train = {
        (row["task_key"], row["task_version_id"])
        for row in catalog["task_versions"]
        if row["classification"] == "immediately_train_eligible"
    }
    heldout = {
        row["transitive_family"]["component_id"]
        for row in catalog["task_versions"]
        if row["classification"] == "heldout_family_overlap"
    }
    assert len(roster) == len(identities) == len(components) == 50
    assert identities == train
    assert not (components & heldout)
    assert value["train_roster"]["component_task_version_roster_sha256"] == plan._digest(  # noqa: SLF001
        roster
    )


def test_pilot_and_expansion_are_disjoint_and_cover_exactly_400_fresh_cells() -> None:
    value = _load()
    roster = value["train_roster"]["component_task_version_roster"]
    seeds = value["cell_universe"]["seeds"]
    pilot = value["stages"]["operational_canary"]["task_versions"]
    pilot_ids = {(row["task_key"], row["task_version_id"]) for row in pilot}
    all_cells = [{**row, "seed": seed} for row in roster for seed in seeds]
    pilot_cells = [{**row, "seed": seed} for row in pilot for seed in seeds]
    expansion_cells = [
        cell for cell in all_cells if (cell["task_key"], cell["task_version_id"]) not in pilot_ids
    ]
    assert seeds == list(range(51, 59))
    assert len(pilot_ids) == 5
    assert len(all_cells) == 400
    assert len(pilot_cells) == 40
    assert len(expansion_cells) == 360
    identity = value["source_identity_sha256"]
    assert identity == plan._digest(value["source_identity"])  # noqa: SLF001
    assert value["cell_universe"]["sha256"] == plan._digest(  # noqa: SLF001
        {"source_identity_sha256": identity, "cells": all_cells}
    )
    assert value["stages"]["operational_canary"]["cell_universe_sha256"] == plan._digest(  # noqa: SLF001
        {"source_identity_sha256": identity, "cells": pilot_cells}
    )
    assert value["stages"]["expansion"]["cell_universe_sha256"] == plan._digest(  # noqa: SLF001
        {"source_identity_sha256": identity, "cells": expansion_cells}
    )
    assert value["cell_universe"]["sha256"] != value["operation"]["retired_cell_universe_sha256"]


def test_pilot_is_deterministic_and_spans_five_applications() -> None:
    value = _load()
    metadata = json.loads((ROOT / plan.METADATA).read_text())
    by_identity = {
        (row["task_key"], row["task_version_id"]): row["lineage"]
        for row in metadata["task_versions"]
    }
    pilot = value["stages"]["operational_canary"]["task_versions"]
    applications = {
        by_identity[(row["task_key"], row["task_version_id"])]["application"] for row in pilot
    }
    assert len(applications) == 5


def test_reservation_and_create_remain_fail_closed() -> None:
    value = _load()
    reservation = value["daily_reservation"]
    assert value["mode"] == "prepared_no_launch_no_reservation"
    assert value["launch_authorized"] is False
    assert value["external_mutations"] == 0
    assert reservation["target_utc_day"] == "2026-09-25"
    assert reservation["count"] == 400
    assert reservation["canonical_cap"] == 500
    assert reservation["reservation_receipt_sha256"] is None
    assert reservation["packet_set_sha256"] is None
    assert reservation["groups"] == [
        {
            "name": "operational_canary",
            "components": 5,
            "count": 40,
            "cell_universe_sha256": value["stages"]["operational_canary"]["cell_universe_sha256"],
        },
        {
            "name": "expansion",
            "components": 45,
            "count": 360,
            "cell_universe_sha256": value["stages"]["expansion"]["cell_universe_sha256"],
        },
    ]
    assert reservation["status"] == "missing_not_reserved"
    assert not any(value["fresh_precreate_gates"].values())
    assert not any(value["implementation_gates"].values())
    assert value["stages"]["qa_one_cell"]["status"] == "missing"
    assert value["stages"]["operational_canary"]["status"] == "missing"
    assert value["stages"]["expansion"]["status"] == "blocked"


def test_cluster_root_contract_is_c1_alert_off_zero_gpu_create_once() -> None:
    contract = _load()["cluster_root_contract"]
    assert contract == {
        "ambiguous_create_replay_allowed": False,
        "backoff_limit": 0,
        "create_attempts_maximum": 1,
        "exact_uid_bound_terminal_reconciliation_and_cleanup_required": True,
        "kind": "Job",
        "priority_class_name": "c1",
        "required_top_level_annotation": {"fleet.ai/failure-alerts": "off"},
        "restart_policy": "Never",
        "worker_gpu_requests_and_limits": 0,
    }


def test_plan_contains_no_private_session_or_task_content() -> None:
    value = _load()
    text = json.dumps(value, sort_keys=True).lower()
    forbidden = (
        '"prompt"',
        '"trace"',
        '"flag"',
        '"answer"',
        '"credential"',
        '"session_id"',
        '"numeric_score"',
    )
    assert all(item not in text for item in forbidden)
    assert value["privacy"] == {
        "credentials_persisted": False,
        "numeric_scores_persisted": False,
        "prompts_or_trajectories_persisted": False,
    }


def test_source_digest_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    changed = copy.deepcopy(plan.SOURCES)
    logical, _file = changed[plan.CATALOG]
    changed[plan.CATALOG] = (logical, "sha256:" + "0" * 64)
    monkeypatch.setattr(plan, "SOURCES", changed)
    with pytest.raises(ValueError, match="drifted"):
        plan.build(ROOT)


def test_retired_operation_state_is_checked_in_and_non_relaunchable() -> None:
    value = _load()
    evidence = json.loads((ROOT / plan.LIVE_CENSUS).read_text())
    retired = value["operation"]["retired_operation_evidence"]
    assert value["sources"][plan.LIVE_CENSUS] == {
        "logical_sha256": evidence["sha256"],
        "file_sha256": plan.SOURCES[plan.LIVE_CENSUS][1],
    }
    assert retired["wave_a_state"] == "consumed_non_relaunchable"
    assert retired["wave_a_reusable_verified_success_trajectories"] == 0
    assert retired["wave_b_state"] == "unused_at_observation_not_launch_authorized"
    assert value["operation"]["retired_names_or_authorizations_may_be_reused"] is False
