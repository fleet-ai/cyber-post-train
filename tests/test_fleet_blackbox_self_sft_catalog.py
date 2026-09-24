"""Contracts for the sanitized Qwen self-SFT task catalog."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from training import fleet_blackbox_self_sft_catalog as catalog
from training import task_family_split

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / catalog.OUTPUT


def _load() -> dict:
    value = json.loads(CATALOG.read_text())
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    assert value["sha256"] == task_family_split.canonical_digest(unsigned)
    return value


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value)) if value else set()
    return set()


def test_checked_in_catalog_rebuilds_from_exact_sources() -> None:
    actual = _load()
    assert catalog.build(ROOT, actual["authority_recheck"]) == actual
    catalog.check(ROOT)


def test_catalog_partitions_every_exact_current_blackbox_version_once() -> None:
    value = _load()
    current = json.loads((ROOT / catalog.INVENTORY).read_text())
    rows = value["task_versions"]
    identities = {(row["task_key"], row["task_version_id"]) for row in rows}
    current_identities = {(row["task_key"], row["task_version_id"]) for row in current["tasks"]}
    assert len(rows) == len(identities) == 1217
    assert identities == current_identities
    assert Counter(row["classification"] for row in rows) == {
        "immediately_train_eligible": 50,
        "heldout_family_overlap": 25,
        "known_broken": 74,
        "qa_needed": 33,
        "unanalyzed": 1035,
    }


def test_transitive_components_cannot_cross_train_or_heldout_roles() -> None:
    value = _load()
    by_component: dict[str, list[dict]] = defaultdict(list)
    for row in value["task_versions"]:
        family = row["transitive_family"]
        if "component_id" in family:
            by_component[family["component_id"]].append(row)
        else:
            assert row["classification"] in {"known_broken", "unanalyzed"}
            assert family == {"role": None, "status": "unresolved_blocked"}

    train = {
        component
        for component, rows in by_component.items()
        if rows[0]["classification"] == "immediately_train_eligible"
    }
    heldout = {
        component
        for component, rows in by_component.items()
        if rows[0]["classification"] == "heldout_family_overlap"
    }
    qa = {
        component
        for component, rows in by_component.items()
        if rows[0]["classification"] == "qa_needed"
    }
    assert (len(train), len(heldout), len(qa)) == (50, 25, 26)
    assert not (train & heldout or train & qa or heldout & qa)
    for rows in by_component.values():
        classifications = {row["classification"] for row in rows}
        roles = {row["transitive_family"]["role"] for row in rows}
        assert len(classifications) == 1
        assert len(roles) == 1


def test_train_ceiling_and_next_wave_are_fail_closed() -> None:
    value = _load()
    ceiling = value["maximum_safe_train_only"]
    wave = value["next_qualification_wave"]
    canary = json.loads((ROOT / catalog.CANARY).read_text())
    assert ceiling == {
        "exact_task_versions": 50,
        "transitive_components": 50,
        "condition": (
            "future collection trains only on the exact frozen train components and performs "
            "a just-in-time exact task/runtime freshness check before rollout"
        ),
        "launch_ready_now": False,
        "reason_launch_not_ready": (
            "this static catalog does not replace a fresh runtime binding and availability "
            "check, and it authorizes no rollout"
        ),
    }
    assert (wave["task_key"], wave["task_version_id"]) == (
        canary["candidate"]["task_key"],
        canary["candidate"]["task_version_id"],
    )
    assert wave["component_id"] == canary["candidate"]["shared_atom_component_id"]
    assert wave["launch_authorized"] is False
    assert wave["model_calls"] == 0
    assert wave["remaining_candidate_components_after_canary"] == 25


def test_live_recheck_and_public_artifact_retain_no_private_payload() -> None:
    value = _load()
    live = value["authority_recheck"]
    assert live["catalog"]["exact_inventory_match"] is True
    assert live["session_store"]["exact_task_versions_checked"] == 75
    assert live["session_store"]["outcomes"] == {
        "proven_model_failure": 33,
        "proven_success": 42,
    }
    assert live["exact_source_lineage"]["exact_match"] is True
    assert live["exact_source_lineage"]["artifact_aliases_used"] is False
    assert value["safety"] == {
        "external_mutations": 0,
        "kubernetes_objects_created": 0,
        "registry_writes": 0,
        "rollouts_launched": 0,
    }
    forbidden = {
        "answer",
        "credential",
        "flag",
        "prompt",
        "score",
        "session_id",
        "trace",
        "verifier_execution_id",
    }
    assert not (_keys(value) & forbidden)
