from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
LINEAGE = ROOT / "configs/data/qwen38-teacher3k-training-lineage-map-20260924-v1.json"
PROTOCOL = (
    ROOT / "configs/evaluation/qwen38-teacher3k-fleet-heldout20-pass4-protocol-v2.json"
)
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"
INVENTORY = ROOT / "configs/data/fleet-blackbox-current-high-quality-20260914-v1.json"
OLD = ROOT / "configs/evaluation/qwen38-teacher3k-fleet-heldout25-proposal-v1.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(value: dict) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def nested_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(nested_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    return set()


def test_lineage_map_is_complete_self_digesting_and_safe() -> None:
    lineage = load(LINEAGE)
    assert lineage["sha256"] == digest(lineage)
    assert lineage["counts"] == {
        "training_task_keys": 496,
        "training_task_versions": 1176,
        "source_sessions": 2886,
        "reviewed_training_task_keys": 496,
        "unproven_training_task_keys": 0,
    }
    assert lineage["method"]["exact_api_metadata_versions"] == 1153
    assert lineage["method"]["exact_key_sibling_versions"] == 1
    assert lineage["method"]["reviewed_legacy_alias_versions"] == 22
    assert lineage["method"]["unproven_versions"] == 0
    keys = lineage["training_task_keys"]
    assert len(keys) == len({row["task_key"] for row in keys}) == 496
    versions = [version for row in keys for version in row["versions"]]
    assert len(versions) == 1176
    assert all(row["lineage_status"] == "reviewed" for row in keys)
    assert all(version["atom_lineages"] for version in versions)

    forbidden = {
        "session_id",
        "prompt",
        "answer",
        "trace",
        "flag",
        "score",
        "reward",
        "output",
    }
    assert not nested_keys(lineage) & forbidden


def test_corrected_protocol_binds_sources_and_rejects_old_proposal() -> None:
    protocol, lineage = load(PROTOCOL), load(LINEAGE)
    split, inventory = load(SPLIT), load(INVENTORY)
    assert not OLD.exists()
    assert protocol["sha256"] == digest(protocol)
    source = protocol["source"]
    assert source["study_split_file_sha256"] == file_digest(SPLIT)
    assert source["study_split_sha256"] == split["sha256"]
    assert source["quality_inventory_file_sha256"] == file_digest(INVENTORY)
    assert source["quality_inventory_sha256"] == inventory["sha256"]
    assert source["training_lineage_map_file_sha256"] == file_digest(LINEAGE)
    assert source["training_lineage_map_sha256"] == lineage["sha256"]
    assert protocol["status"] == "corrected_proposal_not_launched"
    assert protocol["supersedes"]["sha256"] == (
        "sha256:4f9b8357679d39aed8383429491dcaece10fa13aad4a690d6a177ebbe01c2750"
    )


def test_all_25_tasks_are_classified_and_exact_exposures_are_recorded() -> None:
    protocol = load(PROTOCOL)
    classification = protocol["classification"]
    assert classification["original_heldout_tasks"] == 25
    assert classification["clean"] == 20
    assert classification["exposed"] == 5
    assert classification["unproven"] == 0
    assert classification["exposed_training_aliases"] == 6
    assert classification["exact_task_key_overlap"] == 0
    assert classification["exact_task_version_overlap"] == 0

    exposed = {
        row["task_key"]: {
            alias["task_key"] for alias in row["training_aliases"]
        }
        for row in classification["tasks"]
        if row["training_lineage_status"] == "exposed"
    }
    assert exposed == {
        "cysec1-2-current-gen_blackbox-bc363f8239aeeaa3f648046a__blackbox_ctf_v1": {
            "cysec1-2-cmp-current-fentry-oracle-epm-690cfbccaf8e_"
            "blackbox-bc363f8239aeeaa3f648046a__blackbox_ctf_v1"
        },
        "cysec1-2-fakelook-gen_blackbox-8095d44c3871865064a0702c__blackbox_ctf_v1": {
            "cysec1-2-cmp-current-fakelook-fentry-667fe0152cb9_"
            "blackbox-8095d44c3871865064a0702c__blackbox_ctf_v1"
        },
        "cysec1-2-fentry-gen_blackbox-54997075dbb8019cd6e2e0ba__blackbox_ctf_v1": {
            "cysec1-2-fentry-gen_blackbox-54997075dbb8019cd6e2e0ba__blackbox_ctf_v1__hinted"
        },
        "cysec1-2-fira-gen_blackbox-5b51b6dcb4193e59b96b1b13__blackbox_ctf_v1": {
            "cysec1-2-cmp-current-fira-fubspot-77dd1028d7ea_"
            "blackbox-5b51b6dcb4193e59b96b1b13__blackbox_ctf_v1"
        },
        "cysec1-2-fubspot-gen_blackbox-438e3127ab9af7efad692398__blackbox_ctf_v1": {
            "cysec1-2-cmp-5apps-c332fdd79799_blackbox-438e3127ab9af7efad692398__blackbox_ctf_v1",
            "cysec1-2-cmp-current-fira-fubspot-2f105967f87b_"
            "blackbox-438e3127ab9af7efad692398__blackbox_ctf_v1",
        },
    }


def test_corrected_pass4_roster_is_exact_and_lineage_disjoint() -> None:
    protocol, lineage = load(PROTOCOL), load(LINEAGE)
    selection = protocol["selection"]
    assert selection["task_family_count"] == selection["exact_task_version_count"] == 20
    assert selection["source_role_counts"] == {"dev": 13, "final_test": 7}
    assert selection["pass_k"] == 4
    assert selection["planned_rollouts_per_arm"] == 80
    assert len({row["task_key"] for row in selection["tasks"]}) == 20
    assert len({row["task_version_id"] for row in selection["tasks"]}) == 20
    assert len({row["group_id"] for row in selection["tasks"]}) == 20

    training_atoms = {
        atom
        for row in lineage["training_task_keys"]
        for atom in row["atom_lineages"]
    }
    selected_atoms = {
        row["reviewed_task_family"] + ":atom_source" for row in selection["tasks"]
    }
    assert not training_atoms & selected_atoms
    assert protocol["effects"] == {
        "api_mutations": 0,
        "cluster_mutations": 0,
        "eval_launches": 0,
        "model_calls": 0,
        "scoring_calls": 0,
    }
