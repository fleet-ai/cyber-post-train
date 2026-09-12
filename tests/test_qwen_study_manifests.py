import hashlib
import json
from pathlib import Path

from training.io import canonical_json
from training.study_data import _check_split, audit_inventory, check_seal

ROOT = Path(__file__).parents[1]
DATA = ROOT / "configs" / "data"


def load(name):
    return json.loads((DATA / name).read_text())


def test_eligible_allowlist_and_study_inventory_are_exactly_aligned():
    eligible = load("qwen-blackbox-eligible-v1.json")
    expected = hashlib.sha256(
        canonical_json({k: v for k, v in eligible.items() if k != "sha256"}).encode()
    ).hexdigest()
    assert eligible["sha256"] == expected
    assert eligible["counts"]["eligible_versions"] == 89

    inventory = load("qwen-blackbox-study-inventory-v1.json")
    check_seal(inventory, "cyber_study_inventory_v1")
    assert audit_inventory(inventory) == {
        "inventory_task_versions": 89,
        "eligible_task_versions": 89,
        "reviewed_task_families": 89,
        "quarantined_task_versions": {},
        "missing_or_unverified_taxonomy_task_versions": {
            "application": 0,
            "environment": 0,
            "vulnerability_family": 0,
            "difficulty": 0,
        },
        "full_taxonomy_available": True,
        "evidence_scope": (
            "metadata claims bound by digests; unavailable upstream receipts are not re-certified"
        ),
    }
    assert {(r["task_key"], r["task_version_id"]) for r in inventory["tasks"]} == {
        (r["task_key"], r["task_version_id"]) for r in eligible["task_versions"]
    }


def test_split_variants_share_sealed_final_but_materially_vary_dev():
    final = load("qwen-blackbox-study-final-test-v1.json")
    check_seal(final, "cyber_final_test_lock_v1")
    assert len(final["tasks"]) == 10

    splits = [
        load("qwen-blackbox-study-split-a-v1.json"),
        load("qwen-blackbox-study-split-b-v1.json"),
    ]
    trains = [
        load("qwen-blackbox-study-train-a-v1.json"),
        load("qwen-blackbox-study-train-b-v1.json"),
    ]
    for split, train in zip(splits, trains, strict=True):
        _check_split(split)
        check_seal(train, "cyber_task_split_v2")
        assert train == split["training_split"]
        assert split["final_test_lock_sha256"] == final["sha256"]
        assert split["counts"] == {
            "train": {"groups": 59, "task_versions": 59},
            "dev": {"groups": 20, "task_versions": 20},
            "final_test": {"groups": 10, "task_versions": 10},
        }
        gaps = split["representation"]["dev"]
        assert gaps["application"]["max_abs_group_share_gap"] < 0.03
        assert gaps["environment"]["max_abs_group_share_gap"] < 0.03
        assert gaps["vulnerability_family"]["max_abs_group_share_gap"] < 0.06
        assert gaps["difficulty"]["max_abs_group_share_gap"] < 0.04

    def ids(split, name):
        return {r["task_version_id"] for r in split["tasks"] if r["split"] == name}

    assert ids(splits[0], "final_test") == ids(splits[1], "final_test")
    assert len(ids(splits[0], "dev") & ids(splits[1], "dev")) == 5
    assert len(ids(splits[0], "train") & ids(splits[1], "train")) == 44


def test_presplit_receipt_scopes_unexposed_claim_to_this_campaign():
    receipt = load("qwen-blackbox-study-presplit-v1.json")
    check_seal(receipt, "cyber_study_presplit_exposure_v1")
    assert receipt["status"] == "unexposed"
    assert receipt["historical_global_exposure_claimed"] is False
    assert "this exact campaign" in receipt["scope"]


def test_filtered_rl_canary_is_sealed_and_drawn_only_from_eligible_tasks():
    eligible = load("qwen-blackbox-eligible-v1.json")
    task_set = load("qwen38-rl-filtered-canary-task-set-v1.json")
    split = load("qwen38-rl-filtered-canary-split-v1.json")
    catalog = load("qwen38-rl-filtered-canary-tool-catalog-v1.json")

    def sealed_sha(value):
        body = {key: item for key, item in value.items() if key != "sha256"}
        return "sha256:" + hashlib.sha256(canonical_json(body).encode()).hexdigest()

    assert task_set["sha256"] == sealed_sha(task_set)
    assert split["sha256"] == sealed_sha(split)
    assert task_set["source_manifest_sha256"] == "sha256:" + eligible["sha256"]
    catalog_bytes = json.dumps(
        catalog, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode()
    assert task_set["tool_catalog_sha256"] == "sha256:" + hashlib.sha256(
        catalog_bytes
    ).hexdigest()

    eligible_ids = {
        (row["task_key"], row["task_version_id"]) for row in eligible["task_versions"]
    }
    selected = {(row["task_key"], row["task_version_id"]) for row in task_set["tasks"]}
    split_ids = {(row["task_key"], row["task_version_id"]) for row in split["tasks"]}
    assert selected == split_ids
    assert selected <= eligible_ids
    assert [row["split"] for row in split["tasks"]].count("train") == 2
    assert [row["split"] for row in split["tasks"]].count("dev") == 1
