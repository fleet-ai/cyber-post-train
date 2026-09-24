import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs/evidence/qwen38-sft-checkpoint-inventory-20260924.json"


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read() -> dict:
    value = json.loads(RECEIPT.read_text())
    assert value["sha256"] == _digest({key: item for key, item in value.items() if key != "sha256"})
    return value


def test_inventory_is_sanitized_inert_and_self_bound() -> None:
    value = _read()
    assert value["status"] == "sanitized_read_only_inventory_no_launch"
    assert value["scope"]["private_task_identities_included"] is False
    assert value["scope"]["benchmark_content_or_results_read"] is False
    assert value["scope"]["capability_claimed"] is False
    assert all(item == 0 for item in value["effects"].values())


def test_inventory_preserves_the_actionable_order() -> None:
    value = _read()
    assert [item["id"] for item in value["actionable_order"]] == [
        "current_base_step1000_matched_pair",
        "teacher3k96_step400",
        "frozen_top5_remaining_candidates",
    ]
    assert value["actionable_order"][0]["scientific_state"] == "no accepted capability result yet"
    assert value["actionable_order"][1]["operational_state"].endswith(
        "checkpoint absent at observation time"
    )


def test_inventory_binds_repository_authorities() -> None:
    value = _read()
    for name in (
        "teacher3k_lineage_map",
        "teacher3k_heldout_protocol",
        "step1000_acceptance",
        "step400_successor",
        "top5_operational_matrix",
    ):
        authority = value["authorities"][name]
        path = ROOT / authority["path"]
        assert authority["file_sha256"] == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        logical_sha256 = json.loads(path.read_text())["sha256"]
        if not logical_sha256.startswith("sha256:"):
            logical_sha256 = "sha256:" + logical_sha256
        assert authority["logical_sha256"] == logical_sha256


def test_inventory_binds_corrected_teacher3k_heldout_scope() -> None:
    value = _read()
    heldout = value["heldout_lineage"]
    protocol = json.loads(
        (ROOT / value["authorities"]["teacher3k_heldout_protocol"]["path"]).read_text()
    )
    assert heldout["clean_families"] == 20
    assert heldout["clean_strata"] == {"development": 13, "final_test": 7}
    assert heldout["excluded_families"] == 5
    assert heldout["excluded_training_aliases"] == 6
    assert heldout["exact_task_key_overlap"] == 0
    assert heldout["exact_task_version_overlap"] == 0
    assert heldout["shared_atom_alias_lineage_requires_family_exclusion"] is True
    assert heldout["qa33_expansion"]["stable_atom_components"] == 26
    assert heldout["qa33_expansion"]["shared_atom_exposed_versions"] == 17
    assert heldout["qa33_expansion"]["unexposed_singleton_components"] == 16
    assert heldout["qa33_expansion"]["add_now"] == 0
    assert set(heldout["valid_corpus_manifests"]) == {"32768", "65536", "98304"}
    assert set(heldout["valid_corpus_manifests"].values()) == set(
        protocol["compatibility"]["valid_checkpoint_corpora"]
    )


def test_only_base_and_step1000_are_current_live_parity_pair() -> None:
    value = _read()
    artifacts = {item["artifact_id"]: item for item in value["artifacts"]}
    assert artifacts["qwen38-27b-base-control"]["route"]["ready_replicas"] == 1
    assert artifacts["q38-teacher3k-32k-step1000"]["route"]["ready_replicas"] == 1
    for artifact_id in (
        "q38-d32-b16-lr5e6-step200",
        "q38-t3k64-b8-step225",
        "q38-t3k32-lr1-step500",
        "q38-t3k96-b8-step300",
    ):
        assert artifacts[artifact_id]["route"]["phase"] == "paused"
        assert artifacts[artifact_id]["route"]["active_pods"] == 0
    assert value["context96_progression"]["step400_checkpoint_present"] is False
