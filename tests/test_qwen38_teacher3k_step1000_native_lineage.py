from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT / "docs/evidence/qwen38-study/2026-09-24-q38-step1000-native-lineage-absence-v1.json"
)
PROMOTION = ROOT / "docs/evidence/qwen38-teacher3k32-step1000-promotion-20260922.json"
RUN = ROOT / "configs/runs/qwen38-teacher3k-32k-full-b8-lr3e6-v3.json"


def _canonical_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_native_lineage_evidence_is_self_digested_and_private_safe() -> None:
    value = json.loads(EVIDENCE.read_text())
    claimed = value.pop("receipt_sha256")

    assert claimed == _canonical_digest(value)
    assert value["schema"] == "cyber_qwen38_step1000_native_lineage_absence_v1"
    assert value["classification"] == {
        **value["classification"],
        "native_fp32_checkpoint_available": False,
        "immutable_checkpoint_receipt_available": True,
        "immutable_checkpoint_seal_available": True,
        "accepted_bf16_payload_available": True,
        "exact_fp32_to_bf16_replay_possible_from_discoverable_artifacts": False,
        "absolute_external_backup_absence_claimed": False,
    }
    assert all(item is False for item in value["privacy"].values())


def test_lineage_evidence_binds_the_existing_run_and_promotion() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    promotion = json.loads(PROMOTION.read_text())
    run = json.loads(RUN.read_text())

    assert evidence["source_run"]["name"] == run["name"]
    assert evidence["source_run"]["output_root"] == run["output_root"]
    assert (
        evidence["retention_contract"]["checkpoint_interval_steps"]
        == run["recipe"]["checkpoint_interval"]
    )
    assert evidence["retention_contract"]["keep_checkpoints"] == run["recipe"]["keep_checkpoints"]

    checkpoint = evidence["surviving_artifacts"]["checkpoint_receipt"]
    seal = evidence["surviving_artifacts"]["checkpoint_seal"]
    export = evidence["surviving_artifacts"]["bf16_export"]
    promoted_checkpoint = promotion["checkpoint"]
    promoted_export = promotion["export"]

    assert "sha256:" + checkpoint["file_sha256"] == promoted_checkpoint["receipt_file_sha256"]
    assert "sha256:" + checkpoint["receipt_sha256"] == promoted_checkpoint["receipt_sha256"]
    assert "sha256:" + seal["file_sha256"] == promoted_checkpoint["seal_file_sha256"]
    assert "sha256:" + seal["receipt_sha256"] == promoted_checkpoint["seal_sha256"]
    assert "sha256:" + export["file_sha256"] == promoted_export["receipt_file_sha256"]
    assert "sha256:" + export["receipt_sha256"] == promoted_export["receipt_sha256"]
    assert (
        "sha256:" + export["payload_manifest_sha256"] == promoted_export["payload_manifest_sha256"]
    )
    assert export["payload_total_bytes"] == promoted_export["payload_total_bytes"]


def test_replay_classification_does_not_overclaim_bounded_absence() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    observation = evidence["native_checkpoint_observation"]["paths"]
    replay = evidence["replay"]
    limitations = evidence["bounded_search"]["scope_limitations"]

    assert observation["native_step1000"]["state"] == "absent"
    assert observation["retained_step1800"]["state"] == "present_directory"
    assert observation["retained_step1837"]["state"] == "present_directory"
    assert observation["step1000_receipt"]["state"] == "present_file"
    assert observation["step1000_seal"]["state"] == "present_file"
    assert replay["exact_fp32_to_bf16_replay_possible"] is False
    assert replay["conversion"]["requires_eight_native_cpu_fp32_model_shards"] is True
    assert replay["conversion"]["source_to_output_cast_is_lossy"] is True
    assert replay["accepted_bf16_payload_can_be_rehashed_or_reused"] is True
    assert replay["accepted_bf16_payload_is_a_native_replay"] is False
    assert any("cannot be disproved" in item for item in limitations)
