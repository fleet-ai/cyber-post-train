"""Deterministic Fresh75 export packets never submit or mutate external state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import fresh75_export_packets as packets
from training.io import digest_json
from training.sft_runtime import _unsigned_digest


def test_checked_in_packet_set_is_exact_and_inert():
    result = packets.verify_packet_set()
    assert result["status"] == "valid_no_submit_packet_set"
    assert result["packets"] == 7
    assert result["launchable"] is False
    value = json.loads(packets.DEFAULT_PACKETS.read_text())
    assert value["packet_set_sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "packet_set_sha256"}
    )
    assert value["launchable"] is value["submitted"] is False
    assert value["external_mutation_authorized"] is False
    assert value["failure_budget"] == {"limit": 10, "consumed": 10, "remaining": 0}
    assert value["execution_record"] == {
        "kind": "offline_no_submit_packets",
        "external_api_calls_performed": 0,
        "cluster_calls_performed": 0,
        "jobs_submitted": 0,
        "resources_created": 0,
        "sfs_writes_performed": 0,
    }


def test_all_seven_accepted_arms_have_unique_create_once_destinations():
    value = packets.build_packet_set()
    assert tuple(packet["arm_id"] for packet in value["packets"]) == packets.PRIORITY
    assert [packet["priority_rank"] for packet in value["packets"]] == list(range(1, 8))
    paths: list[str] = []
    for packet in value["packets"]:
        assert packet["schema"] == packets.PACKET_SCHEMA
        assert packet["status"] == "no_submit_blocked_waiting_sfs_payload_seal"
        assert packet["launchable"] is packet["submitted"] is False
        assert packet["external_mutation_authorized"] is False
        assert packet["source_checkpoint"]["payload_seal_status"] == ("required_not_yet_accepted")
        assert packet["source_checkpoint"]["payload_manifest_file_sha256"] is None
        assert packet["source_checkpoint"]["payload_manifest_receipt_sha256"] is None
        paths.extend(
            value
            for key, value in packet["destinations"].items()
            if key
            in {
                "checkpoint_manifest",
                "export_root",
                "cpu_check_receipt",
                "gpu_reload_root",
                "staged_model_root",
            }
        )
    assert len(paths) == len(set(paths)) == 35


def test_zero_step_export_reload_and_stage_contracts_are_fail_closed():
    value = packets.build_packet_set()
    phases = value["phase_contracts"]
    assert phases["seal"] == {
        "operation": "training.checkpoints.seal",
        "gpu_count": 0,
        "create_once": True,
        "expected_schema": "cyber_skyrl_checkpoint_manifest_v1",
        "expected_world_size": 8,
        "expected_gpu_reload_verified": False,
    }
    export = phases["export"]
    assert export["gpu_count"] == 0 and export["create_once"] is True
    assert export["manifest_file_sha256_required"] is True
    assert export["expected"]["optimizer_steps_executed"] == 0
    assert export["expected"]["dtype"] == "BF16"
    assert export["expected"]["restored_base_tensors"] == sorted(
        packets.QWEN36_EXACT_MTP_OMISSION_KEYS
    )
    assert export["expected"]["tensor_count"] == 1199
    assert export["expected"]["tensor_bytes"] == 55_562_855_904
    cpu, gpu = phases["cpu_reload"], phases["one_gpu_reload"]
    assert cpu["expected"]["optimizer_steps_executed"] == 0
    assert cpu["expected"]["gpu_reload_verified"] is False
    assert gpu["workers"] == gpu["gpus_per_worker"] == 1
    assert gpu["priority_class"] == "c1"
    assert gpu["expected_queue_priority_class"] == "q1"
    assert gpu["requeue_if_preempted"] is False
    assert gpu["expected"]["optimizer_steps_executed"] == 0
    assert gpu["expected"]["gpu_reload_verified"] is True
    assert gpu["expected"]["finite_logits"] is True
    stage = phases["staging"]
    assert stage["gpu_count"] == 0 and stage["namespace"] == "inference"
    assert stage["destination_must_be_absent"] is True
    assert stage["payload"] == {
        "mapping": "exact EXPORT.files object with bare per-file SHA-256 values",
        "expected_file_count": 29,
        "maximum_file_bytes": 4 * 1024**3,
        "maximum_total_bytes": 64 * 1024**3,
    }
    assert stage["resource_shape"] == {
        "cpu_request": "4",
        "cpu_limit": "8",
        "memory_request": "8Gi",
        "memory_limit": "16Gi",
        "ephemeral_storage_request": "1Gi",
        "ephemeral_storage_limit": "2Gi",
    }
    assert stage["atomic_transaction"] == "directory_rename_noreplace_v1"
    assert stage["registration"] is None


def test_packet_source_receipts_match_frozen_handoffs():
    for packet in packets.build_packet_set()["packets"]:
        handoff_path = packets.REPO_ROOT / packet["accepted_handoff"]["path"]
        handoff = json.loads(handoff_path.read_text())
        checkpoint = handoff["checkpoint_selection"]
        source = packet["source_checkpoint"]
        assert packet["accepted_handoff"]["file_sha256"] == (
            "sha256:" + hashlib.sha256(handoff_path.read_bytes()).hexdigest()
        )
        assert source["path"] == checkpoint["checkpoint_path"]
        assert source["optimizer_step"] == checkpoint["optimizer_step"]
        assert source["saved_receipt_path"] == checkpoint["checkpoint_saved_receipt_path"]
        assert (
            source["saved_receipt_file_sha256"]
            == (checkpoint["checkpoint_saved_receipt_file_sha256"])
        )
        assert source["saved_receipt_sha256"] == (checkpoint["checkpoint_saved_receipt_sha256"])


def test_packet_tamper_is_rejected(tmp_path):
    value = packets.build_packet_set()
    value["packets"][0]["source_checkpoint"]["optimizer_step"] += 1
    path = tmp_path / "packets.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="deterministic generator"):
        packets.verify_packet_set(path)


def test_read_only_sfs_audit_accepts_exact_saved_receipt(tmp_path):
    mount = tmp_path / "sfs"
    logical_checkpoint = "/mnt/sfs/jobs/synthetic/checkpoints/global_step_2"
    checkpoint = mount / "jobs/synthetic/checkpoints/global_step_2"
    (checkpoint / "policy/huggingface").mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"data")
    (checkpoint / "trainer_state.pt").write_bytes(b"trainer")
    (checkpoint / "policy/fsdp_config.json").write_text(
        json.dumps({"fsdp_strategy": "fsdp", "world_size": 1})
    )
    for kind in ("model", "optim", "extra_state"):
        (checkpoint / f"policy/{kind}_world_size_1_rank_0.pt").write_bytes(kind.encode())
    (checkpoint / "policy/huggingface/config.json").write_text("{}")
    logical_receipt = "/mnt/sfs/jobs/synthetic/checkpoint_receipts/step-000002.json"
    receipt_path = mount / "jobs/synthetic/checkpoint_receipts/step-000002.json"
    receipt_path.parent.mkdir(parents=True)
    unsigned = {
        "optimizer_step": 2,
        "checkpoint_path": logical_checkpoint,
        "plan_sha256": "a" * 64,
    }
    receipt = {**unsigned, "receipt_sha256": _unsigned_digest(unsigned)}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    packet_set = {
        "packets": [
            {
                "arm_id": "synthetic",
                "source_plan": {"plan_sha256": "sha256:" + "a" * 64},
                "source_checkpoint": {
                    "saved_receipt_path": logical_receipt,
                    "saved_receipt_file_sha256": (
                        "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()
                    ),
                    "saved_receipt_sha256": "sha256:" + receipt["receipt_sha256"],
                    "optimizer_step": 2,
                    "path": logical_checkpoint,
                    "world_size": 1,
                },
            }
        ]
    }
    packet_set["packet_set_sha256"] = digest_json(packet_set)
    path = tmp_path / "packets.json"
    path.write_text(json.dumps(packet_set))
    result = packets.audit_sfs_saved_receipts(path, mount_root=mount)
    assert result["status"] == "read_only_saved_receipt_and_layout_audit_passed"
    assert result["external_api_calls_performed"] == result["sfs_writes_performed"] == 0
    assert result["audited"][0]["checkpoint_files_observed"] == 7
    assert result["audited"][0]["checkpoint_bytes_observed"] > 0


def test_read_only_sfs_audit_rejects_symlink_receipt(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        packets._read_regular_nofollow(link)


def test_generator_has_no_external_control_plane_client():
    source = Path(packets.__file__).read_text()
    for forbidden in (
        "httpx",
        "requests",
        "kubernetes",
        "subprocess",
        '"POST"',
        '"DELETE"',
        ".submit(",
        ".create(",
    ):
        assert forbidden not in source
