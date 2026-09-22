"""No-launch Fleet dev17 packet for sealed Teacher3K 32K step 900."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
PACKET = EVAL / "qwen38-teacher3k-32k-step900-fleet-dev17-seed43-preparation-v1.json"
STEP800 = EVAL / "qwen38-teacher3k-32k-step800-fleet-dev17-seed43-preparation-v1.json"
BINDINGS = EVAL / "qwen38-fleet-dev17-exact-binding-roster-20260922-v1.json"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
BASE = EVAL / "qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"
PROTOCOL = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_packet_is_self_bound_and_authorizes_no_mutation() -> None:
    packet = read(PACKET)
    assert packet["sha256"] == digest(
        {key: value for key, value in packet.items() if key != "sha256"}
    )
    assert packet["status"] == "blocked_not_launchable"
    assert packet["launchable"] is False
    assert packet["supersedes_for_future_launch"] == read(STEP800)["packet_id"]
    assert packet["operation"] == {
        "jobs_submitted": 0,
        "config_maps_created": 0,
        "databases_created": 0,
        "outputs_created": 0,
        "routes_resumed_or_created": 0,
        "evaluation_sessions_created": 0,
        "gh_pages_writes": 0,
    }
    future = packet["future_candidate"]
    assert future["resource_identities_minted"] is False
    assert all(future[field] is None for field in ("job", "config_map", "database", "output_root"))


def test_step800_stays_historical_and_base_reuse_remains_conditional() -> None:
    packet, step800 = read(PACKET), read(STEP800)
    assert step800["candidate_checkpoint"]["optimizer_step"] == 800
    assert step800["operation"]["jobs_submitted"] == 0
    baseline = packet["accepted_baseline"]
    assert baseline["state"] == "accepted_complete_reusable_only_after_candidate_parity"
    assert baseline["planned_cells"] == baseline["accepted_cells"] == 17
    assert baseline["valid_cells"] == baseline["automatic_checker_results"] == 17
    assert baseline["technical_invalid_cells"] == 0
    assert baseline["database"] == "q38_dev17_s43_base_p1_v1"
    assert baseline["config_file_sha256"] == file_sha256(BASE)
    assert "same frozen treatment and exact binding roster" in baseline["reuse_rule"]
    assert "fresh live parity" in baseline["reuse_rule"]


def test_step900_preserves_the_exact_seed43_opencode_treatment() -> None:
    packet = read(PACKET)
    protocol, roster, task_set = read(PROTOCOL), read(BINDINGS), read(TASK_SET)
    frozen = packet["frozen_protocol"]
    assert frozen["file_sha256"] == file_sha256(PROTOCOL)
    assert frozen["sha256"] == "sha256:" + protocol["sha256"]
    assert frozen["task_set_file_sha256"] == file_sha256(TASK_SET)
    assert frozen["binding_roster_file_sha256"] == file_sha256(BINDINGS)
    assert frozen["bindings_sha256"] == roster["bindings_sha256"]
    assert frozen["split_file_sha256"] == file_sha256(SPLIT)
    assert frozen["selection_sha256"] == task_set["selection_sha256"]
    assert frozen["task_count"] == len(task_set["tasks"]) == len(roster["bindings"]) == 17
    assert all(row["task_version_id"] for row in task_set["tasks"])
    assert frozen["harness"] == "opencode"
    assert frozen["harness_version"] == "1.18.27"
    assert frozen["context_management"] == ("opencode_1.18.27_native_compaction_autocontinue_v2")
    assert frozen["context_window_size"] == 262144
    assert frozen["sampling"] == {"temperature": 0.6, "top_p": 0.95, "seed": 43}
    assert frozen["pass_k"] == 1
    assert frozen["training_data_eligible"] is False


def test_step900_receipts_and_zero_gpu_terminal_objects_are_exact() -> None:
    packet = read(PACKET)
    checkpoint = packet["candidate_checkpoint"]
    assert checkpoint["artifact_id"] == "q38-teacher3k-32k-step900"
    assert checkpoint["optimizer_step"] == 900
    assert checkpoint["checkpoint_path"].endswith("/checkpoints/global_step_900")
    assert checkpoint["checkpoint_receipt"] == {
        "path": "/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/checkpoint_receipts/step-000900.json",
        "file_sha256": "sha256:cb1933fb770c2c77adcc84db25b89a024f32df8abdb7d58cdf5f90515b2a224f",
        "receipt_sha256": "sha256:f7eb50db657486f901db6b03e1a17ebf0dcd0d0124c8c7491cb5a405110e54e8",
    }
    assert checkpoint["supervised_tokens_at_step"] == 27896516
    assert checkpoint["world_size"] == 8
    assert checkpoint["file_count"] == 33
    assert checkpoint["total_bytes"] == 324627486795

    promotion = packet["observed_zero_gpu_promotion"]
    expected = {
        "checkpoint_seal": (
            "db1d2298-ccda-4676-9235-0f4e14d2cc4e",
            "sha256:35c69a1e0b5829bcfa041a109cc961a0c3641267f8e945364cfbbe9f4b8584b7",
        ),
        "bf16_export": (
            "90dd9e6f-41e6-4039-8d2d-f27a6a25d5de",
            "sha256:33fab68ad59aab8ff24753c2edbb81371a94b99e3d0284fc29e200e0a664f7e6",
        ),
        "cpu_layout_check": (
            "4e0c0242-ff4e-4e80-af74-ca6a916b9a98",
            "sha256:6048ddf2286a3849a2e3901dd78a24a5c0ea2719940477d61b6cfc6abe0e87a6",
        ),
    }
    for name, (uid, receipt) in expected.items():
        row = promotion[name]
        assert row["pod_uid"] == uid
        assert row["receipt_sha256"] == receipt
        assert row["phase"] == "Succeeded"
        assert row["exit_code"] == row["restarts"] == row["warning_events"] == row["gpus"] == 0
    assert promotion["bf16_export"]["payload_manifest_sha256"] == (
        "sha256:d7f3e596b3b54736ece2f7d1c67c220ea19c27ba47defedd923e61c48ca2df73"
    )
    assert promotion["bf16_export"]["optimizer_steps_executed"] == 0
    assert promotion["cpu_layout_check"]["gpu_reload_verified"] is False
    assert promotion["cpu_layout_check"]["serving_qualified"] is False


def test_step900_gpu_reload_is_exact_accepted_and_fully_cleaned() -> None:
    reload = read(PACKET)["observed_dev_gpu_reload"]
    assert reload["state"] == "accepted_and_fully_cleaned"
    assert reload["pod_uid"] == "823c5d1c-e409-415d-8820-f9bce13803be"
    assert reload["phase"] == "Succeeded"
    assert reload["exit_code"] == reload["restarts"] == reload["warning_events"] == 0
    assert reload["gpus"] == 1
    assert reload["priority_class"] == "c1"
    assert reload["failure_alerts"] == "off"
    assert reload["active_deadline_seconds"] == 1800
    assert reload["receipt_file_sha256"] == (
        "sha256:da73cfddf4b2a01b3b516cc040dad4b9b86b06ccd484c09fcb907c15940ed6f9"
    )
    assert reload["receipt_sha256"] == (
        "sha256:93a7c4d90afe22f11ebb20d15afe2514078ee3f2f0802801348773ee3c1fdad3"
    )
    assert reload["finite_logits"] is True
    assert reload["generated_tokens"] == 2
    assert reload["optimizer_steps_executed"] == 0
    assert reload["source_unchanged"] is True
    assert reload["serving_qualified"] is False
    assert reload["gpu_pod_deleted"] is reload["cleanup_pod_deleted"] is True
    assert reload["cleanup_pod_uid"] == "83eb451c-f0ec-49e0-b4e6-e1d2c613add8"
    assert reload["final_dev_capacity_census"] == {
        "observed_at": "2026-09-22T17:27:11Z",
        "current_nodes": 0,
        "current_gpus": 0,
        "sha256": "sha256:e5ba0807320e30593952bb8fd402938839d9fc0d225bd9eb26decb98eaf51f2b",
    }


def test_stage_is_cleaned_and_registration_is_paused_zero_gpu() -> None:
    accepted = read(PACKET)["accepted_stage_and_registration"]
    assert accepted["state"] == "accepted_through_paused_zero_gpu_registration"
    assert accepted["stage_plan_sha256"] == (
        "sha256:27c15422d77ebf293007173e9e36b73f931cbcbd0ae109b23b4d1b81243fcdd6"
    )
    assert accepted["payload_manifest_sha256"] == (
        "sha256:d7f3e596b3b54736ece2f7d1c67c220ea19c27ba47defedd923e61c48ca2df73"
    )
    stage = accepted["stage"]
    assert stage["namespace"] == "inference"
    assert stage["pod_uid"] == "dca3df2d-fb42-4e67-991f-aa4559071153"
    assert stage["config_map_uid"] == "be77a0e2-5009-4a68-98aa-e23fc3598fd2"
    assert stage["phase"] == "Succeeded"
    assert stage["exit_code"] == stage["restarts"] == stage["warning_events"] == 0
    assert stage["gpus"] == 0
    assert stage["priority_class"] == "c1"
    assert stage["failure_alerts"] == "off"
    assert stage["pod_deleted"] is stage["config_map_deleted"] is True
    assert stage["acceptance_receipt_sha256"] == (
        "sha256:97585eecea6c9af80764d1ae7e2a212a7832944e28713691cb384364ecdca6d1"
    )

    registration = accepted["registration"]
    assert registration["model_id"] == "chris-q38-t3k32-s900-v1"
    assert registration["post_attempts"] == 1
    assert registration["phase"] == "paused"
    assert registration["desired_replicas"] == 0
    assert registration["ready_replicas"] == registration["active_pods"] == 0
    assert registration["routing_enabled"] is False
    assert registration["matching_kubernetes_pods"] == []
    assert registration["api_uid"] is None
    assert registration["api_uid_readback_supported"] is False


def test_launch_remains_closed_until_parity_capacity_and_preview_pass() -> None:
    packet = read(PACKET)
    gates = "\n".join(packet["remaining_gates"])
    for required in (
        "capacity census",
        "live parity receipt",
        "candidate config compiled",
        "duplicate census",
        "two byte-equivalent Kubernetes server dry-runs",
        "fleet.ai/failure-alerts=off",
    ):
        assert required in gates
    future = packet["future_candidate"]
    assert future["served_id"] == future["session_model"] == "chris-q38-t3k32-s900-v1"
    assert future["model_revision_from_staged_payload_manifest_sha256"] == (
        "sha256:d7f3e596b3b54736ece2f7d1c67c220ea19c27ba47defedd923e61c48ca2df73"
    )
    assert future["evaluation_config_file_sha256"] is None
    assert future["evaluation_plan_sha256"] is None
    assert future["route_catalog"] is None
    assert future["route_model_info"] is None
    assert future["route_server_info"] is None


def test_packet_contains_no_private_or_outcome_material() -> None:
    packet = read(PACKET)
    assert not any(packet["privacy"].values())
    assert packet["scientific_boundary"]["capability_claimed"] is False
    assert packet["scientific_boundary"]["final_test_split_access"] == "sealed"
    contract = packet["result_contract"]
    assert contract["public_result_and_webapp_update_before_acceptance"] is False
    assert contract["selective_replay_forbidden"] is True
    assert contract["valid_zero_is_final"] is True
