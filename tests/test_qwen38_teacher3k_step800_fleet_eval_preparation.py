"""No-launch Fleet dev17 packet for sealed Teacher3K 32K step 800."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "configs/evaluation"
PACKET = EVAL / "qwen38-teacher3k-32k-step800-fleet-dev17-seed43-preparation-v1.json"
BINDINGS = EVAL / "qwen38-fleet-dev17-exact-binding-roster-20260922-v1.json"
TASK_SET = EVAL / "qwen38-fresh75-fleet-dev17-task-set-v1.json"
BASE = EVAL / "qwen38-base-fleet-dev17-opencode-seed43-pass1-v1.json"
PROTOCOL = EVAL / "qwen38-fleet-dev17-seed43-matched-protocol-v1.json"
SPLIT = ROOT / "configs/data/fleet-blackbox-current-study-split-20260914-v2.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_packet_is_digest_bound_and_authorizes_no_external_mutation() -> None:
    packet = read(PACKET)
    expected = digest({key: value for key, value in packet.items() if key != "sha256"})
    assert packet["sha256"] == expected
    assert packet["status"] == "blocked_not_launchable"
    assert packet["launchable"] is False
    assert packet["operation"] == {
        "jobs_submitted": 0,
        "config_maps_created": 0,
        "databases_created": 0,
        "outputs_created": 0,
        "routes_resumed_or_created": 0,
        "evaluation_sessions_created": 0,
        "gh_pages_writes": 0,
    }
    assert packet["future_candidate"]["resource_identities_minted"] is False
    assert all(
        packet["future_candidate"][field] is None
        for field in ("job", "config_map", "database", "output_root")
    )


def test_accepted_base_is_complete_but_reuse_is_strictly_conditional() -> None:
    packet = read(PACKET)
    baseline = packet["accepted_baseline"]
    assert baseline["state"] == "accepted_complete_reusable_only_after_candidate_parity"
    assert baseline["planned_cells"] == baseline["accepted_cells"] == 17
    assert baseline["valid_cells"] == baseline["automatic_checker_results"] == 17
    assert baseline["technical_invalid_cells"] == 0
    assert baseline["database"] == "q38_dev17_s43_base_p1_v1"
    assert baseline["config_file_sha256"] == file_sha256(BASE)
    assert baseline["aggregate_evidence"] == {
        "repository_ref": "origin/gh-pages",
        "commit": "ac8f79e51fc322461d14f80806d1f9aa81d28e68",
        "path": "fleet-dev17-baseline-seed43-result.json",
        "file_sha256": ("sha256:50158a06a9688500b0c1840ba717f7707560d49ec080d4fb32b6b56e482f98cd"),
        "receipt_roster_sha256": (
            "sha256:7a2c57d0e37318a46825afaca2e8361b598c2673f2cdc8af8882d0c723cd565d"
        ),
    }
    reuse = baseline["reuse_rule"]
    assert "same frozen treatment and exact binding roster" in reuse
    assert "fresh live parity" in reuse
    assert "Otherwise freeze a new base/candidate pair" in reuse


def test_frozen_treatment_uses_exact_version_bindings_and_seed43_opencode() -> None:
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


def test_step800_is_sealed_and_exported_but_not_serving_qualified() -> None:
    packet = read(PACKET)
    checkpoint = packet["candidate_checkpoint"]
    assert checkpoint["artifact_id"] == "q38-teacher3k-32k-step800"
    assert checkpoint["optimizer_step"] == 800
    assert checkpoint["checkpoint_path"].endswith("/checkpoints/global_step_800")
    assert checkpoint["checkpoint_receipt"] == {
        "path": ("/mnt/sfs/jobs/chris-q38-t3k32-b8-v3/checkpoint_receipts/step-000800.json"),
        "file_sha256": ("sha256:0cee9cf72021c5ad6362fc986570f56ab63eead876f21d192f3031ef7ea48401"),
        "receipt_sha256": (
            "sha256:c13d136b44931004f9a845dd2fb3031eed975cb92bd773953fe1176626f2af54"
        ),
    }
    assert checkpoint["supervised_tokens_at_step"] == 24722458
    assert checkpoint["world_size"] == 8
    assert checkpoint["file_count"] == 33
    assert checkpoint["total_bytes"] == 324627486795

    promotion = packet["observed_zero_gpu_promotion"]
    assert promotion["checkpoint_seal"]["state"] == "accepted_terminal_and_file_binding"
    assert promotion["checkpoint_seal"]["pod_uid"] == ("1c7bbdd0-54df-4ace-bd14-47b6268a34bf")
    assert promotion["bf16_export"]["state"] == (
        "accepted_terminal_pending_durable_promotion_receipt"
    )
    assert promotion["bf16_export"]["pod_uid"] == ("f5697810-5062-4793-beb5-e8f573433467")
    assert promotion["bf16_export"]["trained_tensors"] == 1184
    assert promotion["bf16_export"]["restored_frozen_tensors"] == 15
    assert promotion["bf16_export"]["optimizer_steps_executed"] == 0
    assert promotion["cpu_layout_check"]["pod_uid"] == ("d41dd66e-8ed5-45d6-8772-e39779b6861c")
    assert promotion["cpu_layout_check"]["synthetic_only"] is True
    assert promotion["cpu_layout_check"]["gpu_reload_verified"] is False
    assert promotion["cpu_layout_check"]["serving_qualified"] is False
    assert all(item["gpus"] == 0 for item in promotion.values())


def test_launch_remains_closed_until_every_identity_and_capacity_gate_passes() -> None:
    packet = read(PACKET)
    gates = "\n".join(packet["remaining_gates"])
    for required in (
        "durable accepted promotion evidence",
        "one-GPU zero-update finite reload",
        "atomic create-once inference stage",
        "paused zero-GPU serving registration",
        "capacity census",
        "live parity receipt",
        "candidate config compiled",
        "duplicate census",
        "two byte-equivalent Kubernetes server dry-runs",
        "fleet.ai/failure-alerts=off",
    ):
        assert required in gates
    assert packet["future_candidate"]["evaluation_config_file_sha256"] is None
    assert packet["future_candidate"]["evaluation_plan_sha256"] is None
    assert packet["future_candidate"]["route_catalog"] is None
    assert packet["future_candidate"]["route_model_info"] is None
    assert packet["future_candidate"]["route_server_info"] is None


def test_packet_contains_no_private_or_outcome_material() -> None:
    packet = read(PACKET)
    assert not any(packet["privacy"].values())
    assert packet["scientific_boundary"]["capability_claimed"] is False
    assert packet["scientific_boundary"]["final_test_split_access"] == "sealed"
    assert packet["result_contract"]["public_result_and_webapp_update_before_acceptance"] is False
    assert packet["result_contract"]["selective_replay_forbidden"] is True
    assert packet["result_contract"]["valid_zero_is_final"] is True
