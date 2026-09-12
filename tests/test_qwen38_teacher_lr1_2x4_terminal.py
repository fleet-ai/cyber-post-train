"""Terminal and open-gate contract for the exact LR1 two-by-four dev segment."""

from __future__ import annotations

import json
import math
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-dev-fallback-v1.json"
TRANSITION = ROOT / ("docs/evidence/qwen38-study/2026-09-12-lr1-topology-transition-dev-v1.json")
EVIDENCE = ROOT / ("docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-terminal-v1.json")


def read(path: Path) -> dict:
    return json.loads(path.read_bytes())


def test_terminal_evidence_is_self_digesting_and_bound_to_sources() -> None:
    evidence = read(EVIDENCE)
    assert evidence["sha256"] == digest_json(
        {key: value for key, value in evidence.items() if key != "sha256"}
    )
    source = evidence["source"]
    assert source["config_file_sha256"] == file_sha256(CONFIG)
    assert source["config_commit"] == "97d761d8f5fcb3fa59ba2bbd036e11e911c4a4f5"
    transition = read(TRANSITION)
    binding = source["topology_transition_evidence"]
    assert binding["file_sha256"] == file_sha256(TRANSITION)
    assert binding["embedded_sha256"] == transition["sha256"]
    assert transition["successor"]["rayjob_uid"] == evidence["run"]["rayjob_uid"]


def test_exact_controller_completed_and_released_all_gpu_resources() -> None:
    evidence = read(EVIDENCE)
    run = evidence["run"]
    assert {
        key: run[key]
        for key in (
            "api_name",
            "api_run_id",
            "api_job_id",
            "rayjob_uid",
            "workload_uid",
            "raycluster_uid",
            "head_pod_uid",
            "worker_pod_uid",
        )
    } == {
        "api_name": "chris-q38-ta8-lr1-2x4-dev-v1-0a67736f",
        "api_run_id": "0a67736f-f40b-4b0c-b4ae-fe494f97235b",
        "api_job_id": "chris-q38-ta8-lr1-2x4-dev-v1-0a67736f-9gndd",
        "rayjob_uid": "70161760-16ae-4f8e-8809-e6a024a16188",
        "workload_uid": "f2ee4249-2554-4433-a6c3-53140433f6b8",
        "raycluster_uid": "08369b1f-956b-46fb-bf5a-aed9a450c180",
        "head_pod_uid": "524ab333-11d9-4dc1-9643-b70595de0d20",
        "worker_pod_uid": "56e3a6b5-d2ad-45c8-84be-30a4f77a9880",
    }
    assert run["topology"] == {"workers": 2, "gpus_per_worker": 4, "world_size": 8}
    assert run["priority_class"] == "c1"
    assert run["queue_priority_class"] == "q1"
    assert run["effective_priority"] == 10000
    assert run["automatic_requeue"] is False

    terminal = evidence["terminal_controller_reconciliation"]
    assert terminal["api_status"] == terminal["rayjob_status"] == "SUCCEEDED"
    assert terminal["rayjob_deployment_status"] == "Complete"
    assert terminal["rayjob_failed_count"] == 0
    assert terminal["rayjob_succeeded_count"] == 1
    assert terminal["workload_finished"] is True
    assert terminal["workload_terminal_reason"] == "Succeeded"
    assert terminal["workload_quota_reserved"] is terminal["workload_admitted"] is False
    assert terminal["raycluster_absent"] is True
    assert terminal["head_pod_absent"] is terminal["worker_pod_absent"] is True
    assert terminal["released_gpus"] == 8
    assert terminal["gpus_held"] == 0


def test_six_optimizer_steps_and_exact_tracking_parity_are_proven() -> None:
    evidence = read(EVIDENCE)
    paused = evidence["receipts"]["training_paused"]
    assert paused["status"] == "training_paused"
    assert paused["optimizer_step"] == paused["optimizer_steps_executed"] == 6
    assert paused["planned_optimizer_steps"] == 76
    assert paused["export_status"] == "pending_separate_zero_step_export"
    assert paused["validation_mode"] == "task_outcomes_only"
    assert paused["reference_cross_entropy_enabled"] is False

    receipts = evidence["receipts"]
    assert all(item["embedded_receipt_digest_valid"] for item in receipts.values())
    assert receipts["started"]["file_sha256"] == (
        "sha256:e5b811aeb64444214b166d411a1ef749611980bc10dec15e42d33a7dc4d0bf64"
    )
    assert receipts["training_paused"]["file_sha256"] == (
        "sha256:1d8b65ab5e687adfaa23cc976287b2117d3d1aae5386dccdbebf9a6a024a6a21"
    )
    assert receipts["checkpoint_step_6"]["file_sha256"] == (
        "sha256:e2db2067685d2875dde79ba36c00ac27624122833d4f4106851517ab61fa1ce8"
    )
    assert receipts["progress"]["file_sha256"] == (
        "sha256:aa84b2ffabf0a205cad8ad324c300479a131c9e8f214aada0af8a8e1900c8332"
    )

    training = evidence["bounded_training"]
    assert training["optimizer_steps"] == list(range(1, 7))
    assert len(training["train_loss"]) == len(training["optimizer_steps"]) == 6
    assert all(math.isfinite(value) for value in training["train_loss"])
    assert training["all_losses_finite"] is True
    assert training["all_gradient_norms_finite"] is True
    assert training["terminal_total_supervised_tokens"] == 70862
    assert training["full_planned_optimizer_steps"] == 76
    assert training["full_arm_completed"] is False

    wandb = evidence["wandb"]
    assert wandb["tracking_status"] == "synced"
    assert wandb["acceptance_ready"] is True
    assert wandb["remote_state"] == "finished"
    assert wandb["local_metric_events"] == wandb["remote_metric_events"] == 6
    assert wandb["allowlisted_scalar_fields_compared"] == 8
    assert wandb["local_remote_exact_parity"] is True


def test_checkpoint_remains_unsealed_unreloaded_and_unaccepted() -> None:
    evidence = read(EVIDENCE)
    checkpoint = evidence["checkpoint"]
    assert checkpoint["optimizer_step"] == checkpoint["latest_pointer"]["value"] == 6
    assert checkpoint["world_size"] == evidence["run"]["topology"]["world_size"] == 8
    assert checkpoint["files"] == 33
    assert checkpoint["bytes"] == 324627486731
    assert checkpoint["receipt_digest_valid"] is True
    assert checkpoint["sealed_manifest_present"] is False
    assert checkpoint["gpu_reload_verified"] is False
    assert checkpoint["accepted_for_resume"] is False

    qualification = evidence["qualification"]
    assert qualification["clean_controller_terminal"] is True
    assert qualification["bounded_optimizer_execution"] is True
    assert qualification["tracking_parity"] is True
    assert qualification["checkpoint_created"] is True
    assert qualification["checkpoint_integrity_seal"] is False
    assert qualification["checkpoint_zero_update_reload"] is False
    assert qualification["exact_two_by_four_dev_segment_accepted"] is False
    assert qualification["production_training_authorized"] is False
    assert qualification["hyperparameter_selected"] is False
    assert qualification["capability_claim_allowed"] is False


def test_topology_and_terminal_pod_provenance_limits_are_explicit() -> None:
    evidence = read(EVIDENCE)
    scope = evidence["topology_scope"]
    assert scope["executed_layout"] == "two_workers_by_four_gpus"
    assert scope["original_lr_v2_layout_gate"] == "one_worker_by_eight_gpus"
    assert scope["same_world_size"] is True
    assert scope["literal_one_by_eight_gate_satisfied"] is False
    assert scope["two_by_four_operational_support_only"] is True

    provenance = evidence["pod_provenance"]
    assert provenance["ready_observation"] == {
        "head_pod_restarts": 0,
        "worker_pod_restarts": 0,
        "requested_image_observed": True,
    }
    terminal = provenance["terminal_observation"]
    assert terminal["pods_ttl_cleaned_before_capture"] is True
    assert terminal["head_pod_final_restart_count"] is None
    assert terminal["worker_pod_final_restart_count"] is None
    assert terminal["head_pod_exit_code"] is None
    assert terminal["worker_pod_exit_code"] is None
    assert provenance["zero_restarts_for_full_lifecycle_claimed"] is False


def test_audit_helper_is_gone_and_authoring_was_read_only() -> None:
    evidence = read(EVIDENCE)
    audit = evidence["independent_terminal_audit"]
    assert audit["helper_pod_uid"] == "dcab8cd3-901c-4029-a60f-41e89ad55852"
    assert audit["gpu_request"] == 0
    assert audit["jobs_api_get_status"] == "SUCCEEDED"
    assert audit["jobs_api_output_root_bound"] is True
    assert audit["deleted_after_audit"] is audit["confirmed_absent"] is True
    assert audit["private_logs_read"] is False
    assert audit["raw_weights_read"] is False
    assert audit["prompts_traces_flags_answers_or_scores_read"] is False

    boundaries = evidence["this_evidence_authoring_task_boundaries"]
    assert not any(
        boundaries[key]
        for key in (
            "jobs_api_calls",
            "kubernetes_calls",
            "cluster_mutations",
            "submissions",
            "cancellations",
            "operator_files_modified",
            "credentials_read_or_persisted",
        )
    )
