"""Offline contract checks for the LR3e-5 continuation operator handoff."""

from __future__ import annotations

import json
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = (
    ROOT / "docs/evidence/qwen38-study/2026-09-14-lr30-resume76-operator-watchdog-handoff-v1.json"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path: Path) -> dict:
    value = read(path)
    assert value["sha256"] == digest_json(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    return value


def test_handoff_is_offline_only_and_not_launch_authorization() -> None:
    handoff = sealed(HANDOFF)
    assert handoff["classification"] == "offline_handoff_not_launch_authorization"
    assert handoff["authority"] == {
        "launchable_now": False,
        "live_gates_satisfied": False,
        "jobs_api_posts_performed": 0,
        "kubernetes_mutations_performed": 0,
        "wandb_runs_created": 0,
        "automation_mutations_performed": 0,
        "document_is_authorization": False,
        "required_before_use": handoff["authority"]["required_before_use"],
    }
    assert len(handoff["live_pre_submit_gates"]) == 8
    assert len(handoff["unresolved_until_live_post"]) == 9


def test_handoff_binds_exact_audit_plan_request_bundle_and_checkpoint() -> None:
    handoff = sealed(HANDOFF)
    candidate = handoff["immutable_candidate"]
    audit_binding = candidate["launch_audit"]
    audit_path = ROOT / audit_binding["path"]
    audit = read(audit_path)

    assert audit_binding["file_sha256"] == file_sha256(audit_path)
    assert audit_binding["embedded_sha256"] == audit["sha256"]
    assert candidate["config"]["file_sha256"] == file_sha256(ROOT / candidate["config"]["path"])
    prepared = audit["offline_prepared_artifact"]
    assert candidate["plan"] == {
        "file_sha256": prepared["plan"]["file_sha256"],
        "canonical_sha256": prepared["plan"]["canonical_sha256"],
    }
    assert candidate["request"] == {
        "file_sha256": prepared["request"]["file_sha256"],
        "canonical_sha256": prepared["request"]["canonical_sha256"],
    }
    assert candidate["prepared_receipt_file_sha256"] == prepared["prepared_receipt"]["file_sha256"]
    assert candidate["runtime_bundle"] == {
        "canonical_gzip_sha256": prepared["runtime_bundle"]["canonical_gzip_sha256"],
        "decompressed_payload_sha256": prepared["runtime_bundle"]["decompressed_payload_sha256"],
        "command_sha256": prepared["command_sha256"],
    }
    source = handoff["source_checkpoint"]
    audited = audit["checkpoint_and_recovery_contract"]
    assert source["path"] == audited["source_checkpoint_path"]
    assert source["source_plan_sha256"] == audited["source_plan_sha256"]
    assert source["manifest_file_sha256"] == audit["source"]["checkpoint_manifest"]["file_sha256"]
    assert (
        source["manifest_receipt_sha256"]
        == audit["source"]["checkpoint_manifest"]["receipt_sha256"]
    )
    assert (source["files"], source["bytes"], source["world_size"]) == (
        audited["source_checkpoint_files"],
        audited["source_checkpoint_bytes"],
        audited["source_world_size"],
    )


def test_monitor_bounds_progress_and_wandb_are_exact() -> None:
    handoff = sealed(HANDOFF)
    audit = read(ROOT / handoff["immutable_candidate"]["launch_audit"]["path"])
    cadence = audit["resume_math_and_cadence"]
    monitor = handoff["monitor_contract"]
    bounds = monitor["runtime_bounds"]
    expected = monitor["expected_progress"]

    assert monitor["external_poll_seconds"] == 600
    assert (
        monitor["runtime_internal_poll_seconds"],
        bounds["startup_seconds"],
        bounds["confirmed_idle_seconds"],
        bounds["hard_runtime_seconds"],
        bounds["checkpoint_drain_seconds_after_hard_bound"],
    ) == (
        cadence["watchdog_poll_seconds"],
        cadence["watchdog_startup_seconds"],
        cadence["watchdog_idle_seconds"],
        cadence["watchdog_hard_seconds"],
        cadence["watchdog_drain_seconds"],
    )
    assert bounds["missing_telemetry_is_idle"] is False
    assert bounds["external_observation_alone_may_cancel"] is False
    assert expected["checkpoint_steps"] == cadence["new_checkpoint_steps"]
    assert (
        expected["expected_retained_checkpoint_steps"]
        == cadence["expected_retained_new_checkpoint_steps"]
    )
    assert expected["new_optimizer_steps"] == cadence["continuation_optimizer_steps"] == 70
    assert expected["first_new_optimizer_step"] == 7
    assert expected["terminal_optimizer_step"] == 76

    wandb = handoff["wandb_contract"]
    assert wandb["run_id"] == wandb["name"] == candidate_name(handoff)
    assert wandb["expected_new_scalar_events"] == 70
    assert wandb["expected_global_step_range"] == [7, 76]
    assert wandb["resume_policy"] == "never"

    scientific = handoff["scientific_contract"]
    audited = audit["scientific_contract"]
    assert scientific["model"] == audited["model"] == "Qwen/Qwen3.8-27B"
    assert scientific["model_revision"] == audited["model_revision"]
    assert (
        scientific["corpus_manifest_sha256"]
        == audit["source"]["corpus_manifest"]["embedded_sha256"]
    )
    assert (
        scientific["dense_windows"],
        scientific["represented_train_tasks"],
        scientific["supervised_tokens"],
        scientific["global_batch_size"],
        scientific["learning_rate"],
        scientific["total_optimizer_steps"],
    ) == (
        audited["dense_windows"],
        audited["represented_train_tasks"],
        audited["supervised_tokens"],
        audited["global_batch_size"],
        audited["learning_rate"],
        cadence["terminal_optimizer_step"],
    )
    assert scientific["reference_cross_entropy_enabled"] is False
    assert scientific["selection_mode"] == "task_outcomes_only"
    assert scientific["webexploitbench_selection_eligible"] is False


def candidate_name(handoff: dict) -> str:
    return handoff["immutable_candidate"]["request_name"]


def test_cluster_release_privacy_and_notification_boundaries_are_fail_closed() -> None:
    handoff = sealed(HANDOFF)
    cluster = handoff["cluster_contract"]
    assert (cluster["cluster"], cluster["priority_class"], cluster["expected_queue_priority"]) == (
        "dev",
        "c1",
        "q1",
    )
    assert cluster["expected_effective_priority"] == 10000
    assert (cluster["workers"], cluster["gpus_per_worker"], cluster["total_gpus"]) == (
        2,
        4,
        8,
    )
    node_cap = (
        "maximum_active_owned_physical_nodes_across_dev_and_prod_excluding_four_inference_endpoints"
    )
    assert cluster[node_cap] == 8
    assert cluster["automatic_requeue"] is False
    assert cluster["duplicate_posts_allowed"] is False
    assert cluster["peer_preemption_allowed"] is False
    assert cluster["production_submission_allowed"] is False

    terminal = handoff["terminal_and_release_contract"]
    assert terminal["training_success_alone_accepts_checkpoint"] is False
    assert terminal["terminal_release_poll_seconds"] == 60
    assert terminal["terminal_release_uncertain_after_seconds"] == 300
    assert terminal["automatic_retry_allowed"] is False
    assert terminal["automatic_requeue_allowed"] is False
    assert terminal["production_promotion_allowed"] is False

    monitor = handoff["monitor_contract"]
    assert len(monitor["material_notifications"]) == 7
    forbidden = " ".join(monitor["forbidden_observations_or_output"])
    for private in ("credentials", "prompts", "traces", "flags", "answers", "private trainer logs"):
        assert private in forbidden
