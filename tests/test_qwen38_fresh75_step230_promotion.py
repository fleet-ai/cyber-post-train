"""Offline contract for the Fresh75 final-checkpoint promotion chain."""

from __future__ import annotations

import json
import re
from pathlib import Path

from training.io import digest_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/qualification/qwen38-fresh75-step230-promotion-v1.template.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-fresh75-step230-promotion-prepared-20260915.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unsigned(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "sha256"}


def test_plan_and_evidence_are_self_digesting_and_inert() -> None:
    plan, evidence = read(PLAN), read(EVIDENCE)
    assert plan["sha256"] == digest_json(unsigned(plan))
    assert evidence["sha256"] == digest_json(unsigned(evidence))
    assert evidence["promotion_template"] == {
        "path": str(PLAN.relative_to(ROOT)),
        "file_sha256": file_sha256(PLAN),
        "embedded_sha256": plan["sha256"],
    }
    assert plan["status"] == "blocked_waiting_export_receipt"
    assert plan["launchable"] is False
    assert plan["cpu_export_check"]["launchable"] is False
    assert plan["one_gpu_reload"]["launchable"] is False
    for record in (plan["execution_record"], evidence["operation"]):
        assert record["live_resource_api_calls_performed"] == 0
        assert record["cluster_calls_performed"] == 0
        assert record["jobs_submitted"] == 0
        assert record["resources_created"] == 0
        assert record["repository_fetch_performed"] is True


def test_exact_terminal_checkpoint_is_bound() -> None:
    plan = read(PLAN)
    source, checkpoint = plan["source"], plan["source"]["checkpoint"]
    config = ROOT / source["training_config"]["path"]
    assert source["training_config"]["file_sha256"] == file_sha256(config)
    assert source["jobs_api_status"] == "SUCCEEDED"
    assert source["optimizer_steps_planned"] == source["optimizer_steps_executed"] == 230
    assert source["supervised_tokens"] == 2_072_122
    assert checkpoint == {
        "path": "/mnt/sfs/jobs/chris-q38-f75-max-full-v4/checkpoints/global_step_230",
        "manifest_path": (
            "/mnt/sfs/jobs/chris-q38-f75-max-full-v4/checkpoint-seals-step230-v1/step-230.json"
        ),
        "manifest_file_sha256": (
            "sha256:bf92e29d9f19dd589201214f25fe8d0e3b0f65de14ec08fc573217f50580a486"
        ),
        "manifest_receipt_sha256": (
            "sha256:9542502631b47cd730ccecb5938650103ed999bf4a4a6cf7d16d9be09c317cab"
        ),
        "optimizer_step": 230,
        "world_size": 8,
        "files": 33,
        "bytes": 324_627_486_795,
        "full_payload_rehash_accepted": True,
    }


def test_export_checks_are_create_once_zero_update_and_ordered() -> None:
    plan = read(PLAN)
    export, cpu, gpu = plan["export"], plan["cpu_export_check"], plan["one_gpu_reload"]
    assert export["create_once"] is True and export["gpu_count"] == 0
    assert export["expected"]["optimizer_step"] == 230
    assert export["expected"]["optimizer_steps_executed"] == 0
    assert (
        export["expected"]["source_checkpoint_receipt_sha256"]
        == (plan["source"]["checkpoint"]["manifest_receipt_sha256"])
    )
    assert (
        export["expected"]["source_manifest_file_sha256"]
        == (plan["source"]["checkpoint"]["manifest_file_sha256"])
    )
    assert export["expected"]["model_revision"] == plan["source"]["model_revision"]
    assert export["expected"]["dtype"] == "BF16"
    assert export["expected"]["tensor_count"] == 1199
    assert cpu["kind"] == "bounded_zero_gpu_pod"
    assert cpu["priority_class"] == "c1" and cpu["gpu_count"] == 0
    assert cpu["restart_policy"] == "Never" and cpu["active_deadline_seconds"] <= 1800
    assert cpu["command_template"][1] == "checkpoint-check"
    assert "<EXPORT_FILE_SHA256>" in cpu["command_template"]
    assert cpu["accept"]["optimizer_steps_executed"] == 0
    request = gpu["jobs_api_request_template"]
    assert request["workers"] == request["gpus_per_worker"] == 1
    assert request["priority_class"] == "c1"
    assert "queue_priority_class" not in request
    assert gpu["expected_queue_priority_class"] == "q1"
    assert gpu["request_must_bundle_exact_checked_runtime_sources"] is True
    assert request["requeueIfPreempted"] is False
    assert request["command_template"][-1] == "--gpu"
    assert gpu["accept"]["optimizer_steps_executed"] == 0
    assert gpu["accept"]["gpu_reload_verified"] is True
    assert "absent" in gpu["terminal_gate"]


def test_serving_is_exact_base_clone_and_dev_is_ephemeral() -> None:
    serving = read(PLAN)["serving_handoff"]
    assert serving["model_id"] == "chris-q38-fresh75-step230-v1"
    assert serving["base_model_id"] == "qwen3.8-27b"
    assert serving["required_execution_contract_sha256"] == (
        "sha256:6092f664d93d2ff8037b5834135727759b9dd03e4b631bb563b5f4c5984d3b65"
    )
    assert serving["precision"] == "bf16" and serving["quantization"] == "none"
    assert serving["max_context_size"] == 262_144
    assert serving["priority_class"] == "c1"
    assert serving["expected_queue_priority_class"] == "q1"
    assert serving["c0_or_q0_allowed"] is False
    dev = serving["dev_qualification"]
    assert dev["purpose"] == "temporary serving test only"
    assert dev["persistent_endpoint_allowed"] is False
    assert 0 < dev["maximum_runtime_seconds"] <= 5400
    assert dev["release_required_on_success_or_failure"] is True
    assert dev["absence_must_be_verified_before_production_registration"] is True
    assert serving["only_permitted_arm_difference"] == "exact weight manifest"


def test_external_benchmark_cannot_select_or_change_checkpoint() -> None:
    firewall = read(PLAN)["evaluation_firewall"]
    assert firewall["checkpoint_selection_surface"] == "frozen Fleet development split only"
    assert firewall["webexploitbench_role"] == "sealed external reporting only"
    assert firewall["webexploitbench_harness"] == "OpenCode"
    assert firewall["webexploitbench_may_select_checkpoint"] is False
    assert firewall["webexploitbench_may_change_training_recipe"] is False
    assert firewall["fleet_final_test_may_select_checkpoint"] is False


def test_public_artifacts_contain_no_secret_or_private_payload() -> None:
    text = PLAN.read_text(encoding="utf-8") + EVIDENCE.read_text(encoding="utf-8")
    assert re.search(r"\bsk_[A-Za-z0-9_-]{20,}", text) is None
    assert re.search(r"\btl_apiKey_[A-Za-z0-9_-]{20,}", text) is None
    assert "FLAG{" not in text
    for forbidden in (
        "prompt_text",
        "trace_text",
        "answer_text",
        "credential_value",
        "score_value",
    ):
        assert forbidden not in text
