"""Offline contracts for the current non-lead Qwen3.8 promotion evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-current-nonlead-promotion-20260922.json"


def _read() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _self_digest(value: dict) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_evidence_is_self_digesting_and_records_the_scientific_boundary() -> None:
    evidence = _read()

    assert evidence["sha256"] == _self_digest(evidence)
    assert evidence["scientific_boundary"] == {
        "evaluation_launched": False,
        "exposure_matched_across_arms": False,
        "external_benchmark_content_or_outcomes_used": False,
        "optimizer_updates": 0,
        "serving_activated": False,
    }


def test_step195_chain_is_zero_update_staged_and_paused() -> None:
    arm = _read()["t3k64_step195"]

    assert arm["checkpoint"]["supervised_tokens"] == 10_561_372
    assert arm["gpu_reload"]["optimizer_steps_executed"] == 0
    assert arm["gpu_reload"]["finite_logits"] is True
    assert arm["gpu_reload"]["workload_pod_deleted"] is True
    assert arm["stage"]["source_gpu_reload_verified"] is True
    assert arm["stage"]["source_optimizer_steps_executed"] == 0
    assert arm["stage"]["payload_file_count"] == arm["export"]["payload_file_count"]
    assert arm["stage"]["payload_total_bytes"] == arm["export"]["payload_total_bytes"]
    assert (
        arm["stage"]["payload_manifest_sha256"].removeprefix("sha256:")
        == (arm["export"]["payload_manifest_sha256"])
    )
    assert arm["stage"]["pod_and_config_map_absent_after_cleanup"] is True
    assert arm["registration"]["phase"] == "paused"
    assert arm["registration"]["desired_replicas"] == 0
    assert arm["registration"]["ready_replicas"] == 0
    assert arm["registration"]["active_pods"] == 0
    assert arm["registration"]["endpoint_ready_addresses"] == 0
    assert arm["registration"]["http_route_count"] == 0
    assert arm["registration"]["routing_enabled"] is False


def test_step50_reload_is_accepted_with_truthful_precreate_caveat() -> None:
    arm = _read()["t3k96_step50"]
    reload = arm["gpu_reload"]
    caveat = arm["precreate_evidence_caveat"]

    assert arm["checkpoint"]["supervised_tokens"] == 3_323_952
    assert reload["finite_logits"] is True
    assert reload["generated_tokens"] == 2
    assert reload["optimizer_steps_executed"] == 0
    assert reload["source_unchanged"] is True
    assert reload["serving_qualified"] is False
    assert reload["workload_pod_deleted"] is True
    assert reload["private_leaf_absent_after_cleanup"] is True
    assert arm["staged_or_registered"] is False

    assert caveat["functional_reload_receipt_accepted"] is True
    assert caveat["capacity_precreate_contract_fully_satisfied"] is False
    assert caveat["canonical_fleet_ai_run_name_label_present"] is False
    assert caveat["full_repo_live_capacity_census_before_create"] is False
    assert caveat["successor_guard_required_for_future_reload"] is True
    assert (
        caveat["successor_guard_merge_commit"]
        == (_read()["repository"]["dev_gpu_guard_merge_commit"])
    )
