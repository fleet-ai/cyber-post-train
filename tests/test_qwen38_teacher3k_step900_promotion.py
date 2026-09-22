from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import model_stage as base_stage
from training import model_stage_current_base as stage

ROOT = Path(__file__).resolve().parents[1]
PACKET = (
    ROOT / "configs/evaluation/qwen38-teacher3k-32k-step900-fleet-dev17-seed43-preparation-v1.json"
)
PLAN = ROOT / "configs/qualification/qwen38-teacher3k32-step900-inference-stage-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k32-step900-preservation-20260922.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_step900_stage_plan_binds_the_accepted_export_and_reload() -> None:
    plan = stage.read_plan(PLAN)
    source = plan["source"]
    desired = plan["desired_registration"]

    assert file_sha256(PLAN) == (
        "sha256:95fc77249f839da58b30f1c7ef1204dcd005f898c153aced713c52fee2b799b3"
    )
    assert plan["plan_sha256"] == (
        "sha256:27c15422d77ebf293007173e9e36b73f931cbcbd0ae109b23b4d1b81243fcdd6"
    )
    assert plan["execution"]["gpus"] == 0
    assert plan["execution"]["priority_class"] == "c1"
    assert plan["destination"]["path"] == "/models/chris-q38-t3k32-s900-v1"
    assert source["export_receipt"]["file_sha256"] == (
        "sha256:84b8d965fe08f988fdab6fbe3a3d9c5153af392f75233b3224ff579998fd069f"
    )
    assert source["gpu_check_receipt"]["file_sha256"] == (
        "sha256:da73cfddf4b2a01b3b516cc040dad4b9b86b06ccd484c09fcb907c15940ed6f9"
    )
    assert source["payload"]["manifest_sha256"] == desired["spec"]["model"]["revision"]
    assert desired["spec"]["desiredState"] == "paused"
    assert desired["spec"]["scaling"] == {"minReplicas": 0}
    assert "s800" not in json.dumps(plan)


def test_step900_receipt_cross_binds_the_merged_packet_and_closed_gates() -> None:
    evidence, packet = read(EVIDENCE), read(PACKET)
    accepted = packet["accepted_stage_and_registration"]

    assert evidence["sha256"] == base_stage.digest_json(base_stage._unsigned(evidence, "sha256"))
    assert evidence["source_eval_packet"]["file_sha256"] == file_sha256(PACKET)
    assert evidence["source_eval_packet"]["sha256"] == "sha256:" + packet["sha256"]
    assert packet["launchable"] is evidence["source_eval_packet"]["launchable"] is False

    durable = evidence["durable_acceptances"]
    promotion = packet["observed_zero_gpu_promotion"]
    for evidence_name, packet_name in (
        ("bf16_export", "bf16_export"),
        ("cpu_layout_check", "cpu_layout_check"),
    ):
        row, observed = durable[evidence_name], promotion[packet_name]
        assert row["file_sha256"] == observed["receipt_file_sha256"]
        assert row["receipt_sha256"] == observed["receipt_sha256"]
        assert row["state"] == "accepted_terminal"

    reload = packet["observed_dev_gpu_reload"]
    assert durable["dev_gpu_reload"]["file_sha256"] == reload["receipt_file_sha256"]
    assert durable["dev_gpu_reload"]["receipt_sha256"] == reload["receipt_sha256"]
    assert durable["dev_gpu_reload"]["cleanup_pod_uid"] == reload["cleanup_pod_uid"]
    assert (
        durable["dev_gpu_reload"]["final_dev_census_sha256"]
        == (reload["final_dev_capacity_census"]["sha256"])
    )
    assert durable["dev_gpu_reload"]["preflight_census"] == {
        "current_gpus": 0,
        "current_nodes": 0,
        "observed_at": "2026-09-22T17:21:34Z",
        "projected_gpus": 1,
        "projected_nodes": 1,
        "sha256": "sha256:6012379d5461c897b42a4f2eaa574307b8f83ecb46d0c9783869ce45573bbcc4",
    }
    assert durable["dev_gpu_reload"]["optimizer_steps_executed"] == 0

    stage_receipt = durable["stage"]
    observed_stage = accepted["stage"]
    assert evidence["stage_plan"]["plan_sha256"] == accepted["stage_plan_sha256"]
    assert stage_receipt["file_sha256"] == observed_stage["acceptance_file_sha256"]
    assert stage_receipt["receipt_sha256"] == observed_stage["acceptance_receipt_sha256"]
    assert stage_receipt["pod_uid"] == observed_stage["pod_uid"]
    assert stage_receipt["config_map_uid"] == observed_stage["config_map_uid"]
    assert stage_receipt["pod_and_config_map_absent_after_cleanup"] is True

    registration = durable["paused_registration"]
    observed_registration = accepted["registration"]
    assert (
        registration["registration_sha256"]
        == (observed_registration["desired_registration_sha256"])
    )
    assert registration["post_attempts"] == observed_registration["post_attempts"] == 1
    assert registration["immediate_get_after_create"] == "not_found_no_replay"
    assert registration["reconciled_after_create"] is True
    assert registration["phase"] == observed_registration["phase"] == "paused"
    assert registration["desired_replicas"] == registration["ready_replicas"] == 0
    assert registration["active_pods"] == registration["matching_kubernetes_pods"] == 0
    assert observed_registration["matching_kubernetes_pods"] == []
    assert registration["routing_enabled"] is False
    assert "replicas" not in read(PLAN)["desired_registration"]["spec"]["scaling"]
    assert registration["server_default_delta"] == {
        "observed": 1,
        "path": "spec.scaling.replicas",
        "requested": "absent",
    }

    assert evidence["next_gates"] == {
        "evaluation": "unlaunched",
        "live_parity": "unaccepted_not_run",
        "route_activation": "unlaunched",
    }
    assert evidence["scientific_boundary"]["optimizer_updates"] == 0
    assert evidence["scientific_boundary"]["capability_claim"] is False
    assert not any(evidence["privacy"].values())
