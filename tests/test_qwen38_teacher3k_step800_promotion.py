from __future__ import annotations

import json
from pathlib import Path

from training import model_stage as base_stage
from training import model_stage_current_base as stage

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/qualification/qwen38-teacher3k32-step800-inference-stage-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k32-step800-preservation-20260922.json"


def test_step800_stage_plan_binds_accepted_zero_update_chain() -> None:
    plan = stage.read_plan(PLAN)
    source = plan["source"]
    desired = plan["desired_registration"]

    assert plan["execution"]["gpus"] == 0
    assert plan["execution"]["priority_class"] == "c1"
    assert plan["destination"]["path"] == "/models/chris-q38-t3k32-s800-v1"
    assert source["export_receipt"]["file_sha256"] == (
        "sha256:f3762da74e131ccf2ff2df5c3e0b54f5352822e98c86539f640b6e3996808970"
    )
    assert source["export_receipt"]["receipt_sha256"] == (
        "sha256:3dca4cf10610bcbaf263e730682ffea9cbe2bd540613b81123c14d087d2e09a9"
    )
    assert source["gpu_check_receipt"]["file_sha256"] == (
        "sha256:48e3381859c138636e7ec815d1d4c131c4bf2ce0f758d80e1dd00430951f92ad"
    )
    assert source["gpu_check_receipt"]["receipt_sha256"] == (
        "sha256:29fe1883f160c29b87f42df8e8d74ead193c8ac25feedd0bffd9fcd09e138de5"
    )
    assert source["payload"]["manifest_sha256"] == (
        "sha256:bd1b01c25d96a5e4f637399282ade26275c2a8b9d0f2a38302b173aff07e3d5b"
    )
    gpu = source["gpu_check_receipt"]["required_fields"]
    assert gpu["finite_logits"] is True
    assert gpu["generated_tokens"] == 2
    assert gpu["source_unchanged"] is True
    assert gpu["optimizer_steps_executed"] == 0
    assert desired["id"] == "chris-q38-t3k32-s800-v1"
    assert desired["spec"]["desiredState"] == "paused"
    assert desired["spec"]["scaling"] == {"minReplicas": 0}
    assert desired["spec"]["placement"]["priorityClassName"] == "c1"
    assert desired["spec"]["model"]["revision"] == source["payload"]["manifest_sha256"]


def test_step800_preservation_evidence_records_cleanup_and_disabled_route() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    plan = stage.read_plan(PLAN)

    assert evidence["sha256"] == base_stage.digest_json(base_stage._unsigned(evidence, "sha256"))
    assert evidence["stage"]["plan_sha256"] == plan["plan_sha256"]
    assert (
        evidence["stage"]["payload_manifest_sha256"]
        == (plan["source"]["payload"]["manifest_sha256"])
    )
    assert evidence["dev_reload"]["source_unchanged"] is True
    assert evidence["dev_reload"]["finite_logits"] is True
    assert evidence["dev_reload"]["optimizer_steps_executed"] == 0
    assert evidence["dev_cleanup"]["all_objects_absent"] is True
    assert evidence["dev_cleanup"]["gpus_after_cleanup"] == 0
    assert evidence["stage"]["pod_and_config_map_absent_after_cleanup"] is True
    registration = evidence["registration"]
    assert registration["post_attempts"] == 1
    assert registration["phase"] == "paused"
    assert registration["active_pods"] == 0
    assert registration["ready_replicas"] == 0
    assert registration["desired_replicas"] == 0
    assert registration["routing_enabled"] is False
    assert registration["serving_qualified"] is False
    assert registration["uid"] is None
    assert registration["uid_readback_supported"] is False
    assert evidence["live_parity"]["state"] == "unaccepted_not_run"
    assert evidence["evaluation_launched"] is False
    assert evidence["production_serving_activated"] is False
