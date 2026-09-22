from __future__ import annotations

import json
from pathlib import Path

from training import model_stage as base_stage
from training import model_stage_current_base as stage

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/qualification/qwen38-teacher3k32-step900-inference-stage-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k32-step900-preservation-20260922.json"


def test_step900_stage_plan_binds_the_exact_accepted_zero_update_chain() -> None:
    plan = stage.read_plan(PLAN)
    source = plan["source"]
    desired = plan["desired_registration"]

    assert plan["execution"]["gpus"] == 0
    assert plan["execution"]["priority_class"] == "c1"
    assert plan["destination"]["path"] == "/models/chris-q38-t3k32-s900-v1"
    assert "s800" not in json.dumps(plan)
    assert source["export_receipt"]["file_sha256"] == (
        "sha256:84b8d965fe08f988fdab6fbe3a3d9c5153af392f75233b3224ff579998fd069f"
    )
    assert source["export_receipt"]["receipt_sha256"] == (
        "sha256:33fab68ad59aab8ff24753c2edbb81371a94b99e3d0284fc29e200e0a664f7e6"
    )
    assert source["gpu_check_receipt"]["file_sha256"] == (
        "sha256:da73cfddf4b2a01b3b516cc040dad4b9b86b06ccd484c09fcb907c15940ed6f9"
    )
    assert source["gpu_check_receipt"]["receipt_sha256"] == (
        "sha256:93a7c4d90afe22f11ebb20d15afe2514078ee3f2f0802801348773ee3c1fdad3"
    )
    assert source["payload"]["manifest_sha256"] == (
        "sha256:d7f3e596b3b54736ece2f7d1c67c220ea19c27ba47defedd923e61c48ca2df73"
    )
    gpu = source["gpu_check_receipt"]["required_fields"]
    assert gpu["checker_sha256"] == (
        "a04811409178eedc6969e34766ca70c82d84c27b718944ecab93407609c4dfe7"
    )
    assert gpu["finite_logits"] is True
    assert gpu["generated_tokens"] == 2
    assert gpu["source_unchanged"] is True
    assert gpu["optimizer_steps_executed"] == 0
    assert desired["id"] == "chris-q38-t3k32-s900-v1"
    assert desired["spec"]["desiredState"] == "paused"
    assert desired["spec"]["scaling"] == {"minReplicas": 0}
    assert desired["spec"]["placement"]["priorityClassName"] == "c1"
    assert desired["spec"]["model"]["revision"] == source["payload"]["manifest_sha256"]


def test_step900_evidence_proves_cleanup_and_keeps_live_evaluation_closed() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    plan = stage.read_plan(PLAN)

    assert evidence["sha256"] == base_stage.digest_json(base_stage._unsigned(evidence, "sha256"))
    assert evidence["checkpoint"]["optimizer_step"] == 900
    assert evidence["checkpoint"]["source_plan_sha256"] == (
        "sha256:8488f03a65e3d158a46697eefda2751736681e3a503c26ee9c39c63fb130679c"
    )
    assert evidence["checkpoint"]["seal"]["status"] == "accepted_terminal_and_file_binding"
    assert evidence["export"]["optimizer_steps_executed"] == 0
    assert evidence["export"]["cpu_layout_check"]["status"] == "passed"
    assert evidence["dev_reload"]["status"] == "accepted_bounded_zero_update_reload"
    assert evidence["dev_reload"]["finite_logits"] is True
    assert evidence["dev_reload"]["optimizer_steps_executed"] == 0
    assert evidence["dev_cleanup"]["all_objects_absent"] is True
    assert evidence["dev_cleanup"]["gpus_after_cleanup"] == 0
    assert evidence["stage"]["plan_sha256"] == plan["plan_sha256"]
    assert evidence["stage"]["payload_manifest_sha256"] == (
        plan["source"]["payload"]["manifest_sha256"]
    )
    assert evidence["stage"]["pod_and_config_map_absent_after_cleanup"] is True
    registration = evidence["registration"]
    assert registration["post_attempts"] == 1
    assert registration["phase"] == "paused"
    assert registration["active_pods"] == 0
    assert registration["ready_replicas"] == 0
    assert registration["desired_replicas"] == 0
    assert registration["routing_enabled"] is False
    assert registration["serving_qualified"] is False
    assert registration["reconciled_after_create"] is True
    assert registration["server_defaulted_scaling_replicas"] == 1
    assert registration["matching_kubernetes_pods"] == 0
    assert evidence["live_parity"]["state"] == "unaccepted_not_run"
    assert evidence["evaluation_launched"] is False
    assert evidence["production_serving_activated"] is False
    assert evidence["scientific_boundary"] == {
        "capability_claim": False,
        "exposure_matched_comparison": False,
        "external_benchmark_content_or_outcomes_used": False,
        "optimizer_updates": 0,
    }
