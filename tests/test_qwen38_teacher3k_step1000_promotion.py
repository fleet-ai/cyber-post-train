from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training import model_stage_current_base as stage


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/qualification/qwen38-teacher3k32-step1000-inference-stage-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-teacher3k32-step1000-promotion-20260922.json"


def test_step1000_stage_plan_binds_the_exact_export_and_reload() -> None:
    plan = stage.read_plan(PLAN)
    source = plan["source"]
    desired = plan["desired_registration"]

    assert "sha256:" + hashlib.sha256(PLAN.read_bytes()).hexdigest() == (
        "sha256:8b5debe64294b02ea2d454d0f7eec73a03177cf46975171f29c7ec4fa75fc69e"
    )
    assert plan["plan_sha256"] == (
        "sha256:c8d99519be6703075afaf2ac0254434b82e4448cb72aaf6cea81d514a99cf47a"
    )
    assert source["export_receipt"]["file_sha256"] == (
        "sha256:e79d2cdaaebe52da61a37e4f4ad35c2c8ccd4878afaf77de0fc4796e8e1c6463"
    )
    assert source["gpu_check_receipt"]["file_sha256"] == (
        "sha256:2a19f9c99f712b81604f8b89ba6c498ca89167f2ad778f9d5fb8fa4a0a1c22a0"
    )
    assert source["payload"]["manifest_sha256"] == (
        "sha256:023c5f8b0559ba050f0d672a6bc27aabecec7d5837595f8ea5bc914446d26db5"
    )
    assert source["payload"]["manifest_sha256"] == desired["spec"]["model"]["revision"]
    assert plan["destination"]["path"] == "/models/chris-q38-t3k32-s1000-v1"
    assert plan["execution"]["gpus"] == 0
    assert plan["execution"]["priority_class"] == "c1"
    assert desired["id"] == "chris-q38-t3k32-s1000-v1"
    assert desired["spec"]["desiredState"] == "paused"
    assert desired["spec"]["scaling"] == {"minReplicas": 0}
    assert "s900" not in json.dumps(plan)


def test_step1000_promotion_evidence_is_self_digested_and_paused() -> None:
    value = json.loads(EVIDENCE.read_text())
    unsigned = {key: item for key, item in value.items() if key != "sha256"}

    assert value["sha256"] == hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert value["status"] == "accepted_through_live_serving_parity"
    assert value["stage"]["stage_pod_and_config_map_released"] is True
    assert value["serving_route"] == {
        **value["serving_route"],
        "model_id": "chris-q38-t3k32-s1000-v1",
        "phase": "paused",
        "active_pods": 0,
        "ready_replicas": 0,
        "desired_replicas": 0,
        "routing_enabled": False,
        "post_attempts": 1,
    }
    assert value["evaluation"] == {
        "capability_claimed": False,
        "fleet_heldout_launched": False,
        "webexploitbench_launched": False,
    }
    assert value["activation"]["capacity"]["projected_nodes"] == 7
    assert value["activation"]["route_at_parity"] == {
        "active_pods": 1,
        "desired_replicas": 1,
        "phase": "ready",
        "ready_replicas": 1,
        "resource_version": "33420207",
        "routing_enabled": True,
    }
    assert value["live_parity"]["status"] == "passed"
    assert value["live_parity"]["scores_observed"] is False
