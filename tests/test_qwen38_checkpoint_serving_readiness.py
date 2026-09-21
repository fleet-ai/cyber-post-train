import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/evaluation/qwen38-checkpoint-serving-readiness-v1.json"
LEDGER = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json"
LONG_CONTEXT_STEP1 = ROOT / "docs/evidence/qwen38-teacher3k-64k-v3-first-update-20260921.json"


def _digest(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("sha256")
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_readiness_matrix_is_self_digesting_and_complete() -> None:
    matrix = json.loads(MATRIX.read_text())
    assert matrix["sha256"] == _digest(matrix)

    artifacts = {row["artifact_id"]: row for row in matrix["artifacts"]}
    assert set(artifacts) == {
        "q38-fresh75-step230",
        "q38-teacher-dense-v5-step186",
        "q38-self-sft-step44",
        "q38-available-a-lr30-step76",
    }

    fresh = artifacts["q38-fresh75-step230"]
    assert fresh["live_target"]["status"] == "ready_requires_fresh_parity"
    assert fresh["live_target"]["model_revision"] == fresh["export"]["payload_manifest_sha256"]
    assert fresh["live_target"]["openai_route_header"] == {
        "name": "x-ai-eg-model",
        "value": fresh["live_target"]["model_id"],
    }
    assert "requires_fresh_matched_parity" in fresh["web_readiness"]

    teacher = artifacts["q38-teacher-dense-v5-step186"]
    assert teacher["registration"]["status"] == "paused"
    assert teacher["registration"]["active_pods"] == 0
    assert teacher["registration"]["gpus_held"] == 0

    self_sft = artifacts["q38-self-sft-step44"]
    assert self_sft["reload"]["status"] == "accepted_one_gpu_forward"
    assert self_sft["stage"]["status"] == "accepted"
    assert self_sft["stage"]["resources_released"] is True
    assert self_sft["registration"]["status"] == "accepted_paused_zero_active_replicas"
    assert self_sft["registration"]["phase"] == "paused"
    assert self_sft["registration"]["active_pods"] == 0
    assert self_sft["registration"]["gpus_allocated"] == 0

    lr30 = artifacts["q38-available-a-lr30-step76"]
    assert lr30["checkpoint_reload"]["status"] == "accepted_zero_update_all_rank_reload"
    assert lr30["export_reload"]["status"] == "missing_one_gpu_inference_forward_receipt"


def test_readiness_matrix_covers_every_accepted_ledger_checkpoint() -> None:
    matrix = json.loads(MATRIX.read_text())
    ledger = json.loads(LEDGER.read_text())
    matrix_ids = {row["artifact_id"] for row in matrix["artifacts"]}
    ledger_ids = {row["artifact_id"] for row in ledger["accepted_checkpoints"]}
    assert matrix_ids == ledger_ids


def test_matrix_never_claims_capability_or_embeds_credentials() -> None:
    matrix = json.loads(MATRIX.read_text())
    assert matrix["scientific_boundary"]["capability_claimed"] is False
    assert matrix["scientific_boundary"]["evaluation_launched_by_this_audit"] is False
    assert matrix["privacy"]["credentials_included"] is False
    serialized = MATRIX.read_text().lower()
    assert "api_key" not in serialized
    assert "bearer " not in serialized


def test_64k_first_update_is_bound_to_both_future_eval_lanes() -> None:
    ledger = json.loads(LEDGER.read_text())
    candidates = {row["artifact_id"]: row for row in ledger["in_progress_candidates"]}
    row = candidates["q38-full-weight-broad-64k-b8-lr3e-6"]

    assert row["run_name"] == "chris-q38-t3k64-b8-v3-fc6b1fbe"
    assert row["status"].startswith("successor_v3_running_first_finite_optimizer_update")
    assert row["run_identity"]["rayjob_uid"] == "545bd9db-479f-4064-b61b-40c8c4a491fe"
    assert row["run_identity"]["failure_alerts"] == "off"
    assert row["run_identity"]["priority_class"] == "c1"

    update = row["first_finite_update"]
    assert update["optimizer_step"] == 1
    assert update["all_numeric_metrics_finite"] is True
    assert update["progress_receipt_sha256"].startswith("sha256:")
    assert update["metrics_file_sha256"].startswith("sha256:")

    route = row["checkpoint_and_evaluation_route"]
    assert route["checkpoint_interval"] == 275
    assert route["planned_final_step"] == 1120
    assert route["status"] == "waiting_for_first_reloadable_checkpoint"
    assert route["fleet_holdout"] == {
        "task_set": "frozen_dev17",
        "harness": "opencode",
        "status": "not_launched_waiting_for_checkpoint",
    }
    assert route["webexploitbench"] == {
        "harness": "opencode",
        "rollout_and_scoring_separated": True,
        "matched_base_control_required": True,
        "status": "not_launched_waiting_for_checkpoint",
    }

    predecessor = row["predecessor_terminal"]
    assert predecessor["resources_released"] is True
    assert predecessor["capability_result"] is False

    evidence = json.loads(LONG_CONTEXT_STEP1.read_text())
    receipt = evidence.pop("receipt_sha256")
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    assert receipt == "sha256:" + hashlib.sha256(encoded).hexdigest()
    assert evidence["run"]["rayjob_uid"] == row["run_identity"]["rayjob_uid"]
    assert (
        evidence["runtime_evidence"]["progress_receipt_sha256"] == update["progress_receipt_sha256"]
    )
    assert evidence["scientific_boundary"]["capability_claimed"] is False
