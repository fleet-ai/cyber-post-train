import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/evaluation/qwen38-checkpoint-serving-readiness-v1.json"
LEDGER = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json"


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
    assert self_sft["stage"]["status"] == "absent"

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
