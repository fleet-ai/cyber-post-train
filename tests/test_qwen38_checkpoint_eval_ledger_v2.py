from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
READINESS_V1 = ROOT / "configs/evaluation/qwen38-checkpoint-serving-readiness-v1.json"
READINESS_V2 = ROOT / "configs/evaluation/qwen38-checkpoint-serving-readiness-v2.json"
LEDGER_V1 = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v1.json"
LEDGER_V2 = ROOT / "configs/evaluation/qwen38-checkpoint-eval-ledger-v2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _self_digest(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("sha256")
    return "sha256:" + hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_v2_versions_instead_of_mutating_the_historical_readiness_snapshot() -> None:
    v1 = _load(READINESS_V1)
    v2 = _load(READINESS_V2)

    assert v1["schema"] == "cyber_qwen38_checkpoint_serving_readiness_v1"
    assert v1["sha256"] == _self_digest(v1)
    assert v2["schema"] == "cyber_qwen38_checkpoint_serving_readiness_v2"
    assert v2["sha256"] == _self_digest(v2)
    assert v2["supersedes"] == {
        "path": str(READINESS_V1.relative_to(ROOT)),
        "file_sha256": _sha256(READINESS_V1),
        "matrix_sha256": v1["sha256"],
        "reason": (
            "v1_is_a_frozen_source_snapshot; v2 adds later sanitized LR30 stage, paused-route, "
            "live-parity, and score-blind Fleet reconciliation evidence without changing v1."
        ),
    }


def test_lr30_v2_binds_later_reload_stage_and_paused_route_evidence() -> None:
    artifacts = {row["artifact_id"]: row for row in _load(READINESS_V2)["artifacts"]}
    lr30 = artifacts["q38-available-a-lr30-step76"]

    assert lr30["checkpoint"]["status"] == "accepted"
    assert lr30["checkpoint_reload"]["status"] == "accepted_zero_update_all_rank_reload"
    assert lr30["export"]["status"] == "accepted"
    assert lr30["export_reload"] == {
        "evidence": "docs/evidence/qwen38-lr30-step76-inference-stage-v1-accepted-20260921.json",
        "evidence_receipt_sha256": (
            "sha256:11ea26c25edab362e5a11010640e12f2e4495fb1b6ae7224138b1f7647f42568"
        ),
        "status": "accepted_one_gpu_finite_forward",
        "gpu_reload_verified": True,
        "optimizer_steps_executed": 0,
        "source_unchanged": True,
        "resources_released": True,
        "synthetic_only": True,
    }
    assert lr30["stage"]["status"] == "accepted"
    assert lr30["stage"]["payload_manifest_sha256"] == (
        "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8"
    )
    assert lr30["registration"] == {
        "status": "accepted_paused_zero_active_replicas",
        "model_id": "chris-q38-lr30-step76-web-v1",
        "inference_model_uid": "ad5f2c04-2112-49dc-b533-125cca207eb0",
        "model_revision": (
            "sha256:3bef11697759b11db150b705e1882fb2d31a3efbdadbebd948aa917dcacc03a8"
        ),
        "phase": "paused",
        "active_pods": 0,
        "gpus_allocated": 0,
        "serving_qualified": False,
        "evidence": "docs/evidence/qwen38-lr30-step76-matched-serving-registration-20260921.json",
        "evidence_receipt_sha256": (
            "sha256:aff60c62dad8168b21dbfcba167b73387168dca759c40a821870689e38d15551"
        ),
    }
    assert lr30["fleet_live_parity"] == {
        "status": "passed_for_completed_descriptive_seed43_dev17_evaluation",
        "evidence": "docs/evidence/qwen38-lr30-step76-fleet-dev17-seed43-launch-20260921.json",
        "evidence_receipt_sha256": (
            "sha256:b6e0f971f3d0497f564fcaef3ac31e2762f4acc7ce8fab5f609d284b195022b2"
        ),
        "route_phase_at_observation": "ready",
        "active_pods_at_observation": 1,
        "fresh_parity_required_for_any_new_evaluation": True,
        "capability_claimed": False,
    }
    assert lr30["web_readiness"].endswith("fresh_matched_parity")
    assert lr30["fleet_readiness"] == (
        "completed_descriptive_seed43_dev17_unpaired_any_new_evaluation_requires_jit_resume_"
        "and_fresh_matched_parity"
    )
    assert lr30["live_target"]["status"] == "paused_registered_requires_jit_resume_and_fresh_parity"

    v1_artifacts = {
        row["artifact_id"]: row for row in _load(READINESS_V1)["artifacts"]
    }
    assert {
        artifact_id: row
        for artifact_id, row in artifacts.items()
        if artifact_id != "q38-available-a-lr30-step76"
    } == {
        artifact_id: row
        for artifact_id, row in v1_artifacts.items()
        if artifact_id != "q38-available-a-lr30-step76"
    }


def test_v2_ledger_labels_completed_unpaired_work_without_a_capability_claim() -> None:
    ledger = _load(LEDGER_V2)
    assert ledger["schema_version"] == 2
    assert ledger["supersedes"]["path"] == str(LEDGER_V1.relative_to(ROOT))
    assert ledger["supersedes"]["file_sha256"] == _sha256(LEDGER_V1)
    assert ledger["serving_readiness_matrix"] == str(READINESS_V2.relative_to(ROOT))
    assert ledger["serving_readiness_matrix_file_sha256"] == _sha256(READINESS_V2)

    rows = {row["artifact_id"]: row for row in ledger["accepted_checkpoints"]}
    lr30 = rows["q38-available-a-lr30-step76"]
    assert lr30["duplicate_guard"] == {
        "historical_seed43_sessions_must_not_be_replayed": True,
        "fresh_identity_required_for_any_future_matched_fleet_comparison": True,
        "fresh_duplicate_census_required_before_any_future_web_or_fleet_launch": True,
        "launch_authorized_by_this_ledger": False,
    }
    assert lr30["fleet_seed43_descriptive"] == {
        "accepted_sessions": 17,
        "status": "complete_stored_session_reconciliation_descriptive_unpaired",
        "live_route_parity": {
            "status": "passed_for_historical_seed43_launch_only",
            "launch_evidence": (
                "docs/evidence/qwen38-lr30-step76-fleet-dev17-seed43-launch-20260921.json"
            ),
            "launch_receipt_sha256": (
                "sha256:b6e0f971f3d0497f564fcaef3ac31e2762f4acc7ce8fab5f609d284b195022b2"
            ),
            "fresh_parity_required_for_any_new_evaluation": True,
        },
        "terminal_evidence": (
            "docs/evidence/qwen38-lr30-step76-stored-session-reconciliation-v3-terminal-"
            "20260921.json"
        ),
        "terminal_receipt_sha256": (
            "sha256:2cff332a1a37d898a591e62ebf41475f9a08fe882058756c1b9717d617afaaf6"
        ),
        "model_generation_or_scoring_by_reconciler": False,
        "final_eight_task_set_accessed": False,
        "interpretation": (
            "The launch proved live route parity for the historical dev17 evaluation, and the "
            "reconciler later accepted already completed exact sessions without rerunning a model "
            "or scorer. Together they form a complete descriptive dev17 record, not a new paired "
            "lift result."
        ),
    }
    assert lr30["evaluation_disposition"] == {
        "fleet_holdout": "complete_descriptive_dev17_unpaired_not_a_matched_result",
        "webexploitbench": "not_started_requires_fresh_matched_opencode_packet_and_gates",
    }

    # This correction is intentionally narrow: all earlier controls remain
    # exactly as recorded in the frozen v1 ledger.
    assert ledger["controls"] == _load(LEDGER_V1)["controls"]
    v1_rows = {
        row["artifact_id"]: row for row in _load(LEDGER_V1)["accepted_checkpoints"]
    }
    assert {
        artifact_id: row
        for artifact_id, row in rows.items()
        if artifact_id != "q38-available-a-lr30-step76"
    } == {
        artifact_id: row
        for artifact_id, row in v1_rows.items()
        if artifact_id != "q38-available-a-lr30-step76"
    }


def test_v2_index_remains_sanitized_and_does_not_claim_capability() -> None:
    serialized = READINESS_V2.read_text().lower() + LEDGER_V2.read_text().lower()
    for forbidden in ("api_key", "bearer ", "authorization:"):
        assert forbidden not in serialized
    matrix = _load(READINESS_V2)
    assert matrix["scientific_boundary"]["capability_claimed"] is False
    assert matrix["scientific_boundary"]["evaluation_launched_by_this_audit"] is False
