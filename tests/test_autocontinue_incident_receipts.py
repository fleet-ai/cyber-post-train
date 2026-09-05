import json
from pathlib import Path

from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]
RECEIPT = (
    ROOT
    / "docs/evidence/qwen38-study/"
    "2026-09-04-opencode-autocontinue-canary-v2-pre-model-failure.json"
)


def test_canary_v2_pre_model_failure_receipt_is_digest_valid_and_noncrediting() -> None:
    receipt = json.loads(RECEIPT.read_text())

    assert receipt["receipt_sha256"] == self_hosted.digest_without(
        receipt, "receipt_sha256"
    )
    assert receipt["status"] == "INFRASTRUCTURE_INVALID_PRE_MODEL"
    assert receipt["scientific_disposition"] == {
        "statistical_cells_consumed": 0,
        "valid_rollouts": 0,
        "retry_requires_fresh_execution_generation": True,
        "reuse_old_run_or_job_identity": False,
        "preserve_old_claims": True,
    }
    assert len(receipt["models"]) == 2
    assert all(row["attempt_directory_count"] == 0 for row in receipt["models"])
    assert all(row["fleet_session_count"] == 0 for row in receipt["models"])
    assert all(row["verifier_execution_count"] == 0 for row in receipt["models"])
    assert receipt["aggregate"]["accepted_rollouts"] == 0
    assert receipt["privacy"] == {
        "credentials_included": False,
        "prompts_traces_flags_or_scores_included": False,
        "source_logs_included": False,
    }
