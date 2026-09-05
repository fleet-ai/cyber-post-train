import json
from pathlib import Path

from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "docs/evidence/qwen38-study/2026-09-05-qwen38-dedicated-r002-a2-accepted-validated-v2.json"
)


def test_rank2_a2_validated_acceptance_is_self_digesting_and_sealed_safe() -> None:
    value = json.loads(RECEIPT.read_text())
    assert value["schema_version"] == ("fleet-qwen38-dedicated-tp1-accepted-validated-v2")
    assert value["status"] == "ACCEPTED_VALIDATED"
    assert value["accepted"] is True
    assert value["credited"] is True
    assert value["retry_allowed"] is False
    assert value["all_artifact_byte_digests_matched"] is True
    assert value["fresh_authoritative_session_reconciled"] is True
    assert value["fleet_api_mutations"] == 0
    assert value["prompts_or_traces_included"] is False
    assert value["scores_included"] is False
    assert value["credentials_included"] is False
    assert value["receipt_sha256"] == self_hosted.digest_without(value, "receipt_sha256")
    assert set(value["artifact_file_sha256"]) == {
        "accepted",
        "claim",
        "cleanup",
        "plan",
        "result",
        "reward",
        "session_ingest",
        "terminal",
    }
