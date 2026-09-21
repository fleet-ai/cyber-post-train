from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
LEDGER = ROOT / "configs/evaluation/qwen38-web-source-snapshot-ledger-v1.json"
EVIDENCE = ROOT / "docs/evidence/qwen38-web-source-refresh-m391-accepted-20260921.json"
PRIOR_INCIDENT = (
    ROOT / "docs/evidence/qwen38-web-source-refresh-v4-dependency-failure-20260921.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def _self_digest(value: dict, field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field)
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_m391_source_snapshot_acceptance_is_durable_but_not_an_evaluation_result() -> None:
    evidence = _load(EVIDENCE)
    ledger = _load(LEDGER)

    assert evidence["schema"] == "qwen38_web_source_snapshot_refresh_acceptance_v1"
    assert evidence["receipt_sha256"] == _self_digest(evidence, "receipt_sha256")
    assert evidence["classification"] == {
        "status": "accepted_and_released",
        "role": "source_preparation_only",
        "fresh_zero_model_snapshot_qualification_required": True,
        "rollout_collection_accepted": False,
        "scoring_accepted": False,
        "capability_claimed": False,
    }

    assert ledger["schema"] == "cyber_qwen38_web_source_snapshot_ledger_v1"
    assert ledger["sha256"] == _self_digest(ledger, "sha256")
    assert ledger["source"]["main_commit"] == "28c3cd0016bf9ca745bef73d8f804607ee07a18b"
    assert ledger["prior_source_refresh_incident"] == {
        "path": str(PRIOR_INCIDENT.relative_to(ROOT)),
        "file_sha256": _file_digest(PRIOR_INCIDENT),
    }

    accepted = ledger["accepted_source_snapshot"]
    assert accepted["evidence_path"] == str(EVIDENCE.relative_to(ROOT))
    assert accepted["evidence_file_sha256"] == _file_digest(EVIDENCE)
    assert accepted["evidence_receipt_sha256"] == evidence["receipt_sha256"]
    assert accepted["status"] == "accepted_and_released_source_preparation_only"
    assert accepted["snapshot_id"] == "ppjk4at6m8t0z5v4f4r7q"
    assert accepted["fresh_zero_model_snapshot_qualification_required"] is True
    assert accepted["collection_or_scoring_accepted"] is False
    assert accepted["capability_result"] is False


def test_m391_source_snapshot_record_is_sanitized() -> None:
    serialized = LEDGER.read_text().lower() + EVIDENCE.read_text().lower()
    for forbidden in ("api_key", "bearer ", "authorization:"):
        assert forbidden not in serialized
