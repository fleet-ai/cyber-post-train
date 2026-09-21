from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-fleet-dev17-seed43-authoritative-readback-20260921.json"


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_seed43_authoritative_readback_is_score_blind_and_self_digesting() -> None:
    receipt = json.loads(EVIDENCE.read_text())
    claimed = receipt.pop("receipt_sha256")
    assert claimed == _digest(receipt)

    assert set(receipt["readbacks"]) == {"self44", "teacher186"}
    for readback in receipt["readbacks"].values():
        assert readback["expected_persisted_model"] == readback["observed_persisted_model"]
        assert readback["authoritative_match_count"] == 1
        assert readback["status"] == "completed"
        assert readback["task_binding_matches"] is True
        assert readback["verifier_execution_binding_matches"] is True
        assert readback["pod_ready"] is True
        assert readback["pod_restarts"] == 0

    boundary = receipt["acceptance_boundary"]
    assert boundary == {
        "capability_claim_made": False,
        "scores_included": False,
        "session_identifiers_included": False,
        "prompts_or_traces_included": False,
        "final_eight_task_set_accessed": False,
        "statement": (
            "This proves only checkpoint identity and infrastructure health. "
            "It is not a capability result and does not make an arm terminal."
        ),
    }

    serialized = EVIDENCE.read_text().lower()
    for forbidden in (
        '"score"',
        '"session_id"',
        '"task_key"',
        '"task_version_id"',
        '"prompt"',
        '"trace"',
        '"flag"',
    ):
        assert forbidden not in serialized
