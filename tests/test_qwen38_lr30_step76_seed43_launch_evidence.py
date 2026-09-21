from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-lr30-step76-fleet-dev17-seed43-launch-20260921.json"


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_lr30_launch_evidence_is_score_blind_bound_and_self_digesting() -> None:
    receipt = json.loads(EVIDENCE.read_text())
    claimed = receipt.pop("receipt_sha256")
    assert claimed == _digest(receipt)

    assert receipt["source"]["repository_commit"] == ("276231a5e202cab0b2be2e06d5f5c0b99cb6e509")
    assert receipt["checkpoint_and_route"]["parity_status"] == "passed"
    assert receipt["checkpoint_and_route"]["session_model"] == ("qwen/chris-q38-lr30-step76-web-v1")
    readback = receipt["checkpoint_and_route"]["first_authoritative_session_readback"]
    assert readback["expected_persisted_model"] == readback["observed_persisted_model"]
    assert readback["authoritative_match_count"] == 1
    assert readback["status"] == "completed"
    assert readback["task_key_binding_matches"] is True
    assert readback["verifier_execution_binding_matches"] is True
    job = receipt["evaluator"]["job"]
    assert job["failure_alerts"] == "off"
    assert job["priority_class"] == "c1"
    assert job["backoff_limit"] == 0
    assert job["gpu_request"] == 0
    assert receipt["evaluator"]["workload"]["admitted"] is True
    assert receipt["evaluator"]["pod"]["ready"] is True
    assert receipt["evaluator"]["pod"]["restarts"] == 0

    boundary = receipt["acceptance_boundary"]
    assert boundary == {
        "authoritative_checkpoint_session_readback_available": True,
        "capability_claim_made": False,
        "scores_included": False,
        "session_identifiers_included": False,
        "prompts_or_traces_included": False,
        "final_eight_task_set_accessed": False,
        "statement": (
            "This proves launch identity, live route parity, admission, and useful "
            "evaluator traffic only. It is not a capability result."
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
