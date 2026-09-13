from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / ("docs/evidence/qwen38-study/2026-09-13-teacher-sft-miles-preflight-v1.json")


def test_teacher_miles_preflight_is_self_digested_and_sanitized() -> None:
    serialized = EVIDENCE.read_text(encoding="utf-8")
    evidence = json.loads(serialized)
    unsigned = {key: value for key, value in evidence.items() if key != "sha256"}
    expected = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )

    assert evidence["sha256"] == expected
    assert "/mnt/" not in serialized
    assert "/private/" not in serialized
    assert ".operator/" not in serialized
    assert "ASIA" not in serialized
    assert "tl_apiKey_" not in serialized
    assert re.search(r"(?<![A-Za-z0-9_])sk_[A-Za-z0-9_-]{16,}", serialized) is None
    assert not any(evidence["privacy"].values())


def test_teacher_miles_preflight_records_only_the_completed_operational_gates() -> None:
    evidence = json.loads(EVIDENCE.read_bytes())

    assert evidence["status"] == "accepted_preflight_only"
    assert evidence["scope"]["gpu_allocations"] == 0
    assert evidence["stage_verifier"]["pod"]["gpus"] == 0
    assert evidence["cpu_preflight"]["pod"]["gpus"] == 0
    assert evidence["stage_verifier"]["gates"]["teacher_source_stability_valid"] is True
    assert evidence["cpu_preflight"]["gates"]["conversion_output_absent"] is True
    assert evidence["cpu_preflight"]["gates"]["submission_journal_absent"] is True

    decision = evidence["scheduling_decision"]
    assert decision["conversion_prepared"] is True
    assert decision["conversion_queued"] is False
    assert decision["conversion_resources"] == {
        "workers": 1,
        "gpus_per_worker": 8,
        "total_gpus": 8,
    }
    assert decision["native_checkpoint_created"] is False
    assert decision["all_rank_reload_queued"] is False
    assert decision["training_queued"] is False
    assert evidence["interpretation"]["classification"] == "operational_gate"
    assert evidence["interpretation"]["capability_claimed"] is False
