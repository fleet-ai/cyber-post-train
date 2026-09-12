"""Regression gates for the bounded dev6 source-only root-cause audit."""

import hashlib
import json
from pathlib import Path

from cyber_post_train.jobs import digest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/qwen38-study/2026-09-12-skyrl-dev6-source-root-cause-audit-v1.json"


def load():
    return json.loads(EVIDENCE.read_bytes())


def test_source_audit_is_sealed_and_binds_terminal_evidence():
    value = load()
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
    assert value["classification"] == (
        "historical_child_exception_unrecoverable_exact_image_disqualified"
    )
    for key in ("dev5", "dev6"):
        terminal = value["terminal_evidence"][key]
        path = ROOT / terminal["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == terminal["file_sha256"]
        receipt = json.loads(path.read_bytes())
        assert receipt["sha256"] == terminal["embedded_sha256"]


def test_source_audit_preserves_the_information_and_authority_boundaries():
    value = load()
    boundary = value["proven_information_boundary"]
    assert "cannot be recovered" in boundary["historical_recovery"]
    assert value["model_support_chronology"]["checkpoint"]["architecture"] == (
        "Qwen3_5ForConditionalGeneration"
    )
    assert "do not prove" in value["model_support_chronology"]["inference_limit"]
    assert value["replacement_image_path"]["replacement_available_now"] is False
    miles = value["alternate_backend_boundary"]
    path = ROOT / miles["terminal_evidence_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == miles["terminal_evidence_file_sha256"]
    assert json.loads(path.read_bytes())["sha256"] == miles["terminal_evidence_embedded_sha256"]
    assert value["audit_boundaries"]["historical_child_cause_inferred"] is False
    assert "remain closed" in value["next_gate"]
