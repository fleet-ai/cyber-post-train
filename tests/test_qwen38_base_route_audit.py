"""Integrity checks for the read-only split-A Qwen base-route audit."""

from __future__ import annotations

import json
from pathlib import Path

from evals.fleet import dev_outcome_protocol as protocol
from training.io import digest_json, file_sha256

ROOT = Path(__file__).parents[1]
AUDIT = ROOT / "docs/evidence/inference/2026-09-11-qwen38-fleet-dev-base-route-audit.json"
CONTROL = ROOT / "configs/evaluation/qwen38-blackbox-fleet-dev-a-base-control-v1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_route_audit_is_sealed_and_keeps_the_control_closed():
    audit = read(AUDIT)
    control = read(CONTROL)
    parent = read(ROOT / control["parent_protocol"]["path"])
    task_set = read(ROOT / control["task_set"]["path"])

    assert audit["sha256"] == digest_json(
        {key: value for key, value in audit.items() if key != "sha256"}
    )
    assert audit["frozen_inputs"]["base_control"] == {
        "path": CONTROL.relative_to(ROOT).as_posix(),
        "sha256": control["sha256"],
        "file_sha256": file_sha256(CONTROL),
    }
    protocol.validate_base_control(control, parent, task_set)
    assert control["serving_binding"]["launchable"] is False
    assert not any(control["serving_binding"]["fields"].values())
    assert set(audit["unresolved_base_serving_binding"]) == set(
        protocol.UNBOUND_BASE_SERVING_FIELDS
    )
    assert audit["routes"]["shared"]["exact_frozen_revision"] is True
    assert audit["routes"]["shared"]["assessment"][
        "readiness_and_catalog_identity_passed"
    ] is True
    assert audit["routes"]["dedicated"]["assessment"][
        "readiness_and_catalog_identity_passed"
    ] is False
    assert audit["decision"] == {
        "classification": "operational_route_ready_but_exact_matched_control_blocked",
        "selected_existing_route": None,
        "base_control_modified": False,
        "base_control_launchable": False,
        "fresh_live_pair_parity_receipt_created": False,
        "next_gate": audit["decision"]["next_gate"],
        "evaluation_submitted": False,
    }
    assert audit["scope"]["prompt_or_completion_requests"] == 0
    assert audit["scope"]["scoring_requests"] == 0
    assert audit["scope"]["cluster_or_api_mutations"] == 0
