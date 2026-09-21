"""Checked-in production v2 campaign is reproducible and science-identical."""

from __future__ import annotations

import json
from pathlib import Path

from evals.fleet import visible_action_collection_v2 as runtime
from training import collection_campaign_v2 as campaign_v2
from training import current_collection_campaigns_v2 as current_v2

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2.source.json"
OUTPUT = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2"
V1_OUTPUT = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v1"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_committed_v2_campaign_is_reproducible_and_create_once() -> None:
    rendered = current_v2.render(_load(SPEC), root=ROOT)
    current_v2.current.check(OUTPUT, rendered)
    operation = rendered["operation-authorization.json"]
    packet = rendered["collection-packet.json"]
    receipt = rendered["materialization-receipt.json"]

    assert operation["planned_cells"] == receipt["planned_cells"] == 200
    assert operation["sha256"] == packet["operation_authorization_sha256"]
    assert operation["sha256"] == receipt["operation_authorization_sha256"]
    assert packet["execution_safety"]["cluster_job_execution_requirements"] == (
        campaign_v2.JOB_EXECUTION_REQUIREMENTS
    )
    assert receipt["execution_mode"] == runtime.EXECUTION_MODE
    assert receipt["submitted"] is False
    assert receipt["fleet_api_calls"] == receipt["model_calls"] == 0
    assert receipt["trace_or_score_reads"] == 0


def test_v2_changes_execution_identity_only() -> None:
    old = {path.name: _load(path) for path in V1_OUTPUT.iterdir() if path.is_file()}
    new = {path.name: _load(path) for path in OUTPUT.iterdir() if path.is_file()}
    assert new["task-selection.json"] == old["task-selection.json"]
    assert new["metadata-inventory.json"] == old["metadata-inventory.json"]
    assert new["family-split.json"] == old["family-split.json"]
    assert new["role-anchor.json"] == old["role-anchor.json"]
    assert new["runtime-bindings.json"] == old["runtime-bindings.json"]
    new_request = dict(new["collection-request.json"])
    old_request = dict(old["collection-request.json"])
    assert new_request.pop("source_authorization_receipt_sha256") == _load(SPEC)["sha256"]
    old_request.pop("source_authorization_receipt_sha256")
    assert new_request == old_request
    for field in (
        "name",
        "task_set",
        "models",
        "routes",
        "harness",
        "images",
        "pass_k",
        "concurrency",
        "sampling",
        "training_data_eligible",
    ):
        assert new["eval-config.json"][field] == old["eval-config.json"][field]
    assert new["collection-packet.json"]["schema"] == campaign_v2.PACKET_SCHEMA
    assert old["collection-packet.json"]["schema"] == "cyber_trajectory_collection_packet_v1"
