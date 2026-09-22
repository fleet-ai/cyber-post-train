"""Checked-in v3 action campaign is reproducible and completion-budget safe."""

from __future__ import annotations

import json
from pathlib import Path

from evals.fleet import visible_action_collection_v3 as runtime
from training import current_collection_campaigns as current
from training import current_collection_campaigns_v3 as current_v3

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3.source.json"
OUTPUT = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3"
V2 = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_committed_v3_campaign_is_reproducible() -> None:
    rendered = current_v3.render(_load(SPEC), root=ROOT)
    current.check(OUTPUT, rendered)
    receipt = rendered["materialization-receipt.json"]
    packet = rendered["collection-packet.json"]

    assert receipt["planned_cells"] == 200
    assert receipt["training_data_eligible"] is True
    assert receipt["submitted"] is False
    assert receipt["model_calls"] == receipt["trace_or_score_reads"] == 0
    assert receipt["completion_budget"] == runtime.COMPLETION_BUDGET_POLICY
    assert receipt["completion_budget_runtime_sha256"] == packet["completion_budget_runtime_sha256"]
    assert packet["corpus_scope"]["visible_reasoning_included"] is False


def test_v3_preserves_current75_selection_and_heldout_boundary() -> None:
    for name in (
        "task-selection.json",
        "metadata-inventory.json",
        "family-split.json",
        "role-anchor.json",
        "protected-family-lock.json",
        "runtime-bindings.json",
    ):
        assert _load(OUTPUT / name) == _load(V2 / name)
    selection = _load(OUTPUT / "task-selection.json")
    split = _load(OUTPUT / "family-split.json")
    role_by_version = {row["task_version_id"]: row["split"] for row in split["tasks"]}
    assert len(selection["tasks"]) == 50
    assert {role_by_version[row["task_version_id"]] for row in selection["tasks"]} == {"train"}
