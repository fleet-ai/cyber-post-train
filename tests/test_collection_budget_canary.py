"""The checked-in completion-budget canary is one exact train family x pass@4."""

from __future__ import annotations

import json
from pathlib import Path

from training import collection_budget_canary as canary
from training import current_collection_campaigns as current

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3-canary.source.json"
OUTPUT = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v3-canary"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_canary_is_reproducible_bounded_and_heldout_excluding() -> None:
    rendered = canary.render(_load(SPEC), root=ROOT)
    current.check(OUTPUT, rendered)
    receipt = rendered["materialization-receipt.json"]
    packet = rendered["collection-packet.json"]
    selection = rendered["task-selection.json"]

    assert len(selection["tasks"]) == 1
    assert receipt["planned_cells"] == 4
    assert receipt["heldout_roles_excluded"] == ["dev", "final_test"]
    assert receipt["submitted"] is False
    assert receipt["model_calls"] == receipt["trace_or_score_reads"] == 0
    assert packet["execution_safety"]["planned_cells"] == 4
    assert packet["execution_safety"]["maximum_planned_cells"] == 4
    assert packet["corpus_scope"]["visible_reasoning_included"] is False
