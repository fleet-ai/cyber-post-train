from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import execution_generation_rollforward_v2 as rollforward
from evals.fleet import self_hosted

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-blocked-execution-rollforward-assessment-v2.json"
)


def test_v2_upgrades_only_rank97_tail() -> None:
    value = json.loads(EVIDENCE.read_text())
    rollforward.validate(value, ROOT)
    assert value["resulting_counts"] == {"eligible": 7, "ineligible": 2, "total": 9}
    assert [row["attempt"] for row in value["upgraded_cells"]] == [2, 3, 4]
    assert all(row["selection_rank"] == 97 for row in value["upgraded_cells"])


def test_v2_cannot_upgrade_an_extra_cell() -> None:
    value = json.loads(EVIDENCE.read_text())
    value["upgraded_cells"].append(copy.deepcopy(value["upgraded_cells"][0]))
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    with pytest.raises(ValueError, match="assessment drifted"):
        rollforward.validate(value, ROOT)
