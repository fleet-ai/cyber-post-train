from __future__ import annotations

import json
from pathlib import Path

from evals.fleet import qwen38_rank3_a2_terminal_reconciliation_v1 as reconciliation

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-rank3-a2-terminal-reconciliation-v1.json"
)


def test_frozen_rank3_a2_reconciliation_validates() -> None:
    reconciliation.validate(json.loads(EVIDENCE.read_text()), ROOT)
