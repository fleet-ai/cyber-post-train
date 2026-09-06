import copy
import json
from pathlib import Path

import pytest

from evals.fleet import exact_pass4_blocked_disposition_v1 as disposition
from evals.fleet import self_hosted

ROOT = Path(__file__).parents[1]
RECEIPT = ROOT / (
    "docs/evidence/qwen38-study/"
    "2026-09-06-exact-pass4-blocked-cell-disposition-v1.json"
)


def _write(tmp_path: Path, value: dict) -> Path:
    value["receipt_sha256"] = self_hosted.digest_without(value, "receipt_sha256")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(value))
    return path


def test_blocked_cell_disposition_is_complete_and_score_blind():
    value = disposition.validate(RECEIPT, root=ROOT)
    assert value["counts"] == {
        "total_blocked_cells": 7,
        "existing_session_acceptable_under_reviewed_rule": 0,
        "deterministic_infrastructure_successor_scientifically_valid": 3,
        "irrecoverable_under_current_reviewed_rules": 4,
    }


def test_missing_cell_or_retry_authorization_fails_closed(tmp_path):
    value = json.loads(RECEIPT.read_text())
    missing = copy.deepcopy(value)
    missing["cells"].pop()
    with pytest.raises(ValueError):
        disposition.validate(_write(tmp_path, missing), root=ROOT)

    unsafe = copy.deepcopy(value)
    unsafe["cells"][4]["successor"]["launch_authorized_by_this_table"] = True
    with pytest.raises(ValueError):
        disposition.validate(_write(tmp_path, unsafe), root=ROOT)
