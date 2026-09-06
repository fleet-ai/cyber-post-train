"""Validate the score-blind disposition of every exact-pass@4 blocked cell."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.fleet import self_hosted

SCHEMA = "fleet-exact-pass4-blocked-cell-disposition-v1"
EXPECTED_CELLS = {
    "sha256:f3921b5927bcf73a0df58ff991841db52f9b23f04cae5d482c5f7e8630308250",
    "sha256:6edac950ae44ff62c07074a775afdb65cc9b394047883e0b57fcad84e74d2fa7",
    "sha256:f5dced06656b0bf069930628978ce31b6ff682f437c86d55c6b241b2f83f4f48",
    "sha256:59df171f4303096371f516f604cf76032b33d4ad7298ff51353854e38c8a3ec4",
    "sha256:160c68de576263202b614038ded2140e6bcf72435a2086fc7af6da6a3a19ba78",
    "sha256:209f2350068810f475a61771b588a2ec4ed8aaa99b82197e037eaca57f929413",
    "sha256:0652b1bc6305c8d1ff4cc4184390d69f67ee2a365551ba59408377dddf333869",
}
DISPOSITIONS = {
    "existing_session_acceptable_under_reviewed_rule",
    "deterministic_infrastructure_successor_scientifically_valid",
    "irrecoverable_under_current_reviewed_rules",
}


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("receipt_sha256") != self_hosted.digest_without(
        value, "receipt_sha256"
    ):
        raise ValueError(f"receipt self digest drifted: {path}")
    return value


def validate(path: Path, *, root: Path) -> dict[str, Any]:
    value = _load_receipt(path)
    if value.get("schema_version") != SCHEMA:
        raise ValueError("blocked-cell disposition schema drifted")
    if value.get("privacy") != {
        "prompts_read": False,
        "traces_read": False,
        "flags_read": False,
        "scores_read": False,
        "credentials_included": False,
    }:
        raise ValueError("blocked-cell disposition privacy contract drifted")
    if value.get("policy") != {
        "launch_authorized": False,
        "existing_session_credit_authorized": False,
        "successor_cells_require_fresh_launch_time_collision_and_capacity_gates": True,
    }:
        raise ValueError("blocked-cell disposition policy drifted")

    ledger = value.get("source_ledger", {})
    ledger_path = root / ledger.get("path", "")
    ledger_value = _load_receipt(ledger_path)
    if (
        ledger.get("file_sha256") != _file_digest(ledger_path)
        or ledger.get("receipt_sha256") != ledger_value["receipt_sha256"]
    ):
        raise ValueError("source ledger binding drifted")

    cells = value.get("cells")
    if not isinstance(cells, list) or len(cells) != 7:
        raise ValueError("blocked-cell table must contain exactly seven rows")
    if {row.get("cell_id") for row in cells} != EXPECTED_CELLS:
        raise ValueError("blocked-cell identity set drifted")
    identities = {
        (row.get("model"), row.get("selection_rank"), row.get("attempt"))
        for row in cells
    }
    if len(identities) != 7:
        raise ValueError("blocked-cell statistical identities are not unique")

    disposition_counts = Counter(row.get("disposition") for row in cells)
    if set(disposition_counts) - DISPOSITIONS:
        raise ValueError("unknown blocked-cell disposition")
    expected_counts = {
        "total_blocked_cells": 7,
        **{name: disposition_counts[name] for name in sorted(DISPOSITIONS)},
    }
    if value.get("counts") != expected_counts:
        raise ValueError("blocked-cell disposition counts drifted")

    for row in cells:
        if not row.get("execution_id", "").startswith("sha256:"):
            raise ValueError("blocked-cell execution identity is missing")
        for evidence in row.get("evidence", []):
            evidence_path = root / evidence.get("path", "")
            evidence_value = _load_receipt(evidence_path)
            if (
                evidence.get("file_sha256") != _file_digest(evidence_path)
                or evidence.get("receipt_sha256")
                != evidence_value["receipt_sha256"]
            ):
                raise ValueError("blocked-cell evidence binding drifted")
        if row["disposition"] == "deterministic_infrastructure_successor_scientifically_valid":
            successor = row.get("successor", {})
            if (
                row.get("session_state", {}).get("exact_matches") != 0
                or row.get("acceptance_state") != "absent"
                or row.get("verifier_state") != "absent"
                or successor.get("launch_authorized_by_this_table") is not False
                or successor.get("execution_generation", 0)
                <= row.get("execution_generation", 0)
            ):
                raise ValueError("deterministic successor proof is incomplete")

    return value
