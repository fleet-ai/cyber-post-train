"""Append-only eligibility upgrade for blocked executions with new absence proof."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.fleet import execution_generation_rollforward_v1 as v1
from evals.fleet import qwen38_rank97_session_absence_observer_v1 as rank97_observer
from evals.fleet import self_hosted

SCHEMA = "fleet-execution-generation-rollforward-assessment-v2"
BASE_PATH = Path(
    "docs/evidence/qwen38-study/"
    "2026-09-05-qwen38-blocked-execution-rollforward-assessment-v1.json"
)
BASE_SELF_SHA256 = "sha256:40c2c6a91be6b57164ebc46b5283c2bd1d8d684b69ab5dfad9421e961a787cec"
OBSERVER_PATH = Path(
    "docs/evidence/qwen38-study/2026-09-05-qwen38-rank97-tail-session-absence-v1.json"
)
OBSERVER_SELF_SHA256 = "sha256:c7b810a3d48a3447eb67cbb0bf61f4e2a97d1e105d614ac0e030fadff027ac3a"
OBSERVER_FILE_SHA256 = "sha256:b124b440527a8e99dd5ad8125e2051c34c53b8ed7a3548a762dddefac5b5ef14"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("receipt_sha256") != self_hosted.digest_without(value, "receipt_sha256"):
        raise ValueError(f"roll-forward v2 evidence digest drifted: {path}")
    return value


def build(root: Path, observed_at_utc: str) -> dict[str, Any]:
    base = v1.load(root / BASE_PATH)
    v1.validate(base, root)
    observer_path = root / OBSERVER_PATH
    observer = _load(observer_path)
    rank97_observer.validate(observer, root)
    if (
        base["receipt_sha256"] != BASE_SELF_SHA256
        or observer["receipt_sha256"] != OBSERVER_SELF_SHA256
        or self_hosted.sha256(observer_path.read_bytes()) != OBSERVER_FILE_SHA256
    ):
        raise ValueError("roll-forward v2 authority bytes drifted")
    target_attempts = {row["attempt"] for row in observer["targets"]}
    upgraded = []
    for row in base["cells"]:
        if row["selection_rank"] != 97 or row["attempt"] not in target_attempts:
            continue
        successor_generation = row["prior_execution_generation"] + 1
        successor = v1.exact.execution_for(row["cell_id"], successor_generation)
        upgraded.append(
            {
                "cell_id": row["cell_id"],
                "selection_rank": row["selection_rank"],
                "attempt": row["attempt"],
                "prior_execution_generation": row["prior_execution_generation"],
                "prior_execution_id": row["prior_execution_id"],
                "prior_claim_path": row["prior_claim_path"],
                "prior_claim_receipt_sha256": row["prior_claim_receipt_sha256"],
                "prior_claim_file_sha256": row["prior_claim_file_sha256"],
                "successor_execution_generation": successor_generation,
                "successor_execution_id": successor["execution_id"],
                "proof_upgrade": {
                    "authoritative_sessions_absent": True,
                    "observer_receipt_sha256": OBSERVER_SELF_SHA256,
                    "all_other_v1_zero_side_effect_proofs_preserved": True,
                },
            }
        )
    if [row["attempt"] for row in upgraded] != [2, 3, 4]:
        raise ValueError("roll-forward v2 target set drifted")
    body = {
        "schema_version": SCHEMA,
        "status": "HELD_REVIEWED",
        "observed_at_utc": observed_at_utc,
        "campaign_id": v1.EXPECTED_CAMPAIGN,
        "base_assessment": {
            "path": str(BASE_PATH),
            "receipt_sha256": BASE_SELF_SHA256,
        },
        "session_absence_observer": {
            "path": str(OBSERVER_PATH),
            "receipt_sha256": OBSERVER_SELF_SHA256,
            "file_sha256": OBSERVER_FILE_SHA256,
        },
        "upgraded_cells": upgraded,
        "resulting_counts": {"eligible": 7, "ineligible": 2, "total": 9},
        "statistical_accounting": base["statistical_accounting"],
        "launch_authorized": False,
        "objects_created": False,
        "privacy": base["privacy"],
    }
    return {**body, "receipt_sha256": self_hosted.digest_without(body, "receipt_sha256")}


def validate(value: dict[str, Any], root: Path) -> None:
    if value != build(root, value.get("observed_at_utc")):
        raise ValueError("roll-forward v2 assessment drifted")
