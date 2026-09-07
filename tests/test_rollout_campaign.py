from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from evals.fleet import rollout_campaign, rollout_ledger

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json"
SELECTION = ROOT / "evals/fleet/configs/opencode-easiest-train100-selection-v2.json"
SNAPSHOT = ROOT / "docs/evidence/qwen38-study/2026-09-06-exact-pass4-ledger-v50.json"


def test_frozen_campaign_reconciles_exactly(tmp_path: Path) -> None:
    rows = rollout_campaign.build_plan_rows(CAMPAIGN, SELECTION, SNAPSHOT)
    assert len(rows) == 800
    assert len({(row["task_version_id"], row["model_id"], row["attempt"]) for row in rows}) == 800

    plan = tmp_path / "plan.csv"
    database = tmp_path / "ledger.sqlite3"
    rollout_campaign.write_plan(plan, rows)
    rollout_ledger.initialize(database, plan)
    result = rollout_campaign.import_snapshot(database, SNAPSHOT)

    assert result["accepted"] == 66
    assert result["terminal"] == 13
    assert result["pending"] == 721
    assert rollout_campaign.import_snapshot(database, SNAPSHOT)["created"] is False
    assert rollout_ledger.summary(database)["by_state"] == {
        "pending": 721,
        "claimed": 0,
        "running": 0,
        "grading": 0,
        "accepted": 66,
        "retry_review": 0,
        "terminal": 13,
    }


def test_attempts_for_one_task_stay_on_one_serving_block() -> None:
    rows = rollout_campaign.build_plan_rows(CAMPAIGN, SELECTION, SNAPSHOT)
    assignments: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in rows:
        assignments.setdefault((row["model_id"], row["task_version_id"]), set()).add(
            (row["serving_block"], row["endpoint_model_id"])
        )
    assert all(len(routes) == 1 for routes in assignments.values())
    assert {row["serving_block"] for row in rows} == {
        "qwen-shared",
        "qwen-dedicated",
        "glm-shared",
        "glm-dedicated",
    }


def test_selection_digest_drift_is_rejected(tmp_path: Path) -> None:
    changed = json.loads(SELECTION.read_text(encoding="utf-8"))
    changed["selected_count"] = 99
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(rollout_ledger.LedgerError, match="selection file digest"):
        rollout_campaign.build_plan_rows(CAMPAIGN, path, SNAPSHOT)


def test_snapshot_self_digest_drift_is_rejected(tmp_path: Path) -> None:
    changed = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    changed["status"] = "DRIFTED"
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(rollout_ledger.LedgerError, match="self-digest"):
        rollout_campaign.build_plan_rows(CAMPAIGN, SELECTION, path)


def test_plan_csv_contains_only_score_blind_fields(tmp_path: Path) -> None:
    plan = tmp_path / "plan.csv"
    rollout_campaign.write_plan(
        plan,
        rollout_campaign.build_plan_rows(CAMPAIGN, SELECTION, SNAPSHOT),
    )
    with plan.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert not ({"score", "prompt", "trace", "flag", "response"} & set(reader.fieldnames or ()))
        assert len(list(reader)) == 800
    assert hashlib.sha256(plan.read_bytes()).hexdigest()
