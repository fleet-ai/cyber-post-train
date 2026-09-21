"""Offline tests for the isolated non-thinking collection runtime."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from evals.fleet import collection_fixed_proxy as proxy
from evals.fleet import evaluate
from evals.fleet import visible_action_collection as runtime

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v1"


def _load(name: str) -> dict:
    return json.loads((CAMPAIGN / name).read_text())


def test_collection_runtime_is_new_schema_and_leaves_historical_file_set_unchanged() -> None:
    plan = runtime.compile_eval(_load("eval-config.json"), relative_to=CAMPAIGN)
    assert plan["schema"] == runtime.PLAN_SCHEMA
    assert plan["planned_cells"] == 200
    assert plan["pass_k"] == 4
    assert plan["training_data_eligible"] is True
    assert plan["automatic_retry"] is False
    assert plan["max_reviewed_infrastructure_retries"] == 0
    assert plan["treatment"]["thinking_mode"] == runtime.THINKING_DISABLED
    assert plan["collection_runtime"] == _load("eval-config.json")["collection_runtime"]
    assert set(plan["runtime_files"]) == {"base_evaluator", "collection_adapter"}
    assert "visible_action_collection.py" not in evaluate.RUNTIME_FILES
    assert "collection_fixed_proxy.py" not in evaluate.RUNTIME_FILES


def test_nonthinking_settings_command_and_wire_rewrite_are_all_explicit() -> None:
    plan = runtime.compile_eval(_load("eval-config.json"), relative_to=CAMPAIGN)
    route = next(iter(plan["routes"].values()))
    settings = runtime.nonthinking_opencode_settings(
        {
            "model": {"served_id": route["served_id"]},
            "harness": {
                "name": plan["treatment"]["harness"],
                "version": plan["treatment"]["harness_version"],
                **plan["treatment"],
            },
        }
    )
    model = settings["provider"]["fleet-cluster"]["models"][route["served_id"]]
    assert model["reasoning"] is False
    assert "interleaved" not in model

    arguments = runtime.collection_docker_args(
        (
            "run",
            "bash",
            "-lc",
            "opencode run --format json --thinking --model fleet-cluster/qwen3.8-27b",
        )
    )
    assert "--thinking" not in arguments[-1]

    rewritten = json.loads(
        proxy.rewrite_completion(
            json.dumps(
                {
                    "model": "agent-choice",
                    "enable_thinking": True,
                    "chat_template_kwargs": {"enable_thinking": True, "other": "unsafe"},
                    "max_completion_tokens": 1,
                }
            ).encode(),
            {
                "model": route["served_id"],
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": 43,
                "max_tokens": 32768,
            },
        )
    )
    assert rewritten["model"] == route["served_id"]
    assert rewritten["chat_template_kwargs"] == {"enable_thinking": False}
    assert "enable_thinking" not in rewritten
    assert "max_completion_tokens" not in rewritten


def test_prepare_and_load_are_offline_create_once(tmp_path: Path) -> None:
    output = tmp_path / "wave"
    result = runtime.prepare(_load("eval-config.json"), output, relative_to=CAMPAIGN)
    assert result["sessions"] == 200
    assert result["submitted"] is False
    plan = runtime.load(output)
    assert plan["sha256"] == result["plan_sha256"]
    assert len(runtime.plan_rows(plan)) == 200


def test_external_duplicate_census_must_bind_exact_cells_and_be_zero(tmp_path: Path) -> None:
    output = tmp_path / "wave"
    runtime.prepare(_load("eval-config.json"), output, relative_to=CAMPAIGN)
    plan = runtime.load(output)
    census = {
        "schema": runtime.DUPLICATE_CENSUS_SCHEMA,
        "plan_sha256": plan["sha256"],
        "planned_cell_universe_sha256": runtime.cell_universe_sha256(plan),
        "planned_cells": 200,
        "coverage": runtime.DUPLICATE_CENSUS_COVERAGE,
        "authority_snapshot_sha256": "sha256:" + "a" * 64,
        "issuer": "reviewed-private-collection-authority",
        "completed_at": datetime.now(UTC).isoformat(),
        "exact_duplicate_count": 0,
        "ambiguous_cell_count": 0,
    }
    census["receipt_sha256"] = "sha256:" + runtime.digest(census)
    (output / runtime.DUPLICATE_CENSUS_FILE).write_text(json.dumps(census))
    assert runtime.validate_duplicate_census(output, plan, require_fresh=True) == census

    census["ambiguous_cell_count"] = 1
    census["receipt_sha256"] = "sha256:" + runtime.digest(
        {key: item for key, item in census.items() if key != "receipt_sha256"}
    )
    (output / runtime.DUPLICATE_CENSUS_FILE).write_text(json.dumps(census))
    try:
        runtime.validate_duplicate_census(output, plan, require_fresh=False)
    except ValueError as error:
        assert "reconciliation" in str(error)
    else:
        raise AssertionError("ambiguous cells must fail closed")

    census["ambiguous_cell_count"] = 0
    census["completed_at"] = (datetime.now(UTC) - timedelta(seconds=601)).isoformat()
    census["receipt_sha256"] = "sha256:" + runtime.digest(
        {key: item for key, item in census.items() if key != "receipt_sha256"}
    )
    (output / runtime.DUPLICATE_CENSUS_FILE).write_text(json.dumps(census))
    try:
        runtime.validate_duplicate_census(output, plan, require_fresh=True)
    except ValueError as error:
        assert "not fresh" in str(error)
    else:
        raise AssertionError("stale census must fail closed")
