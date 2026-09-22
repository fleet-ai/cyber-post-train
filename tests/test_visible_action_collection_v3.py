"""Offline regressions for the completion-budget-safe v3 collection rail."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from evals.fleet import visible_action_collection_v2 as v2
from evals.fleet import visible_action_collection_v3 as runtime

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v2"


def _load(name: str) -> dict:
    return json.loads((V2 / name).read_text())


def _runtime(maximum_planned_cells: int) -> dict:
    old = _load("eval-config.json")["collection_runtime"]
    return {
        **old,
        "schema": runtime.RUNTIME_SCHEMA,
        "maximum_planned_cells": maximum_planned_cells,
        "completion_budget": runtime.COMPLETION_BUDGET_POLICY,
    }


def test_v3_changes_runtime_identity_without_mutating_v2() -> None:
    old_config = _load("eval-config.json")
    config = copy.deepcopy(old_config)
    config["collection_runtime"] = _runtime(200)
    old = v2.compile_eval(old_config, relative_to=V2)
    new = runtime.compile_eval(config, relative_to=V2)

    assert old["sha256"] == _load("collection-packet.json")["eval_plan_sha256"].removeprefix(
        "sha256:"
    )
    assert new["schema"] == runtime.PLAN_SCHEMA
    assert new["sha256"] != old["sha256"]
    assert new["planned_cells"] == old["planned_cells"] == 200
    assert new["tasks"] == old["tasks"]
    assert new["routes"] == old["routes"]
    assert new["treatment"] == old["treatment"]
    assert new["collection_runtime"]["completion_budget"] == (runtime.COMPLETION_BUDGET_POLICY)
    assert "collection_fixed_proxy.py" not in new["runtime_files"]["collection_adapter"]
    assert runtime.PROXY_FILE in new["runtime_files"]["collection_adapter"]


def test_one_family_pass4_canary_is_a_distinct_create_once_operation(tmp_path) -> None:
    selection = _load("task-selection.json")
    task = selection["tasks"][0]
    canary_selection = {**selection, "tasks": [task]}
    canary_selection.pop("sha256")
    from cyber_post_train.jobs import digest

    canary_selection["sha256"] = "sha256:" + digest(canary_selection)
    (tmp_path / "task-selection.json").write_text(json.dumps(canary_selection))

    config = _load("eval-config.json")
    config["name"] = "q38-base-actions-budget-canary-p4-v1"
    config["routes"]["base"]["task_versions"] = [task["task_version_id"]]
    config["collection_runtime"] = _runtime(4)
    plan = runtime.compile_eval(config, relative_to=tmp_path)
    authorization = runtime.build_operation_authorization(plan)
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    prepared = runtime.prepare(config, authorization, private, relative_to=tmp_path)
    directory = Path(prepared["prepared"])

    loaded, exact_authorization = runtime.load(directory)
    assert loaded["planned_cells"] == 4
    assert loaded["pass_k"] == 4
    assert len(runtime.identity_map(loaded)) == 4
    assert exact_authorization == authorization
    assert exact_authorization["campaign_id"] == config["name"]
    assert prepared["submitted"] is False


def test_v3_refuses_missing_completion_budget() -> None:
    config = _load("eval-config.json")
    config["collection_runtime"] = _runtime(200)
    config["collection_runtime"].pop("completion_budget")
    try:
        runtime.compile_eval(config, relative_to=V2)
    except ValueError as error:
        assert "unknown or missing" in str(error)
    else:
        raise AssertionError("v3 must not fall back to the historical shared request counter")
