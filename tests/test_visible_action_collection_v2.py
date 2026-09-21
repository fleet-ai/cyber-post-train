"""Exactly-once regressions for the visible-action collection v2 rail."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import pytest

from cyber_post_train.jobs import digest
from evals.fleet import visible_action_collection_v2 as runtime

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "configs/collection/qwen38-base-current75-actions-pass4-v1"


def _config() -> dict:
    value = json.loads((V1 / "eval-config.json").read_text())
    old = value["collection_runtime"]
    value["collection_runtime"] = {
        "schema": runtime.RUNTIME_SCHEMA,
        "source_template_sha256": old["source_template_sha256"],
        "reasoning_generation": runtime.THINKING_DISABLED,
        "reasoning_request_override": {"chat_template_kwargs": {"enable_thinking": False}},
        "opencode_model_reasoning": False,
        "opencode_cli_thinking_flag": False,
        "maximum_planned_cells": old["maximum_planned_cells"],
        "operation_authorization_required": True,
        "canonical_private_operation_root_required": True,
        "exclusive_pre_mutation_intent_required": True,
        "dedicated_empty_ledger_required": True,
        "automatic_replay_of_ambiguous_cells": False,
        "external_submission": False,
        "execution_mode": runtime.EXECUTION_MODE,
        "cluster_wrapper_supported": False,
    }
    return value


def _prepared(tmp_path: Path) -> tuple[Path, dict, dict]:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    config = _config()
    plan = runtime.compile_eval(config, relative_to=V1)
    authorization = runtime.build_operation_authorization(plan)
    result = runtime.prepare(config, authorization, private, relative_to=V1)
    directory = Path(result["prepared"])
    proof = {
        "schema": runtime.PREFLIGHT_SCHEMA,
        "plan_sha256": plan["sha256"],
        "operation_authorization_sha256": authorization["sha256"],
        "task_bindings": {
            task["task_version_id"]: ["synthetic-offline-binding"] for task in plan["tasks"]
        },
        "routes": {name: {"synthetic": True} for name in plan["routes"]},
        "images": plan["images"],
        "reasoning_generation": runtime.THINKING_DISABLED,
        "model_calls": 0,
    }
    proof["sha256"] = digest(proof)
    (directory / "EVAL_PREFLIGHT.json").write_text(json.dumps(proof))
    return directory, plan, authorization


def _created(authorization: dict) -> dict:
    return {
        "created": True,
        "cells": authorization["planned_cells"],
        "plan_sha256": "0" * 64,
        "operation_authorization_sha256": authorization["sha256"],
        "dedicated_ledger_id": authorization["dedicated_ledger_id"],
    }


def test_v2_authorization_binds_every_deterministic_identity() -> None:
    plan = runtime.compile_eval(_config(), relative_to=V1)
    rows = runtime.identity_map(plan)
    authorization = runtime.build_operation_authorization(plan)

    assert plan["schema"] == runtime.PLAN_SCHEMA
    assert plan["campaign_id"] == "q38-base-train50-actions-p4-v1"
    assert len(rows) == 200
    assert len({row["ledger_cell_id"] for row in rows}) == 200
    assert len({row["scientific_cell_id"] for row in rows}) == 200
    assert len({row["execution_id"] for row in rows}) == 200
    assert authorization["identity_map_sha256"] == "sha256:" + digest(rows)
    assert authorization["identity_derivation"] == {
        "ledger_cell_id": runtime.LEDGER_CELL_DERIVATION,
        "scientific_cell_id": runtime.SCIENTIFIC_CELL_DERIVATION,
        "execution_id": runtime.EXECUTION_DERIVATION,
        "execution_generation": 1,
    }


def test_same_path_and_alternate_path_prepare_are_refused(tmp_path: Path) -> None:
    directory, plan, authorization = _prepared(tmp_path)
    with pytest.raises(FileExistsError, match="canonical collection operation root"):
        runtime.prepare(_config(), authorization, directory.parent, relative_to=V1)

    alternate = directory.parent / "alternate-operation-root"
    shutil.copytree(directory, alternate)
    with pytest.raises(ValueError, match="outside its canonical private root"):
        runtime.load(alternate)
    assert runtime.validate_operation_authorization(authorization, plan) == authorization


def test_ambiguous_initialize_writes_intent_and_never_replays(tmp_path: Path) -> None:
    directory, _plan, authorization = _prepared(tmp_path)
    calls = 0

    def ambiguous(_dsn: str, _plan_path: Path, _authorization: dict) -> dict:
        nonlocal calls
        calls += 1
        raise TimeoutError("synthetic uncertain response after possible commit")

    with pytest.raises(TimeoutError, match="uncertain"):
        runtime.initialize_once(directory, dsn="postgresql://redacted", initializer=ambiguous)
    intent = json.loads((directory / runtime.CREATE_INTENT_FILE).read_text())
    assert intent["state"] == "CREATE_INTENT_DO_NOT_RETRY"
    assert intent["operation_authorization_sha256"] == authorization["sha256"]
    assert not (directory / runtime.LEDGER_RECEIPT_FILE).exists()

    with pytest.raises(RuntimeError, match="reconcile, never retry"):
        runtime.initialize_once(directory, dsn="postgresql://redacted", initializer=ambiguous)
    assert calls == 1


def test_concurrent_initializers_have_one_external_mutation_winner(tmp_path: Path) -> None:
    directory, _plan, authorization = _prepared(tmp_path)
    barrier = Barrier(2)
    calls = 0
    lock = Lock()

    def initialize(_dsn: str, _plan_path: Path, _authorization: dict) -> dict:
        nonlocal calls
        with lock:
            calls += 1
        return _created(authorization)

    def contender() -> object:
        barrier.wait()
        try:
            return runtime.initialize_once(
                directory, dsn="postgresql://redacted", initializer=initialize
            )
        except Exception as error:  # return the exact losing boundary for assertion
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: contender(), range(2)))

    assert calls == 1
    assert sum(isinstance(outcome, dict) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, (FileExistsError, RuntimeError)) for outcome in outcomes) == 1


def test_successful_initialize_and_receipts_refuse_overwrite(tmp_path: Path) -> None:
    directory, plan, authorization = _prepared(tmp_path)
    receipt = runtime.initialize_once(
        directory,
        dsn="postgresql://redacted",
        initializer=lambda *_args: _created(authorization),
    )
    assert receipt["created_cells"] == plan["planned_cells"]
    assert receipt["operation_authorization_sha256"] == authorization["sha256"]
    assert runtime._validated_ledger_receipt(directory, plan, authorization) == receipt  # noqa: SLF001

    with pytest.raises(RuntimeError, match="reconcile, never retry"):
        runtime.initialize_once(
            directory,
            dsn="postgresql://redacted",
            initializer=lambda *_args: pytest.fail("existing intent reached initializer"),
        )


def test_changed_authorization_is_rejected_before_operation_use(tmp_path: Path) -> None:
    directory, _plan, _authorization = _prepared(tmp_path)
    changed = json.loads((directory / runtime.OPERATION_AUTHORIZATION_FILE).read_text())
    changed["dedicated_ledger_id"] = "sha256:" + "f" * 64
    changed["sha256"] = "sha256:" + digest(
        {key: item for key, item in changed.items() if key != "sha256"}
    )
    (directory / runtime.OPERATION_AUTHORIZATION_FILE).write_text(json.dumps(changed))

    with pytest.raises(ValueError, match="differs from the exact campaign identity"):
        runtime.load(directory)
