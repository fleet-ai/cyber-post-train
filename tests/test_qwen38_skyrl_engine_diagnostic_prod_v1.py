"""Exact prod fallback boundary for the capacity-blocked dev11 engine gate."""

import copy
import json
from pathlib import Path

import pytest

from cyber_post_train.jobs import digest
from training import skyrl_training

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-prod-v1.json"
)
RUN_CONFIG = ROOT / (
    "configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-prod-v1.json"
)


def test_prod_fallback_source_config_is_exact_c1_zero_work() -> None:
    data = json.loads(DATA_CONFIG.read_bytes())
    run = json.loads(RUN_CONFIG.read_bytes())

    assert skyrl_training.is_prod_engine_diagnostic_config(run)
    assert digest(run) == skyrl_training.ENGINE_DIAGNOSTIC_PROD_CONFIG_SHA256
    assert run["name"] == data["name"] == skyrl_training.ENGINE_DIAGNOSTIC_PROD_CONFIG_NAME
    assert run["output_root"] == skyrl_training.ENGINE_DIAGNOSTIC_PROD_OUTPUT_ROOT
    assert run["data"]["root"] == data["output"] == skyrl_training.ENGINE_DIAGNOSTIC_PROD_DATA_ROOT
    assert run["cluster"]["target"] == "prod"
    assert run["cluster"]["priority"] == "c1"
    assert {key: run["recipe"][key] for key in ("nodes", "steps", "groups", "samples_per_prompt")} == {
        "nodes": 1,
        "steps": 1,
        "groups": 2,
        "samples_per_prompt": 4,
    }


def test_prod_fallback_source_config_rejects_any_drift() -> None:
    run = json.loads(RUN_CONFIG.read_bytes())
    changed = copy.deepcopy(run)
    changed["cluster"]["priority"] = "c2"
    assert not skyrl_training.is_prod_engine_diagnostic_config(changed)


def test_partial_prod_fallback_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="prod1 engine diagnostic identity is incomplete"):
        skyrl_training.is_prod_engine_diagnostic(
            {"run_name": skyrl_training.ENGINE_DIAGNOSTIC_PROD_CONFIG_NAME}
        )


def test_unrelated_prod_plan_is_not_the_fallback() -> None:
    assert not skyrl_training.is_prod_engine_diagnostic(
        {"run_name": "other", "execution": {"cluster_target": "prod"}}
    )
