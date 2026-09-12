"""Offline synthetic tests for the private paired Fleet-development estimator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evals.fleet import dev_outcome_estimator as estimator
from training.io import digest_json

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs/evaluation"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


@pytest.fixture
def frozen() -> tuple[dict, dict]:
    protocol = json.loads(
        (CONFIG / "qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json").read_text()
    )
    task_set = json.loads((CONFIG / "qwen38-blackbox-fleet-dev-a-task-set-v1.json").read_text())
    return protocol, task_set


def outcomes(
    protocol: dict,
    task_set: dict,
    *,
    arm_id: str,
    candidate_success,
) -> dict:
    rows = []
    for task_index, task in enumerate(task_set["tasks"]):
        for seed_index, seed in enumerate(protocol["sampling"]["attempt_seeds"]):
            rows.append(
                {
                    "task_version_id": task["task_version_id"],
                    "attempt_seed": seed,
                    "base": {"valid_outcome": True, "full_task_success": False},
                    "candidate": {
                        "valid_outcome": True,
                        "full_task_success": candidate_success(task_index, seed_index),
                    },
                }
            )
    unsigned = {
        "schema": estimator.OUTCOME_SCHEMA,
        "parent_protocol_sha256": protocol["sha256"],
        "task_set_sha256": task_set["sha256"],
        "arms": {
            "base": {
                "arm_id": "base",
                "evaluation_child_sha256": SHA_A,
                "raw_result_manifest_sha256": SHA_B,
            },
            "candidate": {
                "arm_id": arm_id,
                "evaluation_child_sha256": SHA_C,
                "raw_result_manifest_sha256": "sha256:" + arm_id[-1] * 64,
            },
        },
        "rows": rows,
    }
    return {**unsigned, "sha256": digest_json(unsigned)}


def reseal(value: dict) -> dict:
    value["sha256"] = digest_json({key: item for key, item in value.items() if key != "sha256"})
    return value


def test_primary_uses_all_four_fixed_attempts_and_pass_at_4_is_secondary(frozen):
    protocol, task_set = frozen
    value = outcomes(
        protocol,
        task_set,
        arm_id="candidate-1",
        candidate_success=lambda _task, seed: seed == 0,
    )
    result = estimator.estimate(protocol, task_set, value)

    assert result["primary"] == {
        "name": estimator.PRIMARY_METRIC,
        "base_mean_success": 0.0,
        "candidate_mean_success": 0.25,
        "candidate_minus_base_paired_delta": 0.25,
    }
    assert result["pairing"]["task_count"] == 20
    assert result["pairing"]["matched_pair_count"] == 80
    assert result["pairing"]["attempt_seeds"] == [42, 43, 44, 45]
    assert result["secondary_only"] == {
        "name": "fleet_dev_pass_at_4",
        "base": 0.0,
        "candidate": 1.0,
        "candidate_minus_base": 1.0,
        "hyperparameter_or_checkpoint_selection_eligible": False,
    }
    assert result["release_policy"] == {
        "private_only": True,
        "wandb_export_allowed": False,
        "webexploitbench_input_or_tiebreaker": False,
    }
    assert result["sha256"] == digest_json(
        {key: item for key, item in result.items() if key != "sha256"}
    )


def test_selection_ignores_different_pass_at_4_when_primary_is_tied(frozen):
    protocol, task_set = frozen
    widespread = estimator.estimate(
        protocol,
        task_set,
        outcomes(
            protocol,
            task_set,
            arm_id="candidate-1",
            candidate_success=lambda _task, seed: seed == 0,
        ),
    )
    concentrated = estimator.estimate(
        protocol,
        task_set,
        outcomes(
            protocol,
            task_set,
            arm_id="candidate-2",
            candidate_success=lambda task, _seed: task < 5,
        ),
    )
    assert widespread["primary"] == concentrated["primary"]
    assert widespread["secondary_only"]["candidate"] == 1.0
    assert concentrated["secondary_only"]["candidate"] == 0.25

    decision = estimator.select_primary_arm([widespread, concentrated])
    assert decision["winner_arm_ids"] == ["candidate-1", "candidate-2"]
    assert decision["status"] == "tie_requires_fresh_predeclared_confirmation"
    assert decision["pass_at_4_used"] is False
    assert decision["webexploitbench_used"] is False


def test_task_clustered_bootstrap_is_seeded_and_resamples_whole_tasks():
    first = estimator.task_clustered_bootstrap(
        [-1.0, 1.0], resamples=10000, seed=20260911, interval=0.95
    )
    second = estimator.task_clustered_bootstrap(
        [-1.0, 1.0], resamples=10000, seed=20260911, interval=0.95
    )
    assert first == second == [-1.0, 1.0]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "invalid", "extra_seed"])
def test_incomplete_or_unmatched_pairs_cannot_produce_an_estimate(frozen, defect):
    protocol, task_set = frozen
    value = outcomes(
        protocol,
        task_set,
        arm_id="candidate-1",
        candidate_success=lambda _task, _seed: False,
    )
    if defect == "missing":
        value["rows"].pop()
    elif defect == "duplicate":
        value["rows"][-1] = copy.deepcopy(value["rows"][0])
    elif defect == "invalid":
        value["rows"][0]["candidate"] = {
            "valid_outcome": False,
            "full_task_success": None,
        }
    else:
        value["rows"][0]["attempt_seed"] = 99
    reseal(value)
    with pytest.raises(ValueError):
        estimator.estimate(protocol, task_set, value)


def test_selection_requires_the_same_exact_base_outcomes(frozen):
    protocol, task_set = frozen
    first = estimator.estimate(
        protocol,
        task_set,
        outcomes(
            protocol,
            task_set,
            arm_id="candidate-1",
            candidate_success=lambda _task, _seed: False,
        ),
    )
    second_outcomes = outcomes(
        protocol,
        task_set,
        arm_id="candidate-2",
        candidate_success=lambda _task, _seed: True,
    )
    second_outcomes["rows"][0]["base"]["full_task_success"] = True
    reseal(second_outcomes)
    second = estimator.estimate(protocol, task_set, second_outcomes)
    with pytest.raises(ValueError, match="one exact matched base design"):
        estimator.select_primary_arm([first, second])


def test_v1_protocol_is_historical_and_not_accepted_by_corrected_estimator(frozen):
    _v2, task_set = frozen
    v1 = json.loads((CONFIG / "qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json").read_text())
    value = outcomes(
        _v2,
        task_set,
        arm_id="candidate-1",
        candidate_success=lambda _task, _seed: False,
    )
    value["parent_protocol_sha256"] = v1["sha256"]
    reseal(value)
    with pytest.raises(ValueError, match="requires a v2"):
        estimator.estimate(v1, task_set, value)


def test_cli_writes_private_create_once_output_and_prints_only_digest(frozen, tmp_path, capsys):
    protocol, task_set = frozen
    protocol_path = CONFIG / "qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json"
    task_set_path = CONFIG / "qwen38-blackbox-fleet-dev-a-task-set-v1.json"
    outcome_path = tmp_path / "matched.json"
    output_path = tmp_path / "estimate.json"
    outcome_path.write_text(
        json.dumps(
            outcomes(
                protocol,
                task_set,
                arm_id="candidate-1",
                candidate_success=lambda _task, _seed: False,
            )
        )
    )
    argv = [
        "estimate",
        "--protocol",
        str(protocol_path),
        "--task-set",
        str(task_set_path),
        "--outcomes",
        str(outcome_path),
        "--output",
        str(output_path),
    ]
    assert estimator.main(argv) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"output_sha256", "written"}
    assert output_path.stat().st_mode & 0o777 == 0o600
    original = output_path.read_bytes()
    with pytest.raises(FileExistsError):
        estimator.main(argv)
    assert output_path.read_bytes() == original
