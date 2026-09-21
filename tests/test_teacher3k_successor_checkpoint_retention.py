import json
from pathlib import Path

import pytest

from training import sft, sft_runtime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
POLICY = ROOT / "docs" / "evidence" / "qwen38-broad-sft-checkpoint-retention-policy-20260921.json"


SUCCESSORS = (
    (
        "qwen38-teacher3k-32k-full-b8-lr3e6-v2.json",
        "qwen38-teacher3k-32k-full-b8-lr3e6-v3.json",
        53_300,
    ),
    (
        "qwen38-teacher3k-32k-full-b16-lr3e6-v2.json",
        "qwen38-teacher3k-32k-full-b16-lr3e6-v3.json",
        53_300,
    ),
    (
        "qwen38-teacher3k-32k-full-b8-lr1e6-v2.json",
        "qwen38-teacher3k-32k-full-b8-lr1e6-v3.json",
        53_300,
    ),
    (
        "qwen38-teacher3k-64k-full-b8-lr3e6-v4.json",
        "qwen38-teacher3k-64k-full-b8-lr3e6-v5.json",
        17_460,
    ),
)


def _without_successor_fields(config: dict) -> dict:
    normalized = json.loads(json.dumps(config))
    normalized["name"] = "<run-name>"
    normalized["output_root"] = "<output-root>"
    normalized["checkpoint_recovery_horizon_seconds"] = "<recovery-horizon>"
    normalized["recipe"]["keep_checkpoints"] = "<keep-checkpoints>"
    normalized["wandb"]["run_id"] = "<wandb-id>"
    normalized["wandb"]["name"] = "<wandb-name>"
    normalized["wandb"]["tags"] = [
        tag for tag in normalized["wandb"]["tags"] if tag != "bounded-checkpoint-retention"
    ]
    return normalized


@pytest.mark.parametrize(("source_name", "successor_name", "horizon"), SUCCESSORS)
def test_full_weight_successors_bound_retention_without_changing_science(
    source_name, successor_name, horizon
):
    source = json.loads((RUNS / source_name).read_text())
    successor = json.loads((RUNS / successor_name).read_text())

    assert _without_successor_fields(successor) == _without_successor_fields(source)
    assert successor["name"] != source["name"]
    assert successor["output_root"] != source["output_root"]
    assert successor["wandb"]["run_id"] != source["wandb"]["run_id"]
    assert successor["recipe"]["checkpoint_interval"] == source["recipe"]["checkpoint_interval"]
    assert successor["recipe"]["keep_checkpoints"] == 2
    assert source["recipe"]["keep_checkpoints"] == 5
    assert successor["wandb"]["tags"].count("bounded-checkpoint-retention") == 1

    plan = sft.compile_sft(successor, relative_to=RUNS)
    assert plan["checkpoint_recovery_horizon_seconds"] == horizon
    assert sft_runtime.sft_first_checkpoint_seconds(plan) == horizon
    request = sft.job_request(plan)
    assert request["failureAlerts"] is False
    assert request["priority_class"] == "c1"
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8


def test_retention_policy_is_bound_to_the_exact_successor_set():
    policy = json.loads(POLICY.read_text())
    expected = {(source, successor) for source, successor, _ in SUCCESSORS}
    observed = {
        (
            row["source"].removeprefix("configs/runs/"),
            row["successor"].removeprefix("configs/runs/"),
        )
        for row in policy["successors"]
    }

    assert observed == expected
    assert policy["measured_full_checkpoint_size_gib"] == 303
    assert policy["retention"]["latest_checkpoints"] == 2
    assert policy["retention"]["estimated_live_retention_gib"] == 606
    assert policy["retention"]["separate_terminal_or_promoted_milestones"] is True
