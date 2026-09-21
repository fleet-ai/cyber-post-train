import json
from pathlib import Path

import pytest

from training import sft

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "configs" / "runs"
INCIDENT = ROOT / "docs" / "evidence" / "qwen38-broad-sft-eight-hour-runtime-bound-20260921.json"


SUCCESSORS = (
    (
        "qwen38-teacher3k-32k-full-b8-lr3e6-v1.json",
        "qwen38-teacher3k-32k-full-b8-lr3e6-v2.json",
        "32k_batch8_lr3e-6_reference",
        100,
    ),
    (
        "qwen38-teacher3k-32k-full-b16-lr3e6-v1.json",
        "qwen38-teacher3k-32k-full-b16-lr3e6-v2.json",
        "32k_batch16_lr3e-6_control",
        50,
    ),
    (
        "qwen38-teacher3k-32k-full-b8-lr1e6-v1.json",
        "qwen38-teacher3k-32k-full-b8-lr1e6-v2.json",
        "32k_batch8_lr1e-6_control",
        100,
    ),
    (
        "qwen38-teacher3k-64k-full-b8-lr3e6-v3.json",
        "qwen38-teacher3k-64k-full-b8-lr3e6-v4.json",
        "64k_batch8_lr3e-6_context_control",
        15,
    ),
)


def _without_successor_fields(config: dict) -> dict:
    normalized = json.loads(json.dumps(config))
    normalized["name"] = "<run-name>"
    normalized["output_root"] = "<output-root>"
    normalized["recipe"]["checkpoint_interval"] = "<checkpoint-interval>"
    normalized["wandb"]["run_id"] = "<wandb-id>"
    normalized["wandb"]["name"] = "<wandb-name>"
    normalized["wandb"]["tags"] = [
        tag for tag in normalized["wandb"]["tags"] if tag != "recovery-checkpoint-cadence"
    ]
    return normalized


@pytest.mark.parametrize(("source_name", "successor_name", "arm", "interval"), SUCCESSORS)
def test_broad_sft_checkpoint_successors_change_only_identity_and_recovery_cadence(
    source_name, successor_name, arm, interval
):
    del arm
    source = json.loads((RUNS / source_name).read_text())
    successor = json.loads((RUNS / successor_name).read_text())

    assert _without_successor_fields(successor) == _without_successor_fields(source)
    assert successor["name"] != source["name"]
    assert successor["output_root"] != source["output_root"]
    assert successor["wandb"]["run_id"] != source["wandb"]["run_id"]
    assert successor["recipe"]["checkpoint_interval"] == interval
    assert successor["recipe"]["keep_checkpoints"] == 5
    assert successor["recipe"]["checkpoint_interval"] < source["recipe"]["checkpoint_interval"]
    assert successor["wandb"]["tags"].count("recovery-checkpoint-cadence") == 1

    plan = sft.compile_sft(successor, relative_to=RUNS)
    request = sft.job_request(plan)
    assert request["failureAlerts"] is False
    assert request["priority_class"] == "c1"
    assert request["workers"] == 1
    assert request["gpus_per_worker"] == 8


@pytest.mark.parametrize(("source_name", "successor_name", "arm", "interval"), SUCCESSORS)
def test_broad_sft_first_checkpoint_is_within_historical_three_and_a_half_hour_window(
    source_name, successor_name, arm, interval
):
    del source_name, successor_name
    incident = json.loads(INCIDENT.read_text())
    old_limit = incident["defect"]["old_hard_runtime_seconds"]
    observed = next(row for row in incident["runs"] if row["arm"] == arm)

    estimated_checkpoint_seconds = old_limit * interval / observed["last_optimizer_step"]
    assert interval < observed["first_checkpoint_step"]
    assert estimated_checkpoint_seconds <= 3.5 * 60 * 60
