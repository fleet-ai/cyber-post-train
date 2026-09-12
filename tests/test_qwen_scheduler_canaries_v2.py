"""The concrete scheduler canaries stay paired, bounded, and dev-only."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/qualification/qwen38-teacher-scheduler-constant-dev-v2.json",
    ROOT / "configs/qualification/qwen38-teacher-scheduler-cosine-dev-v2.json",
)
PROTOCOL = "sha256:3c2c65eed748b16b50ef992327324b1fcaad84b190456c4aefe69c6efe748f75"


def read() -> list[dict]:
    return [json.loads(path.read_text()) for path in CONFIGS]


def test_concrete_scheduler_canaries_are_an_exact_pair() -> None:
    constant, cosine = read()
    assert [
        (value["recipe"]["scheduler"], value["recipe"]["warmup_ratio"])
        for value in (constant, cosine)
    ] == [("constant_with_warmup", 0.0), ("cosine", 0.05)]
    for field in ("name", "output_root"):
        assert len({value[field] for value in (constant, cosine)}) == 2
    assert len({value["wandb"]["run_id"] for value in (constant, cosine)}) == 2


def test_concrete_scheduler_canaries_are_bounded_and_outcome_only() -> None:
    for value in read():
        recipe = value["recipe"]
        assert (recipe["nodes"], recipe["gpus_per_node"]) == (1, 4)
        assert (recipe["batch_size"], recipe["microbatch_per_gpu"]) == (8, 1)
        assert (recipe["lr"], recipe["epochs"]) == (1e-5, 1)
        assert (recipe["checkpoint_interval"], value["pause_after_step"]) == (2, 6)
        assert value["cluster"]["priority"] == "c1"
        assert value["validation_mode"] == "task_outcomes_only"
        assert value["fleet_dev_protocol_sha256"] == PROTOCOL
        assert "scientific_rejection" not in value


def test_concrete_scheduler_canaries_bind_scalar_only_tracking() -> None:
    for value in read():
        wandb = value["wandb"]
        assert wandb["entity"] == "thefleet"
        assert wandb["project"] == "cyber-post-train"
        assert wandb["group"] == "q38-blackbox-sft-dev-scheduler-v2"
        assert wandb["run_id"] == wandb["name"] == value["name"]
