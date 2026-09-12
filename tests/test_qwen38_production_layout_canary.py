"""The eight-GPU dev layout canary changes only topology and identities."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-scheduler-cosine-dev-v2.json"
LAYOUT = ROOT / "configs/qualification/qwen38-teacher-production-layout-dev-v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_layout_canary_preserves_the_accepted_cosine_treatment() -> None:
    source, layout = _read(SOURCE), _read(LAYOUT)

    assert layout["model"] == source["model"]
    assert layout["data"] == source["data"]
    assert layout["validation_mode"] == source["validation_mode"]
    assert layout["fleet_dev_protocol_sha256"] == source["fleet_dev_protocol_sha256"]
    assert layout["cluster"] == source["cluster"]
    assert layout["pause_after_step"] == source["pause_after_step"] == 6

    expected_recipe = dict(source["recipe"])
    expected_recipe["gpus_per_node"] = 8
    assert layout["recipe"] == expected_recipe
    assert (layout["recipe"]["scheduler"], layout["recipe"]["warmup_ratio"]) == (
        "cosine",
        0.05,
    )


def test_layout_canary_has_new_create_once_identities() -> None:
    source, layout = _read(SOURCE), _read(LAYOUT)

    assert layout["name"] == "chris-q38-ta8-cos5-dev3"
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", layout["name"])
    assert layout["output_root"] == f"/mnt/sfs/jobs/{layout['name']}"
    assert layout["name"] != source["name"]
    assert layout["output_root"] != source["output_root"]

    wandb = layout["wandb"]
    assert wandb["run_id"] == wandb["name"] == layout["name"]
    assert wandb["group"] == "q38-blackbox-sft-dev-layout-v1"
    assert wandb["run_id"] != source["wandb"]["run_id"]
    assert {"production-layout", "scheduler-cosine", "warmup-5pct"} <= set(wandb["tags"])


def test_layout_canary_is_one_eight_gpu_c1_worker_without_auto_requeue_input() -> None:
    layout = _read(LAYOUT)

    recipe = layout["recipe"]
    assert (recipe["nodes"], recipe["gpus_per_node"]) == (1, 8)
    assert (recipe["batch_size"], recipe["microbatch_per_gpu"]) == (8, 1)
    assert (recipe["checkpoint_interval"], recipe["keep_checkpoints"]) == (2, 3)
    assert layout["cluster"]["priority"] == "c1"
    assert "queue_priority_class" not in layout["cluster"]
    assert "requeueIfPreempted" not in layout
