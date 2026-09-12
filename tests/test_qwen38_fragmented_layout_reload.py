"""The fragmented-capacity reload changes only rank placement and identities."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-production-layout-reload-dev-v1.json"
SUCCESSOR = (
    ROOT / "configs/qualification/qwen38-teacher-production-layout-reload-fragmented-dev-v2.json"
)
MANIFEST = "/mnt/sfs/jobs/chris-q38-study-corpora-v1/ta8-cos5-dev3-reload-v1/checkpoint-step-6.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_fragmented_reload_changes_only_topology_identities_and_layout_tag() -> None:
    source, successor = _read(SOURCE), _read(SUCCESSOR)
    restored = copy.deepcopy(successor)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["wandb"]["run_id"] = source["wandb"]["run_id"]
    restored["wandb"]["name"] = source["wandb"]["name"]
    restored["wandb"]["tags"] = source["wandb"]["tags"]
    restored["recipe"]["nodes"] = source["recipe"]["nodes"]
    restored["recipe"]["gpus_per_node"] = source["recipe"]["gpus_per_node"]
    assert restored == source


def test_fragmented_reload_preserves_exact_eight_rank_checkpoint_compatibility() -> None:
    source, successor = _read(SOURCE), _read(SUCCESSOR)
    assert (source["recipe"]["nodes"], source["recipe"]["gpus_per_node"]) == (1, 8)
    assert (successor["recipe"]["nodes"], successor["recipe"]["gpus_per_node"]) == (
        2,
        4,
    )
    assert source["recipe"]["nodes"] * source["recipe"]["gpus_per_node"] == 8
    assert successor["recipe"]["nodes"] * successor["recipe"]["gpus_per_node"] == 8
    assert (
        successor["recovery"]
        == source["recovery"]
        == {
            "manifest": MANIFEST,
            "sha256": "7b04d753c77855551ae3c3b1a9cf674d55aba609f0b5f82c85b54c1f29a85be4",
            "mode": "validate",
        }
    )


def test_fragmented_reload_uses_new_c1_nonrequeued_run_identities() -> None:
    source, successor = _read(SOURCE), _read(SUCCESSOR)
    assert successor["name"] == "chris-q38-ta8-cos5-reld4"
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]{0,29}[a-z0-9])?", successor["name"])
    assert successor["name"] != source["name"]
    assert successor["output_root"] != source["output_root"]
    assert not Path(successor["output_root"]).is_relative_to(source["output_root"])
    assert not Path(source["output_root"]).is_relative_to(successor["output_root"])
    assert successor["name"] == successor["wandb"]["run_id"] == successor["wandb"]["name"]
    assert successor["wandb"]["run_id"] != source["wandb"]["run_id"]
    assert "fragmented-dev-layout" in successor["wandb"]["tags"]
    assert "production-layout" not in successor["wandb"]["tags"]
    assert successor["cluster"]["priority"] == "c1"
    assert "queue_priority_class" not in successor["cluster"]
    assert "requeueIfPreempted" not in successor
