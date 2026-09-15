"""The eight-rank dev reload changes identity only and binds its own seal."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-production-layout-dev-v1.json"
RELOAD = ROOT / "configs/qualification/qwen38-teacher-production-layout-reload-dev-v1.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_production_layout_reload_preserves_the_source_treatment() -> None:
    source, reload = _read(SOURCE), _read(RELOAD)
    restored = copy.deepcopy(reload)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["wandb"]["run_id"] = source["wandb"]["run_id"]
    restored["wandb"]["name"] = source["wandb"]["name"]
    restored["pause_after_step"] = source["pause_after_step"]
    del restored["recovery"]
    assert restored == source


def test_production_layout_reload_is_zero_update_eight_rank_dev_recovery() -> None:
    source, reload = _read(SOURCE), _read(RELOAD)
    assert reload["name"] == reload["wandb"]["run_id"] == reload["wandb"]["name"]
    assert reload["name"] != source["name"]
    assert reload["output_root"] != source["output_root"]
    assert not Path(reload["output_root"]).is_relative_to(source["output_root"])
    assert reload["recipe"]["nodes"] == 1
    assert reload["recipe"]["gpus_per_node"] == 8
    assert reload["cluster"]["priority"] == "c1"
    assert reload["recovery"] == {
        "manifest": (
            "/mnt/sfs/jobs/chris-q38-study-corpora-v1/"
            "ta8-cos5-dev3-reload-v1/checkpoint-step-6.json"
        ),
        "sha256": "7b04d753c77855551ae3c3b1a9cf674d55aba609f0b5f82c85b54c1f29a85be4",
        "mode": "validate",
    }
    assert "pause_after_step" not in reload
