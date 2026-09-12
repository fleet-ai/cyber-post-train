"""Concrete scheduler reload configs preserve the accepted source treatment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    (
        ROOT / "configs/qualification/qwen38-teacher-scheduler-constant-dev-v2.json",
        ROOT / "configs/qualification/qwen38-teacher-scheduler-constant-reload-dev-v2.json",
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/scheduler-reload-dev2/constant-step-6.json",
        "7a04f5b1257a9b3d13a870cc5116d9c91b68e5221c90085dd2a90e9299d63867",
    ),
    (
        ROOT / "configs/qualification/qwen38-teacher-scheduler-cosine-dev-v2.json",
        ROOT / "configs/qualification/qwen38-teacher-scheduler-cosine-reload-dev-v2.json",
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/scheduler-reload-dev2/cosine-step-6.json",
        "f727971a29354db947a8cd2b011076ead4cb269b2531e62ed39ed9922b7a9a02",
    ),
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_reload_configs_make_only_the_authorized_source_transform() -> None:
    for source_path, reload_path, _, _ in PAIRS:
        source, reload = _read(source_path), _read(reload_path)
        restored = copy.deepcopy(reload)
        restored["name"] = source["name"]
        restored["output_root"] = source["output_root"]
        restored["wandb"]["run_id"] = source["wandb"]["run_id"]
        restored["wandb"]["name"] = source["wandb"]["name"]
        restored["pause_after_step"] = source["pause_after_step"]
        del restored["recovery"]
        assert restored == source


def test_reload_configs_bind_their_own_exact_step_six_seal() -> None:
    for _, reload_path, manifest, sha256 in PAIRS:
        value = _read(reload_path)
        assert value["recovery"] == {
            "manifest": manifest,
            "sha256": sha256,
            "mode": "validate",
        }
        assert len(sha256) == 64
        assert value["recipe"]["gpus_per_node"] == 4
        assert value["recipe"]["nodes"] == 1
        assert value["cluster"]["priority"] == "c1"
        assert value["validation_mode"] == "task_outcomes_only"
        assert "pause_after_step" not in value


def test_reload_run_output_and_wandb_identities_are_new_and_disjoint() -> None:
    sources = [_read(source) for source, _, _, _ in PAIRS]
    reloads = [_read(reload) for _, reload, _, _ in PAIRS]
    for source, reload in zip(sources, reloads, strict=True):
        assert reload["name"] != source["name"]
        assert reload["output_root"] != source["output_root"]
        assert reload["wandb"]["run_id"] != source["wandb"]["run_id"]
        assert reload["wandb"]["name"] != source["wandb"]["name"]
        assert reload["name"] == reload["wandb"]["run_id"] == reload["wandb"]["name"]
        assert not Path(reload["output_root"]).is_relative_to(source["output_root"])
        assert not Path(source["output_root"]).is_relative_to(reload["output_root"])
    assert len({value["name"] for value in sources + reloads}) == 4
    assert len({value["output_root"] for value in sources + reloads}) == 4
    assert len({value["wandb"]["run_id"] for value in sources + reloads}) == 4


def test_reload_pair_cannot_cross_bind_scheduler_checkpoints() -> None:
    constant, cosine = [_read(reload) for _, reload, _, _ in PAIRS]
    assert constant["recipe"]["scheduler"] == "constant_with_warmup"
    assert constant["recipe"]["warmup_ratio"] == 0.0
    assert constant["recovery"]["manifest"].endswith("/constant-step-6.json")
    assert cosine["recipe"]["scheduler"] == "cosine"
    assert cosine["recipe"]["warmup_ratio"] == 0.05
    assert cosine["recovery"]["manifest"].endswith("/cosine-step-6.json")
    assert constant["recovery"] != cosine["recovery"]
