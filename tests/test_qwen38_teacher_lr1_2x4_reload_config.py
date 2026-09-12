"""The LR1 2x4 reload is an identity-only, zero-update validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-dev-fallback-v1.json"
RELOAD = ROOT / "configs/qualification/qwen38-teacher-lr1-cosine-2x4-reload-dev-v1.json"
SOURCE_PREPARED = PurePosixPath(
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/lr-dev-v2-qualified-inputs-v1/"
    "lr1-2x4-fallback-v1"
)
RELOAD_PREPARED = PurePosixPath(
    "/mnt/sfs/jobs/chris-q38-study-corpora-v1/lr1-2x4-reload-dev-v1/prepared-v1"
)
SEAL = {
    "manifest": (
        "/mnt/sfs/jobs/chris-q38-study-corpora-v1/"
        "lr1-2x4-dev-v1-reload-v1/checkpoint-step-6.json"
    ),
    "sha256": "0cbdb07de2ffabc928ba8b443a59814266653d658f4190aa81f6d9801a6fbcd2",
    "mode": "validate",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_reload_makes_only_the_authorized_source_transform() -> None:
    source, reload = read(SOURCE), read(RELOAD)
    restored = copy.deepcopy(reload)
    restored["name"] = source["name"]
    restored["output_root"] = source["output_root"]
    restored["wandb"]["run_id"] = source["wandb"]["run_id"]
    restored["wandb"]["name"] = source["wandb"]["name"]
    restored["pause_after_step"] = source["pause_after_step"]
    del restored["recovery"]
    assert restored == source


def test_reload_binds_the_verified_seal_and_preserves_full_horizon() -> None:
    source, reload = read(SOURCE), read(RELOAD)
    assert reload["recovery"] == SEAL
    assert "pause_after_step" not in reload
    assert reload["data"] == source["data"]
    assert reload["recipe"] == source["recipe"]
    assert reload["recipe"] == {
        "epochs": 1,
        "batch_size": 8,
        "microbatch_per_gpu": 1,
        "nodes": 2,
        "gpus_per_node": 4,
        "lr": 0.000001,
        "scheduler": "cosine",
        "warmup_ratio": 0.05,
        "max_length": 16384,
        "checkpoint_interval": 6,
        "keep_checkpoints": 3,
        "seed": 42,
    }
    preparation = read(
        ROOT
        / "docs/evidence/qwen38-study/2026-09-12-lr1-2x4-dev-fallback-preparation-v1.json"
    )
    assert preparation["scientific_binding"]["expected_max_steps"] == 76
    assert preparation["scientific_binding"]["expected_corpus_manifest_sha256"] == (
        reload["data"]["manifest_sha256"]
    )


def test_reload_identities_are_new_disjoint_and_high_priority() -> None:
    source, reload = read(SOURCE), read(RELOAD)
    assert reload["name"] == reload["wandb"]["run_id"] == reload["wandb"]["name"]
    assert reload["name"] != source["name"]
    assert reload["output_root"] != source["output_root"]
    assert not PurePosixPath(reload["output_root"]).is_relative_to(
        PurePosixPath(source["output_root"])
    )
    assert not PurePosixPath(source["output_root"]).is_relative_to(
        PurePosixPath(reload["output_root"])
    )
    assert RELOAD_PREPARED != SOURCE_PREPARED
    assert not RELOAD_PREPARED.is_relative_to(SOURCE_PREPARED)
    assert not SOURCE_PREPARED.is_relative_to(RELOAD_PREPARED)
    assert reload["cluster"]["priority"] == "c1"
    assert reload["recipe"]["nodes"] == 2
    assert reload["recipe"]["gpus_per_node"] == 4
    assert reload["recipe"]["nodes"] * reload["recipe"]["gpus_per_node"] == 8


def test_sft_request_rail_keeps_reload_non_requeueing() -> None:
    from training import recovery, sft_runtime
    from training.sft import job_request

    plan = {
        "run_name": "reload",
        "output_root": "/mnt/sfs/jobs/reload",
        "runtime_sha256": hashlib.sha256(Path(sft_runtime.__file__).read_bytes()).hexdigest(),
        "recovery": {},
        "recovery_runtime_sha256": hashlib.sha256(
            Path(recovery.__file__).read_bytes()
        ).hexdigest(),
        "wandb": {
            "entity": "thefleet",
            "project": "cyber-post-train",
            "group": "reload",
            "run_id": "reload",
            "name": "reload",
        },
        "execution": {
            "image": "example.invalid/image@sha256:" + "a" * 64,
            "priority": "c1",
            "resources": {},
        },
        "recipe": {"nodes": 2, "gpus_per_node": 4},
    }
    request = job_request(plan)
    assert request["workers"] == 2
    assert request["gpus_per_worker"] == 4
    assert request["priority_class"] == "c1"
    assert request["requeueIfPreempted"] is False
