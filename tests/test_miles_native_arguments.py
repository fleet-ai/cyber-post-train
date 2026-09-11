"""Exact-image CPU checks; no model, environment, Ray cluster or optimizer starts."""

import os

import pytest

from training.miles import MilesConfig, arguments


def test_real_qwen_recipe_generates_a_bounded_argument_vector():
    pytest.importorskip("fti.trainers.miles.run_fleet")
    config = MilesConfig(
        name="synthetic-miles-args",
        output_root="/mnt/sfs/jobs/synthetic-miles-args",
        model_root="/mnt/sfs/models/synthetic-hf",
        torch_dist_root="/mnt/sfs/models/synthetic-dist",
        train_data="/mnt/sfs/data/synthetic/train.jsonl",
        dev_data="/mnt/sfs/data/synthetic/dev.jsonl",
        data_manifest="/mnt/sfs/data/synthetic/manifest.json",
        wandb_entity="synthetic",
        wandb_project="synthetic",
        wandb_run_id="synthetic",
    )
    argv = arguments(config)

    def value(key):
        return argv[argv.index("--" + key) + 1]

    assert value("tensor-model-parallel-size") == "4"
    assert value("context-parallel-size") == "2"
    assert value("rollout-num-gpus-per-engine") == "1"
    assert value("rollout-max-context-len") == "98304"
    assert value("num-steps-per-rollout") == "1"
    assert value("global-batch-size") == "2"
    assert value("over-sampling-batch-size") == "1"
    assert value("offload-train-target") == "cpu"
    assert "--check-weight-update-equal" in argv
    assert value("check-weight-update-skip-list") == "visual."
    assert "--sglang-disable-radix-cache" in argv
    assert value("sglang-attention-backend") == "triton"
    assert "--no-save-optim" not in argv and "--no-save-rng" not in argv
    assert "--dynamic-sampling-filter-path" not in argv
    assert "--use-fault-tolerance" not in argv


def test_native_wandb_primary_uses_environment_run_id_without_key(tmp_path, monkeypatch):
    native = pytest.importorskip("miles.utils.tracking_utils.wandb_utils")
    from types import SimpleNamespace

    import wandb

    # Offline synthetic telemetry proves identity propagation without credentials
    # or public uploads. Training requests will require online W&B separately.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_RUN_ID", "synthetic-miles-id")
    monkeypatch.setenv("WANDB_MODE", "offline")
    args = SimpleNamespace(
        use_wandb=True,
        wandb_mode="offline",
        wandb_key=None,
        wandb_host=None,
        wandb_random_suffix=False,
        wandb_group="synthetic",
        wandb_team="synthetic",
        wandb_project="synthetic",
        wandb_dir=str(tmp_path),
        rank=0,
        env_report=None,
    )
    try:
        native.init_wandb_primary(args)
        assert args.wandb_run_id == wandb.run.id == os.environ["WANDB_RUN_ID"]
        assert wandb.run.entity == "synthetic" and wandb.run.project == "synthetic"
        assert wandb.run.settings.mode == "offline"
        assert wandb.run.config["wandb_key"] is None
        wandb.log({"train/step": 0, "train/synthetic": 1.0})
    finally:
        wandb.finish()
