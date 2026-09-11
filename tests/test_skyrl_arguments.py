"""Pure configuration checks plus explicitly native-image-only contract tests."""

import copy
import importlib.util
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from training import skyrl


@pytest.fixture
def config():
    return skyrl.SkyRLConfig(
        name="synthetic-rl",
        output_root="/mnt/sfs/jobs/synthetic-rl",
        model_root="/mnt/sfs/models/synthetic-hf",
        train_data="/mnt/sfs/data/synthetic/train.jsonl",
        dev_data="/mnt/sfs/data/synthetic/dev.jsonl",
        data_manifest="/mnt/sfs/data/synthetic/manifest.json",
        train_rows=8,
        dev_rows=2,
        wandb_entity="synthetic",
        wandb_project="synthetic",
        wandb_run_id="synthetic",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "zai-org/GLM-5.3"},
        {"model": "zai-org/GLM-5.3-Flash"},
        {"model": "Qwen/Qwen3.6-27B"},
        {"name": "--flag"},
        {"wandb_entity": "bad/name"},
        {"wandb_project": "bad name"},
        {"wandb_run_id": "\n"},
        {"nodes": 0},
        {"nodes": 3},
        {"nodes": True},
        {"steps": 0},
        {"steps": 1.5},
        {"groups": -1},
        {"groups": 16},
        {"train_rows": 7},
        {"dev_rows": 0},
        {"samples_per_prompt": 1},
        {"samples_per_prompt": 3},
        {"nodes": 2},
        {"checkpoint_interval": False},
        {"keep_checkpoints": -1},
        {"eval_interval": 0},
        {"seed": -1},
        {"seed": True},
        {"lr": 0},
        {"lr": -1},
        {"lr": float("nan")},
        {"lr": float("inf")},
        {"lr": True},
        {"context_tokens": 100000},
        {"context_tokens": 81920},
        {"response_tokens": 1},
        {"tokens_per_turn": 0},
        {"max_turns": 0},
        {"output_root": "/mnt/sfs/models/output"},
        {"model_root": "/"},
        {"model_root": "/mnt/sfs/models/../source"},
        {"model_root": "/mnt/sfs/models/source/"},
        {"model_root": "/mnt/sfs/models/line\nbreak"},
        {"dev_data": "/mnt/sfs/data/synthetic/train.jsonl"},
        {"output_root": "/mnt/sfs/jobs/data", "train_data": "/mnt/sfs/jobs/data/train"},
        {"train_data": "/mnt/sfs/jobs/data", "output_root": "/mnt/sfs/jobs/data/output"},
    ],
)
def test_invalid_config_stops_before_native_import(config, changes):
    with pytest.raises(ValueError):
        skyrl.overrides(replace(config, **changes))


@pytest.mark.parametrize("nodes,groups,repetitions", [(1, 2, 4), (1, 1, 8), (2, 4, 4)])
def test_bounded_native_recipe(config, nodes, groups, repetitions):
    config = replace(config, nodes=nodes, groups=groups, samples_per_prompt=repetitions, steps=9)
    values = skyrl.overrides(config)
    assert values["trainer.epochs"] * (config.train_rows // groups) >= 9
    assert values["trainer.max_training_steps"] == 9
    assert values["trainer.train_batch_size"] == values["trainer.policy_mini_batch_size"] == groups
    assert values["trainer.update_epochs_per_batch"] == 1
    assert values["generator.n_samples_per_prompt"] == repetitions
    assert values["trainer.placement.policy_num_nodes"] == nodes
    assert values["generator.inference_engine.num_engines"] * 4 == nodes * 8
    assert values["trainer.ckpt_interval"] == 1
    assert values["trainer.eval_before_train"] is True
    assert values["trainer.eval_batch_size"] == 2
    assert values["trainer.resume_mode"] == "none"
    assert values["trainer.max_ckpts_to_keep"] == 2
    assert values["trainer.policy.optimizer_config.lr"] == config.lr
    assert values["generator.sampling_params"]["logprobs"] == 1
    assert values["generator.eval_sampling_params"]["temperature"] == 0
    assert values["trainer.max_prompt_length"] == config.context_tokens - config.response_tokens
    assert (
        values["generator.inference_engine.engine_init_kwargs.max_model_len"]
        == config.context_tokens
    )
    assert values["trainer.algorithm.dynamic_sampling.type"] is None
    for key in (
        "trainer.algorithm.zero_variance_filter",
        "generator.zero_reward_on_non_stop",
        "generator.apply_overlong_filtering",
        "trainer.dump_eval_results",
        "trainer.dump_data_batch",
    ):
        assert values[key] is False
    for kind in ("train", "eval"):
        assert values[f"trainer.num_logger_{kind}_samples"] == 0
    assert values["trainer.print_example_interval"] == values["trainer.log_example_interval"] == -1
    assert "WANDB_API_KEY" not in str(values) and "FLEET_API_KEY" not in str(values)
    before = copy.deepcopy(values)
    values["generator.eval_sampling_params"]["temperature"] = 0.1
    assert values["generator.sampling_params"] == before["generator.sampling_params"]


def test_native_boundary_does_not_rewrite_config(config, monkeypatch):
    from training import skyrl_episode

    result, calls = NS(), []

    def module(name, sha):
        assert skyrl.NATIVE_SOURCES[name] == sha
        if name.endswith("config"):
            return NS(
                SkyRLTrainConfig=NS(
                    from_cli_overrides=lambda values: (calls.append(values), result)[1]
                )
            )
        return NS(validate_cfg=lambda cfg: calls.append(cfg))

    monkeypatch.setattr(skyrl_episode, "_module", module)
    assert skyrl.native_config(config) is result
    assert calls == [skyrl.overrides(config), result]


@pytest.mark.skipif(importlib.util.find_spec("skyrl") is None, reason="pinned SkyRL image only")
@pytest.mark.parametrize("steps,interval", [(1, 1), (9, 2)])
def test_real_native_config_and_optimizer_step_semantics(config, monkeypatch, steps, interval):
    # This tests actual native methods using synthetic batches, not GPU optimization.
    monkeypatch.setenv("WANDB_API_KEY", "synthetic-offline-test-only")
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = skyrl.native_config(replace(config, steps=steps, eval_interval=interval))
    assert cfg.trainer.max_training_steps == steps
    assert cfg.trainer.algorithm.temperature == cfg.generator.sampling_params.temperature == 1
    assert not cfg.trainer.policy.inference_only_init
    assert not cfg.trainer.fully_async.simulate_training
    assert cfg.trainer.policy.model.lora.rank == 0
    assert cfg.trainer.policy.fsdp_config.cpu_offload is False
    assert cfg.trainer.policy.optimizer_config.offload_after_step is True
    from skyrl.train.trainer import RayPPOTrainer

    calls = []
    dispatch = NS(
        stage_data=lambda model, data, boundaries: ["one-global-mini-batch"],
        forward_backward_from_staged=lambda model, chunk: (
            calls.append(("forward_backward", model, chunk)),
            NS(metrics={"loss": 0.5}),
        )[1],
        optim_step=lambda model: (calls.append(("optimizer", model)), 1.0)[1],
    )
    instance = NS(
        cfg=cfg, dispatch=dispatch, _normalize_advantages=lambda data, boundaries, prompts: data
    )
    data = NS(metadata={"policy_mini_batch_boundaries": [(0, 8)]})
    output = RayPPOTrainer._execute_training_step(instance, "policy", data)
    assert calls == [
        ("forward_backward", "policy", "one-global-mini-batch"),
        ("optimizer", "policy"),
    ]
    assert output["loss"] == 0.5 and output["grad_norm"] == 1.0
