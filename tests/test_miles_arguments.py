"""Pure plan tests; native-image tests separately verify the real recipe."""

import hashlib
import sys
from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from training import miles


@pytest.fixture
def config():
    return miles.MilesConfig(
        name="synthetic-rl",
        output_root="/mnt/sfs/jobs/synthetic-rl",
        model_root="/mnt/sfs/models/synthetic-hf",
        torch_dist_root="/mnt/sfs/models/synthetic-dist",
        train_data="/mnt/sfs/data/synthetic/train.jsonl",
        dev_data="/mnt/sfs/data/synthetic/dev.jsonl",
        data_manifest="/mnt/sfs/data/synthetic/manifest.json",
        wandb_entity="synthetic",
        wandb_project="synthetic",
        wandb_run_id="synthetic",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"model": "zai-org/GLM-5.3"},
        {"model": "zai-org/GLM-5.3-Flash"},
        {"name": "--flag"},
        {"wandb_entity": "bad/name"},
        {"wandb_project": "bad name"},
        {"wandb_run_id": "\n"},
        {"nodes": 0},
        {"nodes": 9},
        {"nodes": True},
        {"nodes": 3, "groups": 2, "samples_per_prompt": 2},
        {"nodes": 4, "groups": 3, "samples_per_prompt": 2},
        {"steps": 0},
        {"steps": 1.5},
        {"groups": -1},
        {"samples_per_prompt": 1},
        {"checkpoint_interval": False},
        {"eval_interval": 0},
        {"steps": 9, "eval_interval": 2},
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
def test_invalid_plan_stops_before_import_or_launch(config, changes):
    with pytest.raises(ValueError):
        miles.arguments(replace(config, **changes))


@pytest.fixture
def native_boundary(tmp_path, monkeypatch):
    # Local-only stand-in to cover drift handling, not model qualification.
    template = tmp_path / "synthetic.jinja"
    template.write_text("synthetic template")
    monkeypatch.setattr(miles, "TEMPLATE_SHA256", hashlib.sha256(template.read_bytes()).hexdigest())
    profile = NS(
        backend="megatron",
        vision=False,
        tito_model="qwen35",
        chat_template=template.name,
        megatron_model_type="qwen3.8-27B",
        parallel_args_by_shape={
            (1, 8): "--tensor-model-parallel-size 4 --context-parallel-size 2",
            (2, 8): "--tensor-model-parallel-size 4 --context-parallel-size 2",
        },
        extra_train_args="--offload-train-target cpu",
        extra_sglang_args="--sglang-disable-radix-cache",
        rollout_num_gpus_per_engine=1,
        sglang_mem_fraction_static=0.8,
        max_tokens_per_gpu=49152,
        log_prob_pass_multiplier=2,
    )
    runtime = ModuleType("fti.trainers.miles.run_fleet")
    runtime.TEMPLATES, runtime._RECIPES = tmp_path, {"qwen3.8-27b": profile}
    model = ModuleType("miles.utils.external_utils.model_args_utils")
    model.load_model_args = lambda name: "--num-layers 64"
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(sys.modules, model.__name__, model)
    return profile, template


def value(argv, flag):
    return argv[argv.index("--" + flag) + 1]


def test_bounded_counts_and_native_optimizer(config, native_boundary):
    cfg = replace(config, nodes=2, steps=9, groups=4, samples_per_prompt=4, lr=3e-6)
    argv = miles.arguments(cfg)
    assert value(argv, "num-rollout") == "9"
    assert value(argv, "num-steps-per-rollout") == "1"
    assert value(argv, "rollout-batch-size") == value(argv, "over-sampling-batch-size") == "4"
    assert value(argv, "global-batch-size") == "16"
    assert value(argv, "lr") == "3e-06"
    assert value(argv, "eval-prompt-data") == "fleet-dev" and cfg.dev_data in argv
    assert value(argv, "custom-generate-function-path") == "training.rl_episode.generate"
    assert (
        value(argv, "rollout-function-path")
        == value(argv, "eval-function-path")
        == "training.miles_rollout.Rollout"
    )
    assert value(argv, "cyber-data-manifest") == cfg.data_manifest
    assert "--use-wandb" in argv and "--wandb-key" not in argv
    assert "--dynamic-sampling-filter-path" not in argv and "--use-fault-tolerance" not in argv
    assert "--no-save-optim" not in argv and "--no-save-rng" not in argv
    assert value(argv, "load") == value(argv, "save") == cfg.output_root + "/checkpoints"
    assert value(argv, "ref-load") == cfg.torch_dist_root
    assert value(argv, "fleet-tito-model") == "qwen35"
    assert value(argv, "rollout-max-prompt-len") == str(cfg.context_tokens - cfg.response_tokens)
    assert value(argv, "rollout-seed") == str(cfg.seed)


@pytest.mark.parametrize("fault", ["template", "backend", "vision", "tito", "overlap", "retry"])
def test_native_drift_is_rejected(config, native_boundary, fault):
    profile, template = native_boundary
    if fault == "template":
        template.write_text("changed")
    else:
        key, val = {
            "backend": ("backend", "fsdp"),
            "vision": ("vision", True),
            "tito": ("tito_model", "different"),
            "overlap": ("extra_train_args", "--num-rollout 900"),
            "retry": ("extra_train_args", "--use-fault-tolerance"),
        }[fault]
        setattr(profile, key, val)
    with pytest.raises(ValueError):
        miles.arguments(config)


def test_paths_remain_literal_argv(config, native_boundary):
    path = "/mnt/sfs/data/synthetic/a space.jsonl"
    assert value(miles.arguments(replace(config, train_data=path)), "prompt-data") == path


def test_unknown_configuration_fields_rejected(config):
    with pytest.raises(TypeError):
        miles.MilesConfig(**config.__dict__, extra_args="--arbitrary-override")


def test_whole_node_layout_is_data_parallel_over_one_replica(config):
    layout = miles.topology(replace(config, nodes=4, groups=4, samples_per_prompt=4))
    assert layout["gpus"] == 32 and layout["data_parallel_size"] == 4
    assert layout["global_batch_size"] == layout["concurrent_train_environments"] == 16


def test_multi_node_arguments_carry_the_requested_whole_nodes(config, native_boundary):
    cfg = replace(config, nodes=4, steps=2, groups=4, samples_per_prompt=4, eval_interval=2)
    argv = miles.arguments(cfg)
    assert value(argv, "actor-num-nodes") == "4"
    assert value(argv, "actor-num-gpus-per-node") == value(argv, "num-gpus-per-node") == "8"
    # The recipe names no four-node shape, so the reviewed whole-node partition
    # is reused unchanged and the extra nodes are data-parallel replicas.
    assert value(argv, "tensor-model-parallel-size") == "4"
    assert value(argv, "context-parallel-size") == "2"
    assert value(argv, "global-batch-size") == "16"


def test_node_dependent_recipe_shapes_are_not_extrapolated(config, native_boundary):
    profile, _ = native_boundary
    profile.parallel_args_by_shape[(2, 8)] = "--tensor-model-parallel-size 8"
    with pytest.raises(ValueError, match="no reviewed whole-node shape"):
        miles.arguments(replace(config, nodes=4, groups=2, samples_per_prompt=2))


def test_replica_larger_than_a_node_is_rejected(config, native_boundary):
    profile, _ = native_boundary
    shape = "--tensor-model-parallel-size 8 --context-parallel-size 2"
    profile.parallel_args_by_shape = {(1, 8): shape, (2, 8): shape}
    with pytest.raises(ValueError, match="pinned replica size"):
        miles.arguments(replace(config, nodes=2, groups=2, samples_per_prompt=2))


def test_unreadable_partition_size_stops_before_launch(config, native_boundary):
    profile, _ = native_boundary
    profile.parallel_args_by_shape = {(1, 8): "--tensor-model-parallel-size auto"}
    with pytest.raises(ValueError, match="unreadable partition size"):
        miles.arguments(config)
