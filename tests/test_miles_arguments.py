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
        {"nodes": 3},
        {"nodes": True},
        {"gpus_per_node": 0},
        {"gpus_per_node": 4},
        {"gpus_per_node": True},
        {"nodes": 1, "gpus_per_node": 4},
        {"nodes": 2, "gpus_per_node": 7},
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
        {"temperature": 0},
        {"temperature": float("nan")},
        {"temperature": True},
        {"kl_loss_coef": -1},
        {"kl_loss_coef": float("nan")},
        {"kl_loss_coef": True},
        {"max_tokens_per_gpu": 0},
        {"max_tokens_per_gpu": True},
        {"max_tokens_per_gpu": 100000},
        {"context_tokens": 100000},
        {"context_tokens": 81920},
        {"response_tokens": 1},
        {"tokens_per_turn": 0},
        {"output_root": "/mnt/sfs/models/output"},
        {"model_root": "/"},
        {"model_root": "/mnt/sfs/models/../source"},
        {"model_root": "/mnt/sfs/models/source/"},
        {"model_root": "/mnt/sfs/models/line\nbreak"},
        {"policy_identity_root": "/"},
        {"policy_identity_root": "/mnt/sfs/models/../accepted"},
        {"policy_identity_root": "/mnt/sfs/models/accepted/"},
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
            (1, 8): (
                "--tensor-model-parallel-size 4 --sequence-parallel "
                "--pipeline-model-parallel-size 1 --context-parallel-size 2 "
                "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1"
            ),
            (2, 8): (
                "--tensor-model-parallel-size 4 --sequence-parallel "
                "--pipeline-model-parallel-size 1 --context-parallel-size 2 "
                "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1"
            ),
        },
        extra_train_args="--offload-train-target cpu",
        extra_sglang_args="--sglang-disable-radix-cache",
        rollout_num_gpus_per_engine=1,
        sglang_mem_fraction_static=0.8,
        max_tokens_per_gpu=49152,
        log_prob_pass_multiplier=2,
    )
    runtime = ModuleType("fti.trainers.miles.run_fleet")
    long_profile = NS(
        **{
            **profile.__dict__,
            "parallel_args_by_shape": {
                (4, 8): (
                    "--tensor-model-parallel-size 8 --sequence-parallel "
                    "--pipeline-model-parallel-size 1 --context-parallel-size 4 "
                    "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1"
                )
            },
            "max_context_len": 262144,
            "max_response_len": 245760,
            "max_tokens_per_gpu": 65536,
        }
    )
    runtime.TEMPLATES, runtime._RECIPES = (
        tmp_path,
        {
            "qwen3.8-27b": profile,
            "qwen3.8-27b-256k": long_profile,
        },
    )
    model = ModuleType("miles.utils.external_utils.model_args_utils")
    model.load_model_args = lambda name: "--num-layers 64"
    tito = ModuleType("miles.utils.chat_template_utils.tito_tokenizer")
    tito.resolve_fixed_chat_template = lambda name: (
        (str(template), {"preserve_thinking": True, "reasoning_effort": "xhigh"})
        if name == "qwen38small"
        else (None, {})
    )
    tito.resolve_reasoning_and_tool_call_parser = lambda name: (
        ("qwen3", "qwen3_coder") if name == "qwen38small" else (None, None)
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setitem(sys.modules, tito.__name__, tito)
    return profile, template


def value(argv, flag):
    return argv[argv.index("--" + flag) + 1]


def test_bounded_counts_and_native_optimizer(config, native_boundary):
    cfg = replace(
        config,
        nodes=2,
        steps=9,
        groups=4,
        samples_per_prompt=4,
        lr=3e-6,
        temperature=0.7,
        kl_loss_coef=0.001,
        max_tokens_per_gpu=8192,
    )
    argv = miles.arguments(cfg)
    assert value(argv, "num-rollout") == "9"
    assert value(argv, "num-steps-per-rollout") == "1"
    assert value(argv, "rollout-batch-size") == value(argv, "over-sampling-batch-size") == "4"
    assert value(argv, "global-batch-size") == "16"
    assert value(argv, "lr") == "3e-06"
    assert value(argv, "rollout-temperature") == "0.7"
    assert value(argv, "kl-loss-coef") == "0.001"
    assert value(argv, "max-tokens-per-gpu") == "8192"
    assert value(argv, "log-probs-max-tokens-per-gpu") == "16384"
    assert value(argv, "tensor-model-parallel-size") == "4"
    assert value(argv, "pipeline-model-parallel-size") == "1"
    assert value(argv, "context-parallel-size") == "2"
    assert "--sequence-parallel" in argv
    assert value(argv, "recompute-granularity") == "full"
    assert value(argv, "recompute-method") == "uniform"
    assert value(argv, "recompute-num-layers") == "1"
    assert value(argv, "eval-prompt-data") == "fleet-dev" and cfg.dev_data in argv
    assert value(argv, "custom-generate-function-path") == "training.rl_episode.generate"
    assert (
        value(argv, "rollout-function-path")
        == value(argv, "eval-function-path")
        == "training.miles_rollout.Rollout"
    )
    assert value(argv, "cyber-data-manifest") == cfg.data_manifest
    assert "--use-wandb" in argv and "--wandb-key" not in argv
    assert "--calculate-per-token-loss" in argv
    assert "--disable-grpo-std-normalization" in argv
    assert "--dynamic-sampling-filter-path" not in argv and "--use-fault-tolerance" not in argv
    assert "--no-save-optim" not in argv and "--no-save-rng" not in argv
    assert value(argv, "load") == value(argv, "save") == cfg.output_root + "/checkpoints"
    assert value(argv, "ref-load") == cfg.torch_dist_root
    assert value(argv, "fleet-tito-model") == "qwen35"
    assert value(argv, "fleet-policy-identity-root") == cfg.model_root
    assert value(argv, "rollout-max-prompt-len") == str(cfg.context_tokens - cfg.response_tokens)
    assert value(argv, "rollout-seed") == str(cfg.seed)


def test_two_by_four_is_rejected_even_when_native_shapes_share_arguments(config, native_boundary):
    with pytest.raises(ValueError, match="unsupported Qwen Miles node/GPU layout"):
        miles.arguments(replace(config, nodes=2, gpus_per_node=4))


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


def test_runtime_and_policy_identity_paths_are_distinct_native_arguments(config, native_boundary):
    accepted = "/mnt/sfs/jobs/synthetic-sft/hf-export"
    argv = miles.arguments(replace(config, policy_identity_root=accepted))
    assert value(argv, "hf-checkpoint") == config.model_root
    assert value(argv, "fleet-policy-identity-root") == accepted


def test_long_context_uses_native_256k_shape_and_session_v2(config, native_boundary):
    argv = miles.arguments(
        replace(
            config,
            nodes=4,
            native_profile="qwen3.8-27b-256k",
            harness="opencode",
            runtime_image="registry.test/miles-opencode@sha256:" + "a" * 64,
            session_node_cap=4096,
            context_tokens=262144,
            response_tokens=245760,
            tokens_per_turn=32768,
            max_tokens_per_gpu=65536,
        )
    )
    assert value(argv, "tensor-model-parallel-size") == "8"
    assert value(argv, "context-parallel-size") == "4"
    assert value(argv, "rollout-max-context-len") == "262144"
    assert value(argv, "rollout-max-response-len") == "245760"
    assert value(argv, "max-tokens-per-gpu") == "65536"
    assert value(argv, "custom-generate-function-path") == "training.miles_opencode.generate"
    assert value(argv, "custom-agent-function-path") == "training.miles_opencode.run"
    assert value(argv, "use-session-server") == "v2"
    assert value(argv, "tito-model") == value(argv, "fleet-tito-model") == "qwen38small"
    assert value(argv, "fleet-session-node-cap") == "4096"
    assert value(argv, "sglang-router-policy") == "consistent_hashing"
    assert "--chat-template-path" not in argv


@pytest.mark.parametrize(
    "changes",
    [
        {"nodes": 1},
        {"context_tokens": 98304},
        {"response_tokens": 81920},
        {"tokens_per_turn": 4096},
        {"max_tokens_per_gpu": 32768},
        {"session_node_cap": 1024},
        {"runtime_image": miles.IMAGE},
        {"harness": "direct"},
    ],
)
def test_long_context_rejects_partial_contract_changes(config, native_boundary, changes):
    base = replace(
        config,
        nodes=4,
        native_profile="qwen3.8-27b-256k",
        harness="opencode",
        runtime_image="registry.test/miles-opencode@sha256:" + "a" * 64,
        session_node_cap=4096,
        context_tokens=262144,
        response_tokens=245760,
        tokens_per_turn=32768,
        max_tokens_per_gpu=65536,
    )
    with pytest.raises(ValueError, match="exact native 256K contract"):
        miles.arguments(replace(base, **changes))


def test_unknown_configuration_fields_rejected(config):
    with pytest.raises(TypeError):
        miles.MilesConfig(**config.__dict__, extra_args="--arbitrary-override")
