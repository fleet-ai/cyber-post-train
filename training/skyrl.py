"""Bounded native SkyRL GRPO configuration, not a second training loop.

Only Qwen's full-weight FSDP profile is wired here. Preparing a configuration
does not qualify a model, acquire reward, create environments or allocate GPUs.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

NATIVE_SOURCES = {
    "skyrl.train.config.config": "a3b36099c5308fc9bc658394f10cfff0da3f0cadcc067aa2b44e95870a609c90",
    "skyrl.train.utils.utils": "abf91c865735dfc052ffddcc0a1ff06afe32814855b5328b08d3a3f7ca8f418c",
}


@dataclass(frozen=True)
class SkyRLConfig:
    name: str
    output_root: str
    model_root: str
    train_data: str
    dev_data: str
    data_manifest: str
    train_rows: int
    dev_rows: int
    wandb_entity: str
    wandb_project: str
    wandb_run_id: str
    model: str = "Qwen/Qwen3.8-27B"
    nodes: int = 1
    steps: int = 1
    groups: int = 2
    samples_per_prompt: int = 4
    lr: float = 1e-6
    eval_interval: int = 1
    checkpoint_interval: int = 1
    keep_checkpoints: int = 2
    seed: int = 42
    context_tokens: int = 98304
    response_tokens: int = 81920
    tokens_per_turn: int = 4096
    max_turns: int = 64

    def validate(self):
        if self.model != "Qwen/Qwen3.8-27B":
            raise ValueError("SkyRL RL loader unqualified for this model; no model substitution")
        for key in ("name", "wandb_entity", "wandb_project", "wandb_run_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", getattr(self, key)):
                raise ValueError("invalid run or W&B identifier")
        paths = []
        for key in ("output_root", "model_root", "train_data", "dev_data", "data_manifest"):
            value = getattr(self, key)
            path = PurePosixPath(value)
            if (
                path.parts[:3] != ("/", "mnt", "sfs")
                or len(path.parts) < 5
                or str(path) != value
                or ".." in path.parts
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("SkyRL inputs/output require canonical staged SFS paths")
            paths.append(path)
        if paths[0].parts[:4] != ("/", "mnt", "sfs", "jobs") or any(
            a == b or a in b.parents or b in a.parents
            for i, a in enumerate(paths)
            for b in paths[i + 1 :]
        ):
            raise ValueError("owned output and input paths must not overlap")
        for key in (
            "nodes",
            "steps",
            "groups",
            "samples_per_prompt",
            "train_rows",
            "dev_rows",
            "eval_interval",
            "checkpoint_interval",
            "keep_checkpoints",
            "context_tokens",
            "response_tokens",
            "tokens_per_turn",
            "max_turns",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("SkyRL counts must be positive integers")
        if self.nodes not in (1, 2) or self.samples_per_prompt < 2 or self.max_turns < 2:
            raise ValueError("profile requires 1–2 whole nodes and grouped GRPO samples")
        if self.train_rows % self.groups or self.groups * self.samples_per_prompt % (
            8 * self.nodes
        ):
            raise ValueError(
                "complete prompt batches and exactly divisible GPU sample groups required"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not self.tokens_per_turn <= self.response_tokens < self.context_tokens <= 98304:
            raise ValueError("generation budgets exceed the reviewed Qwen context envelope")


def overrides(config: SkyRLConfig) -> dict:
    """One native prompt batch = one policy mini-batch = one optimizer step.

    Native SkyRL owns accumulation, GRPO, clipping, weight synchronization and
    checkpointing. End-of-run evaluation/save is native, even off the interval.
    The Fleet generator separately enforces exact task identity and no retries.
    """
    config.validate()
    output = config.output_root
    sampling = {
        "max_generate_length": config.tokens_per_turn,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": 0,
    }
    return {
        "data.train_data": [config.train_data],
        "data.val_data": [config.dev_data],
        "data.dataloader.num_workers": 0,
        "trainer.strategy": "fsdp",
        "trainer.policy.model.path": config.model_root,
        "trainer.ref.model.path": config.model_root,
        "trainer.critic.model.path": None,
        "trainer.policy.language_model_only": True,
        "trainer.ref.language_model_only": True,
        "trainer.policy.model_config_kwargs.fleet_force_qwen35_torch_gdn": True,
        "trainer.ref.model_config_kwargs.fleet_force_qwen35_torch_gdn": True,
        "trainer.policy.optimizer_config.lr": config.lr,
        "trainer.policy.optimizer_config.scheduler": "constant_with_warmup",
        "trainer.policy.optimizer_config.num_warmup_steps": 0,
        "trainer.placement.colocate_all": True,
        "trainer.placement.policy_num_nodes": config.nodes,
        "trainer.placement.policy_num_gpus_per_node": 8,
        "trainer.placement.ref_num_nodes": config.nodes,
        "trainer.placement.ref_num_gpus_per_node": 8,
        "trainer.policy.sequence_parallel_size": 1,
        "trainer.ref.sequence_parallel_size": 1,
        "trainer.flash_attn": False,
        "trainer.remove_microbatch_padding": False,
        "trainer.gradient_checkpointing": True,
        "trainer.epochs": math.ceil(config.steps / (config.train_rows // config.groups)),
        "trainer.max_training_steps": config.steps,
        "trainer.update_epochs_per_batch": 1,
        "trainer.train_batch_size": config.groups,
        "trainer.policy_mini_batch_size": config.groups,
        "trainer.micro_train_batch_size_per_gpu": 1,
        "trainer.micro_forward_batch_size_per_gpu": 1,
        "trainer.eval_batch_size": config.dev_rows,
        "trainer.eval_before_train": True,
        "trainer.eval_interval": config.eval_interval,
        "trainer.max_prompt_length": config.context_tokens - config.response_tokens,
        "trainer.ckpt_path": output + "/checkpoints",
        "trainer.ckpt_interval": config.checkpoint_interval,
        "trainer.max_ckpts_to_keep": config.keep_checkpoints,
        "trainer.resume_mode": "none",
        "trainer.hf_save_interval": -1,
        "trainer.export_path": output + "/exports",
        "trainer.log_path": output + "/private-native-logs",
        "trainer.seed": config.seed,
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.use_kl_loss": True,
        "trainer.algorithm.kl_loss_coef": 0.001,
        "trainer.algorithm.dynamic_sampling.type": None,
        "trainer.algorithm.zero_variance_filter": False,
        "trainer.algorithm.use_entropy_loss": False,
        "trainer.logger": "wandb",
        "trainer.project_name": config.wandb_project,
        "trainer.run_name": config.name,
        "trainer.dump_data_batch": False,
        "trainer.dump_eval_results": False,
        "trainer.print_example_interval": -1,
        "trainer.log_example_interval": -1,
        "trainer.num_logger_train_samples": 0,
        "trainer.num_logger_eval_samples": 0,
        "generator.n_samples_per_prompt": config.samples_per_prompt,
        "generator.eval_n_samples_per_prompt": 1,
        "generator.max_turns": config.max_turns,
        "generator.max_input_length": config.context_tokens,
        "generator.sampling_params": sampling,
        "generator.eval_sampling_params": {**sampling, "temperature": 0.0},
        "generator.step_wise_trajectories": False,
        "generator.zero_reward_on_non_stop": False,
        "generator.apply_overlong_filtering": False,
        "generator.inference_engine.num_engines": config.nodes * 2,
        "generator.inference_engine.tensor_parallel_size": 4,
        "generator.inference_engine.language_model_only": True,
        "generator.inference_engine.enforce_eager": True,
        "generator.inference_engine.enable_prefix_caching": False,
        "generator.inference_engine.max_num_seqs": config.groups * config.samples_per_prompt,
        "generator.inference_engine.engine_init_kwargs.max_model_len": config.context_tokens,
    }


def native_config(config: SkyRLConfig):
    """Parse and validate through the exact installed native code, without Ray init."""
    from .skyrl_episode import _module

    values = overrides(config)
    modules = {name: _module(name, sha) for name, sha in NATIVE_SOURCES.items()}
    cfg = modules["skyrl.train.config.config"].SkyRLTrainConfig.from_cli_overrides(values)
    modules["skyrl.train.utils.utils"].validate_cfg(cfg)
    return cfg
