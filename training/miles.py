"""Bounded native Miles arguments, not a second trainer or a launch wrapper.

Run inside the pinned Miles image after exact model/data preflight. This module
does not download weights, convert checkpoints, start/stop Ray, call pkill, or
submit a job. Full GLM is deliberately not mapped to the native Flash recipe.
"""

from __future__ import annotations

import hashlib
import math
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "d1d37c584e2aafdd47df1e1f3492ff3343eb3b83a5f658cf7ce2432ee8d6ef33"
)
TEMPLATE_SHA256 = "38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"


@dataclass(frozen=True)
class MilesConfig:
    name: str
    output_root: str
    model_root: str
    torch_dist_root: str
    train_data: str
    dev_data: str
    wandb_entity: str
    wandb_project: str
    wandb_run_id: str
    model: str = "Qwen/Qwen3.8-27B"
    nodes: int = 1
    steps: int = 1
    groups: int = 1
    samples_per_prompt: int = 2
    lr: float = 1e-6
    eval_interval: int = 1
    checkpoint_interval: int = 1
    seed: int = 42
    context_tokens: int = 98304
    response_tokens: int = 81920
    tokens_per_turn: int = 4096

    def validate(self):
        if self.model != "Qwen/Qwen3.8-27B":
            raise ValueError("Miles full-model profile unqualified; GLM Flash is not full GLM")
        for key in ("name", "wandb_entity", "wandb_project", "wandb_run_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", getattr(self, key)):
                raise ValueError("invalid run or W&B identifier")
        paths = []
        for key in ("output_root", "model_root", "torch_dist_root", "train_data", "dev_data"):
            value = getattr(self, key)
            path = PurePosixPath(value)
            if (
                path.parts[:3] != ("/", "mnt", "sfs")
                or len(path.parts) < 5
                or str(path) != value
                or ".." in path.parts
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("Miles inputs/output require canonical staged SFS paths")
            paths.append(path)
        if any(
            a == b or a in b.parents or b in a.parents
            for i, a in enumerate(paths)
            for b in paths[i + 1 :]
        ):
            raise ValueError("Miles input and output paths must not overlap")
        if paths[0].parts[:4] != ("/", "mnt", "sfs", "jobs"):
            raise ValueError("output must be an owned jobs directory")
        for key in (
            "nodes",
            "steps",
            "groups",
            "samples_per_prompt",
            "eval_interval",
            "checkpoint_interval",
            "context_tokens",
            "response_tokens",
            "tokens_per_turn",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Miles counts must be positive integers")
        if self.nodes not in (1, 2) or self.samples_per_prompt < 2:
            raise ValueError("Qwen profile requires 1–2 whole nodes and grouped GRPO samples")
        if self.steps % self.eval_interval:
            raise ValueError(
                "eval interval must divide steps so native Miles evaluates the final model"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not self.tokens_per_turn <= self.response_tokens < self.context_tokens <= 98304:
            raise ValueError("generation budgets exceed the native Qwen context envelope")


def arguments(config: MilesConfig) -> list[str]:
    """Use native model/topology defaults; one rollout batch is one optimizer step.

    Paths and counters alone are not scientific acceptance. The caller must pin
    model revision, converted checkpoint, exact task splits/tools, prompt bytes,
    and source hashes before using these arguments with the native train driver.
    """
    config.validate()
    from fti.trainers.miles.run_fleet import _RECIPES, TEMPLATES
    from miles.utils.external_utils.model_args_utils import load_model_args

    profile = _RECIPES["qwen3.8-27b"]
    template = TEMPLATES / profile.chat_template
    if hashlib.sha256(template.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("native Qwen chat template changed")
    if profile.backend != "megatron" or profile.vision or profile.tito_model != "qwen35":
        raise ValueError("native Qwen profile changed")
    argv = shlex.split(load_model_args(profile.megatron_model_type))
    argv += shlex.split(profile.parallel_args_by_shape[(config.nodes, 8)])
    argv += shlex.split(profile.extra_train_args + " " + profile.extra_sglang_args)
    batch = config.groups * config.samples_per_prompt
    values = {
        "train-backend": "megatron",
        "hf-checkpoint": config.model_root,
        "ref-load": config.torch_dist_root,
        "load": config.output_root + "/checkpoints",
        "save": config.output_root + "/checkpoints",
        "save-interval": config.checkpoint_interval,
        "prompt-data": config.train_data,
        "input-key": "input",
        "metadata-key": "metadata",
        "num-rollout": config.steps,
        "num-steps-per-rollout": 1,
        "rollout-batch-size": config.groups,
        "over-sampling-batch-size": config.groups,
        "n-samples-per-prompt": config.samples_per_prompt,
        "global-batch-size": batch,
        "rollout-max-context-len": config.context_tokens,
        "rollout-max-response-len": config.response_tokens,
        "rollout-max-prompt-len": config.context_tokens - config.response_tokens,
        "rollout-temperature": 1,
        "seed": config.seed,
        "rollout-seed": config.seed,
        "custom-generate-function-path": "training.rl_episode.generate",
        "cyber-run-id": config.name,
        "cyber-output-root": config.output_root + "/episodes",
        "fleet-tito-model": profile.tito_model,
        "fleet-max-tokens-per-turn": config.tokens_per_turn,
        "chat-template-path": str(template),
        "actor-num-nodes": config.nodes,
        "actor-num-gpus-per-node": 8,
        "num-gpus-per-node": 8,
        "rollout-num-gpus-per-engine": profile.rollout_num_gpus_per_engine,
        "sglang-mem-fraction-static": profile.sglang_mem_fraction_static,
        "router-policy": "round_robin",
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        "micro-batch-size": 1,
        "max-tokens-per-gpu": profile.max_tokens_per_gpu,
        "log-probs-max-tokens-per-gpu": profile.max_tokens_per_gpu
        * profile.log_prob_pass_multiplier,
        "advantage-estimator": "grpo",
        "kl-loss-coef": 0,
        "kl-loss-type": "low_var_kl",
        "entropy-coef": 0,
        "eps-clip": 0.2,
        "eps-clip-high": 0.28,
        "optimizer": "adam",
        "lr": config.lr,
        "lr-decay-style": "constant",
        "weight-decay": 0.1,
        "adam-beta1": 0.9,
        "adam-beta2": 0.98,
        "eval-interval": config.eval_interval,
        "n-samples-per-eval-prompt": 1,
        "wandb-team": config.wandb_entity,
        "wandb-project": config.wandb_project,
        "wandb-group": config.name,
        "wandb-run-id": config.wandb_run_id,
        "wandb-mode": "online",
        "wandb-dir": config.output_root + "/wandb",
    }
    for key, value in values.items():
        argv.extend(("--" + key, str(value)))
    argv.extend(("--eval-prompt-data", "fleet-dev", config.dev_data))
    argv.extend(
        "--" + flag
        for flag in (
            "colocate",
            "rollout-shuffle",
            "use-dynamic-batch-size",
            "use-kl-loss",
            "use-wandb",
            "disable-wandb-random-suffix",
            "log-multi-turn",
        )
    )
    flags = [x for x in argv if x.startswith("--")]
    if len(flags) != len(set(flags)):
        raise ValueError("native profile overlaps explicit run arguments; review drift")
    forbidden = {
        "--dynamic-sampling-filter-path",
        "--partial-rollout",
        "--use-fault-tolerance",
        "--wandb-key",
        "--no-save-optim",
        "--no-save-rng",
        "--skip-eval-before-train",
        "--apply-chat-template",
        "--disable-rollout-global-dataset",
    }
    if forbidden.intersection(flags):
        raise ValueError("native profile enables forbidden retries, credentials or lost state")
    return argv
