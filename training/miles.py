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

TEMPLATE_SHA256 = "38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"
# Whole nodes only: the native shape owns every GPU on a node. The ceiling is the
# experiment's eight actively allocated nodes and the Jobs API's worker ceiling,
# not an allocation any recipe is entitled to; count other owned capacity first.
MAX_NODES = 8
GPUS_PER_NODE = 8
PARTITION_FLAGS = {
    "tensor_model_parallel_size": "--tensor-model-parallel-size",
    "pipeline_model_parallel_size": "--pipeline-model-parallel-size",
    "context_parallel_size": "--context-parallel-size",
}


@dataclass(frozen=True)
class Profile:
    """A reviewed (recipe, trainer image, context budget, replica shape) tuple.

    `replica_gpus` is the product of the recipe's tensor/pipeline/context parallel
    sizes: how many GPUs hold one copy of the model. `arguments` rechecks it
    against the recipe inside the image before any argument is built. A replica of
    eight is one node, and extra nodes are data-parallel replicas; a replica
    larger than a node spans several nodes and is one copy of the model, so the
    supported node counts are fixed by the recipe rather than extrapolated.
    """

    recipe: str
    image: str
    max_context: int
    replica_gpus: int
    supported_nodes: tuple[int, ...]


# The native 262144-context row (fti 0.7.8's qwen3.8-27b-256k, TP8 x CP4) is one
# replica across four nodes; deniz-qwen38-256k-03 ran it at 113.7 GB peak per GPU.
# It needs its own trainer image because the recipe does not exist in the base
# image. Its chat template is the base row's, so TEMPLATE_SHA256 covers both.
PROFILES = {
    "qwen3.8-27b": Profile(
        recipe="qwen3.8-27b",
        image=(
            "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
            "d1d37c584e2aafdd47df1e1f3492ff3343eb3b83a5f658cf7ce2432ee8d6ef33"
        ),
        max_context=98304,
        replica_gpus=8,
        supported_nodes=tuple(range(1, MAX_NODES + 1)),
    ),
    "qwen3.8-27b-256k": Profile(
        recipe="qwen3.8-27b-256k",
        image=(
            "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
            "259c216ec120d22b9f4d6192c0c870a15f04e2993c5f8838c181371944e396fb"
        ),
        max_context=262144,
        replica_gpus=32,
        supported_nodes=(4,),
    ),
}
# The base image remains the module default so existing conversion/RL paths and
# their receipts are unchanged.
IMAGE = PROFILES["qwen3.8-27b"].image


def profile_for(name: str) -> Profile:
    if name not in PROFILES:
        raise ValueError("unknown Miles profile")
    return PROFILES[name]


def image_for(name: str) -> str:
    return profile_for(name).image


@dataclass(frozen=True)
class MilesConfig:
    name: str
    output_root: str
    model_root: str
    torch_dist_root: str
    train_data: str
    dev_data: str
    data_manifest: str
    wandb_entity: str
    wandb_project: str
    wandb_run_id: str
    model: str = "Qwen/Qwen3.8-27B"
    profile: str = "qwen3.8-27b"
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
    # Miles' synchronous path provisions a whole rollout wave at once, so without
    # a gate the batch size sets simultaneous environment creates. The dataminer
    # arms measured this: 2,100 simultaneous provisions returned gateway 502s and
    # left engines idle for two hours (2026-08-28), and their training arms ran at
    # 32. The slot is held across provisioning, so a queued episode does not burn
    # its own TTL. Raising nodes raises the batch, not this ceiling.
    max_concurrent_episodes: int = 32

    def validate_paths(self):
        """Require canonical, disjoint staged inputs and an owned output directory."""
        paths = []
        for key in (
            "output_root",
            "model_root",
            "torch_dist_root",
            "train_data",
            "dev_data",
            "data_manifest",
        ):
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

    def validate(self):
        if self.model != "Qwen/Qwen3.8-27B":
            raise ValueError("Miles full-model profile unqualified; GLM Flash is not full GLM")
        for key in ("name", "wandb_entity", "wandb_project", "wandb_run_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", getattr(self, key)):
                raise ValueError("invalid run or W&B identifier")
        self.validate_paths()
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
            "max_concurrent_episodes",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError("Miles counts must be positive integers")
        prof = profile_for(self.profile)
        if self.samples_per_prompt < 2:
            raise ValueError("grouped GRPO requires at least two samples per prompt")
        if self.nodes not in prof.supported_nodes:
            raise ValueError("profile does not pin a reviewed shape for this node count")
        gpus = self.nodes * GPUS_PER_NODE
        if gpus % prof.replica_gpus:
            raise ValueError("node count does not hold a whole number of model replicas")
        if (self.groups * self.samples_per_prompt) % (gpus // prof.replica_gpus):
            raise ValueError("global batch must divide across the data-parallel replicas")
        if self.max_concurrent_episodes > 256:
            raise ValueError("simultaneous environment ceiling exceeds any observed safe wave")
        if self.steps % self.eval_interval:
            raise ValueError(
                "eval interval must divide steps so native Miles evaluates the final model"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("learning rate must be finite and positive")
        if (
            not self.tokens_per_turn
            <= self.response_tokens
            < self.context_tokens
            <= prof.max_context
        ):
            raise ValueError("generation budgets exceed the native Qwen context envelope")


def topology(config: MilesConfig) -> dict[str, int]:
    """Derive the whole-node replica layout and batch fan-out this recipe implies."""
    config.validate()
    prof = profile_for(config.profile)
    gpus = config.nodes * GPUS_PER_NODE
    batch = config.groups * config.samples_per_prompt
    return {
        "nodes": config.nodes,
        "gpus_per_node": GPUS_PER_NODE,
        "gpus": gpus,
        "replica_gpus": prof.replica_gpus,
        "data_parallel_size": gpus // prof.replica_gpus,
        "global_batch_size": batch,
        "max_concurrent_episodes": config.max_concurrent_episodes,
        # Every sample in flight holds one authorized Fleet environment instance.
        # Check the account's headroom for this number of simultaneous instances
        # before submitting, not only the GPU budget.
        "concurrent_train_environments": min(batch, config.max_concurrent_episodes),
    }


def parallel_shape(recipe, nodes: int, replica_gpus: int) -> str:
    """Return the native parallel arguments the recipe pins for this whole-node shape.

    The recipe keys one argument string per (nodes, GPUs) pair. When the replica
    is a single node and the recipe does not name this node count, every
    single-node shape it names must be identical: that identity is what makes the
    extra nodes data-parallel replicas of one partition. A recipe whose replica
    spans several nodes, or whose single-node shapes differ, is not extrapolated;
    the exact node count must be named, or preparation stops for review.
    """
    shapes = recipe.parallel_args_by_shape
    exact = shapes.get((nodes, GPUS_PER_NODE))
    if exact is not None:
        return exact
    if replica_gpus != GPUS_PER_NODE:
        raise ValueError("multi-node replica recipe must name this exact node count")
    whole = {value for (_, gpus), value in shapes.items() if gpus == GPUS_PER_NODE}
    if len(whole) != 1:
        raise ValueError("native recipe pins no reviewed whole-node shape for this node count")
    return whole.pop()


def partition(parallel_args: str) -> dict[str, int]:
    """Read the per-replica model partition sizes out of native parallel arguments."""
    argv = shlex.split(parallel_args)
    sizes = {}
    for key, flag in PARTITION_FLAGS.items():
        if flag not in argv:
            sizes[key] = 1
            continue
        index = argv.index(flag) + 1
        value = argv[index] if index < len(argv) else ""
        if not re.fullmatch(r"[1-9][0-9]*", value):
            raise ValueError("native parallel shape carries an unreadable partition size")
        sizes[key] = int(value)
    return sizes


def arguments(config: MilesConfig) -> list[str]:
    """Use native model/topology defaults; one rollout batch is one optimizer step.

    Paths and counters alone are not scientific acceptance. The caller must pin
    model revision, converted checkpoint, exact task splits/tools, prompt bytes,
    and source hashes before using these arguments with the native train driver.
    """
    config.validate()
    prof = profile_for(config.profile)
    from fti.trainers.miles.run_fleet import _RECIPES, TEMPLATES
    from miles.utils.external_utils.model_args_utils import load_model_args

    recipe = _RECIPES[prof.recipe]
    template = TEMPLATES / recipe.chat_template
    if hashlib.sha256(template.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("native Qwen chat template changed")
    if recipe.backend != "megatron" or recipe.vision or recipe.tito_model != "qwen35":
        raise ValueError("native Qwen profile changed")
    shape = parallel_shape(recipe, config.nodes, prof.replica_gpus)
    if math.prod(partition(shape).values()) != prof.replica_gpus:
        raise ValueError("native parallel shape no longer matches the pinned replica size")
    argv = shlex.split(load_model_args(recipe.megatron_model_type))
    argv += shlex.split(shape)
    argv += shlex.split(recipe.extra_train_args + " " + recipe.extra_sglang_args)
    batch = config.groups * config.samples_per_prompt
    values = {
        "train-backend": "megatron",
        "hf-checkpoint": config.model_root,
        "ref-load": config.torch_dist_root,
        "load": config.output_root + "/checkpoints",
        "save": config.output_root + "/checkpoints",
        "save-interval": config.checkpoint_interval,
        "prompt-data": config.train_data,
        "data-source-path": "training.miles_text.TextDataSource",
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
        "rollout-function-path": "training.miles_rollout.Rollout",
        "eval-function-path": "training.miles_rollout.Rollout",
        "cyber-run-id": config.name,
        "cyber-output-root": config.output_root + "/episodes",
        "cyber-data-manifest": config.data_manifest,
        "fleet-tito-model": recipe.tito_model,
        "fleet-max-tokens-per-turn": config.tokens_per_turn,
        "cyber-max-concurrent-episodes": config.max_concurrent_episodes,
        "chat-template-path": str(template),
        "actor-num-nodes": config.nodes,
        "actor-num-gpus-per-node": GPUS_PER_NODE,
        "num-gpus-per-node": GPUS_PER_NODE,
        "rollout-num-gpus-per-engine": recipe.rollout_num_gpus_per_engine,
        "sglang-mem-fraction-static": recipe.sglang_mem_fraction_static,
        "router-policy": "round_robin",
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        "micro-batch-size": 1,
        "max-tokens-per-gpu": recipe.max_tokens_per_gpu,
        "log-probs-max-tokens-per-gpu": recipe.max_tokens_per_gpu * recipe.log_prob_pass_multiplier,
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
