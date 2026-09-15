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
from pathlib import Path, PurePosixPath

IMAGE = (
    "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/miles-trainer@sha256:"
    "d1d37c584e2aafdd47df1e1f3492ff3343eb3b83a5f658cf7ce2432ee8d6ef33"
)
TEMPLATE_SHA256 = "38d42166599348d47ded69776c5389c89924045e6827089923a031379f8a3dfe"
NATIVE_LAYOUTS = frozenset({(1, 8), (2, 8)})
LONG_CONTEXT_PROFILE = "qwen3.8-27b-256k"
LONG_CONTEXT_LAYOUT = (4, 8)
# Native entrypoints in the FTI 0.8.4 image that carries the 256K profile.
# Keep these separate from the legacy image pins in the conversion/training
# modules: accepting either digest for either image would weaken the boundary.
LONG_NATIVE_DRIVER_SHA256 = "85dbfd31d41a84f9c2e79a2918583851fb53925630afa229e9cd0a154b170f46"
LONG_NATIVE_CONVERTER_SHA256 = "0c2541d30073777a30344273a3773844a70ca1961287520c0496a1cec18d43f6"
LONG_INSTALLED_SESSION_TREE_SHA256 = (
    "59bed80a62a8ab94e0bb9012f4f9f0290245a5c8db6feadd0997a7bfb57025ee"
)
LONG_TITO_TEMPLATE_SHA256 = TEMPLATE_SHA256


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
    # The checkpoint used by the native runtime may be a byte-identical staged
    # copy of an accepted SFT export.  Keep that runtime path separate from the
    # scientific policy identity sealed into every episode row.
    policy_identity_root: str | None = None
    model: str = "Qwen/Qwen3.8-27B"
    nodes: int = 1
    gpus_per_node: int = 8
    steps: int = 1
    groups: int = 1
    samples_per_prompt: int = 2
    lr: float = 1e-6
    temperature: float = 1.0
    kl_loss_coef: float = 0.0
    max_tokens_per_gpu: int | None = None
    eval_interval: int = 1
    checkpoint_interval: int = 1
    seed: int = 42
    context_tokens: int = 98304
    response_tokens: int = 81920
    tokens_per_turn: int = 4096
    native_profile: str = "qwen3.8-27b"
    harness: str = "direct"
    runtime_image: str = IMAGE
    session_node_cap: int = 1024

    def validate(self):
        if self.model != "Qwen/Qwen3.8-27B":
            raise ValueError("Miles full-model profile unqualified; GLM Flash is not full GLM")
        for key in ("name", "wandb_entity", "wandb_project", "wandb_run_id"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", getattr(self, key)):
                raise ValueError("invalid run or W&B identifier")
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
        policy_root = resolved_policy_identity_root(self)
        policy_path = PurePosixPath(policy_root)
        if (
            policy_path.parts[:3] != ("/", "mnt", "sfs")
            or len(policy_path.parts) < 5
            or str(policy_path) != policy_root
            or ".." in policy_path.parts
            or any(ord(c) < 32 for c in policy_root)
        ):
            raise ValueError("policy identity requires a canonical staged SFS path")
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
            "gpus_per_node",
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
        long_horizon = self.native_profile == LONG_CONTEXT_PROFILE or self.harness == "opencode"
        if long_horizon:
            if (
                self.native_profile != LONG_CONTEXT_PROFILE
                or self.harness != "opencode"
                or (self.nodes, self.gpus_per_node) != LONG_CONTEXT_LAYOUT
                or self.context_tokens != 262_144
                or self.response_tokens != 245_760
                or self.tokens_per_turn != 32_768
                or self.max_tokens_per_gpu != 65_536
                or self.session_node_cap != 4_096
                or self.runtime_image == IMAGE
            ):
                raise ValueError("long-horizon Qwen requires the exact native 256K contract")
        elif (
            self.native_profile != "qwen3.8-27b"
            or self.harness != "direct"
            or (self.nodes, self.gpus_per_node) not in NATIVE_LAYOUTS
            or self.runtime_image != IMAGE
            or self.session_node_cap != 1024
        ):
            raise ValueError("unsupported Qwen Miles node/GPU layout or runtime contract")
        if not re.fullmatch(r"[^@\s]+@sha256:[a-f0-9]{64}", self.runtime_image):
            raise ValueError("Miles runtime image must be immutable")
        if self.samples_per_prompt < 2:
            raise ValueError("Qwen profile requires grouped GRPO samples")
        if self.steps % self.eval_interval:
            raise ValueError(
                "eval interval must divide steps so native Miles evaluates the final model"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.lr) not in (int, float) or not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("learning rate must be finite and positive")
        if (
            type(self.temperature) not in (int, float)
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("rollout temperature must be finite and positive")
        if (
            type(self.kl_loss_coef) not in (int, float)
            or not math.isfinite(self.kl_loss_coef)
            or self.kl_loss_coef < 0
        ):
            raise ValueError("KL loss coefficient must be finite and nonnegative")
        if self.max_tokens_per_gpu is not None and (
            type(self.max_tokens_per_gpu) is not int
            or self.max_tokens_per_gpu < 1
            or self.max_tokens_per_gpu > self.context_tokens
        ):
            raise ValueError("dynamic token budget must fit the native Qwen context envelope")
        context_ceiling = 262_144 if long_horizon else 98_304
        if (
            not self.tokens_per_turn
            <= self.response_tokens
            < self.context_tokens
            <= context_ceiling
        ):
            raise ValueError("generation budgets exceed the native Qwen context envelope")


def resolved_policy_identity_root(config: MilesConfig) -> str:
    """Resolve the scientific policy identity for legacy base-policy plans."""
    return config.model_root if config.policy_identity_root is None else config.policy_identity_root


def arguments(config: MilesConfig) -> list[str]:
    """Use native model/topology defaults; one rollout batch is one optimizer step.

    Paths and counters alone are not scientific acceptance. The caller must pin
    model revision, converted checkpoint, exact task splits/tools, prompt bytes,
    and source hashes before using these arguments with the native train driver.
    """
    config.validate()
    from fti.trainers.miles.run_fleet import _RECIPES, TEMPLATES
    from miles.utils.external_utils.model_args_utils import load_model_args

    profile = _RECIPES[config.native_profile]
    profile_template = TEMPLATES / profile.chat_template
    if hashlib.sha256(profile_template.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError("native Qwen chat template changed")
    if profile.backend != "megatron" or profile.vision or profile.tito_model != "qwen35":
        raise ValueError("native Qwen profile changed")
    long_horizon = config.native_profile == LONG_CONTEXT_PROFILE
    if long_horizon and (
        profile.max_context_len != 262_144
        or profile.max_response_len != 245_760
        or profile.max_tokens_per_gpu != 65_536
        or set(profile.parallel_args_by_shape) != {LONG_CONTEXT_LAYOUT}
        or profile.rollout_num_gpus_per_engine != 1
    ):
        raise ValueError("native Qwen 256K profile changed")
    tito_family = profile.tito_model
    template: Path | None = profile_template
    if long_horizon:
        from miles.utils.chat_template_utils.tito_tokenizer import (
            resolve_fixed_chat_template,
            resolve_reasoning_and_tool_call_parser,
        )

        # FTI 0.8.4 predates Miles' Qwen3.8 TITO registration and labels this
        # profile qwen35 even though it selects the distinct Qwen3.8 template.
        # The derived image backports upstream Miles #2760 exactly.
        tito_family = "qwen38small"
        resolved, kwargs = resolve_fixed_chat_template(tito_family)
        if (
            resolved is None
            or hashlib.sha256(Path(resolved).read_bytes()).hexdigest() != LONG_TITO_TEMPLATE_SHA256
            or kwargs != {"preserve_thinking": True, "reasoning_effort": "xhigh"}
            or resolve_reasoning_and_tool_call_parser(tito_family)
            != ("qwen3", "qwen3_coder")
        ):
            raise ValueError("native Qwen3.8 TITO family changed")
        template = None
    argv = shlex.split(load_model_args(profile.megatron_model_type))
    parallel_args = profile.parallel_args_by_shape.get((config.nodes, config.gpus_per_node))
    if not isinstance(parallel_args, str):
        raise ValueError("native profile lacks the requested node/GPU layout")
    argv += shlex.split(parallel_args)
    argv += shlex.split(profile.extra_train_args + " " + profile.extra_sglang_args)
    max_tokens_per_gpu = (
        profile.max_tokens_per_gpu
        if config.max_tokens_per_gpu is None
        else config.max_tokens_per_gpu
    )
    if type(max_tokens_per_gpu) is not int or max_tokens_per_gpu < 1:
        raise ValueError("native Qwen profile has an invalid dynamic token budget")
    batch = config.groups * config.samples_per_prompt
    values = {
        "train-backend": "megatron",
        "hf-checkpoint": config.model_root,
        "ref-load": config.torch_dist_root,
        # The first run starts from the accepted zero-step conversion.  A
        # later resume must be a separately bound successor, never an implicit
        # read from a create-once output directory.
        "load": config.torch_dist_root,
        "save": config.output_root + "/checkpoints",
        "save-interval": config.checkpoint_interval,
        "prompt-data": config.train_data,
        "data-source-path": "training.miles_text.TextDataSource",
        "input-key": "input",
        "metadata-key": "metadata",
        # Miles defaults this to None.  Spell out the first rollout so the
        # native parser and our create-once checkpoint semantics agree.
        "start-rollout-id": 0,
        "num-rollout": config.steps,
        "num-steps-per-rollout": 1,
        "rollout-batch-size": config.groups,
        "over-sampling-batch-size": config.groups,
        "n-samples-per-prompt": config.samples_per_prompt,
        "global-batch-size": batch,
        "rollout-max-context-len": config.context_tokens,
        "rollout-max-response-len": config.response_tokens,
        "rollout-max-prompt-len": config.context_tokens - config.response_tokens,
        "rollout-temperature": config.temperature,
        "seed": config.seed,
        "rollout-seed": config.seed,
        "custom-generate-function-path": (
            "training.miles_opencode.generate" if long_horizon else "training.rl_episode.generate"
        ),
        "rollout-function-path": "training.miles_rollout.Rollout",
        "eval-function-path": "training.miles_rollout.Rollout",
        "cyber-run-id": config.name,
        "cyber-output-root": config.output_root + "/episodes",
        "cyber-data-manifest": config.data_manifest,
        "fleet-policy-identity-root": resolved_policy_identity_root(config),
        "fleet-tito-model": tito_family,
        "fleet-max-tokens-per-turn": config.tokens_per_turn,
        "actor-num-nodes": config.nodes,
        "actor-num-gpus-per-node": config.gpus_per_node,
        "num-gpus-per-node": config.gpus_per_node,
        "rollout-num-gpus-per-engine": profile.rollout_num_gpus_per_engine,
        "sglang-mem-fraction-static": profile.sglang_mem_fraction_static,
        # The native Qwen profile keeps the radix cache enabled.  Preserve
        # FTI's measured affinity route so every turn reaches the engine that
        # owns that episode's cached prefix.
        "sglang-router-policy": "consistent_hashing",
        "recompute-granularity": "full",
        "recompute-method": "uniform",
        "recompute-num-layers": 1,
        "micro-batch-size": 1,
        "max-tokens-per-gpu": max_tokens_per_gpu,
        "log-probs-max-tokens-per-gpu": max_tokens_per_gpu * profile.log_prob_pass_multiplier,
        "advantage-estimator": "grpo",
        "kl-loss-coef": config.kl_loss_coef,
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
    if template is not None:
        values["chat-template-path"] = str(template)
    if long_horizon:
        values.update(
            {
                "custom-agent-function-path": "training.miles_opencode.run",
                "use-session-server": "v2",
                "max-seq-len": config.context_tokens,
                "tito-model": tito_family,
                "session-sample-picker-path": "training.miles_opencode.pick_compaction_segments",
                "session-sample-postprocessor-path": (
                    "training.miles_opencode.postprocess_compaction_segments"
                ),
                "fleet-session-node-cap": config.session_node_cap,
            }
        )
    for key, value in values.items():
        argv.extend(("--" + key, str(value)))
    argv.extend(("--eval-prompt-data", "fleet-dev", config.dev_data))
    argv.extend(
        "--" + flag
        for flag in (
            "colocate",
            "rollout-shuffle",
            "use-dynamic-batch-size",
            "calculate-per-token-loss",
            "disable-grpo-std-normalization",
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
    options = {
        flag: argv[index + 1]
        for index, flag in enumerate(argv[:-1])
        if flag.startswith("--") and not argv[index + 1].startswith("--")
    }
    expected_tp, expected_cp = ("8", "4") if long_horizon else ("4", "2")
    if (
        options.get("--tensor-model-parallel-size") != expected_tp
        or options.get("--pipeline-model-parallel-size") != "1"
        or options.get("--context-parallel-size") != expected_cp
        or "--sequence-parallel" not in flags
        or options.get("--recompute-granularity") != "full"
        or options.get("--recompute-method") != "uniform"
        or options.get("--recompute-num-layers") != "1"
    ):
        raise ValueError("native Qwen parallelism or full-recompute safety envelope changed")
    return argv
