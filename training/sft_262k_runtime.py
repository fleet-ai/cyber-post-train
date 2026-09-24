"""Isolated four-node Qwen3.8 262K runtime derived from retained v12."""

from pathlib import Path

from training import sft_runtime as base

VARIANT = "qwen38_sft_262k_4node_v1"
BASE_RUNTIME_SHA256 = "8cb671f377d089e1303248e237f386c256b212b15e41dadc07ac49cf2695ee17"
QUALIFICATION = {
    "schema": "qwen38_262k_long_context_runtime_port_v1",
    "historical_parent_plan_sha256": (
        "3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690"
    ),
    "historical_parent_runtime_sha256": (
        "b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3"
    ),
    "historical_runtime_source": (
        "git:45c04d709f855e20d931456233b85f427558525f:training/sft_runtime.py"
    ),
    "base_runtime_sha256": BASE_RUNTIME_SHA256,
    "hook_ast_sha256": {
        "chunked_sft_forward": ("554cae31621e5f98eda3c0ad1fae3d64cac8af73f35086b494189784cb47792d"),
        "checkpointed_qwen35_gdn_rule": (
            "b27d5d2b2b76ff00b58dbe2442ed5d1af032e64b65ca7a47c42ee8301f72781a"
        ),
        "chunked_qwen35_mlp_forward": (
            "2fcc9e9e4206ae8a141ba248b9af8f2ff39f92ddb7f2f561f877be7dc2b70f2b"
        ),
        "chunked_qwen35_rmsnorm_forward": (
            "97d07b8ea509208727fb26c33840098bd9b1412efaadc87de3961a2be888e53d"
        ),
        "chunked_qwen35_rmsnorm_gated_forward": (
            "af9f01ad825870a06486942a1a221ddfd2e484f921b0d752c4ea1ad1b27e3a57"
        ),
        "grouped_qwen35_text_forward": (
            "e75bda3427314c0bd1dbe5d54fe438c470817212b39dc115d9bcf95a22acf9ec"
        ),
        "install_chunked_sft_worker": (
            "f11637a5fbbf39bf689afd246ee2eb617b67675b2fd41af53dc20ba9d32a624a"
        ),
    },
}
RELEASE_SUPERVISION = {
    "schema": "qwen38_262k_4node_release_supervision_v1",
    "status": "absent_blocks_submission",
    "poll_seconds": 60,
    "external_deadlines": {
        "gpu_allocation_to_authenticated_started_seconds": 1800,
        "authenticated_started_to_forced_terminal_action_seconds": 29100,
        "gpu_allocation_to_forced_terminal_action_seconds": 30900,
        "gpu_allocation_to_release_confirmation_outer_bound_seconds": 31500,
    },
    "runtime_watchdog_bounds": {
        "anchor": "ProgressWatchdog construction in _wait_for_training",
        "startup_seconds": 1800,
        "idle_seconds": 1200,
        "hard_seconds": 28800,
        "checkpoint_drain_seconds": 300,
    },
    "required_uid_bindings": [
        "RayJob",
        "RayCluster",
        "Kueue Workload",
        "all Pods",
    ],
    "terminal_release_grace_seconds": 300,
    "post_delete_confirmation_seconds": 300,
    "terminal_proof": [
        "all bound Kubernetes objects absent",
        "all 32 requested GPUs released",
    ],
    "uncertain_create_policy": ("reconcile the exact rendered name and run ID before any retry"),
}
SUBMISSION_GATE = {
    "preview_authorized": True,
    "preflight_authorized": True,
    "submission_authorized": False,
    "blockers": [
        "zero-GPU preflight receipt absent",
        "independent exact-UID release supervision absent",
        "four-node GPU launch has not received root review",
    ],
    "required_release_supervision": RELEASE_SUPERVISION,
}

_BASE_VALIDATE_PLAN = base.validate_plan
_BASE_ENTRYPOINT_SOURCES = base._validate_entrypoint_sources
_BASE_SFT_OVERRIDES = base.sft_overrides
_BASE_MAKE_TRAINER_CLASS = base._make_trainer_class
_BASE_RUN_TRAINING = base._run_training


def chunked_sft_forward(
    self,
    sequences,
    num_actions,
    attention_mask=None,
    temperature=1.0,
    return_output=False,
    compute_entropy=False,
    entropy_requires_grad=True,
    pixel_values=None,
    image_grid_thw=None,
    mm_token_type_ids=None,
):
    """Project long Qwen hidden states without materializing all vocabulary logits.

    The native wrapper projects every token to the 248,320-wide vocabulary at
    once.  A 262,144-token row therefore asks for a 121-GiB BF16 tensor on each
    rank.  SFT needs only the selected-token log probability, so checkpoint the
    projection/cross-entropy in bounded token slices.  Backward recomputes one
    slice at a time and preserves the exact causal CE objective.
    """
    import types

    import numpy as np
    import torch
    from skyrl.backends.skyrl_train.workers import model_wrapper
    from torch.utils.checkpoint import checkpoint

    if (
        pixel_values is not None
        or image_grid_thw is not None
        or mm_token_type_ids is not None
        or (compute_entropy and entropy_requires_grad)
    ):
        raise ValueError("chunked LM head is restricted to text-only SFT without entropy loss")
    chunk_tokens = getattr(self, "fleet_sft_lm_head_chunk_tokens", None)
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("chunked LM-head token bound is missing")

    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    sequences_fwd, position_ids_fwd, attention_mask_fwd = sequences, position_ids, attention_mask
    nnz_indices = None
    if self.remove_microbatch_padding:
        with torch.no_grad():
            sequences_fwd, nnz_indices, _, _, _ = model_wrapper.unpad_input(
                sequences.unsqueeze(-1), attention_mask=attention_mask
            )
            sequences_fwd = sequences_fwd.transpose(0, 1)
            position_ids_fwd, _, _, _, _ = model_wrapper.unpad_input(
                position_ids.unsqueeze(-1), attention_mask
            )
            position_ids_fwd = position_ids_fwd.transpose(0, 1)
            attention_mask_fwd = None

    causal_model = self.model
    labels = torch.roll(sequences_fwd, shifts=-1, dims=1)
    if self.sequence_parallel_size != 1:
        raise ValueError("chunked Qwen SFT forbids sequence parallelism across recurrent GDN")
    backbone_kwargs = {
        "input_ids": sequences_fwd,
        "attention_mask": attention_mask_fwd,
        "position_ids": None if self.is_vlm else position_ids_fwd,
        "use_cache": False,
    }

    def selected_logprobs(module, hidden, target):
        logits = module.lm_head(hidden)
        if temperature != 1.0:
            logits = logits / temperature
        return model_wrapper.logprobs_from_logits(logits, target, inplace_backward=True)

    def chunked_causal_forward(module, **kwargs):
        hidden_states = module.model(**kwargs).last_hidden_state
        chunks = []
        for start in range(0, hidden_states.shape[1], chunk_tokens):
            stop = min(start + chunk_tokens, hidden_states.shape[1])
            hidden, target = hidden_states[:, start:stop], labels[:, start:stop]
            if torch.is_grad_enabled() and hidden.requires_grad:
                value = checkpoint(
                    lambda h, t: selected_logprobs(module, h, t),
                    hidden,
                    target,
                    use_reentrant=False,
                )
            else:
                value = selected_logprobs(module, hidden, target)
            chunks.append(value)
        return torch.cat(chunks, dim=1)

    # The causal model itself is the FSDP2 root. Calling `.model` directly
    # bypasses its pre/post-forward hooks and leaves root-owned parameters as
    # DTensors. Projecting after the root returns is also invalid because its
    # LM head may already be re-sharded. Temporarily replace only the Python
    # forward implementation so one FSDP lifecycle covers both the backbone
    # and the bounded vocabulary projection.
    had_instance_forward = "forward" in causal_model.__dict__
    instance_forward = causal_model.__dict__.get("forward")
    causal_model.forward = types.MethodType(chunked_causal_forward, causal_model)
    try:
        log_probs = causal_model(**backbone_kwargs)
    finally:
        if had_instance_forward:
            causal_model.forward = instance_forward
        else:
            del causal_model.forward

    if self.remove_microbatch_padding:
        batch_size, seqlen = attention_mask.shape
        log_probs = model_wrapper.pad_input(
            log_probs.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
        ).squeeze(-1)

    if isinstance(num_actions, list):
        num_actions = num_actions[0] if len(num_actions) == 1 else np.array(num_actions)
    action_log_probs = log_probs[:, -num_actions - 1 : -1]
    return (action_log_probs, {}) if return_output else action_log_probs


def checkpointed_qwen35_gdn_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    *,
    outer_chunk_tokens,
    implementation,
):
    """Preserve Qwen's Torch delta rule with bounded recurrent autograd history.

    The fallback kernel already carries an exact recurrent state between its
    64-token blocks, but one 262k call leaves every block in a single autograd
    graph.  Checkpoint larger, 64-aligned sequence segments and pass the same
    recurrent state between them.  Backward then recomputes one segment at a
    time without changing any token, state transition, output, or gradient.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    if (
        type(outer_chunk_tokens) is not int
        or outer_chunk_tokens < chunk_size
        or outer_chunk_tokens % chunk_size
    ):
        raise ValueError("Gated DeltaNet checkpoint chunks must align to its kernel blocks")
    if query.ndim != 4 or query.shape[1] <= 0:
        raise ValueError("Gated DeltaNet needs a nonempty sequence")

    outputs = []
    state = initial_state
    for start in range(0, query.shape[1], outer_chunk_tokens):
        stop = min(start + outer_chunk_tokens, query.shape[1])
        segment = (
            query[:, start:stop],
            key[:, start:stop],
            value[:, start:stop],
            g[:, start:stop],
            beta[:, start:stop],
        )
        needs_grad = torch.is_grad_enabled() and any(x.requires_grad for x in segment)
        if state is None:

            def run(q, k, v, decay, step):
                return implementation(
                    q,
                    k,
                    v,
                    decay,
                    step,
                    chunk_size=chunk_size,
                    initial_state=None,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )

            output, state = (
                checkpoint(run, *segment, use_reentrant=False) if needs_grad else run(*segment)
            )
        else:

            def run(q, k, v, decay, step, previous_state):
                return implementation(
                    q,
                    k,
                    v,
                    decay,
                    step,
                    chunk_size=chunk_size,
                    initial_state=previous_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                )

            inputs = (*segment, state)
            output, state = (
                checkpoint(run, *inputs, use_reentrant=False)
                if needs_grad or state.requires_grad
                else run(*inputs)
            )
        outputs.append(output)
    return torch.cat(outputs, dim=1), state if output_final_state else None


def chunked_qwen35_mlp_forward(self, hidden_states):
    """Evaluate Qwen's pointwise MLP in bounded token slices.

    Chunking the token dimension preserves the gate/up/down computation for
    every token while avoiding the full 262k-token intermediate activation.
    Per-chunk activation checkpointing also keeps backward from retaining all
    gate/up intermediates at once during the decoder-layer recomputation.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    chunk_tokens = getattr(self, "fleet_sft_mlp_chunk_tokens", None)
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("chunked Qwen MLP token bound is missing")
    if hidden_states.ndim < 2 or hidden_states.shape[-2] <= 0:
        raise ValueError("chunked Qwen MLP requires a nonempty token dimension")

    def project(value):
        return self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))

    outputs = []
    for start in range(0, hidden_states.shape[-2], chunk_tokens):
        value = hidden_states[..., start : start + chunk_tokens, :]
        if torch.is_grad_enabled() and value.requires_grad:
            value = checkpoint(project, value, use_reentrant=False)
        else:
            value = project(value)
        outputs.append(value)
    return torch.cat(outputs, dim=-2)


def chunked_qwen35_rmsnorm_forward(self, hidden_states):
    """Preserve Qwen RMSNorm while bounding its full-sequence F32 scratch."""
    import torch

    chunk_tokens = getattr(self, "fleet_sft_rmsnorm_chunk_tokens", None)
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("invalid Qwen RMSNorm chunk size")
    scale = 1.0 + self.weight.float()
    outputs = []
    for value in hidden_states.split(chunk_tokens, dim=-2):
        normalized = self._norm(value.float())
        outputs.append((normalized * scale).type_as(value))
    return torch.cat(outputs, dim=-2)


def chunked_qwen35_rmsnorm_gated_forward(self, hidden_states, gate=None):
    """Preserve gated RMSNorm while bounding its F32 state and gate scratch."""
    import torch

    chunk_tokens = getattr(self, "fleet_sft_rmsnorm_chunk_tokens", None)
    if type(chunk_tokens) is not int or chunk_tokens <= 0:
        raise ValueError("invalid Qwen gated RMSNorm chunk size")
    if gate is None or gate.shape != hidden_states.shape:
        raise ValueError("Qwen gated RMSNorm requires a matching gate")
    outputs = []
    for value, gate_value in zip(
        hidden_states.split(chunk_tokens, dim=-2),
        gate.split(chunk_tokens, dim=-2),
        strict=True,
    ):
        input_dtype = value.dtype
        value = value.to(torch.float32)
        variance = value.pow(2).mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        value = self.weight * value.to(input_dtype)
        # Qwen3.5 fixes this gate to SiLU. The exact-image kernel wrapper does
        # not preserve the local fallback class's descriptive `activation` attr.
        value = value * torch.nn.functional.silu(gate_value.to(torch.float32))
        outputs.append(value.to(input_dtype))
    return torch.cat(outputs, dim=-2)


def grouped_qwen35_text_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """Run exact Qwen layers in checkpoint groups, retaining fewer 262k boundaries."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)
    if position_ids is None:
        seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + seen
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )
    linear_attn_mask = (
        self._update_linear_attn_mask(attention_mask, past_key_values)
        if "linear_attention" in self.config.layer_types
        else attention_mask
    )
    position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
    layers = self.layers[: self.config.num_hidden_layers]
    group_size = getattr(self, "fleet_sft_layer_checkpoint_group_size", None)
    if type(group_size) is not int or not 1 <= group_size <= len(layers):
        raise ValueError("invalid Qwen layer checkpoint group size")

    hidden_states = inputs_embeds
    for start in range(0, len(layers), group_size):
        stop = min(start + group_size, len(layers))

        def run_group(value, start=start, stop=stop):
            for index in range(start, stop):
                layer = layers[index]
                # Transformers otherwise checkpoints every 262k layer boundary.
                layer.gradient_checkpointing = False
                layer_mask = (
                    linear_attn_mask
                    if self.config.layer_types[index] == "linear_attention"
                    else causal_mask
                )
                value = layer(
                    value,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )
            return value

        if self.training and torch.is_grad_enabled() and hidden_states.requires_grad:
            # Reentrant outer checkpointing deliberately runs the first group
            # forward under no_grad.  That prevents the nested GDN segment
            # checkpoints from retaining every group's full Q/K/V projections
            # until the end of a 262k forward.  During backward the group is
            # recomputed with gradients enabled, so the segment checkpoints
            # bound recurrent history while preserving exact gradients.
            hidden_states = checkpoint(run_group, hidden_states, use_reentrant=True)
        else:
            hidden_states = run_group(hidden_states)
    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=self.norm(hidden_states),
        past_key_values=past_key_values,
    )


def install_chunked_sft_worker(
    lm_head_chunk_tokens: int,
    mlp_chunk_tokens: int,
    rmsnorm_chunk_tokens: int,
    gdn_chunk_tokens: int,
    layer_checkpoint_group_size: int,
):
    """Install one job-local Ray worker using bounded Qwen projections."""
    import ray
    from skyrl.backends.skyrl_train.workers.fsdp import fsdp_worker
    from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import FSDPPolicyWorkerBase

    class ChunkedSFTPolicyWorker(FSDPPolicyWorkerBase):
        def init_model(self, *args, **kwargs):
            from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper
            from transformers.models.qwen3_5 import modeling_qwen3_5
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5MLP,
                Qwen3_5RMSNorm,
                Qwen3_5RMSNormGated,
                Qwen3_5TextModel,
            )

            configured_lm_head = self.cfg.policy.model_config_kwargs.pop(
                "fleet_sft_lm_head_chunk_tokens", None
            )
            configured_mlp = self.cfg.policy.model_config_kwargs.pop(
                "fleet_sft_mlp_chunk_tokens", None
            )
            configured_rmsnorm = self.cfg.policy.model_config_kwargs.pop(
                "fleet_sft_rmsnorm_chunk_tokens", None
            )
            configured_gdn = self.cfg.policy.model_config_kwargs.pop(
                "fleet_sft_gdn_chunk_tokens", None
            )
            configured_group = self.cfg.policy.model_config_kwargs.pop(
                "fleet_sft_layer_checkpoint_group_size", None
            )
            if configured_lm_head != lm_head_chunk_tokens:
                raise ValueError("chunked LM-head worker configuration drift")
            if configured_mlp != mlp_chunk_tokens:
                raise ValueError("chunked MLP worker configuration drift")
            if configured_rmsnorm != rmsnorm_chunk_tokens:
                raise ValueError("chunked RMSNorm worker configuration drift")
            if configured_gdn != gdn_chunk_tokens:
                raise ValueError("chunked Gated DeltaNet worker configuration drift")
            if configured_group != layer_checkpoint_group_size:
                raise ValueError("layer checkpoint group worker configuration drift")
            HFModelWrapper.fleet_sft_lm_head_chunk_tokens = lm_head_chunk_tokens
            HFModelWrapper.forward = chunked_sft_forward
            Qwen3_5MLP.fleet_sft_mlp_chunk_tokens = mlp_chunk_tokens
            Qwen3_5MLP.forward = chunked_qwen35_mlp_forward
            Qwen3_5RMSNorm.fleet_sft_rmsnorm_chunk_tokens = rmsnorm_chunk_tokens
            Qwen3_5RMSNorm.forward = chunked_qwen35_rmsnorm_forward
            Qwen3_5RMSNormGated.fleet_sft_rmsnorm_chunk_tokens = rmsnorm_chunk_tokens
            Qwen3_5RMSNormGated.forward = chunked_qwen35_rmsnorm_gated_forward
            native_gdn_rule = modeling_qwen3_5.torch_chunk_gated_delta_rule
            if getattr(native_gdn_rule, "__name__", None) != "torch_chunk_gated_delta_rule":
                raise ValueError("native Qwen Gated DeltaNet rule has already been replaced")

            def bounded_gdn_rule(*args, **kwargs):
                return checkpointed_qwen35_gdn_rule(
                    *args,
                    **kwargs,
                    outer_chunk_tokens=gdn_chunk_tokens,
                    implementation=native_gdn_rule,
                )

            modeling_qwen3_5.torch_chunk_gated_delta_rule = bounded_gdn_rule
            Qwen3_5TextModel.fleet_sft_layer_checkpoint_group_size = layer_checkpoint_group_size
            Qwen3_5TextModel.forward = grouped_qwen35_text_forward
            return super().init_model(*args, **kwargs)

    fsdp_worker.PolicyWorker = ray.remote(num_gpus=1)(ChunkedSFTPolicyWorker)


def validate_plan(plan: dict, *, check_files: bool = True) -> None:
    """Admit only the exact four-node derivative of retained v12."""
    _BASE_VALIDATE_PLAN(plan, check_files=check_files)
    recipe = plan.get("recipe", {})
    expected_recipe = {
        "epochs": 1,
        "batch_size": 32,
        "microbatch_per_gpu": 1,
        "nodes": 4,
        "gpus_per_node": 8,
        "sequence_parallel_size": 1,
        "max_length": 262_144,
        "lm_head_chunk_tokens": 1024,
        "mlp_chunk_tokens": 1024,
        "rmsnorm_chunk_tokens": 1024,
        "gdn_chunk_tokens": 512,
        "layer_checkpoint_group_size": 1,
        "eval_interval": 0,
        "checkpoint_interval": 1,
        "keep_checkpoints": 2,
        "max_steps": 4,
        "lr": 3e-6,
        "seed": 20260916,
    }
    gate = {
        "schema": "qwen38_262k_four_node_submission_gate_v1",
        "submission_gate": SUBMISSION_GATE,
    }
    expected_wandb = {
        "entity": "thefleet",
        "project": "cyber-post-train",
        "group": "qwen38-teacher3k-262k-v1",
        "run_id": "chris-q38-t3k262-4n-can-v1",
        "name": "chris-q38-t3k262-4n-can-v1",
        "tags": [
            "qwen38",
            "teacher-sft",
            "broad-successes",
            "262k",
            "four-node-capacity-canary",
            "full-weight",
            "gdn-checkpoint512",
            "chunk1024",
            "group1",
            "reentrant-outer-checkpoint",
            "planned-pause-step1",
            "held-nonlaunchable",
        ],
    }
    expected_keys = {
        "schema",
        "runtime_variant",
        "validation_mode",
        "model",
        "datasets",
        "recipe",
        "execution",
        "wandb",
        "run_name",
        "output_root",
        "pause_after_step",
        "split_manifest_sha256",
        "corpus_manifest_sha256",
        "runtime_sha256",
        "long_context_qualification",
        "qualification",
    }
    expected_execution = {
        "image": (
            "661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@"
            "sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4"
        ),
        "priority": "c1",
        "resources": {
            "cpu_limit": "64",
            "cpu_request": "64",
            "memory_limit": "1024Gi",
            "memory_request": "512Gi",
        },
        "cluster_target": "prod",
        "jobs_api_base_url": "https://api.ft.flt.build",
    }
    if (
        set(plan) != expected_keys
        or plan.get("runtime_variant") != VARIANT
        or plan.get("model", {}).get("repo") != "Qwen/Qwen3.8-27B"
        or "lora" in plan
        or base._unsigned_digest(plan.get("model", {}))
        != "dcfdcd6ecb6661741cd3a4b24dc5af7259642c8a6824773e0de70d55d7501179"
        or base._unsigned_digest(plan.get("datasets", {}))
        != "65bc752ad8597c68d6ff35f2078918a53fc2a6427f3be6eda33610a98555d46a"
        or plan.get("split_manifest_sha256")
        != "sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c"
        or plan.get("corpus_manifest_sha256")
        != "sha256:c5c7d82127d524593ed7e1cf5375fb3457e36c0d92376531808b6eeaacda10f5"
        or plan.get("run_name") != "chris-q38-t3k262-4n-can-v1"
        or plan.get("output_root") != "/mnt/sfs/jobs/chris-q38-t3k262-4n-can-v1"
        or plan.get("wandb") != expected_wandb
        or plan.get("pause_after_step") != 1
        or recipe != expected_recipe
        or plan.get("long_context_qualification") != QUALIFICATION
        or plan.get("execution") != expected_execution
        or plan.get("qualification") != gate
    ):
        raise ValueError("four-node 262K plan differs from its exact qualified hypothesis")
    if (
        any(
            recipe[key] > recipe["max_length"]
            for key in (
                "lm_head_chunk_tokens",
                "mlp_chunk_tokens",
                "rmsnorm_chunk_tokens",
                "gdn_chunk_tokens",
            )
        )
        or recipe["gdn_chunk_tokens"] % 64
    ):
        raise ValueError("four-node 262K chunk bounds are invalid")
    if check_files:
        base._checked_file(Path(base.__file__), BASE_RUNTIME_SHA256)
        base._checked_file(Path(__file__), plan["runtime_sha256"])


def sft_overrides(plan: dict) -> dict:
    options = _BASE_SFT_OVERRIDES(plan)
    recipe = plan["recipe"]
    options.update(
        {
            "sequence_parallel_size": recipe["sequence_parallel_size"],
            "model_config_kwargs.fleet_sft_lm_head_chunk_tokens": recipe["lm_head_chunk_tokens"],
            "model_config_kwargs.fleet_sft_mlp_chunk_tokens": recipe["mlp_chunk_tokens"],
            "model_config_kwargs.fleet_sft_rmsnorm_chunk_tokens": recipe["rmsnorm_chunk_tokens"],
            "model_config_kwargs.fleet_sft_gdn_chunk_tokens": recipe["gdn_chunk_tokens"],
            "model_config_kwargs.fleet_sft_layer_checkpoint_group_size": recipe[
                "layer_checkpoint_group_size"
            ],
            "fsdp_config.cpu_offload": False,
            "optimizer_config.offload_after_step": False,
        }
    )
    return options


def _make_trainer_class():
    parent = _BASE_MAKE_TRAINER_CLASS()

    class LongContextTrainer(parent):
        def _init_workers(self):
            recipe = self.plan["recipe"]
            install_chunked_sft_worker(
                recipe["lm_head_chunk_tokens"],
                recipe["mlp_chunk_tokens"],
                recipe["rmsnorm_chunk_tokens"],
                recipe["gdn_chunk_tokens"],
                recipe["layer_checkpoint_group_size"],
            )
            super()._init_workers()

    return LongContextTrainer


def _validate_entrypoint_sources(plan: dict, *, verify_qwen_files: bool) -> None:
    _BASE_ENTRYPOINT_SOURCES(plan, verify_qwen_files=verify_qwen_files)
    base._checked_file(Path(base.__file__), BASE_RUNTIME_SHA256)
    base._checked_file(Path(__file__), plan["runtime_sha256"])


def _run_training(plan: dict) -> dict:
    install_runtime()
    return _BASE_RUN_TRAINING(plan)


def install_runtime() -> None:
    """Patch only this process and its candidate-specific Ray entrypoint."""
    base.validate_plan = validate_plan
    base.sft_overrides = sft_overrides
    base._make_trainer_class = _make_trainer_class
    base._validate_entrypoint_sources = _validate_entrypoint_sources
    base._run_training = _run_training


def main() -> None:
    install_runtime()
    base.main()


if __name__ == "__main__":
    from training import sft_262k_runtime as canonical

    canonical.main()
