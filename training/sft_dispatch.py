"""Select the narrowly versioned SFT compiler named by immutable input."""

from __future__ import annotations

from . import sft


def _variant(value: dict) -> str | None:
    binding = value.get("runtime_variant")
    if binding is None:
        return None
    if isinstance(binding, str):
        return binding
    if isinstance(binding, dict) and isinstance(binding.get("name"), str):
        return binding["name"]
    raise ValueError("SFT runtime variant binding is malformed")


def compiler_for_config(config: dict):
    variant = _variant(config)
    if variant is None:
        return sft
    if variant == "native_save_return_v1":
        from . import sft_checkpoint_telemetry_v1

        return sft_checkpoint_telemetry_v1
    if variant == "qwen38_lora_anchor_a2_v1":
        from . import sft_lora_anchor_a2_v1

        return sft_lora_anchor_a2_v1
    raise ValueError("SFT runtime variant is not supported by this launch rail")


def compiler_for_plan(plan: dict):
    return compiler_for_config(plan)
