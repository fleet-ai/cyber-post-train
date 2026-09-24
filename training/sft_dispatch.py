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


def _unqualified_four_node_262k_full(value: dict) -> bool:
    recipe = value.get("recipe", {})
    return (
        isinstance(recipe, dict)
        and "lora" not in value
        and recipe.get("nodes") == 4
        and recipe.get("gpus_per_node") == 8
        and recipe.get("max_length") == 262_144
    )


def compiler_for_config(config: dict):
    variant = _variant(config)
    if variant is None:
        if _unqualified_four_node_262k_full(config):
            raise ValueError(
                "four-node 262K full SFT requires the separately qualified long-context runtime"
            )
        return sft
    if variant == "native_save_return_v1":
        from . import sft_checkpoint_telemetry_v1

        return sft_checkpoint_telemetry_v1
    if variant == "qwen38_lora_anchor_a2_v1":
        from . import sft_lora_anchor_a2_v1

        return sft_lora_anchor_a2_v1
    if variant == "qwen38_sft_32k_matched_v1":
        from . import sft_32k_matched_v1

        return sft_32k_matched_v1
    if variant == "qwen38_sft_32k_domain_anchor_v1":
        from . import sft_32k_domain_anchor_v1

        return sft_32k_domain_anchor_v1
    if variant == "qwen38_sft_262k_4node_v1":
        from . import sft_262k_4node_v1

        return sft_262k_4node_v1
    raise ValueError("SFT runtime variant is not supported by this launch rail")


def compiler_for_plan(plan: dict):
    return compiler_for_config(plan)
