"""Validate Qwen's PyTorch gated-delta fallback at the exact B300 training shape.

The FLA/Triton operator is deliberately not called here: a device-side address
fault poisons the CUDA context and would make the fallback result ambiguous.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import time

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule


def tensor_summary(tensor: torch.Tensor) -> dict[str, float | bool]:
    detached = tensor.detach().float()
    return {
        "finite": bool(torch.isfinite(detached).all().item()),
        "norm": float(detached.norm().item()),
        "abs_max": float(detached.abs().max().item()),
    }


def main() -> int:
    torch.manual_seed(20260831)
    torch.cuda.set_device(0)

    # Qwen3.6-27B declares 16 key heads, then repeats q/k threefold before the
    # gated-delta operator. These are therefore the operator's real dimensions.
    batch, tokens, heads, key_dim, value_dim = 1, 14336, 48, 128, 128
    dtype = torch.bfloat16
    device = torch.device("cuda", 0)

    common = {"device": device, "dtype": dtype, "requires_grad": True}
    query = torch.randn(batch, tokens, heads, key_dim, **common)
    key = torch.randn(batch, tokens, heads, key_dim, **common)
    value = torch.randn(batch, tokens, heads, value_dim, **common)
    float_common = {"device": device, "dtype": torch.float32, "requires_grad": True}
    raw_g = torch.randn(batch, tokens, heads, **float_common)
    raw_beta = torch.randn(batch, tokens, heads, **float_common)
    gate = -torch.nn.functional.softplus(raw_g)
    beta = torch.sigmoid(raw_beta)

    started = time.monotonic()
    output, final_state = torch_chunk_gated_delta_rule(
        query,
        key,
        value,
        g=gate,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    loss = output.float().square().mean() + final_state.float().square().mean() * 1e-6
    gradients = torch.autograd.grad(loss, (query, key, value, raw_g, raw_beta))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started

    result = {
        "schema": "qwen_blackwell_torch_gated_delta_probe_v1",
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "transformers": metadata.version("transformers"),
        "fla": metadata.version("flash-linear-attention"),
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "heads": heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "dtype": str(dtype),
        },
        "elapsed_seconds": elapsed,
        "loss": float(loss.detach().item()),
        "output": tensor_summary(output),
        "final_state": tensor_summary(final_state),
        "gradients": {
            name: tensor_summary(gradient)
            for name, gradient in zip(
                ("query", "key", "value", "raw_g", "raw_beta"), gradients, strict=True
            )
        },
    }
    result["passed"] = (
        result["output"]["finite"]
        and result["final_state"]["finite"]
        and all(summary["finite"] for summary in result["gradients"].values())
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
