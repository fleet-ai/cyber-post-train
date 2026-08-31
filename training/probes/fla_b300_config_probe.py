"""Find a safe FLA gated-delta backward launch config on Qwen's B300 shape.

This file is intended to run inside the exact trainer image. Each candidate is
tested in a fresh subprocess because a Triton misaligned-address fault poisons
the current CUDA context. Only categorical outcomes are printed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

INNER_PROBE = r"""
import importlib.metadata as md
import json

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

torch.manual_seed(20260830)
# Qwen declares 16 key heads but repeats query/key threefold before invoking
# FLA, so the kernel itself receives 48 query, key and value heads.
B, T, H, HV, K, V = 1, 14336, 48, 48, 128, 128
q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g0 = torch.randn(B, T, HV, device="cuda", dtype=torch.float32, requires_grad=True)
b0 = torch.randn(B, T, HV, device="cuda", dtype=torch.float32, requires_grad=True)
g = torch.nn.functional.logsigmoid(g0)
beta = torch.sigmoid(b0).to(torch.bfloat16)
output, final_state = chunk_gated_delta_rule(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    output_final_state=True,
)
loss = output.float().square().mean() + final_state.float().square().mean() * 1e-6
loss.backward()
torch.cuda.synchronize()
print(json.dumps({
    "ok": True,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "fla": md.version("flash-linear-attention"),
    "torch": torch.__version__,
    "triton": md.version("triton"),
}))
"""


def candidate_config(warps: int, stages: int) -> dict[str, object]:
    return {
        "kernel_name": "prepare_wy_repr_bwd_kernel",
        "triton_version": None,
        "autotune_entries": None,
        "default_config": {
            "kwargs": {},
            "num_warps": warps,
            "num_stages": stages,
            "num_ctas": 1,
            "maxnreg": None,
            "ir_override": None,
        },
    }


def categorize(completed: subprocess.CompletedProcess[str]) -> str:
    combined = (completed.stdout + "\n" + completed.stderr).lower()
    if completed.returncode == 0:
        return "passed"
    if "misaligned address" in combined:
        return "misaligned_address"
    if "out of memory" in combined:
        return "out_of_memory"
    return "other_failure"


def main() -> int:
    results: list[dict[str, object]] = []
    for warps, stages in ((2, 4), (2, 2), (2, 3), (4, 2), (4, 3), (4, 4)):
        config_dir = Path(tempfile.mkdtemp(prefix=f"fla-{warps}-{stages}-"))
        (config_dir / "prepare_wy_repr_bwd_kernel.json").write_text(
            json.dumps(candidate_config(warps, stages))
        )
        env = {
            **os.environ,
            "FLA_CACHE_MODE": "default",
            "FLA_CONFIG_DIR": str(config_dir),
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-c", INNER_PROBE],
                env=env,
                text=True,
                capture_output=True,
                timeout=420,
                check=False,
            )
            status = categorize(completed)
            result: dict[str, object] = {
                "warps": warps,
                "stages": stages,
                "status": status,
            }
            if status == "passed":
                detail = next(
                    json.loads(line)
                    for line in reversed(completed.stdout.splitlines())
                    if line.startswith("{")
                )
                result.update(detail)
            elif status == "other_failure":
                result["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            result = {"warps": warps, "stages": stages, "status": "timeout"}
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"summary": results}, sort_keys=True))
    return 0 if any(result["status"] == "passed" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
