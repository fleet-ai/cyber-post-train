"""Probe Qwen3.6 gated-delta backward paths on one B300.

The failing CUDA fault poisons a process, so every case runs in a fresh child.
The RL case mirrors one per-rank microbatch from ``ft-run-7b21930b``; the SFT
case mirrors the long-sequence reproducer used for ``ft-run-b786dd74``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


INNER_PROBE = r"""
import importlib.metadata as md
import json
import os
import time

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

case = json.loads(os.environ["FLEET_FLA_PROBE_CASE"])
torch.manual_seed(20260831)
B, T, H, HV, K, V = case["shape"]

q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g0 = torch.randn(B, T, HV, device="cuda", dtype=torch.float32, requires_grad=True)
b0 = torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g = -torch.exp(torch.zeros(HV, device="cuda")) * torch.nn.functional.softplus(g0)
beta = torch.sigmoid(b0)

started = time.monotonic()
output, final_state = chunk_gated_delta_rule(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    output_final_state=False,
    use_qk_l2norm_in_kernel=True,
    chunk_size=case["chunk_size"],
)
assert final_state is None
loss = output.float().square().mean()
loss.backward()
torch.cuda.synchronize()

print(json.dumps({
    "ok": True,
    "case": case["name"],
    "shape": case["shape"],
    "chunk_size": case["chunk_size"],
    "elapsed_seconds": round(time.monotonic() - started, 3),
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "fla": md.version("flash-linear-attention"),
    "torch": torch.__version__,
    "triton": md.version("triton"),
    "finite_output": bool(torch.isfinite(output).all()),
    "finite_q_grad": bool(torch.isfinite(q.grad).all()),
}), flush=True)
"""


CASES = (
    # Qwen declares 16 key heads, then repeats q/k threefold for grouped-value
    # attention before entering FLA. The kernel therefore sees H=HV=48.
    {"name": "rl_default_reproduction", "shape": [1, 329, 48, 48, 128, 128], "chunk_size": 64},
    {"name": "rl_chunk32", "shape": [1, 329, 48, 48, 128, 128], "chunk_size": 32},
    {"name": "rl_chunk16", "shape": [1, 329, 48, 48, 128, 128], "chunk_size": 16},
    {"name": "sft_chunk32", "shape": [1, 14336, 48, 48, 128, 128], "chunk_size": 32},
)


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
    for case in CASES:
        env = {
            **os.environ,
            "CUDA_LAUNCH_BLOCKING": "1",
            "FLEET_FLA_PROBE_CASE": json.dumps(case),
        }
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, "-c", INNER_PROBE],
                env=env,
                text=True,
                capture_output=True,
                timeout=900,
                check=False,
            )
            status = categorize(completed)
            result: dict[str, object] = {
                **case,
                "status": status,
                "wall_seconds": round(time.monotonic() - started, 3),
            }
            if status == "passed":
                detail = next(
                    json.loads(line)
                    for line in reversed(completed.stdout.splitlines())
                    if line.startswith("{")
                )
                result.update(detail)
            else:
                result["returncode"] = completed.returncode
                result["stderr_tail"] = completed.stderr[-1200:]
        except subprocess.TimeoutExpired:
            result = {
                **case,
                "status": "timeout",
                "wall_seconds": round(time.monotonic() - started, 3),
            }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    alternatives = [r for r in results if r["name"] != "rl_default_reproduction"]
    print(json.dumps({"summary": results}, sort_keys=True), flush=True)
    return 0 if any(r["status"] == "passed" for r in alternatives) else 1


if __name__ == "__main__":
    raise SystemExit(main())
