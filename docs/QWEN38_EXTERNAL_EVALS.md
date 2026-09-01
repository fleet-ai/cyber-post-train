# Qwen3.8-27B external-evaluation track

## State

The exact base is `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The Fleet route
`qwen3.8-27b` was observed Ready 1/1 with BF16 weights, tensor parallelism 1,
the same digest-pinned SGLang image used by Qwen3.6, 262,144-token context,
FP8 KV cache, TensorRT-LLM MHA, `qwen3` reasoning parsing, and
`qwen3_coder` tool parsing. A forced structured-tool request returned exactly
one valid call.

The following are frozen:

- the official model revision, 18-shard weight manifest, tokenizer manifest,
  chat template, and live serving-spec digest;
- WebExploitBench Level 0 with CAGE
  `09a191c565230cebb8255899d622d23c7ddeff33`, Qwen Code 0.22.3, the same
  linux/amd64 harness image, 15 targets / 110 vulnerabilities, pass@1,
  one-target concurrency, 262,144-token student context, 65,536-token GLM-5.3
  judge context, and the existing grading and retry policy;
- ExploitGym's exact five task identities through the immutable Qwen3.6 source
  protocol digest, together with the independently verified control image
  `ghcr.io/fleet-ai/cyber-post-train-exploitgym-control@sha256:466027a5b270e822acbba676ccd775f443e7ed897fdf2d5a3f824f776141e7a0`,
  dynamic graders, firewall, pass@1, single-worker, and 3,600-second budget.

## Interpretation boundary

Qwen3.8-27B was released on 2026-08-14, after WebExploitBench. This track can
measure a controlled delta between exact Qwen3.8 base and post-training
checkpoints. It is not a temporally clean holdout claim, because benchmark
contamination cannot be ruled out. External scores and traces remain sealed and
evaluation-only; none may influence training, rewards, checkpoint selection,
or prompts.

## Launch gates

Run the static control check, then the inexpensive live route/chat/tool probe:

```sh
uv run python -m evals.qwen38_study validate
uv run python -m evals.qwen38_study probe --output /restricted/qwen38-live-preflight.json
```

The WebExploitBench scored base arm may launch only from its single-use absent
run root with the pinned local harness image. ExploitGym remains prepared but
unlaunched until a post-training checkpoint exists, so its two arms can be
counterbalanced and run close in time under one final protocol. A successful
probe is an operational gate, not a capability result.
