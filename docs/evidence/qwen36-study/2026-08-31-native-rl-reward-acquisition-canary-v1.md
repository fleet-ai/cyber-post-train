# Native Qwen3.6 reward-acquisition canary

Status: **blocked pre-submission**. This note and its companion config do not authorize a paid run.
That state reflects unresolved technical identity/preflight gates, not missing user authorization.

The canary is deliberately smaller than the three-task Agent Runtime parity study. It asks one
question first: can native Qwen acquire any authoritative binary reward across eight rollouts,
complete exactly one optimizer step, and retain one checkpoint when given a materially adequate
cyber horizon?

## Frozen shape

- Two historically nonzero source versions: `54425601-6fd2-43d8-8cb9-e565b767676a` and
  `f31ebe83-0ff1-4660-bcba-59ffa4b82d5a`. Their metadata-only successor IDs are explicitly pending;
  the source versions must not be substituted at launch.
- One 8xB300 worker; TP4 yields two colocated inference engines.
- GRPO group size 4, train batch 2, policy mini-batch 2: eight rollouts and one true optimizer step.
- 65,536-token context: 16,384 prompt plus 49,152 flat post-prompt trajectory; at most 2,048 sampled
  tokens per turn and a 600-turn ceiling.
- Ordered task surface `[bash, submit_report]`, 16,000-character tool-result cap, temperature 1,
  top-p 1, binary exact-version Fleet verifier reward, no eval, no compaction, one retained checkpoint.

The source identities and model lineage remain anchored in
[`qwen36-27b-native-rl-two-task-gate-v1.json`](../../../configs/data/qwen36-27b-native-rl-two-task-gate-v1.json).
The executable shape and all unresolved identities are in
[`qwen36-27b-native-rl-reward-acquisition-canary.pre-submit.json`](../../../configs/runs/qwen36-27b-native-rl-reward-acquisition-canary.pre-submit.json).

## Why it remains blocked

Merged Theseus PRs #28407 and #28433 provide the long-horizon controls and fail-closed tool
execution boundary; #28441 pins the corresponding `2bc0ba51` trainer build. Merge status is not
deployment evidence. Before launch, the exact trainer catalog row and image digest must be Ready;
both task versions need exact metadata-only successors and authoritative preview evidence; prompt
plus complete schema tokenization must fit; and each episode must bind its exact verifier execution.

Until then, the request intentionally has no trainer UUID and no task-version bindings. The current
paid-launch gate rejects it. Filling only those blanks is not sufficient: every structured blocker
in the companion file must have immutable evidence.

The canary itself measures whether the requested 65k TP4/two-engine shape fits on 8xB300. CUDA OOM,
engine initialization failure, or invalid KV-cache sizing is an infrastructure/configuration
failure and cannot support an optimizer or model-incapability claim. The native loop may execute
multiple parsed calls from one assistant turn sequentially; this run records that frequency as a
diagnostic. It remains a blocker for a later Agent Runtime parity claim, not for this native-only
reward-acquisition measurement.
