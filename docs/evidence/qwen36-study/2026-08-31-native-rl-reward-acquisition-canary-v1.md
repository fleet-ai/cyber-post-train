# Native Qwen3.6 reward-acquisition canary

Status: **blocked pre-submission**. This note and its companion config do not authorize a paid run.
That state reflects unresolved technical identity/preflight gates, not missing user authorization.

The canary is deliberately smaller than the three-task Agent Runtime parity study. It asks one
question first: can native Qwen acquire any authoritative binary reward across eight rollouts,
complete exactly one optimizer step, and retain one checkpoint when given a materially adequate
cyber horizon?

## Frozen shape

- Two historically nonzero source versions have exact metadata-only successors. Source
  `54425601-6fd2-43d8-8cb9-e565b767676a` version 8 maps to successor
  `c99340e2-3801-5c3c-a50c-3b96cee4572f` version 9. Source
  `f31ebe83-0ff1-4660-bcba-59ffa4b82d5a` version 12 maps to successor
  `ab5f2956-9fbb-54fb-afd8-67fe11402fa4` version 13. The source versions must not be substituted at
  launch.
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

## Successor identity evidence

The one-shot production operation used fixed idempotency key
`qwen36-harness-parity-bash-submit-report-v1` on deployed Theseus revision
`bb9cc39d9386dc4376cb501c72908444ed0de033`. Its two per-version receipts prove that every
content-bearing field and hash is equal to the corresponding source except the single semantic
path `/metadata/tools`, whose exact ordered value is `[bash, submit_report]`. The operation created
both versions in one transaction, did not promote either successor, and left both parent rows and
their `current_version_id` pointers unchanged.

The canary request is now bound to those two successor IDs and never resolves mutable current
versions. A non-submitting Training API preview then resolved those exact IDs from authoritative
`metadata.tools`, returned ordered `[bash, submit_report]` for both with binding digest
`sha256:5a8fa677c0938d8b9c16f6e9c4b8f9819375e2aaeeb3b6b4e3c42f3ab897d62f`, and rendered exactly
one `trainer.max_training_steps=1` in the top-level entrypoint. The sanitized preview receipt is
[`qwen36-27b-native-rl-reward-acquisition-canary.preview.json`](../../../configs/runs/qwen36-27b-native-rl-reward-acquisition-canary.preview.json).

The preview did not expose hydrated prompt bytes or complete runtime tool schemas, and no secure
local Fleet task credential was available to obtain those private values independently. Therefore
the model-visible prompt-plus-schema token-fit measurement remains explicitly unproven.

## Why it remains blocked

Merged Theseus PRs #28407 and #28433 provide the long-horizon controls and fail-closed tool
execution boundary; #28441 pins the corresponding `2bc0ba51` trainer build. Merge status is not
deployment evidence. Before launch, the exact trainer catalog row and image digest must be Ready;
prompt plus complete schema tokenization must fit; and each episode must bind its exact verifier
execution. Exact successor resolution and the authoritative ordered tool-name gate are now passed.

Until then, the request intentionally has no trainer UUID. Its exact task-version bindings are
present, but the current paid-launch gate still rejects any preview that lacks authoritative tool
evidence. Filling only the trainer blank is not sufficient: every structured blocker in the
companion file must have immutable evidence.

The canary itself measures whether the requested 65k TP4/two-engine shape fits on 8xB300. CUDA OOM,
engine initialization failure, or invalid KV-cache sizing is an infrastructure/configuration
failure and cannot support an optimizer or model-incapability claim. The native loop may execute
multiple parsed calls from one assistant turn sequentially; this run records that frequency as a
diagnostic. It remains a blocker for a later Agent Runtime parity claim, not for this native-only
reward-acquisition measurement.
