# Qwen Fleet final outcome protocol

This protocol freezes the study's common 10-family Fleet final test as a
score-blind, non-launchable base-versus-post-SFT comparison. Creating it did not
create an endpoint, scored request, API write, or paid job.

## Exact final-test binding

[`qwen38-blackbox-fleet-final-task-set-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-final-task-set-v1.json)
is the evaluator-ready projection of the immutable
[`qwen-blackbox-study-final-test-v1.json`](../configs/data/qwen-blackbox-study-final-test-v1.json)
lock. The same lock is present in split A and split B. Each of its 10 unique task
families is bound to one exact task version, environment version, starting-data
version, runtime-seed digest, verifier/version/source digest, cyber contract and
task projection hashes from the sanitized binding audit observed at
`2026-09-11T21:04:03Z`.

The task set contains no prompt text, trace, flag, answer, historical outcome, or
score. It is not eligible for training, preferences, rewards, retrieval,
hyperparameter selection, or checkpoint selection. Execution must still refetch
each exact version and reject any task, environment, starting-data, or verifier
drift before a scored attempt.

## Matched comparison

[`qwen38-blackbox-fleet-final-outcome-protocol-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-final-outcome-protocol-v1.json)
fixes the following treatment for both the base and post-SFT arms:

- Qwen3.8-27B base lineage at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- OpenCode 1.18.27 with ordered `bash` and `submit_report`, native compaction and
  automatic continuation;
- a 262,144-token context, 20,000-token compaction headroom, 32,768 maximum
  output tokens, 600 model requests and a 28,800-second task timeout;
- temperature 0.6, top-p 0.95, and pass@4 with paired seeds 42–45;
- the same exact task/version and attempt seed as the pairing unit, with one
  concurrent rollout per worker and one symmetric validity/retry policy.

Only the post-SFT weight manifest may differ. Tokenizer, chat template,
precision, quantization, server image and arguments, harness images, tools,
budgets, tasks, verifiers and retry handling must match. Hosted and dedicated
serving blocks cannot be silently pooled.

The primary result after unsealing is the paired change in full-task pass@1 on
attempt 1 across the 10 families. Pass@4, fractional outcomes, validity,
submission and cost/runtime measures are descriptive. Infrastructure-invalid,
interrupted, output-limited and unknown attempts never become capability zeros,
and valid outcomes are never retried.

## Sealing and per-checkpoint children

The parent has explicit `null` placeholders for both arms' checkpoint/export,
weights, tokenizer/template, staging, registration, live-parity, served-model,
route and harness-image identities. These are required bindings, not wildcards.
For every frozen training checkpoint to be evaluated, create one immutable child
that:

1. references this exact parent protocol digest;
2. binds every base and post-SFT placeholder once from accepted receipts;
3. proves exact base/post live serving parity before evaluation preparation;
4. preserves all frozen tasks, seeds, budgets and validity rules;
5. writes to create-once private result roots.

The child remains non-launchable until all bindings pass. Final-test outcomes,
including failure analyses, stay sealed during hyperparameter search. They may be
unsealed only after the dev-outcome selection rule, chosen recipe and chosen
checkpoint identities are frozen in an immutable study decision receipt. They
must never be used to retrain, retune, choose a checkpoint, develop prompts, or
create exploit guidance.

## Reproduction

The compiler consumes only the frozen lock and the private sanitized binding
audit, and refuses to replace existing outputs:

```sh
uv run python -m evals.fleet.dev_outcome_protocol \
  --final-lock configs/data/qwen-blackbox-study-final-test-v1.json \
  --bindings data/private/qwen-study-20260911/current-bindings.json \
  --task-set configs/evaluation/qwen38-blackbox-fleet-final-task-set-v1.json \
  --protocol configs/evaluation/qwen38-blackbox-fleet-final-outcome-protocol-v1.json
```

A newer exact binding audit requires a new version; never overwrite this frozen
protocol or silently change an in-flight study.
