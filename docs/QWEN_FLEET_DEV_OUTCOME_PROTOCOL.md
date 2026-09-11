# Qwen Fleet development outcome protocol

The split-A and split-B parent protocols freeze how SFT arms are selected on
genuinely held-out Fleet blackbox tasks. They are score-blind, non-launchable
templates: no checkpoint, model route, scored request, or paid job was created
while producing them.

## Frozen task bindings

Each split has 20 exact task families. The evaluator-ready task sets join the
representative study split to the sanitized exact-version binding audit observed
at `2026-09-11T21:04:03Z`. Every row binds the task/version, environment version,
starting-data version, runtime-seed digest, verifier/version/source digest, task
contract and hashes of the prompt/schema/environment-variable projections. They
contain no prompt text, trace, flag, answer, historical outcome, or score.

| Split | Task set | Parent protocol | Matched base control |
|---|---|---|---|
| A | [`qwen38-blackbox-fleet-dev-a-task-set-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-a-task-set-v1.json) | [`qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json) | [`qwen38-blackbox-fleet-dev-a-base-control-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-a-base-control-v1.json) |
| B | [`qwen38-blackbox-fleet-dev-b-task-set-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-b-task-set-v1.json) | [`qwen38-blackbox-fleet-dev-b-outcome-protocol-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-b-outcome-protocol-v1.json) | [`qwen38-blackbox-fleet-dev-b-base-control-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-b-base-control-v1.json) |

The task-set format compiles through `evals/fleet/evaluate.py`. Live preflight
and execution must still refetch the exact version and reject any runtime or
verifier drift. A historical clean binding is not permission to skip that gate.

## Scientific treatment

Both variants fix OpenCode 1.18.27, ordered `bash` and `submit_report` tools,
native compaction plus automatic continuation, a 262,144-token context, 20,000
tokens of compaction headroom, 32,768 maximum output tokens, 600 model requests,
and a 28,800-second task timeout. Sampling is temperature 0.6, top-p 0.95 and
pass@4 with attempt seeds 42–45. Concurrency is one per worker so the base and
post-SFT arms have the same scheduling treatment.

The sole HPO metric is full-task pass@1 on attempt 1 across all 20 valid task
outcomes. Pass@4, fractional verifier outcomes, validity, report submission and
cost/runtime statistics are descriptive. Training loss is a fitting diagnostic,
not a selection signal. Teacher-reference validation cross-entropy is forbidden:
it measures teacher-token imitation rather than blackbox exploitation.

Infrastructure-invalid, interrupted, output-limited and unknown attempts never
become model failures. Valid outcomes are never retried. The parent protocol
does not authorize replacement attempts; a symmetric reviewed amendment would
be required before any replacement execution.

## Matched base controls

Each base-control file binds the exact 20 task-version UUIDs and immutable
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` to its parent protocol. The
parent's OpenCode treatment, ordered tools, decoding, seeds, pass@4 and
concurrency are referenced by digest, so the post-SFT arm cannot change them.
The base outcome set may be reused across several post-SFT arms only when every
arm references the same parent and fresh live-pair evidence proves the same
serving engine, precision, quantization, parallelism, context, parsers, runtime
image and arguments, tokenizer/template, and agent/proxy images. Served IDs and
filesystem paths may differ as labels; the only scientific difference is the
weight manifest. Hosted and dedicated serving blocks are never pooled.

Both controls intentionally remain non-launchable. The repository has historical
Qwen serving observations, but no current immutable registration plus fresh
base/post live-parity receipt for this study. An existing endpoint may be bound
only after those exact current proofs exist; a catalog entry, model revision
label, Ready replica, or historical acceptance note is insufficient.

The prepared, zero-new-GPU path for producing those proofs is the
[matched-base certification workflow](QWEN_FLEET_DEV_BASE_CERTIFICATION.md).

Fleet grader outcomes are stored in the private evaluation ledger and accepted
receipts. Evaluation metrics and scores must never be sent to W&B, and
teacher-reference cross-entropy is forbidden. W&B remains limited to the
training-loss diagnostics defined by the training study.

## Checkpoint binding

The parent protocol deliberately stores `null` for every checkpoint/export,
tokenizer/template, staging, registration, live-parity, route and harness-image
identity. This is not a wildcard. An arm-specific immutable child must:

1. reference the exact parent protocol digest;
2. fill every null once from accepted receipts;
3. leave every frozen task, harness, sampling, metric and retry field unchanged;
4. pass exact base/post live parity before `eval prepare`.

Until then `launchable` is false. WebExploitBench remains a separately sealed
evaluation and cannot rank or retune these arms.

## Reproduction

The compiler consumes only a frozen study split and a private, sanitized binding
audit. It refuses to replace an existing output:

```sh
python -m evals.fleet.dev_outcome_protocol \
  --split configs/data/qwen-blackbox-study-split-a-v1.json \
  --bindings data/private/qwen-study-20260911/current-bindings.json \
  --variant a \
  --task-set configs/evaluation/qwen38-blackbox-fleet-dev-a-task-set-v1.json \
  --protocol configs/evaluation/qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json
```

Run the same command with the split-B paths and `--variant b`. Regeneration from
a newer live audit creates a new version; it must never overwrite these files or
silently change an in-flight study.
