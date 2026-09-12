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

| Split | Exact task set | Current parent protocol | Current matched base control |
|---|---|---|---|
| A | [`qwen38-blackbox-fleet-dev-a-task-set-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-a-task-set-v1.json) | [`qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json) | [`qwen38-blackbox-fleet-dev-a-base-control-v2.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-a-base-control-v2.json) |
| B | [`qwen38-blackbox-fleet-dev-b-task-set-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-b-task-set-v1.json) | [`qwen38-blackbox-fleet-dev-b-outcome-protocol-v2.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-b-outcome-protocol-v2.json) | [`qwen38-blackbox-fleet-dev-b-base-control-v2.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-b-base-control-v2.json) |

The v1 protocols and controls remain byte-for-byte historical records. They
selected on attempt 1 alone and are not edited or silently reinterpreted. The
v2 protocols are new immutable artifacts with new schemas and digests; they
reference the same v1 task-set bytes, split digest, development-selection
digest, final-test lock and 20 exact task versions.

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

The sole HPO metric is the matched candidate-minus-base mean full-task success
over all four fixed attempts. For each task, average the four binary outcomes;
then average the 20 task means with equal task weight. Each candidate outcome is
paired with the base outcome for the same exact task version and attempt seed,
so the primary comparison contains 80 matched pairs. All 80 pairs must be valid
for both arms. No available attempt may be dropped because its outcome is
unfavorable.

The 95% interval uses 10,000 seeded percentile-bootstrap resamples of the 20
tasks. Each resampled task retains all four within-task matched seed
differences. This treats task—not attempt—as the independent sampling unit.
Pass@4 remains a secondary descriptive metric and cannot rank or break ties
between arms. Fractional verifier outcomes, validity, report submission and
cost/runtime statistics are also descriptive. Training loss is a fitting
diagnostic, not a selection signal. Teacher-reference validation cross-entropy
is forbidden: it measures teacher-token imitation rather than blackbox
exploitation.

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
The v2 control additionally requires all 80 exact task-seed pairs and binds the
base vector used by every candidate comparison. A candidate estimate cannot be
ranked against another base outcome vector.
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

## Private estimator

`evals/fleet/dev_outcome_estimator.py` accepts only a self-digesting private
document containing the 80 exact task-seed rows, a valid binary outcome for
both arms, and exact evaluation-child and raw-result-manifest digests. Missing,
duplicate, extra or invalid pairs fail closed. It computes the v2 primary
estimate, task-clustered interval and secondary pass@4 without any network,
launch or W&B path. Real inputs and estimates belong under `data/private` or an
external private result root; the CLI prints only the output digest.

```sh
python -m evals.fleet.dev_outcome_estimator estimate \
  --protocol configs/evaluation/qwen38-blackbox-fleet-dev-a-outcome-protocol-v2.json \
  --task-set configs/evaluation/qwen38-blackbox-fleet-dev-a-task-set-v1.json \
  --outcomes data/private/<matched-outcomes>.json \
  --output data/private/<paired-estimate>.json
```

Run `select` with one or more private estimate paths after they share the same
protocol, task set, attempt seeds and exact base-outcome digest. Selection reads
only the corrected primary point estimate. It records that neither pass@4 nor
WebExploitBench was used. WebExploitBench results remain sealed until the Fleet
development decision is frozen and are never an HPO or tie-breaking signal.

## Reproduction

The compiler consumes only a frozen study split and a private, sanitized binding
audit. It refuses to replace an existing output. The command below documents
historical v1 reconstruction only; new work uses the checked-in v2 protocol
that preserves this exact task-set identity:

```sh
python -m evals.fleet.dev_outcome_protocol \
  --split configs/data/qwen-blackbox-study-split-a-v1.json \
  --bindings data/private/qwen-study-20260911/current-bindings.json \
  --variant a \
  --dev-protocol-version v1 \
  --task-set configs/evaluation/qwen38-blackbox-fleet-dev-a-task-set-v1.json \
  --protocol configs/evaluation/qwen38-blackbox-fleet-dev-a-outcome-protocol-v1.json
```

Run the same command with the split-B paths and `--variant b`. Regeneration from
a newer live audit creates a new version; it must never overwrite these files or
silently change an in-flight study.
