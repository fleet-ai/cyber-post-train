# Resumable evaluation campaigns

`python -m evals.campaign` is a small controller above the existing
benchmark adapters. It does not know how to run WebExploitBench, Fleet held-out,
or another benchmark. It gives those adapters one immutable plan and one state
machine, so `baseline + checkpoints × targets × pass@4` can resume without
repeating accepted work.

## Frozen inputs

The reviewed config fixes:

- the exact matrix receipt (`matrix_sha256`) and budget receipt;
- every model's checkpoint ID, weight digest, serving-route receipt, and live
  parity receipt;
- one shared matched-treatment receipt across models. That receipt must bind
  tokenizer, chat template, serving runtime, context limit, numeric precision,
  and quantization;
- every harness identity receipt, which must bind its version, source, image,
  prompt, tool schema, and settings;
- each benchmark task-set and scoring-protocol digest;
- each target identity receipt, which must bind its exact task/environment and
  grader;
- each benchmark's own temperature, top-p, and four reviewed attempt seeds.

`attempt_seeds` must contain four different integers or four explicit `null`
values. `null` preserves a protocol that genuinely left the provider seed
unset; the controller never invents one. Attempts 1–4 are independent pass-1
runs whose aggregate is pass@4. There is intentionally no campaign-wide
sampling override: WEB, Fleet, and external benchmarks keep their own protocols.

An experiment key is the logical statistical cell: exact model treatment,
benchmark, target, attempt, sampling, and budgets. It deliberately excludes the
campaign name, adapter source, serving-route receipt, and live-parity receipt.
Changing code or repairing an otherwise identical route therefore cannot mint
a duplicate scientific cell. Those execution details still change the sealed
plan digest and every driver packet.

The deduplication registry has one code-owned path:
`~/.local/state/cyber-post-train/evaluation-registry-v1`. It is frozen into the
plan and cannot be selected on the command line or in campaign config.

## Safe state transitions

Each cell has two independent phases: rollout collection, then scoring. Each
phase moves through `preview → readiness → launch → observe`.

- Preview is always allowed and never creates remote work.
- Readiness is checked immediately before launch. Ordinary capacity or
  dependency delays produce `deferred_not_ready`; no dedup claim or launch
  intent exists yet, so a later scheduler tick may try again.
- A ready cell receives its global dedup claim and create-once launch intent
  immediately before the adapter's launch command.
- If launch becomes ambiguous after that intent, the cell remains
  `launch_uncertain`; it is never automatically replayed.
- Observation may report running many times. A terminal receipt is immutable.
- Scoring can begin only after it binds an accepted rollout terminal receipt.
- An infrastructure-invalid or failed target is preserved without stopping
  sibling cells.

The scheduler round-robins benchmarks and rotates model order across attempts.
`max_launches_per_step` bounds new launches. Optional serial canaries admit one
canary at a time and hold broad cells until every canary is accepted.

## Adapter contract

Each benchmark supplies rollout and score adapters as direct argument arrays—no
shell. Both implement `preview`, `ready`, `launch`, and `observe` and write a
self-digested `cyber_eval_campaign_driver_receipt_v1`. The controller passes
only immutable file paths:

- `{packet}` and `{receipt}` for every action;
- `{preview_receipt}` for readiness and launch;
- `{readiness_receipt}` for launch;
- `{launch_receipt}` for observation;
- `{terminal_receipt}` for score preview.

Fleet previews must include all rendered root `Job`/`RayJob` objects. Every root
must have `fleet.ai/failure-alerts: "off"`; a Jobs API request must also carry
`failureAlerts: false`.

TensorLake preview, readiness, and launch receipts must use an attempt-unique
remote name containing the experiment-key prefix. They must bind fresh shared
capacity and full provider-inventory receipts and prove that create/start
claims, the remote name, and output root were absent. Launch repeats these
read-only checks after readiness and before POST. Routine lack of capacity is a
readiness deferral, not an ambiguous launch.

Raw prompts, traces, answers, flags, logs, and scores never enter controller
state. Benchmark adapters remain responsible for their detailed scientific and
provider validation.

### Concrete WebExploitBench mapping

The follow-up adapter belongs in
`evals/webexploitbench/tensorlake/campaign_adapter.py`; none of the existing WEB
implementation should move into the controller. Its rollout actions map to the
current code as follows:

1. `preview` loads the existing sealed plan with
   `collection_launcher.load_plan`, checks that its model, target, OpenCode
   harness, snapshot, and output bindings equal the campaign packet, performs
   the complete read-only TensorLake inventory, and emits the required provider
   gates and attempt-unique sandbox name.
2. `ready` repeats the shared-capacity, full-inventory, local claim, remote-name,
   and output-root checks immediately before create. No mutation is permitted.
3. `launch` calls the existing `CollectionLauncher.create` and
   `start_collection` path once. Those methods retain their own permanent
   before-request claims. The adapter emits a create receipt only after both
   exact provider identities are known; uncertainty returns no receipt, leaving
   the campaign launch intent for reconciliation.
4. `observe` only reads the existing collection-supervisor/replica-pump state.
   It imports an accepted rollout-bundle digest or a sanitized infrastructure
   terminal digest; it never restarts collection or scoring.

The score actions use the existing `deferred_score` module over the accepted
rollout bundle: preview validates the sealed score plan without a judge call,
readiness checks its create-once score root, launch calls `run_score(...,
execute=True)` once, and observation imports `SCORE_COMPLETE.json` or a
sanitized failed-score receipt. This preserves the required collection/scoring
split and lets another judge repair scoring without replaying the rollout.

### Concrete Fleet final8 mapping

`evals/fleet/campaign_adapter.py` reuses the existing held-out launch packet and
its duplicate checks, server preview, create-once journal, exact-UID
observation, and score-blind terminal receipt. It does not render or submit a
second kind of Fleet job.

The prospective comparison is locked to the eight `final_test` task versions
in `configs/data/fleet-blackbox-current-study-split-20260914-v2.json`. The
adapter verifies all three identities before it can report readiness:

- file SHA-256
  `28a3dcaf31f14d724def9023d9435681772d5b8b3a8647cacc7f72ea8fc8adcb`;
- logical split SHA-256
  `05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c`;
- final-eight selection SHA-256
  `9623149c4a021bc13ed2cf94ca26e107b30c18816cf3c02d76b5020cab5066f4`.

The descriptive `dev17` evidence remains descriptive. It cannot be imported as
a final-eight result. The final eight are labelled
`historically_exposed_locked_confirmation_set` because their historical
exposure is part of the interpretation boundary, not something the adapter can
erase. The adapter binds the operational matrix receipt
`389a496130f9a4ce0379d9dd901ee6f5003b7cb1d9fae3bbf02100de202962e4`.
That receipt keeps the prospective selection order and applies only three
outcome-blind retention fallbacks: batch-16 step 200, 64K step 225, and
learning-rate-1e-6 step 500. The exact six arms are base, step 1000, those
three fallbacks, and 96K step 300. It also binds the original prospective
matrix digest
`d72f52ce7035c5b2f48eb75de9513141a98f87d5fd7e4caf552ed625e967593a`
and the exposure audit at
`docs/evidence/qwen38-fleet-final8-exposure-audit-20260923.json` (file SHA-256
`17fc69469673018c84d9d11d92f33989eb229730ba9da1d77fc7fffff2953504`,
receipt SHA-256
`b61dd5bcba1320b11a40abb1836b6b32cf434fe1cc641da07622cc9d0a34dd5f`).
That audit records that all eight exact versions have prior quality-certification
execution and two have accepted pre-split model-comparison execution. No
historical cell is reused, and this comparison must never be described as
pristine or untouched.

Pass@4 is four independent pass-1 source jobs per model, using seeds 46, 47,
48, and 49 with retries disabled. For the six-arm comparison this is exactly 24
Fleet jobs, not 192 jobs and not one pass-4 job per model. Each source job runs
all eight tasks. The controller still keeps 192 independent logical records
(`6 models × 8 tasks × 4 attempts`): one elected record creates the source job,
and its seven siblings bind to the same immutable Job UID. This gives efficient
execution without weakening per-task deduplication or partial-failure
isolation.

Every server preview is checked through the existing Fleet launcher and must
show the root Job with `fleet.ai/failure-alerts: "off"`, Kueue queue
`training-lq`, queue priority `q1`, and Pod priority `c1` before creation is
possible. Collection starts only after that preview and the existing duplicate
census pass. Results are read only after the exact source Job and ConfigMap UIDs
are terminally bound. Each task is then classified independently from its
score-blind database row, and only a zero-retry row can be accepted, so one
broken task does not discard seven accepted siblings. Because one source Job
atomically runs all eight tasks, these campaign cells are all non-canary, so the
generic controller's serial-canary mode has no effect.

Fleet already performs its authoritative task grading during the rollout. The
campaign score phase is therefore local: it seals the accepted native grading
receipt and never calls a second judge, replays a rollout, or reads a score
value. A restarted controller reuses immutable shared preview, launch, and
terminal records. If a create response is ambiguous, the existing create-once
journal and the campaign's `launch_uncertain` state require reconciliation
rather than another POST.

## Commands

Prepare once:

```sh
python -m evals.campaign prepare exact-campaign.json \
  --output /restricted/eval-campaign
```

Generate previews and observe already-created work without permitting a create:

```sh
python -m evals.campaign step /restricted/eval-campaign
```

After reviewing the frozen plan and previews, permit bounded creates:

```sh
python -m evals.campaign step /restricted/eval-campaign --execute
```

Run `step` repeatedly. `status` reports score-blind aggregate states and
experiment keys. `record` can import one already-produced,
identity-bound receipt; it never creates work. Reconcile `*_launch_uncertain`
against the provider and import the recovered receipt—never replay the POST.
