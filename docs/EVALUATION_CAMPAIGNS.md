# Resumable evaluation campaigns

`cyber-post-train eval campaign-*` is a small controller above the existing
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

## Commands

Prepare once:

```sh
cyber-post-train eval campaign-prepare exact-campaign.json \
  --output /restricted/eval-campaign
```

Generate previews and observe already-created work without permitting a create:

```sh
cyber-post-train eval campaign-step /restricted/eval-campaign
```

After reviewing the frozen plan and previews, permit bounded creates:

```sh
cyber-post-train eval campaign-step /restricted/eval-campaign --execute
```

Run `campaign-step` repeatedly. `campaign-status` reports score-blind aggregate
states and experiment keys. `campaign-record` can import one already-produced,
identity-bound receipt; it never creates work. Reconcile `*_launch_uncertain`
against the provider and import the recovered receipt—never replay the POST.
