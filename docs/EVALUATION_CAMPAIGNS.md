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
3. `launch` writes its local receipt, starts one detached worker, and waits for
   its signed `STARTED` receipt. The worker does not touch TensorLake until the
   campaign controller has durably recorded that exact launch receipt. A local
   spawn failure therefore becomes an ordinary observed failure rather than an
   uncertain remote create. The worker repeats the provider gates under
   the shared create lock, then calls the existing `CollectionLauncher.create`
   and `start_collection` path once. Their permanent before-request claims are
   unchanged. An ambiguous sandbox create is attached by exact name and spec;
   an ambiguous collection-process response is attached by the exact claimed
   runner command. Neither POST is replayed. If provider ownership cannot be
   proven, the exact create claim and capacity reservation remain held for
   reconciliation; they are never cleared from a single inventory miss. The
   worker then delegates
   preservation, release, and export to the existing replica-pump supervisor
   and snapshot exporter.
4. `observe` only reads the worker's signed state and independently reopens the
   existing supervisor, release, export, and rollout-bundle evidence. It imports
   an accepted rollout-bundle digest or a sanitized infrastructure terminal
   digest; it never restarts collection or scoring.

The score actions use the existing `deferred_score` module over the accepted
rollout bundle: preview validates the sealed score plan without a judge call,
readiness checks its create-once score root, launch starts one detached worker
that waits for the campaign controller's durable launch receipt before calling
`run_score(..., execute=True)` once, and observation independently
reopens the score bundle or a sanitized failed-score receipt. This preserves
the required collection/scoring split and lets another judge repair scoring
without replaying the rollout.

`campaign_adapter.py` is the concrete bridge. Each statistical attempt binds
one existing WEB `pass_k=1` collection plan; the campaign's four independently
named attempts form pass@4. WEB sampling is fixed to temperature 1.0, top-p
0.95, and an explicitly unset provider seed. A binding is sealed beside each
campaign `packet.json` as `web-binding.json`, so every benchmark driver can use
the same direct-argument commands. Generate the exact rollout and score driver
blocks with `campaign_adapter driver-spec --phase rollout|score`; this avoids
manual command drift. `seal-binding` validates the exact
launch plan, task, model artifact, serving-route and live-parity receipts,
OpenCode harness, sampling, runtime-budget digest, deterministic sandbox name,
state paths, and deferred-score draft without a provider call. Collection
serving evidence also has one reviewed signed mapping to the plan's complete
public student identity; a base arm must supply the canonical standalone route
proof plus a fresh paired live-parity receipt, while a checkpoint arm must use
the exact registration and parity files already validated by its plan.
Collection
launch delegates to `CollectionLauncher`; terminal preservation delegates to
the replica-pump supervisor; scoring delegates to `deferred_score`. Immediately
before a provider create, launch requires the live owner selected by the latest
bound capacity marker, so an older packet source cannot reject a valid
generation-2 owner. An ambiguous sandbox create is reconciled by its permanent
claim and exact provider identity rather than repeated. Rollout acceptance
waits for both the supervisor's exact
released terminal chain and the existing snapshot-export path's verified local
rollout bundle. The adapter does not invent a second export implementation.

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

For a foreground fire-and-forget run, use the same explicit execution gate:

```sh
python -m evals.campaign run /restricted/eval-campaign --execute
```

`run` repeats score-blind steps, polling when a round makes no immediate
progress or any driver reports an error. Three consecutive matching driver
errors for the same target hold that target for repair instead of looping
forever. Per-cell infrastructure failures remain terminal evidence and do not
stop siblings. It exits when every cell is terminal, or when the remaining cells
are explicitly held by an uncertain launch or a failed serial canary; it never
replays an uncertain create.

When serial canaries are enabled, only one canary cell can be remotely active at
a time. The controller observes and records that cell before another canary may
launch; no broad cell launches until the canary barrier is resolved.

`status` reports score-blind aggregate states and experiment keys. `record` can
import one already-produced, identity-bound receipt; it never creates work.
Reconcile `*_launch_uncertain` against the provider and import the recovered
receipt—never replay the POST.
