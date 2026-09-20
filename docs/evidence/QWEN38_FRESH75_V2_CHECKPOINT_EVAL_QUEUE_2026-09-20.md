# Fresh75 V2 checkpoint evaluation queue

Eight Fresh75 V2 supervised-training runs were active when this queue was
first sealed. Each run uses the same exact Qwen3.8-27B base, the same 916-example
Fresh75 corpus, one eight-GPU node, and `c1` priority. The deliberate
differences are batch size, learning rate, and epoch count.

The machine-readable source of truth is
[`qwen38-fresh75-v2-checkpoint-eval-queue-v1.json`](../../configs/evaluation/qwen38-fresh75-v2-checkpoint-eval-queue-v1.json).
It binds every Training API run, RayJob, Workload, Pod, output directory,
runtime plan digest, and recipe. The validator is
[`fresh75_v2_eval_queue.py`](../../evals/webexploitbench/tensorlake/fresh75_v2_eval_queue.py).

The queue now also carries immutable terminal observations as individual arms
finish. Seven arms are terminal and released: the one-epoch arm at step 115,
batch 16 at step 116, batch 32 at step 58, batch 64 at step 30, and the
two-epoch batch-8 reference, lower-learning-rate, and higher-learning-rate arms
at step 230. Each training-complete receipt, final
checkpoint-saved receipt, metrics file, and exact resource release is
digest-bound in the queue. This establishes training completion only. It does
not establish a complete checkpoint payload, an exported model, a reloadable
model, a serving route, or capability lift. The fail-closed handoffs are:

- [`qwen38-fresh75-e1-step115-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-e1-step115-post-training-handoff-v1.json)
- [`qwen38-fresh75-b32-step58-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-b32-step58-post-training-handoff-v1.json)
- [`qwen38-fresh75-b64-step30-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-b64-step30-post-training-handoff-v1.json)
- [`qwen38-fresh75-base-step230-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-base-step230-post-training-handoff-v1.json)
- [`qwen38-fresh75-lr3-step230-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-lr3-step230-post-training-handoff-v1.json)
- [`qwen38-fresh75-b16-step116-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-b16-step116-post-training-handoff-v1.json)
- [`qwen38-fresh75-lr1-step230-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-lr1-step230-post-training-handoff-v1.json)

## What the queue does

For every training arm and every checkpoint step that its immutable recipe can
emit, it can reserve unique names for:

- one new checkpoint-serving route;
- one matched OpenCode WebExploitBench task-0 canary;
- four independent 15-task, pass-at-one WebExploitBench collection replicas
  for the base model and checkpoint; and
- one 17-task Fleet development evaluation.

The four WebExploitBench replicas form pass@4 only after all four have been
collected for both models. Rollout collection makes no judge calls. Scoring is
a later, resumable step over the immutable rollout bundles, so a scoring outage
does not require a model rollout to be repeated.

The eight Fleet final-test tasks are intentionally absent from each arm's
evaluation packet. Every arm may run on the 17 development tasks because those
tasks were excluded from training and are the declared checkpoint-selection
set. The eight final-test tasks may run once, after a single checkpoint has
been selected without using the final-test outcomes.

Checkpoint-directory presence is not acceptance. An interval checkpoint can
enter this queue only with its own full checkpoint seal, payload rehash,
zero-update export, and zero-update reload receipt. A caller selects an interval
with `--optimizer-step`; a step the recipe could not have emitted is rejected.
Because a full-weight checkpoint is about 303 GB and each run retains only the
latest three, the operational priority is scientifically spaced states rather
than every adjacent save: the final state, an early state, and a middle state
when that arm can change the learning-curve decision. The mechanism still
supports every independently accepted step under identical controls.

The preservation and evaluation order is:

1. every accepted final checkpoint, with batch-64 and batch-32 first because
   they finish earliest and test the widest batch boundary;
2. one early and one middle checkpoint for batch-32, learning-rate `3e-6`, the
   `1e-5` reference arm, and batch-16, which currently provide the most useful
   learning-rate/batch-size comparison;
3. one-epoch final, then a middle state, to separate training duration from
   optimizer settings;
4. sparse checkpoints from the low-learning-rate and four-epoch boundaries;
   and
5. any additional interval only when its training metrics or Fleet-development
   result can change a decision already represented above.

This ordering is a storage and decision policy, not an acceptance claim. A
surviving directory that has not passed the receipt gates remains ineligible.

## Why the queue cannot launch a model prematurely

The checked-in queue is inert. It cannot call Fleet, Kubernetes, TensorLake, or
a model endpoint. An arm stays `blocked_pending_training_qualification` until a
self-digesting qualification receipt proves all of the following:

1. The exact Training API run and Kubernetes objects succeeded.
2. Its Workload, Pod, and all eight GPUs were released.
3. The final native checkpoint was sealed and fully rehashed.
4. A separate export performed zero optimizer updates and produced BF16 model
   weights with a complete payload manifest.
5. A separate reload performed zero optimizer updates, produced finite model
   output, and left the source checkpoint unchanged.
6. The exported model was staged at a new create-once inference path.

Only then can the tool render an arm packet whose next state is
`qualified_waiting_for_paused_route_registration`. The route must still be
created paused at zero replicas with a new UID. Live base-versus-checkpoint
serving checks and a fresh duplicate census remain required before rollout
collection.

Qualification schema v2 does not accept a list of unattached digest strings.
The private qualification must point to the exact checkpoint-saved,
checkpoint-manifest, export, GPU-reload, and staging receipts. The validator
reopens each regular file, checks its full file SHA-256 and self-digest, and
then checks the step, checkpoint path, source plan, zero-update export,
reload, and create-once staging chain. For a final checkpoint, the saved
checkpoint receipt and its file SHA-256 must also be exactly the receipt
already recorded in the queue's terminal training observation. A copied or
freshly invented acceptance summary cannot open the paused-route gate.

## Fleet development evaluation protocol is complete

The standalone protocol in
[`qwen38-fresh75-fleet-dev17-protocol-v1.json`](../../configs/evaluation/qwen38-fresh75-fleet-dev17-protocol-v1.json)
seals the complete checkpoint-independent Fleet evaluation identity. Its file
SHA-256 is
`f72c34e3e5bd5ca76959717e4091c4fb0f10fa3a2639e4dbd2eb10f2d9718469`
and its self-digest is
`sha256:5db333575ac97cac30c8ccaeab1983e63937f774dda90ed9eda3b81c245ec21e`.
It binds all 17 exact task, environment and data version tuples and the accepted
GET-only verifier census. It also binds the exact upstream-base model and live
route identity, BF16 SGLang image and serving profile, OpenCode 1.18.27
treatment, ordered tools, immutable agent and proxy images, sampling, budgets,
pass count, retry policy, and evaluator-file digests.

The packet validator reopens and hashes every referenced file. It rejects an
incomplete task tuple, a task outside the development split, a changed runtime
file, an unpinned image, a placeholder serving source, any treatment or sampling
change, and any attempt to open the final-test set. A qualified packet compiles
an exact 34-session plan: 17 upstream-base sessions and 17 candidate sessions
under the same treatment.

Only checkpoint-specific facts remain unavailable before qualification:

1. the candidate payload-manifest digest used as its exact model revision;
2. its create-once staged model path;
3. the new serving-route UID returned by create-once registration; and
4. a fresh readback proving that route still matches the sealed image, model,
   server, tokenizer, chat-template and tool-call contract.

The first two come from the accepted export/staging qualification chain. The
last two cannot be guessed or checked in before the route exists, so the packet
names their exact receipt/readback sources and fails closed until they are
present. They are operational identity checks, not open scientific choices.

The eight-task final-test selection remains closed. It may be rendered only
once, after development evidence has selected one checkpoint.

## Resource and duplicate safety

Each checkpoint route requires one whole eight-GPU node. The queue permits one
checkpoint route at a time and requires that the route reuse capacity released
by its own training run. Admitting it must leave the whole project at no more
than eight active GPU nodes. The historical Fresh75 step-230 route and its
temporary successor are explicitly forbidden for these V2 arms.

Every arm receives its own route, campaign, pair, sandbox-prefix, and Fleet
evaluation names. The validator checks all names for collisions. A fresh live
inventory is still mandatory immediately before each create because a checked-in
plan cannot prove that another human or agent did not launch the same work later.

## Validate or render a packet

Validation is read-only:

```sh
uv run python -m evals.webexploitbench.tensorlake.fresh75_v2_eval_queue \
  --queue configs/evaluation/qwen38-fresh75-v2-checkpoint-eval-queue-v1.json
```

To render one blocked packet before training finishes:

```sh
uv run python -m evals.webexploitbench.tensorlake.fresh75_v2_eval_queue \
  --queue configs/evaluation/qwen38-fresh75-v2-checkpoint-eval-queue-v1.json \
  --arm b64-lr1e5-e2 \
  --output /private/tmp/b64-eval-packet.json
```

To render an interval-checkpoint packet, add for example
`--optimizer-step 20`. Its route and campaign names include that step so they
cannot collide with the final-checkpoint packet.

Adding `--qualification <accepted-receipt.json>` does not launch anything. It
only reopens the exact receipt, checks every terminal and zero-update gate, and
seals the next create-once handoff.

## Interpretation boundary

An accepted task-0 canary proves that transport, the OpenCode harness, and
evidence collection work for that exact pair. It is not a capability result.
Likewise, collecting the base arm earlier than the checkpoint arm is allowed
when every model, task, harness, image, protocol, and sampling digest is frozen.
The work becomes a matched capability comparison only after both complete
collections are accepted and the same separate scoring contract has graded
both immutable bundles.
