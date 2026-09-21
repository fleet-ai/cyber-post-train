# First compact Qwen3.8 SkyRL arm

This is the smallest credible plan for the first real Qwen3.8 cyber RL arm
after the collector diagnostic is accepted. It is deliberately not a general
RL launcher.

## Why one node

Use one eight-GPU node first. The current SkyRL setup puts two rollout engines
on a node, each split across four GPUs. One prompt with eight sampled answers
keeps those eight GPUs busy. The policy and reference models are sharded across
the same eight GPUs.

Two nodes would double the number of four-GPU rollout engines and make the
training-model shards smaller. It would **not** make a single rollout engine
larger: each remains split across four GPUs. Therefore two nodes do not solve
a 262K-context rollout cache failure. Only move to two nodes if the first real
one-node run proves that the training-model shards, optimizer, or activations
are the limit. If an engine runs out of long-context cache, repair the engine
parallelism or context implementation instead.

The base weights total 55,563,006,776 bytes. A four-way rollout shard holds
about 13.9 GB of weight bytes per GPU. The policy and reference model each have
about 6.95 GB of weight bytes per GPU when sharded eight ways, before gradients,
optimizer state, temporary gathers, activations, and the long-context cache.
That makes one node plausible, but only the diagnostic and real canary can
prove that it fits.

## Fixed first-arm settings

- Qwen3.8-27B, exact locked revision.
- 262,144-token context; compaction starts at 163,840 tokens and retains an
  8,192-token summary. A turn may produce up to 32,768 tokens; the episode can
  run for up to 1,200 turns.
- One prompt group per update, eight sampled answers, eight updates, and a
  learning rate of `1e-6`.
- Checkpoint every update; keep the newest two. Evaluate the development set
  before training and after update eight, rather than spending a long task
  evaluation after every update.
- `c1`, one worker with eight GPUs, generic Jobs API only, and online W&B
  scalar logging. W&B receives scalar training and rollout counters only—never
  prompts, trajectories, sample tables, or private exception text.

Use the existing `training.skyrl_training` path with a freshly staged,
sealed `cyber_skyrl_data_v1` manifest. Its limits must exactly be the compact
values above, and its current broad split is 59 training tasks, 20 development
tasks, and 10 untouched final-test tasks.

## Exact gate order

Before preparing the standard SkyRL config, accept both of these small,
sanitized records:

- `RL_DIAGNOSTIC.json`: one collection completed with no leaf failure, zero
  optimizer steps/checkpoints/W&B events, and confirmed local engine cleanup.
- The external exact-UID observer result: the exact development RayJob and its
  children released after terminal success, used at most eight GPUs, and read
  no private logs.

Before a real arm:

1. Obtain a fresh generic Jobs API server preview and verify the root RayJob
   annotation is exactly `fleet.ai/failure-alerts: "off"`.
2. Run and accept one real one-step reward/update/checkpoint canary.
3. Recheck the output path, duplicate identity, and capacity; then prepare and
   submit the eight-update config through the generic Jobs API.

No direct Kubernetes manifest is valid for this RL path.
