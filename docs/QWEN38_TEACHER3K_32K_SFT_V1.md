# Qwen3.8 broad-teacher 32K SFT

This treatment uses the existing exact-version Fleet blackbox corpus without
weakening its identity or outcome contracts. It changes only the physical
window size so the first broad-data run fits the proven one-node full-weight
topology more safely than an unqualified 262K sequence.

## Data contract

- Parent corpus: 2,886 successful stronger-teacher sessions, 1,176 exact task
  versions, 496 task keys, and 176,654 visible assistant responses.
- Unique supervised signal: 57,384,881 target-token occurrences after the
  parent's exact duplicate removal. One epoch therefore exposes 57.38M unique
  supervised tokens; this is not token inflation through repeated epochs.
- Held-out protection: all 25 frozen dev/final task families are excluded
  across every task version. WebExploitBench tasks are absent.
- Validation: task outcomes only. There is no held-out teacher-token
  cross-entropy file.
- Privacy: private transcripts remain on SFS. The repository contains only
  aggregate counts, immutable digests, and public task keys.

The successor groups whole assistant targets into sequences of at most 32,768
tokens and copies at most 8,192 preceding tokens as loss-masked context. Every
source assistant target must occur exactly once in the new loss masks. The
largest independently measured target is 4,700 tokens, so this rule never needs
to split or truncate a target.

## First treatment and controls

The first paid action is a one-step canary at batch 8 and learning rate 3e-6.
Only a finite update, checkpoint, source-stability proof, and clean resource
release admit the matching one-epoch full run. Both use one 8-GPU node, 32K
maximum sequences, microbatch one per GPU, c1 priority, and W&B.

Two already-prepared follow-ups change one important optimization choice at a
time while holding the corpus, seed, topology, context, and epoch count fixed:

1. learning rate 1e-6 at batch 8;
2. global batch 16 at learning rate 3e-6 (two accumulation steps).

All three arms were admitted as create-once c1 jobs on September 20, 2026, and
each completed at least one finite optimizer update with zero container
restarts. The exact Kubernetes identities, immutable bindings, first-step
evidence, motivations, and W&B links are recorded in
[`qwen38-teacher3k-fullweight-sweep-launch-20260920.json`](evidence/qwen38-teacher3k-fullweight-sweep-launch-20260920.json).
That receipt is nonterminal operational evidence, not a checkpoint-acceptance or
capability claim.

The original 57.38M-token 262K corpus remains sealed for later 64K/96K
curriculum treatments. Those are separate topology/length studies and are not
evidence that exact 262K full-weight training fits on one or two nodes.

## Bounded checkpoint retention

An observed Qwen3.8 full-weight training checkpoint occupies about 303 GiB.
The current batch-8 successors save every 100 optimizer steps and the batch-16
successor saves every 50 steps. Each keeps the latest two checkpoints. Over the
complete epoch this produces 19 saves per 32K treatment, including the forced
final save, while bounding steady live storage near 606 GiB. A third checkpoint
can coexist briefly before pruning, so the safe transient allowance is at least
about 909 GiB.

Historical step rates project a first save after roughly three hours. The much
larger 53,300-second `checkpoint_recovery_horizon_seconds` value is a
conservative watchdog ceiling, not the target save cadence. It prevents a slow
but still progressing run from being killed before its next save; the separate
20-minute no-progress check remains unchanged.

The 64K context control saves every 15 steps, keeps the latest two, and has a
17,460-second recovery ceiling. Its higher save frequency is intentional because
each 64K optimizer step takes materially longer. Exact SFT and RL cadence,
storage cost, telemetry boundaries, and the distinction between a saved
checkpoint and a proven restart are recorded in
[`CHECKPOINT_POLICY.md`](CHECKPOINT_POLICY.md).

The project-level goal of at least 100 checkpoint/evaluation pairs should be
met across distinct data, learning-rate, batch, and method treatments rather
than by storing 100 nearly adjacent 303 GB optimizer snapshots from one run.
