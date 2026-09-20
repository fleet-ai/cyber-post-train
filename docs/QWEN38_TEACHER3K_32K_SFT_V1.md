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

The original 57.38M-token 262K corpus remains sealed for later 64K/96K
curriculum treatments. Those are separate topology/length studies and are not
evidence that exact 262K full-weight training fits on one or two nodes.

## Bounded checkpoint retention

An observed Qwen3.8 full-weight training checkpoint occupies 303 GB. Retaining
100 native checkpoints from one run would consume about 30 TB and mostly
compare adjacent points on one trajectory. The batch-8 arms instead checkpoint
every 450 steps and keep five checkpoints. At 1,837 total steps this preserves
steps 450, 900, 1,350, 1,800, and the forced final step 1,837: four roughly
quarter-epoch intermediates plus the final model, bounded near 1.5 TB. The
batch-16 arm uses interval 225 for the same fractional spacing. This retention
choice does not change optimization; it exists so later Fleet and
WebExploitBench evaluations can compare useful, scientifically spaced models.

The project-level goal of at least 100 checkpoint/evaluation pairs should be
met across distinct data, learning-rate, batch, and method treatments rather
than by storing 100 nearly adjacent 303 GB optimizer snapshots from one run.
