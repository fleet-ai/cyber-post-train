# Fresh75 batch-16 and low-learning-rate post-training handoff

Two more Fresh75 V2 supervised-training arms completed successfully and
released their training allocations on 2026-09-20. This record binds their
public terminal receipts to inert post-training handoffs. It does not claim
that either checkpoint is fully sealed, exported, reloadable, served, or
better than the base model.

## Batch 16

- RayJob `chris-q38-f75-b16-v2-9c2c6c68`, UID
  `9627c4e7-a9b7-4fca-b545-0d842ad2adc5`, completed successfully at
  `2026-09-20T15:51:49Z`.
- Training completed all 116 planned optimizer steps over 2,072,122 supervised
  tokens. The recorded training loss was finite at steps 1 and 116 and changed
  from `0.5809828043` to `0.2775996029`.
- The selected final checkpoint is
  `/mnt/sfs/jobs/chris-q38-f75-b16-v2/checkpoints/global_step_116`; retained
  checkpoint steps are 80, 100, and 116.
- The exact training Pod and RayCluster were absent on the independent
  read-only release check. The named Workload remained only as a finished
  historical object, with no active GPU allocation.

The machine-readable handoff is
[`qwen38-fresh75-b16-step116-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-b16-step116-post-training-handoff-v1.json).

## Learning rate 1e-6

- RayJob `chris-q38-f75-lr1-v2-bb3c7aae`, UID
  `e0d7e0bf-fb27-4bdd-aa06-26ad5725a64e`, completed successfully at
  `2026-09-20T16:04:36Z`.
- Training completed all 230 planned optimizer steps over 2,072,122 supervised
  tokens. The recorded training loss was finite at steps 1 and 230 and changed
  from `0.6419628859` to `0.3512610197`.
- The selected final checkpoint is
  `/mnt/sfs/jobs/chris-q38-f75-lr1-v2/checkpoints/global_step_230`; retained
  checkpoint steps are 200, 220, and 230.
- The exact training Pod and RayCluster were absent on the independent
  read-only release check. The named Workload remained only as a finished
  historical object, with no active GPU allocation.

The machine-readable handoff is
[`qwen38-fresh75-lr1-step230-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-lr1-step230-post-training-handoff-v1.json).

## Remaining gates

Each selected checkpoint still needs a complete payload seal and rehash, a
zero-update BF16 export, an independent zero-update reload with the source
unchanged, a create-once paused serving route with a new UID, live serving
parity against the base model, a fresh duplicate census, and then matched
score-free OpenCode WebExploitBench and frozen Fleet-development evaluation.
Scoring must consume the preserved rollout bundles separately.
