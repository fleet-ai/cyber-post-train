# Fresh75 one-epoch checkpoint 115 post-training handoff

The exact Fresh75 V2 one-epoch arm finished successfully at optimizer step
115. This document records what that proves, what it does not prove, and the
next safe operation.

The machine-readable handoff is
[`qwen38-fresh75-e1-step115-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-e1-step115-post-training-handoff-v1.json).
It is checked by
[`fresh75_post_training_handoff.py`](../../evals/webexploitbench/tensorlake/fresh75_post_training_handoff.py).

## Accepted training evidence

- Training API run: `9c8fd91c-87a7-4b3a-92dd-63b97033e4e4`
- RayJob: `chris-q38-f75-e1-v2-9c8fd91c`
- RayJob UID: `fac778b3-ac20-41d7-89a0-7cd470f45428`
- Workload UID: `53eea815-332b-43d3-803f-b7075d994a15`
- Pod UID: `c186a25a-5399-4455-9e8f-1eb1ded50cb3`
- Output: `/mnt/sfs/jobs/chris-q38-f75-e1-v2`
- Terminal optimizer step: 115 of 115
- Supervised tokens: 1,036,061
- Training-complete file SHA-256:
  `00ce20cf0f96ea10bb30f06c7a2ab8fd534530270e2c89b0751a1a754378a881`
- Metrics file SHA-256:
  `c4ec7aa4dc872b3be37786aa9eeb41b522784d8140fbcb21fb7ee8c479822df6`
- Final checkpoint-saved receipt file SHA-256:
  `fd92b96e9bb8f843eca6c3cde9f155d6a99c96faf3f89ce52dc602c99ff7a319`

The metrics file has one row for every optimizer step. The recorded training
loss is finite at both endpoints: `0.6419628859` at step 1 and `0.3676480651`
at step 115. This is evidence that optimization ran and training loss fell on
the training corpus. It is not held-out performance evidence.

At the read-only terminal check, the exact Pod, RayCluster, and Workload were
absent, so the arm held zero GPUs. Checkpoint directories 80, 100, and 115 were
still present. Directory presence is not checkpoint acceptance.

## Why evaluation remains closed

The checkpoint-saved receipt says that the trainer wrote checkpoint 115. It
does not contain a complete file manifest or a byte-for-byte rehash of the
roughly 303 GB payload. The handoff therefore remains closed until a separate
qualification proves all of the following:

1. Every checkpoint payload file is present and rehashes to a sealed manifest.
2. A separate export performs zero optimizer updates and emits BF16 model
   weights with a complete manifest.
3. A separate reload performs zero optimizer updates, produces finite output,
   and leaves the source checkpoint unchanged.
4. The exact export is staged once at a new immutable path.
5. A new `c1` route is created paused at zero replicas, with a new UID.
6. Admitting that route keeps the project at or below eight active GPU nodes.
7. Live base and checkpoint routes pass the same serving checks.
8. A fresh duplicate census passes before a matched, score-free OpenCode
   WebExploitBench task-0 canary and Fleet development evaluation can begin.

Scoring remains a separate, resumable operation over immutable rollout
bundles. The Fleet final-test split remains closed until one checkpoint is
selected using development evidence only.

## Safe next action

Run one create-once checkpoint-115 seal/export/reload/stage qualification. It
must not perform an optimizer update. The checked-in handoff does not authorize
that external action; it only binds the exact accepted input and the gates the
result must satisfy.

Validate the inert handoff locally with:

```sh
uv run python -m evals.webexploitbench.tensorlake.fresh75_post_training_handoff \
  --handoff configs/qualification/qwen38-fresh75-e1-step115-post-training-handoff-v1.json
```
