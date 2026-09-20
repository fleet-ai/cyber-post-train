# Fresh75 batch-32 and batch-64 post-training handoff

Two more Fresh75 V2 arms have terminal, digest-bound training evidence and
released their training GPUs. This is training evidence only. Neither
checkpoint is yet sealed, exported, reloaded, served, or evaluated.

The exact machine-readable handoffs are:

- [`qwen38-fresh75-b32-step58-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-b32-step58-post-training-handoff-v1.json)
- [`qwen38-fresh75-b64-step30-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-b64-step30-post-training-handoff-v1.json)

Both are validated by
[`fresh75_post_training_handoff.py`](../../evals/webexploitbench/tensorlake/fresh75_post_training_handoff.py)
against the shared eight-arm queue. The validator derives the exact recipe,
checkpoint step, serving name, WebExploitBench canary pair, and Fleet
development campaign from the corresponding queue arm. This prevents an
accepted receipt from one arm being reused for another.

## Batch 32

- API run: `d66a9521-bac5-45eb-b4ff-d98792f22f13`
- RayJob UID: `6f8b96a2-96a5-4919-8c11-0ad17a4f37a1`
- Workload UID: `8e2425d8-1df2-4505-9c18-5515568d4da7`
- Pod UID: `e096c40b-c4ab-400c-a1da-2fcb8fdfa8f4`
- Output: `/mnt/sfs/jobs/chris-q38-f75-b32-v2`
- Terminal optimizer step: 58 of 58
- Supervised tokens: 2,072,122
- Training-complete file SHA-256:
  `9f61cadc7fcfdc58680e8e70d9d4707c78bfe91940cddc508677c925c80a9b35`
- Final checkpoint-saved receipt file SHA-256:
  `d42aa664fb99704bd21cd663107f7fa275b0209f9bd3476c8a2c8341bedcc2a4`
- Retained checkpoint steps: 20, 40, 58
- Training loss: `0.5639616847` at step 1 and `0.2245684117` at
  step 58

## Batch 64

- API run: `fe98cb6d-3087-4c1a-a7ad-4aa769d4704c`
- RayJob UID: `133f7c5d-d3f5-4d43-868a-cd18ffb7fd26`
- Workload UID: `8a3c8584-8888-4df0-b68f-4d61187c1d35`
- Pod UID: `a07da9dd-16c6-43f7-9e1d-0221511bb79a`
- Output: `/mnt/sfs/jobs/chris-q38-f75-b64-v2`
- Terminal optimizer step: 30 of 30
- Supervised tokens: 2,072,122
- Training-complete file SHA-256:
  `86337727ec105b0c116f1ba84828d85860c37d636ba007c0cde73b0a5fdf8ac9`
- Final checkpoint-saved receipt file SHA-256:
  `50d3d1d058d09f3137334146fd7a44157a3d65d472b56b02013ed74fe43fc0dd`
- Retained checkpoint steps: 20, 30
- Training loss: `0.5758959055` at step 1 and `0.2480294108` at
  step 30

For both arms, the exact Pod, RayCluster, and Workload were absent at the
terminal read-only check, so each training run held zero GPUs. The source-plan
files are exact copies of the immutable `.runtime/plan.json` bytes that produced
the accepted training receipts; their canonical digests match the queue's
runtime-plan digests.

## Remaining gates

Each final checkpoint still needs an independent full payload rehash, zero-step
BF16 export, zero-step reload with the source unchanged, immutable staging, and
a new paused `c1` serving route. Only after live base/candidate parity and a
fresh duplicate census can score-free WebExploitBench collection or the frozen
Fleet development evaluation begin. WebExploitBench scoring remains a separate,
resumable stage over immutable rollout bundles. The Fleet final-test split stays
closed until development evidence selects one checkpoint.

The checked-in handoffs authorize no external action. They make the exact next
inputs duplicate-safe while the global cluster failure budget is reconciled.
