# Fresh75 reference and lower-learning-rate post-training handoff

The two-epoch batch-8 reference arm and its `3e-6` learning-rate treatment
both completed all 230 optimizer steps and released their training resources.
This is accepted training evidence, not a model-quality result. Neither final
checkpoint is yet fully rehashed, exported, reloaded, served, or evaluated.

The exact machine-readable handoffs are:

- [`qwen38-fresh75-base-step230-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-base-step230-post-training-handoff-v1.json)
- [`qwen38-fresh75-lr3-step230-post-training-handoff-v1.json`](../../configs/qualification/qwen38-fresh75-lr3-step230-post-training-handoff-v1.json)

## Reference arm

- API run: `68c57437-f64d-41b4-a04e-e0ca3dd246a0`
- RayJob UID: `943f6be5-0f9e-465e-b50a-71df07e9b2ca`
- Workload UID: `c6c618fb-23bb-4f28-9eff-0e4055412452`
- Pod UID: `54c39aa7-4b29-405b-bfa2-fada6e84f3f6`
- Final checkpoint: `/mnt/sfs/jobs/chris-q38-f75-base-v2/checkpoints/global_step_230`
- Retained checkpoints: 200, 220, and 230
- Training-complete file SHA-256: `4dcdeae98de7c084f221d366674f84f750320d56a5570857361fd9799a78beb0`
- Final checkpoint receipt file SHA-256: `b3dc481ed81d513961f1e4717837aed14e0cbcc5c33269d34595de01ca3b62b0`
- Training loss: about `0.6420` at step 1 and `0.2577` at step 230

## Lower-learning-rate arm

- API run: `ec1a75e8-3aa8-4db2-8264-70290c9c308e`
- RayJob UID: `b8167ef5-086b-4126-aac4-aa26a1b231ed`
- Workload UID: `e23abc98-0164-47fc-a4d2-8e558a098c38`
- Pod UID: `bc219d6a-89f5-4ca4-872d-4838a597532d`
- Final checkpoint: `/mnt/sfs/jobs/chris-q38-f75-lr3-v2/checkpoints/global_step_230`
- Retained checkpoints: 200, 220, and 230
- Training-complete file SHA-256: `7d3ed17c72f56c7801889bec7e7b73df8c442c7c921ad427befa5dc2f22e9c2f`
- Final checkpoint receipt file SHA-256: `9f8a834d8ac97e0df0c20d2a8501bdc2c50895a58bbc34ff2de237fad0a795f5`
- Training loss: about `0.6420` at step 1 and `0.2565` at step 230

For both arms, the terminal read-only check found no Pod, RayCluster, Workload,
or GPU allocation. The checked-in source plans reproduce the exact canonical
plan digests in the accepted training receipts.

## What remains

Each checkpoint still needs a full payload seal, a zero-update BF16 export, a
zero-update reload that proves the source is unchanged, immutable staging, and
a newly created paused `c1` serving route. Matched OpenCode WebExploitBench and
frozen Fleet-development evaluations remain closed until the serving and live
parity checks pass. Rollout collection will remain separate from scoring.

These handoffs authorize no external action while the global cluster
failure-budget hold is active.
