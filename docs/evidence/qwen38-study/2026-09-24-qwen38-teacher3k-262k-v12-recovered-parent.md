# Recovered Qwen3.8 262K parent evidence

This is a sanitized identity record for the historical eight-node mechanics
canary used as the parent of the held four-node hypothesis. It does **not**
declare the checkpoint scientifically accepted or authorize a new GPU run.

## Immutable identities

- Run: `chris-q38-t3k262-can-v12`
- Jobs API / RayJob name: `chris-q38-t3k262-can-v12-a421f0a3`
- Jobs API run ID: `a421f0a3-5ed2-48cb-99ee-f830fa395cb7`
- RayJob UID: `36f915ca-d882-46b3-9d8b-5e4e2bfddae7`
- Output: `/mnt/sfs/jobs/chris-q38-t3k262-can-v12`
- Plan: `/mnt/sfs/jobs/chris-q38-t3k262-can-v12/.runtime/plan.json`
- Plan SHA-256: `3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690`
- Runtime source: `git:45c04d709f855e20d931456233b85f427558525f:training/sft_runtime.py`
- Runtime SHA-256: `b6b81c9876ddf8379a0837a0aaf5f0426acf8ccc4388d79bd93bad94c59e38d3`
- Runtime bundle SHA-256: `e1eb27e77aa42088feac8913de75414910fe919e1f818dfbe5a086e4a210b5b8`
- Image: `661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4`
- Model revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Model-weight manifest SHA-256: `06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352`
- Dataset: `/mnt/sfs/jobs/chris-q38-study-corpora-v1/teacher3k-262k-v1/capacity-v5/train.parquet`
- Dataset SHA-256: `2359c54e5c5cd756761a0f6e8c250ec8b32b87c8f84f0888252ddac932cfc5ec`
- Corpus-manifest SHA-256: `c5c7d82127d524593ed7e1cf5375fb3457e36c0d92376531808b6eeaacda10f5`
- Split-manifest SHA-256: `05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c`

## Exact parent recipe

The train file contains 112 rows. The one-epoch parent used 8 nodes × 8 GPUs,
global batch 64, microbatch 1 per GPU, and therefore two optimizer steps. It
used 262,144 tokens, learning rate `3e-6`, seed `20260916`, sequence parallel
size 1, Gated DeltaNet chunks of 512 tokens, LM-head/MLP/RMSNorm chunks of
1,024 tokens, layer-checkpoint group size 1, no CPU parameter or optimizer
offload, a checkpoint every step, retention of the latest two checkpoints, and
an intentional pause after optimizer step 1.

## What execution proved

The RayJob reached `SUCCEEDED`. The pause receipt is available at
`/mnt/sfs/jobs/chris-q38-t3k262-can-v12/TRAINING_PAUSED.json`; its self-digest is
`0b00877a5c40ec25fc1da622177ede0c4c35e3e0febe7e07563416b3ff42e919` and
file digest is `399b18c57e03c9c6c6429a4d64bf4c31da24ed88e050eff1facf54e5886d1c5a`.
The step-1 checkpoint receipt is at
`/mnt/sfs/jobs/chris-q38-t3k262-can-v12/checkpoint_receipts/step-000001.json`;
its self-digest is
`3f0cbcd958121499859560f521a8b2f262ffc5d20c6be3cdc472c68f65706758`
and file digest is
`834b1c11c57c64806b71b583e9c9b916f73bdc8fca868a60f0be0bd0e866f6af`.

This proves one finite optimizer update and the native simple-checkpoint path.
It does not prove a sealed checkpoint file inventory, GPU checkpoint reload,
final model export, or scientific acceptance. The historical four-node v9 arm
used global batch 32 and layer-checkpoint group size 2 and OOMed; it is not
positive evidence for the new group-size-1 hypothesis.
