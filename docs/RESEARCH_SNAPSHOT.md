# Research snapshot — 2026-09-24

These are **historical counts and records**, not a live cluster status page.
They were retained because they are useful starting facts for a fresh study.

## Fleet blackbox tasks

The last recorded production census contained **1,217 task versions**. Of
those, **75** had the exact task and run receipts required by the previous
qualification process. Another **33** candidates had their task lineage
identified, but **zero** of those 33 had the necessary runtime receipt in that
snapshot. The coverage categories overlap in the source data; do not add them
as if they were a partition or call all 1,217 tasks working.

The retained split of the 75 receipt-backed versions assigned **50 train, 17
development, and 8 final-test task versions**. It grouped related versions by
shared task lineage. This is a small historical split, not a representative
evaluation of every Fleet cyber task. Read the exact records before reusing it:

- [`fleet-blackbox-training-coverage-20260924-v1.json`](../configs/data/fleet-blackbox-training-coverage-20260924-v1.json)
- [`fleet-blackbox-receipt-proven-20260924-v1.json`](../configs/data/fleet-blackbox-receipt-proven-20260924-v1.json)
- [`fleet-blackbox-qa33-live-lineage-20260924-v1.json`](../configs/data/fleet-blackbox-qa33-live-lineage-20260924-v1.json)
- [`fleet-blackbox-lineage-safe-split-20260924-v2.json`](../configs/data/fleet-blackbox-lineage-safe-split-20260924-v2.json)

## Teacher SFT data and checkpoints

The Teacher3K source selection recorded **496 task keys, 1,176 task versions,
and 2,886 source sessions**, yielding about **57 million supervised tokens**
in the 96k-labelled data route. Three immutable corpus manifests are retained:
[`32k`](../configs/data/qwen38-teacher3k-32k-v1.manifest.json),
[`64k`](../configs/data/qwen38-teacher3k-64k-v1.manifest.json), and
[`96k`](../configs/data/qwen38-teacher3k-96k-v1.manifest.json). Their
`max_length` values are 32,768, 65,536, and 98,304 respectively; the separate
`context_tokens` fields are 8,192, 16,384, and 24,576. The names do not imply
every example fills its maximum context.

The earlier held-out selection was corrected from 25 candidate task families
to **20 clean families** after excluding five related families and six training
aliases. Even those 20 are held out only relative to the recorded Teacher3K
source selection; they are not a globally untouched test set.

The [checkpoint inventory](evidence/qwen38-sft-checkpoint-inventory-20260924.json)
lists the base control and six trained/exported candidates with their recorded
locations, digests, and readiness limits. A separate monitor snapshot at
**2026-09-24 17:06:53 UTC** observed the 96k-labelled run
`chris-q38-t3k96-b8-v3-e7d5c0f8` at optimizer step 400, with a native
checkpoint at
`/mnt/sfs/jobs/chris-q38-t3k96-b8-v3/checkpoints/global_step_400`.
The snapshot reported 27,029,867 supervised tokens processed, finite training
loss, zero Pod restarts, and a checkpoint receipt file SHA-256 of
`ac26e98c5acb6e8a27ce8fb8fd8ed1b38daf32aaabbd8597b0372e2071fa9f72`.
That is a dated observation, not a fresh verification of checkpoint access.

An earlier WebExploitBench comparison had a tool-interface defect and cannot
establish a reliable performance lift or regression. A clean, matched baseline
comparison is still needed before claiming either. The retained inventory
does not certify a completed RL learning run.
