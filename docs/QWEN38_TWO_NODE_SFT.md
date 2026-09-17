# Qwen3.8 two-node SFT qualification

## What is being tested

The eight-node 262k teacher-SFT canary succeeded, but that does not prove eight
nodes are necessary. The first two-node qualification therefore changes one
scientific input only: `nodes: 8` becomes `nodes: 2`. It keeps the exact Qwen3.8
revision, 262,144-token window, 64-example global batch, one example per GPU,
data, target masks, learning rate, seed, checkpointing and runtime.

With 16 GPUs and no sequence parallelism, a 64-example global batch takes four
data-parallel microbatch rounds per optimizer step: `64 / (2 × 8 × 1) = 4`.
The canary must stop after one finite optimizer update and save a reloadable
checkpoint. A later full run is a separate decision.

## What transfers from Neeraj's recipe

Neeraj's Qwen3.8 Miles/Megatron RL recipe demonstrates that long-context memory
can be bounded using activation recomputation, small per-GPU token work units
and optimizer offload. It used TP4 × CP2 on one eight-GPU node and deliberately
capped rollout context at 98,304 tokens. A roughly 131k trajectory had already
exhausted that one-node shape.

Those observations motivate testing fewer nodes, but the topology cannot be
copied literally into this SkyRL SFT runtime. Qwen3.8 mixes attention with
recurrent Gated DeltaNet layers, and SkyRL's available sequence-splitting path
does not propagate the recurrent state between sequence partitions. Enabling it
would change the model computation. The SFT path therefore keeps exact sequence
semantics and relies on its already-qualified chunked projections, chunked
Gated DeltaNet work and activation checkpointing, plus FSDP across 16 ranks.

## Bound canary

- name: `chris-q38-t3k262-2n-can-v1`
- output: `/mnt/sfs/jobs/chris-q38-t3k262-2n-can-v1`
- parent plan SHA-256: `3b9e81acb301327c40007a40ab13c3bf5c164152655fb49f96a8db52eb117690`
- plan SHA-256: `5d4e59b26db80d975075abac53a745c064aa5c7348ed2701d75ceb129d35e63a`
- request SHA-256: `cb74560ea0532a959134068af6f2f9a477d45ec2e2598768546e50e540e4f18b`
- resources: two whole eight-B300 nodes, c1 priority

The zero-GPU dev-cluster preflight succeeded with exit zero and zero restarts.
It checked the pinned trainer sources, all model/data digests, native config and
loader behavior, tokenization, and target accounting: 112 windows, 27 tasks and
3,022,959 supervised tokens. Its receipt SHA-256 is
`46648ee8ea7805fd88fe58234ec317c174623a53a50f695a72fb4ed4f61a6024`.

This preflight rules out source, packaging, binding and data defects. Only a
real two-node GPU canary can determine peak memory and distributed runtime fit.

## V1 result and bounded successor

V1 reached `device_ready`, loaded and actively used all 16 GPUs, and spent about
44 minutes in the first accumulated training step with zero restarts. It peaked
near 271 GiB per GPU. The sanitized failure receipt then recorded a CUDA
out-of-memory error in backward recomputation of the Qwen MLP. No progress or
checkpoint receipt was committed, so V1 is failed capacity evidence—not a
successful optimizer update.

The failed allocation was 2.50 GiB with 2.00 GiB physically free while PyTorch
held 9.10 GiB reserved but unused. The exact runtime recommended expandable
allocator segments for this fragmentation signature. V2 therefore changes only
the allocator to `expandable_segments:True`; all model, data, objective,
context, batch, learning-rate and checkpoint settings remain identical. If V2
cannot commit a finite step and checkpoint, two-node 262k SFT remains blocked
pending a separately reviewed memory change such as optimizer offload.

V1 run evidence is bound to API run
`chris-q38-t3k262-2n-can-v1-7cc622e0`, RayJob UID
`acfc81f3-5bb9-43f0-9be1-ee01f7c45a77`, Workload UID
`6404bae8-1c81-4b09-9f60-6c7a7b17db19`, failure receipt SHA-256
`8b9f1d21fd6d5c852d016ebc3cf07fd3933bf72b794cfa9ca6f4c3160d61385f`,
and failure-stage receipt SHA-256
`03a62045ce6faff4560c2f676269dc90c3c6af381c7288f1d299a74f72594a5b`.
Its Workload is terminally `Finished/Failed`; its RayCluster and Pods are gone.

V2 is prepared under `chris-q38-t3k262-2n-can-v2`, plan SHA-256
`6241e3cb1d0eae75fb37339911d0464661f6920ea736dc8b293c483af8f3e7ba`
and request SHA-256
`145fdff44eb3330eb931198beee73fc5e55c0690e29d947261a192617cddd836`.

## V2 terminal result

V2 was submitted exactly once as API run
`chris-q38-t3k262-2n-can-v2-07c9e6c2`. It was bound to RayJob UID
`4ab5bb14-c93d-4c29-97aa-fa3b32d2a66b`, Workload UID
`d58969a2-45f5-4784-9556-ba105a74ad8a`, RayCluster UID
`d40d468c-d2c7-42e6-a386-71346d09013e`, head Pod UID
`26ced027-9a23-4e83-98b4-faa9ae1bc62f`, and worker Pod UID
`cd79a9fe-aa4f-4f73-bbfc-fa186db1efa2`. Both Pods used the exact image, became
Ready with zero restarts, and the runtime environment contained exactly
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

The run reached `device_ready` and performed sustained long-context compute on
all 16 GPUs. GPU memory crossed several activation peaks, reaching about 272
GiB per GPU, then falling and rising again across the four required gradient
accumulation rounds. It survived beyond V1's failure duration, demonstrating
that expandable segments fixed the first fragmentation failure. Nevertheless,
the Ray job terminally failed at 2026-09-17T06:25:41Z with a nested
out-of-memory error. The W&B terminal summary reported optimizer counter 1,
but no durable `PROGRESS.json`, checkpoint receipt or planned-pause receipt was
committed. The counter therefore cannot be treated as an accepted optimizer
update and V2 produced no reusable checkpoint.

The Workload is terminally `Finished/Failed`; the RayCluster and both Pods are
gone, so all 16 GPUs were released. A read-only CPU observer did not finish
pulling the pinned image within its five-minute deadline and was deleted. No
further observer or training successor was submitted after the operator asked
that nothing new be launched.

After the terminal evidence was captured, the operator requested full cleanup.
The exact terminal V1 and V2 RayJobs were deleted through the supported Jobs
API (HTTP 204 for each). Their owner-bound per-run Secrets and terminal
Workloads were garbage-collected. A final inventory found zero matching Pods,
Jobs, RayJobs, RayClusters, Workloads, Services, ConfigMaps or Secrets on both
the production and development clusters. The Jobs API history rows and durable
SFS output roots remain as provenance; neither reserves cluster resources.

The two-node, exact-262k, full-parameter shape is therefore **not qualified**.
The next experiment must be a separately reviewed memory intervention, not an
unchanged retry. The highest-information candidates are persistent optimizer
state offload (the closest transferable part of Neeraj's Miles recipe) and a
smaller exact context ceiling. Either change needs its own one-step canary and
cannot inherit V1/V2 acceptance.
