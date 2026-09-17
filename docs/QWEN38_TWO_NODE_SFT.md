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
