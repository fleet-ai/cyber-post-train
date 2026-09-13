# Qwen + Miles RL prior-art audit (2026-09-12)

## Decision

Fleet has strong evidence that the **Qwen3.8-27B + Miles execution and checkpoint
path works**, but not a durable, terminally proven Qwen3.8 cyber-RL run that we
can copy wholesale. We should reuse the proven model/topology/memory and
rollout-batch choices, while retaining this repository's stricter Jobs API,
reward, checkpoint, reload, release, and benchmark-isolation gates.

The audit cutoff is 2026-09-13T04:46Z. No prior run proves the complete chain
`verifier executions -> policy update -> sealed native checkpoint -> all-rank
zero-update reload -> immutable HF export -> downstream eval -> controller
success -> external GPU release`.

This audit intentionally excludes prompts, trajectories, flags, task contents,
sealed scores, credentials, and private logs.

## Evidence tiers

### 1. Terminally proven operational RL: Qwen3.6-27B

The strongest self-contained terminal evidence is the one-node/eight-GPU Miles
canary recorded in
[`docs/evidence/qwen36-study/2026-08-31-miles-canary03-terminal-v1.json`](./evidence/qwen36-study/2026-08-31-miles-canary03-terminal-v1.json).
It binds the exact Qwen3.6 revision, trainer image, source commit, RayJob and Pod
UIDs, eight-rank model/optimizer/scheduler/RNG checkpoint inventory, W&B run,
controller success, and RayCluster cleanup.

What it proves:

- Qwen full-parameter RL can execute at one node × eight GPUs.
- Eight grouped samples can reach the optimizer path.
- An eight-rank native checkpoint can be written with optimizer, scheduler,
  metadata, and RNG state.
- The controller can terminate successfully and release the RayCluster.

What it does **not** prove:

- useful reward variation or a nonzero policy update;
- preserved per-episode verifier IDs after cleanup;
- semantic all-rank checkpoint reload; or
- Qwen3.8 model compatibility.

It is an operational capability gate, not evidence of learning or lift.

### 2. Strong Qwen3.8 operational evidence: Neeraj's `rl-transfer-v003`

The closest Qwen3.8 precedent is
[`fleet-ai/dataminer_v2/experiments/rl-transfer-v003`](https://github.com/fleet-ai/dataminer_v2/tree/a3aff4ae4eadb96c49f5a8b374329469f9aa5f53/experiments/rl-transfer-v003).
The materially relevant launcher last changed in
[`b0ae23f97f67544d6795a19b1d6f14fb1c0ce0ef`](https://github.com/fleet-ai/dataminer_v2/commit/b0ae23f97f67544d6795a19b1d6f14fb1c0ce0ef),
and the pinned direct-Job image came from
[`687efc5c2c396d9898995790f18006da80ad3799`](https://github.com/fleet-ai/dataminer_v2/commit/687efc5c2c396d9898995790f18006da80ad3799).

Proven/useful recipe details:

- exact model family: Qwen3.8-27B;
- one eight-B300 node;
- TP4, PP1, colocated CP2 at the eight-GPU/96k-token shape;
- sequence parallelism, full recomputation, dynamic batching, and an 8,192
  token/GPU ceiling;
- SGLang static memory fraction 0.8;
- Adam, constant learning rate, weight decay 0.1, betas 0.9/0.98, CPU optimizer
  offload, and overlapped optimizer transfer;
- GRPO with per-token loss, KL coefficient 0.001, clipping 0.2/0.28, and one
  accumulated optimizer update per rollout wave;
- group-level filtering that removes any group containing an aborted sample;
- same-root checkpoint save/resume and stable W&B grouping across generations.

Retained conversion jobs consumed checkpoints through later iterations, which
is strong evidence that the native files were usable. However, the main run was
operator-stopped and lacks a self-digested terminal/release receipt, an exact
immutable model revision, a Jobs API request digest, a terminal W&B identity,
source-unchanged proof, and a semantic all-rank zero-update reload. Treat it as
strong operational/checkpoint-consumption evidence—not as a terminally proven
production RL result.

### 3. Canonical public recipe: `miles-fleet`

[`fleet-ai/miles-fleet` at `2799fe386320c156334bf763ad4d7ca0f85dca4e`](https://github.com/fleet-ai/miles-fleet/tree/2799fe386320c156334bf763ad4d7ca0f85dca4e)
documents the same broad Qwen3.8 one-node recipe: TP4, sequence parallelism,
8,192 tokens/GPU, recomputation, CPU Adam/offload, one rollout engine/GPU, and
SGLang memory fraction 0.8. This is recipe evidence only; it does not identify a
terminal controller, exact checkpoint, or release receipt.

### 4. Deniz's FTI Qwen3.8 work

The audited preparation source is `fleet-ai/theseus` at
`956c17d796fcbb10d3ca773f4fba1b6a972e1622`. The visible 256k run uses four
eight-B300 nodes, TP8/PP1/CP4, rollout batch 16, eight samples, group-validity
filtering, and periodic native saves. It is useful scale and memory prior art,
but it is not an appropriate first cyber qualification shape: the running image
cannot be mapped to indexed public source, model preparation is not pinned to an
immutable revision, and no sanitized terminal/reload/export/release chain is
available. We therefore start with the peer-proven one-node TP4/PP1/CP2 shape.

### 5. Miles native checkpoint to HF export

Fleet has successfully run a native-to-HF converter from `fleet-ai/dataminer_v2`
at `cf5131074dda38a39320d72aa1525803d02d0605`; the converter script SHA-256 is
`e2f45be7b5c9e56e13ddfe6ffe5554387ea56b9c9fcb285a4fbe09cda42ecb1d`.
For example, `neeraj-offeval-convert-i35-preemptible` (Kubernetes UID
`83b6df7e-97ff-4cd7-a48a-589c4f0f5712`) completed successfully. This proves the
conversion mechanism can execute. It does not prove immutable output bytes:
that job used mutable shared paths and retained no HF manifest or digest.

Miles also supports `--save-hf`, but the audited peer runs did not enable it.
For this campaign, native training remains unchanged and a separately pinned,
create-once export gate must hash the complete HF result and prove zero-update
reload before evaluation consumes it.

## What we should reuse

1. Qwen3.8 model arguments and the proven one-node TP4 memory topology.
2. One optimizer update per complete prompt-group wave.
3. Group-level abort filtering before advantage calculation.
4. Per-token loss, conservative KL/clipping defaults, and explicit LR sweeps.
5. Frequent same-root native checkpoints and stable W&B experiment grouping.
6. The task/runtime template only after its exact version is pinned and
   independently validated for Fleet blackbox tasks.

## What we should not copy

- direct Kubernetes batch submission instead of the create-once Jobs API;
- mutable shared output roots or automatic restart/preemption loops;
- blanket multi-terabyte memory requests without measured justification;
- missing priority/requeue declarations;
- aggregate reward claims without preserved authoritative verifier IDs;
- checkpoint existence without source immutability and semantic all-rank reload;
- training or checkpoint selection based on WebExploitBench.

## Required gates for this campaign

Before production Qwen3.8 RL, this repository must still prove, in order:

1. a fresh corrected one-node/eight-GPU dev4 canary, explicitly using per-token
   loss with GRPO standard-deviation normalization disabled, is admitted on one
   whole B300 node and retains its exact source, image, request, and no-requeue
   bindings; the retired unallocated dev3 request is not promotion evidence;
2. a real dev reward canary with authoritative verifier IDs, valid nontruncated
   episodes, nonzero reward and within-group variation;
3. exactly one independently demonstrated optimizer update plus a value-level
   policy-tensor delta from the exact base model;
4. a sealed native step-1 checkpoint after external release;
5. a value-sensitive zero-update semantic reload of model, optimizer, scheduler,
   and RNG state on all eight ranks with unchanged source;
6. a create-once HF export with an exhaustive manifest and zero-update model and
   tokenizer reload;
7. an exact c1/q1/no-requeue production Jobs API preview and one POST; and
8. checkpoint selection from frozen Fleet-dev criteria only, followed by sealed
   Fleet holdout and WebExploitBench evaluation.

No prior Fleet run removes any of these scientific or operational gates.
