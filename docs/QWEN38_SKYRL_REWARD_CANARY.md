# Qwen3.8 SkyRL reward-acquisition canary

Status: **prepared, not submitted, not yet production-qualified**.

This is the smallest real RL run that preserves the qualified Qwen3.8 SkyRL
shape: one dev-cluster node, eight GPUs, two TP4 rollout engines, eight training
episodes on one train task, one native optimizer update, pre/post dev evaluation, W&B, and a
step-1 checkpoint. It is an operational/scientific gate, not a capability
comparison and not authority for a production run.

## Exact inputs

- Run config:
  [`qwen38-rl-reward-canary-dev-v1.json`](../configs/qualification/qwen38-rl-reward-canary-dev-v1.json)
- CPU/GET-only data config:
  [`qwen38-rl-reward-canary-data-dev-v1.json`](../configs/qualification/qwen38-rl-reward-canary-data-dev-v1.json)
- Frozen task set and split:
  [`qwen38-rl-reward-canary-task-set-v1.json`](../configs/data/qwen38-rl-reward-canary-task-set-v1.json)
  and
  [`qwen38-rl-reward-canary-split-v1.json`](../configs/data/qwen38-rl-reward-canary-split-v1.json)
- Exact ordered tool schema: the existing reviewed `bash`, `submit_report`
  catalog, whose canonical digest is bound in the task set.
- Model: exact `Qwen/Qwen3.8-27B` lock, revision, weight manifest and staged SFS
  root already used by the dev8 engine diagnostic.
- Image: the corrected immutable SkyRL image bound in
  `training/skyrl_training.py`. A different image is a different experiment.

The training row is one exact metadata-only successor recorded in the
sanitized Qwen3.6 reward-canary evidence. Its source version had a historical
nonzero outcome. The successor operation changed only the ordered task-tool
metadata and preserved the prompt, environment, verifier and data binding. This
is task-selection evidence, not a claim that Qwen3.8 will receive nonzero
reward. One frozen task from another family is dev-only and is never optimized.
No external benchmark row is included.

Task preparation fetches only those exact versions. It privately binds the
authoritative task, environment, runtime-seed and verifier identities into each
row. Neither this document nor terminal public evidence may contain prompts,
traces, flags, answers, task responses, credentials or individual scores.

## Why this is the minimum useful shape

The native profile requires grouped GRPO samples and exact divisibility across
eight GPUs. One prompt group times eight samples gives eight training episodes
on the exact receipt-proven task and produces exactly one prompt batch and one
optimizer update. It also matches the `groups: 1`, `samples_per_prompt: 8`
batching shape required by the current 59-row production split, rather than
qualifying a second batching layout. One sample on the dev row runs before
training and one after it, for ten real rollouts total.

The frozen recipe is:

- one eight-GPU worker on the dev cluster;
- Jobs API priority `c1` (the q1/high queue, value 10000), no requeue;
- one step, one group, eight samples per prompt, learning rate `1e-6`;
- 98,304-token context, 81,920 response-token budget, 4,096 tokens per turn,
  80 turns, 2,400-second episode limit and 330-second tool limit;
- evaluation before training and after step 1;
- checkpoint at step 1, keeping at most two checkpoints;
- scalar-only W&B run `thefleet/cyber-post-train/chris-q38-rlreward-dev1`, plus
  the durable local scalar stream;
- create-once output and episode identities, single-attempt generation and
  Fleet API calls, and confirmed instance cleanup before a sample is accepted.

The config now carries `cluster.target: dev`. Preparation seals this as
`execution.cluster_target`, and the CLI rejects both preview and submission to
another cluster. This is an enforcement gate, not a runbook convention.

## Pre-submit gates

Do not submit until all of these are true:

1. The exact dev8 diagnostic has a committed or otherwise independently
   preserved digest-valid terminal receipt proving both engines started, the
   exact corrected image ran, no task/reward/optimizer work occurred, and every
   GPU was released. The current commit contains the dev8 config but not that
   terminal receipt, so repository-only evidence is incomplete.
2. Revalidate the immutable reviewed tool catalog at the path named by the data
   config. Run `cyber-post-train rl-data` in the pinned image as user `1000:100`
   on CPU. Its new output destination must not already exist. Accept only a digest-valid
   `cyber_skyrl_data_v1` manifest with exactly one train row and one dev row.
3. Prepare a fresh immutable launch directory with `cyber-post-train rl`, run
   CPU `preflight`, and run `preview --cluster dev`. The rendered request must
   remain one worker times eight GPUs, priority `c1`,
   `requeueIfPreempted=false`, and Secret-backed `fleet-api` plus `wandb-api`.
4. Prove the run name, output root, W&B ID and request digest have no existing
   or ambiguous submission across the dev Jobs API, SFS and W&B. A timeout is
   ambiguous; reconcile it and never POST a copy.
5. Confirm one topology-compatible whole eight-GPU dev node is available.
   Engine-only success on fragmented `2 x 4` allocations does not prove the
   one-worker training allocation is schedulable.

This repository change performs none of those external checks and makes no
submission.

## Terminal acceptance

`NATIVE_TRAINING_COMPLETE.json` is necessary but insufficient. Accept the dev
canary only after an independent, digest-bound terminal audit proves:

- all eight train episodes and both dev episodes have valid `ACCEPTED.json`
  receipts, exact task/environment bindings, an authoritative nonempty
  `verifier_execution_id`, a valid reward, and confirmed environment release;
- at least one training reward is nonzero and the eight-sample training
  group has reward variance. Uniform valid rewards are genuine outcomes but do
  not establish a usable GRPO learning signal;
- exactly one optimizer update occurred and independently verified model or
  optimizer state changed. A step counter or checkpoint filename alone is not
  enough;
- the step-1 policy, optimizer, scheduler, sampler and RNG checkpoint is
  complete and sealed create-once, with the source unchanged;
- the exact W&B run exists and contains finite scalar rollout, reward, policy,
  optimizer and step telemetry without sample tables or private text;
- the native reload gate below passes; and
- the Job, Workload, RayCluster, Pods and all eight GPUs are released.

A zero reward, zero within-group variance, missing verifier ID, uncertain
instance cleanup, failed W&B identity, missing independent update proof, or
unreloadable checkpoint keeps production closed. It is not repaired by silently
resampling or reclassifying an infrastructure failure as reward zero.

## Native reload gap

The inert
[`qwen38-rl-reward-canary-reload-dev-v1.template.json`](../configs/qualification/qwen38-rl-reward-canary-reload-dev-v1.template.json)
records the exact next gate. It is deliberately `launchable: false`.

The repository currently has no `cyber_skyrl_training_v1` checkpoint-seal
schema and no RL CLI path that binds such a seal to a zero-update all-rank
restore. `training/recovery.py` is reached through the SFT compiler/runtime; it
cannot truthfully validate this RL plan. Before production promotion, implement
and CPU-test a native RL reload command that:

1. binds the exact source plan, checkpoint inventory and file digest;
2. restores model, optimizer, scheduler and RNG on all eight ranks plus the
   exact sampler cursor at global step 1;
3. executes a model forward but zero rollouts, optimizer updates and checkpoint
   writes;
4. uses a fresh dev-only run/output/W&B identity, `c1`, one eight-GPU worker and
   no requeue; and
5. emits a digest-valid terminal receipt and proves GPU release while leaving
   the source checkpoint byte-identical.

Only after that receipt and the reward/update proofs pass may this canary be an
input to a separately reviewed production configuration.
