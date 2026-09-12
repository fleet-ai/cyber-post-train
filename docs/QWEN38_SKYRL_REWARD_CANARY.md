# Qwen3.8 SkyRL reward-acquisition canary

Status: **blocked before data preparation; not submitted or production-qualified**.

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
- Exact-version and horizon evidence:
  [`qwen38-rl-reward-canary-exact-version-evidence-v1.json`](../configs/data/qwen38-rl-reward-canary-exact-version-evidence-v1.json)
- Exact ordered tool schema: the existing reviewed `bash`, `submit_report`
  catalog, whose canonical digest is bound in the task set.
- Model: exact `Qwen/Qwen3.8-27B` lock, revision, weight manifest and staged SFS
  root shared with the historical dev8 and current dev9 engine diagnostics.
- Image: the worker-RPC-sanitized immutable SkyRL image and its exact CPU
  qualification are pinned in `training/skyrl_training.py`. The prior dev8
  relay image is privacy-disqualified for every renamed run. A different image
  is a different experiment and requires a new reviewed diagnostic closure.

The intended training row is a successor of one historically informative exact
version. It is not currently eligible. The authoritative per-version GET showed
that the successor lost the source's exact starting-data binding, so the two
versions are not a metadata-tools-only pair. The committed v1 summaries also do
not retain the canonical sanitized response bytes and GET acquisition journal
required by the v2 gate. A pair of newly self-digested summaries cannot repair
that absence. `training_data_eligible` is therefore false, and both data
validation and launch compilation fail closed. Enabling this canary requires a
new immutable successor plus independently retained source/successor sanitized
GET bodies and journals whose file/self digests are bound by the exact-version
evidence; historical reward is only a selection prior.

Dev8 is terminal historical evidence and is not a runtime prerequisite for the
new image. The fresh [dev9 diagnostic](QWEN38_SKYRL_ENGINE_DIAGNOSTIC_DEV9.md)
must carry the request/preview `1000:100` binding, enforce it inside the
allocated process, and cross-bind each Pod's observed UID/GID, effective
security context, imageID and UID ownership through terminal release. Dev8 is
never modified, retroactively upgraded, or replayed.

The frozen dev row remains from another family and is never optimized. No
WebExploitBench, final-test, external-benchmark row, task content or outcome is
included in either split or in public evidence.

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
- effective Pod and container identity `runAsUser=1000`, `runAsGroup=100`, and
  `runAsNonRoot=true`, rechecked by the allocated GPU process itself;
- Jobs API priority `c1` (the q1/high queue, value 10000), no requeue;
- one step, one group, eight samples per prompt, learning rate `1e-6`;
- 98,304-token context, 81,920 response-token budget, 4,096 tokens per turn,
  a 600-turn ceiling, 2,400-second episode limit and 330-second tool limit;
- evaluation before training and after step 1;
- checkpoint at step 1, keeping at most two checkpoints;
- scalar-only W&B run `thefleet/cyber-post-train/chris-q38-rlreward-dev1`, plus
  the durable local scalar stream;
- create-once output and episode identities, single-attempt generation and
  Fleet API calls, and confirmed instance cleanup before a sample is accepted.

The config now carries `cluster.target: dev`. Preparation seals this as
`execution.cluster_target`, and the CLI rejects both preview and submission to
another cluster. This is an enforcement gate, not a runbook convention.

The 600-turn ceiling is bound to the exact selected task version's prior
reviewed native-harness contract and to the digest of the aggregate harness
audit, which records source sessions that exceeded 80 turns. Raising the turn
ceiling does not raise the response-token, per-turn-token, tool-time or
2,400-second episode bounds; it only prevents the turn counter from truncating
an otherwise in-budget episode.

## Pre-submit gates

Do not submit until all of these are true:

1. The exact dev9 diagnostic has a digest-valid sanitized terminal receipt with
   schema `cyber_skyrl_engine_diagnostic_terminal_v1`. It must reopen and
   cross-bind the exact dev9 config, plan, request, CPU preflight and two-row
   create-once submission journal; the dev API and fresh Kubernetes object UIDs
   and owner chain; both Pods' exact image IDs, `1000:100` runtime identity and
   non-root security context; zero task/reward/optimizer/checkpoint/W&B work;
   clean owned-engine shutdown; terminal success; and absence of the RayCluster
   and GPU Pods with zero active GPUs. No dev9 API or Kubernetes identity is
   predeclared before it exists. Until the currently-null dev9 terminal path
   and digest are replaced by that independently audited receipt, reward-canary
   compilation fails before a launch request can be prepared.
2. Revalidate the immutable reviewed tool catalog at the path named by the data
   config. Run `cyber-post-train rl-data` in the pinned image as user `1000:100`
   on CPU. The canary-specific validator in `training/rl_reward_canary.py`
   binds the exact task-set, split, retained authoritative version observations
   and 600-turn horizon. The currently missing v2 sanitized GET bodies and
   acquisition journals are a hard blocker; repo-authored hashes or booleans do
   not substitute. The generic
   `training/rl_data.py` stays byte-identical for historical frozen receipts.
   Its new output destination must not already exist. Accept only a digest-valid
   `cyber_skyrl_data_v1` manifest with exactly one train row and one dev row.
3. Prepare a fresh immutable launch directory with `cyber-post-train rl`, run
   CPU `preflight`, and run `preview --cluster dev`. The rendered request must
   remain one worker times eight GPUs, priority `c1`, effective runtime user
   `1000:100` with `runAsNonRoot=true`,
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
  `verifier_execution_id`, a valid reward, and confirmed environment release.
  Each acceptance must also bind the exact create intent, returned instance and
  evidence-run IDs, and runtime-evidence-only score intent; the audit reopens
  those bytes and cross-checks them against the verifier attestation;
- at least one training reward is nonzero and the eight-sample training
  group has reward variance. Uniform valid rewards are genuine outcomes but do
  not establish a usable GRPO learning signal;
- exactly one optimizer update occurred. Before the first native optimizer
  call, every policy rank seals its semantic state digest only after proving an
  empty optimizer state. The terminal audit reopens those eight create-once
  receipts and the independently decoded step-1 checkpoint, requires all
  optimizer/scheduler counters to equal one, and requires at least one exact
  base-to-checkpoint policy-rank digest delta. Counters, booleans or a checkpoint
  filename alone are not enough;
- the step-1 policy, optimizer, scheduler, sampler and RNG checkpoint is
  complete and sealed create-once, with the source unchanged;
- the exact W&B run exists and its local and remote step histories match. Step 1
  must contain `trainer/global_step`, `reward/avg_raw_reward`, the four
  `cyber/train/*` episode/reward/variance counters, `cyber/optimizer_step`, and
  `policy/{policy_loss,policy_lr,grad_norm}`, all finite and cross-checked
  against the sealed episode and optimizer receipts. Logged artifacts, tables,
  media, text and other rich payloads are rejected by inspection;
- the native reload gate below passes; and
- the Job, Workload, RayCluster, Pods and all eight GPUs are released.

A zero reward, zero within-group variance, missing verifier ID, uncertain
instance cleanup, failed W&B identity, missing independent update proof, or
unreloadable checkpoint keeps production closed. It is not repaired by silently
resampling or reclassifying an infrastructure failure as reward zero.

## Native checkpoint and reload gate

The native implementation is now present in
`training/skyrl_rl_checkpoint.py`; that fact alone does not qualify a
checkpoint. The inert, self-digested
[`qwen38-rl-reward-canary-reload-dev-v1.template.json`](../configs/qualification/qwen38-rl-reward-canary-reload-dev-v1.template.json)
records the unresolved live identities. Never submit the wrapper itself. Copy
only its `resolved_config` into a new private JSON file after the source run is
terminal and its exact evidence exists. Its `source` block retains the exact
reward-canary run, source plan/request, step-1 checkpoint manifest, source
release and independently audited reward-terminal receipt bindings.

There are four separate create-once receipts:

1. `cyber_skyrl_rl_reward_canary_terminal_v1` independently accepts the source
   run's reward, reward variance, optimizer update, W&B evidence, checkpoint
   linkage and external release. It does not claim reload acceptance;
2. `cyber_skyrl_rl_checkpoint_manifest_v1` seals the terminal source checkpoint
   after an independent source-release audit;
3. `cyber_skyrl_rl_reload_validated_v1` proves the in-process eight-rank reload,
   forward and current-Ray-job cleanup, but explicitly says external release is
   still unverified; and
4. `cyber_skyrl_rl_reload_accepted_v1` may be written only after a second
   independent audit proves the reload Job, RayCluster, Pods and all eight GPUs
   are gone.

Do not collapse those phases. A process cannot prove that its own Pod and GPU
allocation disappeared after it exited.

### 1. Audit and seal the source

Wait for the exact reward-canary Jobs API run, RayJob and Workload to reach
`SUCCEEDED`. A read-only monitor must bind their immutable UUIDs, the RayCluster
UUID and every Pod UUID. While those objects are live, it captures the canonical
payload for a `cyber_skyrl_controller_audit_v1` receipt. Do not add that file to
an active runtime output tree: persist it create-once at the exact bound path
after the process exits. That receipt records the
Workload owner as the exact RayJob UID, the RayCluster owner as that same
RayJob UID, and every observed Pod owner as the exact RayCluster UID. Each Pod
also binds its name, UID, immutable runtime imageID and zero restart count. It
binds the exact dev API run/request, effective priority 10000, disabled requeue
and one-worker/eight-GPU shape. After terminal success, the monitor observes the
RayCluster and GPU Pods absent and active GPU count zero and writes a
create-once `cyber_skyrl_external_release_v1` receipt on SFS. The release binds
both file and self digests of the earlier controller audit. Its `observed_at`
must be a valid UTC `Z` timestamp no earlier than either the controller audit
or `NATIVE_TRAINING_COMPLETE.json`; its `sha256` is the canonical JSON digest
of every other field. A name-only or API-only observation is insufficient.

The controller audit has exactly these fields: `schema`, `cluster`,
`kube_context`, `namespace`, `namespace_uid`, `api_base_url`, `run_name`,
`request_sha256`, `api_run_id`, `rayjob_name`,
`rayjob_uid`, `workload_name`, `workload_uid`,
`workload_owner_rayjob_uid`, `raycluster_name`, `raycluster_uid`,
`raycluster_owner_rayjob_uid`, `pods`, `effective_priority`,
`automatic_requeue`, `workers`, `gpus_per_worker`, `total_gpus`, `observed_at`
and `sha256`. Sort `pods` by UID. Every Pod item has exactly `name`, `uid`,
`owner_raycluster_uid`, `runtime_image_id`, `container_restarts` and `gpus`.
The exact dev context is
`nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb`; the namespace is
`fleet-train-jobs` with UID `10394b76-e1d4-40b1-a8e2-7575e95df216`. Exactly
one eight-GPU Pod is required, and the RayJob, Workload, RayCluster and Pod UIDs
must all be distinct.

The release receipt has exactly these fields: `schema`, `status`, `cluster`,
`kube_context`, `namespace`, `namespace_uid`, `run_name`, `plan_sha256`,
`request_sha256`, `api_base_url`,
`submission_journal_path`, `submission_journal_file_sha256`,
`controller_audit_path`, `controller_audit_file_sha256`,
`controller_audit_sha256`, `api_run_id`, `rayjob_name`, `rayjob_uid`,
`workload_name`, `workload_uid`, `raycluster_name`, `raycluster_uid`,
`pod_uids`, `api_status`, `controller_status`, `effective_priority`,
`automatic_requeue`, `workers`, `gpus_per_worker`, `total_gpus`,
`runtime_image_id`, `container_restarts`, `raycluster_present`,
`gpu_pods_present`, `active_gpus`, `gpu_release_proven`, `observed_at`, and
`sha256`. The receipt is accepted only when its copied two-line create-once
Jobs API journal proves the exact dev API, request digest, API run UUID and
generated RayJob name and its earlier controller audit independently proves the
exact UID ownership chain. Both terminal states are `SUCCEEDED`, effective
priority is 10000, automatic requeue is false, every observed Pod imageID
resolves to the requested immutable image digest, and the requested/observed
shape is one worker and eight GPUs. Restart and active-GPU counts are zero; the
two presence fields are false; and `gpu_release_proven` is true.

After terminal release, copy the original prepared directory's
`SUBMISSION.jsonl` create-once to exactly
`<source-output>/SOURCE_SUBMISSION.jsonl`; never reconstruct its intent or
response rows. The source release receipt path is exactly
`<source-output>/SOURCE_RELEASE.json`; the live controller audit path is exactly
`<source-output>/SOURCE_CONTROLLER_AUDIT.json`. The release binds the copied
journal's file SHA-256 plus the controller audit's file and self SHA-256.

On CPU, as UID/GID `1000:100` in the exact pinned SkyRL image, seal step 1:

```sh
cyber-post-train rl-checkpoint-seal <source-prepared-directory> 1 \
  --release-evidence /mnt/sfs/jobs/chris-q38-rlreward-dev1/SOURCE_RELEASE.json \
  --output /mnt/sfs/jobs/chris-q38-rlreward-dev1/RL_CHECKPOINT_STEP_1_MANIFEST.json
```

The command refuses CUDA and an existing destination. It revalidates the exact
source plan/runtime/image/model, native completion, original submission and
release receipt; requires
the native `data.pt`, `trainer_state.pt`, FSDP topology, eight model shards,
eight optimizer shards, eight extra-state shards and HF sidecars; reopens all
optimizer steps, scheduler counters, sampler cursor and RNG records; records an
exact semantic digest for every rank's policy, optimizer, scheduler and RNG;
hashes every file with inode/timestamp stability; seals the model/checkpoint and
terminal-file inode/size/mtime/ctime snapshot; and writes a self-digested
manifest. Its `sealer` block records the exact pinned image contract, sealer
runtime digest, native-source-set digest and required UID/GID `1000:100`.
Native PyTorch metadata uses pickle, so run this only for the trusted checkpoint
created by the bound source plan.

Record both identities: the manifest's embedded `sha256` and the SHA-256 of the
manifest file bytes. Put the absolute SFS manifest path and the latter digest in
the copied reload config. They are not interchangeable. Compilation also
requires the independently written create-once
`<source-output>/REWARD_CANARY_TERMINAL.json` with schema
`cyber_skyrl_rl_reward_canary_terminal_v1`. Its source run/plan/request, step,
checkpoint-manifest, controller-audit and source-release path/file/self digests
must exactly match. The terminal is deliberately thin: it binds separately
sealed episode, optimizer-update and W&B evidence receipts rather than copying
operator assertions. A consumer must reopen those files and rederive the ten
episode statuses/verifier IDs/reward vector and variance, decoded rank optimizer
state plus the eight pre-update policy receipts and semantic checkpoint delta,
exact scalar-history parity, checkpoint identity and GPU release.
`NATIVE_TRAINING_COMPLETE.json` cannot substitute for this terminal acceptance.

### 2. Prepare and validate without a POST

The copied config must retain exactly:

- schema `cyber_skyrl_rl_reload_config_v1`;
- a fresh run name and disjoint, absent SFS output root;
- dev cluster, `c1`/rendered `q1` priority 10000;
- one worker with eight GPUs, 64 CPUs and 512 GiB requested;
- 30-minute startup and five-minute cleanup bounds;
- no secrets, W&B, requeue, rollouts, verifier calls, optimizer updates or new
  checkpoints; and
- the exact source run, plan/request digests, checkpoint manifest file/self
  digests, source release file/self digests and reward-terminal receipt
  file/self digests inside `resolved_config.source`.

Prepare it once:

```sh
cyber-post-train rl-reload <resolved-private-reload-config.json> \
  --output <new-shared-prepared-directory>
```

Run `cyber-post-train preflight <new-shared-prepared-directory>` on CPU as
`1000:100` in the pinned image with the model, source checkpoint and SFS
manifest mounted. The preflight rehashes every source file, reloads the exact
trainer/optimizer/scheduler/RNG metadata, verifies the staged base model,
tokenizer and chat-template identity, and restores the native sampler into a
synthetic index-only dataloader. It reads no task prompts or trajectories and
does no reward work.

Then run:

```sh
cyber-post-train preview <new-shared-prepared-directory> --cluster dev
```

Accept only a warning-free preview matching the fixed shape above, with the
effective Pod/container security context explicitly proving UID/GID `1000:100`
and `runAsNonRoot=true`. Immediately
before the single POST, reconcile the dev Jobs API, RayJob/Workload/RayCluster,
SFS output and submission journal. Confirm one whole topology-compatible
eight-GPU dev node is available. Any existing or ambiguous identity blocks the
submit; never copy the prepared directory, replay a timeout or enable requeue.

### 3. Submit once and monitor sanitized evidence

Only after those checks:

```sh
cyber-post-train submit <new-shared-prepared-directory> --cluster dev
```

The validator directly creates eight one-GPU native FSDP policy actors on one
strictly packed node. All ranks initialize the exact base, restore their model,
optimizer, scheduler and RNG shards, and must match the corresponding sealed
rank's semantic digests before one fixed four-token forward under inference
mode. Policy and optimizer state are streamed through hashes before and after
the forward, gradients must remain absent, and the worker's optimizer method is
hard-disabled. The GPU process compares the sealed inode/size/mtime/ctime
snapshot before and after native use; the expensive content rehash and trusted
pickle checks happen in CPU preflight and again after GPU release, not twice
while eight GPUs are held. The runtime bundle and private-log tree are checked
recursively against their exact allowlists. No inference engine, task row,
rollout, verifier, W&B run, episode directory or checkpoint writer is created.
The driver and every rank separately prove UID/GID `1000:100` and the exact
native-source-set digest. A source snapshot change before or during reload is
an unexpected integrity failure: it writes `FAILED.json` and exits nonzero,
never the clean rejection receipt.

Monitor only API/Kubernetes state, restart count, GPU/process utilization and
the presence/digests of `RELOAD_STARTED.json`, `RELOAD_VALIDATED.json`,
`RELOAD_REJECTED.json` or `FAILED.json`. Never inspect the private
infrastructure log. A narrowly recognized base/native-state contract mismatch
writes create-once `RELOAD_REJECTED.json` only after internal cleanup and exits
zero so it is not mislabeled as a failed cluster job; it closes promotion and
authorizes no successor. Unexpected runtime or cleanup defects still write
`FAILED.json` and exit nonzero. On either terminal defect or a confirmed stall,
preserve evidence and release only this exact owned dev run through the Jobs
API; no automatic or unchanged retry is allowed.

`RELOAD_VALIDATED.json` is not final acceptance. After the process reaches
terminal `SUCCEEDED`, preserve the live ownership observation as exactly
`<reload-output>/RELOAD_CONTROLLER_AUDIT.json`, then independently prove the
exact reload RayCluster and GPU Pods absent, active GPUs zero and restart count
zero. Write a new `cyber_skyrl_external_release_v1` receipt whose plan digest
and run name bind the reload plan, whose UTC observation is no earlier than the
reload result and live controller audit, and which binds that audit's file and
self digests. Copy the original prepared directory's `SUBMISSION.jsonl`
create-once to exactly `<reload-output>/RELOAD_SUBMISSION.jsonl`; the release
receipt must bind its file digest, and the journal's API run UUID/RayJob name
must equal the runtime start/result receipts, the controller audit and the
observed release. Seal the final handoff on CPU as UID/GID `1000:100` in the
pinned native image:

```sh
cyber-post-train rl-reload-accept <new-shared-prepared-directory> \
  --release-evidence /mnt/sfs/jobs/chris-q38-rlreward-reload-dev1/RELOAD_RELEASE.json \
  --output /mnt/sfs/jobs/chris-q38-rlreward-reload-dev1/RELOAD_ACCEPTED.json
```

Only a digest-valid `RELOAD_ACCEPTED.json`, linked to that exact
`cyber_skyrl_rl_reward_canary_terminal_v1` receipt, may satisfy the production
template's native-reload gate. This reload test never establishes capability
improvement by itself.
