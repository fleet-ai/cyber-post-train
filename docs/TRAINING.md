# Training

The public command compiles YAML directly to an exact plan and a generic Jobs API
request. There is no component factory or separate experiment-description language.
Qwen's SkyRL path has a real training/checkpoint qualification. Full GLM5.3
LoRA has passed pinned-image CPU prerequisites; its full-size GPU qualification
is still pending. Other model/backend combinations remain gated.

## Pin a model

```sh
uv run cyber-post-train model-lock zai-org/GLM-5.3 \
  30333038ada1f1dacb294a93270305a890b50c14 --output output/my-model
```

This downloads public configuration/tokenizer metadata, not weights. It follows
the complete exact-revision inventory, checks Git-blob/LFS identities and writes
`model.lock.json`, `model.weights.json` and a final `COMPLETE.json` into a new
directory. Use those paths in the configuration below. The model must have indexed
safetensors and a standard tokenizer/chat template; remote model code is not
executed. Pinning does not stage weights or establish training compatibility.
Private/gated models need a separately reviewed authenticated staging path.

Reviewed Qwen and full-GLM5.3 locks are under `configs/models/`. Changing model
identity never silently changes the data split, precision or training method.

## Prepare data

Run `cyber-post-train data data.yaml` on CPU with the pinned SkyRL image and
locally staged tokenizer. This uses normalized Fleet records, not raw model
responses. Teacher and self-distillation differ only in the source file/model
filter; use the same frozen dev references when comparing them.

```yaml
source:
  path: /private/data/trajectories.jsonl
  sha256: <exact-source-file-sha256>
split: /private/data/split.json
model_lock: configs/models/qwen38-27b-1d4bf0f2.lock.json
tokenizer_root: /mnt/sfs/models/qwen3.8-27b-1d4bf0f2
native_helper: /opt/skyrl/skyrl/train/generators/utils.py
train_models: [gpt-5.6-sol]       # or the exact student ID in normalized records
max_length: 16384
context_tokens: 4096
dev_windows: 5
output: /private/data/new-corpus # must not exist
```

The split is a self-digesting JSON object with schema `cyber_task_split_v1` and
`tasks` entries containing `task_key`, `task_version_id`, `split`, and, for every
dev task, one fixed `reference_session_id`. It must be reviewed before data
preparation. The builder never moves tasks between splits or replaces missing
references. It rejects family leakage across versions, duplicate sessions,
unverified successes and unsupported tool transitions. External benchmark data
is prohibited. A new split cannot make an already-exposed checkpoint held out.

Outputs are private `train.parquet`, `dev.parquet`, `manifest.json` and a copy of
the split. Training covers each fitting assistant response once; tool observations
and copied earlier responses receive no loss. Complete recent tool rounds must
fit; exclusions remain accounted for. Dev uses a deterministic bounded spread
of last-response windows, not whichever windows happen to have the lowest loss.

For a current-catalog teacher corpus, use `cyber-post-train
data-fleet-teachers request.json`. This offline path additionally binds each
normalized record to a success-evidence receipt, the exact high-quality task
version, and the representative 50/17/8 split. It emits train data only; select
checkpoints with fresh task outcomes instead of held-out teacher-token loss.

## Configuration

```yaml
backend: skyrl
name: my-qwen-sft                 # unique DNS label, at most 31 characters
output_root: /mnt/sfs/jobs/my-qwen-sft
# New immutable full-run configs should bind the maximum acceptable wall time
# before their first recoverable checkpoint.
checkpoint_recovery_horizon_seconds: 28800
model:
  lock: configs/models/qwen38-27b-1d4bf0f2.lock.json
  weights: configs/models/qwen38-27b-1d4bf0f2.weights.json
  root: /mnt/sfs/models/qwen3.8-27b-1d4bf0f2
data:
  manifest: /path/to/new-corpus/manifest.json
  root: /mnt/sfs/jobs/my-corpus/data
recipe:
  epochs: 1
  batch_size: 8
  microbatch_per_gpu: 1
  nodes: 1
  gpus_per_node: 8
  lr: 0.000003
  max_length: 16384
  eval_interval: 50
  checkpoint_interval: 50
  keep_checkpoints: 3
  seed: 42
wandb:
  entity: your-team
  project: cyber-post-train
  group: teacher-sft-lr-study
  run_id: my-qwen-sft
  name: my-qwen-sft
  tags: [teacher, sft]
cluster:
  priority: c1
```

Relative manifest paths resolve beside the YAML file. Model and data roots are
specific staged SFS directories, not names to download during GPU startup. Unknown
fields are errors. Hyperparameters do not authorize a larger resource budget.

For full GLM5.3, select `configs/models/glm53-30333038/model.lock.json` and
`model.weights.json`, use a corpus built with that exact tokenizer, and add
`lora: {rank: 16, alpha: 32}`. The loader converts the FP8 base to frozen BF16
weights and updates only attention adapters—not all model weights and not GLM
Flash. Explicitly request at least two eight-GPU nodes, global batch 16, and
`cluster.resources: {memory_request: 2048Gi, memory_limit: 2304Gi}` per node.
Rank zero must first materialize the roughly 1.5 TB BF16 base on CPU. These are
preflight resource minimums, **not a completed full-model capacity proof**; a
bounded exact-model canary is required before production use.
The pinned loader uses sixteen CPU threads only while loading/converting/merging,
then restores the prior setting; the compiler requires at least sixteen requested
CPUs. [Real-weight checks](evidence/cleanup-glm-loader-throughput-20260911.json)
cover a complete 256-expert gate/up group and prove identical values. They do not
prove that loading every layer and distributing it fits the startup allowance;
keep GPUs released until that execution plan is justified.

Global batch must be divisible by nodes × GPUs/node × microbatch/GPU. Native SkyRL
keeps the tail batch, so optimizer steps are `ceil(train_rows / batch_size) × epochs`.
One row is a bounded token window, not one task or one rollout. The dense format
trains every eligible assistant response once per epoch, masking copied context.

The corpus manifest binds `tokenizer: {repo, revision}`, `split_sha256`, and
`files.train`/`files.dev`. Each file entry has relative `path`, `sha256`, `rows`,
`task_keys` and the format-specific token/response inventory. Its top-level
`sha256` is the SHA-256 of canonical compact sorted JSON excluding that field,
prefixed `sha256:`. Generate it with the data-preparation pipeline, not hand-edited
counts. Train and dev are distinct immutable artifacts and task families.

## Prepare → preflight → preview → submit

1. `cyber-post-train train config.yaml --output output/my-run` resolves the model,
   data and recipe, embeds the small runtime, and writes `plan.json`, `request.json`
   and `PREPARED.json`. It does not use the network or allocate a GPU.
2. Run `cyber-post-train preflight output/my-run` on an authorized **CPU-only**
   worker with the pinned training image, this package and staged SFS inputs.
   It verifies file digests, native source/config compatibility, tokenization and
   target accounting. Preserve its `PREFLIGHT.json` in the prepared directory.
   CPU validation cannot prove CUDA kernels or distributed startup; changes to
   those require a bounded exact-model canary before a full run.
3. Set the standard `FLEET_API_KEY` via your secret manager. Run
   `cyber-post-train preview output/my-run`. It checks the actual image, command,
   resources, normal queue, derived priority, Secret references, release policy,
   and the root RayJob annotation `fleet.ai/failure-alerts: "off"`. A missing
   annotation stops before submission even when the request asked for alerts off.
4. Review ownership, access expiry, experiment-wide allocated resources and the
   canary evidence. Use a shared durable prepared directory with one submitter.
   `cyber-post-train submit output/my-run` checks preflight, exhaustively checks API
   duplicates, previews again and records a fsynced intent before its only POST.
   An uncertain POST blocks reuse: reconcile it, never delete the journal or make
   a differently named copy to retry.
5. `cyber-post-train status <returned-name>` reads sanitized state. Monitor the
   exact API/Kubernetes UIDs, progress receipts, utilization and checkpoints too.

When an operator must transfer a prepared archive from a local machine into an
already-running CPU preflight Pod, never copy directly to the filename that the
worker watches. A direct copy makes the final name visible before all bytes have
arrived. Use the digest-checked atomic publisher instead:

```sh
uv run cyber-post-train pod-publish-file prepared.tar.gz \
  --context <explicit-cluster-context> \
  --pod <exact-preflight-pod> \
  --container <exact-container> \
  --destination /tmp/prepared.tar.gz
```

It copies to a unique sibling name, checks the complete file's SHA-256 and byte
count inside the Pod, then creates the final name atomically without replacing an
existing file. A watcher therefore sees either no final file or the complete
verified archive. A failed transfer is not retried automatically; inspect the
exact Pod and use a fresh reviewed destination or successor as appropriate.

### SFT-only direct-create fallback

Use the normal Jobs API `submit` command whenever its preview carries the exact
root annotation `fleet.ai/failure-alerts: "off"`. If the deployed API still
omits only that annotation, a maintained fallback is available for a prepared
SFT run after the same CPU preflight and source-freshness gates:

```sh
# On a host that sees the shared SFS mount, no more than five minutes before create:
uv run cyber-post-train sfs-output-receipt /shared/prepared-run \
  --output /shared/prepared-run/OUTPUT_ABSENT.json

# Or, when the submitter cannot mount SFS, create one reviewed CPU-only observer.
# It requires the already-accepted native PREFLIGHT.json; it does not replace preflight.
uv run cyber-post-train sfs-output-job-create /shared/prepared-run \
  --context <explicit-production-or-development-context> \
  --attempt 1

# After that exact Job succeeds, collect its still-fresh receipt immediately.
uv run cyber-post-train sfs-output-job-collect /shared/prepared-run \
  --context <explicit-production-or-development-context> \
  --attempt 1 \
  --output /shared/prepared-run/OUTPUT_ABSENT.json

# On the submitter host. Omit the receipt option only when this host sees SFS itself:
uv run cyber-post-train direct-submit-sft /shared/prepared-run \
  --output-absence-receipt /shared/prepared-run/OUTPUT_ABSENT.json \
  --context <explicit-production-or-development-context>
```

The fallback is deliberately narrow. It fetches a fresh API preview, proves the
saved request is the current SFT render and needs only the `wandb-api` Secret,
then rejects warnings or drift in identity, c1/q1 priority, normal suspension,
release-on-exit, image, command, resources, environment, Secret references, or
node count. It replaces the API's zero UUID/name placeholders with one fresh
UUID/name, removes only the preview-generated run-scoped `*-fleet-key` Secret
reference that SFT does not consume, and adds the alert annotation to the root
RayJob. Every other preview field is preserved.

Before creation it checks the complete Jobs API history and Kubernetes Job and
RayJob inventories for the name, output directory and run identity, performs a
Kubernetes server dry-run, and repeats the duplicate checks. It also proves the
create-once SFS output absent before preview and again immediately before the
durable create intent. A submitter without the SFS mount must consume the exact
plan/request-bound receipt above; the receipt expires after five minutes and is
revalidated after the server dry-run. That remote receipt is a recent SFS
observation, not a live post-dry-run filesystem read; the runtime's exclusive
`.runtime` creation remains the final output create-once guard. The supported
local check accepts the SFS root only when `lstat` proves it is an actual
directory entry; a symlink to a directory is treated as an unavailable mount.
The supported
remote observer is itself create-once: it uses the prepared request's exact
digest-pinned image and pull-secret names, root alert opt-out, c1/q1 priority,
normal `training-lq` admission, one CPU, 1 GiB memory, zero GPUs, a read-only SFS
mount and a five-minute deadline. Its submitted Job starts with `suspend: true`;
collection requires the live admitted Job to be unsuspended, terminally
successful, and bound by exact Job UID to one zero-restart Pod running the exact
image at effective c1. It also requires exactly one Kueue Workload owned by that
exact Job UID, still carrying nonempty admission, `Admitted=True`,
`Finished=True`, `training-lq` and effective priority 10000. Fleet's current
Kueue v1beta2 Workload binds q1 through the exact `priorityClassRef` tuple and
binds indexed Job topology through the exact completion-index label; the
retired scalar priority-class fields must remain absent. Numeric 10000 is proved
on the Workload, and unsuspension by itself is not admission evidence. The
scheduled Pod may gain the namespace default ServiceAccount's sole `ecr-pull`
image-pull reference only when the reviewed Job and Workload PodSet had no pull
secret. Collection re-reads that exact `default` ServiceAccount in the exact
namespace and rejects a missing, changed, or additional pull reference; it
never permits a Secret-backed environment, volume, or sidecar. The Pod emits
only one sanitized, plan/request-bound receipt line. Server dry-run and live
readback reject added Secrets, sidecars, privilege, host pinning, affinity,
runtime overhead, GPUs, or writable SFS.
The observer requires the accepted native `PREFLIGHT.json` and proves only that
the output path was absent at its timestamp; it does not validate the dataset,
trainer, CUDA path, or model.

A completed CPU preflight Job whose name ends in `-pre-v1` is not a training
duplicate; only the exact rendered
`<run-name>-<8 lowercase hex>` shape, run labels, run UUID, or output annotation
claims the training identity. The command then writes and fsyncs
`DIRECT_SUBMISSION.jsonl` before exactly one `kubectl create`. It never uses
`apply`, `patch`, automatic retry, or `POST /v1/runs`. A transport error after
that intent is ambiguous: reconcile the exact recorded name and UUID; never
delete the journal or invoke the command again. Because direct-created runs are
not Jobs API records, monitor them by their Kubernetes UID and durable training
receipts rather than `cyber-post-train status`.

This path refuses RL, conversion, requests with Fleet credential Secrets,
non-c1 priority, an already-qualified API preview, or any unreviewed placeholder
or generated field. It is a compatibility bridge, not permission to bypass
normal admission or scientific gates.

The API injects W&B from the existing `wandb-api` Secret. Never put its value in
YAML or argv. Track scalars, configuration identities and checkpoint metadata;
do not upload task text, traces or source code. Reuse neither a W&B run ID nor a
training output directory for a new treatment.

## Checkpoints and completion

Validation runs before training, at the configured interval and at the final step.
Periodic validation and checkpoint intervals must agree. Keep the latest N
checkpoints plus the best checkpoint by fixed held-out loss. Saves include native
optimizer, scheduler, sampler and trainer state; a weights-only file is not a
recoverable checkpoint. Before recording a save, the wrapper reopens the small
sampler/trainer files and checks their counters—SkyRL can catch a sampler-write
error after creating a partial file. Both paused and completed runs require
digest-valid final checkpoint and validation receipts bound to the same plan and
step. Full payload hashing and GPU reload remain separate handoff checks.

For a planned interruption/recovery check, add top-level `pause_after_step: 1`
to a plan whose full recipe has more than one step. This does not shorten the
epoch or scheduler horizon. It saves, validates and logs that step, then shuts
down cleanly with `TRAINING_PAUSED.json`, **not** `TRAINING_COMPLETE.json`.
Seal that checkpoint and use the exact recovery procedure below in a new run,
omitting `pause_after_step` to finish the original recipe. A pause must be after
the source checkpoint and before full completion; it cannot extend a finished
run or turn an unexpected failure into success. Native CPU qualification and
real resumed-GPU qualification are reported separately in the consolidation
checklist.

Before handing a native checkpoint to an export or resume operation, seal it on
CPU using the original prepared directory:

```sh
uv run cyber-post-train checkpoint-seal /shared/prepared-run 50 \
  --output /shared/checkpoint-step-50.json
```

This verifies the saved step and sampler cursor, hashes every rank's model,
optimizer and random-state files, and writes a create-once manifest without
changing the checkpoint. Only use trusted checkpoints from the bound run: native
PyTorch metadata uses pickle. Sealing proves file identity, **not GPU reload**.
When a seal or independent verifier runs as a direct CPU Pod, send it through
the repository's CPU-checkpoint create boundary. That boundary requires c1,
zero GPUs, the shared CPU pool and architecture, the failed-job alert opt-out,
and rejects a hostname, `nodeName`, or extra placement affinity before kubectl
is called. A busy host must never strand a seal while equivalent CPU nodes are
free.
The GLM LoRA path seals adapters plus optimizer/scheduler, per-rank random state,
sampler and trainer state. It cross-checks the exact base, adapter configuration
and consumed-data cursor; frozen base weights are never copied into checkpoints.
It does not use the Qwen full-weight export command below.

The CPU-only export command consumes the sealed manifest, with its externally
recorded **file** SHA-256 (not the embedded receipt digest):

```sh
uv run cyber-post-train checkpoint-export /shared/checkpoint-step-50.json \
  --sha256 <manifest-file-sha256> --output /shared/qwen-step-50-bf16
```

This path currently targets Qwen3.8 only. It rechecks all source files, joins
native rank shards one tensor at a time, casts trained FP32 weights to BF16,
restores only the exact frozen-base MTP allowlist and runtime sidecars, then
reopens every written tensor before atomic no-replace publication. It writes
`EXPORT.json`, **not a GPU reload acceptance**. Use the pinned training image on
CPU; the implementation bounds a source tensor to 8 GiB and output shards to
3 GiB. A 32 GiB request / 48 GiB limit is the qualification shape, not permission
to occupy a GPU. A leftover `.partial` directory is preserved for diagnosis and
blocks reuse; do not remove it automatically. Real-checkpoint export/reload
qualification remains recorded separately in `docs/CONSOLIDATION.md`.

Check the exported model with the same pinned image before handing it to serving:

```sh
# CPU-only: hashes all files and checks the complete HF model's meta layout.
cyber-post-train checkpoint-check /shared/qwen-step-50-bf16/EXPORT.json \
  --sha256 <export-file-sha256> --output /shared/export-cpu-check.json
# In a separately authorized, bounded one-GPU Jobs API run:
cyber-post-train checkpoint-check /shared/qwen-step-50-bf16/EXPORT.json \
  --sha256 <same-export-file-sha256> --output /shared/export-gpu-check.json --gpu
```

The GPU check loads the complete BF16 model, applies the training image's exact
Qwen Torch GDN patch, runs a finite forward pass and generates two tokens from a
fixed synthetic input, then rehashes the export unchanged. It logs counts, not
tokens. Receipt paths must be new and outside the export. The loaded model's
parameters and artifact-only MTP tensors are checked separately; HF warning text
and `hf_device_map` representations are not parameter-identity evidence. This
eager-attention smoke test does **not** qualify optimizer recovery, a serving
engine, tool behavior, throughput, or model quality.

The runtime has fixed startup and no-progress bounds. Its absolute run limit is
derived from the immutable optimizer-step count, context length, topology and
gradient accumulation; the SFT waiter fails closed if that plan is missing.
For new immutable task-outcome full-run configs, prefer binding
`checkpoint_recovery_horizon_seconds`; when present, it requires the next
periodic or terminal checkpoint to fit inside an explicit recovery-loss budget.
This is a conservative watchdog ceiling, not the expected checkpoint cadence;
the expected cadence is `recipe.checkpoint_interval` multiplied by the observed
step time. See [`CHECKPOINT_POLICY.md`](CHECKPOINT_POLICY.md) for the current
Qwen3.8 SFT and RL values and their recovery-qualification boundaries.
Teacher-cross-entropy plans are rejected because their pre-training development
scan is not represented by this bound. Choose a new checkpoint interval and run
identity when the bound does not fit; do not edit a historical config in place.
A confirmed stall preserves evidence and exits truthfully; the Jobs API releases
the allocation.
An independent monitor must confirm release and handle access failures explicitly.
The supported failed-job notification opt-out does not hide terminal failure state.
Preserve that evidence and never hold GPUs while debugging a failed allocation.

`TRAINING_COMPLETE.json` means optimization and checkpoint production completed;
it does not mean the checkpoint is inference-ready or improves task success.
Export/reload is a separate no-optimizer operation. Resume must bind the exact
source checkpoint and consumed-data cursor.

### Validate or resume a saved run

Copy the original training YAML, give it a **new name, output directory and W&B
run ID**, and add:

```yaml
recovery:
  manifest: /shared/checkpoint-step-50.json
  sha256: <manifest-file-sha256>
  mode: validate  # resume continues training; validate performs no optimizer step
```

Use the same prepare/preflight/preview/submit commands. Model, data, recipe,
topology and trainer image must stay unchanged. Both modes restore every rank's
optimizer/scheduler and the saved sampler cursor; a missing state is an error,
never a warning followed by a fresh start. Validation performs held-out forward
passes and writes `RELOAD_VALIDATED.json`, without saving or updating the model.
Continuation requires recorded supervised-token progress and starts at the next
step; completed runs and older checkpoints without that progress cannot be
silently continued. Never run a source trainer and its recovery concurrently.

SkyRL restores the remainder of the interrupted epoch exactly. Its native random
sampler can reshuffle **later epochs differently** after recovery, while still
covering each example once. This is not a promise of bit-identical final weights
to an uninterrupted multi-epoch run. GPU recovery qualification is tracked in
`docs/CONSOLIDATION.md`; CPU tests alone do not establish it.

## RL integration status

The `train` command prepares **SFT**. `rl-data` and `rl` support native **Miles
or SkyRL RL** through the same preparation/submission interface. Both RL backends
still require live reward, optimizer and recovery qualification; implemented and
CPU-tested does not mean production-ready. Historical typed-API requests are not
launch shortcuts. See the dated [qualification evidence](CONSOLIDATION.md).

`cyber-post-train rl-data rl-data.yaml` prepares private native-trainer input files on
CPU in the selected backend's pinned image. It performs only Fleet account/task GETs, never
creates an environment, generates tokens or submits a training job:

```yaml
name: my-qwen-rl
backend: miles  # or skyrl; use that backend's exact runtime image
task_set: reviewed-tasks.json
split: reviewed-split.json
tool_catalog: reviewed-mcp-tools.json
model_lock: configs/models/qwen38-27b-1d4bf0f2.lock.json
model_root: /mnt/sfs/models/qwen3.8-27b-1d4bf0f2
output: /private/data/my-qwen-rl
limits:
  context_tokens: 98304
  response_tokens: 81920
  max_tokens_per_turn: 4096
  max_turns: 80
  episode_seconds: 2400
  tool_seconds: 330
  tool_result_chars: 50000
```

These are explicit budgets, not a claim that every task fits. The task set is a
self-digesting `cyber_rl_task_set_v1` object with `training_data_eligible: true`,
the exact canonical MCP `tool_catalog_sha256`, and `tasks`. Each task has
`task_key`, `task_version_id`, `env_key`, `env_version`,
`environment_version_id`, `data_key`, `data_version`, and reviewed
`lineage: {application, task_family}`. The split uses the existing
`cyber_task_split_v1` format. Both files must cover the same complete set so
lineage can be checked across train/dev/test; test and reserved-dev tasks are
never fetched. Do not approve external benchmark tasks for this input.

`tool_seconds` must exceed the catalog's maximum `bash.timeoutMs` value, leaving
time for transport. For the five-minute tool maximum, 330 seconds leaves 30 seconds
of headroom. Preparation rejects a smaller/equal client deadline before network
or native model loading; runtime rechecks the live catalog. It never changes the
advertised tools to fit a short client timeout. Changing this budget requires a
new prepared run; existing episodes and their outcomes stay unchanged.

The catalog must be the exact previously observed task-facing MCP catalog, in
`bash`, `submit_report` order—not an invented tool schema. Runtime checks it
again. The command writes `train.jsonl`, `dev.jsonl`, split/task-set copies and
a final self-digesting manifest. It renders actual task prompts plus tools with
the selected backend's native Qwen tokenizer, checks recorder-prefill token equality, reserves
the full response budget and proves the native dataset retains every row.
It fails instead of truncating/filtering a task or moving its split. Output is
create-once and private; no prompts or task responses are printed. Reuse neither
a partial destination nor data bound to a different run name/model/budget.

SkyRL keeps chat messages and a canonical JSON binding in each native dataset
row; Miles keeps rendered text and its native metadata. They use different
templates and cannot share a prepared data manifest. SkyRL additionally binds
the initial token sequence, explicitly requests token-list output from
Transformers, and checks the actual `PromptDataset` preserves every prompt,
environment and binding. Preparing data is not a real rollout or RL qualification.

### Prepare a native Miles checkpoint

Miles' Megatron backend needs a distributed checkpoint, not the SkyRL checkpoint
format. `miles-convert` prepares this separate, zero-optimizer operation:

```yaml
name: my-qwen-miles-base
output_root: /mnt/sfs/jobs/my-qwen-miles-base
model:
  lock: configs/models/qwen38-27b-1d4bf0f2.lock.json
  weights: configs/models/qwen38-27b-1d4bf0f2.weights.json
  root: /mnt/sfs/models/qwen3.8-27b-1d4bf0f2
cluster:
  priority: c1
```

Run `cyber-post-train miles-convert conversion.yaml --output output/conversion`,
then the same CPU `preflight`, Jobs API `preview`, and explicit `submit` commands
described above. This uses one eight-GPU node, the pinned Miles image and its
unchanged native converter, with a 64 CPU / 512 GiB minimum loading reservation.
The Jobs API's driver attaches to its existing Ray cluster and assigns exactly
eight GPUs to one conversion task; it does not start or stop node-wide Ray.
Native automatic pipeline splitting avoids loading eight full model copies.
No downloads, source-model edits, Fleet sessions, optimizer steps or training
W&B run are involved. Conversion output and private logs are create-once. The
fixed 30-minute execution limit includes input I/O, with bounded cleanup of only
the converter's child process group. A partial output cannot be replayed.

After exact terminal success **and independently confirmed GPU release**, run
`cyber-post-train miles-seal output/conversion --output /shared/miles-checkpoint.json`
on CPU. It rechecks source identity, the native release tracker and complete file
inventory, then hashes the checkpoint without modifying it. This is a candidate
checkpoint, not an RL optimizer or GPU reload qualification. Actual conversion
and subsequent native reload remain required before using a new checkpoint in RL.

### Native training integration

After preparing the exact prompts and sealing the base conversion, use:

```yaml
backend: miles
name: my-qwen-rl  # must match rl-data's name
output_root: /mnt/sfs/jobs/my-qwen-rl
model:
  lock: configs/models/qwen38-27b-1d4bf0f2.lock.json
  weights: configs/models/qwen38-27b-1d4bf0f2.weights.json
  root: /mnt/sfs/models/qwen3.8-27b-1d4bf0f2
data:
  manifest: /shared/rl-data/manifest.json
  root: /mnt/sfs/data/my-qwen-rl
checkpoint:
  manifest: /shared/miles-checkpoint.json
  sha256: <manifest-file-sha256>
recipe:
  nodes: 1
  steps: 1
  groups: 1
  samples_per_prompt: 2
  lr: 0.000001
  eval_interval: 1
  checkpoint_interval: 1
wandb:
  entity: <your-team>
  project: cyber-post-train
  run_id: <new-run-id>
cluster:
  priority: c1
```

`cyber-post-train rl rl.yaml --output output/my-rl` prepares only. The same
`preflight`, `preview`, explicit `submit`, and `status` commands apply. All input
roots must be staged and disjoint from the new output; the preparation manifest's
paths and target hashes are rechecked before native training sees any data.

The current Qwen Miles profile reserves 1,536 GiB host RAM and caps it at
2,048 GiB per node (64 CPUs requested). Its colocated rollout engines, trainer
and CPU offload exhausted the former SFT-derived 768-GiB cap before any episode.
Smaller reservations/limits are rejected; larger ones remain configurable via
`cluster.resources`. These are conservative startup bounds, not a completed
capacity proof. Check the node's allocatable RAM and normal admission before
submitting. This change does not alter the model, task split or optimizer recipe.

CPU preflight checks the native FTI argument builder and actual text-only data
source, including template identity and train/dev row retention. Qwen's automatic
vision processor must not reinterpret Miles' already-rendered text prompts;
the adapter preserves native cursor/checkpoint behavior and passes no processor.
Megatron's actual parser imports Transformer Engine and `libcuda.so.1`; it is
not usable on a driver-less CPU node. The receipt explicitly records
`native_parser_checked: false`. The bounded GPU child then runs the unchanged
native parser and driver; no stub driver or fake GPU is used to claim readiness.
CPU constructor checks and GPU startup are separate qualification gates.

The wrapper attaches to Jobs API's Ray allocation and never calls native FTI's
process-killing launcher. Its watchdog covers input I/O as well as training and
cleans only its own child process group. W&B is configured for scalar telemetry with
console/code capture disabled. `NATIVE_TRAINING_COMPLETE.json` means the native
loop returned and expected batch/checkpoint records exist, **not independently
verified learning or recoverability**. Preserve the private recordings, verify
optimizer changes and reload the checkpoint before accepting/scaling the run.
Native checkpoint numbers are zero-based rollout indices: the first training
batch saves index `0`, not an assertion of zero optimizer steps.
The text adapter reopens each native cursor save and checks task/sample counters
against completed batches. Recovery rejects missing, partial or inconsistent
cursors instead of Miles' silent reset to the first task; the initial base load
is the sole exception. It also rejects weights-only/finetuning flags as recovery.
These guards do **not** yet expose an RL resume command: a complete, sealed native
optimizer/RNG/checkpoint handoff and real distributed restore still need qualification.

For **SkyRL**, use the same YAML and commands, with these differences:

- Run CPU preflight as the pinned image's user `1000:100`, not root. Keep private
  staged data owned/readable by that user and its parent directories traversable.
  Root-only `0700` staging passed an earlier root preflight but failed on GPU.
  Do not make private data world-readable; prepare a correctly owned successor.
- Set `backend: skyrl` and prepare a matching `rl-data` manifest with that backend.
- Omit `checkpoint`: this initial profile loads the exact HF base directly. It
  does not consume Miles checkpoints or automatically resume old SkyRL runs.
- Use complete prompt batches, for example `groups: 2` and
  `samples_per_prompt: 4` on one eight-GPU node. The training-row count must be
  divisible by `groups`, and `groups × samples_per_prompt` by the GPU count.
- `keep_checkpoints` defaults to two. Native SkyRL saves model/optimizer/sampler
  state and performs dev evaluation before training, at the configured interval
  and at the final step. A missing sampler save is a terminal evidence defect.

The initial SkyRL profile uses Qwen full-weight FSDP with colocated TP4 inference
engines, one native update per prompt batch, and no reward filtering or replacement
episodes. It is not a qualified full-GLM recipe. CPU preflight checks the actual
native parser, source versions, complete task-family split and native dataset
retention. Its native validator requires `WANDB_API_KEY` even during CPU preflight;
inject the existing `wandb-api` Secret and set `WANDB_MODE=disabled` for this
non-training check. Do not put a key in the saved config. Runtime input checks
repeat inside the bounded child before loading.
W&B receives finite scalar metrics and fixed configuration only; native sample
tables, console capture and private exception uploads are disabled. A durable
local scalar stream is retained if tracking fails. Completion still requires
independent real reward, optimizer and checkpoint-reload evidence before scaling.

Use the native trainers, not a new optimizer implementation. The inspected
[Theseus FTI integration](https://github.com/fleet-ai/theseus/tree/cc18d2cd3e9370abf4f6f19df317d96ce6b619e4/services/fti/src/fti/trainers/miles)
provides Miles token recording and a Qwen3.8 text recipe (Megatron TP4/CP2,
one or two eight-GPU nodes; TP1 SGLang engines). Its GLM recipe is **Flash**,
not our full GLM5.3. Its stock agent uses Platform V2 and `fleet_submit`, so it
must not silently replace our exact V1 `bash`, `submit_report` contract.
Reuse its token recorder and native training recipe only with a qualified cyber
adapter. Never call its W&B argv helper, which places a key in process arguments;
use workload Secret injection. Stage exact model bytes before GPU startup rather
than calling its unpinned model-download helper.

The inspected [SkyRL adapter](https://github.com/fleet-ai/theseus/tree/6b7e1304e0782b9586d0fb03e955f2264a8f32db/services/rl-rollout/rl_rollout)
already supports exact V1 cyber tasks and authoritative verifier execution IDs.
Its default episode/SDK retries and best-effort cleanup still need a bounded
cyber policy: never replay an ambiguous create/score request or train an
infrastructure failure as reward zero. Both backends must retain per-episode
verifier IDs, confirmed cleanup, sampled-token/log-probability alignment and
masked tool observations. A group with identical valid rewards is a valid
zero-signal group, not proof of useful learning. Require real reward acquisition,
an optimizer update and a recoverable checkpoint before scaling either backend.

Native SkyRL has its own post-training evidence adapter in
`training/skyrl_posttrain.py`; it never sends a SkyRL plan through the SFT plan
validator. `seal_checkpoint` is a CPU-only, create-once gate over the exact
terminal run receipt, every planned train/development batch, every accepted
Fleet episode, authoritative reward and verifier IDs, confirmed environment
cleanup, finite per-update loss/KL/entropy/gradient scalars, a nonzero parameter
change, checkpoint cadence, sampler cursor, and the complete FSDP payload. It
hashes the checkpoint, staged base model, private train/dev package, and evidence
files twice and rejects any replacement or drift. The public manifest contains
only counts, digests, and immutable identities; it never copies prompts, traces,
flags, answers, or private trainer logs.

`export_checkpoint` accepts only the resulting
`cyber_native_skyrl_rl_checkpoint_manifest_v1` receipt. It reuses the qualified
one-tensor-at-a-time native FSDP reconstruction writer, creates a complete BF16
Hugging Face destination exactly once, executes zero optimizer updates, reopens
every output tensor, restores only the exact frozen MTP allowlist from the bound
base, and proves the checkpoint/base/data inputs stayed unchanged. Its
`cyber_native_skyrl_rl_hf_export_v1` receipt is still not a GPU reload or serving
certificate. A separate zero-update reload, paused new-UID registration, and
live parity check remain mandatory before evaluation traffic.

After the exact run has terminally succeeded and released its GPUs, invoke the
same gates through `cyber-post-train rl-checkpoint-seal <prepared-dir> --output
<new-manifest.json>` and then `cyber-post-train rl-checkpoint-export
<manifest.json> --sha256 <manifest-file-sha256> --output <new-export-root>`.
Both destinations are create-once. Do not run either command against a live,
failed, rejected, partial, or unbound run.

### Adapter boundaries for maintainers

Both adapters preserve sampled token IDs/log probabilities, mask tool/template
observations, enforce the exact tool catalog and save private evidence before
returning a sample. Each run/phase/batch/sample has a create-once identity.
Ambiguous requests and exhausted budgets are held, never resampled or turned
into reward zero. A normally stopped, authoritatively graded zero remains valid.
Multiple tool calls in one turn are rejected, not silently discarded.

SkyRL uses its native client, tokenizer helpers and `BasePPOExp`; the included
Qwen XML parser corrects the older JSON-only parser in the pinned image. Miles
uses FTI's native token recorder and digest-bound `qwen3.8_fixed.jinja` on both
sampling and training sides. The original HF template cannot encode Miles'
incremental tool-history prefix. Never re-tokenize sampled history to hide drift.
Transport tests run against each pinned image's real MCP version, not only mocks.

Do not restore native automatic generation retries, exception-driven replacement
groups, raw response logging, model downloads or broad process-killing launchers.
The adapters use single-attempt requests, await sibling cleanup and attach only
to the Jobs API's Ray allocation. No module-global patching across episodes.

Miles uses unique native sample indices for `Sample.rollout_id`; a batch ID would
incorrectly merge reward-normalization groups. Baseline and post-update dev calls
also have distinct durable identities even when the native rollout index is zero.
`COLLECTED.json` proves collection, not optimization. The launcher explicitly sets
`MILES_USE_LEGACY_ROLLOUT_V1=0` and `WANDB_RUN_ID`; credentials stay in Secret-backed
environment variables. See [consolidation](CONSOLIDATION.md) for the exact test,
runtime and real-execution evidence instead of inferring readiness from this guide.
