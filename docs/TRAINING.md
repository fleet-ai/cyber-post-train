# Training

The public command compiles YAML directly to an exact plan and a generic Jobs API
request. There is no component factory or separate experiment-description language.
The current compiler uses the qualified Qwen SkyRL loader; other model/backend
combinations remain gated until their real-model tests pass.

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

## Configuration

```yaml
backend: skyrl
name: my-qwen-sft                 # unique DNS label, at most 31 characters
output_root: /mnt/sfs/jobs/my-qwen-sft
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
3. Set `FLEET_TRAINING_API_TOKEN` via your secret manager. Run
   `cyber-post-train preview output/my-run`. It checks the actual image, command,
   resources, normal queue, derived priority, Secret references and release policy.
4. Review ownership, access expiry, experiment-wide allocated resources and the
   canary evidence. Use a shared durable prepared directory with one submitter.
   `cyber-post-train submit output/my-run` checks preflight, exhaustively checks API
   duplicates, previews again and records a fsynced intent before its only POST.
   An uncertain POST blocks reuse: reconcile it, never delete the journal or make
   a differently named copy to retry.
5. `cyber-post-train status <returned-name>` reads sanitized state. Monitor the
   exact API/Kubernetes UIDs, progress receipts, utilization and checkpoints too.

The API injects W&B from the existing `wandb-api` Secret. Never put its value in
YAML or argv. Track scalars, configuration identities and checkpoint metadata;
do not upload task text, traces or source code. Reuse neither a W&B run ID nor a
training output directory for a new treatment.

## Checkpoints and completion

Validation runs before training, at the configured interval and at the final step.
Periodic validation and checkpoint intervals must agree. Keep the latest N
checkpoints plus the best checkpoint by fixed held-out loss. Saves include native
optimizer, scheduler, sampler and trainer state; a weights-only file is not a
recoverable checkpoint.

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

The runtime has fixed startup, no-progress and hard-runtime bounds. A confirmed
stall preserves evidence and exits truthfully; the Jobs API releases the allocation.
An independent monitor must confirm release and handle access failures explicitly.
Do not suppress alerts or hold GPUs while debugging a failed allocation.

`TRAINING_COMPLETE.json` means optimization and checkpoint production completed;
it does not mean the checkpoint is inference-ready or improves task success.
Export/reload is a separate no-optimizer operation. Resume must bind the exact
source checkpoint and consumed-data cursor; the compiler currently rejects an
unreviewed resume instead of starting from the base and pretending to resume.
