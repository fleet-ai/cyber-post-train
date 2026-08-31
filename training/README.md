# Modular cyber post-training stream

This directory implements a model-swappable reproducible boundary around training: read-only
Fleet export, lossless tool-trajectory normalization, lineage/leakage tracking,
SFT and preference materialization, online-RL prompts, reward composition, and
a fail-closed model-compatibility/run-plan gate.

The current student is dense `Qwen/Qwen3.6-27B` at exact revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. Model-specific checkpoint,
tokenizer, topology and serving details live behind model adapters and run
configs; the corpus, split, reward and evaluation contracts do not depend on
Qwen. The earlier GLM-5.2 path remains historical provenance, not the current
optimizer source.

## Data flow

```text
Fleet completed jobs
  -> private raw export (transcript route only; no grader stdout/source)
  -> normalized immutable trajectories
     -> verified successes -> leakage-safe SFT train/dev/test
     -> matched success/failure -> optional preference data
     -> every valid task -> fresh online GRPO environment prompt
  -> manifest + exact digests
  -> compatibility-gated SFT -> pre-RL eval -> GRPO -> post-RL eval
```

Use credentials only through the environment or the cluster secret store:

```bash
python -m training export \
  --job-id a62dd51f-a52b-4941-8207-4679e4b25b51 \
  --output /secure/fleet/raw-sessions.jsonl

python -m training normalize \
  --input /secure/fleet/raw-sessions.jsonl \
  --output-dir /secure/fleet/glm52-v1

python -m training plan \
  --config training/configs/glm52_all_fleet.json \
  --manifest /secure/fleet/glm52-v1/manifest.json \
  --compatibility-receipt /secure/receipts/glm52-nebius.json \
  --evaluation-protocol /secure/protocols/glm52-cyber-v1.json \
  --output /secure/plans/glm52-fleet-v1.json

# Dry-run is the default; no cluster process starts.
python -m training run \
  --plan /secure/plans/glm52-fleet-v1.json \
  --work-dir /secure/runs/glm52-fleet-v1
```

For full historical discovery, use `--discover-completed-cyber
--created-after <ISO timestamp>`. The discovery path fetches completed jobs,
then retains only jobs containing a blackbox task key. An explicit lower bound
is required so a typo cannot issue an unbounded historical export.

## Training choices

- Start with shared-expert LoRA on attention and expert MLP projections. It is
  recoverable, small to checkpoint, and permits fast rank/target ablations.
- Run one epoch of verified-success SFT. A deterministic split unit binds the
  application, vulnerability family and exact task lineage, so sessions from a
  lineage cannot cross train/dev/test. All exported data remains accounted for.
- Use failures as matched preference negatives and, most importantly, as fresh
  online-RL prompts. Never imitate an unverified failing action sequence.
- GRPO reward is `verifier_score × behavior_score × integrity_gate`.
  Infrastructure failures and grader/reward hacking are always zeroed.
- Keep WebExploitBench outside this pipeline. Its terms make it an evaluation
  holdout, not training or reward-design data.

## Windowed SFT corpus

Historical successful agent sessions are much longer than the 16,384-token SFT
ceiling. Right-truncating whole sessions retained only 9.36% of assistant targets
and no final-success turn in the 508-record training set. Build five
assistant-ending windows per trajectory instead: the original task instruction
is retained, recent contiguous context is maximized, the final success is always
selected, and only the last assistant message receives loss.

```bash
uv run python -m training.stage_sft_corpus \
  --trajectories data/processed/glm52-fleet-v2/trajectories.jsonl \
  --output-root data/processed/qwen36-windowed-v4 \
  --team-id a1025f0b-ad67-49fc-a023-51800ab43e84 \
  --tokenizer Qwen/Qwen3.6-27B \
  --tokenizer-revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --window-max-tokens 14336 \
  --targets-per-trajectory 5 \
  --artifact-stem chris-cyber-fleet-a62dd51f-qwen36-windowed-v3 \
  --corpus-job-id chris-cyber-qwen36-windowed-v2
```

The unique corpus job ID is intentional: the Fleet Training API uses it to
select only these immutable windows even when older rows from the same source
job remain on SFS. The stage manifest separately preserves the original Fleet
job UUID and source trajectory digest.

## Exact Fleet RL task split

The task-first multi-environment RL request is built from a committed,
secret-free split lock rather than mutable catalog `current` pointers. Create
or audit that lock with the read-only task picker, then regenerate the typed
request deterministically:

```bash
FLEET_TRAINING_API_TOKEN="$(gh auth token)" uv run python -m training rl-snapshot \
  --trajectories data/processed/glm52-fleet-v2/trajectories.jsonl \
  --dataset-manifest data/processed/glm52-fleet-v2/manifest.json \
  --source-job-id a62dd51f-a52b-4941-8207-4679e4b25b51 \
  --output configs/data/fleet-a62-task-split-v1.json

uv run python -m training rl-config \
  --task-split configs/data/fleet-a62-task-split-v1.json \
  --template configs/runs/qwen36-27b-rl-base-full.template.json \
  --output configs/runs/qwen36-27b-rl-base-full.json

uv run python -m training rl-config \
  --task-split configs/data/fleet-a62-task-split-v1.json \
  --template configs/runs/qwen36-27b-rl-base-full.template.json \
  --exclusions configs/data/fleet-a62-rl-exclusions-v1.json \
  --treatment-receipt configs/data/fleet-a62-rl-treatment-v1.json \
  --output configs/runs/qwen36-27b-rl-base-full-runnable.json
```

The lock binds 160 exact source-job task versions: 130 train, 10 dev, and 20
untouched test. The runnable request contains only the 130 train and 10 dev
bindings. Every runtime task has an empty `env_variables` object; prompts,
verifiers, flags, credentials, image URLs, and catalog environment values are
never copied into the repository. Preview/submit re-resolves the authoritative
task and environment rows by UUID.

As of 2026-08-29, 159 bindings resolve through the task picker. One archived
train task (`f956619f-ab6d-4851-b45d-496a331f90c6`) is still bound from the
exact source-job roster and exact environment catalog, but the server correctly
refuses it as no longer runnable. The non-submitting preview receipt is
`configs/runs/qwen36-27b-rl-base-full.preview.json`. The intent-to-treat arm
remains frozen at 130 tasks. The separately named as-treated request contains
129 runnable train tasks and ten dev tasks, records the sole exclusion and its
evidence in `configs/data/fleet-a62-rl-treatment-v1.json`, and has a green HTTP
200 server preview in
`configs/runs/qwen36-27b-rl-base-full-runnable.preview.json`. Never describe
that execution as 130/130, and never silently substitute a newer task version.

The causal ablation is base, SFT-only, RL-from-base, and SFT→RL when compute
allows. This separates tool-format adaptation from verifier-grounded capability
learning. Recipe selection uses lineage-held-out Fleet tasks; after the recipe is
frozen, a final all-data fit may be measured only on untouched external
benchmarks. Training-distribution scores are reported separately and are never
presented as generalization.

## Nebius Jobs API

Typed SFT and RL jobs use the queue-aware Fleet Jobs API at
`https://api.ft.flt.build`; direct `kubectl` submission is retained only for
historical custom manifests. Preview is the default, exact duplicate titles are
reported, and submission requires `--execute`:

```bash
FLEET_TRAINING_API_TOKEN="$(gh auth token)" uv run python -m training jobs-run \
  --config configs/runs/qwen36-27b-sft-full.json

FLEET_TRAINING_API_TOKEN="$(gh auth token)" uv run python -m training jobs-run \
  --config configs/runs/qwen36-27b-sft-full.json --execute

FLEET_TRAINING_API_TOKEN="$(gh auth token)" uv run python -m training jobs-status \
  ft-run-1c54ba33
```

The API renders the authoritative RayJob, submits it through Kueue, and exposes
durable run status. Never place the bearer token in a config, receipt, command
argument, or repository file.

### Native RL successor safety gate

The first full native run (`ft-run-0081ca94`) established two launch defects that
must not be inherited by a successor:

1. The current Train API maps `grpo.max_steps` to SkyRL `trainer.epochs`. With
   129 rows and train batch size two, `max_steps: 130` rendered 130 complete
   dataloader passes: 8,320 batches, not 130 steps. A successor request must carry
   the explicit `trainer.max_training_steps=130` argument, and the live preview
   must contain that exact argument. The historical runnable config is preserved
   byte-for-byte as evidence; build successors from
   `configs/runs/qwen36-27b-rl-successor.template.json`.
2. The exact resolved runtime parquet had empty `tools` for all 129 train and all
   ten dev task versions. Empty means “expose every environment tool”; the fira
   schema alone is about 7,198 tokens, and the observed run remained 100%
   truncated through its step-10 eval. Task names and prompts are not authority
   for guessing a smaller list.

`jobs-run` therefore reports `launch_blockers`, and `--execute` refuses a paid RL
POST unless both conditions are proven. Tool proof must be returned by the live
Train API preview as `task_tool_allowlist_evidence` with schema
`fleet_rl_task_tool_allowlists_v1`, source
`authoritative_task_version_metadata`, source field `metadata.tools`, an ordered
binding for every exact train+eval `task_version_id`, exactly the two intended
task-facing tools (`bash` and `submit_report`) per binding, and a canonical
`bindings_sha256`. Caller-authored config
cannot supply this proof. The current API does not emit it, so the successor is
intentionally blocked until authoritative task metadata and preview support land.

## Exact pre/post identity

`training.science` makes the evaluation protocol a hard input to the training
plan. Both arms must share one protocol digest binding the base checkpoint shard
manifest, tokenizer files, chat template, inference image and engine commit,
precision/quantization recipe, harness image and commit, tool schema, system
prompt, benchmark tasks/environments/prompts/verifiers, sampling parameters,
budgets and seeds. The comparison receipt permits exactly one changed variable:
checkpoint or adapter bytes.

Current hosted Fleet GLM-5.2 jobs are operational shakedowns because the hosted
route does not expose those exact artifact identities. A publishable experiment
will self-host both base and intervention arms from the same pinned checkpoint
and serving image; it will not compare a hosted FP8 base to a differently served
post-trained model.

## Live-launch gate

Copy `compatibility_receipt.example.json` and prove every check on the exact
Nebius topology. The planner refuses a missing check, mutable model revision,
or mismatched dataset digest. After the gate passes, bind the sealed plan to a
pinned NeMo RL/Megatron container and site-specific Slurm or Kubernetes job.
The cluster injects `SFT_TRAIN_COMMAND`, `RL_TRAIN_COMMAND` and
`CYBER_EVAL_HOOK` as argv strings. Add `--execute` only inside the approved
scheduler allocation. Successful stage receipts make requeues idempotent.

Useful primary documentation:

- GLM-5.2 BF16 checkpoint: https://huggingface.co/zai-org/GLM-5.2
- NeMo RL model extension: https://docs.nvidia.com/nemo/rl/latest/guides/add-new-model.html
- NeMo RL LoRA: https://docs.nvidia.com/nemo/rl/latest/guides/lora.html
- Megatron Bridge MoE LoRA controls: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/peft.html
