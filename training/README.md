# Training

Start with [the training guide](../docs/TRAINING.md) for configurable SFT:
prepare → CPU preflight → API preview → explicit submit → status.
Use [the repository README](../README.md) for installation and qualification status.

`sft.py` compiles the editable configuration; `sft_runtime.py` uses the pinned
SkyRL trainer. Model files, tokenizer, train/dev data, target masks, W&B identity,
batch size, learning rate, scheduler/warmup pair, stopping rule and checkpoint
policy are bound before submission. Omitting both schedule fields preserves the
legacy constant/no-warmup path; new plans may explicitly select
`constant_with_warmup` with ratio `0`, or `cosine` with a positive ratio such as
`0.05`. Never interpret a historical config as a currently supported recipe.

## Data utilities

The existing read-only export and normalization commands remain available:

```sh
uv run python -m training export --job-id <exact-source-job-uuid> \
  --output /private/data/raw-sessions.jsonl
uv run python -m training normalize --input /private/data/raw-sessions.jsonl \
  --output-dir /private/data/normalized
```

Keep raw transcripts and normalized messages private. Export only authorized
Fleet sessions, never external benchmark material. Split task families across
all their versions **before** selecting demonstrations or generating windows.
Only training-split verified successes are SFT targets; held-out data never
enters SFT, preference pairs or online-RL prompts.

Export uses credential-free HTTPS endpoints, refuses redirects and retries only
transient GET failures (at most five attempts). It prints counts and safe error
categories, never transcript/error bodies. Files and their digest manifests are
private; normalization also redacts injected temporary AWS session tokens.

Normalization is not tokenization. Dense SFT preparation must account for each
eligible assistant target exactly once, mask copied context and tool observations,
and record excluded overlong targets. The old fixed-five-window corpora are
historical treatments, not the new dense policy.

## Backend and evidence boundaries

- SkyRL Qwen full-weight SFT has historical real-run evidence. Changes to this
  consolidated runtime still need their own native-image and real-run checks.
- Full GLM-5.3 is not GLM-5.3-Flash. Its loader, memory requirements and distributed
  training/checkpoint path need separate qualification.
- Miles and online RL integration are tracked in
  [the consolidation checklist](../docs/CONSOLIDATION.md). Do not use old typed
  launch templates or model-name substitutions as evidence of support.
- RL needs exact task/environment/verifier/tool identities, a useful interaction
  horizon and a canary that acquires genuine reward. An all-zero, truncated run
  is not proof of learning or model incapability.

Miles may start from either the exact frozen base conversion or an accepted SFT
BF16 export. For SFT→RL, add exact SFS `export.path` and `gpu_check.path`
references, separately hash-bound local `snapshot` copies of both receipts, plus
`sft_source.plan_sha256` and `sft_source.checkpoint_receipt_sha256` to the model
mapping used by both conversion and RL, together with the exact
`sft_source.checkpoint_manifest_sha256`,
`sft_source.export_code_sha256` and `sft_source.checker_sha256` producer pins.
Point `model.root` and the freshly
prepared RL data at that export directory, then convert it to a new create-once
native checkpoint. Local preparation validates the snapshots and pins only the
SFS runtime paths into the plan. Run the zero-GPU preflight where SFS is mounted:
it deeply reopens the export and requires its exact base revision, tensor layout
and tokenizer/runtime sidecars, plus a matching one-GPU, zero-update reload
receipt from the pinned checker. Submission requires that exact preflight, so an
unavailable or changed runtime export is rejected before GPU allocation. Rebuild
RL data because its private episode bindings include the initial-policy root and
run ID.
The existing production promoter remains deliberately bound to RL-from-base;
an SFT→RL arm must first earn its own reward/update/reload dev evidence and then
freeze a distinct production candidate. External benchmarks remain post-training
only and cannot select the SFT export, RL recipe, checkpoint, or retry.

Current teacher SFT→RL handoff (2026-09-13): the accepted step-186 stronger-
teacher export passed a teacher-only stage verifier and the exact zero-GPU Miles
CPU preflight. The compiled next operation is one create-once, one-worker,
eight-GPU HF→Miles conversion; it is prepared but intentionally not queued while
RL-from-base is the critical path. No native checkpoint, all-rank reload, RL
training or evaluation has started for this arm. After RL-from-base releases its
allocation, repeat the exact duplicate, output, capacity and authority gates
before that single conversion. The sanitized immutable evidence is
[`2026-09-13-teacher-sft-miles-preflight-v1.json`](../docs/evidence/qwen38-study/2026-09-13-teacher-sft-miles-preflight-v1.json).

Use [cluster operations](../docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md) and the
[training skill](../skills/cyber-train-operator/SKILL.md) before paid operations.
Historical model-specific reports and immutable run configs remain provenance;
they do not override today's API contract, resource budget or user authorization.
