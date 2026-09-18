# Training

Start with [the training guide](../docs/TRAINING.md) for configurable SFT:
prepare → CPU preflight → API preview → explicit submit → status.
Use [the repository README](../README.md) for installation and qualification status.

`sft.py` compiles the editable configuration; `sft_runtime.py` uses the pinned
SkyRL trainer. Model files, tokenizer, train/dev data, target masks, W&B identity,
batch size, learning rate, stopping rule and checkpoint policy are bound before
submission. Never interpret a historical config as a currently supported recipe.

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

Use [cluster operations](../docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md) and the
[training skill](../skills/cyber-train-operator/SKILL.md) before paid operations.
Historical model-specific reports and immutable run configs remain provenance;
they do not override today's API contract, resource budget or user authorization.

## Copying an accepted model to inference storage

`model_stage.py` supports `cyber_inference_model_stream_stage_plan_v2` for a
new accepted BF16 export. Unlike the historical v1 plan, v2 does not clone a
particular old model registration. It binds the exact export/GPU-check receipts,
their full file inventory, model revision and optimizer step, then streams and
rehashes files into `/models/chris-autoresearch/<owned-destination>`. Publication
is atomic and never replaces another directory. An identical existing result
can be independently read back; corrupted or incomplete copies are not accepted.

The renderer produces an immutable ConfigMap and a c1 CPU-only inference Job
with zero retries and a fixed deadline. Inference storage lives on a different
PVC from training SFS, which is why the worker runs there. The caller must record
its intent before creation, reconcile uncertain creates, monitor exact UIDs and
count real failures even outside the training namespace. The stage does not
register or resume a model, allocate GPUs, train, or evaluate any task.

Historical v1 plans and evidence are unchanged and must be replayed using their
original pinned source. Updating this implementation requires a newly digested
plan; never rewrite an old plan's source hash in place.
