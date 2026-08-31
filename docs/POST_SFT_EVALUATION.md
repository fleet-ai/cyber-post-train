# Post-SFT evaluation handoff for `ft-run-574bd7b3`

## Decision

The intervention checkpoint is the run's **single final promoted checkpoint after the RayJob
succeeds**. Intermediate checkpoints and their development losses are diagnostics, not candidates.
This rule was fixed before viewing post-SFT WebExploitBench or Fleet-test outcomes and prevents
benchmark-driven checkpoint selection.

Do not use this handoff while the run is `RUNNING`, if its RayJob UID changes, if more than one
checkpoint is promoted, or if the final checkpoint is incomplete or lacks a verified durable
archive. The checked-in plan is
`configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json`.

## What is already frozen

- Base weights: `Qwen/Qwen3.6-27B` at
  `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, BF16.
- Active SFT run: `ft-run-574bd7b3`, RayJob UID
  `fe0256e7-ba30-470d-abd9-b148cd3cdbbd`.
- Training config SHA-256:
  `3a56db536d8918cadad8961dc450e1210488742d2981bb53f5ca4347aabcad29`.
- Trainer image:
  `fleet/skyrl-train:q36-torchgdn-6db8d0c9@sha256:ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4`.
- RayJob entrypoint SHA-256 (exact string, without a trailing newline):
  `sha256:f88435165af2bb65a41528aa7d82395431c234178289dba2c9b7fbe58a73024c`.
- Base serving: SGLang image
  `lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`,
  BF16, TP1. The live baseline reported engine version
  `0.0.0.dev0+qwen38.27b.g561c8f3`.
- Primary external harness: the same CAGE/Claude Code arm, judge, Level-0 prompt, pass@1,
  context windows, budgets, and concurrency as the base run.
- Fleet holdout: the 20 exact `test` task-version IDs in
  `configs/data/fleet-a62-task-split-v1.json`. They have zero task-lineage overlap with the 130 SFT
  train tasks or 10 SFT development tasks.

## Why the checkpoint cannot be served directly yet

The SFT trainer writes a resumable FSDP training checkpoint. It does not produce a Hugging Face
`safetensors` directory in the inference filesystem. The `fleet/<run>-step-<n>` Jobs API spelling
routes model calls to `inference.flt.build`, but that spelling alone neither converts weights nor
creates a routed inference model.

The immutable trainer image already contains SkyRL's FSDP-to-Hugging-Face exporter. At the pinned
SkyRL commit `f5bc3b78dfddfb352870d5d7430cd226e5785838`, it gathers the full FSDP state dict to rank 0,
writes with `save_pretrained(..., safe_serialization=True)`, and saves the tokenizer and corrected
model config. The active run left `hf_save_interval=0`, however, so it will not create that export.

A separate, queue-managed zero-optimizer-step export run must therefore:

1. consume the exact promoted checkpoint UUID and archive-manifest digest;
2. use the exact same trainer image, model, FSDP strategy, and eight-GPU world size;
3. set `resume_from` to the final `global_step_N`, `num_steps=N`, and
   `hf_save_interval=N+1`, which makes the pinned loop execute zero optimizer steps and take its
   final HF-export branch;
4. consolidate only model weights into BF16 Hugging Face `safetensors`;
5. preserve the base tokenizer, configuration, and chat template without modification;
6. hash every output file and verify all shards, the parameter count, and a clean `safetensors`
   load; and
7. copy that exact manifest into the separate inference filesystem under an immutable
   `/models/.../<checkpoint-uuid>` path accepted by the staging rail, using a digest-pinned stager;
   and
8. prove the source and destination manifests are byte-identical and bind the generated
   `.fleet-acceptance.json` manifest.

The checked-in offline renderer creates this request without submitting it. The request writes its
duplicate resumable checkpoint and HF export below a new run/output path; it never changes the
source checkpoint directory. The Training API permits these exact typed overrides via
`trainer.args`, but the rendered entrypoint and source checkpoint presence must be checked before
submission. Exporting still requires the same eight B300 GPUs briefly to load and gather the full
state; it is not a CPU file-copy operation.

## Offline freeze and render flow

These commands create local receipts only. They do not submit, register, stage, or evaluate
anything.

Capture the terminal run detail and checkpoint inventory from the authenticated Training API into
restricted local files. Add the SHA-256 of the exact S3 `_MANIFEST.json` body to the promoted
checkpoint observation as `archive_manifest_sha256`; the API currently reports archive presence
and size accounting, but not that content digest. The run observation must use this prompt-free
shape:

```json
{
  "schema": "fleet_training_run_observation_v1",
  "run_name": "ft-run-574bd7b3",
  "status": "succeeded",
  "rayjob_uid": "fe0256e7-ba30-470d-abd9-b148cd3cdbbd",
  "run_config_sha256": "sha256:...",
  "trainer_image": "registry/image@sha256:...",
  "entrypoint_sha256": "sha256:...",
  "latest_checkpoint_step": 318
}
```

The example step is illustrative; use the server's actual final step. Then freeze the selection:

```bash
uv run python -m training.post_sft_cli freeze \
  --plan configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json \
  --run-observation /restricted/ft-run-574bd7b3-run.json \
  --checkpoints /restricted/ft-run-574bd7b3-checkpoints.json \
  --output /restricted/ft-run-574bd7b3-selection.json
```

The promoted checkpoint row must also retain the API's exact `sfs_path` and `sfs_available` fields.
If it has already been removed from SFS, stage the archived checkpoint through the normal API and
recapture the row; the export renderer refuses to invent a path or proceed while it is unavailable.

For `ft-run-574bd7b3`, the deployed checkpoint pipeline was explicitly dry-run-only
(`apply: false`, archive disabled), so the successful run has an expected empty checkpoint API
index even though its final SFS checkpoint is complete. The plan binds that exact Argo CD
application UID, sync commit, and Helm-values digest. Use `freeze-sfs` only after proving the SFS
checkpoint has `.promoted`, `.milestone`, every expected shard-completion marker, and the run's
latest step. Its cheap path/size/mtime structural manifest must be identical before and after the
conversion, and a serialized post-conversion pass must SHA-256 every checkpoint file. This is the
second fail-closed selection leg; it is not permission to mutate the checkpoint index or archive.
Use `python -m training.post_sft_artifacts structural|full` for those read-only manifests. Run the
full pass only after conversion has ended and at low priority because the source contains roughly
302 GB of model and optimizer state.

The prepared normal-queue evidence job wraps both passes and the HF inspection without requesting
a GPU. Preview it with `evals/post_sft/scripts/submit_evidence.sh preview`; the submit mode refuses
to proceed until `ft-run-29f2bedf` is `SUCCEEDED`, refuses an existing output/job collision, and
submits suspended through `training-lq`. Do not submit it while conversion is reading the source.

```bash
uv run python -m training.post_sft_cli freeze-sfs \
  --plan configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json \
  --run-observation /restricted/ft-run-574bd7b3-run.json \
  --sfs-observation /restricted/ft-run-574bd7b3-sfs.json \
  --output /restricted/ft-run-574bd7b3-selection.json
```

First render the review-only export request:

```bash
uv run python -m training.post_sft_cli render-export \
  --plan configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json \
  --selection /restricted/ft-run-574bd7b3-selection.json \
  --request-output /restricted/ft-run-574bd7b3-export-run-config.json \
  --receipt-output /restricted/ft-run-574bd7b3-export-request-receipt.json
```

The receipt is a `submit: false` review artifact, and the separate run config is suitable for the
repository's normal `training jobs-run` preview. Do not execute that preview until the source path
exists and the source run is terminal. Before any execution, inspect the server-rendered entrypoint
and require `resume_from=.../global_step_N` and `num_steps=N`. The run must report zero optimizer
steps. A post-export staging step is still needed because training SFS and the inference `/models`
PVC are different filesystems.

For this study, the predeclared export is uniquely bound in the plan to RayJob
`ft-run-29f2bedf` (UID `51957fb6-c8c5-4e72-ab1b-8ec80e38e68b`) and output root
`/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-v1`. A read-only preflight at
`2026-08-31T16:05:07Z` proved that destination absent and found exactly that one RayJob referring
to it. The downstream receipt must repeat the exact run id, RayJob identity, trainer version and
digest-pinned image, resume path, step/save settings, collision preflight, output path, and output
file/weight/tokenizer/chat-template hashes. Any mismatch is a hard stop; do not reinterpret a
different directory as this export.

The trainer's checkpoint sidecars are not byte-identical to the pinned base. Transformers 5
rewrote `config.json`, `tokenizer.json`, and `tokenizer_config.json`: the effective 248,077-token
mapping remains identical, but seven audio tokens moved from the base tokenizer config's added-token
decoder into the serialized tokenizer JSON. Therefore the trainer output is preserved as an
immutable **raw export**, and is never served directly. Inspect it with `--allow-sidecar-drift` to
record every raw hash rather than disguising this known difference.

The inference bundle is composed weights-only: copy the raw export's safetensors shards and index,
then copy every runtime sidecar from exact base revision `6a9e13bd...`. Before accepting it:

- compare every effective token→ID mapping and core special-token ID;
- prove the seven serialization additions are contiguous, non-remapping, and already declared by
  the base tokenizer config;
- require encode/decode parity over every rendered SFT training window, the ten lineage-held-out
  Fleet prompts, all 15 WebExploitBench prompts and harness protocol strings, all five ExploitGym
  Qwen Code traces, and explicit tool/control strings;
- compare every post-training tensor key, shape, and dtype with the exact base layout; and
- require the final bundle's runtime sidecar hashes to equal the plan's base hashes and its weight
  and index hashes to equal the raw export.

Any corpus-relevant tokenizer difference is a hard stop. After these gates and inference staging
emit one digested `cyber_sft_hf_export_v1` receipt, render all paired evaluation inputs. The receipt
must separately identify raw trainer conversion, composition, and digest-pinned staging; a model
merely appearing under `/models` is not provenance. `training.post_sft_artifacts` hashes every
output file and weight shard, verifies that the index exactly names those shards, and checks every
safetensors tensor is BF16 with the base architecture's exact parameter count.

```bash
uv run python -m training.post_sft_cli render \
  --plan configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json \
  --selection /restricted/ft-run-574bd7b3-selection.json \
  --export /restricted/ft-run-574bd7b3-hf-export.json \
  --output-dir /restricted/ft-run-574bd7b3-eval-handoff
```

The renderer fails unless the export is BF16 safetensors, its checkpoint and archive identities
match the selected final checkpoint, its tokenizer/chat-template hashes match base, its converter
image and command are digest-bound, and the inference destination manifest exactly equals the SFS
export manifest. It produces:

- a post-SFT inference registration cloned from the base SGLang contract;
- a WebExploitBench config differing from base only in `model` and `run_id`;
- an exact, prompt-free 20-task Fleet holdout receipt; and
- a comparison receipt binding all three artifacts.

After one ExploitGym control image has also passed the independent immutable-image
publication/readback rail, freeze both external benchmark inputs together:

```bash
uv run python -m training.post_sft_cli render-external-benchmarks \
  --plan configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json \
  --selection /restricted/ft-run-574bd7b3-selection.json \
  --export /restricted/ft-run-574bd7b3-hf-export.json \
  --control-image /restricted/exploitgym-control-image-receipt.json \
  --output-dir /restricted/ft-run-574bd7b3-external-eval-handoff
```

The ExploitGym renderer reuses the exact frozen five tasks, Qwen Code 0.22.3,
official dynamic-flag verifiers, firewall, pass@1, and time budget. It requires a
single GHCR digest whose manifest, config, and layers were independently pulled
on a second native AMD64 worker. It deterministically counterbalances base-first
and post-SFT-first task pairs; the earlier 0/5 descriptive pilot is not treated
as the byte-identical paired baseline.

## Serving parity gate

Register only the rendered model ID (`ft-run-574bd7b3-step-<actual-final-step>`), so the Fleet
checkpoint provider spelling `fleet/ft-run-574bd7b3-step-<actual-final-step>` resolves to the same
gateway route. Registration must be queue-safe and idempotent. It must clone the base contract,
including:

- the exact SGLang image digest and command;
- BF16 model weights and TP1;
- 262,144-token context;
- FP8 E4M3 KV cache (this is runtime cache precision, not model-weight quantization);
- `trtllm_mha`, chunked-prefill and max-prefill settings;
- Qwen reasoning and tool parsers; and
- the same non-preempting `fleet-serve-low` placement policy.

Before evaluation, save the live catalog and `/server_info` projection and compare every controlled
field with base. Also require a tokenizer identity check, structured tool-call smoke, and a fixed
prompt/logit smoke. A ready Pod or a model-list entry alone is not serving parity.

## Evaluation and leakage gates

WebExploitBench remains evaluation-only and sealed. Do not inspect its base or post-SFT scores until
the selection, export, serving, and comparison receipts are frozen. Run the rendered config through
the existing CAGE path; do not use the separate Qwen Code harness arm for this primary comparison.

For Fleet, do not submit the 20 mutable task keys directly. Create one task group whose members pin
the 20 exact `eval_task_version_id` values in the rendered holdout receipt, and inspect the rendered
job before paid submission. Use `claude_code`, pass@1, the same budgets as base, and the exact
`fleet/<run>-step-<n>` model. Archive each resolved session's task-version, environment-version,
data-version, verifier, model, harness, and job identities. Any mismatch invalidates that pair; it
must never be silently replaced with the current task version.

## Remaining blockers

1. `ft-run-574bd7b3` must finish successfully and the final promoted archive must settle.
2. The rendered zero-step export request must pass review, run through the queue using the exact
   trainer image, report zero optimizer steps, and emit a verified HF export receipt.
3. The exported model must be staged and registered through the inference control plane with the
   rendered parity contract, then pass live parity checks.
4. An exact-version Fleet task group must be created from the 20-task receipt and dry-run through
   the deployed Jobs API.

Until all four are green, neither WebExploitBench nor the Fleet test holdout is ready to launch.
