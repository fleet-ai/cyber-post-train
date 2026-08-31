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
- Primary external harness: exact Qwen Code 0.22.3 at source commit
  `09825973e7d3c3fd07e17909c396aa62f48ce51f`, matching the canonical 10/110 baseline,
  with the same CAGE image, judge, Level-0 prompt, pass@1, target set, context windows, budgets,
  and concurrency.
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

A separate, queue-managed zero-optimizer-step export run was used to:

1. consume the exact promoted checkpoint UUID and archive-manifest digest;
2. use the exact same trainer image, model, FSDP strategy, and eight-GPU world size;
3. set `resume_from` to the final `global_step_N`, `num_steps=N`, and
   `hf_save_interval=N+1`, which makes the pinned loop execute zero optimizer steps and take its
   final HF-export branch;
4. consolidate model weights into Hugging Face `safetensors`; and
5. hash every output file and verify all shards, the parameter count, and a clean `safetensors`
   load.

The terminal export unexpectedly serialized all weights as FP32: three shards total
109,427,064,152 bytes, and the finalized shard headers identify every tensor as `F32`. Those bytes
are preserved as the immutable raw export. They are never relabelled BF16 and are never served.
The reviewed corrective rail must therefore:

6. run a separate CPU-only, queue-managed SFS transaction which verifies the full FP32 manifest;
7. cast every tensor to BF16 in sorted-key, fixed-size shards and reopen every shard to prove its
   raw BF16 bits equal a direct PyTorch FP32→BF16 cast;
8. reject mixed dtypes, missing/extra/remapped tensors, shape drift, non-finite source values,
   memory-bound violations, pre-existing outputs, and incomplete transactions;
9. copy the verified BF16 weights into the inference filesystem under the immutable `/models/...`
   path, composing them with exact base runtime sidecars; and
10. prove the BF16 cast and inference source/destination manifests are byte-identical and bind the
   embedded `.fleet-acceptance.json` receipt while excluding that reserved path from the served
   payload manifest.

The checked-in offline renderer creates this request without submitting it. The request writes its
duplicate resumable checkpoint and HF export below a new run/output path; it never changes the
source checkpoint directory. The Training API permits these exact typed overrides via
`trainer.args`, but the rendered entrypoint and source checkpoint presence must be checked before
submission. Exporting still requires the same eight B300 GPUs briefly to load and gather the full
state; it is not a CPU file-copy operation.

## Offline freeze and render flow

These commands create local receipts only. They do not submit, register, stage, or evaluate
anything. Every receipt/config output is an atomic, no-replace publication; use a new reviewed
path instead of overwriting any prior freeze or render artifact.

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

The prepared normal-queue evidence job wraps both passes and the raw FP32 HF inspection without
requesting a GPU. Preview v4 with `evals/post_sft/scripts/submit_evidence_v4.sh preview`; the submit
mode refuses to proceed until `ft-run-29f2bedf` is `SUCCEEDED`, refuses an existing output/job
collision, and submits suspended through `training-lq`. This evidence job must finish before the
BF16 cast job is submitted, so the cast input is bound to the complete raw and source manifests.

The immutable first attempt, `chris-cyber-qwen36-sft-evidence-v1` (UID
`777191dc-0d80-4453-a74b-609b58c6a848`), was admitted on 2026-08-31 but failed before reading or
hashing any model file: the pinned trainer image runs as UID 1000 and the root-owned
`/mnt/sfs/jobs` directory is not writable. Preserve that failed Job as evidence. The create-only
v2 successor moved the receipt under the UID-1000-owned export tree and proved that storage fix,
but its isolated ConfigMap bundle used the repository's import-heavy `training/__init__.py` without
including all of those unrelated modules. It therefore failed at Python import, again before
reading any model file. Preserve v2 unchanged as well. The create-only v3 successor mounted a
minimal package marker and read the complete source, but correctly failed when it compared the raw
export's 27,356,728,560 parameters with the frozen base's 27,781,427,952 parameters. Immutable
headers then proved the exact difference: no extra tensors or shape mismatches, and exactly 15
enumerated `mtp.*` speculative-draft tensors absent (424,699,392 parameters). Preserve v3 and its
incomplete `.partial` directory unchanged.

The create-only v4 evidence job writes to
`/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/evidence/sfs-v4/receipt`. Its model-specific gate accepts
only those exact 15 names, shapes, BF16 base dtypes, and element counts. Any additional, missing,
renamed, or reshaped tensor is fatal; generic parameter and layout validators remain strict. The
matched SGLang registration contains no speculative-decoding or draft-model argument, so these MTP
heads are inference-inert in both arms. The Job recomputes the checkpoint structure and latest-step
pointer after all source/export scans and fails if either changed. Its receipt also binds the live
Job and Pod UIDs/specs, resolved image digest, immutable ConfigMap identity, mounted code hashes,
and a programmatic check of the exact hash-pinned serving registration's runtime arguments.

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

The inference bundle is composed weights-only: copy the verified BF16 cast's safetensors shards and
index, then copy every runtime sidecar from exact base revision `6a9e13bd...`. Before accepting it:

- compare every effective token→ID mapping and core special-token ID;
- prove the seven serialization additions are contiguous, non-remapping, and already declared by
  the base tokenizer config;
- require encode/decode parity over every rendered SFT training window, the ten lineage-held-out
  Fleet prompts, all 15 WebExploitBench prompts and harness protocol strings, all five ExploitGym
  Qwen Code traces, and explicit tool/control strings;
- compare every present raw post-training tensor key and shape with the exact base layout,
  accepting only the enumerated 15-key MTP omission and recording the FP32/BF16 dtype difference;
  cast all 1,184 trained tensors and restore those 15 frozen-base BF16 auxiliary tensors; then
  compare the resulting 1,199-tensor BF16 layout exactly; and
- require the final bundle's runtime sidecar hashes to equal the plan's base hashes and its weight
  and index hashes to equal the verified BF16 cast.

Any corpus-relevant tokenizer difference is a hard stop. After these gates and inference staging
emit one digested `cyber_sft_hf_export_v1` receipt, render all paired evaluation inputs. The receipt
must separately identify raw trainer conversion, composition, and digest-pinned staging; a model
merely appearing under `/models` is not provenance. `training.post_sft_artifacts` hashes every
output file and weight shard, verifies that the index exactly names those shards, and checks the
raw export is uniformly FP32 while the cast and served outputs are uniformly BF16 with the base
architecture's exact parameter count.

The CPU cast rail is `evals/post_sft/scripts/submit_bf16_cast_v2.sh`. Preview is non-mutating.
Submission accepts only the completed evidence Job's digest-bound observation and raw full
manifest, validates the queue, and creates a suspended Job through `training-lq` using create-only
ServiceAccount, Role, RoleBinding, immutable ConfigMap, and Job operations. Its destination is
`/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/step-318-bf16-v2/global_step_318/policy`.
The embedded `.fleet-bf16-cast-acceptance.json` proves 1,184 direct FP32→BF16 casts and 15
bit-identical copies from the exact frozen base. The latter are explicitly **frozen base
auxiliary-head restoration**, not trained weights. Before copying, the Job hashes the complete
base directory and requires its safetensor-shard digest to equal the signed model lock; it hashes
the complete base again after conversion and fails unless the two full manifests are identical.
After atomic
promotion the same Job publishes a separate terminal evidence directory containing that receipt,
the full post-marker manifest, and a digest-bound `COMPLETE.json`. Merely finding the destination
directory is not success. Every written weight shard, index, and copied sidecar is flushed before
the directory-level atomic promotion.

```bash
bash evals/post_sft/scripts/submit_bf16_cast_v2.sh preview

# Only after the evidence Job is Complete and these are its exact immutable outputs:
bash evals/post_sft/scripts/submit_bf16_cast_v2.sh submit \
  /restricted/ft-run-574bd7b3-observation.json \
  /restricted/ft-run-574bd7b3-raw-export-full-manifest.json
```

After the cast Job is Complete, retrieve `cast-receipt.json` and
`cast-full-manifest.json` from
`/mnt/sfs/exports/cyber-sft/ft-run-574bd7b3/evidence/bf16-cast-v2/receipt/`.
The adjacent
`COMPLETE.json` must bind both files. These two files are inputs to staging; the destination path
or a successful Pod status alone is insufficient.

The CPU-only inference staging rail is
`evals/post_sft/scripts/submit_inference_stage.sh`. Its preview performs server-side validation and
creates nothing. Submission remains blocked until the exact BF16 cast Job succeeds. It streams
the verified BF16 cast through the internal read-only filebrowser transport, verifies
every path, size, and SHA-256, rejects symlinks and unsafe ZIP paths, composes post-training weights
and index with the exact base runtime sidecars, and re-runs BF16/layout/parameter/hash inspection.
The final model path must be absent or contain the exact already-committed transaction. The
acceptance receipt is written at the reserved `.fleet-acceptance.json` path inside the verified
partial directory. The receipt declares that one—and only that one—path excluded from the served
payload manifest. The complete directory is then committed with one Linux `RENAME_NOREPLACE`;
there is no second receipt promotion that can strand an unaccepted model. A destination that wins
the race still causes a hard failure and is never replaced.

Retries have three explicit states. An exact final directory with a valid embedded receipt and
matching payload is a completed idempotent recovery. A complete partial directory with the exact
receipt can be validated and promoted without downloading again. An incomplete/corrupt partial,
an unaccepted final, or simultaneous partial and final paths is a hard stop requiring review; the
rail never deletes or repairs those states automatically.

```bash
bash evals/post_sft/scripts/submit_inference_stage.sh submit \
  /restricted/ft-run-574bd7b3-observation.json \
  /restricted/ft-run-574bd7b3-bf16-cast-receipt.json \
  /restricted/ft-run-574bd7b3-bf16-cast-full-manifest.json
```

The frozen plan records the reviewed SHA-256 map for every code file placed in the ConfigMap; the
stage input copies that map and binds it with its own digest. The ConfigMap is immutable, and the
submission rail uses create-only operations for the versioned ServiceAccount, Role, RoleBinding,
ConfigMap, and Job. Any pre-existing object or concurrent creation fails without replacement. Both
preview and submission use server-side create dry-runs.

The staging receipt records runtime provenance obtained from the Kubernetes API: Job and Pod UIDs,
resourceVersions and spec digests; the exact anchored container imageID; ConfigMap UID and
resourceVersion; and SHA-256 for every mounted code file plus the canonical stage-input bytes. A
read-only, staging-specific service account may read its fixed Job and ConfigMap and the running
Pod; it has no mutation permissions. Kubernetes cannot grant `get` on the controller-generated Pod
name before that name exists, so Pod `get` is the documented namespace-wide exception. Runtime
still accepts only its downward-API Pod UID owned by the exact Job UID. The assembler validates all
of this provenance against the stage input before accepting the staging receipt.

Once staging completes, assemble the final receipt from the exact immutable inputs. Do not hand-edit
the aggregate receipt. The read-only export observer has two phases because KubeRay removes trainer
resources after a terminal run. While the export is admitted, capture its RayCluster, resolved head
Pod, submitter Job, and exact submitter Pod. Their controller UIDs preserve the complete RayJob →
RayCluster → head-Pod and RayJob → submitter-Job → submitter-Pod ownership chains:

```bash
bash evals/post_sft/scripts/collect_export_run_observation.sh runtime-core \
  /restricted/ft-run-29f2bedf-runtime \
  /restricted/ft-run-574bd7b3-export-request-receipt.json

bash evals/post_sft/scripts/collect_export_run_observation.sh runtime-submitter \
  /restricted/ft-run-29f2bedf-runtime \
  /restricted/ft-run-574bd7b3-export-request-receipt.json
```

After the exact RayJob is `SUCCEEDED`, immediately capture the terminal RayJob, retained
submitter/driver log, and Jobs API run/config/metrics record before controller cleanup:

```bash
FLEET_TRAINING_API_TOKEN="$FLEET_TRAINING_API_TOKEN" \
  bash evals/post_sft/scripts/collect_export_run_observation.sh terminal-capture \
  /restricted/ft-run-29f2bedf-runtime \
  /restricted/ft-run-574bd7b3-export-request-receipt.json
```

Once the full checkpoint selection receipt is available, finalize the observation from those
immutable terminal inputs and the earlier runtime snapshots:

```bash
bash evals/post_sft/scripts/collect_export_run_observation.sh terminal \
  /restricted/ft-run-29f2bedf-runtime \
  /restricted/ft-run-574bd7b3-export-request-receipt.json \
  /restricted/ft-run-29f2bedf-terminal-observation.json
```

All four modes are read-only against Kubernetes and Fleet Train. Splitting terminal capture from
receipt assembly allows the ephemeral controller objects and complete log to be frozen immediately,
while the full source checkpoint manifest is produced later by the low-priority evidence job. The
finalizer still requires the complete cryptographic selection and export-request receipts; the
minimal identity file used for urgent runtime capture is not sufficient. The collector verifies the exact
RayJob name and UID, terminal status, queue, entrypoint, head/worker/submitter images, commands,
service accounts, replicas and resources, runtime ownership chain, and container-runtime-resolved
image digest. The live API identity is its top-level `id` plus `config.run_id`. The plan explicitly
freezes the server's stored-config normalization contract: `name`, `run_id`, submitter identity,
node pool, Fleet team selection, nullable data filters, and trainer `command`/`env` defaults. The
collector reconstructs that complete normalized value and requires the RayJob, runtime RayCluster,
and Jobs API to carry it exactly; unreviewed fields or different defaults fail closed. Empty metric
arrays are not zero-step proof. The complete exact-
submitter log must contain the entrypoint and terminal-success marker; any optimizer event,
`Step N:`, JSON step, current/latest/global step beyond the resume boundary, or nonzero API
optimizer field is fatal. The bearer token is passed to curl through a mode-0600 temporary config
file, never a process argument. Only log/API digests and parsed event summaries enter the receipt.

The resulting `cyber_sft_zero_step_export_run_observation_v1` binds the exact predeclared run name,
run id, RayJob UID, trainer version and image; `terminal_status: "SUCCEEDED"`; the exact resume path,
`num_steps`, `hf_save_interval`, and output path; `optimizer_steps: 0`; and the zero-step
export-request digest. Its embedded digest covers the full observation. The proof is the conjunction
of the pinned image, actual command's equal resume/final boundary, terminal success, the complete
submitter log's parsed absence of optimizer events, and bounded API progress fields—not an
operator-authored `optimizer_steps: 0` assertion or empty metric array.

The other inputs are produced directly by the existing gates: `observation.json` from the evidence
Job, `cast-receipt.json` and `cast-full-manifest.json` from the cast terminal evidence directory,
the exact `stage-input.json` stored in the staging ConfigMap, and the embedded
`.fleet-acceptance.json` inside the atomically promoted inference directory. The assembler
validates every embedded digest and cross-checks the checkpoint, run, paths, raw weights, composed
weights, tokenizer, chat template, configuration, runtime sidecars, staging image, staging command,
and zero-step result before it writes anything.

```bash
uv run python -m training.post_sft_cli assemble-export \
  --plan configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json \
  --selection /restricted/ft-run-574bd7b3-selection.json \
  --export-request /restricted/ft-run-574bd7b3-export-request-receipt.json \
  --export-run-observation /restricted/ft-run-29f2bedf-terminal-observation.json \
  --export-observation /restricted/ft-run-574bd7b3-observation.json \
  --cast-receipt /restricted/ft-run-574bd7b3-bf16-cast-receipt.json \
  --cast-full-manifest /restricted/ft-run-574bd7b3-bf16-cast-full-manifest.json \
  --stage-input /restricted/ft-run-574bd7b3-stage-input.json \
  --staging-receipt /restricted/ft-run-574bd7b3-staging-acceptance.json \
  --output /restricted/ft-run-574bd7b3-hf-export.json
```

The command is offline and non-mutating except for its private, atomic output file. A missing,
unsigned, stale, colliding, or cross-run input is a hard failure. Its output is the only
`cyber_sft_hf_export_v1` receipt that should be passed to the downstream renderers.
Its `--output` path is immutable publication: an existing file, directory, valid symlink, or broken
symlink is refused. An atomic no-replace link closes the race between the initial collision check
and publication, so a concurrently created path is never overwritten.

After staging, `evals/post_sft/scripts/submit_registration.sh` validates the rendered serving
receipt and submits one idempotent, priority-zero registration Job through `training-lq`. It refuses
to create a replacement Job and is gated on the exact staging Job's completion. The post-SFT route
must then become Ready and pass the live parity checks below.

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
- a WebExploitBench protocol preserving exact Qwen Code 0.22.3 and a fail-closed paired-identity
  receipt spanning the harness, CAGE/runtime image, prompt/verifier revisions, all 15 targets,
  both request/time budgets, judge, tokenizer, and normalized serving runtime;
- an exact, prompt-free 20-task Fleet holdout receipt; and
- a comparison receipt binding those artifacts.

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

Before evaluation, capture the exact live Kubernetes `InferenceModel` UID, generation, full spec,
observed generation, and Ready status for both arms. The complete live spec must equal the rendered
registration, including source/serving paths, revision, image, command, arguments, environment,
placement, and scaling. Also require `/model_info`, `/server_info`, tokenizer identity, structured
tool-call, and fixed prompt/logit checks. A ready Pod or model-list entry alone is not parity.

## Evaluation and leakage gates

WebExploitBench remains evaluation-only and sealed. Do not inspect its base or post-SFT scores until
the selection, export, serving, and comparison receipts are frozen. Run the rendered config through
the exact Qwen Code 0.22.3 CAGE path used by the canonical 10/110 baseline. If that exact protocol
cannot be reproduced, do not compare against 10/110; run a newly matched base-plus-post pair.
The sole scored launch rail is `evals/webexploitbench/scripts/launch_post_sft_paired.sh`: it
revalidates the digested paired identity against the exact config and protocol at the launch
boundary, requires export/evidence/staging completion and a Ready post-SFT route, and passes that
same receipt into the CAGE run gate. Its preview is non-mutating. Both preview and submit also
require two post-registration receipts:

- `cyber_post_sft_registration_completion_v1` binds the exact completed registration Job and its
  controller-owned Pod, resolved image digest, reviewed Job/Pod execution projections, immutable
  ConfigMap UID/resourceVersion/content and frozen-plan code hashes, the API-returned full
  registration spec, and the exported weight-manifest digest. Extra containers, init containers,
  volumes, mounts, environment, resources, or storage bindings fail closed.
- `webexploitbench_post_sft_live_parity_v1` binds that completion receipt and paired-identity
  receipt to the two live Kubernetes serving objects plus `/model_info` and `/server_info`. It
  requires the exact SGLang image/version, CR-declared BF16 TP1, complete controlled server fields,
  byte-identical complete serving non-weight file manifests, the frozen tokenizer/config/template
  hashes, successful
  structured-tool calls on both revisions, and finite deterministic fixed-prompt logit probes.

The only permitted serving-artifact difference is the weight-manifest digest. The full top-level
non-weight file set is checked: unknown files fail closed. The base-only
Base-only `.gitattributes`, `LICENSE`, `README.md`, and
`.cyber-post-train-lock.json`, plus each arm's `.fleet-acceptance.json`, are explicitly recorded
with hashes and reviewed non-serving reasons; no implicit exclusion is permitted. A Ready replica without
these receipts is not launchable. Before any CAGE preparation or build, and again immediately at
the paid `cage run` boundary, the rail refuses an existing run root at
`examples/agent_pentest_bench/.cage_runs/qwen_code:local-openai-compatible:stateless/<run-id>`.
Retries therefore need a newly reviewed run ID rather than appending to prior results.

These receipts are built directly from production APIs, not edited or staged through raw response
files. The registration collector reads the immutable Job, its sole retained Pod, immutable
ConfigMap, and one-line API result in memory. It reads logs from that exact Pod name and rechecks its
UID afterward rather than resolving logs through a Job selector. It recovers the weight-manifest
digest from the exact export receipt.

```bash
uv run python -m evals.webexploitbench.post_sft_evidence capture-registration \
  --paired-identity /restricted/webexploitbench-paired-identity-receipt.json \
  --serving-receipt /restricted/serving-registration-receipt.json \
  --export-receipt /restricted/ft-run-574bd7b3-hf-export.json \
  --output /restricted/registration-completion.json
```

The command rejects any provenance, code, mounted receipt, full returned spec, model, revision, or
completion mismatch. Its output is private, digest-bound, and published with atomic no-replace.

The post-SFT artifact identity comes only from the already validated export/staging receipt. The
base identity comes from one digest-pinned, read-only PVC inspector Job. Its reviewed execution
contract and code hashes are external inputs frozen in the post-SFT plan; the producer cannot define
its own expected values. It validates the historical checkpoint lock, hashes every shard and every
top-level non-weight file, loads every safetensors header, and binds the exact Job/Pod/image,
immutable ConfigMap, service account, and read-only PVC provenance. Preview and later collect it
without replacing output:

```bash
bash evals/post_sft/scripts/submit_base_artifact_inspection.sh preview
# Submit only after review; this is a CPU-only, read-only Job.
bash evals/post_sft/scripts/submit_base_artifact_inspection.sh submit
bash evals/post_sft/scripts/submit_base_artifact_inspection.sh collect \
  /restricted/base-artifact-receipt.json
```

Use that receipt, the frozen tokenizer-equivalence receipt, exact registrations, and export receipt
to capture both routes. The collector reads both live Kubernetes CRs itself:

```bash
FLEET_API_KEY="$FLEET_API_KEY" uv run python \
  -m evals.webexploitbench.post_sft_evidence capture-live-parity \
  --paired-identity /restricted/webexploitbench-paired-identity-receipt.json \
  --registration-completion /restricted/registration-completion.json \
  --base-registration evals/webexploitbench/serving/qwen36-27b-6a9e13bd-registration.json \
  --serving-receipt /restricted/serving-registration-receipt.json \
  --base-artifact-receipt /restricted/base-artifact-receipt.json \
  --export-receipt /restricted/ft-run-574bd7b3-hf-export.json \
  --tokenizer-probe docs/evidence/post_sft/2026-08-31-tokenizer-equivalence.json \
  --output /restricted/webexploitbench-live-parity.json
```

The tool smoke disables thinking explicitly and is temperature-zero/bounded, matching the live
Qwen/SGLang behavior. The fixed log-probability probe is repeated twice per arm. Raw response and
reasoning text remain in memory only; the receipt contains projections and digests. There is no
production CLI for persisting or replaying raw observations.

Finally pass the two generated paths to `launch_post_sft_paired.sh` as arguments five and six.
The launcher revalidates both producers' exact Job specs, controller-owned Pod UIDs/specs/resolved
images, immutable ConfigMap UIDs/resourceVersions/content, and both live CR
UID/generation/spec/status values before setup and again at the paid boundary. It then creates an
atomic, persistent launch claim
before CAGE can run, so concurrent launchers cannot both pass. A failed paid attempt requires a new
reviewed run ID rather than reusing or appending to its run root.

For Fleet, do not submit the 20 mutable task keys directly. Create one task group whose members pin
the 20 exact `eval_task_version_id` values in the rendered holdout receipt, and inspect the rendered
job before paid submission. Use Qwen Code 0.22.3, pass@1, the same budgets as base, and the exact
`fleet/<run>-step-<n>` model. Archive each resolved session's task-version, environment-version,
data-version, verifier, model, harness, and job identities. Any mismatch invalidates that pair; it
must never be silently replaced with the current task version.

## Remaining blockers

1. The queued zero-step export request must run through the queue using the exact
   trainer image, report zero optimizer steps, and emit a verified HF export receipt.
2. The exported model must be staged and registered through the inference control plane with the
   rendered parity contract, then pass live parity checks.
3. The WebExploitBench Qwen Code 0.22.3 paired-identity receipt must validate against the live
   post-SFT serving registration.
4. An exact-version Fleet task group must be created from the 20-task receipt and dry-run through
   the deployed Jobs API.

Until all four are green, neither WebExploitBench nor the Fleet test holdout is ready to launch.
