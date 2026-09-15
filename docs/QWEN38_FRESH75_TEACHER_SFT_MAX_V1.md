# Qwen3.8 fresh-catalog teacher SFT

This arm trains only on successful stronger-teacher sessions from the frozen
50-task training split in
`configs/data/fleet-blackbox-current-study-split-20260914-v2.json`. The 17
development tasks and 8 final-test tasks never enter the corpus.

## Data receipt

- 424 stronger-teacher successes were found in the session catalog.
- 274 match an exact current task version in the training split.
- 115 sessions pass the Qwen tool-interface and task-richness checks, covering
  37 exact task versions.
- Source mix: 113 `gpt-5.6-sol`, 1 `grok-4.5`, and 1 `kimi-k3` session.
- The corpus contains 916 windows, 1,036,061 supervised tokens, and 4,725
  assistant responses.
- 106 sessions using an unproven `search_tool`/`use_tool` interface, 11 with
  opaque context compaction, and 42 with extra user/system transitions are
  excluded. No exact trajectory or window duplicate was found.
- The final wrapper is removed only when it is an exact two-message
  `submit_final_answer` suffix after a completed `submit_report`. The original
  transcript digest remains bound in every row.

The public aggregate manifest and receipt are
`configs/data/qwen38-fresh75-teacher-sft-train-v1.manifest.json` and
`configs/data/qwen38-fresh75-teacher-sft-train-v1.receipt.json`. Session IDs,
transcripts, scores, and token IDs remain private.

`cyber-post-train data-fleet-teachers` reproduces the private Parquet file from
digest-bound normalized records and success evidence without network or GPU
access. Two independent create-once builds produced the same manifest and
Parquet digests.

## Training recipe

The current successor configuration is
`configs/runs/qwen38-fresh75-teacher-sft-largest-full-v4.json`:

- Qwen3.8-27B at exact revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- two epochs, global batch 8, microbatch 1 per GPU
- learning rate `1e-5`
- 230 optimizer steps: `ceil(916 / 8) * 2`
- checkpoint every 20 steps, retaining the newest three
- one 8-GPU node at `c1` priority
- W&B run `thefleet/cyber-post-train/chris-q38-f75-max-full-v4`

The unsubmitted V1 plan was retired after its zero-GPU pinned-image preflight
found that train-only validation still tried to check an absent teacher-token
development file. V1 never created a training job or allocated a GPU.

## First V2 submission and setup diagnosis

The first V2 production submission stopped at optimizer step zero with a public
`KeyError` classification and produced no checkpoint. Its old runtime preserved
the traceback privately but did not publish the missing key or an exact setup
stage. The approved public evidence therefore cannot identify that key after the
fact, and this document does not guess one.

Two direct, clean-exit development-cluster probes then exercised the same pinned
image, model bytes, eight-rank FSDP setup, and all-rank device backload without
loading training data or creating an optimizer step:

- `chris-q38-f75-setup-probe-v1` passed through `device_ready`; its validated
  setup receipt digest is
  `520f7d397ecd2f81d459790ad385ad795e6d51aff403472e7004271d0c65b69b`.
- `chris-q38-f75-tracker-probe-v2-b5fda280` additionally exercised the exact
  production W&B initialization path with a new run identity. Its W&B receipt
  digest is
  `2e670e3ee7a429fcc7dd9a72d81fb4445f5e232a41e0cb6b7ca30d1ebfdd828c`
  and its validated setup receipt digest is
  `48c1baae3711b8236f73c8085c61486565161151a525f9aa5bc50b51bdd63d75`.

Both Pods exited zero with no restarts and were deleted after terminal receipt
validation. They ruled out a deterministic corpus-schema, W&B binding, model
file, FSDP configuration, or explicit-backload defect in the exercised setup
path. They did **not** exercise the native loop's evaluation-loader
initialization, so they were not a sufficient promotion gate for a plan without
a `dev` dataset.

V3 made that gap observable. `chris-q38-f75-max-canary-v3-6feeb90c`
reached `device_ready`, then failed before optimizer step one because
`load_eval_dataset()` unconditionally indexed `datasets["dev"]`. Its immutable
outcome-only plan intentionally contains only `datasets["train"]`. The
RayCluster, Pod, and all eight GPUs were released; V3 must never be retried.

This repeated because the earlier compiler-side train-only preflight fix and
the runtime loader fix were on different experiment branches. The Fresh75
branch contained the former but did not contain the latter, and its setup probe
ended before the loader call. Treating those partial gates as equivalent was
the operational error.

The correction is commit `a3dce55a`. The loader now returns no evaluation
dataset for a plan without `dev`, and the setup probe explicitly exercises that
branch. A CPU-only Pod in the exact pinned trainer image then ran the real
SkyRL training loop across all 16 checkpoint/evaluation combinations: 16
passed, zero failed, zero GPUs, zero restarts. The durable evidence is
`docs/evidence/qwen38-fresh75-nodev-fix-20260915.json`. V4 uses new run,
output, request, and W&B identities; neither the V2 nor V3 output is reused.

The V4 one-step canary then completed successfully with zero restarts. It
executed one finite optimizer step, wrote a matching native checkpoint and
planned-pause receipt, and released all eight GPUs. A separate zero-GPU Pod in
the pinned image reopened its sampler/trainer metadata and independently
validated the terminal and checkpoint receipts. The acceptance evidence is
`docs/evidence/qwen38-fresh75-v4-canary-accepted-20260915.json`. The exact full
230-step V4 run was submitted once at `c1` only after that gate passed; it was
queued without allocation at the evidence timestamp.

The full run later completed all 230 planned optimizer steps and consumed
2,072,122 supervised tokens. Its final `global_step_230` checkpoint is sealed
by a self-digesting manifest and a separate full-payload rehash: 33 files,
324,627,486,795 bytes, manifest-file SHA-256
`bf92e29d9f19dd589201214f25fe8d0e3b0f65de14ec08fc573217f50580a486`,
and embedded receipt SHA-256
`9542502631b47cd730ccecb5938650103ed999bf4a4a6cf7d16d9be09c317cab`.
This proves terminal training and exact checkpoint bytes. It does not yet prove
that the model can be loaded for inference or served.

The [step-230 promotion template](../configs/qualification/qwen38-fresh75-step230-promotion-v1.template.json)
froze the remaining gates before they ran. The create-once BF16 export, its
CPU integrity check, and the bounded one-GPU reload have now all passed. The
one-GPU check loaded the complete exported model, produced finite output,
executed zero optimizer steps, proved the source unchanged, and released its
GPU. The exact terminal identities and signed receipt hashes are in the
[reload acceptance evidence](evidence/qwen38-fresh75-step230-reload-accepted-20260915.json).

This proves that the final checkpoint is reloadable. It does not yet prove that
the model is served correctly. Serving still requires create-once staging,
temporary serving qualification, production registration, and fresh
base-versus-candidate parity. Any development-cluster serving test is temporary,
has a fixed lifetime, and must release all resources on success or failure; a
persistent development endpoint is forbidden.

The earlier public preparation receipt is
`docs/evidence/qwen38-fresh75-step230-promotion-prepared-20260915.json`. It is
an immutable record of the pre-execution state and does not claim that the then
in-progress export had succeeded. The later acceptance evidence above records
the completed export and reload gates without rewriting that history.

There is no held-out teacher-token loss. W&B records training loss and runtime
health. Checkpoint choice must use task outcomes on the frozen 17-task Fleet
development split.

## Evaluation handoff

After a checkpoint is sealed and exported:

1. Evaluate it on the 17 Fleet development tasks with the same OpenCode-facing
   tool contract used for the training traces.
2. Run matched base-versus-checkpoint WebExploitBench with OpenCode only.
3. Choose a checkpoint without consulting the 8 final-test tasks.
4. Evaluate the chosen checkpoint once on the 8 final-test tasks.

Every evaluation must bind the exact checkpoint receipt, task selection,
harness image/config, model-serving identity, and sampling settings.
The step-230 checkpoint is the recipe's predeclared final step. WebExploitBench
is external reporting only and cannot select this or any other checkpoint,
change the recipe, or serve as a retry signal.
