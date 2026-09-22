# Prod9: staged one-create gates

This document describes the fresh prod9 source rail. It does **not** create a
Kubernetes object, allocate a GPU, or inspect private task rows. A server
preview validates an object and discards it; only a later explicit call to one
of the bounded create functions can create an object.

## What is now implemented

The fresh prod9-only source contains three linked stages:

1. `training.skyrl_prod9_training.stage_rebind` runs only in a zero-GPU Job
   with the SFS mount. It reads the earlier input package, creates the fresh
   prod9 data package, and emits a small receipt containing paths, counts, and
   hashes—not task rows.
2. `training.skyrl_prod9_training.preflight` runs in a separate zero-GPU Job
   from the exact training image. It checks the fresh runtime, token-safe
   compaction, output conditions, and the new W&B run identity.
3. `training.skyrl_prod9_direct.create_once` is the only GPU create path. It
   accepts just the sealed receipts from the first two stages, fresh previews,
   a live cleanup observer, absence checks, W&B absence, and a fresh
   all-namespace capacity proof that includes the planned one node/eight GPUs.
   It records an intent before one create and refuses every retry.

All three root Kubernetes objects are rendered with
`metadata.annotations["fleet.ai/failure-alerts"] == "off"`. The old prod8
rail rejects the fresh prod9 runtime and cannot be used as a fallback.

## Required live sequence after review

No live operation is authorized merely because these functions exist. An
authorized operator must execute the following sequence with fresh evidence:

1. Run the offline preparation check against a sanitized prod9 manifest. It
   confirms the sealed source, identity, one-node/eight-GPU plan, and the
   native 262,144-token context contract without reading data rows.
2. Server-preview the zero-GPU SFS rebind Job in development and production.
   Verify the root alert opt-out, image, SFS mount, non-root user, and zero GPU
   request. Arm the exact-UID cleanup observer, create the Job once, then
   require its receipt and release evidence.
3. Server-preview and create the zero-GPU exact-image preflight Job with the
   same root-annotation and observer rules. Require its exact receipt and
   release evidence.
4. Server-preview the fresh root RayJob in development and production. Verify
   its root alert opt-out, one-node/eight-GPU shape, `c1`/`q1` priority, exact
   image, and fresh prod9 bundle.
5. Immediately before the one GPU create, recheck all prod9 names and output
   paths in both Kubernetes clusters and the Jobs API, prove the W&B run ID is
   unused, take the all-namespace capacity census, and confirm the cleanup
   observer is still alive. Record the no-retry intent, then create exactly
   once.

Every create intent and creator handoff is stored under the single durable
root
`/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/prod9-create-once-v1`.
This is inside the existing `launch-controls` tree proven writable by the
pinned UID 1000/GID 100 runtime; a non-root process cannot safely create the
old root-level child under `/mnt/sfs/jobs`. The exact operation child is
computed from the complete sealed stage, training plan, or reload
specification; it is never chosen by the caller. A second process using a new
working directory therefore sees the same journal and must reconcile the
original object instead of creating another Job or RayJob.

`live_create_is_available()` is an environment check, not a source-code
feature flag. It returns true only when the current process is UID 1000/GID
100 and the exact canonical create-once root already exists, is owned by that
identity, and is readable, writable, and searchable. An operator laptop with
no `/mnt/sfs` mount therefore cannot authorize a create merely because it has
checked out this code.

## Smallest cluster-native controller/observer

The current create rail still uses a same-host process check for its cleanup
observer, so it cannot be driven from a laptop without `/mnt/sfs`. The smallest
safe successor is one bounded, zero-GPU Kubernetes Job that runs the existing
authorization, one-create, and cleanup-observer code inside the pinned image.
It should not be a general cluster operator.

That Job must have the root annotation `fleet.ai/failure-alerts: "off"`, use
`c1`/`q1`, request zero GPUs, run as UID 1000/GID 100, use `backoffLimit: 0`,
and have a fixed active deadline. It must mount the shared SFS claim twice:

1. all of `/mnt/sfs` read-only, for the sealed inputs and evidence; and
2. only `jobs/chris-q38-study-corpora-v1/launch-controls` read-write at
   `/controls`, using the PVC `subPath` mechanism already validated by the SFT
   CPU-control path.

The controller then maps the canonical create-once root to
`/controls/prod9-create-once-v1`, atomically creates the absent operation
child, writes the armed observer receipt before the target create, records the
returned target UID exactly once, observes only that UID, releases only that
owned object if the documented limit is reached, and writes a terminal result
before exiting. It must use the Jobs API for the target create and cleanup; do
not give it broad Kubernetes create/delete permission. A small sanitized
termination receipt lets an operator collect the result without reading SFS.

This document records the design boundary only. The remote receipt handshake
and same-host liveness replacement are not implemented yet, so the cluster
controller must not be submitted and the prod9 live-create gate remains closed
on an operator laptop.

If any gate fails, preserve only its sanitized receipt and release only the
exact owned object. Do not retry a create after an uncertain response, reuse
prod8, or copy private data through the operator machine.

## Terminal acceptance

`NATIVE_TRAINING_COMPLETE.json` only says that the native training process
returned. It is **not** permission to call the resulting checkpoint valid,
evaluate it, serve it, or start a successor from it.

The only terminal acceptance marker is
`/mnt/sfs/jobs/chris-q38-rlreward-prod9/ACCEPTED.json`. It may be written once
by `training.skyrl_prod9_hardening.accept_terminal`; no shell command or
hand-written receipt is a substitute. The historical `skyrl_posttrain` module
can seal/export a checkpoint, but it cannot by itself accept prod9. The fresh
prod9 function accepts only these fixed paths for the exact planned final step:

| Evidence | Required fixed path | What it proves |
| --- | --- | --- |
| Checkpoint seal | `.../checkpoint-seals-v1/step-<final-step>.json` | The final checkpoint is complete, changed model parameters, and has not changed since it was sealed. |
| BF16 export | `.../hf-export-step<final-step>-v1/EXPORT.json` | Every model tensor and required sidecar was rebuilt and re-opened as BF16 from that seal. |
| Training creator handoff | `/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls/prod9-create-once-v1/training-<plan-digest>/TRAINING_OBSERVER_ARMED.json.created.json` | The cleanup observer and the one create call agreed on the exact original RayJob UID, name, plan, and rendered manifest. |
| Training cleanup observer | `.../TRAINING_OBSERVER_RESULT.json` | The original eight-GPU training RayJob succeeded without a restart, carried the exact native completion receipt used by the checkpoint seal, and its RayJob, Ray cluster, workload, Pod, and all eight GPUs were released. |
| One-GPU reload check | `...-p<final-step>-reload-v1/GPU_CHECK.json` | The exact export loads with the model configuration and tokenizer, produces finite output, and performs no optimizer update. |
| Reload cleanup observer | `...-p<final-step>-reload-v1/OBSERVER_RESULT.json` | The exact one-GPU reload RayJob succeeded without a restart and its RayJob, Ray cluster, workload, Pod, and GPU allocation were all released. |

The reload Job is a separate, future one-GPU operation. Its final server
preview must prove the root `RayJob` annotation
`fleet.ai/failure-alerts: "off"`, the fixed `c1`/`q1` priority, and exactly one
GPU before it may be created. Its observer must be armed against the exact
RayJob name and manifest before creation. The acceptance function rejects a
missing or changed seal, a non-BF16 export, a reload that did not use exactly
one GPU, a failed/restarted reload, or any unreleased resource. It also rejects
all alternate file paths, so an old receipt cannot be substituted for prod9.

## Reasoning and long-context safety

This RL run does not ingest teacher reasoning. It trains only on actions and
working-memory summaries generated by the student during its own rollout. A
summary is a normal, separately recorded model action: it is used to continue
the same task and is not presented as an external answer key.

The hard Qwen context envelope is 262,144 tokens, and the complete episode may
generate at most 4,194,304 response tokens across all turns. Before an ordinary
action
would cross the 163,840-token compaction trigger, the model receives the same
visible conversation plus a visible request to write a short working summary.
That summary is capped at 8,192 tokens. The next prompt contains the original
task and that student-written summary; it is not an opaque background rewrite.
The summary generation remains its own rollout step, while the real task reward
is placed only on the final step. A later action is rejected if its rendered
prompt plus its allowed output cannot fit the 262,144-token envelope. This is
what lets a long episode continue without pretending an overflowed transcript
was still available to the model.

If a future SFT data lane uses reasoning, it may contain only reasoning the
model was explicitly allowed to see as an assistant message. It must never
copy a teacher's hidden reasoning, hidden analysis, private prompt, trace,
flag, credential, answer, or sealed score into a training target, W&B record,
or public receipt. Those inputs remain private even when the corresponding
task outcome is useful for training.

The fresh prod9 bundle already binds
`training.skyrl_prod9_rollout.Generator`, which constructs
`training.skyrl_prod9_hardening.Recorder` directly rather than using the frozen
prod8 recorder. It treats
`tool_result_chars` only as an input-size limit. It does **not** assume that
one character equals one model token. Before a tool result becomes part of the
next model prompt, the recorder tokenizes it with the exact Qwen tokenizer and
renders the full next prompt with the exact chat template. If the next action
would need a summary, it also proves that the summary prompt and its reserved
output fit in the configured context. If either does not fit, the episode ends
with a declared context-limit result; it does not silently cut off or discard
text.
