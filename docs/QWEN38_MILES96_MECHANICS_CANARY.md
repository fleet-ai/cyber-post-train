# Qwen3.8 96K Miles mechanics canary

This is a small operational check for the maintained Fleet Miles trainer.  It
answers one practical question before larger RL work: can Qwen3.8-27B use a
real Fleet task reward to make one update, save the result, and load that saved
model again?

It is **not** a replacement for the 262K SkyRL path.  This Miles recipe has a
98,304-token context limit and does not compress old conversation turns.  A
long task therefore stops when it fills its context window.  The separate
SkyRL `prod9` rail remains the path for long cyber episodes with compaction.

## What is reused

The canary uses Fleet's maintained source, rather than the older custom cyber
Miles adapter:

| Item | Bound value |
| --- | --- |
| Theseus source used to build the image | `224d6b81cb698f4785fb16123314583b981493f3` |
| Trainer image | `miles-trainer@sha256:ee273bee346ad8e1cea63d18026b2bc5703bbd2e14d3c65efbbd74f19854874a` |
| FTI version | `0.10.9` |
| Miles source | `9e178ca16839b0600155f3927f57ce0670b8f453` |
| Recipe | `qwen3.8-27b`: one node, eight GPUs, TP4/CP2, 96K context |
| Historical mechanics reference | `fleet-ai/dataminer_v2@10afa8d064bb3dd1c11c50768590e432dfa69097`, recipe `v004` |

The image above is the built 0.10.9 image.  Theseus 0.10.10 exists as source,
but its own changelog says no image was built for that version, so this canary
must not claim 0.10.10.  The runtime checks the installed FTI version and the
exact SHA-256 of `run_fleet.py`, the V1 task-session implementation, and the V1
rollout implementation before it writes a training row or starts the trainer.
It also checks the exact Miles inference-rollout, HTTP-client, and Megatron
checkpoint/export source files plus the maintained recipe shape.  It calls
`python -m fti.trainers.miles.run_fleet` directly.  It does not call the older
`/opt/fleet/run.sh` route, which would otherwise select an old shared model
directory when running outside a FleetJob mount.

The historical `v004` result proves only the broad one-node TP4/CP2 96K
mechanics shape.  This maintained recipe uses 49,152 train tokens per GPU,
offloads the whole trainer between phases, and deliberately does **not** enable
optimizer CPU offload.  The pinned Theseus source describes that exact stack as
unrun, so this canary is its first qualification; it must not be described as a
configuration already proven by `v004`.

The pinned Miles W&B helper is also bound by source hash.  Its primary helper
normally discards the parsed run ID and lets W&B choose another one.  The
runtime patch instead makes the primary create the exact fresh ID checked by
the launcher with `resume=never`.  Miles's secondary processes may use
`resume=allow` only to join that already-live distributed run; they must carry
the same exact ID.  They cannot use that setting to revive a historical run.
The exact ID and `resume=never` policy are forwarded explicitly to the inner
Ray runtime.  The API key is never written into the command or bundle: the
reviewed root RayJob injects the required `wandb-api` Secret into its sole GPU
Pod, and every W&B initializer fails closed if that credential is absent.
The terminal receipt records these two distinct facts as primary
`never` and secondary `allow_same_live_id`; it does not describe the whole
distributed run with the primary's policy alone.

## Exact, deliberately small job

- One unprivileged eight-GPU node at `c1` priority.  The maintained Miles/FTI
  payload does not need host-level access, and the production Jobs API treats
  a privileged whole-node request as a warning.  Because this repository
  requires a warning-free preview, the canary fails closed if that setting
  drifts.
- One task prompt with eight independent samples.  The launcher explicitly
  sets Miles's over-sampling batch size to one; 0.10.9 otherwise asks for two
  candidate prompt groups and can run as many as sixteen episodes in one
  candidate wave.  A later wave is allowed only when the maintained filter
  rejects the first group for lacking valid reward variation.
- A group is eligible for an update only when none of its attempts aborted and
  their real verifier rewards vary.  A uniform reward cannot create a fake
  learning step.
- One optimizer update at learning rate `1e-6`.
- Miles numbers the first rollout and save as `0`.  The plan therefore asks
  for one optimizer update while binding the native checkpoint and raw HF
  export as `step-0`; it never relabels them as step 1.
- Save the Megatron checkpoint and raw text-tensor HF export after that update.
- Compose a complete Qwen3.8 artifact by taking trained text tensors from the
  step-0 export and restoring only the exact frozen `model.visual.*` and
  `mtp.*` tensors and runtime files from the bound base model.  Verify every
  tensor key, shape, dtype, finite value, and digest, require at least one
  trained tensor to change, and publish the complete directory create-once.
- A maintained Miles post-save hook writes a small receipt only after the
  checkpoint and HF export complete.
- A separate one-GPU observer loads that complete HF model and runs exactly one
  ordinary V1 task episode.  It does no optimizer work and writes no raw task
  text, model answer, reward number, or trace.

The training row is created fresh at runtime as
`<run-dir>/data/train.jsonl`.  It contains only a generic placeholder message,
the exact task key/version, and limits; Fleet resolves the actual task and
tools at runtime.  Existing rows, checkpoints, outputs, and W&B IDs are never
reused.

## Required inputs before any create

The committed [template](../configs/qualification/qwen38-miles96-mechanics-canary-v1.template.json)
is intentionally not launchable.  A restricted operator must make a new plan
with `training.miles96_mechanics_canary.build_plan` after all of these are
freshly checked:

1. New `chris-q38-` training and reload names, new W&B run ID, and new SFS
   output paths.  The canonical project prefix lets the cross-namespace census
   classify both jobs while they hold or queue GPU capacity.
2. An exact prepared-model SFS root.  It must contain both
   `Qwen3.8-27B/config.json` and
   `qwen3.8-27B_torch_dist/latest_checkpointed_iteration.txt` with the value
   `release`.  Before building the plan, compute the binding with
   `prepared_model_inventory(Path(root))["sha256"]` from a read-only process
   that can see SFS.  This hashes every file in both the HF and Megatron trees.
   The GPU runtime recomputes the same inventory both before training and
   again before it composes the complete export, and refuses any changed,
   missing, extra, or linked file.  The canary never downloads or converts a
   model on the GPU allocation.
3. A current authoritative observation for the exact task and verifier
   versions, plus the task-set, tool-catalog, and observation digests.  The V1
   wrapper checks those exact identities again when every episode opens.  This
   is only an authority check; no prompt or stored trajectory belongs in the
   plan.
4. A sanitized, digest-sealed prior receipt for that exact task version,
   verifier version, task-set digest, and tool-catalog digest.  It must show at
   least two finite completed episodes, actual reward variation, and released
   instances under no more than 32 turns, 8,192 tokens per turn, and 2,400
   seconds.  Reward values, prompts, answers, and traces do not enter the plan.
5. Immediate duplicate/absence checks for both names and all four output
   destinations: train output, reload output, fresh row directory, and HF
   export.
6. A fresh all-namespace GPU capacity census with the planned one-node request
   included: eight GPUs for training or one GPU for reload.  It must reconcile
   project inference models, contain no unclassified project GPU Pod, and stay
   within the project limit.

The model root and task binding are deliberately arguments to `build_plan`.
That prevents a future operator from accidentally reviving a stale path or a
historical task receipt just because it exists in this repository.

### Prepared-model staging boundary

`training.miles96_model_stage` is the zero-GPU create-once stager used when the
exact HF and Megatron source trees are not already paired below one readable
root.  The source PVC mounts are read-only and the destination must not exist.
The source directories currently reject uid 1000, so this narrow copy process
runs as uid 0.  Root is used only to inventory and copy those two immutable
sources.  The container is not privileged, drops every Linux capability,
cannot gain privileges, has a read-only root filesystem and no service-account
token, and requests no GPU.  Its public terminal receipt repeats those facts.

The stager's rendered Job begins suspended.  After two matching server
previews, an operator creates the immutable ConfigMap once and the Job once and
binds both returned UIDs.  Kueue owns admission and may change
`spec.suspend` to false before the operator can re-read the Job; the A2 staging
attempt proved that this is normal controller behavior.  The operator therefore
sends no unsuspend patch and never retries either create.  It only verifies the
root alert annotation, c1 priority, zero-GPU resources and exact UID, then
monitors the controller-managed Job.

Every runtime operation has a fixed phase name.  A sanitized failure receipt
records that phase, the exception class, and whether the exact destination,
current partial directory, or retired A2 partial directory exists.  It never
prints a source filename or model byte.  This closes the diagnostic gap in A2,
whose `FileNotFoundError` receipt did not identify the failing operation.
Terminal monitoring and cleanup remain bound to the exact Job UID.

## Mandatory live safety gates

`job_request` and `reload_request` only render generic Jobs API requests.
They cannot submit them.  `training.miles96_mechanics_launch.submit_once` is
the only create path for these requests.  It obtains a live Jobs API server
preview for each request and passes it through the shared preview validator.
The current server omits `backoffLimit` from its `ray.io/v1` render, so the
launcher also reads the live `RayJob` CRD and requires its admission default to
be exactly zero.  An explicit rendered value is accepted only when it is zero;
an unreadable or nonzero live default fails closed.

The root rendered `RayJob` must contain exactly:

```yaml
metadata:
  annotations:
    fleet.ai/failure-alerts: "off"
```

The request also has `failureAlerts: false`, but that request field is not
accepted as proof on its own.  The shared preview validator rejects a request
unless the rendered root object carries the annotation, has `c1` priority,
uses the pinned image, owns the expected GPU count, and releases after exit.

The create path writes the plan, request, and preview proof, then starts a
separate cleanup coordinator before any POST.  It freshly proves that the name,
title, output directory, Kubernetes identity, and W&B run ID do not already
exist.  When the submitter cannot see `/mnt/sfs/jobs`, it requires an exact
plan/request-bound SFS absence receipt no more than five minutes old.  The
receipt comes from the tracked zero-GPU, root-alert-off, create-once observer
below; it is validated after server preview and again immediately before the
create intent.  Immediately after that final SFS check, it runs the repository
cross-namespace census with the planned allocation included and seals the full
qualified receipt into the create intent.  Any unreadable, stale, over-limit,
unclassified, or otherwise uncertain census fails closed.  It then writes
`POST_INTENT_DO_NOT_RETRY` durably, issues
exactly one POST, and journals the returned API name and run ID.  The
coordinator binds that creator-returned name to the exact RayJob UID and may
release only that UID after terminal failure or a contract-defined stall.  An
uncertain POST is never retried.  The same process is repeated with a new
evidence directory for the one-GPU reload.

After independent review, a submitter with the real SFS mount can use the
create command directly.  A normal operator laptop first creates and collects
one exact remote absence observation:

```sh
uv run python -m training.miles96_mechanics_launch --sfs-output-job-create \
  --plan /restricted/fresh-plan.json \
  --request /restricted/fresh-request.json \
  --directory /restricted/train-sfs-check

# Run only after that exact zero-GPU Job succeeds.
uv run python -m training.miles96_mechanics_launch --sfs-output-job-collect \
  --plan /restricted/fresh-plan.json \
  --request /restricted/fresh-request.json \
  --directory /restricted/train-sfs-check \
  --output /restricted/train-sfs-absence.json
```

The zero-GPU Job is Kueue-admitted at `c1`, has a five-minute active deadline,
has `backoffLimit: 0`, and carries the required root
`fleet.ai/failure-alerts: "off"` annotation.  Its create journal is
create-once.  Never replay its create after an intent exists; inspect the exact
named Job instead.  Collect immediately before submission so the five-minute
receipt remains fresh.

The final Kubernetes duplicate scan recognizes only that exact terminal
observer Job and its successful zero-restart Pod.  Both must retain the plan
digest, request digest, output path, alert-off annotation, c1 queue labels, and
observer role.  This prevents the required evidence object from colliding with
its own training name while every Ray object, active or failed observer, and
differently bound Job still closes the launch gate.

The only training create command is then:

```sh
uv run python -m training.miles96_mechanics_launch --submit \
  --plan /restricted/fresh-plan.json \
  --request /restricted/fresh-request.json \
  --directory /restricted/fresh-launch-evidence \
  --output-absence-receipt /restricted/train-sfs-absence.json
```

The reload uses the same command with its separately rendered request and
evidence directory plus `--receipt /restricted/UPDATE_AND_EXPORT.json`.  Its
SFS observer create/collect commands also receive that `--receipt`, because the
reload request is bound to the accepted training receipt.  The
Fleet credential stays in the process environment; it is never written into a
plan, request, command, or receipt.  Neither command is safe to repeat after a
POST intent exists, even when the response is uncertain.

## What counts as passing

The selected-group hook keeps the exact verifier execution, instance, and
reward records in a private mode-0700 directory.  Its public receipt exposes
only digests and the facts that all eight rewards were finite, varied, came
from eight unique verifier executions, and all instances were released.  The
training receipt then proves the maintained post-save hook saw native rollout
0, exactly one optimizer update, a non-empty Megatron checkpoint, and a
complete verified HF model.  It also carries the exact W&B run ID.  The reload
receipt proves that complete model loaded on one GPU, obtained a finite real
verifier reward, observed a tool call, and cleaned up its task instance.  Public
receipts deliberately omit reward values, task text, answers, traces, and the
exact verifier execution IDs.

Each exact-UID terminal observer also records the requested digest-pinned image
and the owned GPU Pod's actual `status.containerStatuses[].imageID`.  It accepts
release evidence only when the resolved digest equals the requested digest;
missing or mismatched runtime identity leaves release uncertain for both the
eight-GPU trainer and the one-GPU reload.

Neeraj's pinned `dataminer_v2` `v004` run is useful prior evidence for this mechanical
shape: one node, 96K context, 80 recorded optimizer steps with finite sampled
reward metrics, a resume from 75 through 79, checkpoint/HF output, and
evaluation.  Its records do not independently prove every update tensor was
finite.  Its lift was narrow and other metrics regressed, so it supports the
trainer mechanics only—not transfer or capability improvement.  This
one-update canary has the same limitation.

Only after those receipts, the independent release observation, and the normal
held-out evaluation gates may this route inform a larger RL experiment.  A
successful mechanics canary does not establish a capability improvement or a
long-horizon training result.
