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
| Theseus source | `f2b0cb5db7c0a9dcc2210943f2ee1df31f9b50fc` |
| Trainer image | `miles-trainer@sha256:ee273bee346ad8e1cea63d18026b2bc5703bbd2e14d3c65efbbd74f19854874a` |
| FTI version | `0.10.10` |
| Miles source | `9e178ca16839b0600155f3927f57ce0670b8f453` |
| Recipe | `qwen3.8-27b`: one node, eight GPUs, TP4/CP2, 96K context |

The runtime checks the installed FTI version, the exact `run_fleet.py` digest,
and the maintained recipe shape before it writes a training row or starts the
trainer.  It calls `python -m fti.trainers.miles.run_fleet` directly.  It does
not call the older `/opt/fleet/run.sh` route, which would otherwise select an
old shared model directory when running outside a FleetJob mount.

## Exact, deliberately small job

- One eight-GPU node at `c1` priority.
- One task prompt with eight independent samples.
- A group is eligible for an update only when none of its attempts aborted and
  their real verifier rewards vary.  A uniform reward cannot create a fake
  learning step.
- One optimizer update at learning rate `1e-6`.
- Save the Megatron checkpoint and an HF export at update one.
- A maintained Miles post-save hook writes a small receipt only after the
  checkpoint and HF export complete.
- A separate one-GPU observer loads that HF export and runs exactly one
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

1. New training and reload names, new W&B run ID, and new SFS output paths.
2. An exact prepared-model SFS root.  It must contain both
   `Qwen3.8-27B/config.json` and
   `qwen3.8-27B_torch_dist/latest_checkpointed_iteration.txt` with the value
   `release`.  Its binding digest is recorded in the plan.  The canary never
   downloads or converts a model on the GPU allocation.
3. A current authoritative observation for the exact task version, plus the
   task-set, tool-catalog, and observation digests.  This is only a task
   identity check; no prompt or stored trajectory belongs in the plan.
4. Immediate duplicate/absence checks for both names and all four output
   destinations: train output, reload output, fresh row directory, and HF
   export.

The model root and task binding are deliberately arguments to `build_plan`.
That prevents a future operator from accidentally reviving a stale path or a
historical task receipt just because it exists in this repository.

## Mandatory live safety gates

`job_request` and `reload_request` only render generic Jobs API requests.
They cannot submit them.  Before a create, the operator must use the ordinary
Jobs API server preview for each request and pass it through
`cyber_post_train.jobs.validate_preview`.

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

Arm an exact-name, exact-UID cleanup observer before creating the training
job.  It may release only the matching terminal or confirmed-stalled job and
must record whether all GPUs and task instances were released.  If the update
does not happen—for example because the group has no reward variation—preserve
the sanitized evidence, release the exact allocation, and make a new plan;
never resume or replay this one.

## What counts as passing

The training receipt proves the maintained post-save hook saw update one after
the real-reward variation filter and after a completed HF export.  The reload
receipt proves the saved HF model loaded on one GPU, obtained a finite real
verifier reward, observed a tool call, and cleaned up its task instance.  Both
receipts deliberately omit reward values, task text, answers, and traces.

Only after those receipts, the independent release observation, and the normal
held-out evaluation gates may this route inform a larger RL experiment.  A
successful mechanics canary does not establish a capability improvement or a
long-horizon training result.
