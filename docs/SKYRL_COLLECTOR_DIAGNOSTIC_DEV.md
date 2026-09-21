# Dev SkyRL collector diagnostic

This is a bounded debugging tool for one failed SkyRL collection path.  It is
not a training run and it is deliberately unable to submit work on its own.

It runs exactly one selected training-split task row against the immutable
SkyRL image-native collector.  It does **not** start a trainer, make an
optimizer update, write a checkpoint, or initialize Weights & Biases.  Private
trajectory files stay in a temporary directory inside the container and are
removed before the only durable output is written.  That output is a small,
sealed receipt containing only fixed status categories, identity digests, and,
after engine startup, the public setup module/package version (never a path or
source-file hash).

## What the preparation command does

`scripts/prepare_skyrl_collector_diagnostic.py` accepts a reviewed JSON config
and writes a create-once local packet:

- `PLAN.json` binds the source training plan, image, runtime source bytes, one
  selected row, sampling settings, one dev node with eight GPUs, `c1`, and a
  thirty-minute limit.
- `REQUEST.json` has `failureAlerts: false`, one eight-GPU worker, no W&B
  secret, and W&B disabled in its environment.
- `OFFLINE_PREVIEW.json` makes clear that this is local rendering only, not a
  server preview.
- `RELEASE_OBSERVER_CONTRACT.json` describes the required safe release path.

The command has no Fleet client, Kubernetes command, credential handling,
preview option, or create option.  Preparing a packet never starts a job.

## Conditions before any future create

An approved one-shot creator must complete every condition in one fresh launch
transaction.  This repository does not provide that creator in this change.

1. Ask the **dev** generic Jobs API for a fresh server preview of the exact
   request.
2. Verify that the rendered root RayJob, not merely the request or a Pod
   template, contains `fleet.ai/failure-alerts: "off"`.
3. Verify the exact image, one-node/eight-GPU shape, `c1`, no retry queueing,
   automatic shutdown after the job, and disabled W&B environment.
4. Before POST, perform a read-only collision check for the generated-name
   prefix.  The prefix is only a collision check; it never establishes
   ownership and is never allowed to select an object for cleanup.
5. Create at most once.  Capture the API-returned exact run name and run ID.
6. Bind that exact name to the rendered RayJob UID, then bind all generated
   RayCluster, Workload, and Pod UIDs.  A UID change, an unexpected child, or
   an inaccessible object is an uncertainty result, not permission to act.
7. Use an exact-UID observer for at most thirty minutes.  It may release only
   the creator-bound root UID and must prove that the bound generated objects
   and GPUs are gone before claiming release.  It must never delete by prefix
   or touch a peer workload.

If any condition is not met, the right outcome is no create or an explicit
uncertainty record.  Do not substitute a direct RayJob manifest, silently
relax a check, or reuse a packet after a failed/uncertain POST.

## What a terminal receipt means

`collection_completed_no_training` means the one collector call completed and
the collector confirmed release of its Fleet task instance.  It does not mean
that an optimizer step, checkpoint, or training outcome exists.

`allowlisted_invalid_episode` records only one known, pre-approved failure
category.  Any nested, multiple, or unfamiliar error is
`unclassified_failure`; exception text is intentionally not saved.

The external exact-UID observer is still the authority for release of the GPU
allocation in all terminal cases.
