# Qwen3.8 SFT launch readiness — 2026-09-21

This is a source-only audit.  It made no Fleet API request, Kubernetes request,
GPU allocation, W&B request, or training submission.  It records which next
SFT arm can use the existing launch machinery without changing its scientific
recipe, and which checks must still happen against the live cluster.

## Result

The fresh **96K full-weight** arm is the quickest next SFT execution.  Its
configuration is
`configs/runs/qwen38-teacher3k-96k-full-b8-lr3e6-v3.json`.  It is a new
create-once identity for the already qualified 96K treatment: one eight-GPU
node, batch eight, learning rate `3e-6`, one epoch, 856 updates, and the
existing 57.38M visible-action target-token corpus.

The **LoRA A2 anchor** is also source-qualified for the same SFT launch path:
`configs/runs/qwen38-27b-lora-sft-r64-a32-anchor-a2-v1.json`.  It is the
fresh identity for the qualified rank-64, alpha-32, all-linear LoRA treatment.
It has an extra small entrypoint binding so that historical runtime bytes stay
unchanged; the local guard proves that this binding still renders the exact
reviewed request.

Neither conclusion means either job has been submitted.  A local configuration
cannot prove a live scheduler object.  Each arm still needs its own fresh
zero-GPU CPU preflight, output/duplicate check, and live server preview.

## What is already enforced locally

The regression test
`tests/test_qwen38_sft_launch_readiness.py` prepares both configurations from
scratch and proves all of the following without calling an external service:

- the immutable plan and request render again from current source;
- the request uses exactly one eight-GPU worker at `c1` and sets
  `failureAlerts: false`;
- both are accepted by the maintained SFT-only direct-submit contract;
- the exact source-bound CPU preflight Job requests zero GPUs, uses `c1`, and
  has `fleet.ai/failure-alerts: "off"` on both the root Job and its Pod
  template.

The LoRA A2 qualification packet now also names the maintained fallback.  If
the current generic API preview omits **only** the root alert annotation,
`direct-submit-sft` consumes that preview, adds the root annotation before
creation, and requires a Kubernetes server dry-run.  It is not a manual patch
or an alternate generic launcher.

The A2 anchor now has a named CPU-preflight command pair in `docs/TRAINING.md`.
It reuses the maintained generic zero-GPU Job rather than a separate LoRA Pod,
then wraps its native receipt with the exact Qwen revision, adapter settings,
runtime binding, and one-node/eight-GPU training shape. Generic CPU-preflight
commands reject this A2 LoRA plan, and every later submit path requires the
typed receipt.

## What the live checks must prove

Local intent is not enough for the final RayJob.  Immediately before each
create, the operator must follow the maintained order in `docs/TRAINING.md`:

1. Prepare the exact configuration into a new local prepared directory.
2. Create and collect the source-bound, zero-GPU CPU preflight Job for that
   exact prepared directory.
3. Recheck output absence, duplicate identities, and current project capacity.
4. Ask the Jobs API for a fresh preview.
5. Use normal submit only if that preview itself has
   `fleet.ai/failure-alerts: "off"` on the **root RayJob**.
6. If the preview differs only by omitting that root annotation, use the
   maintained `direct-submit-sft` command.  Its server dry-run must prove the
   annotation before its one create.

Any warning, identity drift, stale preflight, existing output, missing root
annotation, or capacity mismatch stops the launch.  The fallback is restricted
to SFT; it must never be copied to RL, conversion, or another workload type.

## Ordering

Run the 96K full-weight arm first because it answers the already-prioritized
context question using the same full-weight trainer family as the completed
96K canary.  The LoRA A2 arm should follow through the same guarded launch
sequence; it answers a separate full-weight-versus-adapter question and must
not be treated as a learning-rate sweep or as a replacement for the 96K
comparison.

This audit deliberately does not invent a new CUDA or distributed-training
claim.  A CPU preflight checks the image, loader, configuration, tokenization,
and target accounting; the real GPU run remains responsible for proving its
own CUDA/distributed behavior and checkpoint evidence.
