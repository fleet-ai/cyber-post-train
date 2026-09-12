# Qwen3.8 scheduler reload gate

The two six-step scheduler canaries need separate native recovery checks before
their runtime path is accepted. The inert source of truth is
[`qwen38-teacher-scheduler-reload-dev-v2.template.json`](../configs/qualification/qwen38-teacher-scheduler-reload-dev-v2.template.json).
It binds the exact constant/no-warmup and cosine/5%-warmup source names and
paths, but remains non-launchable while either source is active or any terminal,
checkpoint-seal, preview or recovery identity is unresolved.

For each arm, first reconcile the exact Jobs API and UID-bound Kubernetes
objects with a digest-valid step-six `TRAINING_PAUSED.json`, complete scalar
tracking and released GPUs. Then run the template's `checkpoint-seal` command
once on CPU in the pinned training image. Record both the resulting manifest's
file SHA-256 and its embedded receipt SHA-256; they are different identities.
Never seal while the source trainer can still write the checkpoint.

Build a private recovery config from the source plan, not by retyping its
recipe. Remove only `pause_after_step`; assign a new run name, output root and
W&B identity; and add the arm's exact seal path and **file** digest with
`recovery.mode: validate`. The model, data, full 76-step horizon, scheduler,
warmup ratio, topology, image, resources and outcome-only protocol must remain
byte-for-byte equivalent after compilation. Run pinned-image CPU preflight,
Jobs API preview and duplicate destination/API/Kubernetes/W&B checks before a
reviewed POST. A source trainer and its reload must never run concurrently.

The GPU validator uses the same four-rank topology as the source checkpoint.
It may restore model, optimizer, scheduler, RNG and sampler state, but it may
not run CE, call the optimizer, save a checkpoint or publish training metrics.
The sole permitted W&B scalar is `train/global_step = 6` under a new validation
identity. Acceptance requires digest-valid `RECOVERED.json` and
`RELOAD_VALIDATED.json`, exact scheduler-state comparison on all four ranks,
zero new optimizer steps, a full post-run rehash of the unchanged source
checkpoint, zero restarts and verified GPU release. Both arms must pass
independently; a checkpoint or receipt from one arm cannot certify the other.

This is an operational recovery gate only. It does not rank the schedulers,
measure held-out task capability, qualify an inference export or authorize a
production training arm.
