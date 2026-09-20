# Qwen3.8 LoRA development qualification: V4–V8

Date: 2026-09-20  
Scope: development cluster only; no production Jobs API submission occurred.

This note records only public lifecycle and digest evidence. It excludes model
data, task text, trainer logs, traces, credentials, and W&B secrets.

## V4: output-directory initialization contract

- Pod: `chris-q38-lora-sft-c1-v4-dev-eb439ef5`
- Pod UID: `f75c4ce0-5b79-4719-8203-01b14b8f815a`
- Result: the unprivileged init container could not create the shared output
  directory. The trainer never started, there were no restarts, and no
  optimizer step occurred.
- Repair: the init container now proves only that the create-once destination
  is absent. The root training container creates its own destination.
- Cleanup: the Pod and ConfigMap were deleted and the development GPU request
  returned to zero.

## V5: insufficient public step location

- CPU preflight Pod UID: `29071ea7-56f7-434e-8a74-60ef262d9df1`
- Training Pod: `chris-q38-lora-sft-c1-v5-dev-c8dbd81c`
- Training Pod UID: `84f3d6dc-89b0-4ae8-8a7d-0703f70f4bc7`
- Result: exact-image tokenization passed for 866 rows, 35 task versions, and
  998,652 supervised tokens. The GPU process reached device-ready state and
  then ended with a public `ValueError`, but the previous receipt could not
  locate the failure inside the first training step.
- Evidence: digest-valid `FAILED.json` and `FAILURE_STAGE.json` receipts were
  independently read through the bounded CPU-only reader. Their exact bytes
  remain under the create-once V5 output root.
- Repair: V6 added a create-once, sanitized qualification-stage receipt for the
  one-step gate only. It records boundaries, never values or training data, and
  is disabled for full training so it adds no per-step storage overhead.
- Cleanup: training and reader Pods plus the ConfigMap were deleted; no dev GPU
  allocation remained.

## V6: float representation mismatch before optimizer mutation

- CPU preflight Pod UID: `2cd9aca1-cae6-43e8-b421-a5cfe1314bfa`
- Training Pod: `chris-q38-lora-sft-c1-v6-dev-493082fd`
- Training Pod UID: `1477cc5b-d698-4c2f-b298-6497086cf2aa`
- Canonical plan SHA-256:
  `b1c82cd41ecbb52c9eca2f5970ba07bd2977428bc7d534ffdbab66ac79e89ae1`
- Result: CPU preflight again passed the exact 866/35/998,652 accounting. The
  GPU run initialized all ranks with zero restarts. The public qualification
  receipt stopped at `worker_lr_validated`: both learning-rate observations
  were numeric, finite, and positive, but the next exact-equality check rejected
  them. There was no metrics file, training-complete/paused receipt, checkpoint,
  or Qwen3.8 checkpoint receipt; no optimizer artifact is accepted.
- Cause: SkyRL's `all_reduce_metrics` converts a Python scalar into a default
  float32 tensor before returning the worker metric. The independent native
  optimizer query returns the original Python float. Direct equality therefore
  rejects an expected float32 round trip even when both observations describe
  the same learning rate.
- Evidence: `QUALIFICATION_STAGE.json` file SHA-256
  `7c982cc6c255a7d569b2321603bcf9b75a3ed6fe6b61effa8eaedea8a22909c5`;
  `FAILURE_STAGE.json` file SHA-256
  `fe4f0cbbc4e445bd7e7149a2e490189402ccb5952b5dac92b4b846408c617a8e`;
  `FAILED.json` file SHA-256
  `218d3071a7a62712dae36085780e90f41513ff76e6259dc41788308973c7bde5`.
- Repair: V7 uses `math.isclose` with relative tolerance `1e-6` and zero
  absolute tolerance for this cross-representation comparison. This accepts a
  float32 serialization round trip while remaining far tighter than a useful
  schedule change. Per-rank native learning-rate evidence must still be finite,
  positive, complete, and identical before this comparison.
- Cleanup: the failed GPU Pod, CPU reader, and ConfigMap were deleted. A final
  name-scoped census found no remaining V6 objects or GPU requests.

## V7: optimizer step and checkpoint succeeded; final qualification did not

- CPU preflight Pod UID: `7ea90cdc-374e-48d0-9a2e-bb1e324df164`
- Training Pod: `chris-q38-lora-sft-c1-v7-dev-7c1f3ea2`
- Training Pod UID: `12eaf3c1-c24e-41c8-8b41-128be10dbf35`
- Canonical plan SHA-256:
  `796a29f784cd9abc6cd4c4e89254fd8114817ac901c7eb2f92af15927740a129`
- Result: exact-image preflight passed the same 866 rows, 35 task versions,
  and 998,652 supervised tokens. The GPU run had zero restarts and completed a
  finite forward/backward pass plus optimizer update. Its sanitized scalar
  record reports optimizer step 1, finite loss `0.4274914860725403`, finite
  positive LoRA gradient norm `1.200891378263028`, learning rate `3e-5`, and
  185 supervised target tokens. The native checkpoint exists at exact step 1,
  its pointer reads `1`, and the digest-bound checkpoint receipt exists.
- Gate outcome: this is real successful learning work, but the run is not yet
  qualified for production. A later strict post-checkpoint check raised a
  public `ValueError` before `QWEN38_LORA_CHECKPOINT.json` and
  `TRAINING_PAUSED.json` could be published. The failure therefore lies after
  step evidence and native checkpoint creation, not in model setup, the
  forward/backward pass, the optimizer, or checkpoint writing.
- Evidence: `QUALIFICATION_STAGE.json` file SHA-256
  `4bf59963d1ca841b187ffff370dd00fbbfd8a70b2f8e23a81fc908b9f9689aff`;
  `FAILURE_STAGE.json` file SHA-256
  `58caab3cf83895e82280558e0fcfa8401ab59b535387832fd7ffcce4715977dd`;
  `FAILED.json` file SHA-256
  `31d4ac42b5d060148b842e8c1f2eb41a9020e597372d0ca65b37b8f89d84d2c`;
  `metrics.jsonl` file SHA-256
  `f9465fa3e8e882e7b76c31cb5215efc6387fb1abe1f5f2c54c47712e2b088b0d`.
- Next repair: add sanitized one-step-only boundaries around checkpoint drain,
  post-update adapter collection, receipt preparation, terminal validation,
  W&B flush, trainer shutdown, and final receipt validation. A fresh identity
  can then locate and repair the exact post-checkpoint contract without reading
  private trainer logs or weakening any acceptance check.
- Cleanup: the exact GPU Pod and every CPU reader were deleted. The final dev
  census showed zero active GPU requests.

## V8: defect isolated to strict receipt construction

- CPU preflight Pod UID: `7c9ccac9-f358-4727-957e-dd239cd6052c`
- Training Pod: `chris-q38-lora-sft-c1-v8-dev-363e4b02`
- Training Pod UID: `9d4cc598-08df-40cd-a163-23c9f3b8bb62`
- Canonical plan SHA-256:
  `f5702646d1d1dcc88088164f01c13cc1f7dd1e9a119a3760fc5e127f4da14363`
- Result: exact-image CPU preflight again passed 866 rows, 35 task versions,
  and 998,652 supervised tokens. The GPU run had zero restarts. The new public
  stage receipt reached `qualification_receipt_preparation_started`. By the
  immutable stage order, the finite optimizer update, asynchronous checkpoint
  drain, and post-update all-rank adapter snapshot had therefore all completed.
  The remaining `ValueError` is inside strict receipt preparation, before
  terminal-result validation, W&B finalization, and final receipt publication.
- Evidence: `QUALIFICATION_STAGE.json` file SHA-256
  `e3e86d2a6f4db15e7ace86335e027e6f1079970533c3e36b367f4f5f53816947`;
  `FAILURE_STAGE.json` file SHA-256
  `3c8f42f7d7069996483f690b41e02d4d4b251135f982fa4df64bbb0ae30f01b9`;
  `FAILED.json` file SHA-256
  `292fb79756e9d479a126e6087777eca96004a8a6f51bb63a1b4a1e8679f0cabc`;
  `metrics.jsonl` file SHA-256
  `e534e41199b3d46990ce217794d25a884284e344c28e25b5f486c82b529323fd`.
- Next repair: add one-step-only stage boundaries inside receipt preparation:
  native checkpoint inventory, trainer-step evidence, adapter reconciliation,
  source revalidation, immutability comparison, and source-plan validation. A
  fresh identity can then identify the exact rejected invariant without private
  trainer logs or relaxed checks.
- Cleanup: the GPU Pod, CPU reader, and ConfigMap were deleted. The final dev
  census showed zero active GPU requests.

## Production status

These were development qualifications, not production runs. The production
failure count is still **0/10**, and no production identity has been submitted
for this one-step LoRA gate.
