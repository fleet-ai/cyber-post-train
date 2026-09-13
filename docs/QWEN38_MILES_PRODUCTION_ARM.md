# Qwen3.8 Miles production arm

`configs/runs/qwen38-miles-rl-filtered-study-a-prod-v1.template.json` is an
inert, exact production candidate. It cannot be compiled or submitted as-is.
Its reviewed recipe is one 8-GPU B300 node, 59 optimizer updates (one pass over
the 59 training task groups), eight samples per group, LR `1e-6`, checkpoint
every 10 updates, and baseline plus terminal native evaluation over the 20
Fleet dev tasks. The Jobs API request is `c1` (rendered `q1`, live value 10000)
with no automatic requeue.

Production promotion requires this exact create-once chain:

1. The corrected dev4 canary must finish with ten valid episodes (eight train,
   two dev), genuine authoritative rewards with within-group variance, exactly
   one independently verified policy-tensor update, a sealed checkpoint, exact
   W&B identity, and externally verified GPU release. Validate
   `MILES_TERMINAL_ACCEPTED.json` with
   `training.miles_acceptance.validate_terminal(..., check_files=True)`.
2. Create a sanitized, digest-valid active-canary identity with
   `training.miles_promotion.accept_active_canary_binding`, passing the exact
   accepted terminal receipt and a create-once output path. The binding is
   derived from the terminal's immutable source, request, API, RayJob, and
   Workload evidence; it contains no assumed dev generation or mutable
   `current` pointer.
3. Run the single all-rank policy observer described in
   `docs/QWEN38_MILES_POLICY_OBSERVER.md`. Its trained leg restores that same
   checkpoint on all eight ranks without doing any work and also provides the
   value-sensitive policy delta required by step 1. After the training terminal
   receipt exists, perform the CPU-only evidence join and validate the resulting
   create-once `RELOAD_ACCEPTED.json` with
   `training.miles_reload_acceptance.validate_accepted(..., check_files=True)`.
   Do not launch a second eight-GPU reload job.
4. Build the exact 59-train/20-dev Miles data manifest from
   `configs/runs/qwen38-miles-rl-filtered-study-a-prod-v1.data.json`. This gate
   is complete: the create-once manifest is at
   `/mnt/sfs/jobs/chris-q38-miles-rl-prod1-inputs/data/manifest.json`, with its
   sanitized preparation receipt at the parent `DATA_PREPARED.json`. The
   tracked evidence is
   `docs/evidence/qwen38-study/2026-09-13-miles-prod1-data-prepared-v1.json`.
5. Create a production-promotion receipt with
   `training.miles_promotion.accept_promotion`, passing the exact active-canary
   binding, reward terminal, native reload, and production-data manifest paths.
   Null or placeholder terminal, reload, or binding digests are rejected. Copy
   the inert candidate to a new run config and replace only
   `production_promotion` with that receipt's absolute path, file SHA-256, and
   receipt SHA-256.

The create-once BF16 Hugging Face export and its independent model reload may
run concurrently with production after the native reload passes. They are not a
production-training prerequisite because production starts from the exact base
checkpoint, not the dev export. Both remain mandatory before any checkpoint is
served or evaluated.

Compilation and request creation reopen the entire chain. The production CLI
then repeats the warning-free preview, derived `q1` queue binding, live `c1`
value 10000, unique Jobs/W&B/output identities, exact data bytes, and the
eight-node ceiling immediately before its one create-once POST. The allocated
runtime rechecks the sealed promotion and exact candidate before opening model
or task data.

WebExploitBench is not training data, reward input, an HPO signal, a retry
signal, or a checkpoint-selection signal. Its Tensorlake plan appears only in
the post-training handoff, after the checkpoint and recipe are frozen. The
20-task Fleet dev split remains evaluation-only and is the only allowed native
checkpoint-selection signal for this arm.

No production job was submitted while preparing this arm.
