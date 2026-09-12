# Qwen3.8 SkyRL production arm

Status: **fully specified but deliberately not launchable**.

The immutable candidate is
[`qwen38-rl-filtered-study-a-prod-v1.template.json`](../configs/runs/qwen38-rl-filtered-study-a-prod-v1.template.json).
It remains a wrapper, not a `cyber-post-train rl` input, until every dev and
live release gate below is independently satisfied. Do not copy its
`candidate_run` into a runnable file early.

## Data boundary

The candidate uses the evidence-backed 89-task representative study split but
includes only its 59 train and 20 Fleet-dev rows:

- [`qwen38-rl-filtered-study-a-task-set-v1.json`](../configs/data/qwen38-rl-filtered-study-a-task-set-v1.json)
- [`qwen38-rl-filtered-study-a-split-v1.json`](../configs/data/qwen38-rl-filtered-study-a-split-v1.json)
- [`qwen38-rl-filtered-study-a-prod-v1.data.json`](../configs/runs/qwen38-rl-filtered-study-a-prod-v1.data.json)

Every task key, task-version UUID, environment/data version and reviewed
application/task-family lineage is copied from the exact eligibility and study
manifests. The optimizer reads exactly the 59 `train` rows. The 20 `dev` rows
are passed only through native pre/terminal evaluation and never enter the
training dataloader, GRPO reward normalization, replay or optimizer. All ten
sealed final-test tasks and all WebExploitBench tasks are absent. The data build
is CPU/GET-only, create-once and must retain all 79 selected rows; it has not
been run by this preparation.

## Frozen candidate recipe

- exact base `Qwen/Qwen3.8-27B` revision and weight manifest;
- one production node, eight GPUs, one prompt group and eight samples per
  prompt;
- 59 optimizer steps at learning rate `1e-6`, which is one ordered pass over
  the 59 train prompt groups;
- native dev evaluation before training and at the terminal 59th step;
- native checkpoint every 10 steps, retaining the latest three, plus the
  native terminal save;
- W&B `thefleet/cyber-post-train/chris-q38-rl-prod1` with scalar-only logging;
- Jobs API `c1` / rendered queue `q1`, effective priority 10000, no automatic
  requeue;
- fresh output `/mnt/sfs/jobs/chris-q38-rl-prod1` and fresh create-once data
  output; and
- one experiment node, checked against the live maximum of eight active
  production experiment nodes. The four inference endpoints are excluded from
  that accounting exactly as authorized.

This preserves the reward canary's model, one-node/eight-GPU layout,
`groups=1`, `samples_per_prompt=8`, learning rate, seed and token/tool budgets.
Only the dataset size, step/eval/checkpoint counts, cluster target and
create-once identities change. There is therefore no unqualified topology or
GRPO group-scaling change between the canary and production.

## Promotion gates

All gates are conjunctive. A clean Jobs API exit or
`NATIVE_TRAINING_COMPLETE.json` alone is not acceptance.

1. **Dev8 engine gate.** Bind a digest-valid terminal receipt from the exact
   corrected-image dev8 diagnostic. It must prove both TP4 engines started,
   zero task/reward/optimizer/checkpoint work and complete GPU release.
2. **Real reward canary.** Bind the exact prepared dev reward canary and a
   digest-valid independent audit. Every rollout must be nontruncated and
   single-attempt, retain an authoritative nonempty verifier-execution ID and
   prove cleanup. At least one train reward must be nonzero and the eight-sample
   group must have reward variance. Exactly one optimizer update, independent
   model/optimizer change, W&B identity and a complete sealed step-1 checkpoint
   are required.
3. **Native RL reload.** Restore that sealed checkpoint on all eight ranks in a
   fresh dev run, including model, optimizer, scheduler, sampler and RNG state.
   Execute a native model forward but zero rollouts, zero optimizer updates and
   zero checkpoint writes. Prove source-checkpoint stability and GPU release.
4. **Live duplicate/output/W&B gate.** Prove the data destination absent before
   its one create-once CPU build. Immediately before the one permitted POST,
   verify the resulting data manifest against the frozen source digests and
   reconcile the production Jobs API, production output root, W&B run ID and
   any submission journal. Any existing or ambiguous run identity blocks the
   POST. Preview must be warning-free and reproduce one worker × eight GPUs,
   `c1`/`q1` at 10000, Secret-backed credentials and no requeue.
5. **Live resource gate.** Count active production experiment allocations
   immediately before submission. The candidate may admit only if the total,
   including its one node, is at most eight; never touch inference or peer
   workloads to manufacture room.

The template contains null receipt paths/digests and `launchable: false` so the
absence of any gate is visible and machine-tested.

## Exact remaining gap

The repo still has no SkyRL checkpoint-seal schema or zero-update all-rank
native RL reload command. The inert reload contract is
[`qwen38-rl-reward-canary-reload-dev-v1.template.json`](../configs/qualification/qwen38-rl-reward-canary-reload-dev-v1.template.json).
That command and its dev acceptance receipt must exist before production can be
opened. The current SFT recovery path is not a valid substitute.

After the dev engine, reward/update and native-reload receipts exist, update the
production template with their exact paths and file digests, independently
verify every predicate, create a digest-bound release receipt, and only then
materialize `candidate_run` as a fresh runnable config. CPU preflight, prod
preview and exact duplicate checks still happen after materialization. This
preparation submitted no job and made no external mutation.
