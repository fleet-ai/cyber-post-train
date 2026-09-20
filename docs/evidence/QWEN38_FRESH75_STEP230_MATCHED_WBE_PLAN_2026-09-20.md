# Fresh75 step-230 matched WebExploitBench plan

Date: 2026-09-20

Status: **prepared but not launchable**. No model was resumed, no TensorLake
sandbox was created or changed, and no evaluation was launched while preparing
this plan.

## What will be compared

The comparison changes only the model weights:

- base: `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- candidate: the accepted Fresh75 checkpoint after optimizer step 230, served
  under model ID `chris-q38-fresh75-step230-v1`;
- harness: OpenCode;
- benchmark: all 15 WebExploitBench targets at the pinned benchmark and CAGE
  revisions in the sealed plan.

The exact machine-readable plan is
[`configs/evaluation/qwen38-fresh75-step230-opencode-wbe-matched-v1.json`](../../configs/evaluation/qwen38-fresh75-step230-opencode-wbe-matched-v1.json).
Its self-digest is
`sha256:3b37094868cc0dec65411cd552f67424eefbd9e01af1b5035b65bf54f535100d`.

## Two stages, in order

The first stage is a matched canary:

- one benchmark target;
- one rollout from the base model;
- one rollout from Fresh75;
- agent rollouts are saved without a judge call;
- both saved rollouts are scored later under the same judge settings.

The second stage is the full comparison:

- all 15 benchmark targets;
- four rollouts per target and per model;
- 60 rollouts for the base and 60 for Fresh75, 120 total;
- the same collection-first, score-later design.

The full stage cannot begin merely because the canary process exited. It needs
a receipt proving that both rollouts were fully saved, both were fully scored,
and the run was technically valid. The model's benchmark result is not used to
choose the checkpoint; WebExploitBench remains an external reporting benchmark.

## Why rollout collection and scoring are separate

An agent rollout must interact with a live challenge, so it is the expensive
and difficult-to-repeat part. The collector saves that completed interaction as
a read-only bundle and makes no judge call. The scoring command later reads that
same bundle.

This means:

- a judge outage cannot destroy a valid rollout;
- scoring can resume without running the agent again;
- a second judge can inspect the same rollout without changing the model's
  interaction; and
- a lost or uncertain scoring request is held for reconciliation rather than
  causing a duplicate paid call.

The existing `collection_launcher.py`, `collection_pair.py`,
`rollout_bundle.py`, and `deferred_score.py` implement those boundaries. The new
study-plan validator makes the canary-before-full order and duplicate exclusions
explicit.

## The 13 older suspended sandboxes

The read-only capacity audit found 13 suspended TensorLake sandboxes whose names
begin with `q38-s10-web-p4-v1-`. Historical local state identifies them as the
older `q38-t3k262-step10-opencode-web-p4-v1` campaign for the
`teacher3k-262k-step10` checkpoint. They are not Fresh75 work.

Their exact names and provider IDs are preserved in
[`qwen38-q38-s10-lineage-reconciliation-20260920.json`](qwen38-q38-s10-lineage-reconciliation-20260920.json).
They are reserved: the Fresh75 plan uses different campaign and sandbox names,
and no operator may resume, replace, or duplicate those old identities without
a fresh read-only provider check.

At launch time, a new `GET /sandboxes` inventory is mandatory. It must confirm
the current state of all 13 identities and prove that no equivalent Fresh75
base/candidate campaign is active or already accepted. The 2026-09-20 snapshot
is evidence for planning, not permission to launch later.

## Exact blocker

The Fresh75 payload is staged and registered, but its route is **paused**, has
zero active replicas, and is not available to the evaluator. The base model's
older observation is also not a current paired serving proof.

Before the paired canary can be sealed and launched, an authorized operator must
temporarily serve the exact Fresh75 model and establish a current base route,
then produce one fresh receipt proving that:

1. both routes are ready;
2. the candidate route serves the exact accepted step-230 export;
3. tokenizer, chat template, numerical format, context limit and serving runtime
   are identical between the two routes; and
4. identical continuation, forward-generation and tool-call probes pass on
   both models.

Only after that proof and the fresh TensorLake duplicate check may the exact
score-free canary collection plans be sealed. This work deliberately did not
resume or serve either model.

## Next action

When serving is authorized, qualify the two live routes and record fresh parity.
Then reconcile TensorLake inventory read-only, seal the two canary collection
arms under the plan's new identities, seal their pair, and launch only the
paired pass@1 canary. Do not prepare or launch the 15-by-4 stage until the exact
canary acceptance exists.
