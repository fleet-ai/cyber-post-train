# Run a WebExploitBench evaluation

Use the new controller with a reviewed, sealed collection launch plan:

```sh
uv run python -m evals.webexploitbench.tensorlake.campaign \
  --launch-plan /absolute/private/launch-plan.json \
  --state /absolute/durable/evaluation-state \
  --concurrency 4
```

This first command is a **preview**, not a paid submission. Add `--execute` to
run it. Supply `TENSORLAKE_API_KEY` and `FLEET_API_KEY` through your secret manager
or environment, never as command-line arguments. Use an always-on trusted host;
a laptop sleeping is not an unattended execution service.

The command connects the existing tested operators:

1. Create the first task as a canary. The other tasks remain unallocated until
   that canary has produced and preserved a score-free collection. A launch
   plan controls the exact model, task order, OpenCode version, budgets,
   prepared filesystem and source-file hashes.
2. Collect attempts **without a judge**.
3. Check completeness and preserve the prepared Docker daemon, images, files,
   and process state in a memory snapshot. A filesystem-only snapshot cold-boots
   and is not sufficient for this nested-Docker evaluator.
4. Release the collection machine. Temporarily restore the snapshot on a small
   CPU machine to download and verify the collection, then release that machine.
5. Optionally score a separate local copy using a frozen judge configuration.

`task-NNNN/status.json` is readable progress, not scientific acceptance.
The underlying collection, download and score records contain the checked
digests. A technical failure is reported as needing review, never as a zero
model score. If the canary fails before collection, the remaining tasks do not
launch. After the canary succeeds, other independent tasks continue if one
task fails.

## Restart safely

Run **the exact same command, source revision and state directory**. Completed
work is verified and reopened; it is not rerun. If a create response was lost,
the controller reconciles the uniquely named, exactly matching machine. An
ambiguous model-process dispatch stops that task for investigation; it never
blindly sends the same paid attempt twice. Safe GET requests have bounded
retries. POST requests do not.

Keep state and downloaded collections on backed-up durable storage. Provider
timeouts bound cloud allocations if the host disappears, but a timeout is not
proof that results were downloaded. Monitor controller liveness and investigate
`needs_review`; do not call this an infallible workflow.

## Score separately, or score automatically after download

For automatic scoring, add `--score-template /private/judge-template.json` from
the first invocation. The template contains the `schema_version`, `judge`, and
`runtime` sections of a reviewed deferred-scoring draft. The controller supplies
each downloaded collection and a separate output directory. The scorer verifies
the judge's synthetic qualification and official benchmark implementation
before making scoring calls.

To choose a judge after collection, or to compare judges without new rollouts,
use the existing separate scorer:

```sh
uv run python -m evals.webexploitbench.tensorlake.deferred_score seal-plan \
  --draft /private/score-draft.json --output /private/score-plan.json
uv run python -m evals.webexploitbench.tensorlake.deferred_score score \
  --plan /private/score-plan.json --execute
```

GPT, GLM and other supported providers can be configured; each is a separately
identified measurement. Changing the judge must not overwrite or mutate the
original collection. A failed scoring transaction retains its evidence and
requires reconciliation before retrying that same transaction. It does not
authorize a new student rollout.

## Current qualification status

The orchestration tests cover partial failure, cleanup, restart behavior,
judge/collection separation, and the one-task collection canary. Fresh75
campaign `fresh75-step230-opencode-web-p1-v3` is terminally
infrastructure-invalid: nine restored machines failed readiness before any
model call, six tasks were never created, and there are no rollouts or scores.
The nine launched machines were released and Fresh75 serving was paused. The
source-only environment qualification was therefore insufficient. A new
snapshot must be checked after restoration before a replacement canary starts.
The completed GPT rescore demonstrates recovery of older stored work; it is not
by itself end-to-end proof for this controller.

See [Tensorlake execution instructions](../evals/webexploitbench/tensorlake/README.md)
for preparation and sealed-plan requirements. The command deliberately does not
accept an arbitrary model name and silently invent the remaining scientific
settings or bypass checkpoint acceptance.
