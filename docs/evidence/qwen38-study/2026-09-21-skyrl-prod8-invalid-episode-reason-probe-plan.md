# Prod8 InvalidEpisode reason-code probe plan

Status: prepared locally; not previewed, created, or submitted.

The prod8 terminal evidence establishes only this: its first evaluation batch
started at optimizer step zero and later failed with an `InvalidEpisode` inside
an exception group.  It collected no training batch and produced no checkpoint.
That outer error is intentionally non-specific: `skyrl_rollout.Generator`
wraps every failed sibling in one `skyrl_batch_failed_no_replacement` error, and
the root `NATIVE_FAILURE.json` keeps classes and code locations but not error
messages or leaf reason strings.  It is therefore not evidence for a particular
training repair.

## Narrow diagnostic

`scripts/probe_qwen38_prod8_invalid_episode_reason.py` defines one bounded
CPU-only diagnostic Job.  It has no GPUs, no service-account token, a read-only
SFS mount, a read-only root filesystem, priority `c1`, and the required root
`fleet.ai/failure-alerts: "off"` annotation.  It does not read trainer logs,
prompts, conversations, tool calls, reward records, recordings, checkpoints,
or stack traces.

The probe first requires a digest-valid, plan-bound root
`NATIVE_FAILURE.json` that includes `InvalidEpisode`.  It then inspects at most
eight expected batch directories and eight expected episode directories per
batch.  From an exact, bounded `failure.json` schema it emits only a reason
which was already deliberately allowlisted in `training.rl_episode` when that
failure receipt was written.  Any malformed file, unexpected reason, multiple
different reasons, unstable read, or missing root binding produces an
unaccepted or ambiguous receipt instead of an inferred diagnosis.

The command that can produce a passed preview receipt is deliberately limited
to `--server-preview --context <exact-production-context>`. It calls Kubernetes
with `create --dry-run=server`, then validates the returned root Job. It never
creates a Job. The lower-level render validator is used only inside that path
and in tests; it produces a validation record, not a passed server-preview
receipt. The final receipt binds the digest of that validation record to the
server-render digest.

The receipt contains a single reason only when it is deterministic. It carries
only a fixed, non-path evidence-binding label plus the already-bound plan
digest; it never contains a storage path, filename, batch ID, run ID, message,
frame, prompt, tool input or output, score, reward, checkpoint path, or raw
exception text.

Both receipt readers reject duplicate keys, non-finite JSON numbers, malformed
frame records, unexpected fields, oversize files, unstable reads, links, and
out-of-bound directory fan-out. A malformed receipt becomes sealed
`unaccepted`; it must never crash the diagnostic or be treated as a diagnosis.
An absent per-episode failure receipt is ignored; an existing but unreadable or
malformed one is not equivalent to absence and makes the result `unaccepted`.

## Required gate before any diagnostic execution

The Job is only a local prepared artifact.  Before a separate authorized
operator can create it, they must obtain a fresh Kubernetes server dry-run and
verify the rendered root annotation, exact read-only storage shape, zero GPU
resources, and `c1` priority.  A generic Jobs API RL successor remains blocked:
its current live preview does not prove the required root alert annotation, and
the repository policy limits the direct-create fallback to source-bound SFT.

## What a result would mean

* `classified`: one defined leaf reason was found.  Add a focused code guard
  and regression for that exact reason before preparing a successor.
* `ambiguous`, `no_safe_reason`, or `unaccepted`: do not guess a repair and do
  not submit another RL successor.  Preserve the sealed receipt and choose a
  new bounded diagnostic only after reviewing the evidence boundary.

The tests for the probe prove its read scope and redaction behavior, not that a
live episode ran correctly or that RL training is qualified.
