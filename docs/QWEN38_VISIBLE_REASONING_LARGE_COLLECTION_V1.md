# First large Qwen visible-reasoning collection plan

## Outcome

The repository now contains a deterministic, source-only plan for a large
Qwen3.8-27B self-rollout campaign. It is **not approved to run**. It makes no
Fleet, model, trace, or scoring call and cannot create a workload. Independent
review and the immutable evidence listed below are still required.

The scientific purpose is to collect successful Qwen trajectories in which
the written reasoning was explicitly shown to the model's user. This is not a
path for importing a teacher model's private chain of thought. Provider-private
`thinking`, `reasoning`, `analysis`, inferred reasoning, and unknown fields are
rejected.

## Exact task boundary

The first campaign uses every currently receipt-proven training family:

- 50 exact task versions from 50 distinct training families;
- 17 development families that receive zero collection cells; and
- 8 final-test families that receive zero collection cells.

It binds the same immutable family-role anchor and protected-family lock as the
current action-only campaign. The source route itself is bound to the digest of
the ordered 50-version training selection; it does not reuse a development-set
evaluation profile. WebExploitBench and every other external
benchmark receive zero collection cells and remain evaluation-only. Results on
development, final-test, or external tasks cannot change which training tasks
the campaign samples.

This is broad over the currently admitted training set, not a claim that 50
families are the eventual full Fleet catalog. Newly qualified task families
must inherit a role through the anchored split before a later campaign can use
them. This v1 plan never silently absorbs mutable catalog growth.

## Why 20,000 rollout cells

The final corpus gate is **at least 20,000,000 unique supervised tokens** after
deduplication, with no task family contributing more than 25% and at least 20
families contributing verified successes.

The planning reference is aggregate-only. The existing teacher corpus has
57,384,881 supervised tokens from 2,886 verified-success sessions, or about
19,884 tokens per success. At that historical yield, 20 million tokens require
1,006 distinct successful sessions. A 20,000-cell universe can reach that mark
at a 5.03% verified-success rate. This is a sizing calculation, not a promise:
Qwen self rollouts may have a different success rate or token yield. If the
corpus misses the target, it remains unqualified and a new reviewed campaign is
needed. Nothing relabels failures as data.

The cells are balanced into 40 waves. Every wave runs ten fresh attempts on
each of the 50 training families, for 500 cells. Attempt number deterministically
selects a seed, and campaign, task, family, attempt, and seed determine one
create-once cell identity. Ambiguous creates are never replayed. A wave finishes
before checking the aggregate stop gate, so an early stop cannot favor whichever
families happened to run first.

## Exact model and online context treatment

The planned source is base `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, using OpenCode 1.18.27 with
only `bash` and `submit_report`. It requests and preserves written Qwen
reasoning explicitly. Sampling is temperature 0.6 and top-p 0.95, with a
deterministic distinct seed per attempt.

Online trajectories use the existing 262,144-token OpenCode context and 20,000
tokens of compaction headroom. Compaction is allowed online so long tasks can
continue. It is trainable offline only when the collector records the exact
pre-compaction prompt, the student-generated summary, the actual post-summary
prompt, and proves that this is the prompt used for the next target. Compaction
summary tokens are loss-masked. An opaque or unreconstructable compaction
boundary rejects that target and everything after it; the materializer never
replays an invented full history.

## What becomes training data

A record is eligible only when an immutable verifier execution proves success
for the exact task and environment version. Every accepted assistant target
must contain both:

1. explicitly student-visible Qwen reasoning; and
2. a visible action or tool call.

Source sessions, normalized trajectories, and packed windows are deduplicated
in that order. The offline materializer produces a matched action-only arm from
the same selected turns by masking reasoning tokens while leaving action targets
unchanged. It writes a separate manifest for each arm and a cross-arm digest
that proves the record, window, and token inputs are paired. The two corpora
remain separate. This makes it possible to test whether visible
reasoning supervision helps rather than confounding reasoning with different
tasks or successes.

## Remaining gates before collection

Merge or local generation does not authorize a rollout. Before any collection,
an independent reviewer must accept the source plan and the collection
authority must provide:

1. an immutable Fleet Artifact Registry authorization for this exact source;
2. an exact Qwen chat-template round-trip receipt;
3. a request-and-response receipt proving visible reasoning is preserved;
4. an exact native-compaction continuation recording receipt;
5. the distinct visible-reasoning collection packet and create-once operation
   authorization; and
6. the normal pre-create duplicate, identity, and resource checks.

After collection, only verified successes may enter the private materializer.
The source census, immutable verifier-success mapping, private selection,
token-only corpus, and aggregate coverage receipt must satisfy the merged
visible-reasoning contract. Even then the result is only
`source_only_qualified`; no existing SFT launcher may consume it until a later,
separately reviewed training change.

## Reproduce the source-only packet

```sh
uv run --locked python -m training.visible_reasoning_collection_campaign \
  --spec configs/collection/qwen38-self-visible-reasoning-train50-target20m-v1.source.json \
  --root . \
  --output configs/collection/qwen38-self-visible-reasoning-train50-target20m-v1 \
  --check
```

The committed output contains only the campaign plan, wave plan, counts, and
digests. It contains no prompt, trace, answer, flag, score, credential, token
sequence, or session identifier.
