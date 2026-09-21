# Broad stronger-teacher visible-rationale campaign v1

## Outcome

This change freezes the first scaled comparison between two ways of learning
from the **same** successful stronger-teacher sessions:

1. learn the teacher's short, ordinary explanation before each action plus the
   action itself; and
2. learn only the action, while retaining the explanation as loss-masked
   context.

It is source-only. It makes zero Fleet API calls, zero model calls, reads zero
raw traces, and authorizes neither collection nor training. An independent
review and an immutable teacher-source authorization are required before any
external work.

The source files are:

- [`stronger-teacher-visible-rationale-current75-pass64-v1.source.json`](../configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.source.json)
- [`stronger-teacher-visible-rationale-current75-pass64-v1.requirements.json`](../configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.requirements.json)
- [`broad-review.json`](../configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1/broad-review.json)

They build on the reviewed teacher-visible-rationale contract merged at
`11a793d14570b7522eb89705a2038596a16cb56b`.

## Why this is the first broad wave

The current exact, receipt-proven roster contains 50 training families, 17
development families, and 8 final-test families. This wave uses every one of
the 50 training families and none of the 25 held-out families. It does not use
the larger 1,093-version catalog directly because most of that catalog still
lacks the exact task-health and grading evidence required for training-data
collection. A later campaign can replace this frozen packet with a new packet
after more families qualify; it must not enlarge this packet in place.

Each training family has one exact selected task version. The campaign fixes 64
attempts per task before any outcomes are known:

```
50 task families × 64 attempts = 3,200 planned rollouts
```

The admission rule remains strict: only a completed session with authoritative
verifier success can enter either corpus. Failure, incomplete grading,
ambiguous creation, a duplicate session, or a mismatched task/runtime/model
binding produces no training example.

## Scale and cost estimate

The target is at least **20,000,000 unique supervised tokens in each arm** after
the private materializer performs exact Qwen tokenization, windowing, and
deduplication.

The planning reference is the already accepted action-only teacher corpus. It
contains a reviewed mix of stronger teachers rather than only the proposed GPT
source, so it is used only for scale planning:

- 57,384,881 distinct action-target tokens;
- 2,886 successful sessions; and
- an observed mean of `57,384,881 / 2,886`, about 19,884 action-target tokens
  per success.

At that historical mean, the action-only arm needs 1,006 successful sessions
to clear 20 million tokens. Across 3,200 predeclared attempts, that is a
planning success fraction of `1,006 / 3,200`, about **31.4%**. This is a sizing
assumption, not a promised success rate. The visible-rationale arm may contain
more supervised tokens, but it does not borrow those extra tokens to make the
action-only arm look large. Each arm must independently clear its exact 20M
gate.

At the project limit of 500 Fleet rollouts per day, 3,200 attempts need at least
seven calendar days. The sum of the per-attempt 262,144-token context capacities
is 838,860,800 tokens. That number is a capacity scale, not a forecast of billed
tokens: compaction and early task completion change actual use. A credible
dollar estimate is intentionally unavailable until the exact authorized route
supplies its billed input/output token counts and unit prices. The exact cost
formula is
`billed_input_tokens × input_USD_per_token + billed_output_tokens × output_USD_per_token`.
The checked-in review records the 3,200 paid rollout count, the capacity scale,
that formula, and the two missing measured/price pairs instead of inventing a
dollar amount.

## What counts as visible rationale

The teacher is asked to write one to four short sentences in ordinary assistant
text before every tool call. The text says what it learned, what it will test,
and why. This is the same text the student would see in the conversation.

The campaign never requests, copies, infers, or reconstructs provider-private
thinking. Any field named `analysis`, `thinking`, `reasoning`,
`reasoning_content`, `hidden_reasoning`, or `private_reasoning` rejects the
record. A tool call without its ordinary visible explanation also rejects the
record.

The serializer is pinned to the exact Qwen3.8-27B revision and chat-template
digest. It uses `enable_thinking=false`, and requires locally rendered Qwen
tokens to match collection, training, and serving serialization exactly.

## Context and compaction

Collection uses OpenCode 1.18.27 with a 262,144-token context window, 20,000
tokens of reserved headroom, and native compaction with automatic continuation.
Long sessions may continue after compaction, but opaque summaries are never
accepted as training context.

Every accepted compaction boundary must bind the actual visible summary, the
prompt before and after compaction, and the true next supervised target in one
ordered digest chain. The summary remains zero-loss context in both comparison
arms. It is never treated as a rationale target.

## The matched comparison

The comparison is produced from one collection and one exact selected-success
set. Both arms use the same:

- source sessions and verifier-success evidence;
- task families and actions;
- serialized messages and compaction history; and
- packed-window selection.

Only the loss mask changes:

| Target type | Visible rationale + actions | Matched actions only |
| --- | ---: | ---: |
| Ordinary visible explanation before a tool call | 1 | 0 |
| Visible tool action | 1 | 1 |
| Visible compaction summary | 0 | 0 |

There is no second model collection for the comparator. This avoids changing
sampling luck, successful-session selection, actions, or context while testing
whether supervising the visible explanations helps.

## Held-out and benchmark separation

Development and final-test Fleet families are excluded at roster validation,
admission, and private materialization. WebExploitBench is evaluation-only:
none of its prompts, tasks, traces, outputs, metadata, solutions, or derived
hints may enter this campaign. The source spec names this exclusion explicitly
so it cannot disappear behind a generic "train only" label.

## Create-once identity

The offline review derives one stable scientific-cell identity from the exact
requirements, task/version, runtime binding, and attempt number. After an exact
teacher authorization exists, the renderer derives a second execution identity
from the immutable collection packet and scientific-cell identity. The private
`execution-map.private.json` contains every exact task key, task-version ID,
runtime-binding digest, attempt, scientific ID, authorized execution ID,
exclusive-intent ID, and create-once relative path. The authorization binds the
ordered map digest for all 3,200 identities to:

- one canonical operation root;
- one dedicated empty ledger;
- an exclusive intent written before any mutation; and
- no same-path retry, alternate-path retry, or replay after an ambiguous
  create.

Changing the teacher authorization changes the collection packet and therefore
the operation root, while leaving the scientific question unchanged. This
prevents a source change from silently reusing an old operation.

## Local review commands

The first command needs no teacher authorization:

```sh
uv run --locked cyber-post-train \
  data-fleet-teacher-visible-rationale-broad-review \
  configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.source.json
```

After an independently reviewed immutable authorization exists, the second
command can render a local source bundle. It still does not collect or train:

```sh
uv run --locked cyber-post-train \
  data-fleet-teacher-visible-rationale-broad-render \
  configs/collection/stronger-teacher-visible-rationale-current75-pass64-v1.source.json \
  /private/path/to/exact-source-authorization.json \
  /private/create-once/output
```

Before any launch, a separate reviewed launcher must consume the exact
operation authorization, prove its no-duplicate and cleanup behavior, and
preserve the mandatory root `fleet.ai/failure-alerts: "off"` annotation if it
creates a Kubernetes Job or RayJob. This PR deliberately does not implement or
authorize that external step.

After metadata admission, the local private path is executable without any
external mutation:

```sh
uv run --locked cyber-post-train data-fleet-teacher-visible-rationale-materialize \
  /private/path/to/matched-materialization-request.json

uv run --locked cyber-post-train data-fleet-teacher-visible-rationale-authorize \
  /private/corpus/matched-manifest.private.json \
  /private/corpus/coverage.private.json \
  /private/corpus/MATERIALIZATION.json \
  /private/corpus/TRAINING-PERMIT.json
```

The materializer opens the private messages locally, rejects hidden/private
fields, repeats the visible-rationale and OpenCode checks, and re-renders every
window through the exact locked Qwen tokenizer/template with thinking disabled.
It then writes two token-only Parquet files from the same ordered windows. It
derives source-token occurrence IDs without using packed-window positions, so
repacking cannot make one target count twice. It checks the 20M, family-count,
family-coverage, and 25% concentration gates separately for both arms. Before
issuing a permit, the authorizer reopens the bound Parquet/manifests and proves
their rows, masks, counts, family totals, file digests, and same-window pairing.
The permit fails closed unless both arms pass. A final `...-consume` command
creates the standard dense-SFT manifest for exactly one permitted arm; it still
launches nothing.
