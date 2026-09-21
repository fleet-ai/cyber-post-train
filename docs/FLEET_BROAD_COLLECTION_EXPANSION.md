# Expanding Fleet trajectory collection without weakening the holdout

## Purpose

The original 75-task study split is an important fixed comparison set, not the
maximum amount of Fleet data that can ever be used for training.  This document
describes the safe route from the current broad catalog to a substantially
larger, family-safe collection campaign.  It is an offline design and
validation path.  It does not read rollout text, create a rollout, or grant
permission to launch one.

## The catalog is supply, not training data

The current production inventory records 1,093 blackbox task versions in
`configs/data/fleet-blackbox-current-production-20260921-v1.json`.  That is a
useful list of possible work, but it does not by itself prove that a task is
healthy, graded correctly, or bound to its current environment.  The catalog
contains known-broken and not-yet-reviewed entries.

Today, the strict, receipt-proven base is the 75-version inventory in
`configs/data/fleet-blackbox-current-high-quality-20260914-v1.json`.  The
associated study split fixes 50 training, 17 development, and 8 final-test
families.  The 17 development and 8 final-test families remain unavailable for
SFT, preference data, and RL prompts even when a later catalog lists a new
version of the same family.

There are 89 previously examined versions with exact runtime bindings and
reviewed taxonomy.  They are useful inputs to a qualification campaign, but
their historical 89-task split is **not** a safe replacement: it would move
some of the original held-out families into training.  A fresh generic split
of the old 75 also changes historic roles.  Both are rejected by design.

## What makes a new task usable

A task version joins a broad collection roster only through a sealed,
metadata-only qualification record.  The supply catalog and qualified subset
both carry the exact digest of the source catalog snapshot, and the roster
refuses a mismatch.  It must bind all of these facts:

1. It belongs to the exact current production catalog snapshot.
2. Its task shape is the authorized blackbox task shape.
3. Its current environment, data, verifier, and cleanup bindings have exact
   immutable identities.
4. A task-health receipt proves a completed, authoritative grading path and
   no known broken-environment finding for that exact version.
5. Its application, environment, difficulty, task family, and vulnerability
   family are reviewed rather than unknown.
6. Its runtime binding and task-validity receipts have exact digests.

The record carries only identities, labels, and digests.  It never carries a
prompt, solution, trace, flag, score, or model output.  A qualification run
that cannot provide all of those fields remains listed as supply, not silently
promoted into the roster.  This offline code verifies exact bindings and
digests; the system that creates the qualification receipt must separately
authenticate the receipt issuer and verify its health and grading evidence.

## A split that grows safely

The roster uses an anchored task-family split:

1. Convert the immutable 75-task study split once into a sealed role anchor.
   The anchor records every original family role, including training roles, and
   the original task-key-to-family mapping.
2. On each catalog refresh, inherit the anchored role for every old family.
   A later version of an old final-test family stays final-test; it cannot be
   reconsidered as a training example.
3. Assign only genuinely new, qualified families with a deterministic,
   metadata-only allocator.  It balances reviewed task labels and enforces a
   declared maximum share for one family.
4. Refuse a split if the requested train/development/final-test proportions
   cannot be achieved while keeping the anchor, or if the reviewed labels
   cannot be represented honestly.  The correct remedy is to qualify a more
   diverse batch, not to move a holdout family.
5. Derive a new protected-family lock from the resulting split.  Every later
   admission and materialization step checks that lock.

This makes a broad roster reproducible and family-safe even as the catalog
grows from dozens toward thousands of versions.

The two local, CPU-only commands make that sequence concrete:

```sh
uv run --locked cyber-post-train data-fleet-freeze-role-anchor anchor-request.json
uv run --locked cyber-post-train data-fleet-roster roster-request.json
```

The first command validates the frozen v2 study split against its exact old
inventory and writes a create-once role anchor.  The second command accepts a
sealed, content-free supply catalog, a sealed qualified subset, and that role
anchor.  It writes a generic metadata inventory, exact runtime bindings, an
anchored family split, and a protected-family lock for the existing collection
renderer.  Neither command launches a rollout or reads a trace.

## Outcome-blind collection and the 20M target

Roster choice, split assignment, model choice, attempt count, and sampling
budget are fixed before any rollout result is examined.  That is what
"outcome-blind" means here.  It does **not** mean that failed rollouts become
training examples: after the predeclared campaign ends, only authoritative,
verifier-confirmed successful sessions can enter the corpus.

The collection packet declares a target of at least 20 million unique
supervised tokens.  The campaign planner should use previous **aggregate**
success-rate and visible-action-token measurements to choose a sufficiently
large roster and fixed number of attempts.  It should size from a conservative
lower-bound estimate, then preserve the actual aggregate coverage receipt.
The private corpus materializer will retain a smaller result as evidence but
will not mark it ready for SFT until the 20M target and family-concentration
checks pass.

## Two deliberately separate data lanes

### Teacher visible-action lane

For a stronger teacher, use the existing visible-action packet:

- exact teacher model and authorization receipt;
- OpenCode, 262K context, and the approved action tool surface;
- predeclared attempts across only training families;
- verifier-confirmed successful sessions;
- no private reasoning and no opaque compacted transcript;
- visible tool actions rather than a final-report-only corpus.

This lane can feed the current private corpus materializer after admission.

### Future Qwen self visible-reasoning lane

Qwen self collection should use the same broad anchored roster and OpenCode
environment, but it is a separate future packet and materializer contract.  It
may contain only reasoning explicitly intended to be visible to the student;
private, unknown, hidden-thinking, or opaque-compaction fields remain
forbidden.  Its exporter must prove which exact messages are student-visible,
how continuations were represented, and how those targets were masked.

The current materializer deliberately rejects this lane rather than treating
visible reasoning as if it were ordinary tool output.  That prevents an
accidental mixture of teacher action imitation and a different self-reasoning
objective.  A separate schema, tests, and acceptance receipt are required
before self visible reasoning can be admitted to SFT.

## Practical next sequence

1. Produce sealed metadata-only qualification records for a large, diverse
   batch from the 1,093-version supply catalog.  Do not inspect or export
   traces during this stage.
2. Build the qualified broad roster, anchored split, runtime bindings, and
   protected-family lock offline.  Review the aggregate family and taxonomy
   coverage.
3. Render independent teacher-action and Qwen-self collection packets.  Verify
   source authorization before a launch; do not use an opaque receipt digest as
   evidence that the issuer was checked.
4. Collect the predeclared campaigns.  Export only sealed attempt metadata to
   admission; keep raw records private.
5. Admit successes, materialize action-only data, and use aggregate coverage to
   decide whether the 20M gate is met.

No production collection is authorized merely by completing steps 1 or 2.
