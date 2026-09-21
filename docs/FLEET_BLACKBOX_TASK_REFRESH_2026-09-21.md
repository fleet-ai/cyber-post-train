# Fleet blackbox security-task refresh — 2026-09-21

## Result

Fleet's current OTS Cyber catalog contains **1,676 exact selected task
versions**. Of its **1,288 production versions**, **1,093** explicitly identify
as blackbox security tasks.

This is a real increase in available catalog supply, but it is **not yet an
increase in usable training data**. The catalog added 38 production blackbox
tasks since the September 15 observation. None of those 38 has an exact
completed execution record in this refresh. They remain unknown rather than
being treated as either healthy or broken.

The conservative, execution-proven set remains **75 exact task versions** with
the same selection digest as before. The 17 reviewed candidates that still need
end-to-end proof are also unchanged. Therefore this refresh does not change any
current SFT, RL, development, or final-test split.

## Reproducible files

- [All 1,093 current production blackbox task versions](../configs/data/fleet-blackbox-current-production-20260921-v1.json)
- [The unchanged 75-task execution-proven set](../configs/data/fleet-blackbox-current-high-quality-20260914-v1.json)
- [The unchanged task-family-safe study split](../configs/data/fleet-blackbox-current-study-split-20260914-v2.json)
- [Current successful-demonstration coverage](../configs/data/fleet-blackbox-training-coverage-20260921-v1.json)
- [Complete metadata-only catalog census](evidence/fleet-task-inventory-20260921/catalog-census-v1.json)
- [Read-only refresh program](../scripts/refresh_fleet_task_inventory.py)

The new dated files are immutable observations. They retain task and version
identities only; they contain no task instructions, traces, answers, flags,
scores, credentials, or grader source.

## What changed

| Measure | 15 September | 21 September | Change |
| --- | ---: | ---: | ---: |
| Catalog task/version pairs | 1,633 | 1,676 | +43 |
| Production versions | 1,250 | 1,288 | +38 |
| Production blackbox versions | 1,055 | 1,093 | +38 |
| Known-broken blackbox versions | 52 | 52 | 0 |
| Candidates awaiting complete proof | 17 | 17 | 0 |
| Execution-proven high-quality versions | 75 | 75 | 0 |

The 38 additional blackbox versions have no exact current-version execution
receipt in the safe metadata view. Across all 1,093 blackbox tasks, the current
receipt check still finds 44 tasks with one proven successful demonstration and
36 tasks with one proven genuine model failure. The other 1,013 have no
current-version receipt in this audit. A genuine model failure can be valid
data; an absent receipt is not evidence that a task is broken.

The 75 accepted tasks still contain 42 with a known successful demonstration
and 33 with a known genuine model failure. This is unchanged. The complete
selection digest is still
`sha256:23cfc25af183ff474e914443682e4707162110ddc2aa16103a13d984fe05b83c`.

## What this does and does not establish

The metadata-only census confirmed that every catalog item has an attached
automatic answer check. That is necessary, but it does not prove that the
website starts, the answer check can run, the result is saved, and cleanup
finishes. The 75-task set has all of that additional proof. A published source
record or a task marked production is likewise a useful supply signal, not a
training-quality certificate.

The Artifact Registry has 3,084 source keys, including 385 newer source keys.
Those numbers must not be added to the catalog count: source records are not
accepted runnable tasks.

## Safe high-throughput collection plan

The right way to grow the corpus is to move tasks through two separate queues:

1. **Prove tasks before using them.** Start with the 17 reviewed candidates,
   then consider the 1,024 blackbox tasks that have not yet been reviewed. For
   each candidate, run a bounded, isolated end-to-end qualification attempt
   that records the exact task and environment version, answer-check identity,
   finite grading result, and cleanup. This is a task-health check, not a model
   training rollout. A task enters the usable set only after the complete
   evidence joins; a failed website or grader remains excluded.
2. **Freeze a versioned study set before collecting model traces.** Keep every
   version of one `(application, task family)` together. New tasks belong in a
   new inventory and a new split; never rewrite the existing 75-task study or
   move its final held-out families after looking at model outcomes.
3. **Collect training traces only from the new split's training families.**
   Development and final-test families remain entirely out of teacher SFT,
   self-SFT, preference data, and RL prompts. Successful teacher and Qwen
   traces each need their own complete trajectory and verifier-quality checks;
   a task being healthy does not automatically make every rollout useful.
4. **Use one creator and a shared rollout queue.** The PostgreSQL campaign
   queue can keep many workers busy without duplicate attempts. It must be the
   only creator for a frozen route, record the task/version and harness binding
   for every attempt, and quarantine ambiguous or incomplete results rather
   than silently retrying them.
5. **Promote in batches, not by wishful counting.** Once a meaningful batch of
   newly qualified task families is available, build a fresh family-safe split,
   collect broad teacher and student successes from its training portion, and
   reserve the new development/final portions for evaluation. This grows both
   the amount and diversity of data without leaking held-out tasks into
   training.

This plan can run qualification and trace collection in parallel: the existing
75-task training split remains available while the candidate queue is being
proved. It does not launch any workload as part of this refresh.

## Method and safety boundary

The refresh used read-only metadata endpoints, verified the account belongs to
Fleet, and checked that project membership stayed unchanged before and after
per-task reads. It required one current selected version per task, production
membership to agree with the task-quality endpoint, and an attached answer
check. The receipt-coverage pass used only safe session summaries to determine
whether a pre-bound exact execution was a complete success or a genuine model
failure; it did not retain numbers, identifiers, or any private content.

No jobs were launched, changed, or cancelled.
