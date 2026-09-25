# Blackbox data and holdout qualification

`python3 -m training.qualify` is a **read-only metadata report**, not a job
launcher or a replacement for checking signed source receipts. It prints task
keys, exact version IDs, counts, and hashed lineage components; it never prints
prompts, traces, scores, answers, or flags. Supply only sanitized JSON.

There are three separate questions:

1. **Was an old teacher session genuinely successful?** Its exact historical
   task version, completed verifier, finite outcome, infrastructure validity,
   acceptance receipt, and trace digest must agree. Today's task status does
   not invalidate that old session. An SFT example additionally needs an
   intact trace, the successful report still present after any trimming, and
   proof that its model-facing tool interface matches the one we will evaluate.
2. **Can this exact task version run now?** It needs a current production
   catalog readback with attached verifier, plus a bound model-free receipt
   proving environment startup, both tools, verifier execution, a finite
   outcome, and cleanup. An old success receipt or a `clean` QA label alone is
   not this proof. Even a complete receipt describes its observation time,
   not an indefinite guarantee.
3. **Is it independent of training?** All versions of one task key and every
   task sharing a reviewed vulnerability patch are one connected family. A
   family cannot cross train, teacher-validation, development, or final test.
   The tool does not admit unknown atom lineage and reports teacher-exposed heldout
   tasks instead of counting them as clean.

The CLI accepts a metadata-only current catalog at `--catalog` (`-` means
standard input), one or more reviewed `--lineage` files, an optional frozen
`--split`, `--teacher-lineage`, sanitized per-session `--sessions`, and exact
version model-free `--runtime` receipts. `--rosters` adds exact key/version
lists for follow-up; without it the output is compact. Session projections
need opaque SHA-256 session/receipt/trace digests and the five historical
success booleans named in `training/qualify.py`; SFT admission additionally
needs `trace_intact`, `retained_successful_report`, and
`tool_contract_proven`. Runtime projections need an exact receipt digest and
all five runtime booleans. **The producer must independently authenticate and
rehash those upstream receipts.** This tool checks identity, field completeness,
and lineage; it cannot certify a forged JSON assertion. It always emits
`launch_authorized: false`. `--teacher-lineage` means the lineage of a corpus
**already used to train a checkpoint** when auditing that checkpoint's live
holdout; do not pass an unsplit pool as if every candidate had been trained on.
Frozen roles are `train`, `teacher_validation`, `dev`, and `final_test`.

For the last frozen census, the report finds 1,217 blackbox versions. Seventy-
five have prior exact-version execution receipts; another 33 have reviewed
lineage but no complete runtime receipt. The **1,035** `not_analyzed` versions
without prior exact receipts are the discovery pool. First fetch each exact
current version and its atom metadata, then join shared-atom families, then
obtain bounded model-free runtime receipts. The old Teacher3K lineage map
already resolves 194 of those 1,035 exact versions; 841 need initial atom
lineage review. All 1,035 still need current-status and runtime checks before
they can enlarge the runnable heldout. Do not confuse this report with a fresh
runtime test: no new jobs or evaluations were launched to make it.

Teacher3K had 2,886 accepted whole-session successes, but its old packing
clipped most windows inside messages, used tool names that differed from the
OpenCode evaluation, and salvaged 149 prefixes, 148 without a retained report.
Rebuild SFT examples only after preserving complete task/tool context and
verifying success in the retained segment. Its 496 task keys collapse into 370
reviewed shared-atom components; split by those components, not by session or
task-key spelling. Reserve a component-stratified teacher-validation slice
before packing, and keep both that slice and the live development/final tasks
out of all SFT and RL inputs. The existing 20 Teacher3K-clean live families
are a useful debugging panel, **not** a powered test for a modest 10-point
improvement. Four attempts per family do not make four independent tasks.
Expand and freeze a genuinely new final set before repeated checkpoint tuning.
