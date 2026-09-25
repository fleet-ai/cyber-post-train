# Blackbox data and holdout qualification

`training.qualify.components` groups reviewed task versions by shared atoms.
The frozen current candidate order is replayed by
`python3 -m training.qualification_order <file>`; runtime qualification is a
separate live test. Neither metadata step certifies a runnable task.

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

The retired all-in-one metadata report was not an admission authority. The
current tools keep the decisions separate: source receipts prove historical
success, family roles prevent train/test leakage, and exact-version runtime
receipts prove what worked at a particular time. All input summaries must be
independently authenticated; a JSON assertion alone is not proof. Frozen roles
are `train`, `teacher_validation`, `dev`, and `final_test`.

For the September 24 frozen census, the report finds 1,217 blackbox versions. Seventy-
five have prior exact-version execution receipts; another 33 have reviewed
lineage but no complete runtime receipt. The **1,035** `not_analyzed` versions
without prior exact receipts are the discovery pool. First fetch each exact
current version and its atom metadata, then join shared-atom families, then
obtain bounded model-free runtime receipts. The old Teacher3K lineage map
already resolves 194 of those 1,035 exact versions; 841 need initial atom
lineage review. All 1,035 still need current-status and runtime checks before
they can enlarge the runnable heldout. Do not confuse this report with a fresh
runtime test: no new jobs or evaluations were launched to make it.

On September 25, a read-only live QA refresh found 1,109 `not_analyzed`
blackbox versions, one fewer than the frozen census; 26 were `agent_failure`,
one more. Excluding the same 75 prior exact receipts leaves 1,034 unreviewed.
Of those, 363 share a Teacher3K task key; 671 have new keys. Exact-version
metadata GETs succeeded for all 671, including production status, verifier,
environment version, and atom-source locators. After excluding Teacher3K atom
keys, 433 versions across 14 apps in 327 independent atom-key families remain as **possible**
new heldout candidates. That is a discovery count, not a runtime-qualified
count. The 20 existing live-heldout and 16 conditional candidate families must
also be excluded before fixing a new wave. `training/qualify_live.py` performs
this current metadata-only selection; it never creates environments.
After protected-family exclusion, 408 versions across the same 14 apps and
311 independent families remain. The frozen 16-family first wave is metadata
only. A direct API canary on September 25 created one exact environment and
proved its deletion, but did **not** qualify the task: version-scoped rollout
provisioning ignored the caller's create-request ID, so the durable claim could
not be reconciled. Theseus PR #35155 repaired this in source and staging now
advertises the exact capability, but production still returned 404 on September
25. `training/runtime_qualify.py` remains create-disabled until production
readback. Its runtime receipt is deliberately not task acceptance: a positive
intended-path/grade proof is also required. The first 16 candidates have no
Pipeline Lanes and no solvability-stage Registry run, so neither the 16-wave nor
the broader 311-family pool is yet a runnable heldout set.

Teacher3K had 2,886 accepted whole-session successes, but its old packing
clipped most windows inside messages, used tool names that differed from the
OpenCode evaluation, and salvaged 149 prefixes, 148 without a retained report.
Rebuild SFT examples only after preserving complete task/tool context and
verifying success in the retained segment. The historical audit groups its
496 task keys into 370 components using exact atom-version locators;
stripping version suffixes merges those into **356** stricter base-atom
families. `training/family_roles.py` splits by those 356 and requires the
reviewed Fleet holdout split and exact receipts. Five protected families were
exposed by six teacher source versions (16 sessions, 381,734 tokens); all
source versions in those families are quarantined as `test`, never trained or
used for teacher validation. Its 37-family teacher-validation slice covers all
seven source apps, but cannot claim representation on environment, difficulty,
or vulnerability type without those missing source labels. Split by these
families, not by session or task-key spelling. Reserve teacher validation
before packing, and keep both that slice and the live development/final tasks
out of all SFT and RL inputs. The existing 20 Teacher3K-clean live families
are a useful debugging panel, **not** a powered test for a modest 10-point
improvement. Four attempts per family do not make four independent tasks.
Expand and freeze a genuinely new final set before repeated checkpoint tuning.

The frozen metadata-only role file is
`configs/data/qwen38-teacher3k-family-roles-20260925-v1.json` (SHA-256
`6c56b9b1ae9c0e5b21a36e48f0d5b6a451bda9da24ecab8262cb737acb1e2b68`).
It covers all 1,176 exact teacher source versions: train has 1,057 versions,
2,671 sessions, and 52,569,390 source supervised tokens; teacher validation
has 113 versions, 199 sessions, and 4,433,757 tokens; quarantine has six
versions, 16 sessions, and 381,734 tokens. These are source-lineage totals, not
repacked training-window counts. A separate historical packer roster still
needs a truthful new role-anchor binding before corpus materialization.

## Frozen next qualification order (September 25)

`configs/data/fleet-blackbox-qualification-order-20260925-v1.json` freezes
408 exact candidate versions in 311 transitive task-key/shared-atom families
(canonical payload SHA-256 `4e654dab459819dd1afeb043ca10a854c703795a50ef7df011527dd5a2c3703f`).
Its candidate rows came from one read-only live QA/exact-version census, after
excluding Teacher3K and the earlier protected Fleet families. No model outcome
or runtime-qualification result chose its order or roles. Verify its digest and
recompute the order with `python3 -m training.qualification_order <file>`.

The fixed order alternates applications and prefers less represented metadata
within each app. It spans all 14 apps in its first 16 families. Sixty families
are reserved for development and 251 for final testing; **all 14 apps occur in
both roles**. The overall difficulty labels are 17 easy, 256 medium, 37 hard,
and one floor; development has 3 easy, 49 medium, 8 hard. The unique floor
case is reserved for final testing. Source-project labels
are 199 `ots`, 101 `cyber-ots`, and 11 `apollo`. The vulnerability-class column
is only a keyword hint from atom names, **not** a reviewed vulnerability
taxonomy: 181 families are unclassified. Only one family has multiple atoms.

These are reservations, **not 251 valid final tasks**. Each exact version still
needs accepted, independent positive solvability/grading evidence, plus current
environment startup, tool, verifier, negative-control, and cleanup receipts;
without positive evidence it remains provisional. A baseline model failure never makes
a task invalid, and a baseline success must not be used to select the roster.
Keep every failed/ambiguous qualification visible. If fewer than 100 independent
final families pass these gates, report the final set as underpowered and seek
new task families; do not promote development tasks or relax the proof gate.
