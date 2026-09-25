# Blackbox data and holdout qualification

`training.qualify.components` joins exact task versions by shared atoms;
`python3 -m training.qualification_order <file>` replays the frozen candidate
order. Both use metadata, not live task execution. Authenticate their input
receipts independently: a JSON assertion alone proves nothing.

## Three separate gates

1. **Historical SFT source:** prove exact task version, completed verifier,
   genuine finite success, intact trace/receipt digests, retained report, and
   model-facing tool contract. Today's breakage cannot undo an old success.
2. **Runnable now:** prove current production/verifier status and exact-version
   start, tool, verifier, outcome, cleanup, and intended-path grading. An old
   receipt or `clean` QA label is insufficient; proof expires with time.
3. **Independent holdout:** keep every key version and shared-atom task in one
   family across `train`, `teacher_validation`, `dev`, `final_test`. Unknown
   lineage is inadmissible. Four attempts are still one independent family.

## Frozen source and task counts

The September 25 census found 1,034 blackbox versions without the old 75
execution receipts. Of these, 671 have new task keys; exact-version task,
verifier, environment and atom metadata GETs succeeded. Excluding Teacher3K
atom overlaps and protected families leaves **408 candidate versions / 311
families / 14 apps**. Metadata does not prove current runnability, and
`training/qualify_live.py` never runs environments.

One API canary created/deleted an environment but could not reconcile its
create-request ID. Theseus #35155 repaired source; staging advertised it,
production returned 404. `training/runtime_qualify.py` remains create-disabled
pending deployment. No production qualification wave ran. The first 16 lack
Pipeline Lanes/solvability receipts, and the 311-family pool lacks complete
intended-path positive grading, negative-control and cleanup proof. Never
select or reject tasks by baseline model outcomes.

A single staging-only, exact-version rehearsal later proved startup, the two
tools, a zero-credit negative verifier, and cleanup; it did **not** qualify the
task. Private inspection of that task's published source found a static
intended-request sketch and source-level tests, not an executable witness that
captures a flag through the intended path and earns live verifier credit. The
older 75 accepted-execution receipts likewise do not establish positive
grading; zero of them have been independently positive-qualified here. Ask
the task authoring/solvability owners for exact-version, sealed intended-path
witnesses or prior solvability receipts, and keep unknown cases `unproven`, not
`broken`. Do not select holdout tasks by whether a baseline or candidate model
solved them.

## Teacher3K family roles

The 2,886 accepted whole-session successes span 496 task keys and about 57.4
million estimated supervised tokens. The historical
audit found 370 components with exact atom-version locators; joining all
versions of each atom yields **356 stricter families**. Five protected families
were exposed through six source versions (16 sessions, 381,734 estimated
supervised tokens), so those source versions are quarantined as `test`.
`training/family_roles.py` reserves 37 teacher-validation families across all
seven source apps before packing. Missing source labels prevent a claim of
representation by environment, difficulty, or vulnerability type.

The frozen metadata-only roster is
`configs/data/qwen38-teacher3k-family-roles-20260925-v1.json`, canonical
SHA-256 `6c56b9b1ae9c0e5b21a36e48f0d5b6a451bda9da24ecab8262cb737acb1e2b68`.
Across all **1,176** exact source versions: train has 1,057 versions / 2,671
sessions / 52,569,390 estimated source tokens; teacher validation has 113 /
199 / 4,433,757; quarantine has 6 / 16 / 381,734. These are source totals,
not repacked target-token counts. A truthful new historical-packer role-anchor
binding is still required before corpus materialization. Only **20** existing
Fleet heldout families (13 development, 7 final) are Teacher3K-clean and
provisionally receipt-backed; they need fresh runtime/grade checks and are not
a powered final test for a modest 10-point lift.

## Frozen next qualification order

`configs/data/fleet-blackbox-qualification-order-20260925-v1.json` freezes
408 exact versions / 311 task-key/shared-atom families, payload SHA-256
`4e654dab459819dd1afeb043ca10a854c703795a50ef7df011527dd5a2c3703f`.
No model outcome or runtime result selected its order or roles. The order
alternates apps and favors underrepresented metadata; its first 16 families
span all 14 apps (14 medium, two hard). Sixty families are reserved for
development and 251 for final test; all apps occur in both. Difficulty labels
overall: 17 easy, 256
medium, 37 hard, one floor (final); development: 3 easy, 49 medium, 8 hard.
Source-project labels are 199 `ots`, 101 `cyber-ots`, 11 `apollo`. Atom-name
vulnerability keywords are **not** a reviewed taxonomy: 181 families are
unclassified. Only one family contains multiple atoms.

These reservations are not valid eval tasks. Each exact version still needs
independent intended-path/grading evidence plus current start, tool, verifier,
negative-control and cleanup receipts. Preserve failures and ambiguity. If
fewer than 100 independent final families pass, report an underpowered final
set and seek new families; never promote development tasks or relax proof.
