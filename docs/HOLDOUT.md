# Blackbox data and holdout qualification

`training.qualify.components` joins versions by shared atoms; the
`training.qualification_order` module replays the frozen candidate order.
Both are metadata-only; independently authenticate their input receipts.

## Three separate gates

1. **Historical SFT:** exact version, finite verifier success, intact trace,
   retained report and model-facing tools; today's breakage cannot undo it.
2. **Runnable now:** current production/version, start, tools, positive and
   negative grading, cleanup; an old receipt or `clean` label is insufficient.
3. **Independent holdout:** all versions/shared-atom tasks stay in one family
   across TRAIN, teacher DEV, task DEV and final; unknown lineage is excluded.
   Four attempts remain one independent family.

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

One staging rehearsal proved startup, tools, zero-credit negative grading and
cleanup, **not** intended-path positive grading. Its source contained only a
static request sketch, not a live positive witness. The old 75 receipts are
also not automatically positive-qualified. Seek exact-version successful
executions or sealed author witnesses; keep unknown `unproven`, not `broken`.
Never select holdout tasks by baseline/candidate outcomes.

## Teacher3K family roles

The 2,886 whole-session successes span 496 keys and ~57.4M estimated source
tokens. Shared-atom lineage yields **356 strict families**; six source
versions/16 sessions leak into five protected families and are quarantined.
The split reserves 37 teacher-validation families across seven apps. Missing
labels prevent claims of env/difficulty/vulnerability representativeness.

Frozen metadata-only roster: `configs/data/qwen38-teacher3k-family-roles-20260925-v1.json`
(SHA-256 `6c56b9b1ae9c0e5b21a36e48f0d5b6a451bda9da24ecab8262cb737acb1e2b68`).
Across 1,176 exact versions: TRAIN 1,057 versions/2,671 sessions/~52.6M
source tokens; teacher DEV 113/199/~4.4M; quarantine 6/16/~0.38M. These are
**not** native masked-token counts. The 20 older heldout families (13 DEV/7
final) still need fresh checks and cannot power a modest 10-point lift.

## Frozen next qualification order

`configs/data/fleet-blackbox-qualification-order-20260925-v1.json` freezes
408 versions/311 families (SHA-256
`4e654dab459819dd1afeb043ca10a854c703795a50ef7df011527dd5a2c3703f`).
Its outcome-blind order alternates apps; first 16 span all 14. Sixty families
are provisional DEV, 251 final; all apps occur in both. Difficulty: 17 easy,
256 medium, 37 hard, one floor; 181 lack reviewed vulnerability labels.

These reservations are not valid eval tasks. Each exact version still needs
independent intended-path/grading evidence plus current start, tool, verifier,
negative-control and cleanup receipts. Preserve failures and ambiguity. If
fewer than 100 independent final families pass, report an underpowered final
set and seek new families; never promote development tasks or relax proof.
