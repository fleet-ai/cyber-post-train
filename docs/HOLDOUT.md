# Blackbox data and holdout qualification

`training.qualify.components` joins shared-atom versions;
`training.qualification_order` replays frozen order. Both are metadata-only;
authenticate their input receipts independently.

## Three separate gates

1. **Historical SFT:** exact version, finite verifier success, intact trace,
   retained report/tools; today's breakage cannot undo it.
2. **Runnable now:** production version starts, tools work, positive/negative
   grading and cleanup pass; old `clean` labels are insufficient.
3. **Independent holdout:** shared-atom versions stay in one family across
   TRAIN, teacher DEV, task DEV and final; exclude unknown lineage. Four
   attempts still count as one family.

## Frozen source and task counts

September 25 census: 1,034 blackbox versions lacked the old 75 execution
receipts; 671 had new keys. Exact task/verifier/env/atom metadata GETs passed.
Excluding Teacher3K overlaps and protected families left **408 candidate
versions / 311 families / 14 apps**. Metadata is not runnability;
the frozen roster was a read-only census, not an environment check.

Production now exposes durable create-request claims. One rank-ordered,
model-free production check passed startup, tools, negative grading and exact
cleanup on September 25. The second stopped cleanly after an operational
error before tool verification; its cause is not yet attributable to the task.
Neither proves positive solvability, so no task was promoted to a runnable
holdout. Registry/Lanes supplied no exact positive record. A sealed,
DEV-family-first 16-cell roster is frozen. Its manifest authorizes no bulk
launch; after the first two cells' exact cleanup audits, the runtime checker
permits the 16 frozen DEV representatives one at a time as bounded, model-free
diagnostics after fresh live preflight.
That first DEV cell passed startup, both task-tool probes, negative grading,
and exact cleanup on September 25 (terminal SHA-256
`2f79d4bbf13d1a13b08490d0d9fd58860607f69dabe98522b9bc91d35e7b9e2b`;
independent instance readback `stopped`). It is **not** a positive solve or a
qualified holdout task.
The second DEV cell passed the same operational checks and independent stopped
readback (terminal SHA-256
`74dd660ace558d09e123fe944fc98895d50e1b94300d2e68af5ae62f6386997f`).
It also has no positive solve; qualified-task count remains zero.

A census found 257 full-credit leads on 68 keys. Private exact-version checks
proved **20 historical positive versions/18 families** (six DEV, 12 final;
17 medium, three hard). Of 408 candidates, 340 had no positive lead; strict
older-version contract/seed-byte proof transferred two other leads but added
only **one final family**. Historical union: **19 families, six DEV/13 final**;
current production runnability remains unproved. A later metadata-only scan of
252 named session stores found 21 leads on six counted versions (not global);
no new families. Staging cell 08 passed startup/tools/negative grade/cleanup only.
Old wave16 predates frozen rank order; no broader wave proved prod use.
Old 75 receipts/model outcomes are not automatic positives; preserve unknowns.

## Teacher3K family roles

2,886 whole-session successes span 496 keys/~57.4M estimated source tokens
and **356 strict shared-atom families**. Six versions/16 sessions overlap five
protected families and are quarantined. Teacher DEV reserves 37 families in
seven apps; missing labels preclude env/difficulty/vulnerability balance claims.

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
