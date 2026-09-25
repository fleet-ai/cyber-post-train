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

Production now exposes durable create-request claims. An initial model-free
check passed startup/tools/negative grading/cleanup; another stopped after an
operational error. Neither proved positive solvability. A sealed 16-cell DEV
roster permits one-at-a-time diagnostics, not a bulk launch.
A read-only Registry search found no exact positive binding: 28 matching
atom-solvability runs (none positive), 29 atom-QA runs (12 green but not
solvability), and 54 positive-labelled mATG runs (zero candidate IDs).
Registry presence cannot promote this wave.
The September 25 DEV screen attempted all 16 frozen representatives, one at a
time: 12 passed startup, both task tools, zero-score negative grading, and
cleanup; two were rejected before creation because the exact task metadata
lacked `cyber_contract`; two reached `submit_report` but its harmless no-flag
probe failed (one explicit rejection, one unknown). All 14 created instances
were read back `stopped` with matching claims. Neither probe rejection nor
missing metadata proves breakage; without intended-path proof, **zero tasks**
were promoted.

| DEV cell | Result | Sealed terminal SHA-256 |
|---|---|---|
| 0–3 | pass | `2f79d4bbf13d1a13b08490d0d9fd58860607f69dabe98522b9bc91d35e7b9e2b`, `74dd660ace558d09e123fe944fc98895d50e1b94300d2e68af5ae62f6386997f`, `6ce9346e80c088dfdef036b5f5ec7201e4de7dfdb133a88e131634289c523d63`, `74007316d93942311df2dd9099cfb78fea68821c77809c27710794f049a92e49` |
| 4 | report probe indeterminate | `a3552141e1eaa136ab7ae7580f9908dc9a33127c8e98dc6d4abd89ea5cdd3834` |
| 5–8 | pass | `6fa1a53ac41f5a5a30d814c5f32c79aaad549ea8f89d01b812f33512ba918d53`, `16655473ec8b57fe2df3744b8cfa092203f89c7a59a9caa7c3575f6dbd6e0e1d`, `5ced3980896e52fd075ee71777f72e0ca1d7f9f6f1bb9882f56f0896b1427453`, `ebb196b1e49347c93a5009efb2b4ebc44987dff9b6156c96f9e6ce228a738474` |
| 9–10 | preflight rejection; no instance | n/a |
| 11–13 | pass | `b96a08935dfb84b9c997923d3c7476fce61b567f8491083838c50f80c5f478a0`, `2d1225306d619eeea2802fc157593a0bfdba7a9fac125e48475468e2581ed3e9`, `c7fd02da61ff0fae6232585ba4f59212606889166e20a24d3f2b1f9e90e1c211` |
| 14 | no-flag report rejected | `c0ccbaf48843951cb58e2d5583682ef420ce540812341ad8df8110abc7be2e66` |
| 15 | pass | `1d9a44f75a2bbc9105b395b595d0ce588ca2fc3255a17d317f9c31d9b21355e0` |

Two exact-version historical full-credit DEV paths were then replayed through
fresh current instances: cell 0 (47 bash calls, terminal
`0853aeedc42b1e1d43c195f0eda17ea053de0633cc4b3a79a88c4b0277c9c6a7`)
and cell 3 (67 bash calls, terminal
`4ad22edbf4414df38d34273edcbd44e6039f8a389ab15edf6ba86adb6c237492`).
Both again earned full credit and were independently read back `stopped` with
matching durable claims. The later receipt also bound the authoritative grader
execution and score digest; the first recorded only full credit and authoritative
status. This is a **current intended-path replay**, not an independent model
solve or evidence that all family variants work. Cell 3 is the first strictly
recorded current-positive DEV version; final-test current-positive count is
still zero. Keep the remaining candidates provisional until comparable proof.

A census found 257 full-credit leads on 68 keys. Private exact-version checks
proved **20 historical positive versions/18 families** (six DEV, 12 final;
17 medium, three hard). Of 408 candidates, 340 had no positive lead; strict
older-version contract/seed-byte proof transferred two other leads but added
only **one final family**. Historical union: **19 families, six DEV/13 final**;
current production runnability remains unproved. A later metadata-only scan of
252 named session stores found 21 leads on six counted versions (not global);
no new families. Staging cell 08 passed startup/tools/negative grade/cleanup only.
Old wave16 predates frozen rank order; no broader wave proved prod use.
Six of the old 75 split's eight final versions have exact-version historical
full-credit leads and no Teacher3K TRAIN-family overlap; none is yet current-
positive. Old receipts/model outcomes are not automatic positives.

On September 25, read-only current-task checks found all 20 exact historical
positive versions still match their frozen task version, environment version,
blackbox projection, cyber contract, and attached verifier. This is binding
evidence, not runnable proof. A separate bounded fresh-instance replay of one
reserved **final-test** version used its exact historical 49-call intended path,
again earned full credit with authoritative grader agreement, and independently
read back `stopped` with its exact create claim. Its sealed terminal is
`sha256:4360776f9bd24ab2396297576fd5966657ae3235bf0135d160b7af556ee902e9`
(private file SHA-256 `df552493beb574b73b23a59912a42005e7c492559672b25b19e7677c25bad7e3`).
A second exact final-test version was independently replayed (73 bash calls),
with full credit, authoritative grader agreement, and stopped/claim readback:
terminal `sha256:1531e8c3b13e0f59d65b2d79fa13ec1ab97127e7130e745fae8c9802816f8876`
(private file SHA-256 `038217ee2b25617f14050a9e538825f48b2a4fc736197a0daa70dfd44f77ee1b`).
Six further final versions replayed 37/70/54/38/60/59 calls for authoritative
credit, with exact stopped-instance and create-claim readbacks. Terminals:
`sha256:b4b8e5b970862c18fcfb0715e6d45a2b3d78853ea3f90eb4b6d729c205dbe8a4`,
`sha256:93ebfc30039c468d58f044777db312f82983b0917303a03be010c4de689dd2c0`,
`sha256:f3224ac43d09d2eef508b9461c7d0a5f1da19f2a6ade7f8586fbf1ae20f73b17`,
`sha256:4a2554a82a8a6319a82013cac56b3492efa937f82b0a4555f174a088768aca8e`,
`sha256:4c8d8d030614e180c3419ce72815835cdda6d8812ebf3f5147fb9326522b8013`,
`sha256:b74f9c22cce26b8357e925ec6bea99a2513524040fbf109d1e5227e361220836`.
These **eight** current-positive families are not independent model solves or a sufficient panel.
Never use final actions, reports, or scores for training or checkpoint selection.
An additional final-version replay timed out after 76/84 historical bash calls,
before grading; its sealed terminal is
`sha256:9dcd36965083f24a80bdcd499a56dd5e4e66e68ad36fb0389607bcd891081985`.
Exact claim and stopped-instance readback prove release, not task invalidity.

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

September 25 DEV positive replay index 1 stopped at provision/cleanup without a
terminal receipt. The exact create claim later returned 404; exhaustive recent
queued, pending and running instance listings found no matching environment.
Do not retry that create ID or count this version current-positive. Resolve the
create/claim mismatch before another replay; no active allocation was observed.
DEV index 11 failed at environment creation before tools; claim absent, no live
instance in exhaustive listings. Index 12 failed likewise (terminal
`sha256:b8afcc0a424fd9a7607306d929fe8da342f40c5a748b9ec359c2db8cfef1f091`):
claim 404, no exact binding across all queued/pending/2,375 running instances.
These are create defects, not task-quality verdicts or positive proof.
