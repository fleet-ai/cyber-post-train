# Qwen blackbox task filtering report — 2026-09-12

## Result

Filtering is complete for the frozen 160-version Fleet blackbox study roster.
The strict high-quality subset contains **89 exact task versions**. The other
**71 versions remain excluded**.

"All" in this report means all candidates in that historical study roster. It
does **not** mean every task currently present in Fleet's live cyber Registry or
catalog. The live systems contain many more source/catalog records, but there is
not yet one evidence-complete manifest joining every record to a blackbox
projection, immutable runnable environment, verifier, and accepted execution
receipt.

## Canonical manifests

1. [All 160 historical study-roster candidates](../configs/data/fleet-a62-task-split-v1.json)
   - 160 unique blackbox task keys and 160 unique task-version UUIDs
   - raw file SHA-256:
     `ab444cb664ffb4da351e4d33298cc052f848f2833487807bf984a753fa7c3b47`
   - embedded manifest digest:
     `sha256:fb09668f8de77e6baee37dc1e3162a7617ca9a0a450da9ca53422c17bfbc942a`
2. [Filtered high-quality subset](../configs/data/qwen-blackbox-eligible-v1.json)
   - 89 exact task versions
   - raw file SHA-256:
     `b12a755c7e65bc04694eab85d9b0c21fadc222207d7bcc6ed8f0c788e032589b`
   - embedded self-digest:
     `sha256:c45c9cc420a3dbf1aa6b635b84ed1ca3c404254f046ffdac234e053929a1bf12`

The [enriched study inventory](../configs/data/qwen-blackbox-study-inventory-v1.json)
adds verified application, environment, vulnerability-family, difficulty, and
task-family metadata to the same 89 versions. It is useful for representative
splitting, but it is not a third eligibility authority.

## What the filter checked

A task version entered the high-quality subset only when both gates passed:

1. **Historical execution evidence.** At least one independently
   digest-validated, accepted terminal execution proved that the exact
   environment ran, the verifier completed, the grade was finite, and the
   ingest and cleanup path completed.
2. **Current reproducibility.** The exact current starting data, environment,
   runtime seed, and verifier bindings still resolve and match the accepted
   historical execution.

The filter is outcome-blind. A genuine exploit success and a genuine graded
model failure are both valid evidence that the task environment and grader
worked. Model success rate did not decide eligibility.

## Disposition of the 160 versions

| Stage | Exact versions |
| --- | ---: |
| Historical study roster | 160 |
| Complete current immutable bindings | 132 |
| Independently verified accepted execution receipt | 98 |
| Passed both gates | **89** |
| Excluded | **71** |

The 71 exclusions have two mutually exclusive final reasons:

- **62** — no exact historical execution receipt was independently verified.
- **9** — historical execution was verified, but the current starting-data
  binding is absent.

The separate current-binding audit found 17 versions without a starting-data
binding and 11 without an immutable runtime-seed or verifier pin. Those 28 are
not additive to the 62 above: many also lack accepted historical evidence. The
89 is the intersection of the 98 receipt-proven versions and the 132 versions
with complete current bindings.

## Composition of the accepted subset

- Current study split: **74 train, 13 development, 2 reserved development,
  0 original test**.
- Applications: **fira 24, current 22, fakelook 21, fentry 14, fubspot 8**.
- Difficulty: **82 medium, 6 hard, 1 easy**.
- All 89 have accepted certification and verified application, environment,
  task-family, vulnerability-family, and difficulty metadata.

The vulnerability labels are semantically rich but contain synonyms and
different levels of specificity. They are suitable as reviewed stratification
metadata; they should not be presented as a perfectly normalized vulnerability
histogram.

## Representative study splits

The 89 eligible versions are frozen into two outcome-blind outer splits. Each
has 59 training tasks, 20 Fleet development tasks, and the same sealed 10-task
final set:

- [Split A](../configs/data/qwen-blackbox-study-split-a-v1.json)
- [Split B](../configs/data/qwen-blackbox-study-split-b-v1.json)
- [Common final-test lock](../configs/data/qwen-blackbox-study-final-test-v1.json)

The application, environment, vulnerability, and difficulty distributions were
balanced without using model outcomes, traces, scores, or loss. Split A and B
provide a robustness check rather than two unrelated experiments.

## Boundaries and next work

- This is task-version validity evidence, not blanket launch authorization.
- It does not automatically approve an individual teacher or self rollout as
  SFT data; each session still needs model, trace, context, and harness checks.
- It is not a fresh live-health test of every environment today.
- It does not prove application- or vulnerability-family-disjoint generalization.
- The 160-version roster is not Fleet's entire current cyber library. The
  broader metadata census is documented in
  [Fleet task inventory](FLEET_TASK_INVENTORY_2026-09-11.md).

Expanding beyond these 89 should create a new versioned allowlist after the new
task versions gain complete immutable bindings and accepted exact execution
receipts. The current v1 manifests should remain frozen for reproducibility.

## Detailed evidence

The full receipt, binding, split, and exclusion audit is in
[Qwen blackbox task eligibility](QWEN_BLACKBOX_TASK_ELIGIBILITY_2026-09-11.md).
