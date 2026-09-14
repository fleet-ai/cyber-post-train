# Qwen blackbox task filtering report — 2026-09-12

## Result

Filtering is complete for the saved list of 160 Fleet blackbox security tasks.
The strict high-quality group contains **89 tasks**. The other **71 remain
excluded**.

Here, a “task version” means one exact saved copy of a task. “All” means all 160
copies in this study list. It does **not** mean every Fleet cyber task that
exists today. Fleet has many more task records, but there is not yet one complete
list that connects every record to a working website, its hidden answer check,
and proof of a correctly completed test.

## Source files

1. [All 160 tasks reviewed](../configs/data/fleet-a62-task-split-v1.json)
   - 160 unique blackbox task keys and 160 unique task-version UUIDs
   - raw file SHA-256:
     `ab444cb664ffb4da351e4d33298cc052f848f2833487807bf984a753fa7c3b47`
   - fingerprint recorded inside the file:
     `sha256:fb09668f8de77e6baee37dc1e3162a7617ca9a0a450da9ca53422c17bfbc942a`
2. [The 89 tasks that passed](../configs/data/qwen-blackbox-eligible-v1.json)
   - 89 exact task versions
   - raw file SHA-256:
     `b12a755c7e65bc04694eab85d9b0c21fadc222207d7bcc6ed8f0c788e032589b`
   - fingerprint recorded inside the file:
     `sha256:c45c9cc420a3dbf1aa6b635b84ed1ca3c404254f046ffdac234e053929a1bf12`

The [detailed task list](../configs/data/qwen-blackbox-study-inventory-v1.json)
adds the application, website setup, type of weakness, difficulty, and task
family for the same 89 tasks. It helps us make balanced training and test
groups, but it does not change which tasks passed.

## What the filter checked

A task entered the high-quality group only when both checks passed:

1. **It worked before.** At least one independently checked record proved that
   the exact website started, the hidden answer check finished, scoring produced
   a real number, the result was saved, and the temporary website was removed.
2. **It can still be recreated.** The exact starting information, website
   setup, hidden answer-checking code, and scoring program are still available
   and match the earlier test.

The model's score did not affect selection. A genuine success and a genuine,
correctly scored failure both prove that the task worked. A broken website or
broken scorer proves nothing about the model.

## Disposition of the 160 versions

| Stage | Exact versions |
| --- | ---: |
| Tasks examined | 160 |
| All files needed to rerun the task are still present | 132 |
| Independently proven to have run and scored correctly | 98 |
| Passed both checks | **89** |
| Excluded | **71** |

The 71 exclusions have two mutually exclusive final reasons:

- **62** — no independently checked record proves that the exact task completed
  and scored correctly.
- **9** — an earlier test was proven, but required starting information is now
  missing.

A separate file check found 17 tasks without their starting information and 11
without the exact hidden answer check or scoring program. Do not add those 28 to
the 62 above: many tasks appear in both groups. The final 89 are the tasks that
both worked before and still have every file needed to run again.

## Composition of the accepted subset

- Current study split: **74 train, 13 development, 2 reserved development,
  0 original test**.
- Applications: **fira 24, current 22, fakelook 21, fentry 14, fubspot 8**.
- Difficulty: **82 medium, 6 hard, 1 easy**.
- All 89 have independently checked run records and reviewed labels for
  application, website setup, task family, type of weakness, and difficulty.

The vulnerability labels are semantically rich but contain synonyms and
different levels of specificity. They are suitable as reviewed stratification
metadata; they should not be presented as a perfectly normalized vulnerability
histogram.

## Representative study splits

The 89 tasks are assigned to two different balanced study groups. Each has 59
training tasks, 20 tuning tasks, and the same 10 final tasks that remain hidden
until the training recipe is chosen:

- [Split A](../configs/data/qwen-blackbox-study-split-a-v1.json)
- [Split B](../configs/data/qwen-blackbox-study-split-b-v1.json)
- [Common final-test lock](../configs/data/qwen-blackbox-study-final-test-v1.json)

The application, environment, vulnerability, and difficulty distributions were
balanced without using model outcomes, traces, scores, or loss. Split A and B
provide a robustness check rather than two unrelated experiments.

## Boundaries and next work

- Passing this review does not guarantee that a task is healthy at this moment;
  each new evaluation still performs a small launch check first.
- It does not automatically approve an individual teacher or self rollout as
  supervised training data; each saved attempt still needs its own quality
  checks.
- It does not prove that learning transfers to a completely new application or
  a completely new type of weakness.
- The 160-task list is not Fleet's entire current cyber library. The
  broader metadata census is documented in
  [Fleet task inventory](FLEET_TASK_INVENTORY_2026-09-11.md).

To expand beyond these 89, create a new saved list after the additional tasks
have all required files and independent proof of a correctly completed test.
Keep the current files unchanged so the study can be repeated exactly.

## Detailed evidence

The full technical record of the run checks, required files, group assignment,
and exclusions is in
[Qwen blackbox task eligibility](QWEN_BLACKBOX_TASK_ELIGIBILITY_2026-09-11.md).
