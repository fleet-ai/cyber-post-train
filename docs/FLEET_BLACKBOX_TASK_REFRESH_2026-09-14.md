# Fleet blackbox security-task refresh — 2026-09-14

## Result

The current OTS Cyber catalog contains **1,632 exact task versions**, but they
are not all blackbox tasks and they are not all known to work. The current
production catalog contains **1,054 task keys that explicitly identify
themselves as blackbox tasks**.

The conservative high-quality set is **75 exact task versions**. Each of these
75 has all four pieces of evidence:

1. an earlier independently checked attempt proved that the exact task started,
   grading finished with a real result, the result was saved, and the temporary
   environment was removed;
2. it is still the catalog's selected production version;
3. it still has an automatic answer check attached; and
4. the latest task-quality review does not classify it as broken.

This refresh deliberately does **not** treat “production” or “has an answer
check” as proof that a task works.

## Reproducible files

- [All 1,054 current production blackbox task versions](../configs/data/fleet-blackbox-current-production-20260914-v1.json)
- [The conservative 75-task high-quality set](../configs/data/fleet-blackbox-current-high-quality-20260914-v1.json)
- [The fixed 50/17/8 training, development, and final-test assignment](../configs/data/fleet-blackbox-current-study-split-20260914-v2.json)
- [Seventeen promising tasks that still need exact run proof](../configs/data/fleet-blackbox-qa-candidates-20260914-v1.json)
- [Task-key-only successful-demonstration coverage](../configs/data/fleet-blackbox-training-coverage-20260914-v1.json)
- [The complete 1,632-row catalog census and aggregate Registry evidence](evidence/fleet-task-inventory-20260914/catalog-census-v1.json)
- [The read-only refresh script](../scripts/refresh_fleet_task_inventory.py)

The 75-task file uses the repository's `task_versions` eligibility shape, so
training planners can bind it without rewriting task identities. Every row
contains stable task and version IDs plus reviewed application, environment,
task-family, vulnerability-family, and difficulty fields. It also records the
evidence class and source-file hashes. No label was inferred from a model
answer or invented from a task name.

## What each number means

| Current unit | Count | What it establishes |
| --- | ---: | --- |
| OTS Cyber catalog tasks and selected versions | 1,632 | Membership and exact current version |
| Production catalog versions | 1,249 | Platform lifecycle state |
| Production task keys explicitly marked blackbox | 1,054 | Current blackbox inventory |
| Registry task-source keys | 2,729 | Published source products, not runnable tasks |
| Registry source keys explicitly marked blackbox | 894 | Published blackbox source products, not catalog versions |
| Conservative current high-quality task versions | **75** | Prior run proof, current production selection, attached answer check, and no current broken-task finding |

These totals cannot be added together. A source publication, a catalog task, a
selected task version, and an independently checked execution are different
things.

## Current quality review

Fleet's task-quality endpoint summarizes automatic reviews of several model
attempts for the same exact task version. It returns only task identities and a
four-way status for this report; finding text and session contents were not
saved.

| Status among 1,054 current blackbox versions | Count | Decision |
| --- | ---: | --- |
| Known broken task | 52 | Exclude |
| Review found no task problem | 7 | Promising, but still needs the exact run-proof check below |
| Review attributed the problem to model behavior | 10 | A genuine model failure is useful, but the exact grading and cleanup proof is still required |
| Not yet analyzed | 985 | Unknown; neither accepted nor rejected |

The 7 + 10 reviewed candidates form the separate **17-task pending-evidence
file**. They are not mixed into the high-quality set.

The previous strict set contained 89 exact versions. Of those, 80 are still the
catalog's selected production versions. Five of those 80 are now marked as
broken, leaving the refreshed **75**. The other nine historical versions are no
longer the selected catalog versions; this refresh does not claim that they are
irrecoverably broken, only that they are not the current selection.

## What changed since 11 September

The earlier census recorded 1,593 catalog tasks. The stable read from
2026-09-15 00:28–00:30 UTC found **1,632**, an increase of **39**. Production
versions rose from 1,210 to **1,249**; staging remained 379 and discarded
remained 4. Every current row reports an attached answer check.

The Registry added **30 source keys** after the prior census cutoff:

- 7 explicit blackbox products;
- 8 whitebox repair products; and
- 15 unsuffixed source products.

Suffix grouping gives 15 source-name roots: seven pair an unsuffixed source with
a blackbox product and eight pair an unsuffixed source with a whitebox product.
This is useful publication accounting, but it is not proof of 15 independent
vulnerability families. The new source labels cover Current (10), Fakelook
(12), Fira (4), Fubspot (2), and Stratum (2).

The September 11 report did not preserve the 1,593 exact catalog IDs, so this
refresh cannot truthfully identify which 39 catalog rows are new by set
difference. The current file does preserve all 1,632 exact IDs so the next
refresh can make that comparison exactly.

## Composition of the conservative 75

- Applications: Current 21, Fira 19, Fakelook 17, Fentry 11, Fubspot 7.
- Environments: Current 21, Fira 18, Fakelook 17, Fentry 11, Fubspot 7,
  ROPS 1.
- Difficulty: 68 medium, 6 hard, 1 easy.
- Reviewed task-family values: 75 distinct values.
- Reviewed vulnerability-family labels: 95 distinct labels. These include
  overlapping category and class labels, so 95 is not a count of independent
  vulnerability types.

This metadata is sufficient to make a task-family-held-out assignment balanced
over application, environment, vulnerability labels, and difficulty. It does
not make the split application-held-out: tasks from one application appear in
more than one partition by design.

## Fixed representative split

The new versioned split assigns the 75 exact task versions once each:

- **50 training tasks**;
- **17 development tasks** for choosing settings; and
- **8 final-test tasks** that must remain untouched until a recipe is selected.

The generator groups by reviewed application plus task family, so related
versions cannot cross partitions. It gives each application, environment,
difficulty rating, and vulnerability label a proportional target, then uses a
deterministic rarity-first assignment and improving swaps. The assignment has
no exact task-version or reviewed-family overlap. Every label with at least five
tasks appears in all three partitions.

Small groups need an explicit rule because they cannot all appear everywhere:

- a one-task label is recorded where it lands and is not called representative;
- labels with two to four tasks are balanced when compatible with the larger
  groups; and
- labels with five or more tasks must occur in every partition.

The fixed file is validated by the new
`cyber_representative_study_split_v2` code in
[`training/study_split_v2.py`](../training/study_split_v2.py). This is separate
from the older production FTI validator that hard-codes 59/20/10; it does not
silently reinterpret an old experiment plan. Any future training launcher must
bind this file and use a newly versioned plan before running on the refreshed
data.

## Evidence still missing

The current production catalog reports no latest-environment status for any of
its 1,632 rows. The task-quality endpoint identifies 52 broken blackbox tasks
and 17 promising new candidates, but it does not return the complete proof that
the exact environment started, grading finished, a finite result was saved, and
cleanup completed.

Current modular task generation uses a separately authenticated Fleet Admin
view. Registry and catalog access work, but the browser session currently stops
at GitHub sign-in and a Fleet API key is not accepted by that admin endpoint.
Therefore the exact join from an accepted modular-task entry to its catalog
task/version and terminal execution proof remains unavailable in this refresh.

The next concrete step is to restore the Fleet Admin GitHub session, export only
accepted entry IDs and exact source/runtime references, and join them to:

1. current catalog task and version IDs;
2. exact environment and automatic-answer-check versions; and
3. independently checked terminal execution, grading, saved-result, and cleanup
records.

## Successful demonstrations currently available

The refresh also rechecked the one exact receipt-bound session for each of the
75 conservative tasks through Fleet's safe session-summary route. It used the
numeric result only to classify full success versus genuine model failure; no
numeric result or session ID is saved in the public artifact.

- **42 task keys** have an exact current-version receipt proving at least one
  full success. These are the immediately demonstrated candidates for SFT.
- **33 task keys** have an exact receipt proving a genuine model failure for
  the checked attempt.
- Across all 1,054 current blackbox keys, **44** have an exact proven success
  and **36** have an exact receipt for a genuine model failure. Five of those
  80 receipt-proven keys are excluded because they are now marked broken: two
  have a prior success and three have a prior failure.
- The other **974 current blackbox task keys** do not have an exact
  current-version execution receipt in this refresh.

Thus, 1,010 of the 1,054 current blackbox keys lack an exact proven success in
this evidence set. That does **not** mean 1,012 tasks are impossible or have
only failed. In particular, the safe session list does not expose the exact
task version, so it cannot rule out another successful attempt on the same key.
The report therefore leaves the “failure only” count unknown instead of
inventing it. The linked coverage file contains task keys only. It records the
44/36/974 current-inventory partition, the 42/33 high-quality subset, and the
two/three receipt-backed tasks excluded as currently broken.

The 17 pending candidates should be checked first. Once that join is complete,
newly proven tasks can be appended in a new immutable high-quality-set version;
the current 75-task file should not be silently rewritten.

## Public report source

The public project report is served from the repository's `gh-pages` branch at
[fleet-ai.github.io/cyber-post-train](https://fleet-ai.github.io/cyber-post-train/).
Its source is under [`site/`](../site/), and task-quality numbers are stored in
[`site/report-data.js`](../site/report-data.js). This branch updates the page to
the current 1,054 inventory / 75 evidence-qualified / 17 pending / 52 broken
accounting and links the fixed representative split.

## Safety boundary

This was a read-only metadata audit. It launched no jobs and read or persisted
no task prompts, benchmark content, model traces, answers, flags, credentials,
or scores.
