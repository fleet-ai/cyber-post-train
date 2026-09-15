# Fleet blackbox security-task refresh — 2026-09-15

## Result

Fleet's current **OTS Cyber** catalog—the Fleet project that groups the current
cyber tasks—contains **1,633 exact selected task
versions**. Of its **1,250 production versions**, **1,055** have task keys that
explicitly identify them as blackbox security tasks.

The conservative high-quality set remains **75 exact task versions**. It did
not grow in this refresh. Every retained version still has all of the evidence
required by the September 14 filter:

1. an independently checked earlier attempt proved that the exact task
   started, grading finished with a finite result, the result was saved, and
   cleanup completed;
2. the same exact version is still selected for production;
3. an automatic answer check is still attached; and
4. the current task-quality review does not classify it as broken.

The catalog gained one production blackbox task. Its current review status is
`not_analyzed`, so it remains outside the training, development, and final-test
sets. This is the intended conservative behavior: publication and production
status are useful supply signals, but they do not prove that an environment and
its grading work end to end.

### Later same-day recheck

A second metadata-only census ending at **16:16 UTC** found the same 1,633
selected versions, 1,055 production blackbox tasks, 75 retained tasks, and 17
pending candidates. The Artifact Registry contained two additional task-source
records, but neither had entered the authoritative task catalog. The training
split and public counts therefore remain unchanged. The compact
[recheck record](evidence/fleet-task-inventory-20260915/catalog-recheck-1616-v1.json)
binds the exact program, prior census, output digests, and zero-change result
without duplicating another 1,633-row census in Git.

## Reproducible files

- [All 1,055 current production blackbox task versions](../configs/data/fleet-blackbox-current-production-20260915-v1.json)
- [The unchanged conservative 75-task set](../configs/data/fleet-blackbox-current-high-quality-20260914-v1.json)
- [The unchanged 50/17/8 study split](../configs/data/fleet-blackbox-current-study-split-20260914-v2.json)
- [Current successful-demonstration coverage](../configs/data/fleet-blackbox-training-coverage-20260915-v1.json)
- [The complete 1,633-row metadata census](evidence/fleet-task-inventory-20260915/catalog-census-v1.json)
- [The separate Fleet Admin workflow count](evidence/fleet-task-inventory-20260915/admin-workflow-count-v1.json)
- [The read-only refresh program](../scripts/refresh_fleet_task_inventory.py)

The dated files are immutable observations. The older September 14 inventory
remains available so the change can be checked by exact task and version IDs.

## Exact change since the prior refresh

| Measure | 14 September | 15 September | Change |
| --- | ---: | ---: | ---: |
| Catalog task/version pairs | 1,632 | 1,633 | +1 |
| Production versions | 1,249 | 1,250 | +1 |
| Production blackbox versions | 1,054 | 1,055 | +1 |
| Known-broken blackbox versions | 52 | 52 | 0 |
| Promising candidates awaiting complete proof | 17 | 17 | 0 |
| Conservative high-quality versions | 75 | 75 | 0 |

The exact set difference contains one added task/version pair and no removals.
That new row is retained in the full inventory with its current
`not_analyzed` status. No task prompt or task content was needed to establish
this change.

Among all 1,055 current blackbox task keys, the existing exact receipts prove
44 complete successes and 36 genuine model failures. The remaining **975** do
not have exact current-version execution proof in this audit. The high-quality
75 still divide into 42 tasks with a proven successful attempt and 33 with a
proven genuine model failure. A genuine failure is valid data; it is not a
broken task.

## Which source answers which question

The three current systems have different jobs and their counts must not be
added together:

| Source | Current observation | What it establishes |
| --- | ---: | --- |
| OTS Cyber project membership plus per-task status | 1,633 selected versions | Current catalog membership, selected version, production/staging/discarded state, and answer-check attachment |
| Task-quality endpoint | 1,250 production rows, including 1,055 blackbox | A current automatic review category; `broken_task` is an exclusion signal, while other categories are not complete execution proof |
| Artifact Registry | 2,735 published task-source keys | Source products exist; it does not prove catalog registration or a working execution |
| Fleet Admin task-lifecycle page | 176 top-level production workflows whose IDs start with `matg-cyber` | Workflow accounting only; one workflow can contain several entries and a completed workflow is not automatically an accepted task |

The OTS Cyber catalog is therefore the authoritative current inventory used by
this report. The 75-task file is the stricter execution-proven selection used
for experiments. Fleet Admin is authenticated and readable now, but its page
still does not provide a score-free export of accepted entries joined to exact
catalog task/version, environment, answer-check, and cleanup evidence.

The Fleet Admin count rose from 153 in the September 11 observation to 176.
That **does not mean 23 new usable tasks**: the number counts workflows, not
unique accepted blackbox tasks.

## What is still missing

The current catalog added one task, but no new task can enter the 75-task set
without complete exact-version evidence. A durable expansion still requires a
safe metadata export that joins:

1. an accepted modular task entry and exact blackbox source version;
2. the exact catalog task and selected version;
3. the exact environment and answer-check versions; and
4. terminal environment startup, grading, finite-result persistence, and
   cleanup evidence.

Until that join exists, the 17 currently promising tasks are the first review
queue, and another 911 unproven, unreviewed tasks remain unknown. They must
not be silently treated as either good or broken.

## Method and safety boundary

The refresh used read-only metadata endpoints and required stable project
membership before and after the per-task reads. It verified Fleet-team account
identity, exact task/version uniqueness, production membership equality with
the task-quality rows, attached answer checks, immutable file hashes, and the
existing exact receipt-bound session summaries. The Fleet Admin count was read
from the authenticated production page with top-level-only filtering and the
exact workflow-ID prefix shown above.

No jobs were launched. No task instructions, model traces, answers, flags,
credentials, or grader source were read or persisted. The existing safe
session summaries were used only to classify a receipt-bound attempt as a full
success or a genuine model failure; their numeric values and session IDs were
not retained or exposed, and outcome did not affect task-quality selection.
