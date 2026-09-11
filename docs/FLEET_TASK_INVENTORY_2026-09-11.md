# Fleet cyber task supply — 2026-09-11

Read-only metadata census, observed **2026-09-11 18:14–18:22 UTC**, with a
separate authenticated-admin follow-up completed **18:27 UTC** and a final
score-blind exact-version census at **21:54–21:57 UTC**. No jobs,
environments, source publications, catalog edits, or training were started.
No task prompts, source bundles, traces, answers, flags, or scores were opened.

## Bottom line

- Our historical **160** is one frozen experiment roster: 160 distinct task keys
  and 160 exact task-version UUIDs. It is not the size of Fleet's current cyber
  library, nor proof of 160 independent vulnerability families.
- Fleet's Registry now has **2,699 cyber task-graph source keys**. **489** were
  created since September 1: **135 explicitly named blackbox products, 81
  whitebox products, and 273 unsuffixed source products**. These are publications,
  **not 489 additional certified, independent, runnable blackbox tasks**.
- The canonical **OTS Cyber** project contains **1,593 distinct catalog task
  IDs**. This is another unit: catalog membership, not Registry source count or
  readiness. The two totals must not be added or directly compared as growth.
- Recent sources span **seven application labels**, including **Stratum and
  Oracle EPM**, which are absent from the historical roster's environment names.
- A complete score-blind status read found **1,210 production, 379 staging, and
  4 discarded** catalog members; every row had an exact current-version UUID and
  attached verifier. This is an exact platform-state tier, not proof of the
  blackbox projection, runtime receipt, or genuine accepted execution.
- The precise count of newly accepted, runnable, independent blackbox tasks is
  therefore still **not established**. Current generation uses modular ATG
  (mATG), separate from the older Pipeline Lanes. **GitHub admin login is now
  verified working**. The
  remaining census limitations are unavailable structured browser transport and
  a closed-batch status endpoint that does not return entry status—not missing
  user access. The admin search reports **153 `matg-cyber`-prefixed top-level
  workflows**, which must not be counted as 153 accepted tasks. The strict
  receipt-proven task-version set remains the **89 exact versions** frozen in
  `configs/data/qwen-blackbox-eligible-v1.json`; this audit did not enlarge it.

## What the historical 160 actually contains

The tracked [`fleet-a62-task-split-v1.json`](../configs/data/fleet-a62-task-split-v1.json)
binds source job `a62dd51f-a52b-4941-8207-4679e4b25b51` and dataset manifest
`05c7928155f7d3d0c9d387164e559985c2498d0c40b32a2a0884f6c629fb0494`.
Direct counting gives 160 rows, 160 unique task keys, and 160 unique task-version
UUIDs: originally **130 train / 10 development / 20 sealed test**.

| Environment-key grouping in that roster | Tasks |
|---|---:|
| `cysec1-2-current-gen` | 34 |
| `cysec1-2-fira-gen` | 33 |
| `cysec1-2-fakelook-gen` | 29 |
| `cysec1-2-fubspot-gen` | 29 |
| `cysec1-2-fentry-gen` | 23 |
| `cysec1-2-rops-gen` | 6 |
| `cysec1-2-fentry-current-gen` | 4 |
| `cysec1-2-current-fubspot-gen` | 2 |
| **Total** | **160** |

These are environment-name groups, not independently audited application or
vulnerability-family labels. In particular, the compound environments can
contain multiple applications.

The later development-loss split retained all 160 identities as **117 train /
20 development / 20 original test / 3 reserved development**. Its historical
September 10 data handoff had **442 teacher trajectories across 73 training
tasks**, and **35 Qwen3.8 self-generated trajectories across 20 training tasks**.
Those are dated data-handoff counts, not a claim about all subsequently collected
rollouts. The “500-ish” teacher demonstrations were trajectories, not 500 unique
tasks and not hundreds of Qwen3.8 demonstrations.

The independent family audit explicitly found **no independent vulnerability
taxonomy**: all 160 derived `task_family` values were distinct, but 147 were
opaque blackbox fingerprints derived from task keys. It also recorded
`application_disjointness_proven: false`. Therefore the prior split supports
task-identity separation, not a strong claim about unseen vulnerability families
or unseen applications. Sources: the [September 10 handoff](https://github.com/fleet-ai/cyber-post-train/blob/9bc380652040c00e546e407a8721aa19f401e54b/docs/SFT_DEV20_DATA_20260910.md)
and [family audit](https://github.com/fleet-ai/cyber-post-train/blob/9bc380652040c00e546e407a8721aa19f401e54b/docs/evidence/sft-dev20-20260910/application-family-audit-v1.json).

## Live source supply and recent growth

The final Registry census paged all 14 pages, with `total=2697` on every page,
and returned 2,697 unique keys. It ran 18:21:28–18:21:46 UTC. This is complete
for the queried **`kind=cyber_task_graph_source`** namespace as visible to the
Fleet account, not a census of every possible cyber product across all systems.

| Key naming shape | All keys | Created since 2026-09-01 |
|---|---:|---:|
| Explicit `--blackbox_ctf_v1` | 887 | 135 |
| Explicit `--whitebox_patch_v1` | 156 | 80 |
| Explicit `--combined_ctf_patch_v1` | 12 | 0 |
| No recognized projection suffix | 1,642 | 272 |
| **Total** | **2,697** | **487** |

“Created since” refers to the artifact key's creation time. It excludes new
versions of older keys, and does not date certification or catalog registration.
The 487 keys represent about **22% growth over the 2,210 pre-September keys**.
Suffix stripping is useful for spotting sibling publications, but is not a safe
vulnerability-family deduplication algorithm.

### Recent application/projection cross-tab

Current-version `app` labels, queried at 18:21 UTC; projection columns are the
explicit key suffixes. A label names one application and is not a complete
multi-application dependency list.

| Application label | Blackbox | Whitebox patch | Unsuffixed | Total new keys |
|---|---:|---:|---:|---:|
| `fakelook` | 23 | 17 | 48 | 88 |
| `current` | 21 | 9 | 55 | 85 |
| `fentry` | 20 | 14 | 42 | 76 |
| `stratum` | 22 | 11 | 37 | 70 |
| `fira` | 19 | 10 | 37 | 66 |
| `fubspot` | 13 | 15 | 31 | 59 |
| `oracle_epm` | 17 | 4 | 22 | 43 |
| **Total** | **135** | **80** | **272** | **487** |

Publication was actively continuing during this audit. The earlier 18:14 census
had 2,696 total / 486 September keys. The additional Current whitebox product
was created at **18:20:11 UTC**. The differing snapshots are intentional.

### Versions and publication status are not acceptance

All **486** keys from the earlier census had their current version metadata
successfully queried at 18:16–18:18 UTC, with no errors or current-version binding
changes during that sub-census. Together those keys had **1,031 published
versions**. Do not count versions as fresh independent tasks.

Their current source labels were:

| Source status label | Keys |
|---|---:|
| `projection_compiled` | 201 |
| `pipeline_ready` | 187 |
| No status label | 68 |
| `compiled` | 12 |
| Other authoring/migration/review labels | 18 |
| **Total surveyed** | **486** |

The 18 others were `candidate-authored` 5, `verifier_contract_migrated` 5,
`authored_ready` 2, plus one each of `reviewed`, `solvability_hinted_probe`,
`solvability_diagnostic_candidate`, `reminted-generated-carrier`, `DRAFT`, and
`ready`. These are producer labels, not proof of passed build/solvability gates.
The separately queried `tuned_ready` label matched 34 historical source keys;
none of their selected versions was created in September. It cannot stand in
for current mATG acceptance.

## What the new products look like

The source contract distinguishes:

- **Blackbox (`blackbox_ctf_v1`)**: interaction with the service surface using
  the cyber harness and submission of a report; the agent does not receive the
  application's source-code view.
- **Whitebox patch (`whitebox_patch_v1`)**: a sibling repair task with code
  visibility and patch grading. It is not another independent blackbox exploit
  subject and should not be silently pooled into a blackbox training/eval arm.
- **Combined (`combined_ctf_patch_v1`)**: compatibility/transfer-only combined
  shape, requiring explicit justification under current authoring rules.

Most recent keys are labelled `pipeline_intent=standalone_atom` (**456/486**);
2 are labelled `atom_composition`, and 28 have no intent label. This is an
authoring label, not a verified graph-depth measurement: some multi-app/prefix
names still carry `standalone_atom` and require exact source provenance review
before classifying them scientifically.

Representative metadata-only observations:

| Published name / exact selected version | Structural observation, not a solution |
|---|---|
| `single-atom-login-unknown-account-timing-oracle` @3 and whitebox sibling @1 | Single-atom naming; Fakelook label; separate source and patch product |
| `single-atom-activity-body-stored-xss` @0 and whitebox sibling @0 | Single-atom naming; Fubspot label; separate patch visibility |
| `baseline-fentry-stratum-po-approval-v1-prefix-2--blackbox_ctf_v1` @1 | Multi-app and chain-prefix naming; Stratum label; `atom_composition` intent |
| `baseline-current-fakelook-executive-impersonation-v1-prefix-1--blackbox_ctf_v1` @2 | Multi-app/prefix naming; Current label; intent still `standalone_atom` |
| `baseline-fubspot-fira-authenticated-rce-v2-prefix-1--blackbox_ctf_v1` @2 | Multi-app/prefix naming; Fubspot label; semantic family not independently audited |

All keys above are under `cyber/task-graphs/`, selected during the 18:16–18:18
metadata survey. Version indexes are exact locators, **not an approved training
manifest**. Categories suggested by names include authorization/isolation,
token lifecycle, XSS, timing signals, and multi-service chains. **All 486 surveyed
current versions lacked a `category` label**, so this is naming-based inference,
not a verified vulnerability-type histogram. No exploit implementations or
task-local instructions were inspected.

## Catalog membership versus ready task selection

Metadata-only project membership GETs at 18:19 UTC returned:

| Project | Distinct task IDs |
|---|---:|
| OTS Cyber | 1,593 |
| Cyber Blackbox Behavior Pilot | 1 |
| zeuscyber | 16 |
| HiroCyber | 1 |
| **Union of these four projects** | **1,611** |

No IDs overlapped between those four queried projects. This is not a claim that
the project-name search covers every cyber project. Membership is the canonical
`eval_task_projects` relation; it does not certify projection, current version,
environment availability, verifier validity, or training permission.

The old `namespace=cyber` Pipeline Lanes census returned **734 lanes**, including
326 archived and 408 unarchived. The unarchived statuses were 257 failed,
67 waiting, 47 ready, 19 active, 8 superseded, and 10 final-green. **No lane in
that namespace was created after August 10**. A broader namespace check found
no September cyber lane namespace hiding the new production stream. Consequently
these historical lane statuses must not be used to declare the September
Registry supply accepted or failed.

The current source contract explicitly separates **mATG** from Pipeline Lanes.
The mATG preset routes authoring → build → intended-path solvability → blinded
evaluation → accepted, with separate semantic-rejection and infrastructure-error
outcomes. Its admin view is [Task Lifecycle Manager](https://admin.flt.build/modular-atg).
The original read reached **Fleet Admin sign-in**. The later follow-up below
supersedes that access blocker. The existing standard Fleet login also worked
for Registry, projects, and lanes and confirmed team `fleet`
(`a1025f0b-ad67-49fc-a023-51800ab43e84`).

### Authenticated admin follow-up — 18:27 UTC

Chris completed GitHub sign-in. Native Chrome then showed the authenticated
Task Lifecycle Manager, signed in as `chrisisaverted`, with Production selected.
The unfiltered page reported **1,003 top-level workflows**. Applying the visible
Temporal filter **`WorkflowId STARTS_WITH "matg-cyber"`**, retaining **Top-level
runs only**, reported **153 matches**. This scope excludes differently named
cyber workflows and batch children; it is neither an exhaustive cyber-task
count nor an accepted-entry count. No claim of full workflow-page enumeration
is made.

Two bounded metadata-only status checks establish why a completed-workflow
counter cannot substitute for task acceptance:

| Exact workflow / run identity | Observed evidence |
|---|---|
| `matg-cyber-chain-fubspot-fira-v2-recovery-20260911-b`, run `01a09033-c40d-7b71-b242-ba735d3e61e8` | Admin list says completed batch; its `/status?environment=prod` response is `{"detail":"Run status is not available"}`. No accepted-entry count can be recovered from this response. |
| `matg-cyber-whitebox-lean-e2e-20260905-022613`, run `01a06f63-e8e7-72a7-bab1-42d06cc80a86` | Admin list says completed single workflow. The safe status response contains **one entry, `status=errored`, phase `author_finalize`**, 17 completed work items and 16 receipt references. It does not demonstrate an accepted task. |

The single-run snapshot still says `state=finalizing`; its dated work timestamps
are September 5. That is historical workflow-query state, **not proof it is
currently executing**. Its one errored entry must not be generalized to the
other 152 workflows or to all Fleet cyber production.

The source implementation deliberately projects the safe single-run endpoint
to `entry_id`, `phase`, `status`, and `visits`, without source content. For closed
batches, however, the batch query rejects closed executions and falls through
to the single-run query; the observed batch consequently had no safe status
available. The browser automation transport was unavailable in this resumed
session, while native Chrome access worked; native accessibility text truncates
large JSON responses. Therefore **exhaustive accepted-entry pagination and a
catalog/projection/family join were not completed**. No `/result`,
`/live-result`, private receipt body, task source, or credential was read to work
around these restrictions. A score-blind structured metadata export/transport,
including closed-batch accepted entry identities and source/runtime references,
is the remaining requirement; another GitHub login is not.

This does not establish that the platform lacks every possible metadata path.
Current Theseus source also projects **batch-schema** `/result` responses through
`_batch_snapshot`, with aggregate counts and child references. The same route can
return raw content for other schemas and includes free-text error fields, so it
was not opened unfiltered in the native browser. A restored structured transport
could read an allowlisted projection of known batch results and then child status
metadata; it must not expose raw results or confuse aggregate counts with exact
task/family identities. A new platform feature may not be necessary.

### Exact-version, score-blind follow-up — 21:54–21:57 UTC

The private task-status route is a safe, DB-backed projection. For one task it
returns only task ID, current-version UUID, lifecycle status, latest instance
status, and verifier identity/presence. Paging the OTS Cyber membership and
querying this route for every member completed with **1,593/1,593 responses and
zero errors**:

| Evidence tier | Exact count | What it proves |
|---|---:|---|
| Registry source publication | 2,699 source keys; 887 explicitly suffixed `blackbox_ctf_v1` | Immutable source keys exist. It does not prove catalog registration or execution. |
| OTS Cyber catalog membership | 1,593 task IDs | The task belongs to the project. It does not prove projection or readiness. |
| Catalog current-version + attached verifier | 1,593 exact task/version pairs | Every current catalog row has a selected version and verifier pin. |
| Production current-version + attached verifier | 1,210 exact task/version pairs | The platform labels the selected version production and verifier-backed. It still does not expose source projection, environment receipt, or terminal execution evidence. |
| Staging current-version + attached verifier | 379 exact task/version pairs | The selected version remains staging; do not treat it as accepted/runnable. |
| Discarded current-version + attached verifier | 4 exact task/version pairs | The selected version is explicitly excluded from live use. |
| Exact runnable + terminal-receipt-proven training set | **89 task versions** | Existing direct terminal receipts and matching exact task/runtime/verifier bindings passed the stricter eligibility audit. |

Of the 1,210 production current-version UUIDs, **130** are members of the frozen
historical 160-version roster and **1,080 are not**. This is a version-identity
comparison only: an updated version of an older logical task also lands in the
1,080, so it is not evidence for 1,080 new vulnerability families or 1,080 new
accepted blackbox tasks. Likewise, 80 of the strict 89 receipt-proven version
UUIDs are current versions in OTS Cyber; the other nine need not be failures or
drift because OTS membership is not part of that manifest's selection contract.

Registry metadata provides a third, non-joinable signal. At 21:57 UTC it exposed
514 current artifacts labelled `stage=eval`: 433 `measured`, 39 `blocked`, 34
`retry_infrastructure`, 3 `dispatched`, 2 `error`, 2 legacy `ok`, and 1
`unmeasurable`. Of the 433 measured rows, 306 use the Fi mATG-stage handoff path.
The current mATG preset routes a consumed `measured` eval outcome to an accepted
terminal state, regardless of outcome bucket. But the Registry label records the
stage handoff before the scheduler's final transition, and its safe summary does
not expose the exact platform task-version ID. It is therefore supporting
pipeline evidence, not an accepted-task roster.

The three safe surfaces have no common evidence-complete join key:

- Registry source summaries expose exact source versions/projection labels but
  no platform task-version UUID.
- DB task status exposes exact task/version/verifier state but no task-graph
  source locator, projection, promoted environment, or terminal run receipt.
- mATG eval summaries expose terminal stage/status labels but keep the exact
  task/source/session handoff in the receipt body. That body is score-bearing
  and was deliberately not opened for this score-blind audit.

Consequently the **89-version manifest remains the only exact
runnable-and-receipt-proven selection**. The 1,210 production rows are useful as
a candidate ceiling, not as launch authorization. Expanding the strict roster
requires a server-projected, score-free join containing exact task/version,
`blackbox_ctf_v1` source locator, promoted environment/runtime receipt, verifier
pin, and final mATG acceptance. Publication, `production`, verifier presence, or
`measured` alone must not be promoted to that tier.

### What is needed before expanding the Qwen dataset

1. Obtain a score-blind structured mATG accepted-entry inventory, including
   closed batches (admin login already works). Distinguish accepted task entries
   from successful infrastructure workflows; bind exact source/runtime receipts.
2. Join accepted blackbox products to exact catalog task-version and environment
   version UUIDs. Exclude unfinished, superseded, semantically rejected, and
   infrastructure-ambiguous products; publication alone is insufficient.
3. Deduplicate at the underlying atom/graph lineage, including BB/WB siblings,
   prompt variants, and chain prefixes sharing atoms. Preserve application and
   vulnerability-family metadata rather than deriving “family” from opaque keys.
4. Freeze the training/development/test manifest before collecting new teacher
   demonstrations. Keep the original sealed test identities excluded; do not
   silently turn an already trained-on task into a claimed unseen holdout.

This audit did not perform those writes or launch collection. The next exact
number to report is **accepted, uniquely grouped blackbox products after this
join**, not the largest available catalog counter.

## Evidence and reproducibility

Source revisions inspected:

- cyber-post-train: `2b775b827afc87daf01170fd031c5834337aa9e5`
- vulnerability-tracking: `897e94ae92bd9f516fc363d984607c584e627e5d`
- Theseus: `0fff262678a80901fee6aa398e0142039e8a40bc`
- modular_atg: `c7ee1783b10a9717edcc308e990dff1b0b9e1fc5`

Read-only query recipe (use existing standard Fleet authentication through
`vulnerability-tracking/task_graphs/_fleet_api.py` and `_artifact_registry.py`;
never copy credentials into the report or command output):

```text
GET https://orchestrator.fleetai.com/v1/account

# Private Fleet API: https://api.internal.fleet-platform.fleetai.com
GET /v1/registry/artifacts?kind=cyber_task_graph_source&sort=created_at&order=asc&limit=200&offset=N
  Page through total; assert unique keys and stable totals; project only
  key/id/current_version_id/created_at metadata.
GET /v1/registry/artifacts/<artifact-key>/versions?limit=1&summary=true
  Select current version id/index/time and allowlisted labels only.
  Do not download state, blobs, source bundles, or operation payloads.
GET /v1/registry/artifacts?kind=cyber_task_graph_source&label=app:<app>&sort=created_at&order=desc&limit=200&offset=N
  Filter artifact creation time >= 2026-09-01; classify known suffixes.
GET /v1/projects/63d6fda8-48c4-4726-9ec3-d1028f2c47f5/tasks
GET /v1/pipeline/tasks/<each-project-task-id>/status
  Project only eval_task_id/current_version_id/lifecycle_status/instance_status/
  verifier_attached/verifier_id; require one response per project member.
GET /v1/registry/artifacts?kind=cyber_run&label=stage:eval&sort=created_at&order=asc&limit=200&offset=N
  Read current-version summary labels only. Do not open score-bearing run.json.
GET /v1/pipeline-lanes?namespace=cyber&limit=500&offset=N
  Project only immutable IDs, app, projection, stage/status, archive and dates.

# Separate GitHub-authenticated admin gate: login verified at 18:27 UTC.
GET https://admin.flt.build/api/v1/modular-atg/runs?environment=prod&page_size=100&roots_only=true
  Observed UI query: WorkflowId STARTS_WITH "matg-cyber" -> 153 matches.
GET https://admin.flt.build/api/v1/modular-atg/runs/<workflow-id>/status?environment=prod
  For a full census, page all cursors and resolve accepted entry identities,
  not just succeeded workflows. This audit did not complete that enumeration;
  the sampled closed batch's safe status endpoint was unavailable.
  Avoid result/live-result/receipt bodies containing private task content.
```

The compact metadata census checksums below are audit fingerprints, **not task
acceptance receipts**. Sorted keys/IDs use compact JSON before SHA-256:

- Latest 2,699-key Registry metadata tuples
  `(id, key, current_version_id, created_at)`:
  `ee5a3260183cfe4c2a12998bc145c3fbca88322ddc8392729f3140f52e805e31`.
- Latest 2,699 sorted Registry keys:
  `ebee527836d3bbb713178c047a28a0e1a22fe8f52763c9a78de24b77c90f6e7c`.
- Earlier 2,697 Registry keys: `32ed53cafd5fb5f1bcd7528ed91fbc46973e63435709734f1dc7141d837c58aa`.
- Earlier 2,696 keys: `ad9f764472fd9ec2601ebd01a3512622906568301c2b081b360a7c48ee235cca`.
- Earlier 486-key selected-version metadata manifest: `8de81a6f08d936e3e8850b7e007279e66ee76ac4ab9344930d7cc00e9731b44f`.
- OTS Cyber's 1,593 sorted task IDs: `0cc65b7d25d11f301d547b79a740498efa4fc85a299d051cc238ea0fde1c4ea8`.
- OTS Cyber's 1,593 sorted safe status tuples
  `(task_id, current_version_id, lifecycle, instance_status, verifier_attached, verifier_id)`:
  `fa91ee2d5a4a1aa6a067e6962654cb4743034bc47ce7115b9bd25e516f25cd79`.
- The 514 eval-stage artifact/current-version/allowlisted-label tuples:
  `01169dc1440f2f829c50518fa194e81dc883b7d633095f08a6050dfca4786144`.
- Historical 734-lane sorted `(id, stage, status, updated_at)` tuples: `84a737a7fe6af78b20dff442402e03843f1e67ea4c20ca0b8647a5339e53ed5b`.

Source-code authority for interpretation: vulnerability-tracking
[`AGENTS.md`](https://github.com/fleet-ai/vulnerability-tracking/blob/897e94ae92bd9f516fc363d984607c584e627e5d/AGENTS.md)
and [`_artifact_registry.py`](https://github.com/fleet-ai/vulnerability-tracking/blob/897e94ae92bd9f516fc363d984607c584e627e5d/task_graphs/_artifact_registry.py);
Theseus [Registry pagination](https://github.com/fleet-ai/theseus/blob/0fff262678a80901fee6aa398e0142039e8a40bc/orchestrator/private_api/registry/router.py#L1035),
[project membership](https://github.com/fleet-ai/theseus/blob/0fff262678a80901fee6aa398e0142039e8a40bc/orchestrator/private_api/project_page/router.py#L1031),
and [mATG metadata APIs](https://github.com/fleet-ai/theseus/blob/0fff262678a80901fee6aa398e0142039e8a40bc/services/admin-api/routes/modular_atg.py#L1088);
modular_atg [`AGENTS.md`](https://github.com/fleet-ai/modular_atg/blob/c7ee1783b10a9717edcc308e990dff1b0b9e1fc5/AGENTS.md).
