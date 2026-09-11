# Fleet cyber task supply — 2026-09-11

Read-only metadata audit, observed **2026-09-11 18:14–18:22 UTC**. No jobs,
environments, source publications, catalog edits, or training were started.
No task prompts, source bundles, traces, answers, flags, or scores were opened.

## Bottom line

- Our historical **160** is one frozen experiment roster: 160 distinct task keys
  and 160 exact task-version UUIDs. It is not the size of Fleet's current cyber
  library, nor proof of 160 independent vulnerability families.
- Fleet's Registry now has **2,697 cyber task-graph source keys**. **487** were
  created since September 1: **135 explicitly named blackbox products, 80
  whitebox products, and 272 unsuffixed source products**. These are publications,
  **not 487 additional certified, independent, runnable blackbox tasks**.
- The canonical **OTS Cyber** project contains **1,593 distinct catalog task
  IDs**. This is another unit: catalog membership, not Registry source count or
  readiness. The two totals must not be added or directly compared as growth.
- Recent sources span **seven application labels**, including **Stratum and
  Oracle EPM**, which are absent from the historical roster's environment names.
- The precise count of newly accepted, runnable, independent blackbox tasks is
  **not established**. Current generation uses modular ATG (mATG), separate from
  the older Pipeline Lanes. Its authenticated admin view requires a GitHub login
  that was not present in the available browser session. This is the remaining
  readiness-census gap, not a reason to call every published product usable.

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
The attempted read reached **Fleet Admin sign-in**, so no authenticated mATG
run/entry census was possible in this session. The existing standard Fleet
login did work for Registry, projects, and lanes and confirmed team `fleet`
(`a1025f0b-ad67-49fc-a023-51800ab43e84`).

### What is needed before expanding the Qwen dataset

1. Obtain metadata-only mATG accepted-entry inventory through the authenticated
   admin run list/status surface; distinguish accepted task entries from
   successful infrastructure workflows. Bind exact source and runtime receipts.
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
GET /v1/pipeline-lanes?namespace=cyber&limit=500&offset=N
  Project only immutable IDs, app, projection, stage/status, archive and dates.

# Separate GitHub-authenticated admin gate; not completed in this session:
GET https://admin.flt.build/api/v1/modular-atg/runs?environment=prod&page_size=100&roots_only=true
GET https://admin.flt.build/api/v1/modular-atg/runs/<workflow-id>/status?environment=prod
  Page all cursors; count accepted entries, not just succeeded workflows.
  Avoid result/live-result/receipt bodies containing private task content.
```

The compact metadata census checksums below are audit fingerprints, **not task
acceptance receipts**. Sorted keys/IDs use compact JSON before SHA-256:

- Final 2,697 Registry keys: `32ed53cafd5fb5f1bcd7528ed91fbc46973e63435709734f1dc7141d837c58aa`.
- Earlier 2,696 keys: `ad9f764472fd9ec2601ebd01a3512622906568301c2b081b360a7c48ee235cca`.
- Earlier 486-key selected-version metadata manifest: `8de81a6f08d936e3e8850b7e007279e66ee76ac4ab9344930d7cc00e9731b44f`.
- OTS Cyber's 1,593 sorted task IDs: `0cc65b7d25d11f301d547b79a740498efa4fc85a299d051cc238ea0fde1c4ea8`.
- Historical 734-lane sorted `(id, stage, status, updated_at)` tuples: `84a737a7fe6af78b20dff442402e03843f1e67ea4c20ca0b8647a5339e53ed5b`.

Source-code authority for interpretation: vulnerability-tracking
[`AGENTS.md`](https://github.com/fleet-ai/vulnerability-tracking/blob/897e94ae92bd9f516fc363d984607c584e627e5d/AGENTS.md)
and [`_artifact_registry.py`](https://github.com/fleet-ai/vulnerability-tracking/blob/897e94ae92bd9f516fc363d984607c584e627e5d/task_graphs/_artifact_registry.py);
Theseus [Registry pagination](https://github.com/fleet-ai/theseus/blob/0fff262678a80901fee6aa398e0142039e8a40bc/orchestrator/private_api/registry/router.py#L1035),
[project membership](https://github.com/fleet-ai/theseus/blob/0fff262678a80901fee6aa398e0142039e8a40bc/orchestrator/private_api/project_page/router.py#L1031),
and [mATG metadata APIs](https://github.com/fleet-ai/theseus/blob/0fff262678a80901fee6aa398e0142039e8a40bc/services/admin-api/routes/modular_atg.py#L1088);
modular_atg [`AGENTS.md`](https://github.com/fleet-ai/modular_atg/blob/c7ee1783b10a9717edcc308e990dff1b0b9e1fc5/AGENTS.md).
