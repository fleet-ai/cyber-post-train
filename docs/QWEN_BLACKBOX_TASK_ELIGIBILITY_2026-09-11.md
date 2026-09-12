# Qwen blackbox task and SFT-source eligibility — 2026-09-11

Read-only evidence census. PostgreSQL snapshot: **2026-09-11 20:38:36 UTC**.
Live training-key session census: **20:45:06–20:47:44 UTC**.
Local source manifests, source-ID projections and opaque file hashes checked
through **20:51 UTC**; direct receipt and current-binding validation completed
through **21:06:51 UTC**. No jobs, environments, retries, reconciliations, dataset
exports or catalog writes were performed. No prompts, trace/message contents,
flags, answers, grader source/result bodies, or held-out scores were displayed.

## Decision

**89 exact task versions now pass the task-version evidence gate: 74 train,
13 dev and 2 reserved-dev.** The digest-bound
[eligible task manifest](../configs/data/qwen-blackbox-eligible-v1.json) records
their exact environment/verifier and independently validated execution receipts.
The initial receipt-access gap was closed using an **already-running read-only
reader**, not a new job. This is evidence of a genuine previous terminal run plus
matching current immutable bindings, not a new live health test or launch approval.

**The 20-task dev set is not fully qualified yet:** 7 original dev versions still
lack the required historical proof. The 2 reserved-dev versions stay reserved;
none of these 15 non-training versions may enter training. SFT trace-context and
compaction qualification is a separate, still-open gate.

- The frozen a62 roster contains **160 distinct blackbox task keys and exact
  task-version UUIDs**, not 160 proven independent vulnerability families.
- The preserved baseline campaign planned **100 tasks × 4 attempts × 2 models**.
  Qwen has **302 accepted / 400 planned**, and only **42 of its 100 tasks have all
  four accepted attempts**. It is not a completed 160-task pass@4 baseline.
- Across either model, all 100 selected versions have some accepted ledger
  history; **98** have direct `valid` accepted records, and **2** have only
  `valid_historical` records. The other **60 a62 versions are absent from this
  campaign**; this does not mean they are broken or have never run elsewhere.
- All **113 current training task keys** have at least one live completed session
  with a functioning verifier and finite grade. However, the session-summary API
  **does not return the session's pinned task version, runtime environment version,
  or runtime verifier version**. Key-level healthy execution is not exact-version
  lineage.
- The exact ledger/catalog join corroborates **361 direct accepted sessions on
  83 training task versions**, and **74 successful Qwen source candidates on 38
  training versions**. Direct receipt-byte checks now cover every one of those
  83 task versions and all 74 Qwen candidates. Of the latter, **73 have clean
  trace manifests**; **67** also match current valid runtime bindings. These are
  candidates, not 67 newly accepted SFT sources.
- Existing private data is recoverable: **35 prepared Qwen self traces**, the
  older **442 teacher traces**, and the newer **186 GPT-only dense teacher
  sources** have verified local metadata/file bindings. All 35 Qwen and 186
  dense-teacher source IDs have healthy successful live catalog counterparts.
  This is stronger than trusting the normalizer alone, but not a new strict
  SFT acceptance receipt.

## Three different decisions

| Decision | Required evidence | What must not be substituted |
|---|---|---|
| Task-version eligibility | Exact task/runtime/verifier bindings; a real completed environment and working grader; a genuine success **or genuine model failure**; cleanup evidence | Source publication, a catalog key, workflow completion, or a success-only task filter |
| SFT-session eligibility | Eligible exact version; successful authoritative outcome; exact model/harness identity; usable assistant/tool training sequence and compaction provenance; train-only family assignment; immutable source/receipt digests | `accepted` alone, a numeric success alone, or a successfully tokenized file |
| Model success | The authoritative task outcome under the frozen treatment | Verifier process `success=true`, which means the grader ran and can accompany a failed solve |

Infrastructure-invalid, truncated, grader-ambiguous and unresolved sessions must
not enter either denominator as genuine failures or become automatic retries.
No ambiguous/accepted session was repeated during this audit.

## Follow-up: direct receipt and exact-runtime gate closed for 98 versions

The first census below remains frozen at its original timestamps. The bounded
follow-up read **119 accepted training cells**, covering all **83 direct-accepted
train versions** and all **74 Qwen source candidates**, plus one clean,
outcome-blind representative for each of **15 non-training versions**.
No model-success predicate was applied to task eligibility. A functioning grader
that records a genuine failed solve is as valid for this gate as a successful solve.

The existing reader was Job `chris-cyber-rollout-receipt-reader-r5-v1`, UID
`cb56b728-653a-42c2-a885-d9f063cf3311`; Pod
`chris-cyber-rollout-receipt-reader-r5-v1-65s9n`, UID
`3ff27834-d128-4043-8917-50e3e6dd6587`. Its `/campaign` mount is read-only and
selects only `jobs/chris-cyber-q38-glm53-pass4-ledger-v1` on SFS. It was Running,
Ready with zero restarts, with UID rechecked around the reads. The PostgreSQL Pod
does **not** mount this SFS tree. No observer/reader resource was created or changed.

For all **134 checked cells**, the audit independently:

1. Recomputed `ACCEPTED.json`'s canonical self-digest and matched the exact ledger
   receipt, cell, execution, run, session, verifier execution and configuration IDs.
2. Read only allowlisted binding metadata. Reconstructed the original complete
   per-cell configuration from its stored binding and versioned worker recipe;
   **all 134 reconstructed configuration digests match the accepted digest**.
   Thus the historical environment, data, verifier and harness fields are bound
   to the accepted execution—not merely inferred from today's catalog or a
   same-directory JSON file.
3. Verified exact runtime environment/data versions and UUIDs, the private
   runtime-seed content digest, and verifier UUID/version/digest; checked the
   scoring-intent self-digest and exact execution IDs without reading result bodies.
4. Matched the cleanup and completed session-ingest file digests to the stored
   ledger evidence, and verified the expected cleanup booleans and session ID.
5. Compared metadata-only trace manifests with the stored canonical trace digest.
   **Trace bytes were not opened or independently rehashed.** A manifest hash
   match is not a new claim that every recorded tool/context event is usable for SFT.

All 15 non-training session summaries also independently matched their exact
session/verifier-execution IDs and task keys with completed, ended, healthy
verifier-process state at **21:06:51 UTC**. No scores or model-success outcomes
were read for these dev/reserved sessions.

Private, restricted evidence artifacts (canonical self-digests exclude `sha256`):

| Artifact | Coverage | Self SHA-256 |
|---|---|---|
| `/private/tmp/cpt-qwen-receipt-evidence-20260911T2100Z.json` | 119 cells / 83 train versions / 74 Qwen candidates | `722fe6fed4b43be9b1dd855e95292e89e6e3c7fd29d117c7d0cd4f4c44c4f619` |
| `/private/tmp/cpt-qwen-heldout-receipt-evidence-20260911T2106Z.json` | 15 cells / 13 dev + 2 reserved-dev versions | `ae18edf68878603573375fd72a6c7b51126b198f1d76ffd140a23174ea904a60` |

Both artifacts contain only IDs, digests, metadata and validity booleans; both
are mode 0600 and are not committed. The first is historical receipt evidence;
its later current-binding join is recorded in the eligible manifest, not implied
by its earlier timestamp.

### Present bindings are a separate check

The current-binding audit at **21:04:03 UTC** is
`data/private/qwen-study-20260911/current-bindings.json`, self-digest
`15ff4a85bc67279d021c9f1a21a82f4b6751d19b6df3ba8aa214ced4c277a5f2`.
Its complete-file and all **160 individual row self-digests** were independently
verified. **132/160** selected versions have complete current immutable bindings;
**17 lack starting-data binding**, and **11 lack an immutable runtime-seed or
verifier pin**. A valid present binding by itself is not an execution proof.

Intersecting those 132 with the **98 receipt-proven versions** gives **89**;
their current environment and verifier fields exactly match the historical
accepted configuration. Nine receipt-proven train versions currently lack a
starting-data binding and are explicitly excluded from the runnable selection.
That does not invalidate their proved historical executions.

| Dense-v4 split | Receipt-proven exact versions | Also match complete current bindings | Still lack receipt proof |
|---|---:|---:|---:|
| Train | 83 | 74 | 30 |
| Dev | 13 | 13 | 7 |
| Original test | 0 | 0 | 20 |
| Reserved dev | 2 | 2 | 5 |
| **Total** | **98** | **89** | **62** |

The eligible manifest preserves the dense-v4 split unchanged, explicitly lists
all **71 exclusions**, and grants **no SFT-source acceptance or job-launch
authority**. Its self-digest is
`c45c9cc420a3dbf1aa6b635b84ed1ca3c404254f046ffdac234e053929a1bf12`.
Current binding validity intersects 49/61 dense teacher task versions and 17/20
prepared-self task versions; this is not a new teacher-session runtime lineage proof.

### Qwen source-quality exception and the compaction limit

All 35 prepared Qwen source trace manifests are clean. Of the 74 direct Qwen
candidates, one is excluded from SFT: session
`dd0fc50a-e29b-4154-947b-00121dfec0a0`, cell
`c2d818b5-ad40-58bd-aa9b-7d6ab11c4ea3`, task version
`47180c40-447d-4b8c-9387-ca8bae891e46`. Its manifest records **3 malformed lines**
and only partial valid-JSON normalization. It is **not** in the prepared 35.
Two other clean checked receipts cover that same task version, so the task remains
eligible while this specific session does not.

After the present-binding intersection there are **68 Qwen candidates on 33
versions**, of which **67** have clean manifests. **31/35 prepared Qwen sources**
are on the 17 currently bound prepared-self versions. Do not confuse these counts
with 68/67/31 new strict SFT acceptances.

The accepted configuration pins OpenCode **1.18.27**, asset digest
`4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702`, native
compaction/autocontinue, context window **262,144**, and **20,000** token compaction
headroom. This proves the configured context policy, **not whether a particular
session compacted or whether its post-compaction context is faithfully replayable**.
That SFT-context check remains open; no trace contents were inspected to bypass it.

## Historical roster and split quarantine

Selection authority:
[`fleet-a62-task-split-v1.json`](../configs/data/fleet-a62-task-split-v1.json),
source job `a62dd51f-a52b-4941-8207-4679e4b25b51`, source dataset manifest
`05c7928155f7d3d0c9d387164e559985c2498d0c40b32a2a0884f6c629fb0494`.
All 160 keys end in `__blackbox_ctf_v1`; all 160 version UUIDs are unique.
Sorted exact-version ID fingerprint: `63f3bb9667ab4839b8d07bdcfddfae27797f1614b11d146ff92a3e30fba24116`.

The original split was 130 train / 10 dev / 20 test. The September 10 v2 split
was 117 train / 20 dev / 20 test / 3 reserved-dev. The later dense-v4 split,
independently self-digest-validated here, is **113 train / 20 dev / 20 test /
7 reserved-dev**, digest
`cc55a6e7aca2cc56ec1987f6aba8b04e0a6bc5da67b380adfa3786cecc3b6a76`,
at `/private/tmp/cpt-teacher-dense.oem0Om/data/private/teacher-dense-v4/split.json`.

Use the stricter latest quarantine, not the original 130 or intermediate 117,
when counting new training sources. The current source candidate count is
**74**, not the intermediate 75; one candidate is excluded by the newer split.
All 35 old self-SFT sources remain training-only. The older 442 teacher sources
become **410 sources / 69 training keys** after the newer exclusions.

The completed live session census queried only those 113 training keys. An
earlier intermediate-split probe was stopped after discovering the newer split;
no returned intermediate-probe outcome set was used in the conclusions. The 47
dev/test/reserved keys were not included in the completed outcome census.

| Dense-v4 split | Exact a62 versions | In pass@4 ledger | Outside ledger | Versions with direct accepted ledger metadata |
|---|---:|---:|---:|---:|
| Train | 113 | 85 | 28 | 83 |
| Dev | 20 | 13 | 7 | 13 |
| Original test | 20 | 0 | 20 | 0 |
| Reserved dev | 7 | 2 | 5 | 2 |
| **Total** | **160** | **100** | **60** | **98** |

These are task-identity splits, not proof of application-disjoint or independently
audited vulnerability-family-disjoint splits. The previous family audit found
147 opaque fingerprint-derived family labels and no independent vulnerability
taxonomy. Vulnerability classes, chain depth and semantic family identity remain
**unknown** unless supplied by independent source/promotion evidence.

## Baseline completion: authoritative current scheduling state

Namespace `fleet-train-jobs`; PostgreSQL Pod
`chris-cyber-rollout-postgres-v1-0`, exact UID
`dff2a054-eca2-441d-bd19-028dc9e7c446`, Running/Ready with zero restarts.
Read-only, repeatable-read transactions were used; no schema initialization or
state mutation occurred. Metadata reports `fleet_cyber_rollout_ledger_v2`,
frozen plan SHA-256
`e88dca0419b10b50dc5b8d6b1313e8ea18408221185be279c5ce66c47853f9b3`.

Exact experiment:
`chris-cyber-q38-glm53-exact-easiest100-pass4-v1`.
Model bindings: Qwen3.8-27B
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
GLM5.3 `30333038ada1f1dacb294a93270305a890b50c14`.
Qwen3.6 sessions are a different model and are not counted as Qwen3.8.

| Route | Accepted | Review | Terminal nonaccepted | Pending | Active |
|---|---:|---:|---:|---:|---:|
| Qwen shared | 152 | 35 | 5 | 0 | 0 |
| Qwen dedicated | 150 | 54 | 4 | 0 | 0 |
| **Qwen total** | **302** | **89** | **9** | **0** | **0** |
| GLM shared | 96 | 87 | 2 | 19 | 0 |
| GLM dedicated | 91 | 98 | 2 | 5 | 0 |
| **GLM total** | **187** | **185** | **4** | **24** | **0** |

Keep shared and dedicated treatment blocks separate; this count does not claim
that pooling split-task serving treatments is scientifically justified.
Accepted means a credited rollout, not a solved task. Review and terminal states
are not silently converted into model failures.

| Accepted Qwen attempts on a selected task | Task versions |
|---|---:|
| 0 | 3 |
| 1 | 7 |
| 2 | 17 |
| 3 | 31 |
| 4 | 42 |
| **Total** | **100** |

Of Qwen's 302 accepted cells, **260** are `valid` with direct catalog session
IDs; **42** are `valid_historical` without direct session IDs on their current
cell rows. **258** direct accepted Qwen cells join to local results, each with
matching session ID, exit 0, completed termination and completed ingest.
Two further direct accepted cells were reconciled but have no matching
`rollout_local_results` row:
`4cd28290-b2c6-561d-aa93-39897da9f742` and
`eb9a61f4-e92f-5702-a88d-6136d9c86f0b`.
Both carry reconciliation digest
`42b4cdef3f1287172b67b924b65cb740891bd1453545b70974b0c7a31baf7581`.
All accepted cells have receipt digests. The initial 20:38 census did not read
the remote receipt bodies; the follow-up above independently validated 134
selected direct receipts, not every accepted or reconciled campaign cell.

There are **578 local-result records**: 316 Qwen and 262 GLM. Qwen's 58
nonaccepted local records remain review-only: 30 exit-0/ingest-failed,
26 exit-1/ingest-completed, and 2 exit-1/ingest-failed. These persisted attempts
must not be blindly repeated or admitted to SFT.

The source job's safe metadata route
`GET https://orchestrator.fleetai.com/v1/jobs/a62dd51f-a52b-4941-8207-4679e4b25b51`
returned **HTTP 404 at 20:47 UTC**. This proves neither successful completion nor
deletion; its old terminal state is not independently retrievable from that route.
No evidence examined establishes a separate completed Qwen3.8 160 × 4 campaign.

## Independent catalog corroboration and its exact limit

Using existing standard Fleet authentication, the audit fully paginated
`GET /v1/sessions?task_key=<one-current-train-key>&limit=500&offset=N`
for all 113 current training keys. It returned **3,184 distinct session IDs**,
113/113 completed queries, zero errors. Sorted session-ID fingerprint:
`b8c48e9ad0d91082cdfe91365012450bdc671775ad302c96a66322fc41e617f2`.

Only summary identity/state and verifier process health were retained; source
success selection was confined to training keys. No raw numeric scores are
included in this report. Every one of the 113 keys has at least one
`completed`, ended, finite-graded session with verifier process success.

- All **361** direct accepted ledger sessions on **83 train task versions** were
  found, completed and healthy in the independent catalog.
- All **74** current Qwen training success candidates matched catalog session ID,
  Qwen3.8 model name, task key and exact verifier-execution ID.
- All **35** prepared self-SFT IDs and **186** dense GPT source IDs were found
  under the expected task key/model, completed, healthy and successful.
- All **410** older teacher IDs still allowed by the current split also matched:
  204 GPT-5.6-sol, 186 Grok4.6 and 20 Grok4.5.

The catalog additionally contains **97 Qwen3.8 successful-summary candidates on
43 current train keys**. The other **23** do not join to a direct or local-result
session ID in this ledger snapshot. They might belong to other/historical work;
their exact model revision, task version, treatment and acceptance are not
established here, so they are not added to the 74.

The summary route does **not** supply `eval_task_version_id`, runtime
environment UUID/digest, runtime verifier version/digest, cleanup or compaction
proof. Joining a session's **key** to today's selected version does not create
the missing runtime pin. The public job-session route includes prompts and may
resolve mutable task-version data; the grading route includes grader source and
result payloads. Neither was used as an unsafe shortcut.

## Existing SFT sources: exact recoverable identities

These are verified private source locations, not a declaration that a new SFT run
may use them without the remaining gates.

| Source set | Distinct sessions / task keys | Current status |
|---|---|---|
| Older teacher Parquet | 442 / 73 | Metadata/file hash verified; 410 / 69 remain in current train split |
| Existing Qwen self Parquet | 35 / 20 | All remain train; source IDs match current accepted ledger and healthy catalog |
| Newer dense-v4 GPT teacher | 186 / 61 | GPT-5.6-sol only; source-ID coverage and Parquet agree; healthy catalog matches |
| Current direct Qwen success candidates | 74 / 38 | Exact accepted ledger/local-result/catalog join; includes the prepared 35 plus 39 requiring context/source qualification |

Private paths and immutable hashes:

- Older teacher:
  `/private/tmp/cpt-sft-dev20-v2/data/private/sft-dev20-v2/qwen-teacher/train.parquet`,
  file SHA-256 `83917f7af1395f693d24703dcc3f234e1604461e3d8fb4f220e9b05bea64f464`;
  2,210 rows / 442 distinct `source_session_id` values.
  Manifest self-digest
  `1329cec84cd1a755641a86bb1a6bd42ba5c9283f8b7c10eb18800fd5890614b5`.
- Prepared Qwen self:
  `/private/tmp/cpt-sft-dev20-v2/data/private/sft-dev20-v2/qwen-self/train.parquet`,
  file SHA-256 `0796cc615b8060adc4ad21a797e4021f456d165d5b41f0c92738b6d928276ab4`;
  175 rows / 35 distinct source IDs.
  Manifest self-digest
  `c1fb74ed9b1371ddb6d3266cbb45f7f164306b2cf150dd56f0fd7870f1fb69cb`.
- Qwen source JSONL (opaque hashing only, not message inspection):
  `/private/tmp/cpt-sft-dev20-v2/data/private/sft-dev20-v2/self-sources-v3/qwen3.8-27b.jsonl`,
  SHA-256 `cee5e6bf71989cba3affc4181422af95ada5d07aa57b263d231a5ea61c675e00`.
  Companion audit self-digest
  `41b1e1099de0ec92a0eb07d03ca6738fc017c0f00408771fc319faa6334abed3`
  records 20 compaction-related and 14 split-related historical exclusions.
- Dense teacher:
  `/private/tmp/cpt-teacher-dense.oem0Om/data/private/teacher-dense-v4/train.parquet`,
  SHA-256 `16961ea94dd5c082002d490ef1b21facea0900b415ebd33eca6975641f695ab9`;
  1,482 rows / 186 source IDs.
  Manifest self-digest
  `323a20fba88c22a87836a67f547e2ce7770d771434cb5ce3eff1b0f46ba28f36`.
  `source-coverage.private.json` has exactly the same 186 IDs and canonical
  digest `2f8d454553263ce75a7cc9ddde3cb432ac92535eadf072e6f312daac2019649b`.
  `included-sources.private.jsonl` opaque file SHA-256:
  `f3e5e7603e0d8d5014f71d4195547e53adf169e0c3dab5e3fc6a7a403c144d9a`.

All five examined source/split manifests self-validate. Dense-v4 starts with 410
split-allowed source candidates and excludes 206 non-GPT sources, 17 opaque
compactions and 1 unsupported/unproven tool interface, leaving 186. This is a
different source policy from the older mixed-teacher 442 corpus, not loss of
hundreds of new Qwen examples.

The historical staged data location is
`/mnt/sfs/jobs/chris-cyber-sft-dev20-20260910-inputs-v1/data/{qwen-teacher,qwen-self}/`.
The September 10 handoff records independent cluster file-hash checks.
**Current remote existence of these staged SFT files was not revalidated.** The
initial source-path audit inspected PostgreSQL, which mounts its database PVC,
not SFS. The later existing read-only reader mounts only the rollout campaign
subtree, not these separate SFT input paths. No observer/reader Job was created.

Exact source rosters are reproducible without reading text: project only
`source_session_id` and `task_key` from the hash-bound Parquet files; deduplicate
by session ID. Dense-v4 additionally has a metadata-only source-coverage file.
Sorted source-ID fingerprints (compact JSON arrays):

| Roster | SHA-256 |
|---|---|
| Old teacher 442 | `686f8eeea2cab6992c1eef5552d3f27972b654382e7056b8ee8589b621e891cc` |
| Prepared self 35 | `b1eda7b5f95a41889eecb8a1b457efd646b09f07871ffa7cc6cd26ac56438c1c` |
| Dense teacher 186 | `2725516a8a26a92e5f5c65280a30110242d0062d7875647a2eb6d9f4884d3249` |
| Current Qwen 74 candidates | `3ce7f74dd80d66fca1137fb86ebba76af0ecf2aded02b626ea82eb5ba65017f3` |

The generic [normalizer](../training/normalize.py) derives `infra_valid` from
terminal status, messages and task key; it does not by itself establish verifier
process health or exact runtime-verifier identity. This audit adds independent
health/ID corroboration but does not silently reinterpret that inferred boolean
as the stronger scientific gate.

## New Registry/mATG products are still an unresolved extension

The separate [Fleet inventory](FLEET_TASK_INVENTORY_2026-09-11.md) freezes the
earlier 18:21 census: 2,697 task-graph source keys, 887 explicitly blackbox-suffixed
keys overall; 487 new keys since September 1, including 135 explicit blackbox
products. OTS Cyber has 1,593 catalog task IDs. These are **source keys and catalog
memberships**, not accepted exact task versions and not additional SFT examples.

The admin login works, but the prior score-blind mATG status census could not
recover closed-batch entry status. Its 153 `matg-cyber`-prefixed workflow matches
are not 153 accepted task entries. The exact source → accepted entry → catalog
version → promoted runtime/verifier → terminal session join remains incomplete.
No new Registry product was promoted into the experiment on publication alone.

## Remaining gates; no inferred authorization

1. The direct-receipt gate is closed for 98 versions, with 89 matching complete
   current bindings. Recover or repair exact current starting-data bindings for
   the nine historical-only exclusions through the normal task owner workflow;
   this read-only audit does not authorize those mutations.
2. For the 60 a62 versions outside this campaign, plus the two versions with
   only historical acceptance metadata, recover exact-session runtime
   pins and terminal/cleanup evidence from the historical source job or a
   score-blind catalog projection. The current summary API cannot answer this
   exact-version question. Preserve all 47 non-training identities as quarantined.
3. Resolve the new mATG accepted-entry/catalog/runtime join independently of
   model success, including genuine graded failures.
4. For SFT, check exact teacher-session runtime provenance and usable teacher/
   student context and compaction boundaries. Exclude the malformed Qwen session;
   the remaining candidates are not automatically an expansion of the prepared 35.
5. The published eligible selection is a qualified **subset**, not a full-roster
   or full-20-dev-set certificate. Do not choose “easy” tasks solely because existing
   models solved them or call a source-key split vulnerability-family-held-out.

## Reproducible, score-blind evidence queries

Use the existing Fleet helper `vulnerability-tracking/task_graphs/_fleet_auth.py`
or the established environment credential; never log headers or credential
values. Confirm the exact PostgreSQL Pod UID before exec. Scheduling reads use
`BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY`.

```sql
SELECT model_id, serving_block, state, COUNT(*)
FROM rollout_cells
GROUP BY model_id, serving_block, state;

SELECT task_version_id, model_id,
       COUNT(*) FILTER (WHERE state = 'accepted') AS accepted_attempts
FROM rollout_cells
GROUP BY task_version_id, model_id;

-- Project only identity/state/digest fields, never SELECT * or private payloads.
-- A train-only success-source query must join the frozen CURRENT train UUIDs
-- before applying its success predicate. Output session/verifier IDs and hashes,
-- not numeric scores. Keep review/ambiguous cells excluded.
```

For catalog corroboration, fully page the safe session-summary route for the
**113 current train keys only**, checking `has_more`; retain IDs, key, model,
terminal/ended state and verifier process health. The route's absent exact-version
fields must remain absent in the conclusion.

Source-code interpretation:
[PostgreSQL operational contract](ROLLOUT_POSTGRES.md),
[accepted worker receipt](../evals/fleet/rollout_worker.py),
[direct-authority reward binding](../evals/fleet/opencode_self_hosted.py),
and Theseus
[`SessionSummary` and its paginated route](https://github.com/fleet-ai/theseus/blob/0fff262678a80901fee6aa398e0142039e8a40bc/orchestrator/public_api/sessions.py#L1101).
Source revisions examined: cyber-post-train
`fc36a1a24cd111f22188baaae18b83de2d4ac9a0`; Theseus
`0fff262678a80901fee6aa398e0142039e8a40bc`.
Older split/data evidence remains at
[`9bc3806`](https://github.com/fleet-ai/cyber-post-train/tree/9bc380652040c00e546e407a8721aa19f401e54b/docs/evidence/sft-dev20-20260910).

## Complete 160-version disposition

Full keys and exact environment-version UUIDs are in the linked frozen a62
manifest. The UUID below is the task-version selection identity, not a claim that
the summary API returned that session pin. Q/G columns are accepted-attempt
counts, **not solve scores**. `D` = direct accepted metadata exists;
`H` = only historical accepted metadata; `N` = outside this ledger.
This original disposition table remains the 20:38 ledger snapshot. The later
eligible manifest independently identifies the 89 versions that pass both
historical receipt and current-binding gates; `D` alone is not that certificate.

| Exact task-version UUID | Current split | Q accepted | G accepted | Ledger evidence |
|---|---|---:|---:|---|
| `02dd4e3f-d85d-4bf8-9976-eae2f102384d` | train | 3 | 4 | D |
| `032b6acc-13bc-4a9f-a57f-e3cd932988d4` | train | — | — | N |
| `037704a7-fc03-4cf6-a8da-3f3e3933717e` | train | 4 | 4 | D |
| `056f0e20-cd7d-40c8-b3c8-369fcee838be` | train | 4 | 3 | D |
| `05733731-8dfc-44b6-b9ed-231af79f2d53` | test | — | — | N |
| `05d09c55-ccdf-4d26-b06d-b851a2314a35` | train | 4 | 1 | D |
| `0920e798-c7e7-4da6-9d5e-ebeba45ec05a` | train | 2 | 3 | D |
| `09a3fea6-f691-4841-9218-d04459041a1f` | dev | 4 | 1 | D |
| `0b192133-c9b8-4211-b0a1-dbf98be46fe3` | train | 2 | 2 | D |
| `11f420a0-65a4-4819-93f8-8a042c512da8` | dev | — | — | N |
| `139ff10d-8c00-4bd6-9437-e6fbcccf2f4d` | train | 4 | 3 | D |
| `1407dcf7-46ad-4b82-8aa1-f88307dd8264` | train | 0 | 1 | D |
| `1456ccf2-5992-4a7c-9763-75248ef7b732` | test | — | — | N |
| `18079a9c-cb79-4013-aa08-54092c415165` | train | 3 | 3 | D |
| `195b1fca-b418-4920-983b-90c5bdfb904f` | train | 0 | 1 | D |
| `1fdbcb5b-4582-47dc-a794-29e7a40682da` | train | 2 | 4 | D |
| `21baee36-dc26-4dcd-983e-ae58a665dfa9` | train | — | — | N |
| `24030b60-cb61-4366-a3be-b3b808cad8ba` | train | — | — | N |
| `242227d2-065f-46da-8229-dbdecd99c46e` | test | — | — | N |
| `25afa659-5e00-4c5a-9ad4-39779b59c5ef` | train | 2 | 2 | D |
| `25b185c5-6aee-479d-97ba-d94ed9df1f42` | train | — | — | N |
| `2632322c-007c-42b2-9c93-d5b66850d6cb` | train | 2 | 2 | D |
| `2b6f2798-c405-4724-9d33-e867ba616fc3` | train | 4 | 4 | D |
| `2b841931-a8ff-4f5c-b32b-6824ea5b4429` | train | 4 | 3 | D |
| `2b9ba166-6f43-44aa-894d-75761314c150` | train | — | — | N |
| `2c47ef51-29f9-4736-9916-18b748059cc6` | dev | — | — | N |
| `2c62b98d-61e8-4216-bc84-c61bc4cd5ccb` | train | 3 | 0 | D |
| `2cb87286-a35e-4ba0-a949-1d93dfbd229a` | train | 3 | 2 | D |
| `2d62ff6e-92d4-4b85-8741-14bc78209e03` | test | — | — | N |
| `2de89322-296d-42dd-8038-06c8aee262bd` | test | — | — | N |
| `2df3b04b-87bd-4012-b5aa-e16d50698288` | train | 4 | 3 | D |
| `2e363dca-e74d-4855-a002-6761dc7f9f0f` | train | 4 | 1 | D |
| `2f2681bf-c1f7-4056-b6e4-cb2e5058f027` | train | 4 | 4 | D |
| `2fdc9511-f5fe-4386-a4a5-a5a16a30f359` | dev | 4 | 1 | D |
| `32d32c35-52f1-43a1-8d3d-3faaf01aa6a5` | train | — | — | N |
| `33527414-ced7-4005-9c5a-eb9c211a32da` | train | 3 | 0 | D |
| `33d37078-0669-478e-af39-43cd245f0da8` | train | 3 | 4 | D |
| `34fd6ab4-43f6-4f6d-a51e-7f55fb35b878` | train | — | — | N |
| `36dd3619-68f1-464a-859a-9eccd7c70994` | train | 3 | 4 | D |
| `383a2b61-e82b-4426-9a2b-d5372c92e346` | train | 3 | 2 | D |
| `386dff0d-6a22-4654-ba05-5262f61483b9` | train | 4 | 1 | D |
| `390e360e-ad44-43c2-91fd-8e7c9801b0be` | train | 3 | 0 | D |
| `3981d848-0689-47da-8f44-9f7e2132c3ce` | train | 3 | 0 | D |
| `3aafc5a8-2144-4bc3-919f-fcbbebd9f0d7` | train | 3 | 3 | D |
| `3ffeac73-fff8-49d8-b156-e726275edc2f` | train | — | — | N |
| `4283ed2c-bd4e-4ce8-b92c-80d46d5729e4` | train | 3 | 1 | D |
| `43dc2c52-f283-4454-8987-69c6cbbd4160` | train | 2 | 0 | D |
| `45c6cd9f-afef-4e63-b1f8-496c52e98f01` | reserved_dev | 4 | 1 | D |
| `47180c40-447d-4b8c-9387-ca8bae891e46` | train | 3 | 2 | D |
| `47483c98-ec42-4543-89f3-555a33e8f820` | train | 4 | 2 | D |
| `484625a1-a0ee-4e12-b566-40eccc26802e` | dev | 4 | 3 | D |
| `493bc3d8-7806-4e38-9bc3-e964a5c26f5b` | dev | 1 | 4 | D |
| `49789232-945c-43f9-b25f-a16f82457b5c` | train | — | — | N |
| `4a4aba06-38d4-4fc7-83ee-b625fc934d00` | train | — | — | N |
| `4b134baa-bbd7-4448-894d-2444e89a46e7` | train | 4 | 4 | D |
| `4b40f0b0-a8b7-465c-94f6-fce0a4acd2cb` | reserved_dev | — | — | N |
| `4b48f9cf-a5c4-4e86-814d-ebe86957e725` | train | 3 | 2 | D |
| `50ab8f20-7fc5-4e9f-afad-9fb03fae9b4d` | train | 4 | 0 | D |
| `5144890e-3515-497e-aaff-044d062eb664` | train | 4 | 0 | D |
| `51b23680-27fa-4bdd-a077-16669f3f0e18` | dev | 4 | 1 | D |
| `5287e180-64ea-487b-982a-d08a38c05fa2` | train | 1 | 0 | D |
| `53669bab-1389-48cf-9d08-68e411f9cd78` | train | — | — | N |
| `53d2f089-6502-4208-9795-d6ba7277fa9a` | train | 4 | 3 | D |
| `54425601-6fd2-43d8-8cb9-e565b767676a` | train | 3 | 0 | D |
| `560890d1-66cc-4fdc-b191-513797e763b1` | test | — | — | N |
| `5806d715-26cf-4aeb-a4b8-8ec692d79b33` | reserved_dev | — | — | N |
| `588b521c-74c2-4670-a38c-e4d4fd981c38` | test | — | — | N |
| `5990a2c6-0986-4c6f-85db-ca764425d8c5` | train | 4 | 3 | D |
| `5f0c9b28-fe2f-41e0-b73f-0e7749f64195` | dev | 3 | 0 | D |
| `5fdde785-acb4-4140-b234-5811ab1f8d1e` | test | — | — | N |
| `60bd1c32-506e-4df7-884e-f804b6da27bb` | test | — | — | N |
| `60f864ae-49bc-43ee-ae54-4ff8a8e71226` | train | 4 | 0 | D |
| `611014e8-6787-43a0-87a1-5c70fb69e3e1` | train | — | — | N |
| `61ff6681-c1e6-44cc-814c-02c43edce687` | train | — | — | N |
| `62ed9ac8-478d-45a2-a09b-6750f5946d97` | train | — | — | N |
| `6324fee5-2e3f-4538-a9e2-026d82ebb8a3` | train | 4 | 0 | D |
| `63e6ea95-1b08-44ec-aea7-244d9c6253af` | train | 3 | 2 | D |
| `65c8cc68-6ada-421f-ad0a-eb1d81fc9141` | dev | 4 | 4 | D |
| `66051718-314a-4dad-86ab-c7b9b72fa514` | train | — | — | N |
| `6a0e9f5f-dd72-40cc-a50d-6906b8f62c28` | train | — | — | N |
| `6a3023fd-4022-47a4-9ae5-6d15a892bd9b` | train | 3 | 2 | D |
| `6e565e4d-2cb9-402f-a492-457d6802cee6` | train | 3 | 0 | D |
| `78971e60-851b-4f6a-85d8-ecb3391d2af2` | train | 1 | 0 | D |
| `7e1fc9f6-7717-49bf-aee5-c09686dca297` | train | — | — | N |
| `7ec7aeca-0c69-4df6-989e-81278980d0a9` | train | 3 | 0 | D |
| `877f451f-29b2-48ab-bdcb-3c05aade4ccb` | test | — | — | N |
| `8ad6a73f-7ec8-4435-b868-a84327ed4c85` | train | 3 | 2 | D |
| `8c0b249c-8946-4ec3-8609-d74631af547e` | train | 3 | 2 | D |
| `8d84b316-d741-44f8-aaa5-7a0f0d7b8d3a` | train | 2 | 0 | D |
| `8e5c7c45-7259-420d-ba12-0381465d864e` | train | 1 | 2 | D |
| `90338ce0-194b-4cea-bef4-d64e5d426319` | train | 4 | 3 | H |
| `91369354-e904-405e-9b86-8c793fbae289` | test | — | — | N |
| `9375a9b9-04e5-4f6f-ad47-286121278992` | train | 2 | 4 | D |
| `941c1062-2ecd-4263-9d5c-f364eca44c46` | test | — | — | N |
| `96977ecb-4d8c-4c4f-a5e0-02d925bdc724` | train | 2 | 1 | D |
| `9c3de7a0-90eb-470d-b6b8-c8bb2e86467c` | train | — | — | N |
| `9d37921b-5ec7-413f-be13-986b1c2d66e8` | dev | — | — | N |
| `9fd14da1-efd5-4852-81aa-e6020219b75f` | train | 1 | 0 | D |
| `9ff9f2eb-3359-4321-9f65-d42accfdcace` | test | — | — | N |
| `a05eb7c4-3107-4f19-b4bb-4a6d90fea614` | reserved_dev | — | — | N |
| `a0cacaaf-480b-4a4c-9ed7-6b6192bb6783` | train | 2 | 0 | D |
| `a0e4a473-b544-4006-b4fa-8e6b6bfb0a50` | train | 1 | 0 | D |
| `a1b92843-ca06-4852-9f09-f2730da9124d` | train | 3 | 4 | D |
| `ab2fd90d-1ee5-4451-a866-260ddf5c336d` | dev | 3 | 2 | D |
| `ac6a6cfa-ea44-4c11-bb2e-c2bb2d142c8e` | train | 4 | 4 | D |
| `b164ad89-7b40-4baf-a9e8-2eb880c61730` | train | 4 | 4 | D |
| `b26ec4e6-c86f-4005-8f6c-b2473b6de3bd` | train | — | — | N |
| `b2f61fae-a07b-42c7-92ec-eeb4fa2064b6` | train | — | — | N |
| `b51782f9-cb9f-4603-be81-a16eab4b8a36` | dev | 3 | 3 | D |
| `b51ee7e6-78ce-4358-895d-ecebf43cc582` | test | — | — | N |
| `b571037f-bec1-4d59-b987-2d86fd8e0f19` | test | — | — | N |
| `b66062df-4c6a-419f-8568-82e152e9df73` | dev | 1 | 3 | D |
| `b66aad3c-0225-46ab-a8fa-c616e9d3f3ff` | train | — | — | N |
| `b6efa2e4-7743-4b8b-a014-832d90cd5e39` | train | 2 | 4 | D |
| `b6f2349e-fcb1-4a18-9c8d-3278d3f72991` | train | — | — | N |
| `b7b452c2-af1e-4c11-acf3-3d855c846e74` | train | — | — | N |
| `b8b0644b-db81-4d6a-a420-0416eb3dbb66` | test | — | — | N |
| `b90a33ad-c4b6-4ce6-9757-5009bc3b3b17` | reserved_dev | — | — | N |
| `ba912601-1184-4dcd-9e9d-2fce22543e4b` | train | 4 | 3 | D |
| `bdb3778f-33da-4eb6-96ef-6b9a7a828d41` | train | — | — | N |
| `be398179-b809-441f-85d0-b981c64faf43` | train | — | — | N |
| `bf88b387-f5b5-4452-b539-4ad48f0c177e` | train | 2 | 3 | D |
| `c07d363e-b67b-4e0b-8669-8c75bd3eb773` | dev | — | — | N |
| `c0d14c8f-5f33-4957-8e38-af075bbf3df9` | train | — | — | N |
| `c3aeb7aa-d823-4b81-9373-3762539b36b7` | test | — | — | N |
| `c3fbe2bd-b9e6-41b1-a47f-5342fdc28053` | dev | 3 | 4 | D |
| `d05fbffb-db7e-4d27-b6d1-5a854b74e6cd` | train | 3 | 0 | D |
| `d4f5e664-d4fd-4492-a103-429260ad9e99` | train | 0 | 1 | D |
| `d6634975-2918-4037-9cd2-467f365e39b0` | train | 2 | 1 | D |
| `d6c906fc-ef92-4ce6-ab9c-54a9b7eb9037` | train | 4 | 3 | D |
| `d73098dc-4f03-4761-8a81-cc9ec914286c` | train | 4 | 3 | D |
| `daa5bbbb-0f63-481f-be8e-1d217c12740e` | dev | 3 | 4 | D |
| `dcf2aefc-e34d-4a4a-8aa9-b18a3b754d5a` | train | 3 | 2 | D |
| `dd8dd22e-75c0-4b93-8f8e-ea8a292d92bb` | train | 4 | 2 | D |
| `e13f62eb-170a-40c6-86bf-3c3f3829e3d8` | train | 4 | 1 | D |
| `e32f1402-ec95-4df3-9c04-8e5e0400242c` | train | 4 | 3 | D |
| `e33aaed0-491b-46e8-8891-55d99c80ef56` | train | 4 | 2 | D |
| `e4068901-ba6e-4faf-b30c-7535aa11f0fc` | dev | 4 | 2 | D |
| `e42d481e-e0eb-42b2-a4dc-62b1a57ce93d` | test | — | — | N |
| `e53d70e2-c129-422a-b878-033bfb5556ab` | reserved_dev | 4 | 3 | D |
| `e7a41e55-2a4e-48d0-931b-eb4d002ccd43` | dev | — | — | N |
| `e856de31-fe15-4aab-8519-6ae4554561d7` | test | — | — | N |
| `e90d9673-7260-4460-9eb5-6ec2e9e24329` | train | 4 | 0 | H |
| `eae5337c-eeba-48f8-b5b0-a0310a690a05` | dev | — | — | N |
| `eb5a91b3-b2ac-446c-93ee-c4f57f5da0a4` | dev | — | — | N |
| `ee77cfbf-ceed-487f-a19d-3922aaaf4bad` | train | 2 | 0 | D |
| `eefe304c-499d-4c9e-b09c-8aa0fab87213` | train | 2 | 0 | D |
| `f0366278-f58b-4a63-bcd1-55da55ad59a9` | reserved_dev | — | — | N |
| `f1ce26fc-39b6-442c-a03c-3c18f99e119f` | train | 3 | 0 | D |
| `f31ebe83-0ff1-4660-bcba-59ffa4b82d5a` | train | 4 | 0 | D |
| `f50eee68-84c7-44d1-b4de-4c554ecc3e2e` | train | 2 | 1 | D |
| `f7b9d881-c718-488c-8e36-f9cdf62afa38` | train | — | — | N |
| `f8533754-39ad-4683-b9c1-d26bc24dfd14` | train | 4 | 3 | D |
| `f956619f-ab6d-4851-b45d-496a331f90c6` | train | — | — | N |
| `fa04ef0d-3341-4b5f-8a13-d2955feaadc1` | train | 3 | 4 | D |
| `fa83b3a7-be02-4193-9670-2cd794fa28b4` | train | 2 | 0 | D |
| `fb68f600-9850-4cc2-86e7-a4f554e5f93f` | train | 4 | 0 | D |
| `fb8f2178-7dd8-429a-8d3c-f14b21a51e02` | train | 4 | 0 | D |
| `fd07b96f-7aa4-4041-862b-5aa7411eff4b` | train | 4 | 4 | D |
| `fe274b85-707f-4a84-ada9-9470f84e39ea` | test | — | — | N |

## Exact current Qwen success-candidate identity roster

All rows below are current-train only and have matching accepted ledger/local
result and healthy catalog evidence. They are **candidates, not a strict SFT
allowlist**. “Prepared” means membership in the existing hash-verified 35-source
self Parquet; no additional message/context qualification was performed here.

| Session ID | Exact selected task-version UUID | Verifier execution ID | Prepared |
|---|---|---|---|
| `0bf1368d-1176-47ad-8642-ee571ecbdea6` | `d73098dc-4f03-4761-8a81-cc9ec914286c` | `28ec4172-64ee-4182-be67-ff53d979d65e` | no |
| `0cf8a222-36de-4ef2-a4ee-af36ac58a582` | `05d09c55-ccdf-4d26-b06d-b851a2314a35` | `cd0e106b-c1bc-413f-b669-78590f2a8534` | yes |
| `10baaf82-587d-4d36-9383-6240262f8f78` | `fd07b96f-7aa4-4041-862b-5aa7411eff4b` | `2a5ca7c7-f26f-4650-bf6a-cb83430d28fb` | no |
| `1333eb9a-e129-4746-8b0d-89f016656c3a` | `02dd4e3f-d85d-4bf8-9976-eae2f102384d` | `f6c3fe62-4528-46d7-8ede-80aa453f5d4f` | yes |
| `166a3115-5b98-4f2b-b7a0-4107c1fdbbea` | `4b134baa-bbd7-4448-894d-2444e89a46e7` | `7e26446d-75aa-4f85-9486-3090f5ba6bf9` | yes |
| `18247701-3e12-49fd-b909-6f6097c970b1` | `fd07b96f-7aa4-4041-862b-5aa7411eff4b` | `80d81403-5c09-4013-99ea-2fc881797a89` | no |
| `19009ec7-124a-4921-959c-5a31ebb4b6ce` | `47180c40-447d-4b8c-9387-ca8bae891e46` | `bf682372-af19-4301-b0f0-da8a347d9f20` | yes |
| `1f90aa3b-57e3-49ce-9a38-cb5f068f9f37` | `5990a2c6-0986-4c6f-85db-ca764425d8c5` | `f3f0110f-e67a-4a5d-840a-8253a8e1bc2b` | yes |
| `1f92ff48-15a2-40dd-bf32-c94ed3f62196` | `0920e798-c7e7-4da6-9d5e-ebeba45ec05a` | `a3cd56f1-0e72-4a67-b971-220644abe470` | yes |
| `2a09c05b-2f76-488e-adf7-2ac98f07371f` | `f8533754-39ad-4683-b9c1-d26bc24dfd14` | `3e3f1f55-1d69-4510-9474-6631b691bddf` | no |
| `2b39e291-eb6b-4740-8471-e3c7b6af4763` | `fb8f2178-7dd8-429a-8d3c-f14b21a51e02` | `7ebabd70-f17f-4338-bfe8-4aa162684845` | no |
| `2dca568f-11a3-47c4-afa0-4d5a1bb866fd` | `b164ad89-7b40-4baf-a9e8-2eb880c61730` | `b38b7db1-58a7-485b-8401-f53bfe88f35d` | yes |
| `3547d546-9b2f-4de0-b080-ad21448adc0c` | `b6efa2e4-7743-4b8b-a014-832d90cd5e39` | `9355ec95-823c-429a-b6e7-35a704cdde69` | no |
| `367eede4-0677-40dd-afc3-daa202eff1bb` | `8ad6a73f-7ec8-4435-b868-a84327ed4c85` | `4d966a70-5497-40e6-bc81-85f49c57445a` | yes |
| `3f2fdc8c-6638-4ae2-9838-a24afeb36977` | `139ff10d-8c00-4bd6-9437-e6fbcccf2f4d` | `cf5f8fb7-49ff-42f8-a233-42b474b7fed9` | no |
| `458bbdb3-0a23-4f6a-ac1c-df17ecf05175` | `139ff10d-8c00-4bd6-9437-e6fbcccf2f4d` | `4c806396-105d-45c9-9a60-99901536383f` | yes |
| `4baa5ca3-15f6-4f5f-9519-eaa7fc3bfdfd` | `d73098dc-4f03-4761-8a81-cc9ec914286c` | `108df0b1-f8e0-496c-a9d2-b30696d12098` | no |
| `4c91240c-1a0a-4e64-a9f1-e9f528787ce2` | `eefe304c-499d-4c9e-b09c-8aa0fab87213` | `6e61551e-8612-44ca-8451-c060e5b26bfb` | no |
| `4eb71b22-bd58-404e-b009-bb37000aa167` | `a0cacaaf-480b-4a4c-9ed7-6b6192bb6783` | `dbad90ed-4732-4cc1-93cb-862e0338f340` | no |
| `5983504a-9d96-43ae-8f72-48e21208ab42` | `bf88b387-f5b5-4452-b539-4ad48f0c177e` | `27ea5b15-d087-455a-903a-8f08a8d7e292` | yes |
| `6517681d-c0d3-4a8a-adc4-8da683cf4141` | `4b48f9cf-a5c4-4e86-814d-ebe86957e725` | `99ed784d-b321-4b19-9946-ea0c70ef10b6` | no |
| `65f9fc3e-9fcf-4fc8-a9a7-fc65c0570cd5` | `e32f1402-ec95-4df3-9c04-8e5e0400242c` | `2365b0a3-b3b7-4b05-91f2-d112d5e18cd3` | no |
| `725d273d-b56b-4220-a531-f996bf49e4c8` | `4b134baa-bbd7-4448-894d-2444e89a46e7` | `fef68f47-cbef-4094-93c9-dd194588bfd8` | yes |
| `753ba6d6-26de-4a5e-a20e-96a7c4b73162` | `b6efa2e4-7743-4b8b-a014-832d90cd5e39` | `32e02ccc-2ea7-4601-87a8-1a2db896cc6a` | no |
| `7580ee7b-24b5-453e-84f2-544deeb11647` | `037704a7-fc03-4cf6-a8da-3f3e3933717e` | `c0a3fa95-ff19-4d1d-927c-257ad01f6379` | yes |
| `76c3a7a2-d235-4965-93e4-d591d66fddbd` | `33d37078-0669-478e-af39-43cd245f0da8` | `eafec818-4661-4c54-8f11-196f35186139` | yes |
| `7e8f49ce-f9c7-4d01-add1-d326f0f15e6b` | `2df3b04b-87bd-4012-b5aa-e16d50698288` | `357a24fb-4f7c-4b53-ad2d-f1c3cc77af23` | yes |
| `7f0ed7be-2349-4f4c-ab5f-8f00ced4afa5` | `fb68f600-9850-4cc2-86e7-a4f554e5f93f` | `e4a92f25-4dc0-4d9e-980a-18caa2a66310` | no |
| `800e1fe7-8cd4-4c6f-a956-230cbdbceada` | `43dc2c52-f283-4454-8987-69c6cbbd4160` | `fe363993-d60e-44f4-8dfa-435f30356272` | no |
| `832cb7e7-3675-4282-96ad-8553d552d1da` | `b164ad89-7b40-4baf-a9e8-2eb880c61730` | `58a75319-d0a1-4e0c-8e69-316006a2e68d` | yes |
| `8341dec6-9fa4-477a-9355-59e994ef2713` | `4b134baa-bbd7-4448-894d-2444e89a46e7` | `a263b653-cf29-48c4-81b5-0dc1981761fe` | yes |
| `86bfafba-0c09-470e-8aba-5ef035411d13` | `9375a9b9-04e5-4f6f-ad47-286121278992` | `c1ceed70-b151-469a-bb11-ff91e57f69da` | yes |
| `9117154b-e985-4506-b502-8535a1ac0822` | `8ad6a73f-7ec8-4435-b868-a84327ed4c85` | `c4b774c0-d868-498c-b8a4-7017a47a67c3` | no |
| `92e4a9c5-380a-4fb8-814f-17e0f7e0e364` | `33d37078-0669-478e-af39-43cd245f0da8` | `097f975e-7d85-4cfc-9a05-dcfdac30d68d` | yes |
| `98bc4b65-e8a8-402d-a879-cf03591261d5` | `fd07b96f-7aa4-4041-862b-5aa7411eff4b` | `bf2dfeed-fe4b-4a42-848b-3da418274b98` | no |
| `99137892-55dc-4db7-9886-b96713a5fee5` | `383a2b61-e82b-4426-9a2b-d5372c92e346` | `5f67b37b-7ecc-497a-a54f-e1e276b62262` | yes |
| `9a2931f4-c776-4bcb-9718-7693885474a3` | `0b192133-c9b8-4211-b0a1-dbf98be46fe3` | `38add282-77b8-4295-b280-8b0bfe531cfe` | yes |
| `a17b589f-8a0a-40f2-b14b-3198baa35888` | `e33aaed0-491b-46e8-8891-55d99c80ef56` | `ba27c8aa-9559-4353-b4be-c74a3a08fc44` | no |
| `a2df36b5-5e1c-4fde-9932-35f75079c6e0` | `0920e798-c7e7-4da6-9d5e-ebeba45ec05a` | `4c5bf44b-37e1-478d-b615-bb2efa3f7b72` | no |
| `aaf232fb-141a-4aa8-9cee-8685822a9603` | `dcf2aefc-e34d-4a4a-8aa9-b18a3b754d5a` | `5511d029-07e5-4ad0-a819-4fbf6e44f5d3` | no |
| `b089d231-b142-457f-8307-0e7c9b8b407b` | `5990a2c6-0986-4c6f-85db-ca764425d8c5` | `a9c0b0de-2bce-4570-889a-8250f06bbc85` | yes |
| `b0b2a80a-960a-47df-b03d-be3409f96943` | `fb68f600-9850-4cc2-86e7-a4f554e5f93f` | `2c88a995-5177-41f7-918e-1dad6513c51d` | no |
| `b0e76492-75d4-4e5f-99e5-4ae6f6de21d9` | `5990a2c6-0986-4c6f-85db-ca764425d8c5` | `03288a59-6e29-4514-94f1-3085d47a927a` | yes |
| `b1cf29d5-fbfd-47bc-83cc-42270f543275` | `36dd3619-68f1-464a-859a-9eccd7c70994` | `fd85909c-9d71-4ae0-b6aa-f563abfbfb0e` | yes |
| `b2de4060-4a95-4236-be7d-d43cd75e4c99` | `36dd3619-68f1-464a-859a-9eccd7c70994` | `5c35b6c8-eb29-4b93-9521-d2688d5b6b66` | yes |
| `b2e8558e-6311-4ea0-926a-7237bcf90759` | `f8533754-39ad-4683-b9c1-d26bc24dfd14` | `64ca1bd6-6500-47a4-9405-f826e26f6c55` | no |
| `b3f5c8c3-e533-4b45-a6fb-7e5d70a711b6` | `fb68f600-9850-4cc2-86e7-a4f554e5f93f` | `de5fdeb3-4af5-445f-942d-fc08c8140706` | no |
| `b97078f0-a4d8-48a0-ba21-ac92f0b73507` | `47180c40-447d-4b8c-9387-ca8bae891e46` | `68eb7e7f-a11d-4117-83ee-b568d59c9ff2` | yes |
| `b9e16949-06b9-4cdd-8801-9b3911929fb7` | `383a2b61-e82b-4426-9a2b-d5372c92e346` | `2e318964-7b42-4724-b716-445ed4071250` | yes |
| `ba7fae93-408d-4e15-a872-a26e997c0144` | `47483c98-ec42-4543-89f3-555a33e8f820` | `6f73758b-315a-46e9-9d1c-ce94615fdadd` | no |
| `c15e9ce2-f845-4bca-a6e2-68fec370e450` | `e32f1402-ec95-4df3-9c04-8e5e0400242c` | `b8c47598-2c32-4fe6-9866-1fc435a0ae3a` | no |
| `c3ec9c42-3130-4514-acfa-21cbab3aba3b` | `36dd3619-68f1-464a-859a-9eccd7c70994` | `cdf2197e-bc77-448b-95d8-acc137482168` | yes |
| `c811f6c9-49ee-4be5-9213-1b3a5a171795` | `2b6f2798-c405-4724-9d33-e867ba616fc3` | `c768db6c-2a1f-43b3-8711-81e0dce92b85` | yes |
| `ca4cbf88-622d-4522-a3a2-36cc0f491f46` | `e32f1402-ec95-4df3-9c04-8e5e0400242c` | `861762eb-3221-44f4-9be8-99f906dde635` | no |
| `ce16f2ee-ffe1-47ca-aea6-afde605f197b` | `33527414-ced7-4005-9c5a-eb9c211a32da` | `be80758e-12f2-4496-90ee-61f5da7d41f6` | no |
| `cf4cf860-4321-41da-8c68-70280e2241ac` | `8ad6a73f-7ec8-4435-b868-a84327ed4c85` | `32ee906f-45b0-4431-b144-bbebe73ccd29` | no |
| `d1bff33f-7a7c-4385-8814-1de3f62b0298` | `9375a9b9-04e5-4f6f-ad47-286121278992` | `323dd9da-1c26-44b0-a556-a972a1abed58` | yes |
| `d1cd619c-c031-446f-8c35-931669363c73` | `4b134baa-bbd7-4448-894d-2444e89a46e7` | `7de9274f-a960-4b57-b9db-d1c9b75c8ccf` | no |
| `d8ed82a4-319e-41c0-bafe-ac53cd5219b2` | `2b6f2798-c405-4724-9d33-e867ba616fc3` | `4eb95779-f75f-4d7a-a978-2abd78f8eb41` | yes |
| `db29153d-7105-4899-9a7c-aa0cabbbe2a6` | `4283ed2c-bd4e-4ce8-b92c-80d46d5729e4` | `f18dc33f-6b23-49c9-9dd3-a85a41ea2cf4` | yes |
| `dbfb070a-5410-4d9a-bac0-1f4c5b32fe4b` | `2632322c-007c-42b2-9c93-d5b66850d6cb` | `900fbfc7-2266-4fb9-80b7-7894fe5f7f8b` | no |
| `dd0fc50a-e29b-4154-947b-00121dfec0a0` | `47180c40-447d-4b8c-9387-ca8bae891e46` | `9783d0bd-7dc8-4c09-98ef-9e16680eb628` | no |
| `ded24974-d6b1-41eb-897c-38deb904e4f6` | `5990a2c6-0986-4c6f-85db-ca764425d8c5` | `09542f42-c6ae-4b6c-b116-93a125503008` | yes |
| `e0f5ce92-3e43-4a0c-9f9e-bc18d3c24765` | `3aafc5a8-2144-4bc3-919f-fcbbebd9f0d7` | `ceb6e241-4cba-4aa0-936c-2300c6a85b3a` | no |
| `e8c30984-d9df-4dbf-ba14-1d03c8b8a0d2` | `8e5c7c45-7259-420d-ba12-0381465d864e` | `f572604b-e469-45a4-a4b7-5dd3e1befd5b` | yes |
| `ebd25c57-1eec-4b65-a529-570da49365a6` | `037704a7-fc03-4cf6-a8da-3f3e3933717e` | `4a918992-9797-48ff-882a-94485e4b3499` | no |
| `edbf9ef7-e80a-4e14-9be5-05138191c188` | `e32f1402-ec95-4df3-9c04-8e5e0400242c` | `01ad3719-3c13-4550-b51b-e395c29af437` | no |
| `eebffadf-4cab-45b1-a30d-1449f90ef895` | `fb68f600-9850-4cc2-86e7-a4f554e5f93f` | `c8a5ece5-68a8-4720-9a51-335404cc0347` | no |
| `f01edab8-7bf9-4c87-bc7e-d912e376836f` | `2b6f2798-c405-4724-9d33-e867ba616fc3` | `46dd9e78-32f7-4c7d-9cf7-4b1d1e37f851` | no |
| `f0558747-f7d2-4fd6-affd-7f9a68b96fa7` | `dcf2aefc-e34d-4a4a-8aa9-b18a3b754d5a` | `e7ab1cf9-5d31-420c-87aa-9c663098205e` | no |
| `f0b5aab7-0ebb-49f8-86c0-22201312b3c7` | `4b48f9cf-a5c4-4e86-814d-ebe86957e725` | `87e47e27-1b30-47da-a987-782fc33f9811` | no |
| `f0ed1d6f-549c-4b91-ace6-0862be523777` | `383a2b61-e82b-4426-9a2b-d5372c92e346` | `4d1b1476-5902-47f1-8b26-a2054f00cdc9` | yes |
| `f144846a-3c60-4ecd-bf52-7535cd7f2ae0` | `fa83b3a7-be02-4193-9670-2cd794fa28b4` | `f8265a39-f80b-49fc-a1e0-9b2204bc621b` | no |
| `ff8f169a-be4a-42f6-a299-604d06201ba1` | `05d09c55-ccdf-4d26-b06d-b851a2314a35` | `362f3144-95b8-46c4-8975-f42fa9a4db93` | yes |
