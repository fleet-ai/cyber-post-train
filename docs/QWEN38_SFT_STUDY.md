# Qwen3.8 cyber SFT study — live gates (2026-09-25)

Goal: measured lift on new Fleet blackbox tasks. Historical 96k run `chris-q38-t3k96-b8-v3-e7d5c0f8` proves mechanics, **not** this independent corrected-data study. See [HOLDOUT.md](HOLDOUT.md) for source and split evidence. Its checkpoints cannot satisfy this study: a new 96k arm needs separate data, name, output, saves, and matched held-out evaluations.

## Current 96k pilot (September 25)

The reproducible `training/safe_subset.py` selection retains 319 whole successful teacher sessions from 76 reviewed TRAIN families: 369 complete windows and 3,639,296 distinct masked assistant tokens. It excludes wrapped tool calls and every tool result above OpenCode's 50-KiB/2,000-line limit; no DEV/TEST family enters TRAIN. Before the result-size filter, the same direct-call/300k-per-family ceiling offers 538 sessions, 115 families and 7,322,663 target tokens; admitting those requires a proved OpenCode-equivalent result truncation, while removing the family ceiling would restore the old concentration defect. The exact corpus manifest is `sha256:317769bbe55b0911e45083cf05250c8ebcdc9079b5c971bfb0c204f285735110` and TRAIN Parquet is `sha256:33fc76c8c27526f539afb13e8d60b10b7c1c50fd36adb1be869779b4e2568bbb`. Independent native row validation and the exact-image zero-GPU CPU preflight passed (Job UID `034660cc-4882-405c-989b-77a0bc873901`, Pod UID `34e2091d-b57f-40eb-9b1b-c437107dc5d7`).

The create-once c1/q1, root-alert-off pilot is `chris-q38-safe96-sft-v1-50b39697` (API run `50b39697-2553-4941-bb96-c6519986d9d2`, RayJob UID `35837375-7a41-4fb0-b7cb-0c164e5ba311`, Workload UID `02585d60-004c-4f63-b8d6-bcd743724abe`). It was admitted at 23:42:06 UTC; RayCluster UID `779c2056-5a7b-4ec5-b9d3-e8e3933014cb` and Ready Pod UID `c1d791c1-df61-42e1-920c-c726172a2967` bound the exact image with zero restarts. A self-digest-valid `PROGRESS.json` at 23:58:39 UTC proves training optimizer step 1/47; no checkpoint is yet proven. Plan SHA-256 `1e9b8192f46d007121302e1047bfcfc39fe9bafb4bcc6f18fb6178edc04bd127`: 47 steps, batch 8, LR 3e-6, 96k maximum, checkpoint at step 25 and final step 47, W&B, no teacher-CE validation. This is a *pilot*, not an accepted capability result: source-to-live tool-result equivalence and the matched Fleet evaluation route remain open. Do not infer lift from its training loss.
Kueue auto-unsuspends admitted CPU Jobs. For SFS staging, `/mnt/sfs/jobs` is root-owned (0755): preview a zero-GPU, alert-off root container, transfer only sealed data, verify remote digests, then release the exact staging Job. Native teacher DEV needs a unique `window_id`; the first image preflight failed at pinned `sft_runtime.py:1282`, and corrected data v2 has remote SHA-256 `9ea52340b56daba3b2a7b2afd091d078e759bdac45bcab4a29f5a10bbaa0e252`. The first provisional fast96 corpus had 12,869 windows, 1,198,182,936 input tokens and only 20,295,844 masked targets (1.69%); one family supplies 27.9% and the top ten 77.8% of targets. Do not submit its 1,609-step plan as an accepted lift experiment; source/tool parity and family balance remain open. Exact-image zero-GPU CPU preflight passed under Job UID `ed32ed5f-9f1e-4704-ba99-21a99b03c239`/Pod UID `d18e81d2-a13f-4663-8ecd-dd6ef6bb5f9f`. The separate one-step diagnostic `chris-q38-fast96-probe-v1-5dc31065` was admitted at 2026-09-25T17:47:37Z: RayJob UID `8f2b38fd-926d-490b-8ca8-d8ee3d86bf28`, Workload UID `888b9104-e22e-4c41-b888-75d02748b624`, RayCluster UID `36e36ac5-de4b-4a63-bfe2-022a421fafa2`, Pod UID `56f11e8f-86ff-475a-9a74-d298edeaaf56`, root alerts off. Its CPU gate Job UID `b2bd8e21-5577-4b90-853f-1c5f48157dc7`/Pod UID `8c97215b-bc89-4b27-9576-121e6c042661` passed. The GPU POST succeeded once; local command then reported failure because it tried to release a Lease with forbidden duration 0. Code now preserves duration 900, and the exact project Lease was released. Never retry the POST. This is a mechanics probe, not a scientific capability run.

The probe completed one finite optimizer step and wrote a native checkpoint plus teacher DEV receipts for steps 0 and 1. DEV token-weighted CE was 0.3094313→0.3097003 and task-macro CE 0.2914641→0.2925198; one step is insufficient to infer a trend. It failed only after these artifacts, at the planned-pause finalizer: pinned `WorkerDispatch` has no `finalize_pending_saves`, while its FSDP save already synchronously waits via `ray.get`, barrier and CUDA sync. Exact Jobs API deletion and UID-bound Kubernetes readback confirmed all eight GPUs released. The successor overlay skips that nonexistent finalizer only for the reviewed dense FSDP layout; checkpoint receipts remain mandatory. Two zero-allocation CPU seal attempts were removed: one lacked free quota at 96 GiB, and one requested 16 cores where nodes expose only 15.9. The right-sized, CPU-only c1/q1/root-alert-off successor `q38s-cd72bc84-000001-seal` (Job UID `3289bff5-7d4e-475e-9c97-bc0a7d1b4a4d`, Pod UID `f323e1b3-007c-4951-b942-f24ed37358a4`) was admitted and began hashing the exact saved checkpoint. Sealing/export/reload do not turn this diagnostic corpus into accepted scientific data.
The separately named v2 mechanics probe binds the 2,407-window, 20,295,844-target-token repack at `/mnt/sfs/jobs/chris-q38-fast96-dense-data-v3`, the patched runtime SHA-256 `ed02280ec077cf19240c30a15827219fef1f0c3743952051cab431a401e5321d`, plan SHA-256 `695635cc7180dce7d01c40e0da6d4a15166c7675a39a0a0594de0405a2d536d1`, and request SHA-256 `028974fcd8dbfc1b598d01de8692cbc5ba09f9f028145f6d7968f9f0553d0513`. Its exact-image, zero-GPU CPU preflight Job `chris-q38-fast96-probe-v2-pre-a01` (UID `75e0b191-8d24-4f8c-9e45-4cd8c0c46f5f`) was created with c1/q1 and root alerts off; no v2 GPU submission has occurred. V2 is strictly diagnostic while source/tool parity remains unproved.

## Why the old run cannot answer the question

Old packing clipped **11,794/14,693** rows, often losing task/tool anchors; tools differed from OpenCode eval. Of 149 prefix-salvaged sessions, 148 lost the final report. Step-1000 mixed output-limit failures and leaked families: **not defensible lift evidence**. New data must retain private originals, match target anchors/tools, preserve complete early/middle/late rounds without duplicate targets or submit bias, freeze roles, and verify native masks. A controlled repack reduced optional prior context from 96k to 16k with the same 96k maximum and source: **2,407** windows, **156,971,822** input tokens and exactly the same **20,295,844** unique masked targets from 933 sessions/115 families as the 12,869-window source. The independent sorted target-set SHA-256 matched (`43553d9e1c5183e086538fb228451a800ea21c9b038de7d184c5d904fe3e5bdd`); target density rose from 1.69% to 12.93% (7.63× less input), with one Parquet row group per window. The repacked TRAIN receipt is `sha256:f0f143ca2c96bb46f1cd8d34e0881a3c08656eb038677a7d00d464a38f184fe6`. A disjoint 27-window teacher DEV was sealed with it as diagnostic manifest `sha256:a0b8896d03a22b5acc15a14380cb2e2fe52f820368c1ec728dba35380ed05b6d` and staged at `/mnt/sfs/jobs/chris-q38-fast96-dense-data-v3`; local/remote SHA-256s agree for TRAIN (`ce93adc897bf80ec567fefbdd581afff7e6976f53e83d272e5a2b8fc14d26c66`), DEV (`565da059a56f723f0230c96ad464226f1e8f83661013b5a04278cfe708423956`), and manifest (`e283af1cf2e06ed62c1a1390acc6647d08ceeaa9e37f65c277beba06cdb9e98c`). A zero-GPU staging Job first lacked root-owned SFS write permission and was deleted; exact replacement Job UID `8cb6b4cc-0207-4fc4-9893-0e4c799549c5` staged, verified, and was deleted. The corpus is still **diagnostic-only**: tool-result parity and family concentration remain unresolved. `target_dense.audit` correctly rejects the v2 private source because it lacks the later required served-request attestation; do not relabel that source as accepted.

## Two held-out measurements

Teacher cross-entropy on unseen successes diagnoses optimization, **not** exploit ability. The unused provisional teacher-CE builder was removed. Matched base/checkpoint Fleet pass@4 on runnable versions measures ability: select on DEV, confirm once on final, and exclude infrastructure failures.

## Source-to-OpenCode compatibility

All **2,886** envelopes passed session/version/transcript digests. Provisional strict set: **943 TRAIN/116 families**, **27 teacher DEV/9**; 16 protected sessions quarantined. Hidden reasoning is excluded and OpenCode anchors/calls substituted. The current first-request Qwen wire is checked below, but historical tool-result parity is unproved. `use_tool` wrappers stay out: CLI/MCP blocks are unbound; none of 28,312 wrapped bash strings recovers a unique final status. A private aggregate-only recount found `use_tool` in 934 source sessions, with 28,703 `OkayOutput` strings all below OpenCode's 50-KiB/2,000-line caps. Truncation is not the main wrapper obstacle; exact result-to-OpenCode equivalence remains unproved.
One-text-block conversion adds 219 TRAIN/15 DEV: **1,162/42** sessions, **153/12** families, old token proxies **22,295,624/639,694**.
Native masked counts, live wire and tool-result parity remain unproved. OpenCode 1.18.27 truncates above 50 KiB/2,000 lines; only 362 TRAIN/7 DEV sessions are size-safe—too little for accepted full SFT.

## Bounded live target-wire probe (2026-09-25)

The Chris-owned exact-base route (model UID `54beeb64-c498-4e7d-a504-05a8d694808b`, Pod UID `6a7308b4-4e8b-4683-a21f-9c32147dfd0a`) served two frozen TRAIN task anchors through pinned OpenCode 1.18.27 and the fixed proxy. Both first requests matched the reviewed system/user and two-tool hashes; both returned complete HTTP-200 SSE streams with distinct response IDs. The synthetic local MCP did not execute task tools. The first probe entered a tool loop: 10 bounded live chat requests total, each capped at 128 output tokens. The route was paused and independently read back at zero Pods/replicas and routing disabled.
Private sealed diagnostic: `/private/tmp/cpt-q38-live-qwen-wire.YKisC9/LIVE_WIRE_DIAGNOSTIC.json`
(file SHA-256 `c6d475ba5b54bb2a34b1f2f68ee4aa4e10964873de0208d7cc4ae326e81eeeed`).
This proves first-request wire, **not** historical tool-result parity or corpus acceptance; `training_ready` stays false.

An isolated OpenCode 1.18.27 mock confirmed that two one-block MCP text results reached its model request byte-for-byte. A separate current TRAIN task/version was then probed with one exact historical bash call: the stored teacher result was 2,451 characters, while the fresh MCP text was 2,444 characters and marked as a tool error. The fresh instance was independently confirmed stopped with its durable claim (sealed terminal
`sha256:53d52f465cc5428da273428d7551d1131b7aa137cd4a4fe25459f75ad18b7294`).
Different runtime output may explain the difference; it does **not** prove a
renderer bug or equivalence. The historical corpus remains unaccepted. An
earlier attempt against an obsolete TRAIN task version was rejected before an
instance was created; its durable claim was independently absent. Fresh teacher
collection through Fleet OpenCode was also rejected before job creation with
`sales_product_required`. The live OTS Cyber project has no default billing
selection; the active catalog offers both OTS Dataminer and Platform General.
The unsupported native OpenCode Jobs launcher was removed; matched DEV attempts use the direct, version-pinned `evals/direct.py` path.
An exact one-task TRAIN-only probe against current production version
`43dc2c52-f283-4454-8987-69c6cbbd4160` used Platform General. The Jobs API
rejected `harness=opencode` (HTTP 400); it accepts Claude Code, Codex, Grok and
Grok-bot. Direct Fleet-instance creation then returned generic HTTP 500 before
a claim or instance existed on three current TRAIN versions (fira, fentry,
rops). Delayed claim readback remained absent. Stop repeating this route until
its server defect is understood; no model rollout or GPU allocation occurred. A read-only check of a previously full-credit DEV instance found its task's exact seed bindings matched, but `environment_version_id` and `image_url` were absent from the instance readback. The server create route uses the task's version label but does not pass its frozen seed configuration or exact environment-version ID; this is a separate runtime-identity proof gap, not an established cause of the HTTP 500s. Require an authoritative immutable runtime binding before paired evaluation.
Another current TRAIN version returned one successful MCP text block but differed
from its stored result by one character (2,251 versus 2,252); line-ending
equivalence was **not** established. A bounded follow-up timed out during
creation after the server materialized an instance. Its exact durable claim
enabled cleanup, and independent readback confirmed that instance `stopped`.
No further retry of that probe is planned without a changed diagnostic.

A September 25 read-only scan of all 144 strict TRAIN task keys found 6,648 historical sessions (4,143 successful); only seven began after September 20, and the four successes were Qwen on one key, not new teacher data. An exact strict TRAIN session had 53 stored tool-result
strings but neither raw MCP response blocks nor a separately captured next
model-facing tool message. Its harness metadata lacks a renderer binary/image
digest, and its instance lacks environment-version/image pins; exact session
and instance detail readbacks returned 404. A result string alone cannot prove
Grok-to-OpenCode rendering equivalence. The 943 strict TRAIN/27 DEV historical
sessions and one-block salvage remain provisional. Admit none as scientific
training data without immutable renderer and same-call wire evidence; otherwise
collect fresh verifier-backed OpenCode teacher successes on TRAIN families.
The Fleet transcript exporter at Theseus `e2b20f08` explicitly marks its
stored available-tools snapshot `model_facing: false` and
`requires_harness_reconstruction: true`. An exact old TRAIN session returned
two stored names (`bash`, `submit_report`), not the provider-facing bytes.

## Parallel execution lanes

- **96k independent fast lane:** one node/eight GPUs; corrected family-disjoint
  data, update, native checkpoint, zero-step reload and tool match. At **every**
  checkpoint immediately queue teacher loss and matched Fleet DEV pass@4;
  retain state through eval (old keep-two evicted too early).
- **262k capacity lane:** four-node 112-row canary
  `chris-q38-t3k262-4n-can-v2-1502ba9f` was admitted on four B300 nodes on
  2026-09-25, then failed before training: the entrypoint added a plan digest
  field that the candidate's strict validator rejected. No optimizer step or
  checkpoint was proven; the exact Pods and Ray cluster were released. The
  narrowly repaired v3 validator checks that field's digest, and its local
  regression test reproduces the v2 failure. Exact-image zero-GPU preflight v6
  passed and was released. The create-once v3 RayJob UID
  `0b135d1f-3326-401b-a83c-68923d6e1b6b` / Workload UID
  `14050301-32dd-4855-bcc9-9d3919b5619f` was deleted through the Jobs API
  while still queued with zero GPUs: its frozen pause path repeated the known
  nonexistent `finalize_pending_saves` call. V4 changes only that post-save call
  and its create-once identity. Exact-image zero-GPU CPU preflight v7 passed,
  verified 112 rows/3,022,959 targets, and was released. Exactly one v4 GPU
  POST created `chris-q38-t3k262-4n-can-v4-441c5703` (API run
  `441c5703-3e78-481f-a12c-3273ad81977a`, RayJob UID
  `cfede13d-a3e0-4ad3-818b-02f8538f8017`, Workload UID
  `166cf858-82a1-46d0-9f7a-975776745b7b`); it was queued without GPUs at
  00:35 UTC September 26. Request SHA-256 `f8ff4d2f8cbdf81f95547b0e9e1ec53b3015de313efe639f39624c753c7d0389`.
  `training/long_context_reload.py` remains review-only until the
  exact v4 step-1 checkpoint is sealed. This lane is **not
  scientific**. Require real
  near-262k first-batch update, finite gradients, peak memory, checkpoint,
  release and separate zero-step reload. Old eight-node step/zero-GPU preflight
  prove neither; CPU preflight v2/v3 failed, pinned v4 passed.
  Earlier queueing held zero GPUs until the 22:37 UTC admission.
  A restore-only adapter and synthetic test exist, but no source step-1 seal,
  exact-image restore preflight, or GPU reload acceptance exists yet.
- **262k corrected scientific lane (not launchable):** freeze independent
  name/output/data/steps after source/capacity gates. Require transitive roles,
  verified successes, exact OpenCode 1.18.27 anchors/tools, visible complete
  rounds, unique targets, ≥20M **actual** TRAIN masked tokens and disjoint
  contiguous teacher DEV. Final families never train. Old runtime binds a
  capacity-only corpus/no-CE mode; qualify a new binding.

Initial full262 hypothesis after capacity/reload pass: full-weight Qwen3.8-27B,
four 8-B300 nodes, 262,144 context, global batch 32, one epoch, LR 3e-6,
group-1/GDN-512/chunk-1024. Freeze `max_steps = ceil(train_rows / 32)`, ~10–20
spaced CE/checkpoints; retain **all** native saves through eval and budget SFS.
Validate zero-step recovery with a new output/W&B identity. Each checkpoint
needs BF16 readback and matched base/candidate Fleet DEV and OpenCode
WebExploitBench pass@4. Keep final families sealed; train/teacher loss is not lift.

Before full262 POST, server-preview root `fleet.ai/failure-alerts: "off"`,
c1/q1, four-node shape/release; recheck duplicates, empty output and capacity.
The historical 57M-token full request passed an exact-image zero-GPU CPU check
and live four-node/32-GPU preview under the earlier v3 runtime (receipt in
`docs/evidence/qwen38-262k-full57-cpu-preflight-v1-passed.json`). Its data/model
checks remain useful, but the v4-derived request needs a fresh exact-plan CPU
preflight before submission. `submission_authorized` remains false until canary
checkpoint and reload acceptance. No full POST has occurred. Apply [AGENTS.md](../AGENTS.md). Until
Chris clarifies the wording, use stricter **>10 percentage-point absolute**
final-task lift with statistical evidence; never iterate on the final panel.
