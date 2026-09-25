# Qwen3.8 cyber SFT study — live gates (2026-09-25)

Goal: measured lift on new Fleet blackbox tasks. Historical 96k run `chris-q38-t3k96-b8-v3-e7d5c0f8` proves mechanics, **not** this independent corrected-data study. See [HOLDOUT.md](HOLDOUT.md) for source and split evidence. Its checkpoints cannot satisfy this study: a new 96k arm needs separate data, name, output, saves, and matched held-out evaluations.
Kueue auto-unsuspends admitted CPU Jobs. For SFS staging, `/mnt/sfs/jobs` is root-owned (0755): preview a zero-GPU, alert-off root container, transfer only sealed data, verify remote digests, then release the exact staging Job. Native teacher DEV needs a unique `window_id`; the first image preflight failed at pinned `sft_runtime.py:1282`, and corrected data v2 has remote SHA-256 `9ea52340b56daba3b2a7b2afd091d078e759bdac45bcab4a29f5a10bbaa0e252`. The first provisional fast96 corpus had 12,869 windows, 1,198,182,936 input tokens and only 20,295,844 masked targets (1.69%); one family supplies 27.9% and the top ten 77.8% of targets. Do not submit its 1,609-step plan as an accepted lift experiment; source/tool parity and family balance remain open. Exact-image zero-GPU CPU preflight passed under Job UID `ed32ed5f-9f1e-4704-ba99-21a99b03c239`/Pod UID `d18e81d2-a13f-4663-8ecd-dd6ef6bb5f9f`. The separate one-step diagnostic `chris-q38-fast96-probe-v1-5dc31065` was admitted at 2026-09-25T17:47:37Z: RayJob UID `8f2b38fd-926d-490b-8ca8-d8ee3d86bf28`, Workload UID `888b9104-e22e-4c41-b888-75d02748b624`, RayCluster UID `36e36ac5-de4b-4a63-bfe2-022a421fafa2`, Pod UID `56f11e8f-86ff-475a-9a74-d298edeaaf56`, root alerts off. Its CPU gate Job UID `b2bd8e21-5577-4b90-853f-1c5f48157dc7`/Pod UID `8c97215b-bc89-4b27-9576-121e6c042661` passed. The GPU POST succeeded once; local command then reported failure because it tried to release a Lease with forbidden duration 0. Code now preserves duration 900, and the exact project Lease was released. Never retry the POST. This is a mechanics probe, not a scientific capability run.

The probe completed one finite optimizer step and wrote a native checkpoint plus teacher DEV receipts for steps 0 and 1. DEV token-weighted CE was 0.3094313→0.3097003 and task-macro CE 0.2914641→0.2925198; one step is insufficient to infer a trend. It failed only after these artifacts, at the planned-pause finalizer: pinned `WorkerDispatch` has no `finalize_pending_saves`, while its FSDP save already synchronously waits via `ray.get`, barrier and CUDA sync. Exact Jobs API deletion and UID-bound Kubernetes readback confirmed all eight GPUs released. The successor overlay skips that nonexistent finalizer only for the reviewed dense FSDP layout; checkpoint receipts remain mandatory. Two zero-allocation CPU seal attempts were removed: one lacked free quota at 96 GiB, and one requested 16 cores where nodes expose only 15.9. The right-sized, CPU-only c1/q1/root-alert-off successor `q38s-cd72bc84-000001-seal` (Job UID `3289bff5-7d4e-475e-9c97-bc0a7d1b4a4d`, Pod UID `f323e1b3-007c-4951-b942-f24ed37358a4`) was admitted and began hashing the exact saved checkpoint. Sealing/export/reload do not turn this diagnostic corpus into accepted scientific data.

## Why the old run cannot answer the question

Old packing clipped **11,794/14,693** rows, often losing task/tool anchors; tools differed from OpenCode eval. Of 149 prefix-salvaged sessions, 148 lost the final report. Step-1000 mixed output-limit failures and leaked families: **not defensible lift evidence**. New data must retain private originals, match target anchors/tools, preserve complete early/middle/late rounds without duplicate targets or submit bias, freeze roles, and verify native masks. A controlled repack reduced optional prior context from 96k to 16k with the same 96k maximum and source: **2,407** windows, **156,971,822** input tokens and exactly the same **20,295,844** unique masked targets from 933 sessions/115 families as the 12,869-window source. The independent sorted target-set SHA-256 matched (`43553d9e1c5183e086538fb228451a800ea21c9b038de7d184c5d904fe3e5bdd`); target density rose from 1.69% to 12.93% (7.63× less input), with one Parquet row group per window. The repacked TRAIN receipt is `sha256:f0f143ca2c96bb46f1cd8d34e0881a3c08656eb038677a7d00d464a38f184fe6`. A disjoint 27-window teacher DEV was sealed with it as diagnostic manifest `sha256:a0b8896d03a22b5acc15a14380cb2e2fe52f820368c1ec728dba35380ed05b6d` and staged at `/mnt/sfs/jobs/chris-q38-fast96-dense-data-v3`; local/remote SHA-256s agree for TRAIN (`ce93adc897bf80ec567fefbdd581afff7e6976f53e83d272e5a2b8fc14d26c66`), DEV (`565da059a56f723f0230c96ad464226f1e8f83661013b5a04278cfe708423956`), and manifest (`e283af1cf2e06ed62c1a1390acc6647d08ceeaa9e37f65c277beba06cdb9e98c`). A zero-GPU staging Job first lacked root-owned SFS write permission and was deleted; exact replacement Job UID `8cb6b4cc-0207-4fc4-9893-0e4c799549c5` staged, verified, and was deleted. The corpus is still **diagnostic-only**: tool-result parity and family concentration remain unresolved. `target_dense.audit` correctly rejects the v2 private source because it lacks the later required served-request attestation; do not relabel that source as accepted.

## Two held-out measurements

Teacher cross-entropy on unseen successes diagnoses optimization, **not** exploit ability. Matched base/checkpoint Fleet pass@4 on runnable versions measures ability: select on DEV, confirm once on final, exclude infrastructure failures. The old paired structured-window DEV converter was unused and incompatible with the corrected dense TRAIN path. `training/teacher_dev.py` now builds a disjoint, target-compatible DEV panel offline, but deliberately rejects provisional source evidence; no trusted teacher cross-entropy panel exists yet.

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
its server defect is understood; no model rollout or GPU allocation occurred.
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
  `chris-q38-t3k262-4n-can-v2-1502ba9f` is **not scientific**. Require real
  near-262k first-batch update, finite gradients, peak memory, checkpoint,
  release and separate zero-step reload. Old eight-node step/zero-GPU preflight
  prove neither; CPU preflight v2/v3 failed, pinned v4 passed.
  At 2026-09-25 12:42 UTC it remained unadmitted, holding zero GPUs: the
  training queue had 25 GPUs below its 192-GPU quota, short of this job's 32.
  Kueue displayed a `cpu-head` topology fallback error; that alone does not
  establish a job-config defect while the B300 flavor lacks quota.
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
This draft authorizes no full POST. Apply [AGENTS.md](../AGENTS.md). Until
Chris clarifies the wording, use stricter **>10 percentage-point absolute**
final-task lift with statistical evidence; never iterate on the final panel.
