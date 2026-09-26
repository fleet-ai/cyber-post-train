# Qwen3.8 cyber SFT study — live gates (2026-09-25)

Goal: measured lift on new Fleet blackbox tasks. Historical 96k run `chris-q38-t3k96-b8-v3-e7d5c0f8` proves mechanics, **not** this independent corrected-data study. See [HOLDOUT.md](HOLDOUT.md) for source and split evidence. Its checkpoints cannot satisfy this study: a new 96k arm needs separate data, name, output, saves, and matched held-out evaluations.

## Current 96k pilot (September 25)

The reproducible `training/safe_subset.py` selection retains 319 whole successful teacher sessions from 76 reviewed TRAIN families: 369 complete windows and 3,639,296 distinct masked assistant tokens. It excludes wrapped tool calls and every tool result above OpenCode's 50-KiB/2,000-line limit; no DEV/TEST family enters TRAIN. Before the result-size filter, the same direct-call/300k-per-family ceiling offers 538 sessions, 115 families and 7,322,663 target tokens; admitting those requires a proved OpenCode-equivalent result truncation, while removing the family ceiling would restore the old concentration defect. The exact corpus manifest is `sha256:317769bbe55b0911e45083cf05250c8ebcdc9079b5c971bfb0c204f285735110` and TRAIN Parquet is `sha256:33fc76c8c27526f539afb13e8d60b10b7c1c50fd36adb1be869779b4e2568bbb`. Independent native row validation and the exact-image zero-GPU CPU preflight passed (Job UID `034660cc-4882-405c-989b-77a0bc873901`, Pod UID `34e2091d-b57f-40eb-9b1b-c437107dc5d7`).

The create-once c1/q1, root-alert-off pilot is `chris-q38-safe96-sft-v1-50b39697` (API run `50b39697-2553-4941-bb96-c6519986d9d2`, RayJob UID `35837375-7a41-4fb0-b7cb-0c164e5ba311`, Workload UID `02585d60-004c-4f63-b8d6-bcd743724abe`). It was admitted at 23:42:06 UTC; RayCluster UID `779c2056-5a7b-4ec5-b9d3-e8e3933014cb` and Ready Pod UID `c1d791c1-df61-42e1-920c-c726172a2967` bound the exact image with zero restarts. At 07:00:24 UTC September 26, exact RayJob/Jobs API `SUCCEEDED`, Workload `Finished=True`, and absent RayCluster/owned Pods proved eight-GPU release. Zero-GPU read-only inspection Job UID `da9eae90-4a54-4f5b-97ad-13a17a98174e` subsequently verified self-digest-valid step-25 and step-47 receipts, final progress 47/47, and 33 native files totaling 324,627,486,667 bytes at each checkpoint; it and its Pod were then deleted. Step-47 receipt file SHA-256 is `dd329907864fc382e87d76fb219420358707e03b5a99d3d22fd7642078c78a40`; no `COMPLETED.json` or `FAILED.json` existed at inspection. A first zero-GPU inspector was promptly removed after its digest-only image URI failed to pull; future Jobs must use the full registry@digest reference. CPU-only c1/q1/root-alert-off seal Job `q38s-1e9b8192-000047-seal` (UID `ce80a33d-8735-46a2-b25c-2b05f769ea74`) completed at 07:18:59 UTC with zero restarts and self-digest-valid `SEAL.json` file SHA-256 `8273e82a30535e5bb978090eac7a8f0b19638f6353851f575acbcc5fa8bc8d96`; its Job/Pod were then removed. CPU-only export Job `q38-ck-1e9b8192-000047-export` (UID `4ec3a58f-548a-4a6b-ae78-09a017f1aaab`, Pod UID `e424917c-eb36-415e-b5b7-b7483daacd1f`) completed at 07:35:32 UTC with zero restarts and exit 0. Separate zero-GPU inspector Job UID `e6cb507e-7375-47de-b02b-2ed70209a6c5` verified the published BF16 export's self-digest, exact step/seal binding, and 29 files totaling 55,586,032,099 bytes; `EXPORT.json` file SHA-256 is `6e6a1eed06440d4f22282e79c791ed3213d03a05032244c583e86f7c2673f53a`. Both CPU Jobs and Pods were removed after evidence preservation. CPU reload check Job `q38-ck-1e9b8192-000047-cpu` (UID `f509322d-c727-4b75-a67b-0254ce40b5f9`) was active at 07:39 UTC; no reload result is yet proven. Plan SHA-256 `1e9b8192f46d007121302e1047bfcfc39fe9bafb4bcc6f18fb6178edc04bd127`: 47 steps, batch 8, LR 3e-6, 96k maximum, checkpoint at step 25 and final step 47, W&B, no teacher-CE validation. This is a *pilot*, not an accepted capability result: source-to-live tool-result equivalence and the matched Fleet evaluation route remain open. Do not infer lift from its training loss.
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

## Live tool-format check (2026-09-25)

Two first requests matched pinned OpenCode 1.18.27 anchors/tools, but replayed
historical tool results differed from stored strings. Fleet Jobs rejected
`harness=opencode`; direct instance creation returned HTTP 500 on three TRAIN
versions. At 06:25 UTC, deployed readback still lacked immutable environment-version
and seed identity; local Theseus source changes are not deployed proof.
Local Theseus `7aa3bb3473d` supplies a specific failure hypothesis: the
version-scoped creator omits exact task/environment-version IDs and passes a
direct image URL, while network-task admission requires those IDs and forbids
the URL with an exact environment ID. Test this against the deployed revision;
do not treat it as the proven cause of the HTTP 500 or retry unchanged creates.
Across 144 TRAIN keys, 6,648 sessions existed (4,143 successes), but their
stored tool strings are not captured model-facing messages. Keep the 943
TRAIN/27 DEV historical successes provisional pending same-call wire proof or
fresh verifier-backed OpenCode teacher collection. All probes released resources.

## Parallel execution lanes

- **96k fast lane:** one node; family-disjoint data, checkpoint, zero-step
  reload and tool match. The pilot has no teacher-CE validation; run paired
  Fleet DEV pass@4 only after checkpoint and route readiness are proven.
- **262k capacity lane:** the v2 attempt failed before training because its
  entrypoint added a digest field rejected by the frozen validator. V3 was
  deleted while queued: its pause path repeated the known missing finalizer.
  V4 repaired only that post-save call and passed exact-image CPU preflight.
  One c1/q1 POST created `chris-q38-t3k262-4n-can-v4-441c5703` (API run
  `441c5703-3e78-481f-a12c-3273ad81977a`, RayJob UID
  `cfede13d-a3e0-4ad3-818b-02f8538f8017`). It succeeded at 05:27 UTC
  September 26 after one finite optimizer update and released all four nodes.
  Its first 32 rows included 15 at least 250,000 tokens long; the maximum
  was 262,126/262,144. The saved checkpoint has 105 files across 32 ranks,
  302.4 GiB total. The saved sampler state exactly matches replay: step 2 has
  20/32 rows ≥250k (maximum 262,144), and step 3 has 14/32 ≥250k.
  CPU Job `chris-q38-262k4n-step1-seal-a01` (UID
  `e279c843-dbee-4bd1-b816-4ad3da6e143d`) sealed it; the self-digest-valid
  manifest file SHA-256 is `76184b48335e3382bb05510ef70c6628fe119230359b826bad29a7602bec4e2c`.
  The v4 reload was withdrawn through the Jobs API before admission: code review
  found its recovery worker could replace the bounded long-context worker.
  Corrected v5 passed exact-image CPU preflight (Job UID `2086b070-1882-4dc4-913f-2c8c50737c21`, exit 0, zero restarts), then queued
  as `chris-q38-t3k262-4n-reload-v5-e33ced3d` (RayJob UID
  `6ba1dedd-1268-43c9-ba44-d70e114ab4a7`, Workload UID
  `a63ce5c2-c193-4387-9fb8-db0101cb43f1`). Outcome pending; this is capacity
  proof, **not scientific SFT** or lift.
- **262k scientific lane (not launchable):** requires a new identity and ≥20M
  verified masked TRAIN tokens in OpenCode format; disjoint DEV/final families.

Full262 hypothesis: full-weight Qwen3.8-27B, four B300 nodes, 262k context,
batch 32, one epoch, LR 3e-6; `max_steps = ceil(train_rows / 32)`.
Keep all saves for BF16 and matched Fleet DEV/WEB pass@4; loss is not lift.

Before full262 POST, verify c1/q1, root alert-off, four-node release,
duplicates, empty output and capacity. The old 57M-token CPU/preview gates
do not authorize a new run; reload acceptance and fresh gates remain required.
No full POST occurred. Require >10 percentage-point final-task lift with
statistical evidence; never iterate on the final panel.
