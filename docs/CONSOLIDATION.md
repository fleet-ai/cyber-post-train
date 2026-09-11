# Repository consolidation

Status as of 2026-09-11. This checklist is not launch authority. Historical
campaigns and other worktrees remain untouched.

Chris's current execution preference is Qwen first: prioritize its end-to-end
SFT/recovery, Miles/SkyRL RL and evaluation qualification. Full GLM remains a
completion requirement. Its first full-size allocation was released after a
loading bottleneck; prioritizing Qwen does not narrow the final model/backend scope.

## Interface and sources

The public CLI prepares data, pins model metadata, prepares/preflights/previews/
submits training, seals/exports checkpoints and operates PostgreSQL-backed Fleet
evaluations. Configuration is plain YAML. SkyRL/Miles own optimization; Fleet owns
isolated environments and authoritative grading. No new trainer or scheduler.

| Source | Consolidated contribution | Boundary |
|---|---|---|
| main `bac2a4c` | PostgreSQL queue, eval adapters, cluster safeguards | Live 100-task campaign stays on its frozen worker |
| `2f290e4` teacher-devicefix worktree | Dense SFT, held-out loss, W&B, checkpoints | Its full-GLM LoRA work has only CPU prerequisite evidence |
| `25ad2a3` training-prep worktree | Canary/export/reload evidence and failure lessons | Never replay historical runs |
| rollout-postgres-ledger worktree | Lease lifetime and exact instance TTL/data binding | Copied fixes, not a deployment into the campaign |
| webexploitbench-opencode-cluster worktree | Exact-image external eval preparation | Campaign is paused; do not restart it |

Companion tasks `fleet task evals`, `external benchmarks` and `cyber repo prep
and trace analysis` were inspected. Original commits/worktrees remain recoverable.

## What is actually qualified

| Check | Evidence and limits |
|---|---|
| Pinned-image SFT CPU tests | 63 tests, zero skips/failures/restarts/GPUs; [receipt](evidence/cleanup-native-sft-20260911.json) |
| Real data preparation | One complete teacher trace → eight windows/7,770 targets; two fixed dev tasks; [receipt](evidence/cleanup-qwen-cpu-20260911.json). Operational fixture, not a capability study |
| Real Qwen SFT | One optimizer step, held-out validation, W&B and native checkpoint; succeeded 01:03:32 UTC, all eight GPUs released; [receipt](evidence/cleanup-qwen-sft-20260911.json). CPU audit found 353 changed small tensors |
| Native checkpoint seal | 33 files/324,627,486,731 bytes, all eight ranks at step one; CPU-only; [receipt](evidence/cleanup-checkpoint-seal-20260911.json) |
| CPU checkpoint export | 33 pinned-image tests; real export succeeded 02:01:04 UTC; independent audit rehashed 29 files, checked 1,199 BF16 tensors, nine sidecars and tokenizer/config reload; [receipt](evidence/cleanup-qwen-export-20260911.json). Separate GPU qualification is recorded below |
| BF16 export GPU check | Public `checkpoint-check` passed 18 pinned-image CPU tests and real-export preflight, then one B300 loaded the complete model, ran a finite synthetic forward pass and generated two tokens. The run succeeded 09:35:10 UTC and released its GPU/quota; a separate CPU audit verified its receipts and single-POST journal. [Evidence](evidence/cleanup-qwen-export-gpu-20260911.json). No serving-engine, throughput, optimizer or capability claim |
| Public model pinning | Live exact-revision reads reproduced Qwen's existing 18-shard weight/sidecar hashes and pinned full GLM5.3's 141 shards/755,632,050,320 bytes in `configs/models/glm53-30333038`. No weights downloaded or GPUs requested |
| GLM LoRA CPU prerequisites | 68 pinned-image tests, zero failures/skips/restarts/GPUs; actual tiny GLM FP8 load, adapter updates, exact CPU resume and native worker integration; [receipt](evidence/cleanup-glm-native-20260911.json). **Not full-size or distributed GPU qualification** |
| Full GLM data/model preflight | All 141 staged weight shards verified, exact 744B model constructed on meta, 8 train windows/6,287 targets and two fixed dev tasks; 34 native checkpoint/seal tests also passed. [CPU receipt](evidence/cleanup-glm-preparation-20260911.json). The two-node qualification exposed slow rank-zero CPU dequantization while GPUs waited; released all 16 GPUs before off-node repair, with no optimizer step. [Startup/release evidence](evidence/cleanup-glm-startup-release-20260911.json) |
| GLM loading diagnosis | Native decoding rejects partial 128-row scale blocks. The correction matches an independent FP32→BF16 reference bitwise on two actual tensors and five synthetic shapes. All 141 shard headers show 79 affected matrices and no unsupported scale grids. A later actual-shard check converted 80 tensors/1.56B elements bit-identically with one versus four CPU threads (11.86s versus 4.29s). [Partial-block proof](evidence/cleanup-glm-partial-blocks-20260911.json), [actual-shard benchmark](evidence/cleanup-runtime-regression-20260911.json). Full-model loading, merging and distribution remain unqualified; no GPU resubmission |
| Native recovery CPU tests | 41 tests in the pinned image, zero skips/failures/restarts/GPUs; exact model/Adam state through the resumed epoch, complete later-epoch coverage, strict state checks and no-optimizer validation. [Receipt](evidence/cleanup-recovery-cpu-20260911.json). Native later-epoch reshuffles are explicitly not bit-identical |
| Planned pause/recovery | 181 pinned-image CPU tests pass. The real fixture paused after step one, then a fresh run restored all eight ranks and completed exactly one new optimizer update to step two. Succeeded 08:39:53 UTC; all eight GPUs/quota released. Train loss 0.5321; dev token-weighted loss 0.7086→0.6594 after the resumed update. [Result](evidence/cleanup-qwen-resume-result-20260911.json). Independent CPU audit succeeded 08:52:32 UTC: all 33 source files rehashed unchanged and the 33-file step-two checkpoint sealed. [Audit](evidence/cleanup-qwen-resume-checkpoint-audit-20260911.json). No capability claim |
| Real native checkpoint reload | Succeeded 04:34:51 UTC and released all eight GPUs/quota. Eight ranks restored model/optimizer/scheduler/RNG plus sampler; zero new optimizer steps; both held-out losses exactly match before shutdown. Independent CPU audit rehashed all 33 source files/324,627,486,731 bytes unchanged at 04:51:12 UTC; [receipt](evidence/cleanup-qwen-recovery-20260911.json). This is not the separate BF16 export or a resumed GPU optimizer step |
| Miles native CPU integration | 82 tests pass with zero failures/skips/restarts/GPUs; real MCP 2 transport, native FTI token assembly, exact Qwen tokenizer and corrected template. The single-attempt generation boundary covers all 256 module statements and 86 branches; [receipt](evidence/cleanup-miles-single-attempt-20260911.json). Synthetic engine/task replies only, not real RL optimization |
| Miles native argument builder | 128 combined CPU tests, zero failures/skips/restarts/GPUs; real Qwen TP4/CP2 recipe, bounded batches, checkpoint/dev controls and native offline W&B identity. Both argument and episode modules have full focused statement/branch coverage; [receipt](evidence/cleanup-miles-arguments-20260911.json). Generic API preview passed without allocation; not real GPU RL qualification |
| Miles data preparation | Public `rl-data` command; 174 combined pinned-image tests pass without skips/failures/restarts/GPUs. Real Qwen tokenizer/TITO and native Dataset retain every selected task; all three RL modules have complete focused line/branch coverage. [Receipt](evidence/cleanup-rl-data-20260911.json). Fleet HTTP replies are synthetic; no live reward or optimizer qualification |
| Miles checkpoint preparation | Public `miles-convert`/`miles-seal`, reusing preflight/preview/submit. 228 pinned-image tests pass; all 28 staged Qwen files and native converter arguments independently verified on CPU. [Receipt](evidence/cleanup-miles-conversion-cpu-20260911.json). One-node/eight-GPU API preview passes; the subsequent real conversion is recorded below |
| Strict Miles batches | 307 pinned-image tests pass with zero failures/skips/restarts/GPUs; all five RL integration modules have complete focused statement/branch coverage. Native batch/episode identities, no exception-driven refill, and awaited sibling cleanup are exercised with synthetic task/engine replies; [evidence](evidence/cleanup-miles-batches-native-20260911.json). No live RL qualification is claimed |
| Real Miles base conversion | Native GPU conversion succeeded 06:17:40 UTC and released all eight GPUs/quota. CPU `miles-seal` then hashed 11 checkpoint files/53,794,115,609 bytes; [evidence](evidence/cleanup-miles-conversion-result-20260911.json). Zero optimizer steps; native checkpoint GPU reload and RL optimization remain open. The removed GPU Pod's imageID/exit/restart fields were not captured and are explicitly unknown |
| Public Miles RL preparation | `rl` compiles a native launch through the shared preflight/preview/submit rail. In the exact image, 49 CPU tests passed and one native-parser test failed because the CPU-only node has no `libcuda.so.1`; no GPU or training allocation occurred. CPU preflight separates the FTI builder/Dataset checks from GPU-only native parsing. [CPU evidence](evidence/cleanup-miles-training-cpu-20260911.json); real startup is recorded separately below |
| SkyRL native episode integration | 140 pinned-image CPU tests, zero skips/failures/restarts/GPUs in the corrected fixture; actual Qwen tokenizer, native client/helpers, MCP 1.28.0 and corrected XML parsing. Recorder/parser have full focused statement/branch coverage; [evidence](evidence/cleanup-skyrl-episode-native-20260911.json) preserves the first fixture's mount-path failure. Synthetic task/engine responses only |
| SkyRL native data preparation | `rl-data` accepts `backend: skyrl`, reusing the exact-task/split/GET boundary. 189 pinned-image tests pass with zero skips/failures/restarts/GPUs; actual Qwen tokenizer and `PromptDataset` preserve prompts and bindings. [Evidence](evidence/cleanup-skyrl-data-native-20260911.json) records the caught Transformers dictionary-default bug and fix. Synthetic Fleet replies only; not RL optimization |
| SkyRL strict native batches | 227 combined pinned-image CPU tests pass, zero skips/failures/restarts/GPUs. Native trajectory types and output validation, immutable attempt groups, token/log-probability alignment and awaited sibling cleanup pass; the batch module has complete focused statement/branch coverage. [Evidence](evidence/cleanup-skyrl-batches-native-20260911.json). Synthetic responses only, not real optimizer qualification |
| SkyRL native configuration | 275 combined pinned-image CPU tests pass, zero skips/failures/restarts/GPUs. Actual parser and native training-method tests verify whole batches, exactly one optimizer call per batch, bounded steps and checkpoint/dev settings. [Evidence](evidence/cleanup-skyrl-arguments-native-20260911.json). Synthetic dispatch, not real optimization |
| Public SkyRL RL preparation | `rl` now selects Miles or SkyRL and shares the bounded process/submission lifecycle. Corrected exact-image CPU run passed 420 tests, zero skips/failures/restarts/GPUs. It checks native driver/parser/data identities, actual sampling-to-recorder compatibility, Linux bundle transport and scalar-only tracking. [Evidence](evidence/cleanup-skyrl-training-native-20260911.json) preserves two earlier failures and their fixes. No real RL update, reward or checkpoint-reload claim |
| Real RL preparation/startup | Miles V6 passed the earlier startup/RAM failures and reached a real dev episode, then failed before grading. All eight GPUs/quota and its task environment are released. Its old receipt retained only ExceptionGroup, so the cause remains unknown; new tests preserve nested types/code locations and recognize the platform's stopped cleanup state. [V6 evidence](evidence/cleanup-miles-rl-v6-20260911.json). SkyRL's private-staging permission correction passed CPU preflight as the actual image user, but its GPU submission was withheld because it shares this episode lifecycle. [SkyRL diagnosis](evidence/cleanup-skyrl-rl-v3-20260911.json), [unsubmitted preparation](evidence/cleanup-skyrl-preparation-v4-20260911.json). No RL reward, update or recovery is qualified |
| Miles V7 bounded canary | Exact-image CPU preflight and GET-only input checks passed; same task split/model/native recipe, with a 330-second tool deadline and corrected diagnostics/cleanup. One Jobs API POST at 13:13:48 UTC; Workload priority 10,000; queued without GPUs at 13:14:58 UTC. [Submission evidence](evidence/cleanup-miles-rl-v7-20260911.json). No RL reward, update or recovery claim; no additional GPU wave while the conservative node envelope is full |
| Real evaluation lifecycle | Qwen generated, authoritative grading and catalog ingestion completed, local result persisted, and all resources cleaned. OpenCode ended at its output limit, so the frozen natural-stop gate holds the result: zero accepted, one review, no retry. [Evidence](evidence/cleanup-eval-terminal-20260911.json). This qualifies lifecycle execution, not a complete model outcome |
| Evaluation V6 | Terminal: one private result, complete authoritative grading/catalog ingestion and cleanup, but final output-limit stop. Zero accepted and one review; no active worker or retry. This separate operational treatment used 65,536 output tokens per request. [Evidence](evidence/cleanup-eval-v6-20260911.json). Not a capability comparison or training input |
| Latest native regressions | 329 Miles and 472 SkyRL CPU tests pass in their exact images with zero skips, including real MCP transport fault propagation. The episode module covers all 317 statements/116 branches in Miles. Two invalid Miles self-test setups are retained separately, not hidden. [Evidence](evidence/cleanup-runtime-regression-20260911.json). No new RL reward/update qualification |
| Local regression suite | 1,657 tests plus seven subtests pass using disposable PostgreSQL and MCP 2.1.1; 105 native/optional-dependency cases explicitly skip locally. Whole-repo coverage is 13,603/17,929 statements and 4,412/6,956 branches. Native-image and GPU qualification remain separate |
| Installation | Wheel with current episode/GLM fixes, SHA-256 `3524d5eb28013f8971a5a3f71b8a9e1a6f6425118f1f6b9b997db5a4920c2c18`, installed outside the checkout on Python 3.12.14. Eight public and two legacy data help paths, redirect/user guards, numeric JSON, module identity, replay rejection and nested-error privacy pass. [Evidence](evidence/cleanup-runtime-regression-20260911.json). No Torch/PyArrow in this minimal environment; installation is not training readiness |

The newer `rl-data` wheel was independently installed outside the checkout;
command help and RL module imports pass. Its digest is in the data-preparation
receipt. A local Finder `.DS_Store` exposed a portability bug in the skill-content
scan; it now ignores only that OS metadata file and still tests real skill content.
The metadata file was not deleted or committed.

The conversion wheel was rebuilt and installed outside the checkout; six public
command help paths, five training-module imports and portable bundle hashes pass.
Its digest is in the conversion receipt. Native conversion uses the pinned Miles
converter, not a new optimizer or model implementation. Its fixed 30-minute
deadline covers source validation and the owned child process group. Sealing is
CPU-only after resource release and does not claim GPU reload.
The conversion was submitted once at effective priority 10,000, then succeeded
and released its allocation; [submission evidence](evidence/cleanup-miles-conversion-submission-20260911.json)
and [terminal/seal evidence](evidence/cleanup-miles-conversion-result-20260911.json).
That historical receipt records the resource limit at submission, not current
authority. Chris replaced that limit on 2026-09-11 UTC: **at most eight actively
allocated experiment-owned nodes at once**. Queued, unallocated work does not
count and has no fixed numerical cap. Count admitted/startup allocations and
dedicated serving as well as training/evaluation nodes. Whether CPU-only nodes
are exempt has not been explicitly confirmed; until clarified, use the stricter
total-node interpretation rather than assume an exemption. Shared endpoints
owned by colleagues are not our experiment-owned allocations. The old
four-node/32-GPU limit including queued work is superseded.
The generic request validator now accepts up to eight nodes per request and
rejects nine. This removes the obsolete four-worker ceiling; it does not qualify
new model topologies or enforce the sum across independently admitted runs.

Reconcile allocation before submission and admission. The Jobs API admits queued
work independently, so observing eight active nodes is not an enforcement
mechanism: do not queue an unguarded batch that could simultaneously start above
the active limit. Use supported admission control or bounded waves until an
owner-scoped admission cap is proven. Never modify peer queues or bypass admission.

Split, checkpoint, export and model-pinning checks have
100% focused line/branch coverage; this is **not whole-repository coverage**.
Model-pinning tests exercise real HTTP parsing, pagination, tampering, download
bounds, errors and create-once output. Unused desktop `datasets`/`trl` were removed.
The lock now resolves 96 packages including the real MCP transport's test-only
dependencies; these do not expand the minimal installed CLI dependency set.

Earlier evaluation qualification attempts did not complete a rollout. The first check
exposed a claim-directory defect before instance creation; that is fixed and
regression-tested. A distinct successor then hit HTTP 404 at Fleet's exact
instance-create API despite successful exact task lookup. Neither generated
tokens, scored or produced a local result. All held rows are preserved in the
separate qualification databases. Server-side diagnosis is needed; HTTP
status alone does not prove the cause. Future errors retain an allowlisted
category and response digest, never raw server text or an automatic POST retry.
The runner's two direct instance-create calls initially bypassed that diagnostic
helper. Both now use the same single-attempt request boundary; tests exercise the
actual runner/preflight against HTTP 404 and 503, assert persisted safe diagnostics
and prove there is exactly one POST. This repairs future evidence collection, not
the unresolved server-side 404 or an already-held attempt.

A separate non-scored diagnostic used exactly one POST to
`/v1/rollout-rewards/<task>/versions/<version>/instances` and reproduced the 404;
no instance ID, model call or score resulted. Read-only probing confirms the route
exists (405 with `Allow: POST`) and exact task lookup succeeds under explicit
Fleet-team scope. The body fingerprint is retained but its cause remains
unclassified. Exhaustive queued/pending/running instance listings found no newly
created matching environment. [Diagnostic evidence](evidence/cleanup-eval-provisioning-20260911.json).
This is not the generic `/v1/env/instances` create route.

**Resolved diagnosis, 2026-09-11:** the saved response fingerprint exactly matches
the platform's static error for a historical task version without recorded
starting data. The original source review predated that September 10 change.
Current task GET confirms missing seed/data fields. Its mutable current version
has a different prompt and was not substituted. Shared eval/RL preflight now
rejects missing starting-data identity on GET, removing the unsupported deferred
instance-check exception. A score-blind census found 16 of the first 20 exact
train versions outside the active campaign still bind correctly. These are
eligible for a separately frozen operational qualification, not replacement
credits for the held rows. [Diagnosis and source fingerprint](evidence/cleanup-eval-seed-diagnosis-20260911.json).

The next qualification (`eval-prepared-v4`) successfully provisioned and bound an
exact challenge and tool catalog, then stopped during OpenCode startup. A fully
offline synthetic reproduction produced the identical stderr fingerprint:
Docker's arbitrary host UID had no home directory, causing an attempted write
under `/`. Set HOME explicitly and test data-directory initialization under the
same user/mount before claims. This native image check passes. The environment
and containers are released, and the unscored row remains held—not accepted or
automatically retried. [Runtime evidence](evidence/cleanup-eval-runtime-20260911.json).

## Remaining completion gates

- [x] Fix task-family leakage across versions and consolidate dense preparation.
- [x] Establish one generic Jobs API submission boundary with durable journals.
- [x] Qualify Qwen SFT, CPU sealing/export and installed public commands.
- [x] Qualify native GPU checkpoint/optimizer reload with source immutability.
- [x] Qualify resumed GPU optimization after a planned checkpoint pause.
- [x] Rehash the continuation's source unchanged and seal its final checkpoint on CPU.
- [x] Qualify the separate BF16 inference export's GPU reload and synthetic forward/generation.
- [ ] Integrate and qualify **full GLM5.3**, not GLM Flash.
- [ ] Integrate/qualify Miles and SkyRL RL: actual rollouts, authoritative reward,
      optimizer update and recoverable checkpoint.
- [ ] Complete real evaluation qualification after the API issue is resolved.
- [ ] Finish caller-checked retirement, coverage review and final wheel/CLI tests.
- [ ] Update from current main, review final diff and finish concise docs/skills.

## Removed surfaces and retained history

Removed unused spec/catalog and model-adapter facades, typed Jobs launch commands,
fixed-window corpus builder, XBEN/CVE-Bench fallback, GLM5.2 probes/configs,
run-specific export CLI/collector, redundant guidance and generated legacy Jobs.
Also removed ten completed-run cluster shell launchers, the superseded raw
Kubernetes submitter/policy module and their obsolete checks (1,214 net lines).
Current submission safeguards remain in the generic Jobs API boundary and its
behavioral tests. Historical model contracts and receipts remain unchanged.
The maintained surface is the public CLI plus Fleet/WebExploitBench/ExploitGym
adapters. Artifact-integrity, tamper, concurrency and scientific-control tests
remain; external benchmark material never entered training.

Retired the isolated GLM5.2 Agent Runtime launcher, prompt-curriculum authoring
pilot, Qwen Code calibration launcher and raw-Kubernetes Qwen Code smoke/test20
scripts. Their only code callers were their own obsolete launch paths and tests;
the OpenCode/PostgreSQL worker imports none of them. Historical configs and
receipts remain readable at their recorded revisions, not executable recipes.
The duplicate Qwen-only runner and fixed-20-task sequential launcher are also
retired. Read-only historical reconciliation now imports the maintained runner's
identical trace-normalization functions; all 34 reconciliation/runner tests pass
against it, with fixtures updated to the actual task-version and ingest response
contracts. The configurable PostgreSQL evaluation command replaces the old
sequential launch surface. Historical reconciliation still never runs a model.

Removed files are recoverable from base commit `bac2a4c`. Historical receipts
and their source pointers are unchanged and refer to their original revisions,
not current launch instructions. No accepted checkpoint, dataset, campaign
database or unrelated worktree was deleted.

Retired the typed-API RL split/request builders and preview checker, their two
CLI commands and tests that existed only to reproduce the old Qwen3.6 requests.
Caller checks found no maintained runtime dependency. Historical requests and
receipts remain untouched; current split, tool-identity, token-integrity and
Jobs API safeguards remain tested in their supported paths. The public CLI
explicitly rejects the removed commands instead of presenting them as RL support.

Caller review also retired three fixed Qwen3.6/B300 kernel investigation scripts
and their unused FLA launch override (380 lines). No maintained launcher, runtime,
test or guide referenced them. Current pinned-image GDN checks and real Qwen
training/recovery evidence remain; the old probes are recoverable from `bac2a4c`.
The read-only transcript export/normalization CLI is retained because it is the
documented source-data utility. Its HTTP and command paths now have full focused
statement/branch coverage: HTTPS-only, no authorization-bearing redirects,
bounded GET retries, private files/digests and non-disclosing errors. Temporary
AWS session tokens are included in exact normalization redaction. These tests
use synthetic HTTP/data only, not live transcript reads or training qualification.

Formatting now passes in CI. Twenty-eight files were mechanically formatted with
identical parsed Python syntax trees. Seven historical source files remain
format-excluded because execution plans bind their exact bytes; their integrity
tests are unchanged. The dev dependency set now includes the tested MCP 2.1.1
client so CI exercises the real transport boundary instead of silently skipping it.

The exact RL tool catalog (`85fad6bd…15b44a`) advertises a 300,000ms bash maximum.
The old 120-second client deadline could interrupt an allowed call. Both RL data
preparation and live catalog checks now reject that mismatch; the documented
example uses 330 seconds. Thirteen added tests cover the bound, both backends'
pre-network rejection, unchanged schemas and live cleanup before any sampling.
This is an independent configuration defect, not proof of the V6 episode's lost
root cause: its final call did not specify `timeoutMs`. Existing input bundles,
episodes and outcomes remain unchanged; a new preparation is required.
