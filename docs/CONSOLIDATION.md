# Repository consolidation

Status: 2026-09-11. This is a qualification index, not live state or launch
authority. Qwen is first; **full GLM5.3 and both RL backends remain required**.

## Paused-goal handoff

Chris paused the cleanup goal on September 11 and requested publication of the
completed work plus an explicit WIP record. This snapshot is **not** a claim that
the original goal or all supported model/backend combinations are complete.

- **Usable, with real operational evidence:** the configurable Qwen SFT path,
  held-out loss, W&B, native checkpoints, resumed optimization and BF16 export
  reload. The small qualification runs do not establish a capability improvement.
- **Implemented but WIP:** full GLM5.3 SFT, both native RL backends, and an accepted
  complete outcome through the new configurable evaluation path. The table below
  identifies the evidence and missing gate for each.
- **Local validation at handoff:** 1,823 tests and seven subtests passed with the
  disposable PostgreSQL integration database enabled; 166 tests explicitly
  skipped. The public CLI/Jobs client has complete statement/branch coverage.
  Across production Python files, this local run covers 71.9% of combined
  statements and branches, excluding test files. Native-image evidence is separate;
  whole-repository coverage is not complete.
- **Installation:** the isolated non-editable package check passed for 105 Python
  modules and all 22 command-help paths, without checkout imports or credentials.
  [CI](../.github/workflows/tests.yml) now runs the repeatable
  [installation check](../tests/check_installed.py).
- **Still running at 2026-09-11T16:35:17Z:** local evaluation V8, Docker container
  `agent-runtime-3f73210f`, one claimed session, zero accepted/local results and
  no stale lease. It had completed 212 model steps and was progressing. Its exact
  identities and bounded, no-retry treatment are in the
  [launch receipt](evidence/cleanup-eval-v8-20260911.json). No new cluster nodes
  were allocated for it. Pausing this goal did not cancel that existing process.
- **Preserved outside Git:** private data, results, ignored qualification tools
  and runtime outputs remain in the existing worktrees/stores. Do not delete them
  or replay their attempts when starting from a fresh clone.

No new cluster submissions, successor runs or monitors are part of this handoff.
Before resuming live qualification, resolve the alert-safe development/reporting
path with the cluster maintainers and obtain current authorization; publication
does not lift the submission hold. Then finish Qwen evaluation/RL qualification,
full GLM qualification, and the remaining simplification/coverage gates below.

**Submission gate — 2026-09-11 user alert review:** Chris explicitly reauthorized
submissions, conditional on checking that the intended path works first, then
reported another alert. Miles V7 was already running and failed at 15:46:54Z;
no new run was submitted during this investigation. The
[incident review](incidents/2026-09-11-cleanup-rl-alerts.md) now records eight
failed RL runs. Hold the next RL GPU submission until the generation-stop
handling is resolved and tested through the real recorder/batch boundary.
Do not simply submit another run to obtain the missing diagnostic. Existing
useful work remains untouched. This gate does not authorize alert suppression,
false success, cancellation of healthy work, or replay of preserved outcomes.
The already-running GLM CPU staging Job subsequently failed at 16:10:07Z;
its container exited and its Workload is gone. The final CPU sample reported
zero cgroup OOM events, but the specific application failure is not yet recovered.
Following Chris's renewed alert complaint, **hold all new cluster submissions**
while resolving the development/reporting path. Continue local code and tests;
do not count CPU-only Jobs or priority labels as notification exemptions.

## Supported interface

One public CLI and plain YAML configure data, model, hyperparameters, W&B,
checkpointing, held-out validation and cluster resources. SkyRL owns SFT;
Miles or SkyRL owns RL; Fleet owns isolated tasks and authoritative grading.
Start with [README](../README.md), [training](TRAINING.md) or
[Fleet evaluation](../evals/fleet/README.md). There is no new trainer or scheduler.

## Qualification: evidence, not capability claims

| Path | Strongest completed check | Still unqualified |
| --- | --- | --- |
| Qwen SFT | Real optimizer update, held-out loss, W&B and eight-rank native checkpoint; [run](evidence/cleanup-qwen-sft-20260911.json) | Hyperparameter quality and held-out capability improvement |
| Qwen recovery | Eight-rank zero-step restore; source unchanged; then a planned pause followed by exactly one new update to step two; [reload](evidence/cleanup-qwen-recovery-20260911.json), [resume](evidence/cleanup-qwen-resume-result-20260911.json), [checkpoint audit](evidence/cleanup-qwen-resume-checkpoint-audit-20260911.json) | Bit-identical later-epoch reshuffling is not promised |
| Qwen inference export | CPU rehashed 29 files/1,199 BF16 tensors and nine sidecars; a separate GPU loaded the complete model, ran finite synthetic forward/generation and released; [export](evidence/cleanup-qwen-export-20260911.json), [GPU check](evidence/cleanup-qwen-export-gpu-20260911.json) | Production serving engine and matched base/post evaluation |
| Full GLM5.3 | Exact 141-shard/755.6GB model pinned; full CPU/meta preflight; tiny native LoRA update/resume tests; [preflight](evidence/cleanup-glm-preparation-20260911.json), [native tests](evidence/cleanup-glm-native-20260911.json) | Full-size distributed load, optimization, export and recovery |
| GLM loader repair | All 141 headers audited; partial blocks match independent arithmetic. Sixteen threads improve the measured 64-expert load/merge by 3.3× over four; a complete 256-expert gate/up group also passes. Fifty native GLM cases cover 320/334 statements and 101/118 branches; [block proof](evidence/cleanup-glm-partial-blocks-20260911.json), [throughput/guard proof](evidence/cleanup-glm-loader-throughput-20260911.json) | Full-model loading/merging/distribution; no GPU resubmission |
| Miles RL | Real native base conversion and CPU seal completed, GPUs released; exact-image parser, TITO, MCP, batching and preparation tests pass; [conversion](evidence/cleanup-miles-conversion-result-20260911.json), [native tests](evidence/cleanup-runtime-regression-20260911.json) | Authoritative live reward, real RL update and recoverable RL checkpoint |
| SkyRL RL | Exact-image native parser, data, trajectory/batch, MCP and training-dispatch tests; real-user CPU preparation passes; [integration](evidence/cleanup-skyrl-training-native-20260911.json), [preparation](evidence/cleanup-skyrl-preparation-v5-20260911.json) | Real GPU reward/update and checkpoint recovery |
| Fleet evaluation | Real Qwen execution, grading, catalog ingestion, private local result and complete cleanup; [V5](evidence/cleanup-eval-terminal-20260911.json), [V6](evidence/cleanup-eval-v6-20260911.json), [V7](evidence/cleanup-eval-v7-terminal-20260911.json) | V5/V6 ended at output limits; V7 at context overflow. All remain held. Lifecycle success is **not** an accepted model outcome |
| Checkpoint completion guards | 630 exact-image CPU tests including partial writes, later-epoch cursor checks and final validation/save receipt binding; [proof](evidence/cleanup-sft-checkpoint-guard-20260911.json) | Does not create new GPU or capability evidence |

These small runs qualify mechanisms, not a sufficient experiment or production
readiness. All detailed IDs, digests, test counts and earlier failures remain in
[evidence/](evidence/); historical receipts are not rewritten.

### Current execution boundary

- **Miles V7:** failed at 15:46:54Z during the initial dev-baseline episode with
  `generation_incomplete`, before grading or optimization. The old diagnostic
  does not distinguish length, context exhaustion or abort. No accepted reward,
  optimizer update or checkpoint. Its Pod/RayCluster and all eight GPUs are
  released; Workload admission/quota are cleared. The exact Fleet environment
  is independently confirmed stopped. Preserve the
  [terminal evidence](evidence/cleanup-miles-rl-v7-terminal-20260911.json); never
  repeat its POST or infer a specific generation-stop cause from the generic code.
- **SkyRL V5:** CPU-prepared only; no GPU submission. It shares the episode
  lifecycle still being qualified with Miles.
- **Evaluation V7:** terminal context-overflow error after 209 completed model
  steps. Exact grading/catalog identity, one private local result and full cleanup
  are verified; accepted zero, review one, active zero. No retry or training reuse.
  The new v2 compaction treatment reserves a full response's growth before the
  native post-response check; exact-binary synthetic tests prove compaction and
  continuation. It does not modify V7 or the historical campaign.
- **Evaluation V8:** one new unused task under v2, from the installed public CLI;
  [launch evidence](evidence/cleanup-eval-v8-20260911.json). Local Docker only,
  pass@1/concurrency one, no retry or training reuse. No accepted outcome yet.
- **Full GLM:** the two-node loading attempt was released without an optimizer
  step. Keep GPUs released while the CPU loading repair is qualified.
  [Release evidence](evidence/cleanup-glm-startup-release-20260911.json).
  Native fused-format save/reload preserves every synthetic tensor, including
  strict FP32 buffers, without adapters or requantization; all 51 native tests
  pass. [CPU staging research](evidence/cleanup-glm-native-staging-probe-20260911.json)
  records full-model memory and the native ignored-input layer. One
  [full-size CPU/RAM staging pilot](evidence/cleanup-glm-native-stage-pilot-20260911.json)
  was admitted at 15:31:41Z after the complete synthetic transaction passed.
  It failed at 16:10:07Z before a model-loaded/staged receipt was observed;
  its CPU/RAM allocation is released and it never reserved GPUs. No production
  artifact is accepted. Full-size materialization, training integration and GPU
  qualification remain unproven; no staging retry or GLM GPU successor.
- No new automations, peer mutations or changes to shared/dedicated serving.

The queued/frozen training bundles predate the newest checkpoint-drain and cursor guards.
They remain immutable; a future preparation must bind current tested source.

## Resource and scientific rules

Chris replaced the four-node limit with **at most eight actively allocated
experiment-owned nodes**. Count admitted startup and dedicated serving. CPU-only
exemption is unconfirmed, so count those nodes too. Queued unallocated work
doesn't count, but do not queue an unguarded wave that could admit above eight.
Use supported admission or bounded waves; never bypass admission or touch peers.

Authorized high-priority GPU batch work uses the Jobs API's `c1`, verified as
effective 10,000/derived `q1`, with no explicit queue-priority override or
automatic requeue. Release broken/stalled owned capacity before fixing off-node.
Never fake success or suppress real alerts. Full controls:
[AGENTS](../AGENTS.md), [cluster alerts](CLUSTER_ALERTS_AND_INFERENCE_SERVING.md),
[scientific protocol](SCIENTIFIC_PROTOCOL.md), [PostgreSQL](ROLLOUT_POSTGRES.md).

## Durable corrections and tests

- Family-level train/dev isolation, dense assistant-target coverage and fixed
  held-out validation; external benchmarks never enter training or tuning.
- Create-once submission journals, immutable model/data/runtime identities and
  source-preserving checkpoint seal/export/reload.
- Rendered head and worker environments reject duplicate variable names and
  missing, optional, renamed or duplicated required Secret imports before a run
  POST. These local regressions strengthen preview; they do not qualify GPU work.
- Exact task GET rejects missing starting-data bindings before instance creation.
  New errors retain safe classes/digests, not private server bodies.
- OpenCode checks its actual UID/HOME and offline startup, not just its version.
  Final output-limit stops remain held, even with complete catalog ingestion.
- RL checks tool deadlines against the real catalog, preserves nested error
  classes, awaits sibling/environment cleanup and uses bounded checkpoint drain.
  Grouped errors are retained at both the episode and native trainer boundaries;
  the [alert incident](incidents/2026-09-11-cleanup-rl-alerts.md) records the local
  regression evidence and the current conditional submission gate.
- Miles cursor saves/reloads reject silent resets and counter drift. All 82
  cursor tests, including 41 using native Miles, pass in the exact image;
  [evidence](evidence/cleanup-miles-cursor-20260911.json). This is not yet complete RL recovery.
- PostgreSQL claims, ownership fencing, transaction rollback, private-result
  idempotency and no-auto-retry have real disposable-database integration tests.
  One shared result normalizer preserves SQLite/PostgreSQL evidence digests.
  This is not a deployment into the historical campaign.

The [ledger regression evidence](evidence/cleanup-ledger-contract-20260911.json)
records 46 real-PostgreSQL cases, complete focused module coverage and exact
cross-backend result-digest parity. That integration-enabled suite passed 1,780
tests plus seven subtests, with 157 explicit skips. The subsequent local-only
stop-contract revision passed 1,790 tests plus seven subtests, with 189 skips
(optional services/native runtimes were not enabled). The latest preview-boundary
revision passes 1,800 tests plus seven subtests, with the same 189 skips.
The public CLI and Jobs client have complete statement/branch coverage in the
338-test focused suite; that excludes the training and evaluation engines.
Whole-repository coverage is **not complete**.
Optional native dependencies need
the pinned trainer images; local skips are never counted as passing training
qualification. Installed-wheel checks run outside the checkout in CI, with a
fresh home and credential-free environment for command help. The active
evaluation deliberately stays on its frozen installed wheel while source evolves.

## Remaining completion gates

- [x] Qwen SFT, held-out loss/W&B, checkpoint seal/export, native reload and resumed optimization.
- [x] Separate Qwen BF16 GPU reload and finite synthetic generation.
- [ ] Full GLM5.3 distributed SFT and artifact/recovery qualification.
- [ ] Miles **and** SkyRL on both required models: real reward acquisition, optimizer update and checkpoint recovery.
- [ ] Complete model outcomes through the configurable evaluation path.
- [x] Repeatable installed-wheel/runtime-resource and CLI-help qualification.
- [ ] Remaining caller-checked simplification and meaningful whole-repository coverage.

## Consolidated sources and retained history

| Source | Contribution |
| --- | --- |
| Main `bac2a4c` | PostgreSQL coordination, evaluators and cluster controls |
| Teacher-devicefix `2f290e4` | Dense SFT, held-out loss, W&B, checkpoints and GLM prerequisites |
| Training-prep `25ad2a3` | Canary/export/reload evidence and failure lessons |
| Rollout-postgres-ledger worktree | Lease lifetime and exact instance TTL/data binding |
| WebExploitBench/OpenCode worktree | Exact-image external evaluator preparation |

Companion tasks and worktrees were inspected and remain intact. The historical
100-task campaign keeps its frozen worker; external benchmark work stays paused.
Neither is a template to replay.

Retired unused facades, old typed-API request builders, fixed-window data
preparation, duplicate sequential runners, raw-Kubernetes submission paths,
completed-run launchers and obsolete kernel probes. Caller checks preserved the
maintained public CLI, paired evaluators, integrity tests and read-only historical
reconciliation. Removed source is recoverable at `bac2a4c`; no accepted checkpoint,
dataset, campaign database or unrelated worktree was deleted.

The September 11 caller check also retired four fixed Qwen3.6 ExploitGym submit
wrappers: paired study, acceptance collector, failure observer and control-image
publication. No matching Job/RayJob remained at the read-only cluster check.
Their frozen bootstrap bytes, historical manifests, protocols and functional
receipt/UID validation tests remain intact. Only tests of the deleted wrappers
and stale launch instructions were removed; no cluster resource was changed.
