# Repository consolidation

Status: 2026-09-11. This is a qualification index, not live state or launch
authority. Qwen is first; **full GLM5.3 and both RL backends remain required**.

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

- **Miles V7:** one create-once Jobs API submission, effective priority 10,000.
  Last observed queued without GPU allocation. Monitor its exact UIDs in
  [submission evidence](evidence/cleanup-miles-rl-v7-20260911.json); do not repeat
  the POST or infer a stuck job from an observation timeout.
- **SkyRL V5:** CPU-prepared only; no GPU submission. It shares the episode
  lifecycle still being qualified with Miles.
- **Evaluation V7:** terminal context-overflow error after 209 completed model
  steps. Exact grading/catalog identity, one private local result and full cleanup
  are verified; accepted zero, review one, active zero. No retry or training reuse.
  The new v2 compaction treatment reserves a full response's growth before the
  native post-response check; exact-binary synthetic tests prove compaction and
  continuation. It does not modify V7 or the historical campaign.
- **Full GLM:** the two-node loading attempt was released without an optimizer
  step. Keep GPUs released while the CPU loading repair is qualified.
  [Release evidence](evidence/cleanup-glm-startup-release-20260911.json).
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
- Exact task GET rejects missing starting-data bindings before instance creation.
  New errors retain safe classes/digests, not private server bodies.
- OpenCode checks its actual UID/HOME and offline startup, not just its version.
  Final output-limit stops remain held, even with complete catalog ingestion.
- RL checks tool deadlines against the real catalog, preserves nested error
  classes, awaits sibling/environment cleanup and uses bounded checkpoint drain.
- Miles cursor saves/reloads reject silent resets and counter drift. All 82
  cursor tests, including 41 using native Miles, pass in the exact image;
  [evidence](evidence/cleanup-miles-cursor-20260911.json). This is not yet complete RL recovery.
- PostgreSQL claims, ownership fencing, transaction rollback, private-result
  idempotency and no-auto-retry have real disposable-database integration tests.
  One shared result normalizer preserves SQLite/PostgreSQL evidence digests.
  This is not a deployment into the historical campaign.

The [ledger regression evidence](evidence/cleanup-ledger-contract-20260911.json)
records 46 real-PostgreSQL cases, complete focused module coverage and exact
cross-backend result-digest parity. The latest full suite passes 1,776 tests plus seven
subtests, with 156 explicit skips. Whole-repository coverage is **not complete**.
Optional native dependencies need
the pinned trainer images; local skips are never counted as passing training
qualification. Installed-wheel checks run outside the checkout. The active
evaluation deliberately stays on its frozen installed wheel while source evolves.

## Remaining completion gates

- [x] Qwen SFT, held-out loss/W&B, checkpoint seal/export, native reload and resumed optimization.
- [x] Separate Qwen BF16 GPU reload and finite synthetic generation.
- [ ] Full GLM5.3 distributed SFT and artifact/recovery qualification.
- [ ] Miles **and** SkyRL on both required models: real reward acquisition, optimizer update and checkpoint recovery.
- [ ] Complete model outcomes through the configurable evaluation path.
- [ ] Final caller-checked retirement, meaningful coverage, current-main review and wheel/CLI qualification.

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
