# Repository consolidation

This is the implementation checklist for the September 2026 cleanup, not a run
authorization or a statement that every backend is qualified.

## Target

One documented interface for preparing data, previewing/submitting SFT and RL,
running evaluations, and reading status. Model, task set, recipe, resources and
tracking are explicit inputs. Reuse SkyRL/Miles and the existing evaluator;
do not implement a new trainer or scheduler. Preserve exact inputs, split
independence, create-once submission, checkpoints, and truthful failure status.

## Sources being consolidated

| Source | Useful work | Boundary |
|---|---|---|
| main `bac2a4c` | PostgreSQL rollout coordination, eval adapters, cluster safeguards | Active campaign must not be changed by cleanup |
| `2f290e4` (`codex/q38-teacher-devicefix-v5`) | Successful dense teacher SFT, task-held-out loss, W&B, checkpointing, CPU-tested GLM LoRA prerequisites | Qwen full run proven; GLM distributed training not yet proven |
| `25ad2a3` (`codex/qwen38-glm53-training-prep-v1`) | Teacher canary/export/reload evidence, historical training lessons | Historical runs must not be replayed |
| `codex/rollout-postgres-ledger` working tree | Active GLM rollout worker fixes | Copy/review changes; do not edit that working tree or its live deployment |
| `codex/webexploitbench-opencode-cluster-20260909` working tree | Image-bound external evaluation preparation | Its campaign is explicitly paused; do not restart it |

The companion tasks `fleet task evals`, `external benchmarks`, and
`cyber repo prep and trace analysis` were inspected. Historical artifacts remain
recoverable in their original commits/worktrees. No blanket worktree deletion.

## Completion checklist

- [x] Fix reproducible data-split leakage, with synthetic regression tests.
- [x] Consolidate the proven SFT runtime and dense-data preparation.
- [ ] Provide usable configurable training/eval/status commands with one submission boundary.
- [ ] Integrate Miles and SkyRL without pretending GLM-5.3-Flash is GLM-5.3.
- [ ] Consolidate active evaluator improvements without modifying live campaigns.
- [ ] Delete superseded launchers, duplicate abstractions and misleading docs after caller checks.
- [ ] Test installation from a built wheel, CLI help/examples, data integrity,
      concurrency, API errors, checkpoints, telemetry and resource-release paths.
- [ ] Validate concise skills and docs against actual commands.
- [ ] Qualify the final code with bounded real training and evaluation runs;
      preserve exact run/image/data/code identities and release evidence.

## Verification log

- Clean worktree starts at main `bac2a4c`; no existing experiment files changed.
- `uv sync --locked --extra dev` followed by pytest failed collection because
  the documented setup omitted Torch and PyArrow. The full local test environment
  currently requires `--extra train` as well. This is a documentation/dependency
  defect, not a passing test suite.
- The generic splitter included versioned `lineage_key` in its hash input.
  Different versions of one task family could cross train/dev/test. Frozen
  historical corpora will not be silently rewritten; newly generated corpora
  must carry a new split-policy identity.
- Baseline after installing the training extra: 599 tests passed, 3 PostgreSQL
  tests skipped. The disposable local PostgreSQL suite was then run successfully.
- New split module has 100% line/branch coverage; generic Jobs boundary has 96%
  coverage in focused tests. These are not claims of whole-repository coverage.
- Consolidated the successful SFT runtime and added configurable epochs, batches,
  manifests and held-out task counts. Local runtime/config tests: 75 passed,
  15 native-image-only tests skipped. Native-image and real-model qualification
  of the final cleanup code is still required.
- Imported the active worker's lease-lifetime and exact-instance TTL/data-binding
  improvements without changing its checkout or deployment. PostgreSQL/worker/
  heartbeat/self-hosted regression set: 35 passed.
- Removed the unused `spec.py`/catalog facade and replaced it with direct prepared
  requests, CPU preflight, preview, submission and status commands. Public CLI
  plus compiler regression set: 47 passed.
- Full local suite after consolidation: 775 passed, 15 native-image-only tests
  skipped, seven subtests passed; PostgreSQL integration tests ran against a
  disposable local database. Ruff passed. Native-image skips are not verification.
- The native-image CPU self-test then passed all 63 tests with zero skips,
  failures or restarts. Its digest-verified receipt and exact image/Pod binding
  are in `docs/evidence/cleanup-native-sft-20260911.json`. The original test
  launcher failed because pytest was absent; the successor added hash-pinned
  test-only wheels outside the trainer installation. Neither used a GPU.
- Built a wheel and installed it into a fresh environment outside the checkout;
  CLI help and training/evaluator imports succeeded without the source tree.
- Removed the last hardcoded 20-task assumption from validation aggregation;
  regression tests cover 1, 2, 20 and 21 tasks. Local suite then passed 815 tests.
- Consolidated dense target preparation from the v3/v4 worktree into `dense.py`
  and `corpus.py`, without the run-specific split migration or fixed teacher.
  Added explicit source-model filters and immutable dev-reference selection.
  Real Parquet/manifest and synthetic segmentation/selection tests pass; the
  new public data path still needs pinned-image qualification.
- Full local suite after dense-data consolidation: 839 passed, 15 native-only
  skips, seven subtests passed. The skipped native cases are covered by the
  separate pinned-image receipt above. Ruff and `git diff --check` passed.
- The live generic Jobs API preview accepted the new request shape with `c1`,
  one node/eight GPUs, the exact training image and the `wandb-api` Secret
  reference. No GPU run was created by preview. Missing Secret references
  produced a warning, which the client correctly treated as a blocked preview.
- Committed the first consolidation as `5d20d02`. The exact-image CPU builder
  then processed one complete teacher demonstration into eight training windows
  with 7,770 supervised tokens and two fixed held-out task windows. Its data,
  tokenizer and native-runtime preflight passed without GPU allocation;
  `docs/evidence/cleanup-qwen-cpu-20260911.json` binds the verified receipt.
- The new CLI submitted `chris-cpt-q38-qual-v1-7eab6d07` exactly once through a
  durable SFS journal. RayJob UID `b6e0d69b-3e23-47fe-939d-1dedfea85d67`, Workload
  UID `2f4ab61c-5b24-450d-b052-12bc9ce24312`, effective priority 10000. At the
  2026-09-11 00:45 UTC observation it remained queued, holding zero GPUs. This
  is a one-step operational qualification, not a new capability experiment.
- Added transactional initialization for a fresh PostgreSQL campaign, concurrent
  initialization tests, and explicit OpenCode terminal-event classification.
  None of these changes were deployed into the active rollout campaign.
- Added generic `eval prepare/preflight/init/run/status` commands. Synthetic
  tests cover arbitrary task counts, configurable pass@k, complete-task route
  partitions, manifest/CSV/runtime drift, bounded workers and no duplicate worker
  identity. Model sampling is enforced by the credential-holding proxy and tested
  through real loopback HTTP. Real evaluation qualification is still pending.
- Removed the unused hardcoded Job-pair generator/runtime, GLM5.2-specific plan
  and command wrapper, their tests and two generated Qwen3.6 Job manifests.
  Their retained callers were removed or routed to the actual configurable API
  path. No historical checkpoints, datasets or accepted evidence were deleted.
- The Qwen qualification was admitted at 2026-09-11 00:51:37 UTC. RayCluster UID
  `76ab1133-e9c9-43a6-8723-05a27a9165a7`, Pod UID
  `ffd5ea72-d88f-406e-a5fc-ea29c6910868`, exact pinned image, zero restarts and
  eight GPUs. Initial held-out validation completed; checkpoint writing was
  observed. Neither fact alone is terminal acceptance. W&B run `cpt-q38-qual-v1`
  is in `thefleet/cyber-post-train`.
- Local suite after the generic evaluation boundary: 901 passed, 15 native-only
  skips, seven subtests. Ruff and wheel build/install passed. Execution-host
  image bytes, architecture, OpenCode release label and actual executable version
  are checked before claims. Synthetic loopback tests prove sampling/output
  limits override agent-provided values without exposing credentials.
- Real evaluation preflight passed for one train-only task outside the active
  100-task campaign, on both exact shared model revisions. It created no task
  environment or scored session. The two planned operational attempts remain
  separate from the campaign and are ineligible for training.

## Removed surfaces

The unused component-spec/catalog layer, its tests, the future-interface design
document and redundant experiment-composition skill were removed. Stable controls
now live in `AGENTS.md`, the scientific protocol and the training/eval skills.

The XBEN/CVE-Bench fallback adapter and its dedicated tests were removed after
checking that no retained module imports them. The supported benchmark scope is
Fleet, WebExploitBench and ExploitGym. Historical results are not deleted; all
removed implementation files are recoverable from base commit `bac2a4c`.

The old `training/NEBIUS_ACCESS.md` prescribed retired GLM-5.2 naming, priorities
and direct Kubernetes launch rules. It was removed in favor of the maintained
cluster guide. `training/README.md` now points at the actual commands and labels
historical recipes explicitly, instead of presenting Qwen3.6 as the global default.
