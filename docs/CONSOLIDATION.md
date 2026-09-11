# Repository consolidation

Status as of 2026-09-11. This checklist is not launch authority. Historical
campaigns and other worktrees remain untouched.

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
| CPU checkpoint export | 33 pinned-image tests; real export succeeded 02:01:04 UTC; independent audit rehashed 29 files, checked 1,199 BF16 tensors, nine sidecars and tokenizer/config reload; [receipt](evidence/cleanup-qwen-export-20260911.json). **GPU/optimizer reload remains unverified** |
| Public model pinning | Live exact-revision reads reproduced Qwen's existing 18-shard weight/sidecar hashes and pinned full GLM5.3's 141 shards/755,632,050,320 bytes in `configs/models/glm53-30333038`. No weights downloaded or GPUs requested |
| GLM LoRA CPU prerequisites | 68 pinned-image tests, zero failures/skips/restarts/GPUs; actual tiny GLM FP8 load, adapter updates, exact CPU resume and native worker integration; [receipt](evidence/cleanup-glm-native-20260911.json). **Not full-size or distributed GPU qualification** |
| Full GLM data/model preflight | All 141 staged weight shards verified, exact 744B model constructed on meta, 8 train windows/6,287 targets and two fixed dev tasks; 34 native checkpoint/seal tests also passed. [Receipt](evidence/cleanup-glm-preparation-20260911.json). One two-node/16-GPU one-step run was submitted once and is queued without allocation |
| Native recovery CPU tests | 41 tests in the pinned image, zero skips/failures/restarts/GPUs; exact model/Adam state through the resumed epoch, complete later-epoch coverage, strict state checks and no-optimizer validation. [Receipt](evidence/cleanup-recovery-cpu-20260911.json). Native later-epoch reshuffles are explicitly not bit-identical; full-size GPU reload remains pending |
| Local regression suite | 1,085 tests plus seven subtests pass with disposable PostgreSQL; 32 native-only cases explicitly skip locally. Ruff/whitespace checks pass |
| Installation | Built wheel installed outside the checkout; public commands and recovery/checkpoint/GLM imports work; retired modules are absent |

Split, checkpoint, export, model-pinning and historical RL-preview checks have
100% focused line/branch coverage; this is **not whole-repository coverage**.
Model-pinning tests exercise real HTTP parsing, pagination, tampering, download
bounds, errors and create-once output. The dependency lock now resolves 75
packages, 19 fewer after removing unused desktop `datasets`/`trl`.

The new evaluation wrapper has not completed a real rollout. Its first check
exposed a claim-directory defect before instance creation; that is fixed and
regression-tested. A distinct successor then hit HTTP 404 at Fleet's exact
instance-create API despite successful exact task lookup. Neither generated
tokens, scored or produced a local result. All held rows are preserved in the
separate qualification databases. Server-side diagnosis is requested; HTTP
status alone does not prove the cause. Future errors retain an allowlisted
category and response digest, never raw server text or an automatic POST retry.

## Remaining completion gates

- [x] Fix task-family leakage across versions and consolidate dense preparation.
- [x] Establish one generic Jobs API submission boundary with durable journals.
- [x] Qualify Qwen SFT, CPU sealing/export and installed public commands.
- [ ] Qualify GPU checkpoint/optimizer reload and exact resume.
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

Removed files are recoverable from base commit `bac2a4c`. Historical receipts
and their source pointers are unchanged and refer to their original revisions,
not current launch instructions. No accepted checkpoint, dataset, campaign
database or unrelated worktree was deleted.
