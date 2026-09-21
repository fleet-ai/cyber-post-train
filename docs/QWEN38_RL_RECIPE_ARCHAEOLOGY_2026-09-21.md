# Qwen3.8-27B RL recipe archaeology — 2026-09-21

## Decision in one paragraph

Keep the SkyRL `prod9` canary as the primary Fleet-cyber RL path.  It is the
only candidate in this repository with an explicit plan for a native 262,144
token context *and* controlled compaction for very long tool-using episodes.
Do not replace it with Miles merely because Miles has more production history.
The maintained Miles/FTI recipe is a valuable, separately-qualified
compatibility lane: it has the strongest evidence that Qwen3.8-27B can collect
groups, compute rewards, update full weights, synchronize rollout servers, and
save checkpoints on Fleet.  But its 256K route ends an episode at
`context_full`; it does not summarize or discard old turns.  That is not an
acceptable substitute for the cyber program's long-horizon requirement.

Nothing below establishes a successful Fleet-cyber RL update.  A route is
eligible to scale only after it proves the exact task/version binding, finite
mixed rewards, verifier execution identifiers, a finite non-zero update, a
reloadable checkpoint, and release of every created task instance and GPU.

### 96K mechanics update

The separate
[`QWEN38_MILES96_MECHANICS_CANARY.md`](QWEN38_MILES96_MECHANICS_CANARY.md)
now specifies a one-node, maintained-Miles operational check. It is not a
scientific alternative to prod9: it proves only the shortest useful chain
(fresh V1 rows → varying real rewards → one update → HF export → one-GPU
reload) at the maintained 96K hard context limit. It has no compaction and
must never be described as a long-horizon cyber result.

## Evidence sources

This review used source and run records only; it did not inspect task prompts,
tool interactions, answers, credentials, or raw rewards.

| Source | Immutable reference | Relevant files |
|---|---|---|
| `fleet-ai/cyber-post-train` | audited head `8498507d2678769c1e59cc596f36c49aa4722129`; SkyRL prod9 inputs committed in `8e9ad6fcf031c862052acd4931fb0d73067cb17e` and the fresh runtime/terminal-acceptance closure in `7518b3bc4647bbe141fa1be5abad158e4ef7f348` | `configs/qualification/qwen38-rl-reward-canary-prod-v9.json`, `configs/qualification/qwen38-rl-reward-canary-data-prod-v9.json`, `training/skyrl_prod9_{training,rollout,hardening,direct}.py`, `docs/QWEN38_SKYRL_PROD9_NONSUBMITTING_GATES.md` |
| `fleet-ai/cyber-post-train` historical Miles plan | FTI 0.9.2 source `0b1af5684310ee244bf7bdb0028e5ef78c08098b`; immutable trainer image is recorded in the plan | `configs/runs/qwen38-27b-fti-v1-rl-reward-canary-v3.json`, `configs/runs/qwen38-27b-fti-v1-rl-production-a1.json`, `docs/QWEN38_FLEET_RL_PATHS_2026-09-13.md` |
| `fleet-ai/theseus` maintained FTI/Miles | recipe files last changed by `f2b0cb5db7c0a9dcc2210943f2ee1df31f9b50fc` | `services/fti/src/fti/trainers/miles/run_fleet.py`, `agent.py`, `README-miles-v1.md`, `README-miles-training.md`, `payloads/tool-use-qwen38-256k-v1.json`, `services/fti/CHANGELOG.md` |
| `fleet-ai/theseus` SkyRL image | image source `8d62868d6dc00eee793d83efe5738dc21e42758d`, which pins upstream SkyRL `f5bc3b78dfddfb352870d5d7430cd226e5785838`; the qualified digest is bound by the prod9 qualification file | `services/skyrl-train/Dockerfile`; upstream `skyrl/train/config/config.py` and `skyrl/backends/skyrl_train/distributed/fsdp_strategy.py` |
| `fleet-ai/dataminer_v2` Neeraj v003 | experiment files last changed by `6c3d7aa79075d4ed0a6ed476664de71249febba0` | `experiments/rl-transfer-v003/{protocol.md,launch.md,run_v003.py,proto-results.md,jobs/neeraj-v003-sy-2e6.yaml}` |

The Dataminer tasks and evaluation distribution are not Fleet-cyber tasks.  Its
evidence validates training mechanics and exposes failure modes; it is not
evidence that its task mixture, learning rate, or transfer result should be
copied into cyber.

The default trees of `fleet-ai/SkyRL-Fleet@3da8f633f1c75125e6ab359f38c827238807e6a5`
and `fleet-ai/skyrl-fleet-v2@fe1ad6c154f6c40bfcfb3c514f5afe3b1dbed5a4`
contain no exact Qwen3.8-27B runnable recipe.  They are therefore not evidence
for this model's topology or optimizer.  Prod9's relevant SkyRL authority is
the exact upstream commit and Fleet image source pinned above.

## Comparison

| Route | Framework and proven scope | Nodes / GPUs and context | Episode, reward, and update | Checkpoint evidence | Compaction | Status for Fleet cyber |
|---|---|---|---|---|---|---|
| **SkyRL prod9** | Repository SkyRL full-weight FSDP adapter over upstream `f5bc3b78`.  It has local qualification and an explicit cyber-tool/verifier binding plan, but not yet one accepted end-to-end cyber RL run. | 1 node / 8 B300 GPUs; policy and reference each declare 8 colocated GPUs, sequence parallel size 1; rollout is 2 engines × TP4; 262,144 context. | One prompt group × 8 samples, one update; train/policy mini-batch 1 group, per-GPU micro-batch 1.  The pinned upstream FSDP path constructs AdamW with LR `1e-6`, betas `.9/.999`, WD `.01`, gradient clip `1.0`, constant-with-zero-warmup schedule, and optimizer offload after each step.  Episode limit is 4 hours / 1,200 turns; each generation chunk is 4,096 tokens. | Save every update, retain two.  Current source requires a sealed step-1 checkpoint, BF16 export, independent one-GPU finite reload with zero optimizer steps, and observed release.  Repository evidence contains no accepted prod9 run artifact yet. | **Yes, by design:** at a 163,840-token trigger, a separate 8,192-token policy step summarizes the prior messages; the next prompt contains the original task plus that lossy summary. | **Primary canary.** It is the minimum-delta current design compatible with the required long, compacted trajectories; it is not yet a proven RL outcome. |
| **Maintained FTI/Miles V1 256K** | Fleet-supported Miles (SGLang rollout + Megatron training + GRPO), V1 Fleet task hook and stored-verifier grading.  Maintained source and separate Qwen runs establish the engine/trainer path; exact cyber loop remained unproved. | Exactly 4 nodes × 8 B300 GPUs.  Qwen recipe uses TP8 × CP4, 262,144 context, 245,760 response cap, and 65,536 train tokens/GPU.  The recipe exposes only `(4, 8)` for this model/context shape. | Payload uses 8 prompt groups × 8 samples at normal scale.  Current defaults are GRPO, Adam, LR `1e-6`, WD `0.1`, betas `.9/.98`; Qwen builds a zero-coefficient KL reference path. | Default normal save interval is 20.  A maintained four-node Qwen run wrote iteration 19 and the trainer later restored it after requeue; the next weight-sync check failed because rollout engines still held base weights.  That proves trainer restore mechanics, not an accepted resumed update or independent reload.  The historical cyber canary instead specified one 8-sample group, one update, and checkpoint interval 1, but never reached acceptance. | **No.** `agent.py` returns `context_full` when there is no room; a length-capped turn can continue, but earlier context is neither summarized nor removed. | **Parallel compatibility/qualification lane only.** Do not call it production cyber RL until it passes the cyber acceptance gate and either gains safe compaction or is restricted to an explicitly bounded-horizon study. |
| **Neeraj Dataminer v003 sync Miles** | Working full-parameter Miles/SGLang/Megatron implementation with custom agent/reward bridge.  Smoke completed rollout → reward → update → save → weight sync. | The successful single-node route was 8 GPUs, TP4 × CP2, 8,192 dynamic train tokens/GPU, and CPU optimizer offload.  The practical cap was 96K; a roughly 131K update OOMed.  Later sync sweep arms also packed TP4 × CP1 on 4 GPUs. | Wave-synchronous GRPO.  Production-style study: 8 prompts × G=4 = 32 episodes/wave, one optimizer epoch/wave, fixed sampling (`T=.7`, `top-p=.8`, `top-k=20`).  Reward bridge reads the agent's verified reward from sample metadata and drops malformed groups rather than inventing zero. | Checkpoint every 4 waves in the study; smoke saved and synchronized a full-weight checkpoint.  The source also loads from the same checkpoint root, but it does not provide an independent sealed reload receipt.  A cyber canary must save/reload after its first update. | **No implementation.** The practical 96K route used truncation-with-reward.  The protocol requires token-native IDs and a per-turn prefix assertion, but it does **not** explicitly prove compaction is impossible; a compacted objective would need a separate design and qualification. | **Mechanics reference, not a cyber recipe.** Reuse its template/reward/update safeguards, not its distribution or conclusion. |
| **Earlier custom cyber Miles** | Repository-specific adapter around Miles. | 96K generation envelope in the latest real canary. | A long interaction ended before grading with an undifferentiated `generation_incomplete`; no trusted reward/update. | No accepted RL checkpoint. | No safe long-horizon compaction. | **Retire as a launch path.** Keep only its incident tests and diagnosis history. |

## What can be reused immediately

These are portable safeguards, not permission to combine incompatible recipes.

1. **Chat/tool template must be qualified, not guessed.**  Dataminer found that
   `qwen3` did not match the tool parser and the checkpoint-native template did
   not work with TITO partial renders; `qwen35` matched the required tool
   instruction.  The maintained FTI path carries a fixed Qwen3.8 template.
   Require a one-episode tool/verifier qualification before any train run.
2. **Rewards must be real and group-useful.**  Preserve the verifier execution
   identifier; reject aborted/malformed samples; require reward variation in a
   group before treating an update as meaningful.  A live process with only
   all-zero or uniform groups is not useful RL progress.
3. **Use a single synchronized path first.**  Dataminer's disaggregated async
   path observed a weight-delivery corruption failure.  Start with synchronous
   rollout/update/weight-sync.  Make async a later, separately qualified
   throughput experiment.
4. **Make the first checkpoint a hard gate.**  For a one-update canary, save at
   update 1, then independently reload it.  Do not infer checkpoint health from
   a directory's existence.  The FTI historical Qwen resume path also had a
   weight-equality/resume defect, so resume needs its own qualification.
5. **Bound platform fan-out.**  Dataminer saw gateway failures when a full wave
   provisioned thousands of environments at once.  Use a small, explicit
   concurrent-environment cap; increase it only after evidence shows the
   platform, grader, and cleanup path remain healthy.
6. **Do not use high learning-rate results as a shortcut.**  In the Dataminer
   study, `1e-5` degenerated quickly, while lower rates were more stable.  For
   a first cyber canary, preserve `1e-6`; tune only after a valid reward/update
   loop exists.  The study's transfer metrics were poor because its reward
   distribution was weak, not because a larger LR was needed.

The minimum-delta choice is therefore to keep prod9's existing one-node FSDP
topology, upstream AdamW defaults, 262K context, and step-wise compaction
objective unchanged.  Reuse only the proven *guards* from Miles and Dataminer:
template qualification, authoritative reward metadata, a synchronous first
update, bounded environment fan-out, and independent checkpoint reload.  Do
not transplant their TP/CP layouts, optimizer constants, sampling policy,
truncation rule, task mixture, or conclusions into the SkyRL rail.

## Recommended next sequence for the active RL lane

1. **Finish the SkyRL prod9 non-submitting gates.**  In particular, prove the
   compacted transcript remains token-safe, capacity estimates match the full
   262K context, and the final receipt seals a reloadable checkpoint.  Preserve
   its exact cyber task/version binding and root failure-alert annotation at
   rendering time.
2. **Run exactly one SkyRL reward/update/reload canary.**  The target is not a
   capability result: it is one mixed-reward 8-sample group, one finite
   non-zero update, a saved/reloaded checkpoint, and confirmed cleanup.  Its
   `prod9` numbers are already deliberately minimal: one node, eight GPUs,
   one group, eight samples, one step, `1e-6`.
3. **Only after that succeeds, grow SkyRL in controlled steps.**  First increase
   useful task groups/updates while keeping the same compaction and exact
   verifier contract.  Evaluate each saved checkpoint on held-out Fleet tasks
   and WebExploitBench; do not use training reward alone as a success signal.
4. **In parallel, optionally qualify Miles V1 on one bounded episode.** The
   one-node 96K mechanics canary is the fastest current route for template,
   verifier, reward-variation, update, and reload mechanics. The maintained
   256K recipe remains a four-node compatibility route. Neither route may be
   scaled into long cyber trajectories until it has a scientifically sound
   segment/compaction objective.
5. **Do not relaunch the older custom Miles adapter.**  Its 96K failure cannot
   be repaired by repeating the same opaque path; it lacks the required
   trajectory and evidence guarantees.

## Exact reusable starting points

### SkyRL cyber canary (primary)

- `configs/qualification/qwen38-rl-reward-canary-prod-v9.json`
- `configs/qualification/qwen38-rl-reward-canary-data-prod-v9.json`
- `scripts/prepare_qwen38_skyrl_prod9_successor.py`
- `docs/QWEN38_SKYRL_PROD9_NONSUBMITTING_GATES.md`

The current configuration is intentionally a one-update qualification: one
node, 8 samples/prompt, one group, one optimizer step, LR `1e-6`, checkpoint
and evaluation interval 1.  Its limits are 262,144 context tokens, 4,096-token
generation chunks, 32,768 tokens per action, compaction at 163,840 tokens to an
8,192-token summary, up to 1,200 turns, 4,194,304 generated tokens in total,
and a 4-hour episode timeout.

The exact optimizer is inherited from the image's pinned upstream SkyRL source,
not invented by the prod9 compiler: FSDP AdamW, betas `.9/.999`, weight decay
`.01`, gradient clipping `1.0`, constant schedule with zero warmup, and optimizer
offload after each full step.  The compiler overrides the LR to the same
`1e-6` default and binds one update epoch, one prompt-group train/mini-batch,
and per-GPU micro-batch 1.

### Miles V1 256K compatibility qualification (secondary)

- `fleet-ai/theseus@f2b0cb5db7c0a9dcc2210943f2ee1df31f9b50fc:services/fti/payloads/tool-use-qwen38-256k-v1.json`
- `fleet-ai/theseus@f2b0cb5db7c0a9dcc2210943f2ee1df31f9b50fc:services/fti/src/fti/trainers/miles/run_fleet.py`
- `fleet-ai/theseus@f2b0cb5db7c0a9dcc2210943f2ee1df31f9b50fc:services/fti/deploy/validate_miles_v1.py`
- historical cyber bindings: `configs/runs/qwen38-27b-fti-v1-rl-reward-canary-v3.json` and `configs/runs/qwen38-27b-fti-v1-rl-production-a1.json`

The maintained payload is four workers × eight GPUs and invokes
`qwen3.8-27b-256k`, platform `v1`, normal mode, 8 rollout prompt groups, and
8 samples/prompt.  Before any submit, render and inspect the root Kubernetes
object: it must carry `metadata.annotations["fleet.ai/failure-alerts"] ==
"off"`.  The shared maintained payload predates this project policy and is a
reference, not an authority to omit the annotation.

### Neeraj v003 implementation lessons (mechanics only)

- `fleet-ai/dataminer_v2@6c3d7aa79075d4ed0a6ed476664de71249febba0:experiments/rl-transfer-v003/run_v003.py`
- `fleet-ai/dataminer_v2@6c3d7aa79075d4ed0a6ed476664de71249febba0:experiments/rl-transfer-v003/{protocol.md,launch.md,proto-results.md}`
- `fleet-ai/dataminer_v2@6c3d7aa79075d4ed0a6ed476664de71249febba0:experiments/rl-transfer-v003/jobs/neeraj-v003-sy-2e6.yaml`

Use these for the custom reward bridge, exact template qualification, gradient
and checkpoint checks, concurrency control, synchronous weight sync, and
full-parameter optimizer layout.  Do not copy the synthetic task mix,
truncation policy, or reported transfer result into Fleet cyber.
