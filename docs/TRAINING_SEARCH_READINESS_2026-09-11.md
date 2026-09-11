# Qwen SFT and hyperparameter-search readiness

Read-only audit, 2026-09-11, against cleanup/main commit
`2b775b827afc87daf01170fd031c5834337aa9e5`. Historical runs below are explicitly
separate from the new configurable CLI. No training, evaluation, cluster helper,
or W&B run was launched for this audit. Training losses are not cyber solve rates.

## Bottom line

The repository has useful, operationally qualified **individual Qwen SFT run
components**, not a dependable autonomous hyperparameter-search service. It can
prepare different learning rates, global batches, epochs, windowing treatments,
and seeds; log fixed held-out loss to W&B; preserve the best checkpoint; and
resume exact interrupted work. It has no sweep planner, experiment-wide admission
lock, cross-run selection table, or automatic aggregate resource-cap enforcement.
W&B grouping is implemented; W&B Sweeps is not.

Focused local audit tests: **358 passed, 36 explicitly skipped** (SFT compiler,
runtime, corpus/windowing, splits, recovery and Jobs client). Skips require the
pinned native image; they are not GPU qualification. The new worktree's base-only
environment initially lacked Torch, so that collection attempt was discarded;
the reported run used the existing cleanup environment with this worktree's
source explicitly selected. No native model or cluster workload was started.

There is also substantially stronger historical learning evidence than the old
318-step run: the later dense-teacher V5 completed 186 updates and its fixed
held-out teacher-response loss fell by about **48%**. Its cyber task-solving
improvement has **not** been established. Exporting/reloading and evaluating that
checkpoint is a higher-information next step than assuming it did not learn and
immediately scheduling many more epochs.

## What is configurable, and what is not

| Area | Current implementation | Important boundary |
| --- | --- | --- |
| Model | Exact lock, weight inventory, staged root; Qwen3.8/Qwen3.6 qualified loader profiles | Public SFT compiler is SkyRL; arbitrary model identity is not arbitrary loader compatibility. Full GLM remains WIP. |
| Data source | One immutable normalized-record file plus explicit source-model allowlist | Teacher versus self uses the same builder. There is no automatic teacher mixture weighting, per-family balancing, or per-task success cap. |
| Windowing | `max_length`, recent `context_tokens`, number of frozen `dev_windows` | Every fitting assistant target once/epoch; copied context/tool observations masked. Changing the cap can change eligible targets and dev windows, not only memory. |
| Optimization | `lr`, integer `epochs`, `batch_size`, `microbatch_per_gpu`, `seed`, GPU/node shape | Steps are `ceil(train_rows/global_batch) × epochs`; rows are variable-length segments, not tasks. No arbitrary max-step tuning override. |
| Other optimizer settings | Native AdamW, betas 0.9/0.999, weight decay 0.01, max gradient norm 1.0 | Inherited exact native defaults, not exposed YAML search knobs. Wrapper fixes constant LR schedule and zero warmup. Unknown YAML fields fail rather than silently overriding them. |
| Validation | Before training, periodically, and final; token-weighted and task-macro loss | Periodic checkpoint and validation intervals must agree. Dev data must be frozen across comparisons. This is reference-action prediction, not live solve rate. |
| Checkpoints | Latest N plus best by task-macro dev loss, baseline eligible, native optimizer/scheduler/sampler state | Saved receipt is not a full payload hash or serving acceptance. Seal, export, reload and evaluation are separate. |
| Tracking | W&B entity/project/group/run ID/name/tags; exact config identities and scalar metrics | New treatment requires new output, prepared directory and W&B ID. No automatic sweep ranking or W&B ID reservation across submissions. |
| Submission | CPU preflight, actual API preview, duplicate name/output check, durable one-POST journal | At most 8 workers × 8 GPUs **per request**, not an aggregate campaign limit. Multiple independently named submissions can exceed a human-authorized total. |

Code evidence: [SFT recipe/compiler](../training/sft.py#L28),
[runtime overrides](../training/sft_runtime.py#L271),
[data builder](../training/corpus.py#L39),
[dense segmentation](../training/dense.py#L203),
[validation/checkpoint binding](../training/sft_runtime.py#L845),
[W&B](../training/sft_runtime.py#L927),
[per-request resource check](../cyber_post_train/jobs.py#L131),
[single-submit journal](../cyber_post_train/jobs.py#L339).

The defaults above were checked against the exact upstream SkyRL revision
`f5bc3b78dfddfb352870d5d7430cd226e5785838`, not today's upstream main:
[OptimizerConfig](https://github.com/NovaSky-AI/SkyRL/blob/f5bc3b78dfddfb352870d5d7430cd226e5785838/skyrl/train/config/config.py#L158)
and [FSDP optimizer construction](https://github.com/NovaSky-AI/SkyRL/blob/f5bc3b78dfddfb352870d5d7430cd226e5785838/skyrl/backends/skyrl_train/distributed/fsdp_strategy.py#L258).
The fetched config SHA-256 is `a3b36099c5308fc9bc658394f10cfff0da3f0cadcc067aa2b44e95870a609c90`,
matching the repository runtime source lock.

## What the training objective actually weights

The qualified SFT path is **mean assistant-token cross-entropy per realized
global optimizer batch**, not mean-of-window-losses or mean-of-rank-losses:

1. The native collator divides every positive loss-mask entry by the total
   supervised tokens in the complete batch, before data-parallel sharding.
2. Native workers sum those pre-scaled losses across microbatches. The SFT
   backward branch multiplies by data-parallel world size to compensate for
   FSDP's averaging. The actor optimizer does not divide by microbatch count again.
3. Tool observations, copied earlier assistant turns, prompt context and tail
   padding contribute zero direct next-token loss. They can still affect
   predictions through context; masking is not removing them from model input.

Sources: exact [collator](https://github.com/NovaSky-AI/SkyRL/blob/f5bc3b78dfddfb352870d5d7430cd226e5785838/skyrl/train/dataset/collators.py#L34),
[worker CE branch](https://github.com/NovaSky-AI/SkyRL/blob/f5bc3b78dfddfb352870d5d7430cd226e5785838/skyrl/backends/skyrl_train/workers/worker.py#L903),
[actor optimizer step](https://github.com/NovaSky-AI/SkyRL/blob/f5bc3b78dfddfb352870d5d7430cd226e5785838/skyrl/backends/skyrl_train/workers/worker.py#L1031).
Fetched collator and worker hashes exactly match the runtime's source lock
(`891fe6fd…` and `a3db942b…`).
[Native CE regression](../tests/test_sft_runtime.py#L1073) compares sparse masks,
tail padding and logical DP sizes 1/8 against direct causal CE and gradients;
it is an exact-image CPU test, not a distributed GPU accumulation proof.

This is **not family-balanced training**. Longer traces and families with more
successful traces supply more targets. Also, equal-weight optimizer updates on
variable-token batches are not the same thing as one corpus-wide token-normalized
update. Changing global batch changes updates/epoch, grouping, target tokens per
update, gradient noise and the denominators applied to targets. It is not just
a throughput knob. Changing microbatch size alone should preserve a fixed
global-batch objective, subject to numerical/stochastic differences; that needs
a same-global-batch equivalence test before claiming exact invariance.

The wrapper logs per-update `train/supervised_tokens` and cumulative
`train/total_supervised_tokens` and checks final epoch coverage
([train step](../training/sft_runtime.py#L1045),
[completion check](../training/sft_runtime.py#L1166)). Compare runs at matched
supervised-token/epoch exposure as well as compute cost, not equal step number.
An unweighted average of plotted batch losses is not exact corpus NLL. Validation
does reconstruct NLL sums and token counts across all batches before averaging,
and separately averages per-task means
([EvalAccumulator](../training/sft_runtime.py#L553)).

## Historical Qwen studies: do not conflate them

| Study | Training data/recipe | Evidence and limitation |
| --- | --- | --- |
| September 7 original teacher run | 508 successes over 86 success-bearing train lineages; exactly five selected ending targets/source = 2,540 windows; global batch 8; 1 epoch = 318 updates; LR 1e-6 | Completed optimization, then failed final inline export synchronization. Last complete resume checkpoint was step 300. Training-only loss, no held-out curve. |
| September 10 preliminary teacher/self comparison | Teacher: 442 successes, 73 tasks, 2,210 windows, 559,942 target tokens. Self: 35 Qwen successes, 20 tasks, 175 windows, 45,676 target tokens. Initially shared dev100. | Teacher LR1/LR3 and self LR1 plans were distinct treatments, not a finished controlled hyperparameter search. Self plan is 2 epochs/44 steps; its terminal outcome was not independently re-audited here. |
| September 10 dense teacher V5 | GPT-5.6-sol only; 186 compatible successes, 61 task lineages; 1,482 segments; 7,641 target responses; 1,622,545 target tokens; batch8, LR3e-6, 1 epoch/186 updates | Historical terminal receipt records success, all GPUs released and checkpoint 186 saved. Fixed 20-task/100-window dev: token loss 0.719304 → 0.375388; task-macro 0.735316 → 0.381986. No task-solving uplift proven. |
| September 11 generic CLI qualification | One training source/eight windows, 7,770 targets; two dev tasks/two windows/74 targets; one update, followed by separate checkpoint/recovery tests | Confirms new individual-run plumbing. Not an independent full learning study or HPO result. |

The V5 segmentation used the same 16,384-token ceiling, a soft 4,096-token
recent-context budget, and complete immediate preceding tool rounds. It retained
93.69% of assistant responses; 515 responses were explicitly excluded because
the required context plus target would not fit. No tool output was shortened.
Its target-token total is 2.90× the **preliminary dev20 teacher** corpus above,
not automatically 2.90× the original 508-source run. Fewer segments do not mean
less supervision: dense segments can train several fitting assistant responses.

The dense V5 dev set replaced four incompatible old references/tasks while
keeping all old dev/test/reserved lineages out of training. Its dev Parquet is
therefore **different** from self V2's. The exact plans have 61 teacher training
tasks versus 20 self training tasks, with **19 in common**. Their curves cannot
isolate teacher versus self distillation without common evaluation, coverage and
token-budget controls. Historical `task_family` labels in this 160-task corpus
are task-key-derived: this is task-lineage held-out with shared applications,
not independent vulnerability-family or application-held-out evidence.

Immutable historical sources:
[original audit](https://github.com/fleet-ai/cyber-post-train/blob/25ad2a37a96db5a1f2ecdccb24ac557e751a2ac6/docs/evidence/training-preparation/2026-09-08-qwen38-teacher-sft-validity-audit-v2.json),
[V5 run/terminal evidence](https://github.com/fleet-ai/cyber-post-train/blob/2f290e4760a0a9f5a5e8c9394a3ffa95ce1b1efd/docs/QWEN38_TEACHER_DEVICE_FIX_20260910.md),
[V5 exact plan](https://github.com/fleet-ai/cyber-post-train/blob/2f290e4760a0a9f5a5e8c9394a3ffa95ce1b1efd/configs/runs/sft-dense-20260910/teacher-dense-v5.plan.json),
[dense data audit](https://github.com/fleet-ai/cyber-post-train/blob/2f290e4760a0a9f5a5e8c9394a3ffa95ce1b1efd/docs/DENSE_TEACHER_V4_DATA_20260910.md),
[preliminary self/teacher data](https://github.com/fleet-ai/cyber-post-train/blob/01156cf13798cee2871127db000a1f028f3527c0/docs/SFT_DEV20_DATA_20260910.md),
[new CLI qualification](evidence/cleanup-qwen-sft-20260911.json).
These stored audits report earlier independently checked receipts; this audit
did not re-read private checkpoint payloads or contact their retired Pods.

## Minimal next work before a broad search

1. **Qualify dev-first execution.** Use the new dev cluster for each new/changed
   job configuration; preserve exact successful image/runtime/data/topology
   evidence before production promotion. A CPU preflight or a different model's
   canary is insufficient. The CPU data cluster is for preparation, not GPU SFT.
2. **Evaluate existing V5 first.** Seal/export/reload the exact best checkpoint,
   then run a matched, frozen Fleet-only held-out blackbox comparison against
   fresh base and frontier references. Do not change the recipe using sealed
   external-benchmark results.
3. **Freeze one comparison corpus/split.** New tasks need audited lineage and
   provenance, not random row splitting. Use a fresh base when newly held-out
   tasks were exposed to older SFT. Prepare teacher and self with identical
   window semantics and a common dev artifact. Report all-available-data arms
   separately from matched-common-task/equal-token arms.
4. **Add only the missing high-value controls.** Expose and test warmup,
   scheduler and weight decay in the existing recipe, rather than adding a new
   trainer abstraction. Record effective native optimizer defaults explicitly.
   Add unequal-target-count, multi-microbatch/same-global-batch gradient checks.
   Publish per-task/source supervised-token shares before changing balancing.
5. **Use a small frozen trial manifest, not an unattended sweep yet.** Each
   treatment needs unique plan/output/W&B identities, fixed data/dev digests,
   LR/global-batch/seed/token budget, dev qualification proof, and a trial-state
   row. Reuse the existing prepare/preflight/preview/submit rail. A future batch
   launcher must enforce aggregate active/admitted/reserved capacity atomically
   across concurrent submitters; a per-request limit or a W&B agent count is
   not that guarantee. Until then, use one submitter and reviewed bounded waves.
6. **Tune in stages.** After choosing source/window semantics, bracket LR around
   the demonstrated 3e-6 control, holding global batch/data fixed. Test batch
   size on the best LR region at matched token exposure, then evaluate whether
   a second epoch is worthwhile using common dev checkpoints and live Fleet
   validation. Predeclare any 2-epoch horizon; do not silently extend a finished
   1-epoch run or change the scheduler during recovery. A few independent seeds
   for finalists are more informative than dozens of single-seed settings.

The fixed historical dev100 contains only 17,709 targets, includes only fitting
successful-reference actions, and shares applications with training. It is a
useful cheap tuning signal, not a representative estimate of exploration or
long-horizon exploitation ability. Expanding or altering it creates a new study;
first remeasure the base and existing candidates on that same frozen set.

Teacher SFT asks whether verified stronger behavior can be transferred; self-SFT
asks whether the student's already-achievable successes can be made more likely.
Neither comparison identifies a superior method if source coverage, target
tokens, context policy and held-out references differ. RL asks a different
question—whether outcome feedback improves fresh interactive behavior—and still
needs a real, exact-version reward/update/recovery gate before broad tuning.
