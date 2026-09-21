# Qwen3.8 SFT next-wave plan — 2026-09-21

## Decision in one paragraph

Keep the current successful-trajectory SFT program **action-only** while
measuring three things in a small, ordered study: how much preceding task
history to retain (32K, 64K, or 96K tokens), whether a full model update or a
LoRA adapter is the better practical method, and whether a second pass over
the winning recipe helps.  Do not spend the next wave on another dense
learning-rate or batch-size grid: the current 32K study already contains
`batch=8` versus `16` and `3e-6` versus `1e-6` controls.  Instead, use the
protected Fleet development task outcomes to choose among the planned arms,
then use the untouched Fleet final split and WebExploitBench only to confirm a
frozen choice.  Training loss remains a health signal, not evidence of cyber
capability.

This is a plan, not a launch request.  Every proposed row still needs its
normal data, capacity, duplicate, preview, failure-alert annotation, and
checkpoint/reload gates.

## Fixed facts and terms

| Item | Current bound fact |
|---|---|
| Student | `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Broad teacher corpus | 2,886 verified successful sessions; 496 training task keys; 176,654 assistant responses; 57,384,881 unique supervised target tokens |
| Split protection | Every version of each of 25 frozen held-out Fleet task families is excluded before windows are made; WebExploitBench is never training data |
| SFT target | Visible assistant text and tool calls only.  Tool results and copied history have zero loss. |
| Context variants | 32K/8K history: 14,693 rows/1,837 steps; 64K/16K history: 8,953/1,120; 96K/24K history: 6,847/856.  Each contains every target token exactly once. |
| Dense anchor | One epoch, global batch 8 (= 1 microbatch × 8 GPUs), learning rate `3e-6`, one eight-GPU node, seed `20260920` |
| Explicit schedule bindings | Both paths bind `constant_with_warmup` with zero warmup.  The LoRA path additionally binds weight decay `0.01` and maximum gradient norm `1.0`; the dense path does not currently bind those two fields in its source configuration. |

Here, **masked history** means earlier conversation tokens are supplied to the
model so it can condition an action on them, but the optimizer receives no
loss for reproducing them.  A **target token** is a visible assistant token on
which the optimizer does receive loss.  Thus the three context treatments hold
target-token exposure fixed, but not row count, update count, or wall time.
Those differences are part of the declared context treatment, not evidence
that one arm saw more supervised action tokens.

The source of these facts is the three dense manifests
[`32K`](../configs/data/qwen38-teacher3k-32k-v1.manifest.json),
[`64K`](../configs/data/qwen38-teacher3k-64k-v1.manifest.json), and
[`96K`](../configs/data/qwen38-teacher3k-96k-v1.manifest.json), plus the
current dense runtime.  The 57.38M-token corpus already exceeds the project's
20M-token minimum; larger data should add verified task-family coverage, not
repeat existing windows merely to make a larger number.

## The five next training decisions

The existing 32K and 64K dense configurations are controls for this table, not
fresh duplicate launches.  An existing configuration records a reviewed
scientific identity; it is not proof that its create-once output path is still
available.  Before any submission, the operator must prove the output and job
identity are absent, or create a newly qualified successor rather than reuse a
consumed identity.  "After gate" means the row must not start merely because
its predecessor has lower training loss.

| Priority | Proposed run | Reviewed source artifact | Data and method | Context | Global batch / LR / duration | What it answers | Start only after |
|---|---|---|---|---|---|---|---|
| 1 | **Dense 96K action-only** | [`96K dense config`](../configs/runs/qwen38-teacher3k-96k-full-b8-lr3e6-v3.json) | Same 57.38M teacher targets, full-weight SFT | 98,304 tokens; 24,576 masked-history cap | `8` / `3e-6` / one epoch (856 planned updates) | Does retaining still more real trajectory state help beyond the 32K and 64K controls? | A fresh create-once successor passes the exact CPU and GPU qualification gates; it must not reuse an old failed identity. |
| 2 | **LoRA anchor** | [`A2 config`](../configs/runs/qwen38-27b-lora-sft-r64-a32-anchor-a2-v1.json) | Same 32K corpus and split; rank-64, alpha-32 adapter on every linear layer | 32,768 / 8,192 masked history | `8` / `3e-5` / one epoch (1,837 planned updates) | Is a high-capacity all-linear adapter a practical alternative to the 32K dense reference? | The already accepted LoRA save, reload, merge, and full-model reload chain is revalidated for the exact run. |
| 3 | **LoRA lower-LR control** | [`lower-LR config`](../configs/runs/qwen38-27b-lora-sft-r64-a32-lr1e5-v1.json) | Identical to priority 2 except learning rate | 32,768 / 8,192 | `8` / `1e-5` / one epoch | Is the anchor's roughly tenfold dense-to-LoRA LR ratio too aggressive on this long tool-use corpus? | Priority 2 completes with finite updates, a reloadable merged checkpoint, and no operational defect. |
| 4 | **LoRA upper-LR boundary** | [`upper-LR config`](../configs/runs/qwen38-27b-lora-sft-r64-a32-lr1e4-v1.json) | Identical to priority 2 except learning rate | 32,768 / 8,192 | `8` / `1e-4` / one epoch | Brackets the high side rather than assuming `3e-5` is optimal.  It is an intentionally wide boundary check, not a default. | Priority 2 is operationally healthy and its development evaluation is available; do not spend a node on this if the anchor already shows instability. |
| 5 | **Second epoch of the selected dense context** | New successor bound to the selected epoch-one checkpoint | Resume the best of 32K/64K/96K from its exact epoch-one checkpoint; no data or recipe change | Winning epoch-one context | `8` / `3e-6` / exactly one more epoch | Does a second pass improve protected task success instead of merely lowering imitation loss? | One context wins the predeclared Fleet development metric without a material reliability regression. |

### What makes these an intentional set

- **Context first.** The practical unknown is whether long cyber trajectories
  need more authentic preceding state.  The 32K/64K/96K series changes that
  one data representation while preserving the same targets, split, model,
  batch, learning rate, epoch count, and seed.
- **LoRA is a real comparison, not a presumed shortcut.** The LoRA rows match
  model, source, split, target exposure, context, batch, and epoch count to the
  32K dense reference.  They use rank 64, alpha 32, zero dropout, and all
  linear layers because that is the supported, receipt-qualified Qwen3.8
  implementation.  The reviewed dense 32K configuration uses seed `20260920`,
  while the reviewed LoRA configurations use seed `20260919`; their Megatron
  runtime also differs from the dense FSDP runtime, and only the LoRA path
  explicitly binds weight decay and gradient clipping.  The resulting
  comparison is therefore a useful *method-and-runtime-and-seed* result, not a
  claim that only one mathematical parameterization changed.  A causal
  adapter-only claim requires a newly qualified paired successor with one
  identical seed and explicit optimizer bindings on both paths; never edit
  either sealed configuration to manufacture that match.
- **The LR bracket is deliberately sparse.** `3e-5` is the direct tenfold
  companion to the dense `3e-6`; `1e-5` and `1e-4` check materially lower and
  higher regions.  There is no value in adjacent tiny changes before a
  protected task result exists.
- **Epoch two is conditional.** A two-epoch recipe is plausible prior art, but
  launching it before context selection would conflate duration with context.
  The resume must use the same schedule semantics as the completed arm; it is
  not an early slice of a different long cosine schedule.

### Comparison record required for every arm

The 32K, 64K, and 96K rows are **history treatments**: they hold each visible
target token fixed once while changing how much real, loss-masked prior task
history is supplied. They are not a pure “context length only” ablation because
row count, optimizer-update count, masked-token compute, and wall time differ.
Every result must therefore retain this compact comparison record rather than
reduce the treatment to a context-length label.

| Record | Required fields |
|---|---|
| Data and history | Corpus and split digests; source/task-family counts; unique target tokens; explicit source/family caps; history cap and packing rule |
| Adaptation and runtime | Full-weight or LoRA; adapter targets/rank/alpha when relevant; trainer/runtime image; nodes, parallelism, and accumulation |
| Optimization | Effective global batch, learning-rate schedule, duration, checkpoint policy, seed, and explicit weight-decay/gradient-clipping values (or an explicit statement that they are not bound) |
| Usable artifact | Exact checkpoint, export/reload receipt, and matching serving template |
| Choice rule | Protected Fleet development task outcome, invalid-run count, and the predeclared tie-break—not training loss or teacher-token loss |

The dense and LoRA rows are deliberately a **method-and-runtime-and-seed**
comparison: the current dense FSDP and LoRA Megatron paths do not share every
runtime detail, their reviewed seeds differ, and only the LoRA path explicitly
binds weight decay and gradient clipping. A task-outcome result may still be
useful, but it must not be described as isolating only the adapter. The winning
method is the one that improves the protected task protocol without a material
reliability regression; the final Fleet split and WebExploitBench confirm that
frozen choice.

If the LoRA anchor produces an outcome that would change an investment
decision, prepare a **new, source-qualified paired control** before making a
causal full-weight-versus-adapter claim.  It must use the same corpus, context,
global batch, epoch count, and seed on both sides, and both rendered plans must
explicitly bind the optimizer values being compared.  The current FSDP-versus-
Megatron runtime difference must remain disclosed.  This is a conditional
follow-up to a useful anchor result, not a reason to duplicate the current
arms or edit sealed plans.

## Data scale and source plan (parallel, but not a shortcut)

The existing teacher corpus already clears 20M unique supervised target tokens.
The next data question is therefore coverage, especially a new Qwen
self-success corpus, rather than repeated copies of old windows.  The complete
collection and admission design is
[`QWEN38_HIGH_THROUGHPUT_TRAJECTORY_COLLECTION_PLAN.md`](QWEN38_HIGH_THROUGHPUT_TRAJECTORY_COLLECTION_PLAN.md).
Its implications for this experiment plan are deliberately small:

1. More teacher data must add proven training families or define a clean source
   contrast; it must not repeat old attempts just to inflate a count.
2. The first standalone self-SFT treatment requires at least **20M unique
   supervised visible-action target tokens**, material new training-family
   coverage, family-safe splitting, and a manifest-backed admission receipt.
   Its initial practical SFT candidate is 64K dense, `batch=8`, `lr=3e-6`, one
   epoch.
3. Compare teacher-only and self-only sources on common eligible families with
   equal per-family target-token budgets.  An all-qualified corpus and a
   separately declared family-balanced corpus are useful practical treatments,
   but neither may silently reweight or leak a held-out family.
4. Every corpus beyond the 20M-token floor must publish source and task-family
   coverage, concentration, and explicit source/family caps in its manifest.
   Repeating old packed windows never counts as new scale.

That is a data-admission deliverable before it becomes a sixth training row.

## Reasoning and compaction: what is allowed now

### Written reasoning

Do **not** train on provider-private `thinking`, `reasoning`, or
`reasoning_content` fields.  Their presence does not mean the student may see
or learn them.  The current corpus correctly excludes them.

A future written-reasoning arm is worthwhile only after a new corpus proves
that the text is both authorized for training and visible to the student under
the same Qwen template and thinking-mode contract.  The first such experiment
should be a matched action-only versus student-visible-reasoning comparison
with the same task families, action targets, optimizer, and evaluation
protocol.  Until then, generate and train more high-quality visible actions;
do not label that as chain-of-thought distillation.

### Compaction

Serving a long-lived agent and preparing offline SFT examples are different
problems:

- **Online serving/evaluation:** Qwen can work within a 262,144-token context;
  a harness may compact a long conversation and continue.
- **Offline SFT:** the trainer sees static windows.  It does not compact while
  optimizing.  A later action from an opaque compacted rollout is rejected
  because the true summary and post-summary prompt are unknown.  The 32K/64K/
  96K arms honestly test how much original, loss-masked history to keep; they
  do not teach compaction.
- **Future explicit compaction training:** it requires an exact pre-summary
  prompt, generated summary tokens, post-summary prompt, and the later
  action's actual prompt.  That is a new corpus type, not a flag on the
  existing action corpus.

The full policy and required regression gates are in
[`QWEN38_REASONING_AND_COMPACTION_POLICY.md`](QWEN38_REASONING_AND_COMPACTION_POLICY.md).

## How results advance the plan

1. For each terminal checkpoint, first require finite updates, changed intended
   weights/adapters, a complete reloadable export, and resource release.
2. Evaluate the base and candidate checkpoint under the same frozen OpenCode
   Fleet development protocol.  Compare task-level outcomes and invalid-run
   counts, not just a pooled score or training loss.
3. Choose the next dense context, LoRA rate, or epoch count only from the
   protected Fleet development result.  Record the decision before touching a
   final holdout or WebExploitBench.
4. Confirm the selected frozen checkpoint on the untouched Fleet final split
   and a matched OpenCode WebExploitBench protocol.  External benchmark
   material never re-enters training or selection.

If no arm beats the base reliably on protected development tasks, the correct
next step is data-quality/source analysis, not a larger unstructured LR grid.

## Evidence and literature used

| Source | What it reports | How this plan uses it | Why it may not transfer directly |
|---|---|---|---|
| [Current Fleet dense plan](QWEN38_NEXT_SFT_EXPERIMENTS_2026-09-20.md) and the bound manifests above | Same 57.38M targets can be rewindowed at 32K, 64K, and 96K; current dense controls already cover batches 8/16 and LRs `1e-6`/`3e-6`. | Do context and adapter questions before another dense micro-sweep. | The newest live runs and exact checkpoint outcomes must still be read from immutable receipts. |
| [Current reasoning/compaction policy](QWEN38_REASONING_AND_COMPACTION_POLICY.md) | Visible actions are allowed; private reasoning and opaque compaction are excluded. | Keep action-only SFT honest; gate a distinct visible-reasoning corpus. | It is a safety/science contract, not evidence that reasoning data would help. |
| [CTF-Dojo](https://openreview.net/pdf?id=FAPEir5GyY) | Successful verified CTF trajectories; Qwen3 models; global batch 16, `5e-6`, two epochs, and a 32,768-token cap. | Supports trying successful-trajectory SFT and a conditional second epoch. | Different model generation, trainer, tasks, data curation, and much shorter interaction contract.  It does not set our optimum. |
| [OpenThinker-32B training recipe](https://github.com/open-thoughts/open-thoughts/blob/main/train/OpenThinker-32B.yaml) | A full Qwen2.5-32B recipe uses 16K sequences, global batch 96, `1e-5`, three epochs, cosine decay, and warmup. | Shows that nearby 32B reasoning SFT can use a very different regime; it argues against copying a generic default blindly. | Different Qwen generation, noninteractive data, data size, trainer, schedule, and objective. |
| [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) | On Qwen/Llama studies, all-layer LoRA and a roughly 10× higher LR than full tuning can be competitive; LoRA is less tolerant of large batches in some SFT settings. | Justifies all-linear rank-64/alpha-32 LoRA and the `1e-5`/`3e-5`/`1e-4` bracket at batch 8. | It is not a cyber-agent study and does not qualify this repository's exact Megatron runtime. |
| [Qwen thinking guide](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html) | Thinking mode is an explicit request/template setting. | Freeze the template and setting for any future visible-reasoning comparison. | It is API guidance, not authorization to train on hidden provider reasoning. |

## Explicit non-actions

- Do not use teacher cross-entropy on held-out teacher rollouts as a proxy for
  blackbox-exploitation progress.
- Do not mix WebExploitBench prompts, traces, outcomes, or hints into a corpus,
  reward, or experiment choice.
- Do not scale unqualified catalog tasks into SFT merely because they increase
  a row count.
- Do not resume an old failed identity or turn this document into a job request.
