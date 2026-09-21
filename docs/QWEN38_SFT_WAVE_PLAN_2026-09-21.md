# Qwen3.8 SFT next-wave plan — 2026-09-21

## Decision in one paragraph

Keep the successful-trajectory program **action-only**, but make the next
study a small staged search rather than treating one inherited recipe as the
answer.  First compare three deliberately spaced full-weight settings and two
LoRA learning rates on the same 32K corpus.  Only then spend compute on more
LoRA capacity, 96K history, or a second epoch.  Existing reviewed runs count as
evidence when they match a row exactly; never duplicate them merely to fill the
table.  Use protected Fleet development-task outcomes to choose among arms,
then use the untouched Fleet final split and WebExploitBench only to confirm a
frozen choice.  Training loss is a health and capacity signal, not evidence of
cyber capability.

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
| LoRA anchor | One epoch, global batch 8, learning rate `3e-5`, rank 64, alpha 32, zero dropout, every linear layer, seed `20260919` |
| Approximate supervised tokens per update | 32K batch 8: 31.2K; 32K batch 16: 62.5K; 64K batch 8: 51.3K; 96K batch 8: 67.0K.  These are corpus-wide means, not fixed token batches. |
| Explicit schedule bindings | Both paths bind `constant_with_warmup` with zero warmup.  The LoRA path additionally binds weight decay `0.01` and maximum gradient norm `1.0`; the dense path does not currently bind those two fields in its source configuration. |
| Native model context | The exact Qwen3.8 model supports 262,144 tokens natively.  None of the 32K–96K arms needs YaRN or another context extension. |

Here, **masked history** means earlier conversation tokens are supplied to the
model so it can condition an action on them, but the optimizer receives no
loss for reproducing them.  A **target token** is a visible assistant token on
which the optimizer does receive loss.  Thus the three context treatments hold
target-token exposure fixed, but not row count, update count, or wall time.
Those differences are part of the declared context treatment, not evidence
that one arm saw more supervised action tokens.  A global batch of `8` means
eight windows, not eight tokens or a fixed token budget.  Every run must record
actual non-padding input tokens and supervised target tokens per update instead
of using the window count as a token-throughput proxy.

The source of these facts is the three dense manifests
[`32K`](../configs/data/qwen38-teacher3k-32k-v1.manifest.json),
[`64K`](../configs/data/qwen38-teacher3k-64k-v1.manifest.json), and
[`96K`](../configs/data/qwen38-teacher3k-96k-v1.manifest.json), plus the
current dense runtime.  The 57.38M-token corpus already exceeds the project's
20M-token minimum; larger data should add verified task-family coverage, not
repeat existing windows merely to make a larger number.

## Source-supported facts and explicit inferences

### What the sources directly support

- The official [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
  identifies a 27B dense model with native 262,144-token context and warns that
  static YaRN can reduce quality on shorter inputs.  It also makes thinking and
  historical-message handling part of the model's template contract.
- [CTF-Dojo](https://arxiv.org/abs/2508.18370) is the closest public cyber
  reference found.  It full-tuned Qwen3-32B on 486 filtered,
  execution-verified successful CTF trajectories using global batch 16,
  learning rate `5e-6`, two epochs, and a 32,768-token cap.  Its reported 32B
  average increased from 20.3% to 31.9% across its selected CTF benchmarks.
- The [OpenThinker-32B recipe](https://github.com/open-thoughts/open-thoughts/blob/main/train/OpenThinker-32B.yaml)
  uses full Qwen2.5-32B tuning with 16K sequences, global batch 96, learning
  rate `1e-5`, three epochs, cosine decay, and 10% warmup.  The later
  [OpenThinkerAgent-32B model card](https://huggingface.co/open-thoughts/OpenThinkerAgent-32B)
  reports full Qwen3-32B tuning on 100K agent traces at 32K context, batch 96,
  learning rate `4e-5`, five epochs, cosine decay, and 10% warmup.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) reports that
  sufficiently high-rank adapters on all attention and feed-forward linear
  layers can track full-tuning loss early, that its LoRA optima were often
  around ten times the full-tuning learning rate, and that LoRA paid a larger
  penalty from large batches in some settings.
- [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)
  reports that full tuning learned substantially higher-rank changes and won
  in-domain code/math performance in its experiments, while LoRA forgot less.
  Within its LoRA sweeps, learning rate mattered most, all-module targeting
  beat attention-only targeting, and rank 256 beat rank 16.
- The exact-model [Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8/train)
  and official-Qwen-linked
  [Axolotl Qwen3-32B QLoRA example](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.9.2/examples/qwen3/32b-qlora.yaml)
  use zero-dropout adapters over attention and feed-forward projections.  Their
  `2e-4` examples are short-context QLoRA/tutorial settings, not evidence that
  `2e-4` is appropriate for this 57.38M-token, 32K cyber run.

### What this plan infers rather than claims as fact

- Dense full tuning is the likely capability ceiling for this broad skill and
  large corpus; LoRA is the lower-cost and potentially lower-forgetting method
  that still needs a matched outcome comparison.
- The initial dense bracket should be `3e-6`, `5e-6`, and `1e-5`; `3e-5` is a
  conditional upper probe.  No source establishes an absolute optimum for this
  exact model, corpus, trainer, or task.
- LoRA should test rank 64 at `3e-5` and `1e-4` before testing rank 256 at the
  winning rate.  Holding alpha at 32 in the rank test isolates capacity; it
  does not prove alpha 32 is globally optimal.
- AdamW with cosine decay, 5% warmup, weight decay `0.01`, and gradient clipping
  at `1.0` is a proposed clean comparison contract synthesized from nearby
  recipes.  It is not a result from Qwen3.8 cyber SFT.
- One epoch is the first stopping point because the corpus already contains
  57.38M unique supervised targets.  A second epoch is justified only by
  protected task outcomes, not by a prior paper's duration or falling loss.
- Searching at 32K first is a compute and attribution decision.  Testing 96K
  only after method/LR selection isolates the long-history question more
  efficiently; native 262K support does not imply every training window should
  be 262K.

## Prioritized eight-arm conditional grid

This table is a **research order**, not eight immediate launches and not eight
executable configurations.  Existing exact results may satisfy a row; otherwise
a new immutable configuration and normal qualification are required.  Rows 6–8
are conditional and should not consume compute until their stated earlier
question has an answer.

For a clean new comparison, hold the model revision, corpus and split digests,
row order, Qwen chat template, visible-action loss mask, seed, precision,
optimizer family, schedule, weight decay, gradient clipping, and evaluation
protocol fixed unless the row names that field.  The proposed common optimizer
contract is AdamW, cosine decay, 5% warmup, weight decay `0.01`, maximum gradient
norm `1.0`, and BF16.  That contract is an **inference from nearby recipes**, not
a source-proven optimum; it must be qualified before it becomes executable.

| Priority | Arm | Method | Context / masked history | Global window batch (mean supervised tokens/update) | Peak LR | Duration | Motivation and start gate |
|---|---|---|---|---|---|---|---|
| 1 | **D-low paired control** | Full-weight | 32K / 8K | `8` (~31.2K) | `3e-6` | 1 epoch | Reproduce the conservative dense anchor under fully explicit optimizer bindings. Reuse an existing result only if every common field matches. |
| 2 | **D-domain anchor** | Full-weight | 32K / 8K | `16` (~62.5K) | `5e-6` | 1 epoch | Closest practical translation of CTF-Dojo's Qwen3 cyber recipe, with duration held to one epoch so duration remains a later decision. |
| 3 | **D-mid LR** | Full-weight | 32K / 8K | `8` (~31.2K) | `1e-5` | 1 epoch | Test whether `3e-6` under-updates this 57.38M-token corpus. Do not jump to `3e-5` unless this arm is finite, stable, and better on protected development tasks. |
| 4 | **L-low** | LoRA rank 64, alpha 32, all attention and MLP linear layers, dropout 0 | 32K / 8K | `8` (~31.2K) | `3e-5` | 1 epoch | Roughly tenfold LR companion to D-low and the currently qualified LoRA capacity. The existing A2 result may satisfy this operational arm but is not a causal adapter-only comparison. |
| 5 | **L-mid LR** | Same rank-64 LoRA | 32K / 8K | `8` (~31.2K) | `1e-4` | 1 epoch | Test the wider adapter LR region supported by nearby LoRA work. Start only if L-low is operationally sound; stop for predeclared non-finite or divergence conditions. |
| 6 | **L-capacity** | LoRA rank 256, alpha 32, same all-linear targets and dropout | 32K / 8K | `8` (~31.2K) | Winning LoRA LR | 1 epoch | Test an adapter-rank bottleneck only if rank 64 is stable but plateaus or trails dense. Holding alpha fixed isolates rank before any later LR retune. |
| 7 | **Long-history winner** | Winning dense or LoRA method | 96K / 24K | `8` (~67.0K) | Winning method LR | 1 epoch | Test richer authentic history after method/LR selection. Compare primarily with 32K batch 16 (~62.5K supervised tokens/update), while disclosing different row grouping and processed history. |
| 8 | **Second epoch** | Winning method and context | Winning context | Winning batch | Winning LR | Exactly 1 additional epoch | Continue only if protected development-task success is still improving without a material reliability regression. Treat this as an explicit continuation, not as the midpoint of a separately scheduled two-epoch cosine run. |

### Why this grid is small and ordered

- **Dense first sets the ceiling.** The corpus is large and the target skill is
  broad, so full-weight training is the higher-capacity reference rather than
  an expensive afterthought.
- **The dense rates are intentionally spaced.** `3e-6` preserves the current
  conservative anchor, `5e-6` comes from the closest cyber recipe, and `1e-5`
  is supported by nearby 32B full-tuning recipes.  `3e-5` is only a conditional
  upper probe after `1e-5`, not an initial arm.
- **LoRA tunes rate before rank.** The cited LoRA studies found learning rate
  more consequential than rank within the tested ranges.  Rank 256 therefore
  follows evidence of a stable rank-64 capacity limit rather than running in
  parallel by default.
- **Long context comes after method selection.** The 32K representation already
  contains every supervised target.  The 96K row tests access to more authentic
  history, not missing target labels.  Approximately matching supervised tokens
  per update reduces one confound but does not make it a pure context-only test.
- **A second epoch is earned by task outcomes.** Lower imitation loss alone is
  not enough.  A checkpoint at epoch one inside a two-epoch cosine schedule is
  not equivalent to a separately planned one-epoch run because its learning-rate
  horizon differs.

### Current dense-versus-LoRA confounds

The current dense and LoRA anchors are useful operational evidence, but they do
not isolate mathematical adaptation method:

- dense uses seed `20260920`; LoRA uses `20260919`;
- dense uses the FSDP path; LoRA uses the Megatron path;
- both currently use constant LR with zero warmup, while only LoRA explicitly
  binds weight decay `0.01` and maximum gradient norm `1.0`;
- checkpoint intervals and retention differ; and
- the two runtimes may differ in low-level batching and numerical order even
  when the high-level data fields match.

Describe the existing result as a **method-and-runtime-and-seed comparison**.
Before making a causal adapter-only claim, qualify a new paired dense and LoRA
comparison with the same data order, seed, explicit optimizer values, schedule,
precision, and evaluation.  If the backend cannot be shared, keep the backend
difference in the label and add a small parity check; do not edit sealed plans
to manufacture a match.

### LoRA module contract

"All linear" means every trainable projection in attention and the feed-forward
network, normally the Q/K/V/output and gate/up/down projections.  A future plan
must enumerate the actual Qwen3.8 module names from the pinned model and prove
coverage before launch.  Embeddings and the output head stay frozen unless a
separate row explicitly studies them.  Rank 64, alpha 32, zero dropout, and the
standard initialized adapter are the primary setting; rank 256 is a conditional
capacity test, not an assumed improvement.

### Checkpoint and measurement policy

Checkpoint cadence is an operational choice, not a capability hyperparameter.
For future arms:

- write a restartable dense checkpoint about every 10% of an epoch and at the
  end; for 32K this is roughly every 180 updates at batch 8 or 90 at batch 16;
- LoRA adapters are much smaller, so every 100 updates plus the end is a
  practical default;
- retain the latest two restart checkpoints, while separately sealing exact
  25%, 50%, 75%, and 100% candidates chosen for comparison; and
- record windows, non-padding input tokens, supervised target tokens, learning
  rate, gradient norm, loss, throughput, and checkpoint identity in W&B.

The exact interval still needs an observed storage/write-time check.  This rule
is an operational recommendation, not evidence that more frequent checkpointing
improves learning.

### Comparison record required for every arm

The 32K, 64K, and 96K rows are **history treatments**: they hold each visible
target token fixed once while changing how much real, loss-masked prior task
history is supplied. They are not a pure "context length only" ablation because
row count, optimizer-update count, masked-token compute, and wall time differ.
Every result must retain this compact comparison record rather than reduce the
treatment to a context-length label.

| Record | Required fields |
|---|---|
| Data and history | Corpus and split digests; source/task-family counts; unique target tokens; explicit source/family caps; history cap and packing rule |
| Adaptation and runtime | Full-weight or LoRA; exact adapter targets/rank/alpha when relevant; trainer/runtime image; nodes, parallelism, and accumulation |
| Optimization | Global window batch; measured input and supervised tokens/update; LR and schedule; warmup; duration; checkpoint policy; seed; explicit optimizer, weight-decay, and clipping values |
| Usable artifact | Exact checkpoint, export/reload receipt, and matching serving template |
| Choice rule | Protected Fleet development-task outcome, invalid-run count, and predeclared tie-break—not training loss or teacher-token loss |

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

That is a data-admission deliverable before it becomes a later source-treatment
row; it is not silently folded into any of the eight hyperparameter arms above.

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
3. Advance through the grid from dense rate/batch, to LoRA rate/capacity, to
   long history, to epoch count only when the prior protected Fleet development
   result answers its stated question.  Record the decision before touching a
   final holdout or WebExploitBench.
4. Confirm the selected frozen checkpoint on the untouched Fleet final split
   and a matched OpenCode WebExploitBench protocol.  External benchmark
   material never re-enters training or selection.

If no arm beats the base reliably on protected development tasks, the correct
next step is data-quality/source analysis, not a larger unstructured LR grid.

## Evidence and literature ledger

| Source and type | Directly supported observation | Use in this plan | Transfer limit |
|---|---|---|---|
| [Bound Fleet manifests](../configs/data/qwen38-teacher3k-32k-v1.manifest.json) and [current dense plan](QWEN38_NEXT_SFT_EXPERIMENTS_2026-09-20.md) (local immutable/config evidence) | The same 57.38M supervised targets are represented at 32K, 64K, and 96K; current dense controls include batches 8/16 and rates `1e-6`/`3e-6`. | Establishes local counts, current anchors, and the history-treatment interpretation. | Current job/checkpoint state still comes from exact terminal receipts, not this narrative plan. |
| [Reasoning and compaction policy](QWEN38_REASONING_AND_COMPACTION_POLICY.md) (local policy) | Visible actions are allowed; provider-private reasoning and opaque compaction are excluded. | Keeps this wave action-only and gates any future visible-reasoning corpus. | A policy is not evidence that reasoning data would improve capability. |
| [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) (official model documentation) | 27B dense model, native 262,144-token context, exact template/thinking controls, and a caution about static YaRN on shorter inputs. | Uses native context only and requires exact template binding. | Model support does not determine the best training-window length. |
| [CTF-Dojo](https://arxiv.org/abs/2508.18370) (primary cyber paper) | Verified successful CTF trajectories; full Qwen3-32B SFT at batch 16, `5e-6`, two epochs, and 32,768 tokens; reported 20.3% to 31.9% average for its 32B model. | Supplies the dense `5e-6`/batch-16 domain anchor and makes a conditional second epoch credible. | Different model generation, tasks, curation, trainer, and much smaller corpus; it does not set this optimum. |
| [OpenThinker-32B recipe](https://github.com/open-thoughts/open-thoughts/blob/main/train/OpenThinker-32B.yaml) and [OpenThinkerAgent-32B](https://huggingface.co/open-thoughts/OpenThinkerAgent-32B) (maintainer recipes) | Nearby 32B full-tuning recipes use `1e-5` to `4e-5`, cosine decay, warmup, multiple epochs, and much larger global batches. | Motivates a spaced dense bracket and explicit schedule bindings rather than copying the current constant schedule without question. | Different Qwen generations, datasets, batch sizes, objectives, and agent harnesses. |
| [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (lab research report) | All-layer, sufficiently high-rank LoRA can track full-tuning loss early; LoRA often preferred about tenfold higher LR and could suffer more from large batches. | Motivates all-linear targets, batch 8, and `3e-5`/`1e-4` LoRA rates. | Not a cyber-agent study and not the repository's exact Megatron implementation. |
| [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (primary paper) | Full tuning learned higher-rank changes and better in-domain code/math behavior; LoRA forgot less; LR, target coverage, and rank all mattered. | Keeps dense as the ceiling, prioritizes LoRA LR before rank, and makes rank 256 conditional. | Different models, tasks, data volume, and context lengths. |
| [Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8/train) and [Axolotl Qwen3-32B QLoRA example](https://github.com/axolotl-ai-cloud/axolotl/blob/v0.9.2/examples/qwen3/32b-qlora.yaml) (exact/near-model maintainer examples) | Zero-dropout LoRA over attention and feed-forward projections; short examples use `2e-4`, warmup, and short context. | Supports module coverage and zero dropout, not direct LR transfer. | Tutorial/QLoRA settings at about 2K context and short horizons are not this 32K, 57.38M-token run. |
| [QLoRA](https://arxiv.org/abs/2305.14314) (primary paper) | Quantized-base adapters can reduce memory substantially while retaining strong fine-tuning performance in its studied settings. | Keeps QLoRA available as a resource control if memory is the binding constraint. | It does not show QLoRA is superior to BF16 LoRA or dense tuning for long cyber trajectories. |

## Explicit non-actions

- Do not use teacher cross-entropy on held-out teacher rollouts as a proxy for
  blackbox-exploitation progress.
- Do not mix WebExploitBench prompts, traces, outcomes, or hints into a corpus,
  reward, or experiment choice.
- Do not scale unqualified catalog tasks into SFT merely because they increase
  a row count.
- Do not resume an old failed identity or turn this document into a job request.
