# Literature-informed Qwen experiment planning

Research snapshot: 16 September 2026. This replaces the old 75–90-column
experiment grid. It proposes experiments; it does not report live jobs, qualify
an implementation, or submit anything. Historical results remain on the results
page and in their original evidence reports, not relabeled as new proposals.

## Recommendation

Start with three **full SFT** runs on the same broad, valid teacher corpus:

| Order | Peak learning rate | Effective batch | Epochs | Context cap |
|---|---:|---:|---:|---:|
| S01 | 5e-6 | 16 | 2 | 262,144 |
| S02 | 2e-5 | 16 | 2 | 262,144 |
| S03 | 1e-6 | 16 | 2 | 262,144 |

These are research priors, not estimates of success probability. The order
combines closeness of published evidence, size of the question answered, and
engineering cost. Prefer a broad range before a local search. If all three fail
to improve held-out Fleet task success, investigate data/target coverage and
evaluation validity before buying a larger sweep. If the best rate is at an
edge, propose one further logarithmic step in that direction before declaring
an optimum. Do not let a grid boundary masquerade as an optimum.

**Proposed common optimizer:** AdamW, betas (0.9, 0.95), epsilon 1e-8,
weight decay 0, gradient norm clipping 1, cosine decay to zero after a 10% update
warmup. Seed 42. These are explicit controlled design choices; not all were
reported by CTF-Dojo. The existing main runtime uses constant LR and zero warmup,
so this is **not** a drop-in reproduction of our previous run. Resolve and test
the schedule first, or explicitly version a constant-schedule study. Never
silently ignore a requested setting.

## What the literature actually establishes

The [page's source records](../site/training-decision-space.json) are the single
versioned source for observed values, exact links, differences and uses. They
contain no task content. The most informative evidence is:

- [CTF-Dojo v1 §3.1](https://arxiv.org/html/2508.18370v1): the closest task
  match. Security demonstrations, Qwen3 including 32B, LR 5e-6, batch 16,
  two epochs. Its **486 retained** trajectories were capped at 32,768 tokens;
  do not confuse collected trajectory counts or later paper versions with this
  training subset. Our longer contexts and newer architecture are real gaps.
- [OpenThinkerAgent-32B's model card](https://huggingface.co/open-thoughts/OpenThinkerAgent-32B):
  nearby model scale and tool use; 4e-5, batch 96, five epochs and cosine/10%
  warmup on 100k demonstrations. It motivates a higher-rate alternative, not
  blindly copying a much larger dataset's batch or duration.
- [OpenThinker-32B's author YAML](https://github.com/open-thoughts/open-thoughts/blob/main/train/OpenThinker-32B.yaml):
  a fully specified same-size Qwen2.5 reasoning recipe at 1e-5, batch 96,
  three epochs, 16k context. Different task, useful numerical cross-check.
- [OpenThoughts-Agent's data study](https://www.openthoughts.ai/blog/openthoughts-agent):
  invest in source diversity and useful trajectories, not only optimizer knobs.
- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) and
  [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2):
  tune adapter LR independently; target more than attention alone; test adequate
  rank and retention. Neither proves that adapters win on our long cyber tasks.
- [Hyperband](https://www.jmlr.org/papers/v18/16-558.html): spend adaptively
  instead of taking the Cartesian product. This generator is an expert-ranked
  staged design, **not an implementation of Hyperband**, Bayesian optimization,
  or an automatic performance predictor.
- [Qwen3.8-27B's official card](https://huggingface.co/Qwen/Qwen3.8-27B):
  the native 262,144 context is a model property, not proof that any given
  training kernel or placement fits a chosen node count.

We inspected related Qwen-family and recent SFT research as well. There is no
verified published optimum for **this exact model + Fleet corpus + 262k tool
history**. Unrelated benchmark tasks, solutions and traces are not imported.
Mutable author model cards/YAML were checked on the snapshot date; pin their
revision again when refreshing the study. Paper versions above are explicit.

## The stages

1. **Start here (S01–S03):** broad teacher learning-rate range. All target tokens
   and the dataset split are identical. Two full passes, not toy smoke updates.
2. **Compare data (S04–S06):** teacher, self-success and 50:50 source mixture.
   Restrict all three to the family intersection; hold per-family supervised
   token weights and total tokens fixed. The broad teacher arm is still the
   practical maximum-data candidate. The intersection answers a different,
   causal source question. If Qwen lacks successes on some families, do not
   invent them or present a smaller/easier subset as a pure source effect.
3. **Spend on promising ideas (S07–S13):** selected first-stage LR at 1 versus
   4 epochs; effective batch 64, then a targeted 2× LR interaction; independent
   LoRA rates 5e-5/2e-4 with rank 64, alpha 128, dropout 0 and all eligible linear
   layers; an optional 65,536-context cost study. These are conditional choices,
   not instructions to launch all of them. If tuned LoRA rank 64 underfits,
   examine rank 256 before concluding full tuning is intrinsically superior.
4. **Confirm (S14–S15):** repeat the selected completed recipe at seeds 43/44.
   Keep evaluation seeds fixed. A selected row can be teacher, self, mixed,
   adapter, or a duration/batch variant; inheritance must retain its actual
   data and method, not silently return to the teacher anchor.

All duration variants start from the same initial checkpoint. Each has a full
schedule for its declared horizon: the epoch-1 checkpoint of a four-epoch cosine
run is not equivalent to a separately trained one-epoch run. More epochs are
worth testing, not automatically better. Compare task outcomes and basic
instruction/tool-format retention, not just training loss.

The four separate RL proposals compare base-start LR, SFT-start, and breadth
versus repeated attempts at constant collection size. [GRPO](https://arxiv.org/abs/2402.03300)
supplies the algorithmic idea, not a proven cyber recipe. They wait for genuine
reward acquisition and checkpoint qualification; no current readiness is
inferred. Do not switch tool programs between training/evaluation, and do not
train actions against the wrong pre-compaction history. This turn does not
change the runtime, active jobs or their monitoring.

## Data and the metric that matters

Freeze a current, eligible training manifest before extracting examples. Never
hardcode the old 75-task, 916-example or approximate 3,000-session snapshot as
today's training inventory. Existing dev/final families stay excluded when
adding data. Group all versions and sessions of a task family, then stratify
by application, vulnerability category and difficulty where counts permit.
Record coverage/imbalance; family-held-out is not application-held-out.

Train all eligible assistant action targets once per epoch. Preserve useful
early and intermediate steps; do not give final submission text extra weight.
Mask user messages, tool observations, and repeated prefixes. Cross-session
packing must have correct attention boundaries, labels and position handling;
avoid it until those are proven. Retain each original model-visible context and
recorded compaction. Long examples cannot silently become final-answer-only or
tail-only examples to make a job fit. The optional shorter-context row changes
history deliberately: audit full target coverage and report the context lost.

Measure **task success on Fleet dev** with the same OpenCode implementation,
model precision, tool schema, task versions, scoring, seeds and time/token
budgets. Use paired differences and confidence intervals at the task-family
level. A 17-task dev set has little power: one changed task is about 5.9
percentage points, not strong evidence of a general improvement. Expand dev
only with a new frozen, uncontaminated manifest before looking at results;
never repeatedly tune against final tasks. Small differences merit repeats,
not winner certainty. Within statistical uncertainty prefer the cheaper stable
recipe. Freeze the decision rule and maximum candidates before evaluation.

W&B records numeric training loss (token-weighted, with optional task-macro
view), LR, gradient norms, unique supervised tokens, total processed tokens,
actual batch tokens, steps, memory, time and checkpoint progress. Also record
Fleet dev outcomes and failed/invalid episodes separately. No held-out teacher
CE loss is requested, and imitation loss is not the promotion metric. Do not
upload task content or credentials. Use study/stage/run/seed grouping and exact
config/data hashes so runs remain comparable.

WebExploitBench is **report-only**, after freezing an accepted checkpoint and
matched protocol. Run it for each accepted checkpoint as separately budgeted
work, but do not feed its results back into this generator, checkpoint
selection, curriculum, prompts, or rewards. The planner does not read the
site's benchmark results. This keeps the target capability distinct from
optimizing to a test set.

## A small reusable generator

One dependency-free implementation serves both the website and CLI:

- [Curated evidence + proposed rows](../site/training-decision-space.json)
- [Shared planning logic](../site/experiment-plan.js)
- [Read-only CLI](../scripts/plan-experiments.cjs)
- [Reusable research skill](../skills/literature-informed-experiments/SKILL.md)

```sh
# Three initial research specifications; no API call, job, or credential.
node scripts/plan-experiments.cjs --stage first --limit 3
node scripts/plan-experiments.cjs --stage data --format csv

# Only after S02 has been selected on Fleet dev:
node scripts/plan-experiments.cjs --stage efficient --winner S02

# If the completed LoRA S11 is the final selected recipe:
node scripts/plan-experiments.cjs --stage confirm --finalist S11

# If a dependent candidate won, include its original first-stage parent:
node scripts/plan-experiments.cjs --stage confirm --winner S02 --finalist S08
```

The generator cannot verify that a chosen parent really completed; choice is an
explicit researcher input, not an automatic launch gate. Without that choice,
dependent rows remain marked `resolved: false`; they are not executable recipes.
Even resolved rows are **proposed, not launch-qualified**. No output is a Jobs API
request. LoRA, cosine warmup and exact 262k training placement need corresponding
runtime support and evidence. Effective batch = data-parallel replicas ×
microbatch × accumulation, **not** necessarily all GPUs when tensor/context
parallelism is used. Reject incompatible placement rather than rounding batch.

### Count examples and compute honestly

With the native kept-tail batching rule, updates = epochs × ceil(examples/batch).
Supervised tokens = epochs × the corpus's unique supervised tokens per pass.
This is not equal to total processed tokens because history can be repeated.
When rebuilding windows or changing sampling, recount rather than copying the
old step count. The generator deliberately returns unknown counts without data.

Provide an inventory JSON with an entry for each selected data pool:

```json
{
  "Broad teacher": {
    "examples": 1234,
    "supervised_tokens": 1234567,
    "manifest_sha256": "<actual 64-character SHA-256>"
  }
}
```

Numbers here are **illustrative**, not our current corpus. Add separate entries
for matched pools, including the same `family_budget_sha256` (SHA-256 of the
reviewed per-family token-budget manifest) and identical supervised-token totals.
Shorter-context examples use the key `Broad teacher / context 65536` and need
a separately inventoried corpus; the CLI refuses borrowing the long-context
inventory for that row. Save the
generated specification hash, then provide measured total per-run cost:

```json
{
  "includes": "training_export_and_fleet_dev",
  "runs": {
    "S01": {
      "spec_sha256": "<hash from this exact generated row>",
      "gpu_hours": 80,
      "evidence": "<reviewed timing report: exact runtime, hardware, corpus, and overhead>"
    }
  }
}
```

```sh
node scripts/plan-experiments.cjs --stage first --inventory /path/inventory.json
node scripts/plan-experiments.cjs --stage first --inventory /path/inventory.json \
  --costs /path/reviewed-costs.json --budget-gpu-hours 200
```

The 80/200 GPU-hour figures above are examples only. Cost entries must include
training, startup/export and Fleet dev evaluation. Review timings from the same
image, kernels, parallel layout and length distribution. The tool checks the
recipe/corpus hash, not the truth of a timing report. Do not transfer a measured
LoRA speed or short-context cost to full 262k training. WEB evaluation is a
separate budget; reserve retries and confirmation capacity explicitly.

Budget selection walks the priority order, includes only resolved, inventoried
rows with matching costs that fit, and lists deferred rows with reasons. This
is a transparent greedy plan, not a globally optimal knapsack solver. It can
skip an expensive high-priority row to fit a later one; prerequisites still
require researcher review. Actual spend can exceed an estimate: a plan budget
is not a live scheduler enforcement mechanism. The eight-node concurrency cap
and idle-resource policy remain separate.

## Next research questions, not a Cartesian product

After these stages: refresh valid task coverage; test family-balanced weighting
against natural token weighting; compare teacher-only, self-only and mixed data
at several controlled sizes; try an application-held-out replication; examine
rank 256 if needed; refine LR around the winning region. Revisit compaction and
window extraction only with coverage accounting. Leave optimizer families,
quantization, exotic adapters and large automated grids until a simpler study
establishes a reliable task-success gain. Every added row must answer a question
that could change the next compute-allocation decision.
