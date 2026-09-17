---
name: literature-informed-experiments
description: Research closely related papers and open recipes, then generate a compact, prioritized, compute-budgeted SFT or RL experiment plan. Use for hyperparameter sweeps, data/method comparisons, research planning, or refreshing an experiment-plan page; not for launching or monitoring jobs.
---

# Literature-informed experiments

Turn evidence into a small sequence of decisions—not an exhaustive grid or a
list of settings that merely sound plausible. Announce this skill and explain
which planning artifact will change. Planning alone never authorizes paid jobs.

## 1. Establish the actual problem

Read the repository's operating instructions and scientific protocol. Inspect
exact model/config/data manifests and what the compiler actually supports.
Separate the user's desired capabilities from qualified implementation paths.

Write down model revision/architecture, task, available verified demonstrations,
supervised and context token counts, length distribution, data splits, tools,
context/compaction requirements, success metric and budget. Unknown counts stay
unknown. Do not substitute old task/session counts for a current inventory.
Distinguish concurrent-node limits from total GPU-hours and evaluation cost.

If no total budget is given, offer a small initial stage and conditional later
stages. Do not fabricate hourly costs or ask a blocking question when useful
read-only planning can proceed. Keep private content out of public artifacts.

## 2. Research by relevance, not popularity

Browse primary sources: papers, author model cards, versioned recipe code.
Search the exact model first, then nearby architectures/sizes, then the same
task and interaction style. Give closest task/model evidence more weight than
generic fine-tuning advice. Inspect methods/configurations, not just abstracts.

For each source record: URL/version/date, model size and initialization, data
type/amount, LR, effective batch, epochs, context, loss masking, optimizer and
schedule where available. Mark unreported fields unknown. Distinguish collected
trajectories from the retained training subset. Do not confuse 8B ablations with
the final 32B model card. Preserve disagreements rather than combining numbers
from incompatible versions.

Record three separate statements: **what was observed**, **what we infer**, and
**why it might not transfer**. A source must support the parameter choice, not
be a decorative citation. Keep source summaries short. Start with
[references/method.md](references/method.md) for reusable design checks and a
dated set of relevant starting references; reverify them for the current model.

Never import sealed evaluation prompts, task descriptions, solutions, traces,
scores or derived hints into research planning, training, rewards or this skill.
In cyber-post-train, Fleet dev selects recipes; external benchmarks report
transfer only after checkpoints/protocols are frozen. User goals about improving
a benchmark do not erase the distinction between development and final tests.

## 3. Design a staged study

1. Choose one credible, implementable anchor using broad valid training data.
2. Bracket important numerical uncertainty logarithmically. A several-fold LR
   change often teaches more than adjacent tiny values. If an edge wins, test
   beyond it; a boundary is not an optimum.
3. Compare meaningful data interventions early. Isolate source effects using
   matched families and per-family supervised-token budgets; keep an additional
   broad-data practical candidate instead of shrinking all training forever.
4. Test a small number of interactions that could reverse a conclusion (batch
   × LR; adapter rank × LR). Do not transfer full-training LR directly to LoRA.
5. Spend longer training on promising candidates. Compare independent horizons
   with their declared schedules; an early checkpoint in a long cosine run is
   not a short run with full decay. Do not stop on imitation loss alone.
6. Reserve compute for repeated training seeds and matched task evaluations.
   Tiny dev sets can make a single task look like a large percentage gain.

Keep 6–10 visible columns: run, data, method, LR, batch, duration, context and
stage are usually enough. Put unchanged settings, sources, hypothesis, control,
prerequisites and decision rules in expandable details. Define technical terms
at first use. Clearly label proposals versus historical completed experiments.

## 4. Make it reproducible and affordable

Use the repository's small deterministic generator if present (for example,
`scripts/plan-experiments.cjs`). Keep one source of truth for browser and CLI;
avoid a new scheduler/optimizer service. Emit structured research specifications
and a CSV view. Never disguise them as valid job requests.

Conditional rows require an explicit completed development-result choice.
Inherited settings must include data, method, adapter parameters and schedule,
not only LR. Unknown choices stay conditional. Reject unsupported options,
invalid counts, ambiguous budgets and stale recipe/cost hashes.

Calculate updates from the actual batching rule and frozen example counts.
Track unique supervised tokens separately from processed context tokens.
Rebuild/count examples after window-length or sampling changes. Do not silently
truncate early actions, train copied prefixes repeatedly, or count windows as
tasks. Batch is data-parallel replicas × microbatch × accumulation, not always
the total GPU count under tensor/context parallelism.

Budget with reviewed timings for the same runtime, hardware, dataset length
distribution and parallel layout, plus checkpoint/export/evaluation overhead.
State uncertainty, reserve retries and defer unknown-cost candidates. Adapters
need not save wall time; long-context activations can dominate memory. A planning
budget is not live spend enforcement. Keep launch authority and runtime gates
separate; hand off unsupported features to implementation work.

## 5. Test and deliver

Test deterministic output, invalid input, parent inheritance, corpus/count
changes, matched-data controls, budget exhaustion and unknown costs. Confirm
planning performs no network write or launch. Review the rendered table at
desktop and mobile widths, conditional settings, definitions and downloads.
Preserve unrelated live result pages when publishing a plan-only change.

Deliver the first recommended runs, why they come first, what evidence would
change the plan, actual readiness gaps, links and explicit launch status. If
publishing or shared-main merging is not authorized, prepare a reviewable branch
and ask rather than silently treating historical permission as current.
