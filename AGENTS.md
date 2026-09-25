# Working in cyber-post-train

## Purpose and authority

Measure whether post-training on Fleet's authorized blackbox cyber tasks
improves performance on genuinely held-out tasks. Current research focus is
Qwen; do not start GLM work without a new request. Training loss, a running
job, and an ungraded rollout are not evidence of capability lift.

This repository was reset in September 2026 and currently has no live job
launcher or evaluation service. These instructions constrain future tooling
and separately authorized operations; they do not authorize a new launch,
deployment, cancellation, preemption, or paid evaluation. For status,
diagnosis, or review requests, inspect only.

## Resource and cluster rules

- Use supported **c1** priority for new project-owned cluster jobs; never use
  c0 unless Chris explicitly changes this rule. Verify the effective priority
  in the rendered Job/Workload, not just a request field or label.
- Use at most **8 active GPU nodes (64 GPUs)** across project-owned cluster
  work, and no more than **10 queued jobs**. Count all concurrent agents and
  automations from live cluster state, including dedicated evaluation and
  serving allocations; exclude an existing endpoint only when a current,
  explicit exception identifies it. A queued job holds no GPUs, but still
  occupies a queue slot. Never use a stale chat count as the capacity ledger.
- Use at most **100 concurrent TensorLake executions** across the project.
  Count running work and outstanding launch reservations across all agents
  before starting more. Limit Fleet task-evaluation launches to **500
  rollouts per day** unless Chris changes that allowance.
- Keep available capacity productive with valuable, nonduplicate work, but
  never launch filler work just to hit a quota. Check existing experiments,
  jobs, and evaluations before creating equivalents.
- The Nebius dev cluster may be used for bounded debugging, but dev testing
  is **optional**, not a mandatory gate before production. Release dev test
  workloads, model servers, and their allocations as soon as the test ends;
  do not leave ongoing hosting there.
- Never bypass admission or mutate a peer's workload based on an old general
  permission. A new request to preempt must identify the current targets and
  authority. Failed, stopped, or idle project-owned GPU allocations must be
  reconciled by exact identity and released promptly through their supported
  lifecycle; preserve failure evidence first.

For any Fleet Kubernetes `Job` or `RayJob` created for Chris, the rendered
root object **must** contain this exact annotation before submission:

```yaml
metadata:
  annotations:
    fleet.ai/failure-alerts: "off"
```

If a launcher creates that object indirectly, verify the server-rendered
preview. A request flag, Pod annotation, or patch after creation is not enough;
fail closed if the launcher cannot prove the root annotation. This suppresses
failed-job notifications only; it does not excuse idle GPUs, hide failure
records, or relax cleanup. There is **no numeric failed-job budget**. After a
failure, preserve evidence, release owned resources, diagnose the actual
boundary, and repair deterministically before a successor.

Text instructions are not resource enforcement. Any new launcher must check
these limits and the rendered alert/priority fields at the submission boundary;
shared reservations or admission control must prevent two agents from each
consuming the last available slot.

## Scientific and data rules

- Run offensive work only against explicitly authorized, isolated challenge
  environments. Never use external benchmark tasks, traces, answers, or
  derived hints for training, reward design, prompt development, retrieval,
  or checkpoint selection. Keep benchmark results sealed until the checkpoint
  and evaluation protocol are frozen.
- Split by reviewed task family, keeping every version and session of one
  family together. Keep development and test tasks out of SFT, preferences,
  and RL. Aim for representative train/test coverage; do not call a
  family-held-out split application-held-out. The split helper cannot certify
  task quality or family lineage on its own.
- Compare baseline and trained checkpoints with the same task set, harness,
  budgets, scoring, and retry policy. Record exact model/checkpoint, tokenizer,
  prompt, tool schema, environment, verifier, image, and protocol identities.
  An infrastructure-invalid result is not a model failure, and a valid result
  must not be silently rerun because it is inconvenient. Evaluate selected
  checkpoints against both Fleet held-out tasks and external benchmarks, with
  matching baseline controls. Use **pass@4** for new campaigns unless a frozen
  experiment protocol specifies otherwise.
- For Qwen WebExploitBench comparisons, use the **OpenCode** harness, not Qwen
  Code. Preserve rollouts independently of scoring; score stored rollouts with
  GPT as the judge, including as new rollouts arrive. Do not claim a lift from
  the older tool-interface-defective comparison.
- Track every new training run with W&B and durable checkpoint/step evidence.
  Set a checkpoint cadence that bounds lost work, and verify reloadability
  before treating a checkpoint as an evaluation candidate. Do not infer a
  successful optimizer update from a Ready Pod or a training-loss chart alone.
- For long-horizon RL, validate context handling and compaction before scale;
  do not treat context overflow, truncated trajectories, broken tools, or
  unreachable graders as genuine zero reward. Prove reward acquisition and a
  finite learning update in a bounded canary before a full run.

## Repository and handoff discipline

Keep this repository small and understandable. `main` is the one persistent
local and remote branch. Remove temporary checkouts after use. Every change
must leave fewer than 10,000 tracked text lines; run
`python3 scripts/check_size.py` and the relevant tests.

The retained JSON files are historical observations. Do not turn a dated
checkpoint path, task count, or candidate qualification into a claim about
current availability or scientific validity without checking the source system.
Group task versions by reviewed application and task family before splitting;
the split helper alone does not establish that lineage.

Keep scripts focused, standard-library-only when practical, and tested at
their actual input/output boundary. Remove obsolete versions instead of
stacking wrappers. Do not add generated outputs, virtual environments, raw
prompts/traces/answers/flags, weights, or credentials to Git. Do not print or
forward secrets or sealed benchmark content in chats, logs, or command lines;
use the approved secret store or environment instead. For live status, prefer
exact API and Kubernetes identities and accepted receipts over mutable names,
dashboard labels, or narrative summaries; preserve any disagreement.

Before paid or mutating work, preview the exact operation and recheck
duplicates, budget, identity, and stop conditions. Prefer minimal code,
targeted diagnostic tests, and rendered/API-returned checks over guesses or
source-text checks. Save durable lessons in code, regression tests, or focused
instructions rather than repeatedly rediscovering the same failure. In a
handoff, distinguish what actually completed from what is queued, running,
failed, unscored, or only planned, with exact evidence where available.
