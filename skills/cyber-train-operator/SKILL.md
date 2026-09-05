---
name: cyber-train-operator
description: Operate authorized cyber SFT and RL runs with queue safety, immutable bindings, checkpoint gates, and reward-acquisition checks. Use for planning, launching, monitoring, exporting, staging, or serving a post-training run; not for benchmark scoring.
---

# Cyber Training Operator

Produce a reproducible training artifact without mistaking infrastructure activity for learning.

## Orient before acting

1. Read the repository `AGENTS.md`, then `docs/QWEN36_STUDY_EVIDENCE.md` for the living state and `training/README.md` for the supported launch rails.
2. Fetch current `origin/main` and work in a clean, dedicated worktree. Preserve unrelated user artifacts and branches.
3. Separate immutable experiment identity from live scheduler state. A config or receipt proves intent; the API, Kubernetes objects, resolved image IDs, logs, and durable artifacts prove execution.
4. Identify whether the request authorizes planning, read-only monitoring, or an external mutation. Historical permission does not authorize a new paid launch, deployment, cancellation, or peer-workload change.

Read [references/gates.md](references/gates.md) before submitting training, exporting a checkpoint, staging weights, or registering a serving route.

## Operate through the supported rail

- Prefer the typed Fleet Jobs API and repository CLIs. Preview is the default; submission requires an explicit execute path and a duplicate-title check.
- Distinguish the general Nebius cluster API (`POST https://api.ft.flt.build/v1/runs`) from Fleet's managed evaluation API (`POST https://orchestrator.fleetai.com/v1/jobs`). Read [`../cyber-cluster-jobs-operator/SKILL.md`](../cyber-cluster-jobs-operator/SKILL.md) before using the former. A narrow local SFT/RL wrapper is not evidence that the deployed cluster API is model-catalog limited.
- Use the configured namespace, normal queue, default priority, and a meaningful `chris-cyber-*` name. Never bypass admission, unsuspend manually, preempt, cancel, or modify another owner's workload.
- Resolve images to immutable digests and record both the requested image and runtime `imageID`. A tag, Ready Pod, or catalog row alone is insufficient evidence.
- Keep credentials in environment or workload-secret injection. Never print, serialize, commit, or pass them as command arguments.
- Treat every retry as a new decision. Reuse only when the producer explicitly implements idempotent, create-only semantics and proves the existing artifact is identical.

## SFT gates

- Train only on verified-success demonstrations from the training split. Split by application, vulnerability family, and task lineage before selecting windows.
- Prove tokenizer, chat-template, windowing, target coverage, model revision, precision, topology, trainer image, dataset digest, and stopping rule before launch.
- At terminal state, bind the exact run UID, optimizer steps, metrics, and checkpoint files. Export without adding optimizer steps.
- Do not relabel checkpoint precision. Inspect tensor headers and perform a deterministic conversion when required.
- Stage weights with exact base runtime sidecars, complete hashes, create-only or atomic no-replace publication, and an embedded acceptance receipt.
- Register serving only from the accepted staged artifact. Require completion and live-parity receipts before evaluation.

## RL gates

- Bind exact task-version, environment-version, verifier-version, prompt, and ordered task-tool identities. Never resolve mutable `current` pointers at launch.
- Prove the live preview renders the intended optimizer-step limit; do not assume an API field named `max_steps` maps to optimizer steps.
- Acquire reward before scaling. Run a small version-bound canary that demonstrates real rollouts, authoritative verifier execution IDs, at least one optimizer step, and checkpoint production.
- Choose interaction horizon from successful reference traces and the student model's needs. Record prompt, trajectory, per-turn, tool-result, and turn budgets separately.
- If attempts truncate before report submission, treat zero reward as harness/configuration evidence, not model incapability.
- Expose only the intended task tools and enforce them at execution, not merely in the prompt. For Fleet blackbox exploit tasks the canonical surface is ordered `bash`, `submit_report` when authoritative task metadata proves it.
- Preserve one immutable verifier execution ID per scored episode. Missing, invalid, or version-mismatched IDs are infrastructure-invalid, not zero-reward model outcomes.

## Stop conditions

Stop before a paid or mutating action when any required identity is unresolved, preview differs from the frozen plan, task tools are inferred rather than authoritative, reward evidence cannot be traced, a destination already exists without identical idempotency proof, or the action would alter peer work. Report the exact blocker and continue with safe independent work.
