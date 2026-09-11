---
name: cyber-train-operator
description: Operate authorized cyber SFT and RL runs with queue safety, immutable bindings, checkpoint gates, and reward-acquisition checks. Use for planning, launching, monitoring, exporting, staging, or serving a post-training run; not for benchmark scoring.
---

# Cyber Training Operator

Produce a reproducible training artifact without mistaking infrastructure activity for learning.

## Orient before acting

1. Read `AGENTS.md` and `README.md`, then the selected run's exact configuration and latest sanitized evidence. No model-specific historical report is global live state.
2. Fetch current `origin/main` and work in a clean, dedicated worktree. Preserve unrelated user artifacts and branches.
3. Separate immutable experiment identity from live scheduler state. A config or receipt proves intent; the API, Kubernetes objects, resolved image IDs, logs, and durable artifacts prove execution.
4. Identify whether the request authorizes planning, read-only monitoring, or an external mutation. Historical permission does not authorize a new paid launch, deployment, cancellation, or peer-workload change.

Read [references/gates.md](references/gates.md) before submitting training, exporting a checkpoint, staging weights, or registering a serving route.

## Operate through the supported rail

- Use the generic Jobs API through the repository CLI. Every new/changed job configuration must pass local and pinned-image CPU checks, then a bounded **dev-cluster** canary before production. `preview`/`submit` default to `--cluster dev`; production is explicit `--cluster prod`, not a kubecontext switch. Production is not a debugging target: preserve and release a failed dev allocation, fix it off-node, and qualify its successor on dev. Review the promotion gates and cluster-specific storage, secrets and topology in [`docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md`](../../docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md).
- Preview the rendered request, then explicitly submit once with a shared durable journal. Preserve the exact API origin in the intent. Use distinct prepared/output directories, run names and W&B identities per cluster; reconcile ambiguous POSTs instead of retrying or switching clusters with the same journal.
- Use normal admission and meaningful owner-specific names. Request `c1` when authorized or `c2` for backfill; under current policy `c1` must preview as the derived `q1` queue priority value 10,000. Recheck live policy rather than trusting the label. Never bypass admission, unsuspend manually, preempt, cancel, or modify another owner's workload.
- Resolve images to immutable digests and record both the requested image and runtime `imageID`. A tag, Ready Pod, or catalog row alone is insufficient evidence.
- For experiment-owned dedicated GPU serving, freeze consumer-liveness and lifecycle bounds before launch and follow [`docs/GPU_RESOURCE_LIFECYCLE.md`](../../docs/GPU_RESOURCE_LIFECYCLE.md). Drain and release idle owned capacity while diagnosing off-node; never apply that policy to shared or hosted endpoints.
- Keep credentials in environment or workload-secret injection. Never print, serialize, commit, or pass them as command arguments.
- Treat every retry as a new decision. Reuse only when the producer explicitly implements idempotent, create-only semantics and proves the existing artifact is identical.

## SFT gates

- Train only on verified-success demonstrations from the training split. Group task families across versions before selecting windows; report whether applications are shared or independently held out.
- Prove tokenizer, chat-template, windowing, target coverage, model revision, precision, topology, trainer image, dataset digest, and stopping rule before launch.
- Count supervised assistant tokens, not just window count. Mask copied context and tool observations; train each eligible target once per epoch. Record exclusions instead of silently truncating.
- For `task_outcomes_only`, bind a frozen Fleet development-outcome protocol and the exact source-selection/target-coverage policy through corpus, plan, checkpoint and recovery evidence. A verified-success receipt does not prove training-interface parity: block self traces unless the complete original model-request prefix, ordered tool schemas and calls are proven compatible with the target trainer.
- Require scalar-only W&B telemetry, regular recoverable checkpoints and a fixed idle/drain policy. Outcome-only W&B history contains training loss and optimizer step only; fresh Fleet development-task outcomes choose arms. Teacher-reference CE is neither required nor a tie-breaker, and WebExploitBench may run after an arm only as sealed external evaluation—never as HPO input. CPU checks do not qualify an untested CUDA kernel or distributed model loader.
- Full GLM and GLM Flash are different models. Do not claim full GLM training from a Flash recipe or a tiny CPU fixture; qualify the exact loader, resource shape and checkpoint resume.
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
