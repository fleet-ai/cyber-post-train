---
name: cyber-eval-parity
description: Design and run matched base-versus-post-training cyber evaluations with sealed benchmarks and exact harness identity. Use for WebExploitBench, ExploitGym, or Fleet holdout evaluation; not for changing the training recipe.
---

# Cyber Evaluation Parity

Measure the checkpoint intervention rather than a difference in serving, harness, tasks, or retry behavior.

## Establish the comparison

1. Read the repository `AGENTS.md`, `docs/SCIENTIFIC_PROTOCOL.md`, the relevant evaluator README, and the current living evidence report.
2. Freeze a machine-readable protocol before scored work. Permit exactly the intended checkpoint or adapter bytes to differ between arms.
3. Bind model revision, tokenizer, chat template, precision, quantization, serving image and arguments, harness image and commit, system prompt, tools, task and verifier versions, sampling, budgets, seeds, concurrency, timeout, and retry policy.
4. Require a completed export receipt, serving-registration receipt, and live-parity receipt for the intervention. A model-list entry or Ready replica is not sufficient.
5. Treat a provider catalog name as availability, not checkpoint identity. Pool a
   hosted route with an exact-checkpoint treatment only when immutable evidence
   binds the model repository and revision, provider, endpoint route revision,
   harness image/version, context and compaction policy, tool schema, and exact
   task-version mode. A missing binding or a different provider creates a
   separate result block.

Read [references/paired-protocol.md](references/paired-protocol.md) before preparing or launching any scored arm.

## Preserve benchmark independence

- External benchmark prompts, applications, traces, outputs, scores, solutions, metadata, and failure analyses are evaluation-only. Do not use them for training, retrieval, prompt development, reward design, checkpoint selection, or stopping.
- Keep results sealed until the intervention checkpoint and protocol are frozen.
- If the historical control cannot be reproduced exactly, run a newly matched base and intervention pair. Do not compare across incompatible harnesses or serving formats.

## Benchmark-specific boundaries

- **WebExploitBench:** use the exact pinned CAGE and Qwen Code path defined by the protocol. Refuse an existing run root. Preserve valid zeros and follow the predeclared retry policy for infrastructure-invalid targets.
- **ExploitGym:** use one immutable linux/amd64 harness digest for both arms, the same dynamic graders and task images, single-use output roots, and counterbalanced arm order. Accept a run only after its terminal artifact and live Job UID agree.
- **Fleet holdout:** launch exact `eval_task_version_id` bindings, not mutable task keys or current versions. Preserve exact environment, data, verifier, model-route, harness, and session identities.

## Preflight local execution

- Inspect every local evaluator image before scored execution and require its
  normalized OCI architecture to equal the host architecture. Do not silently
  use cross-architecture emulation for a scored arm.
- Run the repository's deterministic treatment and architecture guards before
  accepting or pooling results. Keep concrete model IDs, revisions, route
  commits, and incident identities in configs or evidence receipts rather than
  this skill.

## Interpret results

- Report exact task-level outcomes, paired deltas, timeouts, infrastructure-invalid cases, and confidence intervals over tasks.
- Keep training-seed variance separate from evaluation-rollout variance.
- Do not silently rerun a valid pass@1 outcome. A retry must be allowed by the frozen symmetric policy and retain the original record.
- Do not call a plumbing smoke, successful registration, or verifier invocation a capability result.

Stop before launch if any arm identity, result root, live route, tool schema, budget, task version, verifier binding, or retry rule cannot be proven from authoritative evidence.
