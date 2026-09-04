---
name: cyber-experiment-operator
description: Compose, validate, and operate reproducible cyber evaluation or post-training experiments when models, harnesses, serving modes, datasets, or benchmarks must be swapped without changing unintended variables. Not for authoring vulnerability tasks or merely auditing one live job.
---

# Cyber Experiment Operator

Turn the user's scientific question into immutable components, then use the repository's
preview-first adapter for each backend. Preserve the distinction between experimental
intent, serving capacity, agent harness, task data, and execution transport.

## Route the work

- Use `cyber-eval-parity` for matched evaluation design and benchmark containment.
- Use `cyber-train-operator` for SFT/RL gates, model staging, serving, and checkpoints.
- Use `cyber-run-evidence` for live monitoring, failure diagnosis, and terminal receipts.
- Use `cyber-experiment-maintainer` when a repeated failure should become code, tests,
  documentation, or a focused skill.

Read [references/component-decisions.md](references/component-decisions.md) when selecting
or changing a model, harness, serving profile, dataset, benchmark, or concurrency policy.

## Compose the experiment

1. State the scientific comparison and the unit of analysis. Decide which variables may
   differ and which must remain identical.
2. Select exact model, serving, harness, dataset, protocol, execution, and—when
   applicable—training components. Resolve mutable names to immutable revisions and
   digests before launch.
3. Compile a machine-readable plan. Validate cross-component identity, task lineage,
   benchmark containment, resource/session totals, output roots, retry rules, and
   duplicate task-attempt ownership.
4. Preview through the authoritative backend. A valid local config does not replace a
   server render, live model parity probe, or duplicate inventory.
5. Launch only with current authorization and an explicit execute flag. Record the
   immutable plan, request, API/controller/Pod UIDs, queue, resources, and create-once
   claims.
6. Monitor score-blind health and evidence. Change concurrency only when the frozen
   policy permits it; never change tasks, tools, prompts, retry semantics, or serving
   identity to improve observed scores.
7. Accept only after terminal controller state, authoritative grading, cleanup, session
   ingestion, artifact digests, and plan counts reconcile.

## Non-obvious invariants

- A harness includes its version/image, system prompt, ordered tool schemas, context
  policy, step/token/time budgets, sampling, and retry behavior.
- Hosted and dedicated serving are different treatment blocks. They may run in parallel
  only on disjoint complete-task partitions and must be reported separately.
- Partition pass@k work at complete task boundaries. A persisted or active attempt is
  never repeated merely because the controller failed to accept it.
- External benchmarks are evaluation-only. Do not use their content or outcomes for
  training, reward design, prompts, retrieval, task selection, or checkpoint selection.
- Model names and Ready status are not exact identity. Bind the revision, weight,
  tokenizer, chat template, serving image/arguments, and live-parity evidence available
  for that route.
- Context must cover the declared agent horizon. If truncation occurs, classify the
  affected outcome under the frozen policy; do not silently enlarge context mid-arm.
- Prefer the repository's stable facade and typed component locks. Treat older
  model-specific scripts and manifests as implementation detail or historical evidence,
  not defaults to copy.

Stop before paid or scored execution if any component cannot be proven, the duplicate
inventory is incomplete, the treatment partition overlaps, or the backend would require
silently changing the protocol.
