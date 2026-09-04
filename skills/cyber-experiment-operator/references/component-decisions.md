# Experiment component decisions

Use this reference when a request changes more than one of model, serving, harness,
dataset, benchmark, training objective, or execution capacity.

## Component ownership

| Component | Owns | Does not own |
|---|---|---|
| Model lock | exact weights, revision, tokenizer, chat template | endpoint capacity or agent tools |
| Serving profile | engine image/args, precision, quantization, context, route, resources | prompt, tool policy, task selection |
| Harness profile | agent implementation, system prompt, ordered tools, sampling and budgets | model weights or verifier |
| Dataset lock | exact task/version manifest, split and lineage | harness or serving behavior |
| Protocol | pass@k, seeds, timeout/retry, sealing and reporting | backend transport |
| Execution profile | API/backend, queue, priority, concurrency, output root | scientific treatment |
| Training profile | objective, trainer, optimizer, reward and checkpoint gates | held-out benchmark selection |

Changing a component creates a new compiled plan. Never edit a live or accepted plan.

## Harness selection

Use the harness that matches the intended deployed agent and can be reproduced for every
comparison arm. Prefer a pinned existing harness over a new adapter. Do not call two
harnesses equivalent because both expose tools with similar names; compare full prompt,
tool schema, context, sampling, budget, and retry identities.

For Fleet blackbox tasks, expose only the task-authorized tools in their declared order.
For WebExploitBench or ExploitGym, use the benchmark's pinned integration and control
image. Do not substitute the Fleet task runtime merely for convenience.

## Serving selection

Use an existing hosted endpoint for rapid descriptive evaluation when its exact-enough
identity is recorded and the scientific claim permits that opacity. Use dedicated serving
when exact weight/runtime control or additional throughput is required.

Never pool hosted and dedicated outcomes silently. To use both concurrently:

1. verify both satisfy the same harness-facing API and protocol;
2. allocate disjoint complete task groups before launch;
3. bind each session to one serving block;
4. report serving-block estimates and heterogeneity before any pooled estimate.

## Dataset and benchmark selection

- Fleet train-split tasks may support SFT/RL and training-distribution evaluation when
  lineage controls are frozen.
- Fleet dev/test and external benchmarks remain evaluation-only.
- Use matched base/post arms for causal lift. A stronger-model result can inform a
  separately declared difficulty study, but must not leak held-out task content into the
  target model's recipe.
- Keep invalid infrastructure outcomes separate from valid zeros.

## Throughput policy

Optimize accepted rollouts per minute, not raw process starts. Measure:

- claim-to-session and session-to-acceptance latency;
- accepted, persisted-but-excluded, failed, and active attempts;
- endpoint queue, token throughput, error rate, restarts, and readiness;
- controller CPU/memory/storage admission and cleanup latency.

Ramp concurrency only from score-blind health. Predeclare the ramp and backoff thresholds.
If acceptance failures dominate, fix that contract before adding model concurrency. Never
increase throughput by weakening duplicate checks, verifier evidence, cleanup, task tools,
or complete-task partitioning.

## Training progression

SFT requires a proven exact corpus, staged model bytes, trainer compatibility, and a
one-optimizer-step capability canary before a full run. RL additionally requires
authoritative positive reward under the exact task-facing tool contract and an accepted
reward-acquisition canary. A successful evaluation runtime is not by itself a training
runtime qualification.
