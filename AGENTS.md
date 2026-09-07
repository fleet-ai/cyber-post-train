# AGENTS.md — cyber-post-train

## Mission

Measure whether post-training an exact pinned open-weight model on Fleet's authorized blackbox cyber tasks improves exploitation capability on genuinely held-out tasks. Model identity belongs in versioned experiment config, not this repository-wide contract.

## Read and route

Read this file before acting. Then select only the skill matching the work and read its complete `SKILL.md`:

| Work | Skill |
|---|---|
| Plan, launch, monitor, export, stage, or serve SFT/RL | [`cyber-train-operator`](skills/cyber-train-operator/SKILL.md) |
| Design, launch, or interpret matched evaluations | [`cyber-eval-parity`](skills/cyber-eval-parity/SKILL.md) |
| Diagnose a job or assemble status/terminal evidence | [`cyber-run-evidence`](skills/cyber-run-evidence/SKILL.md) |
| Convert a lesson into code, tests, docs, or agent guidance | [`cyber-experiment-maintainer`](skills/cyber-experiment-maintainer/SKILL.md) |
| Compose or change a model × harness × data × benchmark experiment | [`cyber-experiment-operator`](skills/cyber-experiment-operator/SKILL.md) |
| Observe, drain, release, or replace experiment-owned dedicated GPU capacity | [`cyber-gpu-steward`](skills/cyber-gpu-steward/SKILL.md) |

Before creating any Kubernetes Job, RayJob, or persistent inference service on the
Nebius training cluster, read
[`docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md`](docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md).
Normal experiment operation must not manufacture failed-Job pages or hold unconsumed
GPUs. Do not evade or suppress a genuine platform alert; choose the supported resource
lifecycle and make expected terminal states clean and explicit.

For current experiment state, read `docs/QWEN36_STUDY_EVIDENCE.md`. For scientific controls, read `docs/SCIENTIFIC_PROTOCOL.md`. Chronological status notes and early example configs are context, not authority.

## Source-of-truth order

When sources disagree, use this order and preserve the disagreement:

1. Immutable accepted artifacts and self-digesting receipts
2. Exact API/Kubernetes object UIDs, runtime image IDs, logs, and terminal conditions
3. Versioned configs, manifests, code, and tests
4. The living evidence report at its stated observation time
5. Narrative status documents and chat summaries

Mutable names, tags, catalog `current` pointers, a Ready Pod, or a dashboard label do not establish exact identity by themselves.

## Non-negotiable boundaries

1. WebExploitBench and other external benchmarks are evaluation-only. Never copy their prompts, applications, traces, outputs, metadata, solutions, embeddings, or derived exploit hints into training, retrieval, prompt development, reward development, checkpoint selection, or skills. Accepted aggregate scores and task-level outcomes may enter sealed evaluation receipts and result reports after unsealing; they must not influence the frozen recipe or checkpoint.
2. Never commit, print, serialize, forward, or place credentials in command arguments. Read them only from environment variables or the cluster secret manager.
3. Run offensive workloads only against explicitly authorized isolated challenge environments.
4. Keep external results sealed until the checkpoint and evaluation protocol are frozen.
5. Split Fleet data by application, vulnerability family, and task lineage. Sessions from one lineage may not cross train/dev/test boundaries.
6. Bind every result to the exact checkpoint, tokenizer, chat template, task, environment, prompt, harness, tool schema, verifier, image, budgets, and retry policy required by its protocol.

## External-state discipline

- Status, diagnosis, and review requests are read-only. Do not infer permission to fix, deploy, submit, rerun, cancel, or mutate.
- Default mutating or paid operations to preview. Recheck duplicates, exact identity, cost/session count, queue, and stop conditions immediately before execution.
- Use meaningful `chris-cyber-*` ownership names, normal queues, and default priority. Never bypass admission, unsuspend manually, cancel, preempt, or change peer workloads.
- Do not mutate an immutable failed Job to retry it. Preserve it and create a reviewed successor only when authorized.
- An experiment-owned dedicated GPU serving or evaluation allocation needs an observed useful rollout consumer or a bounded, predeclared loading, warmup, or handoff exception. When neither holds, stop new claims, preserve in-flight evidence, release the owned allocation through its supported lifecycle, diagnose off-node, and create a fresh successor only after its gates pass. This does not apply to shared or hosted endpoints and never authorizes peer-workload mutation. Follow [`docs/GPU_RESOURCE_LIFECYCLE.md`](docs/GPU_RESOURCE_LIFECYCLE.md).
- Merge authority follows the user's current instruction and repository ownership. Historical approval is not permanent authorization for a new shared-repository merge or deployment.

## Scientific interpretation

Use precise labels:

- **operational gate** — plumbing executed; no capability claim
- **valid model outcome** — exact protocol and authoritative grading completed
- **infrastructure-invalid** — execution or evidence contract failed
- **user-stopped** — the user confirms intentional termination
- **unknown** — authoritative sources conflict without resolution

Do not interpret all-zero reward as model incapability when trajectories truncate, tools differ, report submission is unreachable, or verifier identity is missing. Demonstrate reward acquisition in a small exact-version canary before scaling RL. Do not silently rerun a valid pass@1 evaluation outcome.

## Engineering and worktree discipline

- Fetch current `origin/main` and use a dedicated clean worktree for each coherent change. Preserve unrelated user files and untracked artifacts.
- Prefer small typed CLIs, immutable manifests, atomic no-replace publication, and explicit idempotency.
- Validate rendered or API-returned behavior rather than source text alone. Add cross-field invariants where one artifact identity appears in multiple stages.
- Rebase or update before final tests, then review the final diff again. Keep one logical change per PR.
- Store private/raw evidence only in ignored or restricted locations. Commit sanitized receipts and digests only when policy permits.

## Learning loop

For every material failure, ask whether a durable correction belongs in code, a regression test, `AGENTS.md`, a focused skill, a receipt/config, or narrative docs. Prefer deterministic enforcement in code. Add skill guidance only for stable decisions that recur across runs; never hardcode current job IDs, secrets, mutable state, or benchmark content into a skill.

## Handoff standard

Lead with the outcome. State what is running, what has not launched, the last verified transition, the next gate, blockers, and any user action required. Link exact PRs/jobs and evidence paths where useful. Never make the reader infer whether an optimizer step, scored evaluation, serving route, or artifact publication actually occurred.
