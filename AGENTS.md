# AGENTS.md — cyber-post-train

## Mission

Measure whether post-training the exact pinned open-weight model selected by
the experiment protocol on Fleet's authorized blackbox cyber tasks improves
exploitation capability on genuinely held-out tasks. The current primary model
is `Qwen/Qwen3.6-27B`; model identity belongs in versioned experiment config,
not in this repository-wide operating contract.

## Non-negotiable boundaries

1. WebExploitBench is evaluation-only. Never copy its prompts, applications,
   traces, outputs, metadata, solutions, embeddings or derived artifacts into
   training, retrieval, prompt development or reward development.
2. Never commit, print, serialize or forward credentials. Read them only from
   environment variables or the cluster's secret manager.
3. Run offensive workloads only against explicitly authorized, isolated
   challenge environments.
4. External benchmark results stay sealed until the post-training checkpoint
   and evaluation protocol are frozen.
5. Split Fleet data by application, vulnerability family and task lineage.
   Sessions from one lineage may not cross train/dev/test boundaries.
6. Every result must bind the exact model revision, tokenizer, chat template,
   task revision, prompt revision, harness image and verifier revision.

## Engineering principles

- Prefer small composable CLIs and typed immutable manifests.
- Default every mutating or paid operation to dry-run.
- Make resume/idempotency explicit; a retry must not duplicate paid jobs.
- Store append-only JSONL events and compact summary JSON separately.
- Keep provider adapters separate from benchmark and training policy.
- Unit-test payload construction, split leakage checks and reward parsing.
