# Terminal evidence checklist

## Identity

- Human-readable name and immutable API run ID
- Kubernetes kind, namespace, name, UID, owner references, and creation time
- Requested image and resolved runtime image ID
- Run config, dataset/task binding, code, and protocol digests
- Queue, priority, resources, worker/rank shape, and ownership labels

## Execution

- Controller terminal condition and timestamp
- Every relevant Pod UID, phase, exit code, termination reason, and restart count
- Last application heartbeat and metric step
- Counts of attempted, valid, truncated, and infrastructure-invalid episodes
- Exact verifier execution ID for every authoritative score when required
- Optimizer-step, loss/gradient, and checkpoint evidence kept as separate facts

## Artifact

- Durable output root and single-use lifecycle markers
- File count, byte count, manifest digest, and self-digesting receipt
- Tensor keys, shapes, dtypes, parameter count, shard hashes, and runtime sidecars for model artifacts
- Atomic no-replace or create-only publication proof

## Reconciliation rules

- Prefer UID-bound controller and artifact facts over a stale name-based API summary.
- Do not call a run successful because a checkpoint directory exists; it may precede later failure.
- Do not call a run failed solely because a controller was TTL-cleaned.
- If the user confirms a stop, record `user-stopped`, the last valid evidence, and any recoverable checkpoint; do not infer who issued the deletion.
- When sources disagree, preserve the conflicting projections and use `unknown` until a stronger immutable source resolves them.

## Minimal status handoff

State: outcome, exact last verified transition, what is currently running, what is not launched, next gate, blocker, user action if any, and evidence location. Never make the user infer whether an evaluation or optimizer step actually started.
