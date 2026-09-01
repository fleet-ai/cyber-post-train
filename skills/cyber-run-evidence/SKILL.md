---
name: cyber-run-evidence
description: Audit cyber training or evaluation jobs and produce evidence-backed status or terminal receipts without mutating workloads. Use for monitoring, diagnosing failures, reconciling API and Kubernetes state, or preparing incident handoffs.
---

# Cyber Run Evidence

Determine what actually happened while preserving the workload and the evidence needed to reproduce the conclusion.

## Default to read-only

Read the repository `AGENTS.md` and living evidence report. Resolve the exact run name, API run ID, Kubernetes kind/name/namespace/UID, requested image, runtime image ID, configuration digest, and expected durable output root before interpreting state.

Do not resubmit, delete, cancel, suspend, unsuspend, reprioritize, scale, patch, or exec a state-changing command while auditing. A user request for status or diagnosis does not authorize a fix.

Read [references/terminal-evidence.md](references/terminal-evidence.md) before declaring a run terminal or publishing a receipt.

## Reconcile independent evidence layers

- Query the durable Jobs API, Kubernetes controller object, controller-owned Pods, events, logs, metrics backend, checkpoint store, and run-output markers as independent sources.
- Bind observations to UIDs and timestamps. Names can be reused; selectors can resolve a replacement Pod.
- Capture runtime identity before TTL cleanup when possible. If an object is already gone, say which evidence is irrecoverable rather than reconstructing it from mutable state.
- Never print or persist credentials while using an authenticated observation path.

## Classify conservatively

- `pending` or `running`: controller and workload state agree and no terminal artifact exists.
- `operational gate`: infrastructure executed the intended path, but no capability claim follows.
- `valid model outcome`: the exact task and verifier completed under the frozen protocol.
- `infrastructure-invalid`: execution, identity, timeout, harness, verifier, or evidence integrity failed.
- `user-stopped`: the user confirms intentional termination; record that fact and do not speculate about a deletion actor.
- `unknown`: authoritative sources conflict and the conflict cannot be resolved.

An API row that says running while its UID-bound controller object is gone is stale evidence, not proof of either success or failure. A zero reward accompanied by truncation, missing report submission, or invalid verifier identity is not a valid capability zero.

## Report and persist

Lead with the outcome, then state what is running, what is not launched, the last verified transition, the next gate, and any user action required. Distinguish observed facts from inference.

Publish a terminal receipt only when the terminal classification and artifact identities are provable. Keep private traces and sensitive evidence in ignored or restricted storage; commit only sanitized digests and conclusions permitted by repository policy.
