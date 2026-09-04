---
name: cyber-gpu-steward
description: Govern observed consumer-aware drain, release, and create-once replacement of experiment-owned dedicated GPU serving or evaluation capacity. Use for dedicated GPU lifecycle work; not for shared or hosted endpoints, peer workloads, or benchmark interpretation.
---

# Cyber GPU Steward

Keep experiment-owned dedicated GPU capacity tied to useful, scientifically controlled
rollout work.

## Orient

1. Read repository `AGENTS.md` and [the dedicated GPU lifecycle](../../docs/GPU_RESOURCE_LIFECYCLE.md).
2. Read `cyber-run-evidence` for UID-bound observation. Also read `cyber-train-operator`
   when serving or training capacity is involved, and `cyber-eval-parity` when a serving-block
   transition affects scored evaluation.
3. Determine whether the request authorizes the owned resource lifecycle or is read-only.
   Do not infer release authority from a monitoring or diagnosis request.
4. Confirm the allocation is experiment-owned dedicated capacity. Shared or hosted endpoints
   and peer workloads are out of scope.

## Decide from observed activity

- Bind the exact allocation, runtime, consumer, treatment, and serving-block identities.
- Combine consumer claims or heartbeats, request or acceptance progress, reachability, server
  health, and accelerator utilization. Do not treat Ready, scheduled, or one utilization
  sample as sufficient proof of useful work.
- Compare loading, warmup, liveness, handoff, and drain state against bounds frozen in the run
  plan. Never invent a new grace interval during an incident.
- Use score-blind operational evidence only. Never inspect benchmark answers or scores to make
  a capacity decision.

## Release idle owned capacity

When no useful consumer or valid bounded exception remains:

1. Reconcile attempt evidence and stop new claims for only that serving block.
2. Drain in-flight work within the frozen allowance and preserve authoritative acceptance,
   verifier, ingestion, cleanup, and retry evidence.
3. Write a sanitized lifecycle receipt, then release the owned allocation through its
   supported owner-only API when authorized.
4. Confirm API and Kubernetes release without touching peers. Diagnose and repair off-node.
5. Re-run compatibility, parity, inventory, health, duplicate, and authorization gates.
6. Submit a fresh create-once successor; never revive or mutate the old Job.

If release is required but not authorized, report the resource leak and request authority.
Resource release is operational evidence, not a capability result.

## Preserve scientific blocks

- Keep hosted and dedicated outcomes explicitly separated and every session bound to one
  serving block.
- Preserve valid, active, excluded, and infrastructure-invalid cells across a release. Never
  repeat a valid or active cell.
- Apply the same lifecycle rule to matched arms. A runtime change creates a separately recorded
  block unless exact parity is re-proven under the protocol.
- Do not change model, task, harness, tools, verifier, budgets, retries, or treatment identity
  merely to reclaim or replace capacity.

## Hand off

Report allocation and consumer identity, last useful activity evidence, active exception and
deadline, drain/release state, in-flight attempt disposition, frozen receipt digests, and the
next create-once successor gate. Never include prompts, traces, flags, scores, or credentials.
