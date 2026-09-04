# Dedicated GPU resource lifecycle

This policy keeps experiment-owned dedicated GPU serving and evaluation capacity aligned
with useful rollout work. It supplements the scientific protocol; it does not change task,
model, harness, verifier, retry, or scoring identity.

## Scope and authority

The policy applies only to dedicated GPU Jobs or Jobs API allocations owned by this
experiment. It does not apply to shared or hosted inference endpoints. Never delete, scale,
cancel, reprioritize, preempt, or otherwise mutate a peer workload.

A launch or execute request must explicitly authorize the complete owned-resource lifecycle,
including later drain and release. A status, audit, or diagnosis request remains read-only.
When release is needed but not authorized, report the idle owned allocation and request that
authority; do not reinterpret this policy as permission to delete it.

## Core invariant

Dedicated GPU capacity is justified only while at least one of these conditions is true:

1. an exact, reachable rollout consumer is producing or actively advancing valid attempts;
2. model loading or health validation is inside its immutable plan-bound allowance; or
3. a known next consumer is inside its immutable plan-bound handoff allowance.

The launch plan must bind deadlines for loading, warmup, consumer liveness, handoff, and
drain. Those values belong in versioned run configuration or receipts, not this living
policy. An unbounded "keep warm" state is invalid.

## Proving useful consumption

Use score-blind evidence from more than one layer. Bind observations to exact object UIDs and
record at least:

- the owned serving or evaluation allocation and its runtime identity;
- the consumer/controller Job and Pod identity plus treatment and serving-block binding;
- recent request, claim, heartbeat, or acceptance-progress counters and timestamps; and
- accelerator utilization, server readiness, errors, restarts, and endpoint reachability.

A Ready server, scheduled controller, open claim, or single instantaneous utilization sample
does not independently prove useful consumption. A consumer blocked on credentials,
packaging, storage, admission, acceptance, or another off-node repair is not useful merely
because its process remains alive. Preserve the observations in a sanitized lifecycle
receipt without prompts, traces, flags, scores, or credentials.

## Drain and release sequence

When no useful consumer or valid bounded exception remains:

1. Reconcile accepted, active, persisted-but-excluded, and infrastructure-invalid attempts.
2. Stop issuing new claims for the affected serving block. Do not interrupt unrelated blocks.
3. Let already-started attempts finish only within the plan-bound drain allowance. Preserve
   verifier, session, ingestion, cleanup, and retry evidence for each one.
4. Freeze a sanitized receipt covering the final consumer and allocation observations,
   outstanding evidence, and release decision.
5. Release the experiment-owned Job or Jobs API allocation promptly through its supported
   owner-only lifecycle. Confirm both API and Kubernetes terminal/absence state and that GPU
   capacity is no longer allocated.
6. Diagnose and correct the blocker off-node. Do not keep GPU servers warm while editing
   controllers, images, packages, manifests, or evidence rules.
7. After compatibility, parity, inventory, health, duplicate, and authorization gates pass,
   submit a newly named create-once successor. Never revive or mutate the old Job.

If the drain deadline expires, classify unfinished attempts from authoritative evidence under
the frozen retry policy. Resource release is an operational event, never a model score.

## Scientific controls

- Keep hosted and dedicated serving as explicit blocks. Bind every session to one block and
  report block-level estimates before any justified pooling.
- Apply the same lifecycle rule to matched comparison arms. Do not preferentially preserve
  capacity based on observed scores or benchmark content.
- Preserve valid outcomes and infrastructure-invalid classifications across successors. Never
  repeat an accepted or active cell merely because its original server was released.
- A successor must retain exact model, task version, harness, ordered tools, verifier, budgets,
  retry policy, and treatment identity unless it is declared as a new scientific block.
- Capacity changes may react to score-blind health and throughput evidence only.

## Handoff receipt

For each affected serving block, hand off the last observed allocation identity, consumer
identity and activity evidence, active exception and deadline, drain/release state, in-flight
attempt disposition, preserved receipt locations and digests, and the exact gates remaining
before a create-once successor may launch.
