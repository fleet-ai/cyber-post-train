# Fleet blackbox heldout16 qualification plan (2026-09-24)

## Decision

This change prepares an exact 16-task runtime check. It does **not** authorize or
launch it. The existing clean20 remains the only qualified Teacher3K-compatible
held-out set.

The 16 candidates are the complete conditional pool found by the held-out
expansion audit. Every candidate is an exact task version in its own transitive
shared-atom component, has no shared-atom overlap with Teacher3K, and has no
overlap with the admitted75 or its frozen train50. The sealed plan is
[`fleet-blackbox-heldout16-zero-model-wave-20260924-v1.json`](../configs/qualification/fleet-blackbox-heldout16-zero-model-wave-20260924-v1.json).

“Transitive shared-atom component” means all tasks connected through any shared
vulnerability atom. The component, rather than a task name alone, is the split
unit. This prevents an alternate task version or alias of the same vulnerability
from appearing in both training and evaluation.

## What the zero-model runtime check proves

No language model is called. For each exact task version, the existing task
quality controller must:

1. create the exact environment once;
2. prove the environment starts;
3. prove that the `bash` and `submit_report` tools are present, reachable, and
   accept the fixed probe schema;
4. send the fixed `no_flag` negative control through the real verifier;
5. prove that the verifier finishes with a finite authoritative result;
6. create one metadata-only evidence session; and
7. delete the exact environment and prove that its create claim and resources
   are gone.

A genuine verifier success or failure can both prove that the runtime and grader
work. The numeric result is not used to select tasks. An incomplete, ambiguous,
or unreachable cell is quarantined and is not retried automatically.

The first cell is the already reviewed QA33 canary
`ae236a07-128d-4556-ba2b-25a248818497`. The other 15 may run only after the
canary has a valid terminal receipt and a complete cleanup receipt.

## Outcome-blind split procedure

Before the first runtime mutation, all 16 components must receive an immutable
role:

| Role | Components | Meaning |
|---|---:|---|
| train | 7 | May become training candidates after qualification |
| dev | 6 | Held out for iterative evaluation |
| final | 3 | Held out for final confirmation |

The assignment uses `training.task_family_split.build` with the fixed seed
`fleet-blackbox-heldout16-outcome-blind-20260924-v1`. It balances application,
environment, reviewed difficulty, and vulnerability family. The algorithm may
read only those reviewed metadata fields and exact family identities. It may not
read the QA label, runtime result, model outcome, prompt, trace, answer, flag, or
score.

The current sealed lineage has application, difficulty, and vulnerability
family for all 16, but lacks a reviewed environment label for all 16. Therefore
the split is deliberately not frozen yet and the runtime wave remains blocked.
Once those labels are added, the deterministic split is generated and sealed
before any task is checked.

Roles never move after a result exists. If a preassigned task does not produce a
valid qualification receipt, it keeps its role but is not admitted. Another task
is not moved into its place based on the observed result. If all 16 qualify, the
held-out set grows by nine tasks, from 20 to 29.

## Daily budget and job safety

The 16 metadata-only evidence sessions count against the same 500-rollout daily
Fleet limit as evaluation sessions. Immediately before launch, the merged PR604
budget code (or an exact reviewed successor) must obtain a fresh authoritative
census and atomically reserve all 16 cells. A read-only budget check is not a
reservation. The launch fails closed unless:

`used today + existing durable reservations + 16 <= 500`.

The future Kubernetes wrapper is CPU-only and runs at `c1`. It must have zero GPU
requests and limits, no automatic retry, two identical server previews, and the
root annotation `fleet.ai/failure-alerts: "off"`. Cleanup is armed before the
single create and is bound to exact names, UIDs, and resource versions.

## Why it cannot launch yet

Launch remains unauthorized until all of the following are true:

- PR604 or its reviewed successor has merged the atomic daily-budget reservation
  contract;
- a generalized version of the QA33 job wrapper has merged and proves this exact
  16-task roster (the current wrapper is intentionally hard-coded to one canary);
- reviewed environment metadata exists for all 16;
- the deterministic 7/6/3 split is sealed before any qualification mutation;
- a fresh private controller plan is generated from the exact merged source;
- exact object and output paths are absent;
- two server previews agree and preserve `c1`, zero GPUs, and the root alert-off
  annotation; and
- exact UID-bound cleanup is armed.

No Kubernetes object, Fleet environment, evidence session, budget reservation,
or other external resource was created while preparing this plan.

## Training exclusion rule

Before any future SFT, self-SFT, preference, or RL optimizer update, every exact
training task version must be joined to the transitive shared-atom census. Any
component assigned `dev` or `final` by this plan must be rejected. Unknown or
unresolved training lineage is also rejected. This check must be repeated against
the final exact training roster; the current zero-overlap audit is necessary but
not a substitute for that launch-time check.
