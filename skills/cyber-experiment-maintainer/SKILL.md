---
name: cyber-experiment-maintainer
description: Turn recurring cyber experiment failures and operational lessons into durable code, tests, instructions, and focused skills. Use when maintaining the experiment system or preventing configuration and evidence drift; not for live job operations.
---

# Cyber Experiment Maintainer

Make the next run safer and easier without turning one incident into brittle universal policy.

## Start from evidence

Read the repository `AGENTS.md`, the relevant terminal or incident receipt, affected code and tests, and current `origin/main`. Reproduce the failure at the narrowest deterministic boundary before changing behavior.

Read [references/learning-loop.md](references/learning-loop.md) when deciding where a lesson belongs.

## Choose the durable layer

- Put an invariant in **code** when it can be checked deterministically at preview, submission, rendering, publication, or acceptance.
- Add a **regression test** that fails for the observed defect and verifies behavior through the real boundary when practical.
- Put always-applicable repository authority, safety, and routing rules in **AGENTS.md**.
- Put specialized, reusable decision guidance in a focused **skill** with a discriminating description.
- Put exact model revisions, task IDs, job UIDs, timestamps, hashes, and current results in immutable **configs and receipts**, not skills.
- Put explanation and historical context in **docs**, clearly distinguishing living state from frozen evidence.

Prefer cross-field and rendered-output invariants over tests that match prose or duplicate one literal value. Examples include proving a destination agrees across plan, receipt, staging input, and serving registration; or proving environment-variable names are unique after the real chart renders.

## Keep changes scoped

- Work from current main in a dedicated worktree. Rebase or update before final validation and review the resulting diff again.
- Make one coherent change per PR. Preserve unrelated user work and do not rewrite historical evidence.
- Fix repository-owned mechanisms in this repository. Upstream only the smallest generally shared platform change required, with focused tests and current merge authorization.
- Default live or paid operations to preview. A code fix does not authorize deployment, data mutation, or workload resubmission.
- Do not encode credentials, private traces, hidden benchmark data, or mutable external state.

## Complete the learning loop

After the fix passes, update the living evidence or handoff only when its claims changed. Update a skill only if the lesson will alter future decisions across multiple runs. Remove or replace stale guidance rather than stacking contradictory rules.

When a lesson applies across runs, update
[`docs/OPERATIONAL_LESSONS.md`](../../docs/OPERATIONAL_LESSONS.md) with the
failure class, durable decision, enforcing code, and meaningful regression.
Keep exact workload identities and mutable observations in sanitized receipts,
not in the prevention map.

Use this incident-closeout gate before scaling or handing off the affected path:

1. Preserve sanitized, immutable evidence of what failed and whether any paid or scored side effect occurred.
2. Name the narrowest deterministic boundary that should have rejected or detected it earlier.
3. Put the prevention at that boundary and add a regression through the rendered or API-facing behavior when practical.
4. Update living operator guidance only for a stable cross-run decision; keep mutable identities and one-run facts in receipts.
5. Re-run focused tests and the relevant end-to-end preview, then state any remaining unenforced risk explicitly.

A successful successor does not by itself close the incident. If no durable
prevention exists yet, keep the related scale gate closed or record the residual
risk explicitly rather than relying on chat memory.

Before handoff, validate every skill package, run focused and relevant repository tests, check formatting and the final diff, and state which failures were pre-existing or out of scope. Report material residual risk rather than claiming that tests prove more than they do.
