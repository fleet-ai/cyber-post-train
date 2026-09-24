# Qwen3.8 self-SFT full-50 pass@8 successor plan

This is a plan, not permission to launch or reserve capacity.

The plan fixes one exact current task version from each of the 50 approved
training components. The 17 development and eight final-test components remain
excluded. Each training component has eight fresh attempts, using seeds 51
through 58, for 400 sessions in total. Those seeds do not reuse the retired
43-through-50 cell universe.

The first five components form a representative 40-session operational
canary. They were selected deterministically to cover five applications. If
that canary is accepted, its 40 sessions count toward the 400; they are not run
again. The remaining stage is the disjoint 45-component, 360-session
complement.

## Order of operations

1. The separate one-cell, zero-model task-quality canary must finish cleanly
   and all of its resources must be released.
2. On the target UTC day, the operator must atomically reserve exactly 400
   sessions under the canonical shared daily lock. The reservation must count
   every supported reservation schema and keep the total at or below 500.
3. The five-component pass@8 canary may run only after all fresh pre-create
   gates pass. Expansion requires all 40 cells to finish without ambiguous or
   infrastructure-invalid outcomes, full cleanup, and at least one verified
   successful trajectory that is suitable for the visible-action corpus.
4. Only then may the remaining 360 cells run. A failed canary does not permit
   the reserved capacity to be repurposed or the cells to be replayed.

Immediately before any create, the operator must also recheck every exact task
and runtime binding; prove that no matching cell, active job, queued intent,
accepted trajectory, database, ledger, or output already exists; obtain two
identical server previews; and arm exact-UID cleanup. Any cluster root must be
a zero-GPU `Job` at c1 with top-level
`fleet.ai/failure-alerts: "off"`, backoff zero, restart `Never`, and at most one
create attempt.

The checked-in plan remains closed because the one-cell receipt, 40-cell
canary receipt, daily reservation, current duplicate census, fresh absence
checks, previews, packet set, reviewed pass@8 runtime, dedicated reservation
writer, and cleanup handoff do not yet exist. The existing worker is fixed to
pass@4 and 200 cells, so it cannot execute this plan. The historical Wave A is
consumed and non-relaunchable, and its zero reusable successes do not open this
successor.
