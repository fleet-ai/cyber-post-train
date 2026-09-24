# Fleet blackbox heldout expansion audit — 2026-09-24

## Decision

The scientifically usable Teacher3K heldout set is still **20 task families**.
There are **zero additional tasks that can be admitted today** without weakening
the task-quality gate.

There is an exact conditional pool of **16 task versions in 16 independent
atom-lineage components**. None shares a reviewed atom lineage with the 32K,
64K, or 96K Teacher3K corpora; the planned 262K parent is a 27-task-key subset
of that same bound lineage. The 16 components also have zero overlap with all
75 currently admitted components, including the 50 frozen train components
available to prospective self-SFT.

Those 16 tasks are **not heldout tasks yet**. Each lacks the complete model-free
runtime receipt needed to prove that the environment starts, both required tools
are reachable, the verifier executes, a finite authoritative outcome is emitted,
and the environment and create claim are cleaned up. A prior QA label—whether
`clean` or `agent_failure`—does not prove those facts.

The exact machine-readable audit is
[`fleet-blackbox-heldout-expansion-audit-20260924-v1.json`](../configs/data/fleet-blackbox-heldout-expansion-audit-20260924-v1.json).
It authorizes no task, rollout, reservation, split change, or cluster action.

## Exact population

The current 1,217-version catalog partitions without overlap:

| Evidence state | Exact versions | Transitive components known? | Heldout treatment |
| --- | ---: | ---: | --- |
| Proven-good exact runtime receipts | 75 | 75 | 20 are Teacher3K-clean; 5 heldout-role tasks are Teacher3K-exposed; 50 are frozen train |
| QA-needed with reviewed lineage | 33 | 26 | 16 singleton components are Teacher3K-unexposed but still unqualified; 17 versions in 10 components are exposed |
| Known broken | 74 | No safe count | Excluded |
| Unresolved | 1,035 | No safe count | Discovery only |

“Proven-good” is operational: exact receipts prove the task and grading path ran
truthfully. It does not mean the model solved the task. “Agent failure” in the QA
pool likewise does not mean a valid model failure because complete runtime
receipts are missing.

## Conditional candidate pool

All 16 candidates are exact singletons, so no alias is being counted as an
independent family. The full task keys and version IDs are in the machine-readable
audit. This table uses the stable atom key as the readable family identity.

| Application | Prior QA label | Stable atom family |
| --- | --- | --- |
| concur | agent failure | `per-diem-quote-id-range-bypass` |
| concur | agent failure | `expense-report-fields-sqli` |
| concur | agent failure | `report-search-template-sqli` |
| concur | clean | `extract-job-status-missing-auth` |
| concur | agent failure | `extract-job-playground-leak` |
| current | agent failure | `checkout-csrf-key-hash-disclosure` |
| fakelook | agent failure | `directory-select-private-fields` |
| fanaplan | agent failure | `office-connect-origin-history-dump` |
| fanaplan | agent failure | `version-update-status-mass-assignment` |
| fcompliance | agent failure | `event-log-audit-export-unbounded` |
| fcompliance | agent failure | `customer-trust-favicon-url-forward` |
| fira | agent failure | `wiki-page-list-content-property-sqli` |
| fubspot | agent failure | `ticket-import-hs-ticket-id-sqli` |
| oracle_epm | agent failure | `job-application-existence-oracle` |
| oracle_epm | agent failure | `enable-features-setup-auth-skip` |
| oracle_epm | agent failure | `inbox-postponed-resume-unbounded` |

Strata before runtime qualification:

- QA label: 1 clean, 15 agent failures.
- Application: concur 5; oracle_epm 3; fanaplan 2; fcompliance 2;
  current, fakelook, fira, and fubspot 1 each.
- Transitive atom component: 16 distinct singleton components.

## What the possible sizes mean

Three different ceilings must not be conflated:

1. **20 now:** the only qualified Teacher3K-compatible heldout set.
2. **36 conditional leakage-only maximum:** clean20 plus all 16 candidates, but
   only after all 16 receive complete runtime qualification. This maximizes
   family count and is not a representative split.
3. **29 conditional representative set:** clean20 plus nine of the 16 candidates.
   The existing component-ratio plan would assign the 16 unexposed components as
   seven train, six development, and three final, yielding nine new heldout
   components. The exact nine cannot be frozen responsibly yet because runtime
   qualification is incomplete and the sealed sources do not contain the full
   reviewed difficulty and vulnerability metadata needed for an outcome-blind
   stratified assignment.

The audit therefore records all 16 exact candidates, but deliberately does not
pretend to know which nine are representative.

## Training-lineage checks

### Existing Teacher3K checkpoints

The 32K, 64K, and 96K checkpoints share the exact bound Teacher3K source
selection. Every conditional candidate has
`teacher3k_shared_atom_exposed=false` in the sealed transitive census. This is
an atom-lineage check, not merely a task-key comparison.

### Planned 262K SFT

The prepared 262K parent names 27 task keys. All 27 are present in the exact
496-key Teacher3K lineage map. Therefore a component absent from the full
Teacher3K lineage is also absent from this planned subset.

### Planned self-SFT

The prospective self-SFT roster has 50 frozen train components. The 16
conditional candidates overlap neither those 50 nor any of the 75 admitted
components. If a future self-SFT plan expands beyond the frozen 50, it must
first join its exact task versions to this component roster and keep any newly
reserved heldout component out of training.

### Planned RL

No immutable final RL training selection is merged on current main. The reviewed
phase-one signal wave is zero-update and its successor remains work in progress.
The currently observed successor roster uses existing admitted components and
does not overlap the 16 candidates, but mutable worktree state is not durable
evidence. Before any optimizer update, the final exact RL task versions must be
joined to transitive component IDs and rejected if any component has been
reserved for heldout.

This audit also caught a useful historical distinction: an older static RL
review used task `8095`, which belongs to the frozen development role. It was
acceptable only for a zero-update mechanics/signal check. The current successor
uses a frozen-train replacement. No learning run should use the development
task.

## Missing evidence and next gate

Expansion requires, in this order:

1. Complete zero-model runtime qualification for the 16 exact candidates.
2. Reviewed application, environment, difficulty, and vulnerability labels.
3. An outcome-blind, fixed-seed assignment that freezes either all 16 as a
   leakage-only heldout extension or exactly nine as the representative extension.
4. A final exact-lineage join against every SFT, self-SFT, preference, and RL
   training plan before those plans can run.
5. A new immutable split/protocol; the current split must not be silently edited.

The 74 broken and 1,035 unresolved versions remain excluded. In particular,
the unresolved pool has no defensible component count: task names are not a
safe substitute for reviewed atom lineage.

## Reproduction

```sh
uv run --locked --extra dev pytest -q \
  tests/test_fleet_blackbox_heldout_expansion_audit.py
```

The audit read no prompt, trace, flag, answer, or credential and made no external
mutation.
