# Fleet blackbox shared-atom lineage — 2026-09-24

## Outcome

All 33 pending QA candidates now have an exact, content-free lineage binding
from their immutable task version to their reviewed atom sources. None can yet
enter training or evaluation: the available evidence does not prove environment
startup, tool reachability, verifier execution, a finite outcome, and cleanup.

The read-only observation made no external mutation and no paid model call. Its
result is deliberately **33 lineage-bound, 0 runtime-qualified**.

Artifacts:

- [exact live lineage bindings](../configs/data/fleet-blackbox-qa33-live-lineage-20260924-v1.json)
- [transitive shared-atom census](../configs/data/fleet-blackbox-shared-atom-lineage-census-20260924-v1.json)
- [unchanged no-extension split](../configs/data/fleet-blackbox-lineage-safe-split-20260924-v2.json)
- [qualification decision](evidence/qwen38-study/2026-09-24-fleet-blackbox-qa33-lineage-qualification-v1.json)
- [corrected aggregate ceilings](evidence/qwen38-study/2026-09-24-fleet-blackbox-heldout-component-ceilings-v1.json)

Reproduce the four derived artifacts exactly with:

```sh
uv run python -m training.fleet_blackbox_atom_lineage_expansion
```

## How the grouping works

An atom is one reviewed vulnerability source. Its stable identity is the
Artifact Registry key, for example `cyber/atoms/<application>/<atom>`. The
version suffix is intentionally removed for grouping, so version 0 and version
3 of the same atom stay together.

Each task is connected to every atom it contains. Two tasks belong to the same
component when they share an atom, or when a chain of composite tasks connects
them. Exact versions under one task key also stay together even if later
metadata changes. This is a deterministic connected-component calculation; it
does not infer lineage from task names.

The combined census contains:

| Item | Count |
|---|---:|
| Receipt-proven task versions | 75 |
| Pending QA task versions with exact live lineage | 33 |
| Stable atom artifact keys | 104 |
| Shared-atom components | 101 |
| Candidate components | 26 |
| Candidate components touching an admitted component | 0 |

Most components contain one task version. One candidate component contains
three versions and one contains six. This is the concrete case the old
task-family-string grouping missed: several differently named tasks collapse
because their reviewed atom sources overlap transitively.

The implementation lives in
[`training/shared_atom_lineage.py`](../training/shared_atom_lineage.py). It
rejects malformed atom identities, duplicate exact task versions, and any
component that would bridge frozen train and held-out roles.

## What the 33-candidate observation proves

The read-only Fleet query selected each exact `(task_key, task_version_id)` and
retained only its reviewed atom and task-graph source metadata. It persisted no
task text, session content, model output, answer, or secret.

It proves that all 33 current versions still bind to valid reviewed lineage
metadata. It does **not** turn the QA labels `clean` or `agent_failure` into a
runtime certificate. Those labels are discovery signals, not proof that a
failure or success was genuine.

The exact teacher3k comparison adds one useful checkpoint-specific result:

- 12 candidates are the exact task versions present in the teacher3k corpus.
- Five more current versions were not exact corpus members, but their live atom
  lineage intersects atom lineage that teacher3k trained on.
- Therefore 17 candidates are teacher3k-exposed at the shared-atom level and 16
  have no shared-atom exposure in the bound teacher3k lineage.

This does not qualify any task. It only prevents a later evaluator from calling
an exposed task held out for that checkpoint.

## Corrected expansion ceilings

The 33 exact candidate versions are only 26 transitive components. The earlier
optimistic 108-family / 36-heldout projection assumed all 33 were distinct; the
correct proportional population is 101 components with 34 heldout. Separately,
36 remains the conditional Teacher3K leakage-only maximum: clean20 plus all 16
unexposed singleton candidates after complete zero-model qualification.

The full distinction between qualified-now, absolute lineage-only, and
representative ceilings is recorded in
[`FLEET_BLACKBOX_HELDOUT_COMPONENT_CEILINGS_2026-09-24.md`](FLEET_BLACKBOX_HELDOUT_COMPONENT_CEILINGS_2026-09-24.md).

## Split result

The admitted split remains exactly 50 train, 17 development, and 8 final test.
All 75 parent roles are unchanged, no admitted shared-atom component crosses
roles, and none of the 33 candidates was admitted.

The next qualification step requires an authorized zero-model runtime probe for
each candidate. A task may be promoted only after one exact receipt proves all
five missing facts: environment startup, both task tools, verifier execution, a
finite outcome, and cleanup. This work intentionally did not perform that
mutating step.
