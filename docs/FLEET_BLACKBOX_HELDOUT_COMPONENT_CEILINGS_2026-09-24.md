# Fleet blackbox heldout component ceilings — 2026-09-24

## Decision

No task can be added to the current qualified heldout today. The 33 pending
exact task versions have reviewed lineage but zero complete runtime receipts.
They form 26 transitive stable-atom components, not 33 independent families.

The aggregate-only receipt is
[`2026-09-24-fleet-blackbox-heldout-component-ceilings-v1.json`](evidence/qwen38-study/2026-09-24-fleet-blackbox-heldout-component-ceilings-v1.json).
It contains no task key, version ID, component ID, prompt, trajectory, score, or
credential. Its logical SHA-256 is
`f2f4fb090aca2b39c0e3564298908a8e5dd7738947f37c17308e839c1861d966`
and its file SHA-256 is
`aa8315e011397a445e1be41aa32d5103fd3685eb86ef326a026497ecdd1025ec`.
It authorizes no launch or runtime qualification.

## Corrected ceilings

| Scope | Qualified now | Conditional after complete zero-model qualification |
|---|---:|---:|
| Teacher3K-compatible heldout | 20 components / 20 exact tasks | 36 components / 36 exact tasks |
| General lineage-only maximum | 25 components | 51 components; 58 exact tasks only if all aliases are retained |
| Representative anchored heldout | 25 components | 34 components |

The Teacher3K maximum is the existing clean20 plus the 16 pending singleton
components with no shared-atom exposure to the bound Teacher3K lineage. It is a
leakage-only maximum, not a representative split and not a globally untouched
claim.

The general maximum puts all 26 candidate components in heldout. That preserves
lineage disjointness for a checkpoint trained only on the frozen train role, but
it is not representative. A statistically independent roster must predeclare
one exact qualified representative per component. Keeping every exact alias
would produce 58 tasks across only 51 components, so those aliases cannot be
counted as 58 independent families.

For the representative construction, preserve all 75 inherited roles and use
the existing 50:17:8 ratios over the 101 known components. The deterministic
targets are 67 train, 23 development, and 11 final test. The 26 new components
therefore contribute 17 train, 6 development, and 3 final. To remain compatible
with Teacher3K, put all 10 exposed candidate components in train, then allocate
the 16 unexposed components as 7 train, 6 development, and 3 final. This yields:

- 34 general heldout components for a future checkpoint trained only on the 67
  train components; and
- 29 Teacher3K-compatible heldout components: the current clean20 plus nine new
  unexposed heldout components.

The exact role assignment remains blocked on qualification. Freeze it before
reading capability results, ignore prior `clean` versus `agent_failure` outcome
labels during allocation, and balance only reviewed application, environment,
difficulty, and vulnerability metadata with a fixed seed.

## Why the older proportional count changes

The earlier design recorded a conditional 108-family population and 36-family
heldout under the explicit assumption that all 33 candidates were distinct.
The completed transitive census disproves that assumption:

- 24 candidate components contain one exact version;
- one contains three; and
- one contains six.

The corrected population is therefore 75 + 26 = 101 components, whose
proportional heldout is 23 + 11 = 34. The number 36 still appears for a different
reason: it is the Teacher3K leakage-only maximum of clean20 plus 16 unexposed
singletons after qualification. It is not the corrected general proportional
count.

## Catalog exclusions

The 1,217 current exact versions partition without overlap:

| Evidence state | Exact versions | Current treatment |
|---|---:|---|
| Receipt-proven | 75 | Admitted under frozen roles |
| Known broken | 74 | Excluded |
| QA clean, complete runtime receipt missing | 8 | Excluded pending qualification |
| QA agent failure, complete runtime receipt missing | 25 | Excluded pending qualification |
| Not analyzed and no exact receipt/lineage | 1,035 | Discovery pool only |

The raw `not_analyzed` QA bucket is 1,110 because it also contains the 75
receipt-proven versions. The 1,035 unproven rows have no defensible family count
until exact lineage is bound; they do not contribute to any safe ceiling.

## Family and qualification rules

Selection always uses exact `(task_key, task_version_id)` identities. Leakage
grouping removes the atom version suffix, keeps every version of one task key
together, unions every atom in a composite, and takes the transitive closure.
No resulting component may cross train and heldout.

The 33 pending versions collapse to 26 components. Seventeen versions in 10
components are Teacher3K-exposed; 16 versions in 16 singleton components are
unexposed. Both multi-version components are exposed. None of the 26 candidate
components overlaps an admitted component.

Every admitted exact version still needs a model-free receipt proving environment
startup, `bash` and `submit_report` reachability, verifier execution, a finite
authoritative outcome, and cleanup. A historical QA label is not that receipt.
Promotion must remain outcome-blind.

A 100-component heldout at the same ratio requires 300 qualified components.
The current shortfall is 225; even after all 26 candidate components qualify it
is 199.

## Reproduction

Reproduce the census, unchanged no-extension split, qualification decision, and
this receipt with:

```sh
uv run python -m training.fleet_blackbox_atom_lineage_expansion
uv run pytest -q tests/test_fleet_blackbox_atom_lineage_expansion.py
```

This work made no API call, cluster mutation, model call, evaluation launch, or
runtime qualification.
