# Fleet blackbox lineage-safe split design — 2026-09-24

## Outcome

A 100-task held-out set is **not supported by the current evidence**. The latest
catalog contains 1,217 production blackbox task versions, but only 75 have the
complete receipts needed to show that the environment ran, the verifier ran, a
finite outcome was recorded, and the environment was cleaned up. Those same 75
tasks already have frozen roles: 50 train, 17 development, and 8 final test.

The checked-in output therefore does two things without launching anything:

1. It reproduces the existing 75-family split under the latest Sep24
   qualification authority, preserving every role exactly.
2. It defines the evidence gates and minimum population required to grow the
   split to 200 train, 68 development, and 32 final-test families. That would
   provide the requested 100 held-out families while retaining the current
   two-to-one train-to-heldout ratio.

Machine-readable artifacts:

- [immutable role anchor](../configs/data/fleet-blackbox-lineage-role-anchor-20260924-v1.json)
- [current lineage-safe split](../configs/data/fleet-blackbox-lineage-safe-split-20260924-v1.json)
- [derivation and no-launch decision](evidence/qwen38-study/2026-09-24-fleet-blackbox-lineage-safe-split-design-v1.json)

The follow-up [shared-atom lineage census](FLEET_BLACKBOX_SHARED_ATOM_LINEAGE_2026-09-24.md)
implements the previously missing composite/successor closure. It binds all 33
pending candidates to exact live lineage, but qualifies none for admission, so
the 50/17/8 roles remain unchanged.

Reproduce all three exactly with:

```sh
uv run python -m training.fleet_blackbox_lineage_split_design
```

## What “lineage-safe” means

For the current 75 admitted tasks, the split unit is the reviewed application
plus `task_family`, not a task name. Every exact version with that same identity
must stay in one role. All 75 current rows are single-atom families, so this is
enough to preserve their frozen roles.

The v1 grouping is **not enough for expansion**. Two future composite tasks
could have different family strings while sharing an atom, and successor atom
versions could encode the same underlying vulnerability under different
strings. The follow-up v2 artifact now computes the required transitive closure
of shared reviewed atom identities. It remains no-launch because none of the
newly bound candidates has the complete runtime receipts required for
admission.

Exact qualification uses the pair `(task_key, task_version_id)`. A task key by
itself is not enough because it can acquire later versions. Family grouping then
uses the reviewed application and task-family identity. Application alone is
too broad; task key alone misses aliases.

For future training, the current split is safe if training reads only the 50
train families and never reads development or final-test data. Existing
checkpoints are different: each needs its own audit against the exact corpus it
actually saw.

## Why 1,217 tasks become only 75 admitted tasks

The Sep24 census partitions the catalog as follows:

| Evidence state | Task versions | Admitted to the split? |
|---|---:|---|
| Complete exact receipts | 75 | Yes |
| Known broken | 74 | No |
| QA looked clean, but complete receipts are missing | 8 | No |
| QA ended in agent failure, but complete receipts are missing | 25 | No |
| Not yet analyzed | 1,110 | No |

The 1,110 not-analyzed rows include the 75 already proven rows, leaving 1,035
not-analyzed rows without exact receipts. “Clean” or “agent failure” is useful
discovery information, but it does not prove that the task environment and
grading path are valid. Admitting those rows would turn infrastructure defects
into apparent model failures.

## Existing training lineage

The split design keeps two claims separate:

- **Prospective split:** if a new model trains only on the 50 train families,
  all 25 development/final families remain held out.
- **Existing checkpoint compatibility:** a checkpoint is evaluated only on
  families absent from its exact training corpus, including aliases.

The teacher3k corpus has the strongest completed alias audit. Of the original
25 held-out families, 20 are clean for checkpoints trained only from the bound
teacher3k manifests. Five were exposed through six alternate task keys, so they
must not be used for those checkpoints. The exact 20-family protocol remains
separate because this result is checkpoint-specific, not a new global split.

The fresh75 manifests are bound to the frozen train roles. The original Sep7
teacher run is not compatible with the current heldout by default: all 25
held-out task keys appeared in its 130-key candidate list, while the exact 86
families actually used were not retained. Several other historical corpora also
lack a complete public exact-family list. Consequently, **zero families are
certified as untouched across every historical Qwen training lineage**. That
does not mean every family was trained on; it means the stronger absence claim
cannot be proven.

## Balance of the current split

The frozen roles keep all 75 reviewed families disjoint and preserve the
original balance across application, environment, difficulty, and vulnerability
labels. The machine-readable split records the full counts and share gaps. The
application mix is especially close: the largest absolute difference from the
population share is about 3.5 percentage points in development, 3.2 points in
final test, and 0.7 points in train.

Rare vulnerability labels cannot appear in every partition because many occur
in only one family. The artifact names these cases rather than pretending they
are balanced. When new families are admitted, only new families are allocated;
existing roles never move.

## What is required for a 100-task heldout

Keeping the current proportions requires 300 distinct qualified families:

- 200 train
- 68 development
- 32 final test

That is 225 more qualified families than exist today. The original design also
recorded an explicit best case in which all 33 pending exact versions were
distinct families. The completed shared-atom census now shows that they are 26
transitive components. If all 26 components gain complete receipts, the
population would be 101 and the proportional heldout would be 34—still 66 short
of 100. Reaching 300 would then require 199 more distinct qualified components.

Before a new family can extend the split, it needs:

1. Exact environment, verifier, finite-outcome, and cleanup receipts.
2. An exact task key and task-version binding.
3. A reviewed application and task-family identity for every exact version.
4. Transitive grouping by shared reviewed atom identity, including composites
   and successor atom versions.
5. A family-level exposure check against each training corpus whose checkpoint
   will be evaluated.
6. Deterministic anchored assignment that does not move any existing family.

The 1,035 unproven rows are a large discovery pool, not an evaluation set. The
next useful action is to qualify one preregistered representative per transitive
component, then rerun the deterministic builder. The corrected current and
conditional ceilings are in
[`FLEET_BLACKBOX_HELDOUT_COMPONENT_CEILINGS_2026-09-24.md`](FLEET_BLACKBOX_HELDOUT_COMPONENT_CEILINGS_2026-09-24.md).
No cluster or evaluation job was launched by this design work.
