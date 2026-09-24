# Qwen3.8 self-SFT blackbox task catalog — 2026-09-24

## Outcome

The current Fleet OTS Cyber catalog contains 1,217 exact production blackbox
task versions. The maximum safe prospective train-only roster is **50 exact
versions in 50 transitive atom-lineage components**. Another 25 components are
frozen development or final-test families and must remain held out.

This is catalog-level eligibility for a future Qwen pass@8 collection, not a
rollout receipt and not session-level SFT admission. A fresh exact task/runtime
binding and availability check is still required immediately before collection;
each resulting session must then pass the existing authoritative-success,
visible-action, compaction, and corpus-admission gates.

The machine-readable inventory is
[`qwen38-self-sft-blackbox-task-catalog-20260924-v1.json`](../configs/data/qwen38-self-sft-blackbox-task-catalog-20260924-v1.json).

| Classification | Exact versions | Transitive components | Current use |
| --- | ---: | ---: | --- |
| Immediately train-eligible | 50 | 50 | Prospective train-only pass@8 collection after JIT freshness |
| Heldout-family overlap | 25 | 25 | Development/final evaluation only |
| Known broken | 74 | unresolved | Excluded |
| QA needed | 33 | 26 | Zero-model runtime qualification only |
| Unanalyzed | 1,035 | unresolved | Discovery pool only |

The categories are mutually exclusive and cover all 1,217 exact
`(task_key, task_version_id)` pairs. `task_id` is retained only as a
supplemental catalog identity.

## Evidence refresh

The read-only refresh observed Fleet team
`a1025f0b-ad67-49fc-a023-51800ab43e84` and reconciled three independent
authorities:

- The exact OTS Cyber project/catalog census still byte-matches the sealed
  1,217-row inventory at
  `sha256:3de1271d11769989a4d80b0e329a53744642402a2754544eec1851db7322efe7`;
  all 1,217 exact current versions still have an attached verifier.
- The safe session-summary route still resolves all 75 receipt-proven exact
  versions to 42 proven successes and 33 genuine model failures with their
  recorded verifier executions. Only aggregate outcomes and a digest over the
  receipt/cleanup/ingest bindings were retained; no score or session ID was
  persisted.
- Exact version-selected task GETs reproduce all 33 QA-candidate atom and task
  graph source locators. Every locator includes an immutable Registry version;
  no `current` alias was used.

The observation window and aggregate evidence digests live in the inventory's
`authority_recheck`. No task prompt, answer, flag, trajectory, credential, or
private session content was persisted.

## Leakage boundary

Family grouping uses transitive closure over stable atom Artifact Registry
keys. Atom version suffixes do not split a family, composites union every
constituent atom, and versions sharing one task key stay together. The admitted
roster has 50 train and 25 heldout components with zero intersection. The 33 QA
candidates collapse to 26 additional components, none touching an admitted
component.

The 74 known-broken and 1,035 unanalysed rows do not have sufficient reviewed
lineage for training. They remain `unresolved_blocked`; the inventory never
guesses a family from a task name. This is what keeps unknown aliases from
leaking into the 50-task train ceiling.

## Next qualification wave

The next wave remains the prepared, model-free, one-cell canary:

- task version: `ae236a07-128d-4556-ba2b-25a248818497`
- transitive component:
  `sha256:2d9ece4620167eca71c829e9ae37644999a82a138d8ef8ade7484bdd8cf4ade5`
- atom source:
  `cyber/atoms/concur/extract-job-status-missing-auth@0:atom_source`
- task graph source:
  `cyber/task-graphs/single-atom-extract-job-status-missing-auth@2:task_graph_source`

It is still `prepared_no_launch` and requires separate authorization. If the
canary closes cleanly, preregister one exact representative from each of the
remaining 25 candidate components; do not qualify all aliases as if they were
independent families.

This refresh launched no rollout, made no Registry or Fleet mutation, and
created no Kubernetes object.

## Reproduction

Recheck the sealed artifact offline with:

```sh
uv run --locked python -m training.fleet_blackbox_self_sft_catalog
uv run --locked --extra dev pytest -q tests/test_fleet_blackbox_self_sft_catalog.py
```

An authorized operator can repeat the read-only live reconciliation into a new
versioned output path by updating the module's immutable output version and
running `--refresh-live`. The writer refuses to replace different existing
bytes.
