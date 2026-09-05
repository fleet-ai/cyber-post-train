# Hosted c4 exact-pass@4 successor (held)

This append-only successor increases controller-level hosted concurrency without
changing an evaluation cell. It is **held** and creates no objects.

## Frozen shape

- Qwen3.8-27B: four independent serial controllers, 399 total cells.
- GLM-5.3: four independent serial controllers, 399 total cells.
- Each controller owns 25 complete task ranks. The controller containing the
  accepted Generation-7 canary owns only attempts 2–4 for that task.
- Every cell id, execution id, run id, network, task version, model, OpenCode
  1.18.27 treatment, tool contract, 262,144-token context policy, and compaction
  policy is inherited byte-for-byte from the four-controller v3 package.
- The two existing per-model lease keys are retained, with a maximum of four
  holders per model. Jobs remain CPU-only, `fleet-serve-low`, and
  `preemptionPolicy: Never`.

## Release gates

A later append-only release must prove all of the following:

1. Both Generation-7 canaries remain accepted through the v3 prebulk terminal.
2. The exact G7-gated non-scoring concurrency qualifier Job is exclusively
   Complete, both model receipts are `PASSED`, its terminal is `PASSED`, and its
   recorded lease was released.
3. A fresh GET-only reconciliation checks all eight successor Job, ConfigMap,
   Pod, and output-root identities and all 798 existing cell/run/claim/session
   identities immediately before release, with zero collisions. Its observer
   must be the exact clean package commit and its receipt may be at most 15
   minutes older than the append-only release.
4. The release is create-once and uses the existing global execution claims.
   A restarted controller may continue only with cells that have no accepted,
   active, claimed, or model-started state.

At runtime, each controller must receive the exact release path, file digest,
and package commit. The v4 wrapper validates that release and deliberately
replaces the v3 runtime's controller-map gate; reinterpreting the old v3
reconciliation with the new eight-controller map would fail closed before any
model call.

The qualifier is an operational capacity gate, not a capability result.

## Dedicated-GLM integration boundary

This package is hosted-only. It does not pool hosted and dedicated GLM
treatments. A dedicated-aware package must be a mutually exclusive append-only
successor that transfers complete GLM task ranks, preserves the same statistical
cell identities, records a separate serving block, and reruns duplicate
reconciliation over its final partition. The integration surface is the sorted
exact-identity digest plus the `(model, selection_rank)` whole-task ownership
map; controller names are execution machinery and may differ.

## Preview

Run:

```sh
uv run python -m evals.fleet.exact_pass4_hosted_c4_bulk_v4 preview --repo .
```

The only valid current result is `status=HELD`, `launch_authorized=false`,
`objects_created=false`, eight controllers, and 798 planned sessions.
