# Qwen3.8 Fresh75 V2 Fleet dev17 heldout queue — 2026-09-20

## Decision

The mutation-free Fleet queue is frozen for one exact upstream-base control and
the seven terminally accepted Fresh75 V2 SFT checkpoints. No checkpoint is
launch-authorized: the failure budget is exhausted at 10/10. The upstream base
is the only checkpoint whose exact model and serving identities are currently
ready; every accepted SFT arm is blocked before export and serving. The e4 arm
is still training and is not an accepted checkpoint or a member of the scored
matrix.

No launch, stage, registration, task creation, POST, cancellation or other
external mutation was performed. External evidence came only from GETs.

## Frozen evaluation identity

- Queue: `configs/evaluation/qwen38-fresh75-fleet-dev17-queue-v1.json`
  (`sha256:942a2a0e19df7d68b0066928b1c6bbeb5121d934a79825354015de272c6245f6`).
- Standalone packet protocol:
  `configs/evaluation/qwen38-fresh75-fleet-dev17-protocol-v1.json`
  (`sha256:f72c34e3e5bd5ca76959717e4091c4fb0f10fa3a2639e4dbd2eb10f2d9718469`;
  self-digest
  `sha256:5db333575ac97cac30c8ccaeab1983e63937f774dda90ed9eda3b81c245ec21e`).
  The checkpoint packet generator reopens this file and every file it names,
  then rejects any changed task tuple, split, route profile, image, OpenCode
  treatment, sampling value, retry rule or evaluator byte.
- Task set: 17 exact development-heldout task/environment/data tuples in
  `configs/evaluation/qwen38-fresh75-fleet-dev17-task-set-v1.json`
  (`sha256:79c834e739246da29aca9513965ecfc7032f8df7744eb2245ce0303aba1b97c5`).
  The split object is
  `sha256:05b3a8dc90ca93adc9671942d75ecb54ff8d0951b0dd48a087e29ec1e641840c`
  and the dev selection is
  `sha256:38ea6686afa068c19e69fea2493e01027fdf99f7e01b20376e107b1a0cfa0b68`.
- The GET-only receipt binds all 17 task versions, runtime seeds, verifier
  versions and verifier-contract-v3 projections in
  `docs/evidence/qwen38-fresh75-fleet-dev17-get-census-20260920.json`
  (`sha256:0f083c9dace34382f85aa68f7daecaf417be6eb895355b3b6aee948793fb2565`).
- Harness: OpenCode 1.18.27, release asset
  `sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702`,
  `@ai-sdk/openai-compatible`, native compaction/autocontinue v2, 262,144-token
  context, 20,000-token headroom, 32,768 maximum output tokens, 600 model
  requests, 28,800 seconds, ordered tools `[bash, submit_report]`, and tool
  catalog
  `sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a`.
- Sampling is temperature 0.6, top-p 0.95 and seed 42. This is pass@1 with
  concurrency 1 and is ineligible for training-data collection.
- Agent image ID:
  `sha256:45424cc2e304e114b26aca916486df6e80fdc57c4f5adddf2bb144d4773d510f`.
  Fixed-proxy image:
  `ghcr.io/astral-sh/uv:python3.12-bookworm@sha256:9aa60c50016c0485636ab9a830246a6ef3399aa4a8bab3d17ef4a2358fba2ca7`.
  The queue also seals every evaluator runtime-file digest.
- Automatic retry is disabled. Valid outcomes are never retried. Ambiguous or
  infrastructure-invalid cells are held; any reviewed successor needs a new
  execution ID, incremented execution generation, unused output subroot and new
  failure-budget authorization.

The final-test split is prohibited. Each arm has a distinct campaign ID and
single-use output root below
`/mnt/sfs/jobs/q38-f75v2-fleet-dev17-p1-v1/`.

The protocol is scientifically complete before any checkpoint qualifies. A
qualified packet renders 34 matched sessions: 17 upstream-base sessions and 17
candidate sessions. Candidate payload revision and staged path come only from
the accepted checkpoint qualification. The candidate route UID and its live
runtime readback remain absent until create-once route registration; they must
be supplied by that exact registration receipt and fresh readback and are never
invented in the checked-in packet.

## Checkpoint readiness

| Arm | Accepted checkpoint | Current state | Earliest unmet gate | Planned route |
|---|---:|---|---|---|
| upstream base | exact upstream revision | serving-ready; launch blocked | execution-host image/output/receipt preflight, then new budget | `qwen3.8-27b` |
| b8-lr1e5-e2 | step 230 | export/serving blocked | full checkpoint payload manifest and rehash | `chris-q38-f75v2-b8-lr1e5-e2-wbe` |
| b8-lr3e6-e2 | step 230 | export/serving blocked | full checkpoint payload manifest and rehash | `chris-q38-f75v2-b8-lr3e6-e2-wbe` |
| b32-lr1e5-e2 | step 58 | export/serving blocked | full checkpoint payload manifest and rehash | `chris-q38-f75v2-b32-lr1e5-e2-wbe` |
| b64-lr1e5-e2 | step 30 | export/serving blocked | full checkpoint payload manifest and rehash | `chris-q38-f75v2-b64-lr1e5-e2-wbe` |
| b8-lr1e5-e1 | step 115 | export/serving blocked | full checkpoint payload manifest and rehash | `chris-q38-f75v2-b8-lr1e5-e1-wbe` |
| b16-lr1e5-e2 | step 116 | export/serving blocked | full checkpoint payload manifest and rehash | `chris-q38-f75v2-b16-lr1e5-e2-wbe` |
| b8-lr1e6-e2 | step 230 | export/serving blocked | full checkpoint payload manifest and rehash | `chris-q38-f75v2-b8-lr1e6-e2-wbe` |

The base GET readback had one catalog match at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, routed and ready with two ready
replicas. Its BF16 SGLang catalog profile and model/server readbacks match the
frozen queue. Each of the seven planned candidate route IDs had zero catalog
matches. No candidate repository/revision or serving profile is invented in the
queue; those fields remain absent until accepted export and live readback exist.

Every candidate must pass these gates in order:

1. Seal a full checkpoint payload manifest and independently rehash every byte.
2. Perform a zero-optimizer-step BF16 export that produces an immutable
   repository/revision plus payload digests.
3. Reload the export at zero optimizer steps and prove source bytes unchanged.
4. Stage exactly once, create a paused c1 route with a new UID, and GET-read it
   back before any separately authorized activation.
5. Reconcile fresh base/candidate parity for revision, BF16 engine/precision,
   model info, server info, context, reasoning/tool parsers, tool-calling
   capability and ready replicas.
6. On the authorized CPU execution host, prove both exact Linux/amd64 harness
   images are already staged, recheck all 17 bindings, run the session census,
   and prove the arm's output root and dedicated PostgreSQL experiment identity
   are absent.
7. Obtain a positive failure budget and explicit root launch authorization.

## Duplicate and absence findings

The 17 task-key session inventories contained 455 historical rows. None used a
planned candidate served ID. Sixty-one rows used the persisted base model name
`qwen3.8-27b`; the list projection does not expose campaign, harness, cell or
execution identity. Therefore those rows are neither accepted as reusable
control outcomes nor treated as proof that the new exact campaign already ran.
The base gate is a fresh empty output root/database plus exact per-cell receipts;
historical sessions cannot be silently reused.

Shared-storage output-root absence was not claimed from this audit host. It must
be checked on the authorized execution host immediately before preparation.

## Operational ordering, not a result

Sanitized final-decile train loss can order payload-seal/export work only:
b32-lr1e5-e2, b16-lr1e5-e2, b8-lr1e5-e2, b8-lr3e6-e2,
b64-lr1e5-e2, b8-lr1e6-e2, then b8-lr1e5-e1. All observations were finite.
This is not a capability ranking, must not choose the final model and must not be
combined with heldout scores before the frozen evaluation runs.

The e4 arm was only at optimizer step 279/460 (2.426 epochs and 2,504,941
supervised tokens processed) in the latest sanitized snapshot. Its recent
decile loss was 0.13697, a 72.0% relative drop from the initial window, with
minimum 0.0785 and finite values. It remains pending and outside the matrix. If
it later reaches terminal acceptance and the same export gates, it is a
duration/overfit study arm, not a presumed capability winner.

## Validation

Offline validation:

```sh
uv run python -m evals.fleet.fresh75_holdout_queue
```

Fresh live reconciliation is deliberately GET-only:

```sh
uv run python -m evals.fleet.fresh75_holdout_queue --live-get-only
```

The checker exposes no launch or mutation command. Its tests assert the fixed
arm sets, failure budget, handoff/checkpoint/digest parity, pending e4 exclusion,
retry policy and GET-only HTTP surface.
