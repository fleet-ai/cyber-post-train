# Exact pass@4 final bulk v5 (held)

This append-only successor combines the reviewed hosted-c4 and dedicated-GLM
cores without mixing serving treatments. It is held: no release receipt or
cluster object is included.

## Exact statistical partition

- Qwen hosted: 399 cells, four whole-rank serial streams.
- GLM hosted ranks 1–50: 199 cells, four whole-rank serial streams.
- GLM dedicated A ranks 51–75: one scored canary plus two dependent streams,
  100 cells total.
- GLM dedicated B ranks 76–100: one scored canary plus two dependent streams,
  100 cells total.

The union is exactly the 798 cells remaining after the two accepted
Generation-7 canaries. No task rank crosses a hosted or dedicated serving
block. Results remain stratified by serving block.

## Independent release groups

`hosted-qwen` and `hosted-glm` each require the clear prebulk-v4 terminal and
that model's passing non-scored c2→c4 qualifier receipt. They do not wait for a
dedicated server.

Each dedicated replica is staged separately: Jobs API server creation, exact
UID-bound parity, one accepted scored canary, then its two dependent streams
after the v7 runtime gate. The CPU controllers use `fleet-serve-low` with
`preemptionPolicy: Never`; GPU servers remain Jobs API
`fleet-infra-quiet`/`Never` and retain the 600-second no-heartbeat release
policy.

## Safety and restart behavior

Every controller uses the existing global O_EXCL execution-claim namespace.
Accepted, active, claimed, or model-started cells are nonrepeatable. Controller
restarts skip claimed cells and may only continue the untouched tail. The
harness is OpenCode 1.18.27 with only `bash` and `submit_report`, 262,144-token
context, 20,000-token compaction headroom, and the exact immutable model/task
bindings inherited from the Generation-7 plan.

Preview the held plan and package with:

```sh
uv run python -m evals.fleet.exact_pass4_final_bulk_v5 preview --repo .
uv run python -m evals.fleet.exact_pass4_final_bulk_package_v5 preview --repo .
```

The submitter always renders from the exact release commit, runs a server-side
dry run, and uses create-only semantics. Dedicated and hosted groups use
separate ConfigMaps and Jobs, so hosted work can start while GPU serving is
still gated.

Before any create, the submitter requires the existing Secret
`chris-cyber-opencode-evals-v2` at UID
`e0febd8e-94a2-46b0-a0bf-dd6b3154187b` and confirms through the public
read-only account endpoint that it resolves to Fleet team
`a1025f0b-ad67-49fc-a023-51800ab43e84`. The credential is held only in the
process environment for that check; it is not written to a receipt or package.

## Release and evidence flow

Release receipts are created with `exact_pass4_final_bulk_v5 build-release`.
The command writes once and revalidates the exact package commit, prebulk-v4
terminal, and only the gates needed by the selected group. The separate
Generation-7-aware qualifier gatherer supplies hosted model and terminal
receipts. `exact_pass4_final_dedicated_evidence_v5` turns sanitized, digest-bound
GET-only observer results into the authenticated-preview, parity, scored-canary,
and post-canary runtime receipts, then passes every result through the existing
dedicated-v7 validator before writing it once.

The prebulk and live dedicated evidence live on SFS. Rendering/submission must
therefore run through the reviewed in-cluster create relay (or another runtime
with the exact SFS mount); a desktop checkout without `/mnt/sfs` is not a valid
release environment. This package intentionally exposes the release and
renderer entry points for that relay and never falls back to stale copied
evidence.

Every group release also requires a fresh
`fleet-exact-pass4-final-bulk-fresh-duplicate-v5` receipt observed no more than
15 minutes earlier. It binds that
group's exact v5 Job, ConfigMap, SFS-root, cell, claim, and Fleet-session
absence. The validator and release hook are included here. The matching exact
GET-only source/accept observer is a required integration artifact and must be
reviewed before this held package can receive a launch GO; the generic create
relay is intentionally not allowed to synthesize that evidence.
