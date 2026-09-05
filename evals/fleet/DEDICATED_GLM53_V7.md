# HELD GLM5.3 dedicated serving v7

This append-only successor prepares, but does not authorize, dedicated GLM5.3
serving after the generation-7 hosted canary. It creates no cluster objects and
reserves no GPU.

Validate and print a sanitized local preview with:

```bash
evals/fleet/scripts/submit_glm53_dedicated_v7.sh held-preview
```

The exact model, image, TP8/DP8/EP8 topology, FP8 weight loading, BF16 KV cache,
262,144-token context, SGLang parsers, and ten-minute post-Ready idle release are
unchanged from the reviewed v6 package. Replica A must be the first one-node,
eight-GPU canary. Replica B cannot be requested until A has passed UID-bound
non-scored parity, its separately predeclared one-cell scored canary has a
digest-valid accepted receipt, and a post-canary runtime receipt has accepted
that evidence. The dependent bulk plans never contain either scored canary
cell, so their runtime-gate dependency cannot become circular. The hard ceiling
is two nodes and sixteen GPUs.

## Sanctioned Jobs API route

The signed-in GitHub identity is accepted by the sanctioned Jobs API at
`https://api.ft.flt.build`. Its OpenAPI advertises `GET /v1/whoami`,
`POST /v1/runs/preview`, and `POST /v1/runs`. The token is held only in memory
and sent in the Authorization header; it is never placed in a config, receipt,
or command argument.

At the observation bound in the spec, the API accepted the policy-correct
`fleet-infra-quiet` request and rendered the immutable GHCR image, `ghcr-pull`,
and one eight-GPU Ray worker. The live PriorityClass has
`preemptionPolicy: Never`. The rejected `fleet-serve-low` probe is retained as
negative route evidence only. Neither preview created a cluster object or
authorized launch. Each replica still requires a fresh authenticated preview
immediately before its create-once duplicate gate and submit.

The generation-7 bulk package at commit
`b2934446d93cf34facf5fd007216e6dedcc9c2b4` assigns every remaining GLM cell to
hosted inference. It must not be used together with this dedicated path. Before
any GPU request, freeze an append-only dedicated-aware bulk successor and rerun
reconciliation so the hosted ranks 1–50 and dedicated ranks 51–100 are disjoint
at whole-task boundaries. If the all-hosted bulk has already accepted, claimed,
or model-started a rank in 51–100, keep that whole task hosted and generate a
fresh disjoint partition; never silently move or repeat it.

## Release order

1. Accept the exact generation-7 hosted GLM canary (rank 13, attempt 1), including
   claim, authoritative session, verifier, ingestion, cleanup, zero restarts,
   and released endpoint lease.
2. Freeze and accept a fresh dedicated-aware prebulk reconciliation and complete
   controller packages before requesting GPUs. The executable v4 source and
   acceptance jobs are CPU-only, `fleet-serve-low`, `Never` preempted, and
   create only score-blind, UID-bound receipts. They must prove the exact
   798-cell successor partition is absent before release.
3. Authenticated-preview replica A; immediately recheck Jobs API, Kubernetes,
   SFS, global claims, session, and model-started duplicates; create it once.
4. Prove the exact Jobs API run, RayJob, Workload, RayCluster, head Pod, Service
   DNS/selector/target, and probe Job/Pod traversal. The parity receipt must also
   bind the exact image, model revision, canonical 52-element serving argv,
   context, and content-blind structured `bash` plus `submit_report` behavior,
   with no scored session. A parallel-barrier probe must complete two
   simultaneous non-scored requests with zero errors before concurrency two is
   qualified; otherwise dependent bulk remains held.
5. Run only the separate parity-gated rank 51, attempt 1 plan while its
   controller owns the heartbeat. Release A's dependent bulk only after the
   post-canary runtime receipt validates the exact claim/execution,
   authoritative session/verifier, ingestion, cleanup, and serving identities.
   It may then release exactly the two disjoint controller streams at bounded
   concurrency two, with one heartbeat lease file per stream.
6. Repeat preview, create-once, parity, the separate rank 76, attempt 1 scored
   canary, and post-canary acceptance for B before releasing B's dependent bulk.
7. Drain a replica as soon as its block finishes. If no controller heartbeat is
   fresh for ten minutes after Ready, the embedded lifecycle stops the server.
   Release before diagnosing any controller-side problem.

Results remain stratified by hosted, dedicated A, and dedicated B serving
treatments. They are not pooled without a later reviewed equivalence analysis.
