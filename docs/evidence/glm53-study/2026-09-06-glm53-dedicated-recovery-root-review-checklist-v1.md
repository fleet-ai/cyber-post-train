# Dedicated GLM recovery root-review checklist

This checklist is a launch stop, not an authorization receipt. The future v23
server and rank-100 controller remain held until every item is independently
verified from fresh score-blind evidence.

## PR 99 code and evidence

- [ ] The PR head is mergeable and its unchanged remote workflow is green.
- [ ] Repo-wide Ruff and all workflow test suites pass, including the recovery
  successor, scoring-route preflight, concurrency observer, fail-closed ladder,
  and post-qualification successor tests.
- [ ] The two consumed post-model failures remain `retry_authorized=false` and
  their cell and execution identities are excluded from every successor.
- [ ] The late scoring-route preflight is body-blind, makes zero mutations,
  requires `GET` to return `405`, runs after model finalization and immediately
  before scoring intent, and cannot authorize an automatic retry.
- [ ] A subsequent Fleet request failure persists only method, route, and HTTP
  status; no response body, prompt, trace, flag, or score is persisted.

## Fresh launch-time authorities

- [ ] Hosted rank-3/attempt-1 v2 is terminal and its exact scoring boundary is
  classified. If accepted, its accepted receipt and session/verifier linkage
  are digest-valid. If failed, the sanitized method/route/status are preserved.
- [ ] A self-digested 400-cell global ledger excludes every accepted, ingested,
  active, claimed, blocked, and ambiguous cell from the rank-100 candidate.
- [ ] All four rank-100 cells are still wholly unstarted: generation zero and
  zero exact claim, output, accepted-receipt, session, and verifier collisions.
- [ ] Authenticated Jobs API, Kubernetes, and SFS checks show no v23 title,
  run-directory, controller-name, or output-root collision.
- [ ] The live two-node/16-GPU accounting includes every owned peer; the exact
  topology and quota gate proves a single eligible eight-GPU node.
- [ ] `fleet-infra-quiet` still exists with `preemptionPolicy: Never`, while the
  complete Kueue Workload condition history remains a separate live gate.

## Server and scored-controller gates

- [ ] Create the server once through `POST /v1/runs`; bind API, RayJob,
  Workload, RayCluster, Pod, and Service UIDs without identity rotation.
- [ ] Lifecycle READY, exact model/revision/context, service origin, and actual
  OpenCode parity are bound in fresh self-digested receipts.
- [ ] The actual-request counter—not health, process liveness, or packaging—
  drives the 600-second idle release rule.
- [ ] The exact final projected controller package passes a UID-bound CPU full-path
  canary through the same pre-model callback and scoring-route probe.
- [ ] Root independently reviews the final server-bound release. Only then may
  one rank-100 controller be created once, with one same-task attempt in flight
  and authoritative acceptance required before the next attempt.

The held plan must continue to report `launch_authorized=false`,
`gpu_server_submit_permitted=false`, and `scoring_create_permitted=false` until
a separate immutable launch receipt changes those exact fields.
