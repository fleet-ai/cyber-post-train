# Qwen3.8 campaign unattended-readiness audit

Observed at **2026-09-12T12:35:24Z**. This was a read-only reconciliation of
sanitized Fleet Jobs API state and UID-bound Kubernetes controller state on
Nebius development and production. No logs, task content, results, credentials,
or sealed metrics were read. No API or Kubernetes object was created, changed,
deleted, reprioritized, resumed, or cancelled.

## Current execution state

The Qwen study has **zero live or queued training executions** in both clusters:

- no nonterminal `chris-q38-*` RayJob;
- no nonterminal Qwen-study Kueue Workload;
- no Qwen-study RayCluster; and
- no Qwen-study Pod in `Pending`, `Running`, or `Unknown` phase.

Therefore the observed Qwen study allocation is **zero GPU nodes and zero
GPUs**. This statement is intentionally narrower than “no experiment-owned
Pod.” Production still has historical, zero-GPU rollout infrastructure outside
this training wave: `chris-cyber-rollout-postgres-v1-0` was Running on one CPU
node, and `chris-cyber-rollout-archive-v1-pqq5x` was Pending without a node.
Three related historical Jobs lacked terminal conditions. They were not mutated
by this audit. Conservatively count the occupied CPU node when enforcing a
campaign-wide node ceiling unless a later reviewed policy explicitly exempts it;
the Pending, unallocated Pod counts as zero nodes.

## Stale Jobs API projections

The Jobs APIs returned 14 Chris-owned rows whose status was still `submitted`,
but none had a matching live RayJob, Workload, RayCluster, or Pod in the target
cluster. Their historical terminal outcome is not reconstructible from this
disagreement. Classify each as
`stale_nonterminal_api_projection_without_uid_bound_execution_object`, not as
active, succeeded, failed, or released. Preserve the names as duplicate-history
evidence; never replay one merely because its Kubernetes object is absent.

Development (2):

- `chris-q38-ta-dev1-3b8d80d0`
- `chris-q38-ta8-cos5-reld3-fdcce569`

Production (12):

- `chris-cpt-glm-qual-v1-506dcdba`
- `chris-cyber-q38-tsft-reload-g3-debf9ea8`
- `chris-cyber-runtime-pub-b02d7143`
- `chris-glm-dind-c1-b6ce000c`
- `chris-glm-dind-c2-f73b5b8e`
- `chris-glm-dind-c3-877a2ddc`
- `chris-glm53-eph-sf-v6-4a0514cd`
- `chris-glm53-eph-sf-v8-514ddc4a`
- `chris-glm53-eph-sf-v9-afbb10b2`
- `chris-q38-dp8-scorefree-canary-518f8de4`
- `chris-q38-ephemeral-canary-v3-1068d43c`
- `chris-q38-teacher-lr1-v2-7563ca25`

This is a time-bound observation. Repeat the API and UID-bound Kubernetes
reconciliation immediately before any new POST.

## Overnight control contract

- The three bounded LR development arms are prepared, pinned-image
  CPU-preflighted and development-API-previewed, but none has been submitted.
  Each requests one eight-GPU node with `c1`, whose reviewed effective queue
  class is `q1` at value 10,000, and disables automatic requeue.
- The aggregate ceiling is eight actively allocated or reserved experiment
  nodes **across development and production**, excluding only the four
  pre-existing inference endpoint nodes. Count training, export, temporary
  serving, evaluation, and conservatively CPU allocations. Queued work without
  an allocation counts zero. Recheck the live total immediately before every
  POST; do not queue an unguarded wave that could admit above the ceiling.
- The SFT runtime polls every 60 seconds, allows 30 minutes for startup, declares
  confirmed no-progress idle only after the fixed 20-minute evidence window,
  enforces an eight-hour hard bound, and permits at most five additional minutes
  for an advancing checkpoint drain. Missing telemetry is not idle evidence.
  A real runtime/storage/tracking defect remains a nonzero failure; do not hide
  it to suppress an alert.
- The production failed-object monitor observes new failed Job/RayJob UIDs every
  five minutes. The idle monitor's reviewed suspect rule is an allocation older
  than one hour with average GPU utilization below 1% over the prior 30 minutes.
  Dev-first qualification, bounded startup/idle/drain behavior, and prompt
  release of an exact owned broken allocation are prevention controls, not an
  alert exemption.
- Before an unattended GPU POST, require an authentication window long enough to
  observe startup, progress, terminal evidence, and resource release. If control
  access expires, stop mutations and report possible resource leakage; do not
  infer release from a stale API row.

## RL state

The spawn-safe SkyRL diagnostic-router correction is committed at
`7599706f1b5626d7ad2588804c75ea85ceeb099b`. This records a code and
pinned-image operational prerequisite only. RL dev3 and dev4 remain terminal
and released and must never be retried. No RL GPU successor is live or queued;
a new unique dev diagnostic still needs its own plan, pinned-image preflight,
Jobs API preview, duplicate/output checks, authorization, and immutable runtime
identity before one create-once submission.

## Monitor wording corrections

The recurring controller should:

1. say “no Qwen study training Pod or GPU allocation” rather than “no
   experiment-owned Pod”;
2. record the 14 API-only rows above as stale/unknown evidence, not live work;
3. stop saying the RL correction awaits a commit—the exact commit now exists;
4. count admitted/reserved nodes across both clusters before every POST; and
5. preserve the existing dev-first, exact-UID, no-requeue, no-peer-mutation,
   no-private-log, and fail-closed authentication rules.

This evidence does not authorize a development or production submission.
