# Current Fleet easiest-100 pass@4 evaluation

This is the living operator entry point for the active Qwen3.8-27B and GLM-5.3
Fleet blackbox study. It describes the frozen scientific contract and safe
workflow; it does not claim the campaign is complete and is not blanket launch
authorization.

Before changing or replacing any part of this path, read
[`OPERATIONAL_LESSONS.md`](OPERATIONAL_LESSONS.md). It maps prior operational
failure classes to the guards and regression tests that now define the safe
boundary.

## Question and denominator

Evaluate these exact checkpoints on the same frozen 100 historically easiest
eligible Fleet blackbox exploit task versions:

| Model key | Repository and revision | Sessions required |
|---|---|---:|
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 400 |
| `glm-5.3` | `zai-org/GLM-5.3@30333038ada1f1dacb294a93270305a890b50c14` | 400 |

The selection, exact task-version UUIDs, four attempt numbers, and 800 stable
cell identities are frozen in
[`evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json`](../evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json).
The historical ranking is an ease proxy, not a target-model result. A cell is
the single `(model, exact task version, attempt 1..4)` measurement unit.

Validate the universe without contacting a model:

```bash
uv run python -m evals.fleet.exact_pass4_universe \
  evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json \
  --repo-root "$PWD" --summary
```

## Frozen agent treatment

Every poolable cell uses all of the following:

- OpenCode `1.18.27` through the pinned OpenAI-compatible adapter;
- 262,144 context tokens and 32,768 maximum output tokens;
- OpenCode native compaction plus post-compaction autocontinue, with 20,000
  tokens reserved as compaction headroom;
- at most 600 model requests and an eight-hour task timeout;
- exactly two task-facing tools, ordered `bash` then `submit_report`;
- the exact registered task version, environment/runtime seed, verifier, prompt,
  tool-catalog digest, model revision, and retry policy in the frozen plan.

Compaction is part of the treatment. A run made with compaction but without the
frozen autocontinue behavior belongs to a different result block and cannot be
silently credited here.

## Where each part runs

The OpenCode controller is the harness: it provisions the exact Fleet task,
sends model requests, exposes only the allowed tools, submits the report,
invokes authoritative scoring, cleans up, and emits sanitized receipts. The
model server can be elsewhere. Harness location and model-server location are
separate facts that must both be recorded.

There are two API services with different jobs:

| Service | Endpoint | Purpose in this study |
|---|---|---|
| Fleet evaluation Jobs API | `https://orchestrator.fleetai.com/v1/jobs` | Managed evaluations for models and harnesses present in the deployed catalog. |
| Nebius cluster Jobs API | `https://api.ft.flt.build/v1/runs` | General immutable image plus command workloads admitted as Ray jobs; use for dedicated serving, staging, training, or GPU diagnostics. |

The cluster API is not restricted to the training schemas exposed by one local
convenience client. Conversely, its general container support does not make an
arbitrary checkpoint available through the managed evaluation catalog. Before
using `/v1/runs`, check its deployed schema:

```bash
uv run python skills/cyber-cluster-jobs-operator/scripts/check_contract.py
```

## Hosted and dedicated serving blocks

Hosted Fleet inference and dedicated self-hosted inference may run in parallel
for throughput, but they remain explicit serving blocks. A block binds the
provider/route revision, exact model bytes, serving image and arguments,
precision/quantization, context, harness, tools, task versions, and retry
policy.

Partition only unstarted cells after a fresh score-blind reconciliation. Keep
all four attempts for a task in one serving block. Never repeat an accepted,
active, claimed, or model-started cell. Pool blocks only after sanitized
immutable receipts pass the parity gate:

```bash
uv run python -m evals.fleet.treatment_parity pool \
  --frozen /restricted/frozen-treatment.json \
  --candidate /restricted/candidate-treatment.json
```

A failed parity check is not permission to loosen the comparison: retain the
candidate as a separately reported treatment block.

## Safe launch and monitoring sequence

1. Start from a clean immutable repository commit. Validate the frozen universe
   and current live API contracts.
2. Prove the credential resolves to Fleet team without printing it. Hydrate the
   exact task versions through targeted read-only routes; do not copy prompts or
   verifier bodies into plans.
3. Run the score-blind ledger. Search the complete Jobs API, Kubernetes state,
   global execution-claim namespace, session inventory, and output roots for
   the exact proposed cells. Stop on a collision or contradictory evidence.
4. Preview rendered objects or the `/v1/runs/preview` response. Verify immutable
   images, commands, task/model bindings, resource shape, mounts, queue,
   preemption behavior, and fresh output roots.
5. Run one create-once cell canary per new model/route/harness treatment. Scale
   only after authoritative scoring, session ingestion, cleanup, and a
   digest-valid acceptance receipt all agree.
6. Partition the untouched tail at whole-task boundaries. Each controller must
   obtain an exclusive cell claim before any model request and must not retry a
   mutating request automatically.
7. Monitor Jobs API state, exact Kubernetes UIDs, Pod restarts, controller
   heartbeats, endpoint traffic, GPU utilization, cleanup, and acceptance
   receipts as independent evidence. Do not infer progress from directory or
   session counts.
8. On failure, first preserve a sanitized UID-bound terminal receipt. Release
   dedicated GPU capacity before diagnosing a controller-side issue. Create a
   reviewed successor only for cells proven retry-safe; never mutate the failed
   job or reuse its single-use output root.
9. Finish only when the score-blind ledger proves exactly 400 accepted cells for
   each model and no cell is active, blocked, retryable, or unstarted.

All creates are explicit. Preview output, a healthy endpoint, a completed
controller, or an ingested session is not by itself a valid model outcome.

## Ten-minute GPU idle-release rule

Dedicated serving has a hard ceiling of two nodes and 16 GPUs. Select the
highest live priority class whose deployed preemption policy is `Never`; never
infer nonpreemption from a class name. That setting governs the Kubernetes Pod
scheduler only: Kueue may still preempt and later readmit the Workload. Bind the
exact Workload, RayCluster, Pod, and Service UIDs, inspect the complete Workload
condition history, and invalidate the route after any preemption or identity
rotation. A server must not reserve idle GPUs.
Model loading is productive startup. After the server is Ready, require useful
GPU work or a fresh controller heartbeat. If neither remains fresh for 600
seconds, terminate the server through the supported Jobs API, record the
terminal state, and release the GPUs before debugging or resubmitting. Drain a
server immediately when its assigned block finishes.

Do not keep an idle server alive while repairing a harness, controller,
network, or evidence problem. Do not cancel, reprioritize, preempt, or mutate a
peer workload.

## Score-blind progress ledger

Use
[`evals/fleet/exact_pass4_ledger.py`](../evals/fleet/exact_pass4_ledger.py) as
the progress authority. With no receipt inputs it proves only the 400 + 400
denominator:

```bash
uv run python -m evals.fleet.exact_pass4_ledger --repo-root "$PWD"
```

For a live reconciliation, pass only explicitly reviewed sanitized evidence:

```bash
uv run python -m evals.fleet.exact_pass4_ledger \
  --repo-root "$PWD" \
  --mount-map /mnt/sfs/jobs=/shared/jobs \
  --accepted /absolute/path/to/ACCEPTED.json \
  --active-claim /absolute/path/to/current-claim.json \
  --nonrepeatable-claim /absolute/path/to/preserved-claim.json \
  --tombstone /absolute/path/to/retry-safe-tombstone.json \
  --json
```

For a reviewed point-in-time authority set, prefer an exact evidence manifest
over rebuilding a long command from memory. The manifest lists each receipt or
claim path together with its expected self-digest and fails closed if either
changes. Cluster paths can be projected through one read-only observer mount:

```bash
uv run python -m evals.fleet.exact_pass4_ledger \
  --repo-root "$PWD" \
  --evidence-manifest \
    docs/evidence/qwen38-study/2026-09-05-exact-pass4-ledger-evidence-snapshot-v1.json \
  --mount-map /mnt/sfs=/absolute/read-only/sfs/mount
```

Snapshots are immutable evidence. Publish a successor snapshot when a claim
becomes accepted, blocked, or retry-safe; do not edit a sealed snapshot or
infer an acceptance path from the adjacent raw attempt directory.

Repeat an option as needed. Do not point a claim root at unreviewed historical
claims: a preserved claim behind an accepted result is history, not active
work. The ledger rejects malformed digests, duplicate or contradictory states,
treatment drift, and any input containing prompt, trace, flag, or score fields.

## Laptop controller boundary

A laptop may run the exact OpenCode image for non-scored transport, image, and
tool-catalog parity. It is not automatically a poolable scored controller. The
current create-once claims and hosted endpoint stream leases are coordinated on
the cluster's shared SFS mount; a laptop that cannot mount those exact roots
must fail closed before creating a task instance or making a scored model call.

The future unblocking interface is a small remote atomic coordinator in the
same trust and storage domain as the cluster controllers. It must provide:

1. one transaction that reconciles the proposed cell against the global
   ledger, authoritative session inventory, Kubernetes objects, output roots,
   and canonical claim namespace;
2. atomic create-once claim creation bound to the exact execution identity;
3. acquire, continuously hold, and release operations over the existing hosted
   endpoint lease namespace, including the qualified two-stream ceiling; and
4. sanitized, digest-valid receipts for every decision without prompts,
   traces, flags, scores, model output, or credentials.

Until that reviewed interface exists, do not substitute local files, laptop
locks, or a check-then-create sequence. A laptop lane remains diagnostic or an
unstarted failover partition.

## Evidence and privacy

Persist model/task/treatment bindings, stable cell and execution identities,
exact runtime UIDs, timestamps, request/config digests, terminal classification,
cleanup, and self-digesting acceptance receipts. Keep credentials, raw prompts,
traces, flags, sealed scores, and private verifier content outside source
control. Historical Qwen3.6 reports are frozen provenance and remain under the
`docs/QWEN36_*` paths; they are not current operating authority.
