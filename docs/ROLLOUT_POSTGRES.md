# PostgreSQL rollout operations

This is the reusable operator contract, not a live status report or permission to
launch. Read `AGENTS.md`, the campaign's current authorization and exact bindings,
and `CLUSTER_ALERTS_AND_INFERENCE_SERVING.md` first. Never infer current counts,
credential validity, endpoint health, or next worker numbers from a document.

## What moved, and what did not

PostgreSQL now coordinates the distributed Fleet rollout queue. Workers on different
hosts use database transactions and row locks instead of shared-filesystem SQLite
locking. The migration preserves scientific cell identities, state, claim ownership,
event order, and private local-result records. It does not change the model, task,
harness, sampling treatment, attempt budget, or acceptance contract.

| Component | Authority and permitted use |
| --- | --- |
| `rollout_cells` | Immutable scientific identities and mutable worker-owned state |
| `rollout_events` | Append-only transition evidence; status readers use allowlisted metadata only |
| `rollout_local_results` | Private result index; monitoring may count records, not read outcomes or payloads |
| `ledger_metadata` | Frozen plan identity; a different plan is rejected |
| `ledger_migrations` | Migration hashes and normalized row-parity proof |
| Attempt files and accepted/terminal receipts | Preserved immutable evidence; PostgreSQL does not replace them |
| Archived SQLite, CSV/JSONL exports, narrative docs | Historical evidence/projections, never a second scheduler |

Claims select **only pending rows on the exact route**, atomically with
`FOR UPDATE SKIP LOCKED`. Every state change is fenced by `cell_id`, `worker_id`,
and a fresh `claim_id`. Expired leases never auto-requeue. `retry_review` is not a
source of replacement work; existing review rows must remain untouched by monitoring.
The shared package deliberately provides no orphan-repair/apply command.

The worker chooses one backend: `--postgres-dsn-env ROLLOUT_DATABASE_URL` or the
legacy local `--database` path. Never dual-write or fall back to SQLite after a
PostgreSQL error. Credentials come from a Secret/environment, not CLI values,
rendered YAML, receipts, logs, or Git. A Secret *name* is not a credential.

## Read-only status

For the authorized four-route campaign, the official occupancy observer is:

```sh
uv run --locked python -m evals.fleet.rollout_refiller \
  --namespace fleet-train-jobs \
  --ledger-pod chris-cyber-rollout-postgres-v1-0 \
  --ledger-kind postgres --target-per-route 8
```

It performs read-only Kubernetes and PostgreSQL queries. It never creates a Job,
claims work, repairs an orphan, or reads result content. `refill_blocked=false`
is necessary, **not sufficient**, for a launch: receipt, source, endpoint, access,
and identity-absence gates remain independent checks.

For a separately authorized database diagnostic with an existing environment-held
connection string, the following uses an enforced read-only, repeatable-read
transaction and never initializes a schema:

```sh
uv run --locked python -m evals.fleet.rollout_postgres_status summary
uv run --locked python -m evals.fleet.rollout_postgres_status active
```

Prefer a least-privilege diagnostic role. Neither command grants access or replaces
the official campaign observer. Missing schema/access fails closed; driver errors
are sanitized. Never run `SELECT *`, dump tables, read Pod logs, or open private
result/trace files to answer a status question.

## Refill and priority protocol

Use one designated creator across operators, terminals, agents, and automations.
Parallel independent creators can race even when each saw a deficit. The current
observer/renderer is not a distributed launch lock; serialize handoffs explicitly.

1. Reconcile every new terminal from exact Job/Pod UIDs, zero restarts, exit zero,
   no terminal Job failure, digest-valid sanitized receipt, requested/accepted
   counts or verified absence, cell transitions, and local-result count. Kubernetes
   `Complete` alone does not establish scientific acceptance.
2. Immediately recheck the approved Git/source/harness pins; PostgreSQL Pod UID,
   Ready state, zero restarts and readability; all endpoint UIDs/revisions and
   Pod health; score-blind queue/retraction metrics; and the actual configured
   cluster-access expiry. Follow the campaign's minimum validity window.
3. Observe occupancy again. Queued/initializing **refiller-labelled Jobs occupy
   slots**, even without a Pod or claim. Require zero stale/orphan claims, no
   ambiguity, the intended route deficit and a free total slot. Preserve unrelated
   historical suspended objects; never relabel them to alter counts.
4. Recompute route counters from all preserved Job and ConfigMap names. Prove the
   exact new names and short/expanded worker IDs absent from Kubernetes and
   PostgreSQL. Require a pending row on that route; never select review rows.
5. Render only with `rollout_successor.py`, the exact source Job UID, new immutable
   identities, `--repo-root .`, the approved `--postgres-secret`, and reviewed
   `--concurrency-stage`. Validate the whole List without displaying its embedded
   source or environment values. Server-dry-run the exact result before submitting
   the List with `kubectl create` **once**. A partial/error response is not permission
   to retry blindly; reobserve object UIDs and preserve any objects already created.
6. Immediately reobserve and count the initializing Job. Wait for its UID-bound Pod
   Ready/zero restarts and its matching distinct PostgreSQL claim before another
   successor. Recount all routes. Stop on drift, restart, failure or any missing gate.

The renderer preserves the clean-exit wrapper, annotation
`cyber-post-train.fleet.ai/alert-safe-job-policy=retry-without-terminal-failure-v1`,
backoff limit `2147483647`, and absence of `activeDeadlineSeconds` and
`podFailurePolicy`. Handled outcomes still require truthful receipts; never suppress
or manufacture a platform alert.

Normal queue/default priority remains the default. A current, explicit priority
exception can use `--workload-priority-class <name>` together with
`--expected-priority-class-uid <uid>`. These change the Kueue Job label, not Pod
priority, admission, or peer workloads. Verify the resulting Workload's effective
priority and admission separately: a Job label alone is not queue readback.
Higher priority may cause normal Kueue preemption depending on ClusterQueue policy;
it is not a guarantee of immediate execution. Never manually unsuspend or set admission.

Plan receipt accessibility before a campaign starts. A completed Pod cannot serve
`exec`, and a shared filesystem mount may exist only on a specific node. Never
create observer/archive/status-only Jobs under a campaign prohibition. A one-time
reader needs separate explicit approval, least privilege, read-only exact mount,
bounded lifetime and UID-bound verification; it is not a reusable standing exception.

## Migration is a separate, reviewed operation

The current campaign is already migrated. **Do not reapply the example manifests or
rerun migration on it.** `evals/fleet/manifests/rollout_*_v1.yaml` are deployment/
migration references, not a bootstrap shortcut for joining the team.

For a new migration, require explicit authority, stop all writers through their
supported lifecycle, preserve exact object/receipt identities, and establish a
quiescent source plus an empty, dedicated target. Use `rollout_archive.py` to make
a private lossless copy and consolidated SQLite snapshot. It copies opaque bytes;
it does not authorize viewing them. Validate the archive receipt/manifest, every
file digest, consolidated database integrity and the required terminal receipts.

`rollout_postgres_migrate.py` accepts `--sqlite`, `--archive-root`, an environment
variable name via `--dsn-env`, and a new `--receipt` destination. In one PostgreSQL
transaction it requires empty source tables, imports the four legacy tables,
preserves event sequence, compares every normalized row digest/count, and records
migration provenance separately. Verify source/target parity before switching any
worker. Keep source and archive intact; never clear or overwrite a nonempty target.

Database commit and filesystem receipt publication cannot be one transaction. If
receipt publication fails after commit, stop and reconcile `ledger_migrations`
read-only against the source hashes. Do not retry import, empty the target, or
fabricate a success receipt. Migration grants no authority to change retry states.

## Durability and handoff

The example StatefulSet is a **single PostgreSQL instance**, not HA or a verified
backup system. Before expanding ownership, agree on the database operator, backup
destination/retention/access controls, encryption, and a tested restore procedure.
Back up PostgreSQL and immutable attempt evidence as a coordinated pair. A PVC,
successful migration, or copied manifest is not proof of recoverability. Restore
only into a fresh isolated target; verify row/receipt parity before any cutover.

Each operator handoff records observation time, campaign/plan and Git identities,
PG/endpoint UIDs, per-route occupied/running/initializing counts, accepted/local
counts, review count, stale/orphan status, unresolved terminals, the exact last
submission response, next gate, creator ownership and automation status. Store
only sanitized metadata. Never paste access tokens, auth links, private data or
mutable “current” selectors into the handoff.
