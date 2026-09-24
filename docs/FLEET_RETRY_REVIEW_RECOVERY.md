# Recovering incomplete Fleet evaluation cells

This procedure is for a cell that the evaluator placed in `retry_review`. It
does not grant permission to launch a job. It applies only after the exact
campaign owner has reviewed the private evidence and approved the specific
create-once action.

The central rule for one frozen cell is simple: **never overwrite or retry its
model trajectory when a stored trajectory already exists.** Repeating the same
cell after seeing which cells had infrastructure trouble can bias a comparison,
even if nobody reads a score.

## Outcome-validity correction (2026-09-24)

Stored-session reconciliation preserves what happened; it does not turn an
invalid lifecycle into a capability outcome. Under the current scientific
protocol, only exit zero plus a normal OpenCode completion may enter a pass@k
denominator. An `output_limit` or `process_error` attempt remains held even when
its session was stored and scored. Legacy reconcilers may preserve such a row
as database evidence, but every maintained aggregate must reject its lifecycle.

To finish a matched pass@k study, keep the original held row and use a new,
predeclared replacement seed for the same task in **both** arms. Freeze that
paired replacement before execution and without reading either score. This is
a new comparison cell, not a retry or overwrite of the original cell.

## Find the rule by symptom

| Sanitized symptom | Safe path |
| --- | --- |
| `output_limit` with a stored session | Preserve or reconcile the existing record only; keep it held outside capability metrics. |
| `process_error` with a stored session | Preserve or reconcile the existing record only; keep it infrastructure-invalid outside capability metrics. |
| `output_limit` or `process_error` without a completed stored session | Preserve the terminal evidence and keep the cell held. |
| `post_claim.connecterror` with no generation artifact anywhere | One rollout retry after exact absence proof. |
| `post_claim.operationalerror` with no generation artifact anywhere | One rollout retry after exact absence proof. |
| `post_claim.fleetrequesterror` with an exact source-bound provisioning `POST`/`504`, the complete pre-model file set, and no local or authoritative session | One private-roster rollout retry after two identical observations; this does not make the failure-code class generally retryable. |
| `post_claim.fleetrequesterror` without that exact proof | Manual terminal review; never regenerate. |
| Any evidence conflict, unknown failure, or exhausted limit | Manual terminal review; do not create work. |

## Two separate recovery paths

1. **A stored session exists.** Verify that it belongs to the exact task
   version, model route, harness, seed, and evaluator plan. If it already has a
   complete authoritative score, reconcile that existing result without
   reading or recording the score value. If scoring is incomplete, run at most
   one scoring-only recovery against the immutable stored session. This path
   never calls the model.
2. **No session or generation artifact exists.** A rollout retry is possible
   only for the allowlisted post-claim transport failures and only after proving
   absence from the local result store, authoritative session store, trace
   manifest, and scoring-intent record. Run at most one retry with the exact
   original task, model, harness, seed, budgets, and route.

An output limit or agent-process error never authorizes a retry of that same
cell. Use the stored session path only to preserve its immutable evidence;
otherwise leave the cell in manual terminal review. A complete matched study
may use the separately predeclared, score-blind paired replacement rule above.

The failure-code prefix is not authoritative evidence of how far execution
progressed. In particular, a row recorded as
`authoritative_scoring_started.runtimeerror` can still have a completed
authoritative score when the controller failed after ingest. Classify it from
the exact local lifecycle artifacts and authoritative session metadata. If
those prove one complete already-scored session, use
`accept_existing_scored_session`: do not invoke the scorer and do not regenerate
the trajectory.

## Review order

1. Wait for all four primary seed-43 arms to become terminal. Do not change an
   active frozen run.
2. Read only the allowlisted lifecycle metadata. Do not inspect score values,
   prompts, traces, flags, responses, or rewards.
3. Produce a private, self-digesting cell manifest. It binds every selected
   cell to its original frozen plan and records exactly one action:
   `accept_existing_scored_session`, `scoring_only_recovery`, `rollout_retry`,
   or `manual_terminal_review`.
4. Classify every arm with the same rules before applying any action. Accepted
   cells are final and never replayed. Do not choose an action from model
   success, failure, or score.
5. Run scoring-only actions before considering any rollout retry. Reobserve
   each selected cell immediately before a create.

## Create-once gate

For each approved scoring or rollout recovery, prove all of the following
immediately before submission:

- the reserved Job, ConfigMap, Pod, Workload, output directory, and database
  identity are absent;
- the rendered root Job has `fleet.ai/failure-alerts: "off"`, priority `c1`,
  zero GPUs, and the reviewed CPU and memory request;
- a server dry-run of the exact object succeeds;
- the original model route is Ready with the frozen revision and parity
  receipt;
- the exact frozen task versions, OpenCode harness, sampling seed, budgets,
  and verifier binding are unchanged; and
- the cluster has capacity for immediate useful work.

Create each object once. If the create response is uncertain, reobserve the
exact identities; do not submit again. A failure or missing receipt remains
truthful evidence and must not be hidden by moving the cell back to pending.

## Worker-side digest fence

Applying a reviewed action and checking it from the operator terminal is not
enough. The recovery worker must independently prove, inside the same database
transaction that makes its claim, that it is consuming exactly the approved
work.

For a rollout recovery, create one private reviewed intent after classification
finishes. Use schema `fleet-reviewed-recovery-intent-v1`; bind the frozen
evaluation-plan digest, all recovery-runtime file digests, the exact route,
and the complete selected cell roster, then add the intent's own SHA-256
digest. That self digest is the database `reconciliation_digest`; changing,
reordering, adding, or removing a cell changes the authority. Keep this file
mode `0600` and outside Git because it contains private cell identities.

Run the recovery through `python -m evals.fleet.reviewed_recovery_worker`, not
the ordinary rollout worker. The worker first applies the complete intent in
one bounded database transaction. Application writes that digest on every
selected row but deliberately leaves every row in `retry_review`; ordinary
route-wide workers therefore cannot claim or race the reviewed cells. A
score-blind apply receipt is stored in `ledger_reconciliations`.

Before every recovery claim the worker:

1. locks the complete reviewed roster;
2. requires every row to exist on the frozen route and remain bound to the
   exact self-digesting intent;
3. requires every row's database `reconciliation_digest` to equal the frozen
   intent digest;
4. selects only an eligible `retry_review` row from that exact roster, moving
   it directly to `claimed` while consuming its one retry allowance; and
5. records a score-blind pre-claim receipt in `ledger_reconciliations` in the
   same transaction as the claim.

The receipt contains only digests, counts, and explicit statements that no
score, task, cell, session, prompt, response, flag, reward, or trace content is
present. Both apply and claim use bounded statement and lock timeouts, and the
claim lease starts from the wall clock after the lock succeeds. A missing row,
route difference, digest difference, exhausted allowance, runtime difference,
or changed intent fails before a claim or model call. Never move these rows to
ordinary `pending`, and never fall back to route-wide
`rollout_postgres.claim` after this failure.

This fence does not approve a retry, authorize a launch, or replace the private
review and apply receipts. It prevents a reviewed worker from silently acting
on database state that no longer matches those frozen receipts.

The comparison is usable only when every task has the predeclared number of
paired, normally completed outcomes in both arms. A database `accepted` state
is insufficient by itself. Never splice cells from an older campaign or a
different arm. The final eight tasks remain sealed.

## When the controller dies after the session was stored

A failed controller does not mean its completed Fleet sessions are lost. The
LR30 seed-43 controller ended after all 17 local results had been written: six
cells were already accepted, ten were held for review, and one expired owner
still held its cell. Every unresolved cell had exactly one completed Fleet
session with the frozen task version, model, and verifier identity. The normal
session listing exposes the parent task UUID, not the pinned task-version UUID.
The reconciler therefore uses the session reference-trace metadata surface only
when it returns the exact pinned version and an explicitly empty trace payload
(`reference_trace_count=0`, `reference_traces=[]`). A non-empty payload or an
unknown response field fails closed; task or session content is never accepted
or copied into evidence.

Do not restart the model and do not score those sessions again. Use the
stored-session reconciler instead. It reads the private cell roster from a
mode-0600 intent, validates every local artifact digest, observes the exact
Fleet session metadata twice, compares the authoritative score with the local
result only in memory, and accepts the complete roster in one database
transaction. A still-live owner fails closed; an expired owner is admissible
only when the intent also binds the terminal source Job receipt. Public output
contains counts and digests, never score values or private cell, task, session,
prompt, response, flag, reward, or trace data.

Before rendering the recovery Job, require the sanitized terminal receipt, the
public Job plan, and the private mode-0600 intent to carry the same terminal
receipt self-digest. A mismatch means the packet was assembled from different
reviews and must fail before a ConfigMap, Secret, or Job is created.

The bootstrap bundle is a separate closure gate. Materialize only the exact
ConfigMap module set in an isolated import tree and import the reconciler there.
Importing it from the full repository can hide a missing transitive module; the
live v1 packet demonstrated this when `rollout_worker` imported an unbundled
`rollout_campaign`. A successor needs a fresh ConfigMap, Secret, Job, and output
identity after that failure.

The PostgreSQL secret contains an administrator connection URL, not the
dedicated evaluation database identity. Bind the exact database name in both
the public recovery plan and private intent, replace only the URL database path
with that reviewed name, and then verify the frozen plan against that database
before any row mutation. Passing the secret URL through unchanged can select an
unrelated, internally valid ledger: the LR30 v2 reconciler compared its 17-cell
plan with an 800-cell database and correctly failed the plan-digest check. The
materialized v2 output path is permanently retired. A successor must use fresh
ConfigMap, Secret, Job, and output identities; it must never make the unrelated
database fit by weakening plan verification.

The incident receipt is
[`qwen38-lr30-step76-fleet-dev17-seed43-terminal-census-20260921.json`](evidence/qwen38-lr30-step76-fleet-dev17-seed43-terminal-census-20260921.json).
The guard is [`stored_session_reconciliation.py`](../evals/fleet/stored_session_reconciliation.py),
the CPU-only create-once package is
[`stored_session_reconciliation_job.py`](../evals/fleet/stored_session_reconciliation_job.py),
and focused regressions live in
[`test_stored_session_reconciliation_job.py`](../tests/test_stored_session_reconciliation_job.py)
and [`test_rollout_postgres.py`](../tests/test_rollout_postgres.py).
The sanitized database-selection incident is
[`qwen38-lr30-step76-stored-session-reconciliation-v2-database-selection-failure-20260921.json`](evidence/qwen38-lr30-step76-stored-session-reconciliation-v2-database-selection-failure-20260921.json).

The machine-readable review-only contract is
`configs/evaluation/qwen38-fleet-dev17-seed43-reviewed-recovery-v1.json`.
Its classifier is `evals/fleet/retry_review_policy.py`. The classifier rejects
unknown fields, including a score value, and fails closed on contradictory
evidence.

For a current one-arm campaign created by `heldout_launch.py`, use
[`heldout_stored_session_reconciliation.py`](../evals/fleet/heldout_stored_session_reconciliation.py)
only after `collect_terminal` has written the exact source receipt. The adapter
requires the exact successful source-create JSONL journal as well. That journal
must bind both the source Job UID and source ConfigMap UID later observed in the
terminal receipt. The reviewed mode-0600 authorization binds the source launch
packet bytes, terminal receipt, create journal bytes, database, plan digest,
one reviewed private cell roster, every executable support file, `run.sh`, the
immutable image, dependency versions, fresh object names, output path, and the
canonical digest of the complete ConfigMap/Secret/Job bundle. Changing any one
of them requires a new review.

The adapter renders an immutable CPU-only `c1` bundle with root failed-job
alerts disabled; it has no create operation. Its worker verifies the full
executable closure before installing or running anything, accepts the already
scored sessions only after two identical authoritative observations, and never
calls either the model or scorer. A later operator must still perform the live
exact-context absence census, two identical exact server previews, durable
create intent, and one create request. A preview may differ only by the small,
enumerated set of Kubernetes identity fields and exact API defaults; an added
command, environment variable, init container, service account, volume,
annotation change, or private Secret key fails closed. Do not prepare or run
this packet while the source Job is active.

Any observer or reconciler Pod that reads `ROLLOUT_DATABASE_URL` must carry the
exact Pod-template label
`cyber-post-train.fleet.ai/postgres-client: "true"`. The production PostgreSQL
NetworkPolicy selects that label; omitting it causes a connection timeout
before any database read. Check the label on the server-rendered Pod template
before create. A Job-level label alone does not satisfy this requirement.

A recovery observer must derive its expected plan rows and `harness_id` from
the exact evaluator modules and task set embedded in the sealed launch packet.
Do not copy a harness identifier from another arm, an older packet, or the
current checkout. Use `heldout_launch.sealed_evaluation` and
`heldout_terminal_observer.validate_then_publish`: finish the terminal receipt,
database, private audit, and artifact checks before creating any output, then
publish the complete private directory with one atomic no-replace rename. A
validation failure must leave the final path absent. Preserve an older partial
root as incident evidence; never delete it, overwrite it, or reuse its identity.

## Seed-44 Base narrow repair (historical; not capability-valid)

The procedure below records the historical repair design. Its five
`output_limit` sessions may be preserved, but they cannot satisfy the current
pass@k outcome-validity gate. Do not use this repair to claim a complete
capability comparison; use a predeclared paired replacement design instead.

The seed-44 Base controller left one operationally recoverable but
scientifically incomplete arm: ten accepted cells, five `output_limit` cells
with completed stored and
authoritatively scored sessions, and two cells whose exact source attempt ended
at the provisioning `POST` with HTTP 504 before any session, trace, or scoring
artifact existed. Preserve the ten accepted cells. Atomically accept the five
existing scored sessions without a scorer or model call, then create a fresh
generation-2 execution identity only for the exact private two-cell roster.
Each of those cells may consume its frozen campaign's single retry once.

The public, score-blind source census is
[`qwen38-base-fleet-dev17-seed44-terminal-census-20260921.json`](evidence/qwen38-base-fleet-dev17-seed44-terminal-census-20260921.json).
The inert two-stage plan is
[`qwen38-base-fleet-dev17-seed44-narrow-repair-v1.json`](../configs/evaluation/qwen38-base-fleet-dev17-seed44-narrow-repair-v1.json).
Private intents are prepared by
[`seed44_base_repair_intents.py`](../evals/fleet/seed44_base_repair_intents.py),
the CPU-only packages are rendered by
[`seed44_base_repair_job.py`](../evals/fleet/seed44_base_repair_job.py), and
focused guards live in
[`test_seed44_base_repair.py`](../tests/test_seed44_base_repair.py).

Run the two stages in order. Stage one requires no model route. Before stage
two, prove fresh exact Base serving parity and repeat the duplicate census and
server preview. Both rendered root Jobs must show
`fleet.ai/failure-alerts: "off"` before create. Stage two uses a fresh
Docker-in-Docker data directory, so it must verify the
exact harness archive and build-receipt digests, load that archive, and pull the
immutable proxy image before running the normal image preflight or claiming a
cell. A node-level Docker cache cannot satisfy this gate.
The legacy database repair would reach 17 accepted rows: original ten,
reconciled five, and generation-2 rerolls for two proven pre-session failures.
That database state is not a scientifically complete comparison because the
five reconciled output-limit rows are held. Never infer object absence until
the live command has asserted the expected Kubernetes context and namespace.
