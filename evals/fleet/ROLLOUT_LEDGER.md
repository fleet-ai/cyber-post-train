# Rollout coordination ledger

`rollout_ledger.py` is the small shared work queue for concurrent cyber-evaluation
workers. The SQLite file is authoritative; `progress.csv` is a human-readable snapshot.
Workers must never select or update work by editing the CSV.

The ledger stores no prompt, trace, flag, score, credential, or model response. One row is
one scientific cell: exact task version × logical model × pass@k attempt. Shared and
dedicated replicas are recorded as distinct serving blocks. Use one database per
experiment; initialization rejects a mixed-experiment plan.

## Plan CSV

Create the complete plan before any worker starts. For the current 100-task pass@4 study,
it contains 800 rows. Assign every task's four attempts to the same serving block, then
balance whole tasks across the two exact replicas for that model.

```csv
experiment_id,task_key,task_version_id,model_id,model_revision,serving_block,endpoint_model_id,harness_id,attempt,max_retries
q38-glm53-fleet100-p4-v1,<task-key>,<exact-task-version>,qwen3.8-27b,1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,qwen-shared,qwen3.8-27b,opencode-<exact-version>,1,1
q38-glm53-fleet100-p4-v1,<task-key>,<exact-task-version>,qwen3.8-27b,1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0,qwen-shared,qwen3.8-27b,opencode-<exact-version>,2,1
```

`task_key` is optional descriptive provenance. All other identity fields are required.
`max_retries` counts infrastructure retries of the same scientific attempt; it does not
increase pass@k.

Initialize once. Repeating the command with byte-equivalent rows in a different order is
safe. A changed plan is rejected.

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  init --plan results/rollout-plan.csv
```

The database, CSV, and event export are created with mode `0600`.

## Worker flow

Run one or more workers for each serving block. `claim` uses an immediate SQLite
transaction, so concurrent workers cannot receive the same row.

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  claim --worker-id qwen-shared-worker-1 --serving-block qwen-shared
```

The response contains the immutable `cell_id`, new `claim_id`, endpoint model id, exact
task version, and attempt. After Fleet creates the session, bind it to that claim:

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  start --cell-id "$CELL_ID" --worker-id "$WORKER_ID" --claim-id "$CLAIM_ID" \
  --session-id "$SESSION_ID"
```

Heartbeat well before the lease expires. A lease only detects a silent worker; expiry
never returns a cell to the queue.

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  heartbeat --cell-id "$CELL_ID" --worker-id "$WORKER_ID" --claim-id "$CLAIM_ID"
```

Mark grading, then accept only after the complete authoritative acceptance receipt exists.
A valid zero is accepted exactly like any other valid model outcome; no score is stored in
the ledger.

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  grading --cell-id "$CELL_ID" --worker-id "$WORKER_ID" --claim-id "$CLAIM_ID"

uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  accept --cell-id "$CELL_ID" --worker-id "$WORKER_ID" --claim-id "$CLAIM_ID" \
  --receipt-digest "$ACCEPTED_RECEIPT_SHA256"
```

## Failures and retries

An infrastructure failure moves to `retry_review`, never directly to `pending`:

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  retry-review --cell-id "$CELL_ID" --worker-id "$WORKER_ID" --claim-id "$CLAIM_ID" \
  --failure-code runner_exit_before_acceptance
```

After an authorized read-only reconciliation proves the prior session cannot become a
valid accepted result, approve the bounded retry with that evidence digest:

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  approve-retry --cell-id "$CELL_ID" \
  --reconciliation-digest "$RECONCILIATION_SHA256"
```

If reconciliation says no further run is allowed, use `terminal` with the same evidence
contract. Never treat an expired lease as permission to rerun.

## Human view

```bash
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 status
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 stale
uv run python -m evals.fleet.rollout_ledger --db results/rollouts.sqlite3 \
  export --csv results/progress.csv --events results/rollout-events.jsonl
```

The CSV contains current row state. The JSONL export is the ordered append-only audit
history. A controller should refresh both after every material transition, but neither
export participates in claiming or retry authorization.
