# Working together

Read [AGENTS.md](AGENTS.md) first. This repository contains shared mechanisms and
versioned scientific controls, not permission to submit workloads or unseal results.
Use the relevant evaluation, training, evidence or maintenance skill for your task.

## Get a reproducible checkout

Start from freshly fetched `origin/main` in a dedicated branch/worktree. Preserve
another operator's dirty tree, private artifacts and running campaign checkout.
Use Python 3.11 or 3.12 and the committed `uv.lock`:

```sh
uv sync --locked --extra dev --extra train
uv run --locked cyber-post-train doctor
```

No credentials are needed for these commands. Use `.env.example` only as a template;
never overwrite a colleague's `.env`. Share access through the team's secret manager,
not Git, chat, issue comments, command arguments or saved shell output.

## Repository map and authority

- `cyber_post_train/`: small CLI and shared generic Jobs API boundary.
- `evals/fleet/`: frozen Fleet evaluation treatment, rollout workers and PostgreSQL backend.
- `training/`: authorized Fleet-data export, normalization, SFT/RL and checkpoint handling.
- `evals/webexploitbench/`, `evals/exploitgym/`: evaluation-only external benchmarks.
- `configs/`: versioned model/run controls, not global model defaults.
- `docs/`: protocols, operator runbooks, dated evidence and historical narrative.
- `skills/`: reusable decision guidance; exact live IDs and results do not belong here.

For current rollout operations use [ROLLOUT_POSTGRES.md](docs/ROLLOUT_POSTGRES.md).
PostgreSQL is the distributed queue; preserved SQLite and status exports are not.
Immutable receipts and exact runtime UIDs outrank narratives. Documentation snapshots
must state their observation time and must not be treated as live health checks.

## Test without touching a campaign

The full suite includes optional CPU training-artifact tests. Install the locked
training extra too when running everything; no model download or paid run is needed:

```sh
uv sync --locked --extra dev --extra train
uv run --locked pytest
```

PostgreSQL integration tests skip unless **both** `TEST_POSTGRES_DSN` and
`TEST_POSTGRES_DISPOSABLE=1` are explicitly set. Use only a fresh local test instance
bound to loopback, database name `cyber_post_train_test`, never a tunnel or production
endpoint. The fixture rejects nonlocal hosts, other database names and DSN query
overrides before connecting. Each test creates a uniquely named schema and cleans
up only that schema; it never drops shared ledger tables. CI uses an ephemeral
PostgreSQL service with synthetic rows. Never copy campaign records into tests.

The focused no-training test set is:

```sh
uv run --locked pytest tests/test_rollout_ledger.py tests/test_rollout_worker.py \
  tests/test_rollout_refiller.py tests/test_rollout_successor.py \
  tests/test_rollout_archive.py tests/test_rollout_postgres.py \
  tests/test_rollout_postgres_status.py
```

Run Ruff on files you change, `git diff --check`, and all relevant tests before
review. Do not perform a repository-wide formatting rewrite to hide an unrelated
baseline issue. If a suite needs unavailable optional dependencies or infrastructure,
report that explicitly rather than claiming it passed.

## Review and operational ownership

Keep one coherent change per PR. Describe the invariant, focused regression test,
privacy boundary, migration/backward-compatibility impact, and residual risk.
Keep shared tooling changes separate from scientific treatment changes; never
silently change model/harness/task selection while maintaining queue infrastructure.

Before committing, check the staged **file list** and scan for credentials/private
artifacts without printing matches. `.gitignore` is not a security boundary. Do not
commit databases, environment files, raw output, traces, caches, or credential-rotation
material. Historical sanitized evidence must not be rewritten as cleanup.

Designate one live campaign creator; everyone else is read-only. A monitor heartbeat,
human terminal and another agent are not independent authorities to fill the same
slot. Transfer creator ownership explicitly, preserving exact pending submission
identities. Code review does not grant deployment or merge authority; seek the
current owner's approval for shared-repository publication and live changes.

Do not reapply campaign manifests to onboard. Request least-privilege access, review
the current sanitized handoff, verify immutable bindings and credential validity,
then observe through the official refiller. A fresh clone can diagnose and test
without being allowed to submit, repair, delete, unseal or deploy.
