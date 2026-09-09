# Fleet cyber rollout ledger workers

This is the small control layer for the exact Qwen3.8-27B and GLM-5.3
pass@4 campaign. The model servers are persistent Fleet inference deployments;
the workers are bounded CPU Jobs that create environments, run OpenCode, submit
the authoritative result, ingest the session, clean up, and then claim another
cell.

## Source of truth

PostgreSQL is authoritative for the migrated distributed campaign. The preserved
`ledger.sqlite3` is historical evidence, never a live fallback. Read
[ROLLOUT_POSTGRES.md](ROLLOUT_POSTGRES.md) before operating this campaign.
`plan.csv` is the immutable 800-cell plan and
CSV/JSONL status files are human-readable projections only. A cell is one exact
model + task version + attempt. All four attempts for one model/task stay on the
same serving route.

The private `rollout_local_results` table indexes every completed local result
before Fleet session-catalog acceptance. It stores the numeric score, session and
verifier IDs, and the relative paths plus SHA-256 digests of the trace, result,
reward, ingestion, and cleanup files. Large traces remain immutable files under
the attempt directory rather than database blobs. This preserves our own result if
Fleet catalog ingestion or model attribution fails, while the public progress CSV
and event stream remain score- and trace-free. Existing artifacts are not
automatically backfilled; this contract applies to workers launched with schema v2.

States are:

- `pending`: safe to claim
- `claimed`: one worker owns it; a lease heartbeat is current
- `running` / `grading`: a Fleet session is being finalized
- `accepted`: session, verifier, ingestion, and cleanup all reconciled
- `retry_review`: something failed and a human must prove whether retry is safe
- `terminal`: intentionally closed and never repeated

There is no automatic scientific-cell retry. PostgreSQL claims only `pending`
rows on the exact route using `FOR UPDATE SKIP LOCKED`; a fresh claim token fences
every ownership transition. Lease expiry detects trouble, never permission to
requeue. The global execution-claim file is created before
the model call and is immutable, so another worker cannot silently repeat the
same stochastic cell.

## Fixed treatment

- OpenCode 1.18.27, digest checked at image build
- native compaction with automatic continuation
- 262,144-token context, 32,768-token output, 20,000-token compaction headroom
- only Fleet `bash` and `submit_report`; all local OpenCode tools are disabled
- exact task version, runtime seed, verifier version, model revision, and endpoint
- scores remain private in `rollout_local_results`; coordination receipts and
  exports contain no prompts, traces, flags, answers, or scores

## Serving routes

The plan keeps four explicit blocks: shared Qwen, dedicated Qwen, shared GLM,
and dedicated GLM. Endpoint identity is part of the ledger row and acceptance
receipt. The workers never create or hold a GPU model server.

## Rollout gate

For a new campaign, `rollout-ledger-four-route-canary-v1.yaml` illustrates one new cell per route,
four cells total. Do not widen until all four produce digest-valid accepted
receipts. A rejected canary writes a sanitized terminal receipt and exits the
CPU Job normally. New successors require the official renderer's alert-safe
policy: a clean-exit controller, backoff limit 2147483647, and no active deadline
or pod-failure policy. A restart or Job `Failed=True` still blocks creates and
requires notification; this policy must never hide an actual infrastructure failure.
Always server-dry-run and verify images, secret references, volumes, and endpoints.

The persistent state root is
`/mnt/sfs/jobs/chris-cyber-q38-glm53-pass4-ledger-v1`. Historical state begins
from the reviewed v50 snapshot: 66 accepted, 13 terminal/nonrepeatable, and 721
pending. Never replace the database with a CSV export.

To inspect progress without opening sealed data:

```sh
uv run --locked python -m evals.fleet.rollout_refiller \
  --namespace fleet-train-jobs \
  --ledger-pod chris-cyber-rollout-postgres-v1-0 \
  --ledger-kind postgres --target-per-route 8
```

Only widen a worker pool by a fixed reviewed amount, measure accepted rollouts
per minute by serving block, and keep task/attempt selection score-blind.
The historical v50 counts above are not a current status report. Reobserve live
state for every handoff. Initializing successors occupy slots before they claim a row.
