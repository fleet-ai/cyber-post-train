# Fleet cyber rollout ledger workers

This is the small control layer for the exact Qwen3.8-27B and GLM-5.3
pass@4 campaign. The model servers are persistent Fleet inference deployments;
the workers are bounded CPU Jobs that create environments, run OpenCode, submit
the authoritative result, ingest the session, clean up, and then claim another
cell.

## Source of truth

`ledger.sqlite3` is authoritative. `plan.csv` is the immutable 800-cell plan and
CSV/JSONL status files are human-readable projections only. A cell is one exact
model + task version + attempt. All four attempts for one model/task stay on the
same serving route.

States are:

- `pending`: safe to claim
- `claimed`: one worker owns it; a lease heartbeat is current
- `running` / `grading`: a Fleet session is being finalized
- `accepted`: session, verifier, ingestion, and cleanup all reconciled
- `retry_review`: something failed and a human must prove whether retry is safe
- `terminal`: intentionally closed and never repeated

There is no automatic retry. The global execution-claim file is created before
the model call and is immutable, so another worker cannot silently repeat the
same stochastic cell.

## Fixed treatment

- OpenCode 1.18.27, digest checked at image build
- native compaction with automatic continuation
- 262,144-token context, 32,768-token output, 20,000-token compaction headroom
- only Fleet `bash` and `submit_report`; all local OpenCode tools are disabled
- exact task version, runtime seed, verifier version, model revision, and endpoint
- scores remain private; coordination receipts contain no prompts, traces, flags,
  answers, or scores

## Serving routes

The plan keeps four explicit blocks: shared Qwen, dedicated Qwen, shared GLM,
and dedicated GLM. Endpoint identity is part of the ledger row and acceptance
receipt. The workers never create or hold a GPU model server.

## Rollout gate

Start with `rollout-ledger-four-route-canary-v1.yaml`: one new cell per route,
four cells total. Do not widen until all four produce digest-valid accepted
receipts. A rejected canary writes a sanitized terminal receipt and exits the
CPU Job normally, avoiding a cluster-wide failed-Job page. Pod/image/PVC failures
can still fail the Kubernetes Job, so always server-dry-run and verify every
referenced image, secret, volume, and endpoint before creation.

The persistent state root is
`/mnt/sfs/jobs/chris-cyber-q38-glm53-pass4-ledger-v1`. Historical state begins
from the reviewed v50 snapshot: 66 accepted, 13 terminal/nonrepeatable, and 721
pending. Never replace the database with a CSV export.

To inspect progress without opening sealed data:

```sh
uv run python -m evals.fleet.rollout_ledger \
  --db /mnt/sfs/jobs/chris-cyber-q38-glm53-pass4-ledger-v1/ledger.sqlite3 status
```

Only widen a worker pool by a fixed reviewed amount, measure accepted rollouts
per minute by serving block, and keep task/attempt selection score-blind.
