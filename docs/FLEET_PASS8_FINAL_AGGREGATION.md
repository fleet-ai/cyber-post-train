# Final matched Fleet pass@8 aggregation

This is the private final gate for the frozen Qwen3.8 base-versus-Teacher3K
step-1000 Fleet development comparison. It does not launch, retry, score, or
repair a rollout. It only aggregates the already-authoritative results after
all collection and score-blind reconciliation work is complete.

## What must be true before scores can be read

The comparison consists of the same 17 exact task versions under two model
arms. Protocol v2 starts from original seeds 46 through 53. If score-blind
infrastructure evidence proves any cell in one original seed unusable, the
entire two-arm seed pair is excluded. Excluded originals, sorted ascending,
map to fresh seeds 54, 55, and so on. The final cohort therefore still has
exactly eight separately declared pass@1 replica pairs and each task still has
eight attempts per arm. It never mixes one usable cell from an excluded
original with cells from its replacement.

The final gate consumes the self-digested protocol-v2 migration receipt and
its byte-identical `COMPARISON_DEFINITION.json`. This preserves the original
evidence hashes privately while making the final included seed roster and
whole-pair replacements explicit. The final definition is frozen at
`sha256:bff3b01e6dcc4b189c9acb6140fbf868fda74e9bef4288c39bab0490cfc49fd2`;
it includes seeds 46, 48, 49, 50, 54, 55, 56, and 57. If score-blind evidence
later invalidates any included seed, the current comparison cannot be edited:
a new versioned definition and receipt are required before score unseal. The
gate opens no score until it holds one
read-only, repeatable database snapshot for all 16 included replicas and has
proved all of the following:

- every source evaluator has an exact self-digested terminal receipt;
- the base and candidate receipt for each included seed carry the same matched
  protocol;
- the migration uses the deterministic whole-pair replacement mapping, names
  exactly eight unique included seeds, and supplies both arms for every fresh
  seed;
- every frozen evaluation plan uses the same OpenCode 1.18.27 harness files,
  treatment, images, task versions, sampling settings, and retry rule;
- the sole intended model revision and serving route are exact for each arm;
- every replica ledger contains the same 17 task versions exactly once;
- every cell is `accepted`, is classified `valid`, has no active lease or
  failure code, and has exactly one private local-result record;
- every accepted cell has one matching acceptance event; a result recovered
  from an already-completed stored session also needs its exact private
  reconciliation receipt; and
- there are no pending, active, review-held, terminally invalid, missing, or
  duplicate cells that could be counted as a model outcome.

Only after all 272 cells pass those checks does the process issue a query that
includes the private score column. All 16 database transactions stay open at
repeatable-read isolation, so the identity and score queries see the same
snapshot.

## Outputs and privacy

The job writes into a new directory and renames it into place only after every
file is complete. Existing output stops the run. Private files use mode `0600`:

- `PRIVATE_TERMINAL_INDEX.json` binds the 16 terminal receipts and database
  plans;
- `PRIVATE_SCORED_OUTCOME_INDEX.json` preserves exact per-cell scores and
  evidence digests;
- `PRIVATE_ANONYMIZATION.json` preserves a randomly shuffled task mapping;
- `SANITIZED_AGGREGATE.json` is the only input intended for the public report;
  and
- `FINAL.json` binds the preceding receipts.

The sanitized file contains 17 randomly ordered task rows. A row contains only
the number of valid attempts, the number of technical failures, and whether at
least one of the eight attempts fully solved the task. It contains no task,
cell, session, verifier, prompt, response, flag, reward, or trace identity.
`cyber_post_train.public_eval_import` independently checks the schema and
recomputes the public metrics and uncertainty interval.

## Render and preview; do not create early

Render the immutable CPU-only package without touching Kubernetes:

```sh
uv run python scripts/render_qwen38_fleet_pass8_final_aggregate.py \
  --migration-receipt /reviewed/protocol-v2/MIGRATION_RECEIPT.json \
  --output /safe/new/fleet-pass8-final-render
```

The renderer first checks the migration receipt, its embedded comparison
definition, and the sibling definition file. The package then contains that
self-digested study plan and exact SHA-256 hashes for the migration receipt,
comparison definition, aggregator module, runner, compressed bundle, and
rendered YAML. Its Job is zero-GPU, c1, create-once, and carries the root
annotation `fleet.ai/failure-alerts: "off"`.

Before any later create, run the exact bundle through Kubernetes server dry-run
twice, save both JSON replies, then validate them:

```sh
uv run python scripts/render_qwen38_fleet_pass8_final_aggregate.py \
  --migration-receipt /reviewed/protocol-v2/MIGRATION_RECEIPT.json \
  --render-root /safe/new/fleet-pass8-final-render \
  --preview-one /safe/new/preview-1.json \
  --preview-two /safe/new/preview-2.json \
  --preview-receipt /safe/new/PREVIEWS.json
```

The validator removes only ordinary server metadata and the Job UID-derived
selector labels. The two previews must carry different valid server-assigned
Job UIDs, and neither preview may contain a GPU resource in any container
class. Both preview files must be distinct exact regular files and are hashed
from the same bytes that are parsed. Before comparing either preview, the
validator reopens the source bundle, reconstructs the exact c1 zero-GPU Job and
immutable ConfigMap, rebuilds the study plan from the independently supplied
authoritative migration receipt, and recomputes the full render receipt. The
preview command therefore requires the same reviewed migration-receipt path;
the plan embedded in the render is never accepted as its own authority.
Admission may add only a small set of exact, safe Kubernetes
defaults; a sidecar, secret source, host namespace, security privilege, volume,
resource claim, accelerator, changed command/image, or changed service account
fails closed. A green render or preview does not mean the 272 outcomes are
complete; the private final Job must still pass the live database and
terminal-receipt gates above.
