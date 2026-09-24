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
`sha256:9813713ef2ab023ac6c64df494ca7cbf39b920e8fba2f05f5949daaa006479f1`;
it excludes original seeds 46, 47, 49, 50, 51, 52, and 53 as whole pairs and
includes original seed 48 plus replacement seeds 54 through 60. If score-blind evidence
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
- every included arm is bound to one complete launch-packet identity, exact
  packet digest, compiled evaluation-plan digest, and the exact seven evaluator
  file digests; the retained seed-48 bindings additionally match their frozen
  packet, identity, and plan digests;
- every frozen evaluation plan uses the same OpenCode 1.18.27 harness files,
  treatment, images, task versions, sampling settings, and retry rule;
- the sole intended model revision and serving route are exact for each arm;
- every replica ledger contains the same 17 task versions exactly once;
- every cell is `accepted`, is classified `valid`, has no active lease or
  failure code, has exactly one private local-result record, and that record
  proves exit zero plus a normal OpenCode completion;
- every ordinary accepted cell has one exact, self-digested `ACCEPTED.json`
  that matches its cell, execution, session, verifier, configuration, and
  completed cleanup/ingest lifecycle; a reconciliation may preserve an
  already-scored stored session, but an output-limit or nonzero-process
  lifecycle still remains held and cannot enter a capability aggregate;
- the pre-reconciliation terminal census has exactly as many unresolved cells
  as the final ledger has valid, receipt-backed reconciliations; and
- there are no pending, active, review-held, terminally invalid, missing, or
  duplicate cells that could be counted as a model outcome.

Only after all 272 cells pass those checks does the process issue a query that
includes the private score column. All 16 database transactions stay open at
repeatable-read isolation, so the identity and score queries see the same
snapshot. The earlier query excludes not only the score itself, but also the
full-record, result, reward, and artifact hashes that could act as
low-entropy commitments to that score.

## Outputs and privacy

The job writes into a private temporary directory, flushes every file, and uses
an atomic no-replace rename only after the package is complete. Existing output
or a concurrent claimant stops publication without overwriting either tree.
Evidence JSON and its SHA-256 always come from the same single file read, so a
path cannot be parsed and then silently rebound to different bytes. Private
files use mode `0600`:

- `PRIVATE_TERMINAL_INDEX.json` binds the 16 terminal receipts and database
  plans;
- `PRIVATE_SCORED_OUTCOME_INDEX.json` preserves exact per-cell scores and
  evidence digests;
- `PRIVATE_ANONYMIZATION.json` preserves a randomly shuffled task mapping;
- `SANITIZED_AGGREGATE.json` is the only input intended for the public report;
  and
- `FINAL.json` binds the preceding receipts.

The private task mapping includes a fresh 256-bit random nonce, so its public
commitment cannot be brute-forced by enumerating the roughly 48-bit task
permutation. The nonce and mapping never enter the public file. The sanitized
file contains 17 randomly ordered task rows. A row contains only
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
annotation `fleet.ai/failure-alerts: "off"`. It performs no package-resolution
step at runtime: the three required wheels are named by exact
`files.pythonhosted.org` artifact URLs, byte sizes, and SHA-256 hashes, and are
size- and hash-checked individually before each wheel is unpacked. None of the
unpacked modules is imported or executed until all three downloads and checks
finish. This preserves byte identity but still depends on that artifact host
being reachable; an unavailable download fails before any evidence is opened.
The source evaluators created their
evidence trees as root with mode 0700, so the reader deliberately runs as UID
0/GID 100. It has no service-account token, no Linux capabilities, no
privilege escalation, a read-only root filesystem, and only a read-only SFS
mount. The same reader proves every plan-bound source root is an exact
directory and every required EVAL and terminal receipt is an exact readable
regular file before it opens a database snapshot. Its database sessions start
with PostgreSQL's read-only default. It writes the five final files only to a
shared temporary volume, then makes them group-readable.

A separate publisher runs as the proven SFS writer UID 1000/GID 100. It has no
database credential and cannot see the source SFS tree. It validates the exact
five-file roster, every self digest, and the final receipt's four links to the
terminal, outcome, anonymization, and public receipts in the temporary volume.
It alone receives a writable mount, limited to the existing
`chris-q38-study-corpora-v1/launch-controls` subtree. After proving the exact
parent owner and mode and a temporary write/remove probe, it copies into a
private temporary directory and performs one atomic no-replace rename. This
two-process handoff lets the root reader open historical mode-0700 evidence
without giving that process any writable SFS path.

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
