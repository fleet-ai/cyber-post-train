# Fleet blackbox evaluations

Use `cyber-post-train eval` for a new experiment. It reuses the OpenCode runner
and PostgreSQL claims without assuming 100 tasks, two models or pass@4.
Older campaign launchers remain historical compatibility paths, not defaults.

## Workflow

```sh
cyber-post-train eval prepare my-eval.yaml --output /shared/my-eval
cyber-post-train eval preflight /shared/my-eval
cyber-post-train eval init /shared/my-eval
cyber-post-train eval run /shared/my-eval shared worker-001 --limit 4
cyber-post-train eval status
```

Prepare is offline. Preflight uses GETs to verify Fleet-team access, exact
task/runtime/verifier bindings, ready inference routes and staged Linux/amd64
Docker images, including the release label and actual OpenCode version. The
execution host checks its images again before claims. Its offline startup check
uses the actual controller UID, an explicitly set HOME and a private mounted
directory; `--version` alone misses startup permission errors. Build the agent with
`evals/fleet/Dockerfile.opencode` and freeze its resulting digest. Preflight
creates no challenge or scored session. Init writes only
an empty dedicated PostgreSQL database; it is not a migration/reset command.

Run executes a bounded batch **on the machine running the command**. Use an
authorized CPU worker with Docker and shared durable storage, not an idle GPU.
The generic GPU Jobs API requires at least one GPU and must not be used solely
for this controller. Cluster packaging of this new CLI remains under qualification;
do not redeploy existing campaigns merely because local tests pass.

Set `FLEET_API_KEY` and `ROLLOUT_DATABASE_URL` through the approved secret manager.
Neither value belongs in argv, YAML, receipts or Git. Workers verify the same
frozen plan and claim only pending rows. Use a distinct worker ID per batch.
Expired or ambiguous attempts are held for review, never automatically retried.
The live serving-profile check for each planned session runs only after that
session's atomic ledger claim. A transient profile-check failure is therefore
held as infrastructure-invalid on one exact cell; it cannot consume a worker
slot while leaving an unrelated cell silently pending.

## Configuration

The YAML fields are:

| Field | Meaning |
|---|---|
| `name` | Unique owner-specific experiment name |
| `task_set` | Reviewed JSON/YAML file containing exact task/runtime tuples |
| `models` | Alias → `repository`, immutable `revision`, Fleet `session_model` |
| `routes` | Serving-block assignments and expected live model/server profiles |
| `images` | `agent` and `proxy`: OCI digest references or exact local Docker image IDs |
| `harness` | Explicit supported OpenCode version, tools, compaction and budgets |
| `sampling` | `temperature`, `top_p`, `seed`, enforced by the fixed proxy |
| `pass_k` | Attempts per model/task |
| `concurrency` | Simultaneous sessions **per worker**, not a global reservation |
| `max_reviewed_infrastructure_retries` | Zero or one; defaults to zero. A retry still requires a separate, score-blind human or agent review. |
| `training_data_eligible` | False by default; training collection must explicitly opt in |

The task set's `tasks` rows require `task_key`, `task_version_id`, `env_key`,
`env_version`, `environment_version_id`, `data_key` and `data_version`.
Use reviewed UUIDs and versioned runtime tuples, never mutable current selectors.
Historical outcome fields are not copied into the new plan.

Exact task GET must also expose starting data (`seed_config` or both legacy
data fields). Fleet deliberately rejects historical versions missing this
record; an instance POST cannot recover it. Preflight fails before any rollout.
Preserve the held attempt and review a new exact selection, never borrow a
mutable current version or silently replace the task.

For a new Qwen held-out campaign, read
[`docs/FLEET_HELDOUT_EVALUATION_PRECREATE_GATE.md`](../../docs/FLEET_HELDOUT_EVALUATION_PRECREATE_GATE.md)
before creating anything. Historical Job YAML and prepared configs are not
launch permission. Use `cyber-post-train eval heldout-create` with a new sealed
launch packet, explicit Kubernetes context, and new local create-intent journal.
It performs a fresh duplicate census, two server dry-runs, and requires the
server-rendered root `fleet.ai/failure-alerts: "off"` annotation before its
single create request. After the Job is terminal, use
`cyber-post-train eval heldout-terminal-collect` to record a score-blind
reconciliation; neither command reads rollout traces or interprets a capability
result. The census uses only exact server-side name or Job-label selectors; its
terminal Workload read uses the created Job UID. It never lists the shared
namespace wholesale.

The packet also seals the full representative split and the selected split role.
The evaluator route must contain exactly the selected task versions. Its shared
comparison protocol seals the common harness, sampling, evaluator-image, and
retry treatment for every arm. This makes new larger train/development/final-
test splits safe to introduce without silently changing the held-out family set
or treatment used by a baseline-versus-checkpoint comparison.

## Resumable pass@4 campaign adapter

`evals.fleet.campaign_adapter` connects these sealed source Jobs to the generic
`evals.campaign` controller. The campaign still has one independent cell for
each model, task and attempt. A small bindings file may group several task cells
that share one model and seed onto one CPU source Job. One declared leader cell
creates that Job; the other cells reuse its immutable Job UID. This preserves
per-task failure isolation without creating duplicate Jobs.

The bindings file is immutable JSON with schema
`cyber_fleet_campaign_bindings_v1`. It contains:

- today's UTC rollout census: exact receipt digest, already-used count and the
  fixed daily cap of 500;
- each source packet's relative path and SHA-256 digest;
- the leader and complete cell list for each source Job;
- each campaign cell's exact task version, model alias/revision and source
  attempt.

Before creation, the adapter recompiles the plan with the evaluator code sealed
inside the source packet, requires an exact cell match, checks the Fleet account
and live serving route, runs the existing duplicate census, and relies on the
existing double server dry-run. The source launcher still performs its own
fresh checks immediately before its one create request. The server-rendered
root Job must contain `fleet.ai/failure-alerts: "off"`, request no GPUs, use
priority `c1`, have no retry, and have a bounded deadline.

Daily capacity is reserved under the generic campaign's canonical local state
directory while holding a file lock. The gate counts the frozen already-used
census plus every persistent source-group reservation. Reopening the same exact
group is idempotent; a changed reservation or a total above 500 fails closed.
Use one authoritative census before the first campaign reservation each UTC
day, and reuse that exact census for every campaign on the same controller host.

Terminal observation is score-blind. The adapter binds the Job UID, reuses the
existing terminal collector, and reads only one exact PostgreSQL cell's state
and receipt digest. Cells become accepted independently. An accepted worker
receipt already proves the Fleet instance was closed, its containers were
removed, and the authoritative session was preserved; the adapter exposes that
cleanup fact without exposing the score or any rollout content. A failed cell
does not stop sibling cells, and neither rollout nor score creation is replayed
after an uncertain launch intent.

The four driver commands use this shape (add the corresponding prior-receipt
argument for `ready`, `launch`, or `observe`):

```sh
python -m evals.fleet.campaign_adapter preview rollout \
  {packet} {receipt} --bindings /absolute/bindings.json --context fleet-prod
```

The score phase is only a local evidence-binding phase: the Fleet worker has
already run the exact task verifier. It never launches another rollout or judge.

Each route requires `model` (one alias above), `served_id`, `task_versions`,
`endpoint_origin: https://inference.flt.build`, and these expected projections:

- `catalog`: engine, precision, tensor_parallel_size.
- `model_info`: model_path, model_type, architectures.
- `server_info`: model_path, context_length, tp_size, quantization,
  kv_cache_dtype, reasoning_parser, tool_call_parser.

For data-parallel servers, additionally bind `dp_size` and `load_balance_method`
in `server_info`. Catalog `tensor_parallel_size` is a control-plane declaration,
not a substitute for the runtime's TP/DP values. Investigate disagreements against
the exact Pod arguments, then freeze the observed profile in a new plan; do not
edit a running plan or treat a changed serving topology as a matched control.

Obtain profiles from the exact deployment and read-only gateway endpoints.
Preflight and each new session compare them; never update a profile simply to
accept unexplained drift. Every model must cover every task exactly once.
All attempts for a model/task stay in one serving block. Account for other
workers and existing consumers when setting total authorized concurrency.

A supported harness/sampling block:

```yaml
harness:
  harness: opencode
  harness_version: 1.18.27
  release_asset_sha256: sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702
  provider_adapter: "@ai-sdk/openai-compatible"
  context_management: opencode_1.18.27_native_compaction_autocontinue_v2
  context_window_size: 262144
  compaction_headroom_tokens: 20000
  max_output_tokens: 32768
  max_model_requests: 600
  timeout_seconds: 28800
  tools: [bash, submit_report]
  tool_catalog_sha256: sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a
sampling: {temperature: 0.6, top_p: 0.95, seed: 42}
pass_k: 4
concurrency: 1
max_reviewed_infrastructure_retries: 0
training_data_eligible: false
```

These are a specific treatment, not universal benchmark defaults. Tool/schema
drift stops execution. The agent's other built-in tools are disabled and
credentials remain outside its network boundary. The model proxy enforces
sampling/model/output limits. Attempt N uses base seed + N − 1 (modulo 2³¹), so
pass@k does not deliberately repeat the same seed. A seed does not guarantee deterministic
GPU execution.

The retry limit is not an automatic retry switch. An infrastructure-invalid
cell remains in `retry_review` until a reviewer proves that no valid outcome
exists and records a reconciliation digest. A valid unsuccessful solution is a
real result and must never be retried. For matched comparisons, freeze the same
limit and the same arm-blind decision rule before launch. Do not splice accepted
cells from one experiment into a successor experiment.

When recording a Kubernetes server dry-run digest, use `stable_job_preview`.
The API generates a fresh Job UID and repeats it in the selector and two Pod
labels on every dry run; hashing the raw response therefore produces a false
change. The helper removes only those UID-derived fields and ordinary runtime
metadata. Stable admission changes such as the root failure-alert annotation,
priority, suspension, resources, command, environment, and image remain bound
by the digest. Run the preview twice and require the normalized hashes to match.

The v2 compaction policy reserves one full response of growth **plus** the declared
headroom before the native post-response overflow check. Its trigger is
`context − 2 × max_output_tokens − compaction_headroom_tokens`; room must remain
for input. This prevents a long response from consuming the space reserved for
the next compaction request. The headroom still must cover tool-result and summary
overhead; this is not a guarantee against arbitrary request growth. Native-binary
tests prove both compaction and automatic continuation with synthetic responses.
Existing v1 plans retain their earlier trigger and cannot be silently migrated.
See the [observed context-overflow incident](../../docs/evidence/cleanup-eval-v7-terminal-20260911.json).

## Results and interpretation

The shared directory holds frozen inputs/runtime hashes, plan CSV, preflight,
execution claims and per-attempt private artifacts. PostgreSQL owns state and
the result index. Local trace/results are preserved before catalog acceptance.
Status is score-blind.

Acceptance requires normal terminal model events, exact authoritative grading,
a completed catalog session with matching model/verifier IDs, private results
and environment/container cleanup. A genuine unsuccessful solution is valid.
Truncation, missing ingestion and ambiguous scoring remain review cases.

These generic runs are **serving-block descriptive evaluations**. Gateway
metadata alone does not prove every serving byte or deployment UID. A causal
base/post comparison additionally requires the matched protocol, accepted export,
immutable serving-image/weight receipts and live parity from
[the evaluation skill](../../skills/cyber-eval-parity/SKILL.md).
Never silently pool serving blocks or use external benchmark observations to tune.

See [PostgreSQL operations](../../docs/ROLLOUT_POSTGRES.md) for reconciliation
and backup ownership. Historical Qwen3.6/Qwen Code/exact-800 campaign artifacts
remain reproducible in Git; their names, ledger and retries are not starter templates.
