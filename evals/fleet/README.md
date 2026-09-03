# Fleet blackbox baseline

This evaluator runs the 160 registered tasks from Fleet job
`a62dd51f-a52b-4941-8207-4679e4b25b51` with GLM-5.2 on hosted Agent Runtime v1.

The inference gateway alias and Agent Runtime name are intentionally distinct:

```text
inference.flt.build:  glm-5.2-fp8 (currently not routed for this account)
Fleet Agent Runtime: z-ai/glm-5.2
harness:             Agent Runtime v1 default (no legacy harness override)
```

The source job's immutable launch input is read from Fleet's job-detail API. No
task list is copied by hand. Only canonical
`cysec1-2-*__blackbox_ctf_v1` keys are accepted. Registered tasks carry their
own stamped environment and data/verifier bindings; this launcher does not
override those fields. Resolved session provenance should be archived with the
event log after each job completes.

Every command requires `FLEET_API_KEY`. It first verifies that the credential
belongs to team `fleet` (`a1025f0b-ad67-49fc-a023-51800ab43e84`). Secrets are
never included in plans, events or error messages.

```bash
# Identity and model-catalog check
uv run python -m evals.fleet.cli preflight

# Secret-free plan for the full corpus: 27 jobs, each <= 6 sessions
uv run python -m evals.fleet.cli plan

# Default smoke dry-run: one mid-band task, one session
uv run python -m evals.fleet.cli launch

# Paid one-session smoke launch
uv run python -m evals.fleet.cli launch --submit

# Full pass@1 baseline. Inspect the plan before adding --submit.
uv run python -m evals.fleet.cli launch --full --limit 160 \
  --run-name glm52-fleet-blackbox-full-v1
uv run python -m evals.fleet.cli launch --full --limit 160 \
  --run-name glm52-fleet-blackbox-full-v1 --submit
```

`--submit` is explicit because job creation is paid. Stable idempotency keys
are derived from each logical batch, so retrying an identical command does not
create a second logical measurement. Receipts are appended to
`runs/fleet/events.jsonl` with mode `0600`.

Do not attach `required_key_capabilities: ["cyber"]` to this GLM route. The
deployed capability tag is currently for the provider routes that hold a
matching cyber key; the Fleet-hosted GLM Agent Runtime route is selected by its
provider-qualified model name instead.

## Official Qwen Code arm

The intended Qwen arm must reuse the exact WebExploitBench checkpoint
(`Qwen/Qwen3.6-27B` at
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`) and official Qwen Code
`0.22.3`. It must not silently substitute Fleet Agent Runtime's current
`opencode` fallback for `qwen/*` models.

The readiness record is
`manifests/qwen36-27b-qwen-code-smoke-readiness.json`. As of 2026-08-31, the
deployed Jobs API advertises only `codex`, `grok`, `grok-bot`, `claude_code`,
`muse`, `antigravity`, and `opencode`; latest Theseus `main` likewise maps
Qwen to `opencode`. The one-session launch is therefore intentionally blocked
until both the exact harness and checkpoint binding are deployed and readable.

Once every recorded gate is proven, first render the one-session plan by
removing `--submit` from `blocked_launch_command` in the readiness record.
Review the emitted model, harness, task key, budgets, and idempotency key, then
run the exact command with `--submit`. Do not scale past one task until its
session reaches a terminal state, the verifier executes, and the provenance
receipt confirms the expected harness and model revisions.

### Train-only prompt-curriculum pilot

`prompt_curriculum.py` prepares (but deliberately does not create or launch) a
small reward-acquisition screen for Qwen3.6. The frozen v1 plan selects two
different application families that each produced four valid, authoritative
zero-reward outcomes in the training-only calibration snapshot. Each task gets
one four-member task-group dry run: the exact source prompt plus three
cumulative, generic process-cue levels. Runtime data, atoms, environment,
verifier, flags, model, Qwen Code harness, and task-facing tools remain fixed.

The offline gate proves the split and cited outcome identities without reading
prompt or trace bodies:

```bash
uv run python -m evals.fleet.prompt_curriculum validate \
  --config evals/fleet/configs/qwen36-27b-prompt-curriculum-pilot-v1.json \
  --split configs/data/fleet-a62-task-split-v1.json \
  --campaign-state /private/path/to/campaign-state.json
```

Live preparation additionally requires a Fleet-team credential. It performs
read-only exact-version hydration, requires an integer-versioned Artifact
Registry task-graph locator, and writes one private task-group payload plus a
prompt-free hash receipt per task beneath an ignored, mode-0700 output root.
Use a `results/` destination. Existing output is never replaced.

For review, prefer `prepare-review`. It makes the same exact three predeclared
GETs (Fleet account plus the two exact task versions), records a prompt-free
request audit, and writes only `review-plan.json`. The plan contains prompt
hashes and exact source bindings, never prompt bodies, traces, flags, or hidden
task data. Its invariant block is scoped to the prepared pre-create payload;
created members must still be re-hydrated and verified after any later approved
task-group creation.

```bash
uv run python -m evals.fleet.prompt_curriculum prepare-review \
  --config evals/fleet/configs/qwen36-27b-prompt-curriculum-pilot-v1.json \
  --split configs/data/fleet-a62-task-split-v1.json \
  --campaign-state /private/path/to/campaign-state.json \
  --out-dir results/qwen36-prompt-curriculum-review-v1
```

```bash
uv run python -m evals.fleet.prompt_curriculum prepare \
  --config evals/fleet/configs/qwen36-27b-prompt-curriculum-pilot-v1.json \
  --split configs/data/fleet-a62-task-split-v1.json \
  --campaign-state /private/path/to/campaign-state.json \
  --out-dir results/qwen36-prompt-curriculum-pilot-v1
```

This command makes no POST request. Before a later create-once task-group
operation or paid run, reviewers must re-hydrate every created member and prove
that its environment/data/runtime-seed/verifier bindings match the source,
confirm no prior receipt already owns the exact signature, and repeat the Fleet
team preflight. Each eventual job is exactly four `pass_k=1` sessions, below
the six-session cap. The selection rule is the least revealing level whose
measured Qwen success rate falls in the predeclared 20–70% interval.

### Self-hosted exact-harness canary

`self_hosted.py` is the non-substituting route for the same official Qwen Code
and exact gateway checkpoint. It provisions the exact registered task version,
exposes only the task's Fleet MCP tools to the agent, and grades through the
server-owned version-scoped rollout-reward authority. The agent container gets
neither the Fleet API key nor the runner token; two fixed-upstream proxies hold
those credentials outside the agent's network boundary.

The scored path is intentionally fail-closed until the two routes specified by
the config are proven either by deployed OpenAPI or by their non-mutating
report-only guards. These are the contracts introduced by Theseus PR #28252:
exact task-version provisioning injects server-owned cyber evidence, and exact
task-version scoring validates all production bindings.
The raw Qwen chat JSONL is the canonical trace. A normalized copy retaining
assistant thinking, tool calls, and tool observations is supplied to scoring
and Fleet trace ingestion. All artifacts are marked ineligible for training.

```bash
# Read-only task identity plus deployed-route gate.
evals/fleet/scripts/submit_selfhosted_qwen_smoke.sh preview

# After the route gate passes, create one suspended, queue-managed CPU canary.
evals/fleet/scripts/submit_selfhosted_qwen_smoke.sh submit
```

The canary is one task, one session, and one model. Its Kubernetes Job is
`chris-cyber-qwen36-qcode-fleet-smoke-v1-r1`; it uses LocalQueue `training-lq`,
never cancels another workload, and refuses to replace an existing Job or
ConfigMap. The `-r1` suffix preserves the original infrastructure-terminal
attempt, which stopped on a transient legacy-roster HTTP 503 before instance
creation, model launch, or scoring. The corrected runner treats that large
source-job roster as provenance only: it checks the frozen task/version through
the targeted task route, then the versioned create/score authorities hydrate and
revalidate the exact pair. Idempotent Fleet reads have bounded transient
retries; mutating requests never retry. Do not scale it until cleanup, raw trace,
authoritative reward, session ingestion, model identity, and harness identity
are all terminal and verified.

### Sealed 20-task Qwen Code baseline

`holdout.py` runs the exact untouched `test` rows from
`configs/data/fleet-a62-task-split-v1.json` sequentially with the same model,
Qwen Code, task-version provisioning, scorer, and per-task limits proven by the
canary. The submit preflight resolves each immutable task version through its
targeted route and freezes a prompt-free receipt into the Job's ConfigMap. The
Job rechecks that receipt before creating an instance. It never reads the large
legacy source-job roster and never copies prompt or verifier content into the
receipt.

```bash
# Revalidate all 20 bindings and render the single queue-managed Job.
evals/fleet/scripts/submit_selfhosted_qwen_test20.sh preview

# Submit one CPU Job; tasks execute sequentially (max concurrency 1).
evals/fleet/scripts/submit_selfhosted_qwen_test20.sh submit
```

The Job is `chris-cyber-qwen36-qcode-fleet-test20-base-v1`. It plans exactly
20 pass@1 sessions, refuses replacement, records every exact task-version and
runtime receipt, and keeps every artifact ineligible for training. A per-task
infrastructure failure is recorded separately and does not become a model zero.

### Sealed 20-task Qwen3.8 base baseline

The Qwen3.8 baseline inherits the same exact untouched `test` task-version rows
without reading prompt or verifier bodies into source control. It pins
`Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, Qwen Code 0.22.3, the exact
tokenizer/chat-template digests, the live SGLang serving contract, the ordered
`bash`/`submit_report` tool catalog, a 600-request ceiling, and a 262,144-token
context. The run is evaluation-only and sequential pass@1.

```bash
# Live identity, exact-version task hydration, and server-side manifest preview.
evals/fleet/scripts/submit_selfhosted_qwen38_test20.sh preview
```

The create-only `submit` path is deliberately blocked until the exact active V3
reward canary has one authoritative model outcome with a nonzero verifier UUID
and complete cleanup. At that point the operator supplies the immutable V3
campaign root and frozen receipt through `Q38_CALIBRATION_ROOT` and
`Q38_CALIBRATION_RECEIPT`. The launcher derives a self-digesting sanitized gate
receipt, rechecks the live model and all 20 exact task versions, refuses an
existing Job or ConfigMap, and uses Kubernetes `create` rather than replacement.
The gate accepts a valid zero because it establishes harness/verifier execution;
it does not reinterpret that zero as capability or release any V3 train/dev
tasks. This protocol change does not itself authorize or submit the holdout Job.

### Historical-frontier-ranked 50-task Qwen3.8 capability sweep

The ranked-50 plan is a separate non-test capability baseline for the exact
Qwen3.8/Qwen Code identity above. It selects 50 distinct immutable `train`/`dev`
task versions from the frozen 160-task split and explicitly excludes all sealed
test rows. Historical ease is derived only from prompt-free aggregates of the
authorized 1,265-session Fleet source export: pass rate descending, then passes,
sessions, task key, and task-version UUID. The frozen selection spans historical
rates 1.0 through 0.75 (48 train, 2 dev) and binds its source export, split,
exclusion receipt, and complete ordered task list by digest.
This is a productivity-oriented capability-sweep ordering from historical
frontier-model outcomes, not an unbiased or Qwen-specific difficulty estimate.

Five prior Qwen3.8 attempts are excluded before ranking: both authoritative
valid zeros and the three unresolved V2 attempts. Excluding unresolved versions
is deliberately conservative because missing local output is not proof that the
authoritative scorer never executed. The selector never reads or emits prompt,
transcript, tool-output, or verifier-content fields.

That five-row receipt is exhaustive only for the sanitized local campaign
evidence currently available. The runtime `/v1/sessions` scan covers live,
non-archived sessions and cannot prove absence of archived sessions or a scored
attempt whose session ingestion failed. Paid submission therefore has a second
fail-closed gate pending an authoritative scorer-side/all-history duplicate
inventory. Do not describe the current proof as globally exhaustive.

The create-only controller executes the easiest task alone. It releases the
remaining 49 only after that task produces any authoritative model outcome,
including zero, with a nonzero verifier-execution UUID and verified instance and
container cleanup. Remaining tasks launch in easiest-first waves of at most
three. Every task is pass@1, valid zeros are preserved, and outputs remain
ineligible for training.

```bash
# Static selection and Kubernetes preview. This remains fail-closed until a
# sanitized post-incident credential-rotation receipt is supplied.
FLEET_CREDENTIAL_ROTATION_RECEIPT=/restricted/rotation-receipt.json \
  evals/fleet/scripts/submit_selfhosted_qwen38_ranked50.sh preview

# Paid launch only after review and explicit authorization.
FLEET_CREDENTIAL_ROTATION_RECEIPT=/restricted/rotation-receipt.json \
  evals/fleet/scripts/submit_selfhosted_qwen38_ranked50.sh submit
```

The submitter reads only the Secret's name, namespace, UID, resourceVersion,
and presence of the expected data key. It never decodes the credential. The
evaluator atomically reads the Secret metadata and value in memory at execution
time, revalidates the same sanitized rotation receipt, and passes the value only
through the runner process environment. The value is never written to a
ConfigMap, argument, log, or receipt. An existing Job, ConfigMap, RBAC object,
or single-use output root is a hard refusal.

Paid submission is also deliberately blocked until deployed OpenAPI and a
behavioral probe prove Fleet's zero-message session-ingestion contract. The
ranked-50 client has a dedicated runtime-evidence-only path that submits
`messages: []`, no trace metadata, and requires the response to echo the exact
session, score, verifier, task-version, and instance bindings with
`message_count: 0`. It never calls the full-trace ingestion path. Local task
execution also uses a per-task `emptyDir` scratch root and persists only
minimized receipts to SFS. Do not remove this privacy gate or represent the
ranked-50 plan as launch-ready until the deployed contract is independently
probed.

The authoritative score path is separately constrained to Verifier Contract v3.
Preflight requires every exact task version to expose the complete `2 / 1 / 3`
cyber contract, and the score request contains only the instance ID and scoring
modes—never `conversation` or `final_answer`. A legacy/v2 task fails before any
task instance is created.

The rotation proof is reusable by WebExploitBench and this Fleet run. Capture
metadata before rotation, have the credential owner rotate the Secret, capture
metadata after rotation, then build the self-digesting receipt. The capture
script refuses an existing destination and emits no Secret value:

```bash
evals/fleet/scripts/capture_fleet_secret_metadata.sh /restricted/fleet-api.before.json
# Credential owner rotates fleet-train-jobs/fleet-api out of band.
evals/fleet/scripts/capture_fleet_secret_metadata.sh /restricted/fleet-api.after.json
uv run python -m evals.fleet.qwen38_fleet50 credential-receipt \
  --before-metadata /restricted/fleet-api.before.json \
  --after-metadata /restricted/fleet-api.after.json \
  --confirmed-by "$USER" \
  --rotation-completed-at 2026-09-02T00:00:00Z \
  --output /restricted/rotation-receipt.json
```

The builder requires the UID or resourceVersion to change and records only the
before/after metadata, accountable confirmer, completion time, data-key
presence, and receipt digest. The launcher re-reads live sanitized metadata and
requires an exact match to the `after` record.

### Interrupted-attempt reconciliation

The self-hosted runner writes `resource-plan.json` before starting local
containers and writes an exclusive, durable `scoring-intent.json` immediately
before its one non-retried authoritative scoring request. If the controller is
terminated outside Python, these intent receipts distinguish an unfinished
local attempt from permission to repeat it. Both the intent and its exact
request digest are covered by the intent's own self-digest, and create-once
publication syncs the file and its parent directory before returning.

`reconcile.py` inspects an existing terminal Qwen trace without emitting prompt,
tool arguments, responses, or final-answer content. It always sets
`rerun_model=false`. Before allowing the existing trace to be scored, it
requires a complete observation from the authoritative verifier store, bound to
the exact run, instance, evidence-run, task, and task-version IDs and protected
by a self-digest. One existing verifier result is recovered; multiple or
conflicting results are refused; a new score is eligible only when the complete
lookup proves that no result exists and the exact instance is still running.
Docker cleanup is similarly planned only from a self-digesting durable resource
plan whose exact hash-derived containers and private network match a complete
read-only resource snapshot. Legacy attempts without that resource plan refuse
cleanup rather than inferring ownership from a snapshot.

```bash
# Read-only and fail-closed without an authoritative observation.
uv run python -m evals.fleet.reconcile path/to/interrupted-attempt

# Still read-only: produce a reviewed score/cleanup plan from independently
# collected evidence. This command never calls Fleet, Docker, or the model.
uv run python -m evals.fleet.reconcile path/to/interrupted-attempt \
  --authority-observation path/to/private-authority-observation.json \
  --resource-snapshot path/to/private-resource-snapshot.json
```

The current public rollout-reward API has no read-by-evidence-run endpoint, so
an operator must obtain the authority observation through an authorized
read-only verifier-store export. Absence of a local `reward-result.json` is not
proof that scoring never happened. Do not score, clean up, or advance campaign
state while that lookup is absent or ambiguous.

### Post-score OpenCode session recovery

OpenCode 1.18.27 emits JSON event timestamps as integer milliseconds, while
Fleet trace ingestion requires ISO-8601 strings. `self_hosted.py` normalizes
that boundary and validates every message before the first ingest mutation.
Fleet HTTP failures retain only their method, route, and status in the safe
receipt so schema failures do not collapse into an unactionable exception type.

The v2 Qwen3.8/GLM5.3 smoke recovery reuses only immutable traces whose agent,
authoritative verifier, cleanup, trace digest, and zero-completed-chunk failure
all reconcile. It does not start an agent, model request, task instance, or
verifier. Before writing, it proves there is no session with the same persisted
model and authoritative verifier-execution ID; it then claims a create-once
intent, ingests once, and verifies the resulting completed session through the
public session inventory. The original failed output root is never modified.

```bash
# Server-side validation only; no session mutation.
evals/fleet/scripts/submit_opencode_session_recovery.sh preview

# One create-once recovery Job; zero model rollouts.
evals/fleet/scripts/submit_opencode_session_recovery.sh submit
```

## Qwen3.8-27B reward calibration (non-test)

`qwen38_calibration.py` measures whether the exact served Qwen3.8-27B revision
can acquire binary verifier reward on twenty frozen `train`/`dev` task versions.
It never selects the sealed test20 rows. The twenty versions are a strict subset
of the independent Qwen3.6 calibration slate, so valid attempt-1 outcomes can be
joined later by `task_version_id`; the runner never reads Qwen3.6 outcomes while
executing Qwen3.8.

The protocol pins Qwen Code 0.22.3, a 600-model-request ceiling, a 262,144-token
context, Qwen Code's built-in compaction, exactly `bash` then `submit_report`,
their exact schema-catalog digest, the version-scoped task runtime, and the
authoritative deterministic verifier.
This is the closest current Qwen Code match to the original Fleet sessions, but
does not claim byte-identical Agent Runtime prompting or compaction.

The live preflight requires the gateway catalog, `/model_info`, `/server_info`,
and a structured-tool probe to agree on served id `qwen3.8-27b`, exact revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16, SGLang, TP=1, and a
262,144-token context. It writes only task hashes and immutable bindings to the
receipt. Raw prompts and traces remain under the ignored private artifact root.

```bash
# Read-only identity/task checks plus one tiny structured-tool probe.
evals/fleet/scripts/run_qwen38_reward_calibration_local.sh preview

# Create the pass@1 campaign once. Task 1 is the low-cost canary; the other
# nineteen launch only after positive authoritative reward and verified cleanup.
evals/fleet/scripts/run_qwen38_reward_calibration_local.sh run
```

This path uses the already queue-managed inference deployment and local
linux/amd64 harness containers; it submits no training workload and never
changes, reprioritizes, or cancels cluster jobs.

The historical v2 plan used an infrastructure-only gate and released attempts
2–4 after one authoritative valid zero. It then terminated as an infrastructure
incident: attempts 2–4 are unresolved and excluded pending authorized
read-only verifier-store evidence, and sixteen attempts never launched. V2 is
not scientifically complete and does not satisfy the positive-reward gate. Its
minimized, resource-free incident binding is under `docs/evidence/qwen38-study/`.
V3 fails closed on zero, missing score, invalid cleanup, missing verifier UUID,
canary identity drift, or any unrecognized gate criterion and has not been
submitted.
