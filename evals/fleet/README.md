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

### Interrupted-attempt reconciliation

The self-hosted runner writes `resource-plan.json` before starting local
containers and writes an exclusive, durable `scoring-intent.json` immediately
before its one non-retried authoritative scoring request. If the controller is
terminated outside Python, these intent receipts distinguish an unfinished
local attempt from permission to repeat it.

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
