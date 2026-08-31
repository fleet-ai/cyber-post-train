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
