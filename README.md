# cyber-post-train

Reproducible evaluation and post-training for execution-grounded blackbox cyber
agents.

## Current study

The active Fleet evaluation measures two exact open-weight checkpoints on one
frozen set of the 100 historically easiest eligible Fleet blackbox exploit task
versions:

| Model | Exact revision | Target |
|---|---|---:|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 100 tasks × 4 attempts |
| `zai-org/GLM-5.3` | `30333038ada1f1dacb294a93270305a890b50c14` | 100 tasks × 4 attempts |

This is an 800-cell evaluation universe. It is not complete merely because a
controller ran or a session exists: each cell needs a digest-valid acceptance
receipt under the frozen treatment. Read
[`docs/CURRENT_FLEET_PASS4_EVAL.md`](docs/CURRENT_FLEET_PASS4_EVAL.md) for the
current protocol, safe operating sequence, and score-blind progress command.

The current agent treatment is OpenCode `1.18.27`, a 262,144-token context,
32,768 maximum output tokens, native compaction with 20,000 tokens reserved for
compaction/autocontinue, and only the ordered `bash` and `submit_report` tools.
Model, task-version, route, harness, tool, verifier, and retry identities are
immutable experimental inputs.

## Repository map

- `evals/fleet/` — exact-version Fleet blackbox evaluation and its acceptance
  ledger.
- `evals/webexploitbench/` — evaluation-only WebExploitBench Level 0.
- `evals/exploitgym/` — evaluation-only ExploitGym comparisons.
- `training/` — Fleet-data export, normalization, SFT, and verifiable-reward RL.
- `configs/runs/` — versioned, preview-first training requests.
- `docs/SCIENTIFIC_PROTOCOL.md` — controls shared across studies.

External benchmarks are evaluation-only. Their prompts, applications, traces,
outputs, metadata, solutions, and derived exploit hints must never enter
training, retrieval, prompt development, reward development, or skills.

## Local setup

```bash
cd /Users/christan/Desktop/cyber-post-train
uv sync --extra dev
```

Provide credentials through environment variables or the cluster secret
manager. Never put a credential in a command argument, source file, committed
configuration, receipt, or log.

Run the read-only checks before choosing an execution path:

```bash
# Validate the frozen 800-cell universe.
uv run python -m evals.fleet.exact_pass4_universe \
  evals/fleet/configs/q38-glm53-exact-easiest100-pass4-campaign-v1.json \
  --repo-root "$PWD" --summary

# Print a score-blind baseline ledger (400 cells per model).
uv run python -m evals.fleet.exact_pass4_ledger --repo-root "$PWD"

# Verify the deployed general cluster Jobs API contract without creating work.
uv run python skills/cyber-cluster-jobs-operator/scripts/check_contract.py
```

These commands do not authorize a scored launch. Use create-once plans and the
release/acceptance gates documented in the current study guide.

## Execution services

Do not confuse the two Jobs APIs:

- Fleet's managed evaluation API at `https://orchestrator.fleetai.com/v1/jobs`
  creates supported managed agent evaluations.
- The Nebius cluster API at `https://api.ft.flt.build/v1/runs` runs a general
  immutable container command through the cluster queue. It is used for work
  such as exact model serving, staging, training, and GPU diagnostics.

Hosted inference and dedicated self-hosted inference are separate serving
treatments. Keep their results in separate blocks unless immutable evidence
proves complete treatment parity. Assign whole tasks to one serving block; do
not split a task's four attempts across routes merely to improve throughput.

## Historical studies

Qwen3.6 material remains valuable provenance but is not the current target.
Start at [`docs/QWEN36_STUDY_EVIDENCE.md`](docs/QWEN36_STUDY_EVIDENCE.md) for
that frozen study. `docs/STATUS.md` is chronological history and can contain
superseded intermediate observations. Historical documents and receipts must
not be rewritten to resemble current state.
