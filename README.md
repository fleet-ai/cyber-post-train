# cyber-post-train

Reproducible evaluation and post-training for execution-grounded blackbox cyber agents.

The repository supports exact, versioned open-weight model experiments. Model choice
belongs in an immutable experiment config; it is not a repository-wide default. The
current code has four deliberately separate implementation streams:

1. `evals/webexploitbench/` — evaluation-only WebExploitBench Level 0.
2. `evals/fleet/` — held-out and full-corpus Fleet blackbox task evaluation.
3. `training/` — Fleet-data export, normalization, SFT and verifiable-reward RL.
4. `evals/secondary/` — public XBEN and CVE-Bench fallback evaluations.

The target unified interface and incremental migration plan are documented in
[`docs/REPOSITORY_DESIGN.md`](docs/REPOSITORY_DESIGN.md). Until that facade is complete,
the evaluator and trainer directories remain the authoritative command surfaces.

## Safety and experimental integrity

- Secrets are read from environment variables and are never written to artifacts.
- WebExploitBench is evaluation-only. Its inputs, outputs, traces and derived artifacts
  must never enter training or agent optimization.
- Training data is limited to authorized Fleet challenge environments.
- External benchmark results remain sealed until the training checkpoint is frozen.
- Dataset splits are by task lineage, vulnerability family and application, not by session.
- Every run records immutable model, data, prompt, harness and verifier identifiers.

## Local setup

```bash
cd /Users/christan/Desktop/cyber-post-train
uv sync --extra dev
cp .env.example .env
```

Export credentials in the shell or use a local untracked `.env`; do not put them in
commands, source files, logs or committed configuration.

Check the local installation and inspect the supported adapter lifecycle without any
credentials or network access:

```bash
uv run cyber-post-train doctor
uv run cyber-post-train catalog
```

The facade currently provides inventory, local diagnostics, create-once experiment
scaffolding, component locking, validation, and deterministic plan compilation:

```bash
uv run cyber-post-train experiment init --help
uv run cyber-post-train experiment lock --help
uv run cyber-post-train experiment validate --help
uv run cyber-post-train experiment compile --help
```

Existing evaluator and trainer commands remain authoritative for live
preview/launch/status/accept operations while those paths are migrated behind the facade.
The facade never claims to launch an adapter that has not completed that migration.

## Experiment identity and execution backends

```text
model: selected by an exact lock under configs/models/
revision: immutable commit and weight manifest required
training: Fleet Training Jobs API with the exact trainer selected in each run config
formal evaluation: pinned harness, tasks, verifier and serving contract per protocol
```

Runnable SFT and RL requests live in `configs/runs/`. They keep the model,
dataset filters, objective, trainer version, compute shape, and evaluation split
explicit and independently replaceable. The server preview is always checked
before submission; the resulting RayJobs enter `training-lq` through Kueue.

Model-specific files are preserved as experiment provenance, not as global defaults.
Detailed launch commands currently live beside each evaluation and training
implementation. Evidence-first state and terminal results remain in model-specific
study reports; `docs/STATUS.md` is chronological narrative and can contain superseded
intermediate observations.

The fail-closed checkpoint selection, export, serving-parity, and paired-evaluation
handoff for the active SFT run is documented in `docs/POST_SFT_EVALUATION.md`.
