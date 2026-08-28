# cyber-post-train

Reproducible evaluation and post-training for execution-grounded blackbox cyber agents.

The initial experiment targets GLM-5.2. Fleet currently exposes it to new
tool-use jobs as `z-ai/glm-5.2` through Agent Runtime v1; the historical
`fleet-glm/glm-5.2-fp8` runtime alias and direct gateway alias are retained only
as provenance until routing is restored. The repository has four deliberately
separate streams:

1. `evals/webexploitbench/` — evaluation-only WebExploitBench Level 0.
2. `evals/fleet/` — held-out and full-corpus Fleet blackbox task evaluation.
3. `training/` — Fleet-data export, normalization, SFT and verifiable-reward RL.
4. `evals/secondary/` — public XBEN and CVE-Bench fallback evaluations.

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

## Current model

```text
gateway: https://inference.flt.build
gateway model: glm-5.2-fp8 (not routed as of 2026-08-28)
Agent Runtime model: z-ai/glm-5.2
```

Detailed launch commands live beside each evaluation and training implementation.
Live launch state and external gates are recorded in `docs/STATUS.md`.
