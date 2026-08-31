# cyber-post-train

Reproducible evaluation and post-training for execution-grounded blackbox cyber agents.

The primary experiment targets the dense Apache-2.0
`Qwen/Qwen3.6-27B` checkpoint at exact revision
`6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`. The repository has four
deliberately separate streams:

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

## Current model and execution backend

```text
model: Qwen/Qwen3.6-27B
revision: 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
weights: 27,781,427,952 parameters, 15 verified BF16 safetensor shards
training: Fleet Training API, exact SkyRL trainer version selected in each run config
formal evaluation: self-hosted exact checkpoint through a pinned SGLang serving contract
```

Runnable SFT and RL requests live in `configs/runs/`. They keep the model,
dataset filters, objective, trainer version, compute shape, and evaluation split
explicit and independently replaceable. The server preview is always checked
before submission; the resulting RayJobs enter `training-lq` through Kueue.

The older GLM files are preserved as experiment provenance, not as the current
model choice. Detailed launch commands live beside each evaluation and training
implementation. The evidence-first current state, terminal results, pending
jobs, and remaining causal-comparison gates are indexed in
`docs/QWEN36_STUDY_EVIDENCE.md`. `docs/STATUS.md` remains the chronological
operational narrative and can contain superseded intermediate observations.

The fail-closed checkpoint selection, export, serving-parity, and paired-evaluation
handoff for the active SFT run is documented in `docs/POST_SFT_EVALUATION.md`.
