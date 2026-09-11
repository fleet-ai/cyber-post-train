# cyber-post-train

Train open-weight models on authorized Fleet cyber tasks, then evaluate exact
checkpoints with a pinned harness and verifier. Keep training code thin: SkyRL
and Miles own the optimizers; Fleet owns the challenge environments and grading.

## Start here

```sh
uv sync --locked --extra dev --extra train
uv run --locked cyber-post-train doctor
uv run --locked cyber-post-train --help
```

No credentials or model downloads are needed for local tests. `doctor` checks
installed modules only, not cluster access or model readiness.

## Training

Use one editable YAML file for model/data manifests, hyperparameters, resources
and W&B. The [training guide](docs/TRAINING.md) explains the fields and gates.

```sh
uv run cyber-post-train data my-data.yaml
uv run cyber-post-train train my-sft.yaml --output output/my-sft
# On a CPU worker with the pinned image and staged inputs mounted:
uv run cyber-post-train preflight output/my-sft
# Back on the submitting host, using the same prepared directory:
uv run cyber-post-train preview output/my-sft
uv run cyber-post-train submit output/my-sft
uv run cyber-post-train status <returned-run-name>
```

Preparation and preflight do not request GPUs. Submission is explicit and
create-once; never erase a submission journal to retry a timeout. Check current
authorization and the total experiment-owned resource budget before submitting.

Current consolidation status: the Qwen SFT runtime comes from a successful
full-model run; its new wrapper has completed a real one-step run with held-out
loss, W&B and a native checkpoint. CPU export and native GPU checkpoint reload
are independently verified; the separate BF16 inference-export GPU reload remains
pending. Full GLM5.3,
Miles RL and SkyRL RL must pass their exact-model training/reward gates before
being described as production-ready. GLM Flash is not full GLM5.3.

The Miles preparation commands (`rl-data`, `miles-convert`, `miles-seal`) bind
exact Fleet prompts and a native checkpoint. They are not an RL training run;
the [qualification status](docs/CONSOLIDATION.md) distinguishes tested plumbing
from real reward/optimizer evidence.

## Evaluation

```sh
uv run cyber-post-train eval prepare my-eval.yaml --output /shared/my-eval
# On the authorized Docker worker with the exact images staged:
uv run cyber-post-train eval preflight /shared/my-eval
uv run cyber-post-train eval init /shared/my-eval
uv run cyber-post-train eval run /shared/my-eval shared worker-001 --limit 4
uv run cyber-post-train eval status
```

`ROLLOUT_DATABASE_URL` and `FLEET_API_KEY` come from your secret manager. Init
requires a fresh dedicated PostgreSQL database; run only claims its pending rows.
See the [Fleet guide](evals/fleet/README.md) for configuration and execution scope.

- [Fleet](evals/fleet/README.md): exact task versions, OpenCode, private results,
  and PostgreSQL coordination for distributed workers.
- [WebExploitBench](evals/webexploitbench/README.md) and
  [ExploitGym](evals/exploitgym/README.md): separate, evaluation-only adapters.

The historical 100-task pass@4 campaign remains on its frozen worker revision.
Do not copy its run names, reinitialize its database or replay accepted attempts
to start a new experiment. Shared and dedicated serving remain explicit blocks.

## Rules that matter

- Never train, tune prompts/rewards or select checkpoints on external benchmarks.
- Hold out complete task families across versions, not random windows. State
  explicitly whether applications are shared or held out.
- Bind model, tokenizer, dataset, harness, tools, verifier, runtime and budgets.
- Track held-out loss, supervised tokens and recoverable checkpoints in W&B;
  do not upload traces, task text or credentials.
- Use the Jobs API for GPU batch work and the inference control plane for serving.
  Release broken or idle experiment-owned capacity; never alter peer workloads.
- A running process is not a valid result. Require optimizer/checkpoint evidence
  for training and authoritative grading plus cleanup for evaluation.

See [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md),
[scientific controls](docs/SCIENTIFIC_PROTOCOL.md),
[cluster policy and alerts](docs/CLUSTER_ALERTS_AND_INFERENCE_SERVING.md), and
[consolidation status](docs/CONSOLIDATION.md). Dated evidence is historical, not
live state. Raw outputs, datasets, checkpoints and secrets belong outside Git.
