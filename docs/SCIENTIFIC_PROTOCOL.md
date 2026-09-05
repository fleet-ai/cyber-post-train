# Scientific protocol

## Causal question

For each exact target checkpoint, does post-training on authorized Fleet
blackbox-exploit tasks improve performance on frozen external blackbox cyber
benchmarks? The active targets are Qwen3.8-27B and GLM-5.3. Analyze them as two
separate model studies; do not treat their rollouts or training seeds as one
pooled intervention.

## Experimental arms

For each target model, use at least three arms: unchanged base, SFT-only, and
SFT→online-RL. Add RL-from-base when compute permits. Every arm for that model
starts from the same immutable base checkpoint manifest. Run multiple training
seeds; do not treat evaluation rollouts from one trained checkpoint as
independent training replicates or pool seeds across model families.

The planned sequence is SFT followed by online verifier-reward RL, not a choice
between them. SFT is the low-variance interface/domain adaptation stage and may
imitate only score-1 trajectories. RL is the main capability-learning stage:
it obtains fresh isolated rollouts and can learn on the 58 task lineages for
which the historical corpus has no successful demonstration. Base, SFT-only,
RL-from-base and SFT-to-RL arms distinguish those effects.

## Controlled variables

Before any scored baseline, resolve `configs/evaluation_protocol.example.json`.
Its digest binds all variables other than intervention checkpoint bytes:

- every model shard, tokenizer file and rendered chat template;
- inference engine image/commit, numerical precision and quantization recipe;
- agent harness image/commit, tool schema and system prompt;
- benchmark task, environment, prompt and verifier manifests;
- decoding parameters, agent budgets and random seeds.

The base and intervention endpoints must be deployed from the same serving image
and protocol. If evaluation uses FP8, export both arms with the same pinned
quantization recipe and calibration set. If that cannot be proved, evaluate both
in BF16. A hosted base model with undisclosed bytes is useful for plumbing but is
not an admissible scientific control for a self-hosted intervention.

The active base checkpoints are:

- `Qwen/Qwen3.8-27B` revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- `zai-org/GLM-5.3` revision
  `30333038ada1f1dacb294a93270305a890b50c14`.

Qwen3.6 and GLM-5.2 locks, reports, and cluster manifests are retained only as
historical provenance from superseded feasibility work. They are not active
bases and are not interchangeable controls for either current model.

## Data and leakage

WebExploitBench and all other external benchmarks are evaluation-only. Never use
their files, traces, scores or failure analyses for training, prompt tuning,
reward shaping or checkpoint selection. Choose the recipe on lineage-held-out
Fleet data. The final all-Fleet-data fit is evaluated once on sealed external
benchmarks. Report Fleet training-distribution performance separately.

## Statistics

Pre-register primary metric, task set, exclusions and stopping rule. Report
paired per-task deltas, bootstrap confidence intervals over tasks, and exact
task-level outcomes. Preserve training-seed variance separately from rollout
variance. Report infrastructure failures and timeouts rather than silently
rerunning them; reruns follow a written retry policy and retain original records.
