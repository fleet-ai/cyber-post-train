# Scientific protocol

## Causal question

Does post-training the exact GLM-5.2 base checkpoint on authorized Fleet
blackbox-exploit tasks improve performance on a frozen, external blackbox web
exploitation benchmark?

## Experimental arms

Use at least three arms: unchanged base, SFT-only, and SFT→online-RL. Add
RL-from-base when compute permits. Every arm starts from the same immutable base
checkpoint manifest. Run multiple training seeds; do not treat evaluation
rollouts from one trained checkpoint as independent training replicates.

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
