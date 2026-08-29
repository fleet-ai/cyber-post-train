# Qwen3.6-27B Fleet blackbox post-training experiment

## Claim

Post-training Qwen3.6-27B on execution-verified Fleet blackbox exploit trajectories
improves autonomous web exploitation on lineage-held-out Fleet tasks and a
sealed external benchmark.

## Frozen evaluation order

1. Freeze base model revision, harness, prompts, tools and budgets.
2. Run the base checkpoint on Fleet holdout and WebExploitBench Level 0.
3. Seal external benchmark task-level results and traces from trainers.
4. Train only on the immutable Fleet training manifest.
5. Freeze the post-trained checkpoint before inspecting external results.
6. Evaluate the post-trained checkpoint with the byte-identical protocol.
7. Compare paired task outcomes with task-level bootstrap confidence intervals.

## Primary endpoints

- WebExploitBench Level 0 macro pass@1.
- Future-created, lineage-held-out Fleet blackbox macro pass@1.

## Secondary endpoints

- pass@3;
- success-versus-step-budget area under curve;
- steps, tokens and wall time to success;
- application and vulnerability-family stratification;
- refusal rate;
- undesirable-behavior rate.

## Stop conditions

Do not launch if any of the following is unresolved:

- the credential does not resolve to the Fleet team;
- benchmark access or its evaluation-only conditions are not satisfied;
- the target surface is not an isolated authorized challenge environment;
- model or data revisions are mutable/unknown;
- the planned number of paid sessions is not explicitly printed and accepted;
- train/dev/test lineage leakage is detected.
