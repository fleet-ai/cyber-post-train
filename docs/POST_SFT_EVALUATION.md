# Post-training artifact handoff

For current commands, use [TRAINING.md](TRAINING.md). Train, seal the native
checkpoint on CPU, export it without an optimizer step, then qualify reload and
the exact serving route before evaluating. An export receipt is not a capability
result or a substitute for a successful model reload.

## Required boundaries

- Select the checkpoint by the frozen training/development protocol, never by
  external benchmark results. Keep task families disjoint across splits.
- Bind the source run, completed optimizer step, complete checkpoint inventory,
  tokenizer, chat template, runtime image and code.
- Export into a new directory. Check tensor keys, shapes, real dtypes, hashes and
  exact-base runtime files; never relabel FP32 weights as BF16.
- Validate native optimizer/sampler restoration separately from HF inference
  reload. A weights-only export cannot resume training.
- Bind registration completion and live serving identity before scored use.
  A route name, Ready Pod or catalog row does not prove which weights are loaded.
- Match base/post harness, tools, budgets, sampling, task/environment/verifier
  versions and retry policy. Keep shared and dedicated serving blocks explicit.
- Keep failed/ambiguous attempts and partial artifacts; do not silently overwrite
  or repeat them. Preserve evidence and release unused owned GPU allocations.

See [SCIENTIFIC_PROTOCOL.md](SCIENTIFIC_PROTOCOL.md),
[CLUSTER_ALERTS_AND_INFERENCE_SERVING.md](CLUSTER_ALERTS_AND_INFERENCE_SERVING.md)
and the benchmark-specific guides for their retained acceptance boundaries.

## Historical Qwen3.6 study

The former 600-line runbook described only `ft-run-574bd7b3` and its successive
one-off jobs. It is recoverable from commit `bac2a4c` at this same path. Its
`training.post_sft_cli` and shell observation collector are retired; do not copy
their launch commands into a new experiment.

The original exact plan remains in
`configs/evaluation/qwen36-27b-ft-run-574bd7b3-post-sft.json`; dated receipts remain
under `docs/evidence/qwen36-study/`. Read-only validators used by the historical
paired-evaluation adapters remain tested. Those frozen artifacts describe their
original run and code revision, not current readiness or a new launch authority.
