# Research discovery records

These records separate recoverable historical facts from plans and guesses.
They are written so a new operator can answer four questions before spending
GPU or evaluation capacity:

1. Which model artifacts actually exist and how far were they validated?
2. Which evaluations are complete, partial, invalid, or merely planned?
3. Which internal recipes have real execution evidence for the exact model?
4. Which qualification gate is still missing for the next experiment?

Current Qwen3.8-27B LoRA program:

- [Discovery and experiment handoff](QWEN38_LORA_DISCOVERY_2026-09-19.md)
- [WebExploitBench and TensorLake failure census](../WEBEXPLOITBENCH_TENSORLAKE_FAILURE_CENSUS.md)
- Machine-readable evidence index:
  `../../configs/discovery/qwen38-lora-evidence-index-v1.json`

The machine index is intentionally conservative. A controller reporting
success is not enough to call a checkpoint or evaluation accepted. Each entry
states the strongest proved artifact, the evidence locator, and whether it can
participate in a scientifically matched comparison.
