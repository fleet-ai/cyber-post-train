# Qwen3.8 cyber SFT study — live gates (2026-09-25)

The goal is measured lift on new Fleet blackbox tasks. Historical live 96k
run `chris-q38-t3k96-b8-v3-e7d5c0f8` proves hardware mechanics, **not** the
corrected-data study. Source and frozen-role evidence: [HOLDOUT.md](HOLDOUT.md).

## Why the old run cannot answer the question

Old packing raw-token-clipped **11,794/14,693** rows, often losing the
system/task/tool anchor; its tool names differed from OpenCode evaluation.
Of 149 prefix-salvaged sessions, 148 lost the final successful report. The
step-1000 comparison also mixed output-limit failures and leaked task families:
it is **not** defensible lift evidence. Corrected data must retain originals
privately, use exact target anchors/tools, preserve early/middle/late complete
rounds without duplicate targets or final-submission bias, freeze family roles,
and verify native token/mask files and digests.

## What the two held-out measurements mean

- **Teacher-validation loss:** next-token prediction on successful teacher
  sessions from unseen families diagnoses optimization, not exploit ability.
- **Fleet outcomes:** matched baseline/checkpoint pass@4 on exact runnable
  task versions measures ability. Development selects checkpoints; a separate
  final set confirms once. Infrastructure-invalid is not model failure.

The old 13-dev/7-final families need fresh proof and cannot resolve a modest
10-point lift. Discovery candidates are not qualified tasks; see
[HOLDOUT.md](HOLDOUT.md) for frozen order and API defects.

## Source-to-OpenCode compatibility

All **2,886** envelopes passed exact session/version/transcript digests in a
private cache. Strict checks provisionally retain **943 TRAIN sessions/116
families** and **27 teacher-DEV/9 families**; 16 protected sessions are
quarantined. The importer strips hidden reasoning, substitutes probed OpenCode
anchors and proven calls, but local probes cannot prove live served-Qwen wire.
The detailed historical exclusion census remains in Git history.

`use_tool` wrappers remain excluded: individual CLI versions and MCP block
boundaries are unbound. A synthetic Grok/OpenCode two-block result differed by
one newline, and none of 28,312 wrapped-bash strings has a uniquely
recoverable final status block. A **provisional** one-text-block conversion
adds 219 TRAIN/15 DEV: total **1,162/42** across **153/12** families, with
old source-token proxies **22,295,624/639,694**. Neither proxy is a native
Qwen masked-token count; the ≥20M train floor and historical deployed
tool-result parity remain unproved. Use attested original blocks or fresh
target-harness successes for an accepted corpus.

## Parallel execution lanes

- **96k independent fast lane:** reuse proven one-node/eight-GPU full-weight
  mechanics with corrected, family-disjoint data. Prove an update, native
  checkpoint, zero-step reload and tool match. At **every** checkpoint queue
  teacher loss and matched Fleet DEV pass@4 immediately; retain native state
  through export/reload/eval (the old keep-two policy evicted too early).
- **262k capacity lane:** four-node, 112-row mechanics canary
  `chris-q38-t3k262-4n-can-v2-1502ba9f` is **not** a scientific-corpus run.
  Require a real near-262k first-batch update witness, finite gradients, peak
  memory, sealed checkpoint, release, and separate zero-optimizer reload. The
  old eight-node step and zero-GPU preflight do not prove four-node training.
  CPU preflight v2/v3 failed without GPUs; pinned v4 passed.
- **262k corrected scientific lane (not launchable):** freeze an independent
  name, output, train/DEV digests and step count only after source/capacity
  gates. Require transitive family roles, verified successes, exact OpenCode
  1.18.27 anchors/tools, visible-only complete rounds, unique multi-target
  masks, ≥20M **actual** TRAIN masked tokens, and family-disjoint contiguous
  teacher DEV. Protected final families never train. The old 262k runtime
  binds a different capacity-only corpus/no-CE mode: qualify a new binding.

Initial full262 hypothesis: full-weight Qwen3.8-27B, four 8-B300 nodes,
262,144 context, global batch 32, one epoch, LR 3e-6, and proven
group-1/GDN-512/chunk-1024 mechanics **only after** capacity/reload pass.
Freeze `max_steps = ceil(train_rows / 32)`, align CE with roughly 10–20 spaced
checkpoints, and retain **all** native checkpoints through seal/export/reload/
eval; budget SFS first. Validate native recovery without an optimizer step and
under a new output/W&B identity. Each selected checkpoint needs BF16 readback
and matched baseline/checkpoint Fleet DEV and OpenCode WebExploitBench pass@4;
final families remain sealed until selection. Training loss/teacher CE do not
establish lift.

Before full262 POST, server-preview root `fleet.ai/failure-alerts: "off"`,
c1/q1, four-node shape/release; recheck duplicates, empty output and capacity.
This draft authorizes no full POST. Apply [AGENTS.md](../AGENTS.md). Until
Chris clarifies the wording, use stricter **>10 percentage-point absolute**
final-task lift with statistical evidence; never iterate on the final panel.
