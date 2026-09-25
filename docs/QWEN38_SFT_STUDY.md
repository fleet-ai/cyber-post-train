# Qwen3.8 cyber SFT study — live gates (2026-09-25)

Goal: measured lift on new Fleet blackbox tasks. Historical 96k run
`chris-q38-t3k96-b8-v3-e7d5c0f8` proves mechanics, **not** this independent
corrected-data study. See [HOLDOUT.md](HOLDOUT.md) for source and split evidence.

## Why the old run cannot answer the question

Old packing clipped **11,794/14,693** rows, often losing the system/task/tool
anchor; its tool names differed from OpenCode evaluation. Of 149 prefix-salvaged
sessions, 148 lost the final report. Step-1000 also mixed output-limit failures
and leaked families: **not defensible lift evidence**. New data must retain
private originals, match target anchors/tools, preserve complete early/middle/
late rounds without duplicate targets or submit bias, freeze roles, and verify
native token/mask bytes.

## Two held-out measurements

Teacher cross-entropy on unseen successes diagnoses optimization, **not** exploit
ability. Matched base/checkpoint Fleet pass@4 on runnable versions measures
ability: select on DEV, confirm once on final, exclude infrastructure failures.
Old 13-DEV/7-final families need fresh proof and cannot resolve modest lift.

## Source-to-OpenCode compatibility

All **2,886** envelopes passed session/version/transcript digests. Strict
checks provisionally retain **943 TRAIN/116 families**, **27 teacher DEV/9**;
16 protected sessions are quarantined. Hidden reasoning is excluded and
OpenCode anchors/calls substituted, but local probes cannot prove live Qwen
wire. `use_tool` wrappers stay out: CLI/MCP blocks are unbound and none of
28,312 wrapped bash strings recovers a unique final status. Provisional
one-text-block conversion adds 219 TRAIN/15 DEV: **1,162/42** sessions,
**153/12** families, old token proxies **22,295,624/639,694**. Native masked
tokens, live wire and historical tool-result parity remain unproved; acceptance
needs attested blocks or fresh target-harness successes. OpenCode 1.18.27
truncates tool output above 50 KiB or 2,000 lines; only 362 TRAIN/7 DEV
sessions have size-safe results throughout, too little for accepted full SFT.

## Parallel execution lanes

- **96k independent fast lane:** one node/eight GPUs; corrected family-disjoint
  data, update, native checkpoint, zero-step reload and tool match. At **every**
  checkpoint immediately queue teacher loss and matched Fleet DEV pass@4;
  retain state through eval (old keep-two evicted too early).
- **262k capacity lane:** four-node 112-row canary
  `chris-q38-t3k262-4n-can-v2-1502ba9f` is **not scientific**. Require real
  near-262k first-batch update, finite gradients, peak memory, checkpoint,
  release and separate zero-step reload. Old eight-node step/zero-GPU preflight
  prove neither; CPU preflight v2/v3 failed, pinned v4 passed.
- **262k corrected scientific lane (not launchable):** freeze independent
  name/output/data/steps after source/capacity gates. Require transitive roles,
  verified successes, exact OpenCode 1.18.27 anchors/tools, visible complete
  rounds, unique targets, ≥20M **actual** TRAIN masked tokens and disjoint
  contiguous teacher DEV. Final families never train. Old runtime binds a
  capacity-only corpus/no-CE mode; qualify a new binding.

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
