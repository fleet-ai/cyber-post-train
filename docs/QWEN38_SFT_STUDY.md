# Qwen3.8 cyber SFT study — live gates (2026-09-25)

Goal: measured lift on new Fleet blackbox tasks. Historical 96k run
`chris-q38-t3k96-b8-v3-e7d5c0f8` proves mechanics, **not** this independent
corrected-data study. See [HOLDOUT.md](HOLDOUT.md) for source and split evidence.
Its running Pod, checkpoints, and evaluations cannot satisfy any new 96k
milestone below: the corrected run needs its own frozen data, name, output,
checkpoint chain, and matched held-out evaluations.

## Why the old run cannot answer the question

Old packing clipped **11,794/14,693** rows, often losing task/tool anchors;
tools differed from OpenCode eval. Of 149 prefix-salvaged sessions, 148 lost
the final report. Step-1000 mixed output-limit failures and leaked families:
**not defensible lift evidence**. New data must retain private originals,
match target anchors/tools, preserve complete early/middle/late rounds without
duplicate targets or submit bias, freeze roles, and verify native masks.

## Two held-out measurements

Teacher cross-entropy on unseen successes diagnoses optimization, **not** exploit
ability. Matched base/checkpoint Fleet pass@4 on runnable versions measures
ability: select on DEV, confirm once on final, exclude infrastructure failures.
The old paired structured-window DEV converter was unused and incompatible with
the corrected dense TRAIN path. `training/teacher_dev.py` now builds a disjoint,
target-compatible DEV panel offline, but deliberately rejects provisional
source evidence; no trusted teacher cross-entropy panel exists yet.

## Source-to-OpenCode compatibility

All **2,886** envelopes passed session/version/transcript digests. Provisional
strict set: **943 TRAIN/116 families**, **27 teacher DEV/9**; 16 protected
sessions quarantined. Hidden reasoning is excluded and OpenCode anchors/calls
substituted. The current first-request Qwen wire is checked below, but
historical tool-result parity is unproved. `use_tool` wrappers stay out:
CLI/MCP blocks are unbound; none of 28,312 wrapped bash strings recovers a
unique final status. One-text-block conversion adds 219 TRAIN/15 DEV:
**1,162/42** sessions, **153/12** families, old token proxies
**22,295,624/639,694**. Native masked tokens/live wire/tool-result parity
remain unproved. OpenCode 1.18.27 truncates results above 50 KiB/2,000 lines;
only 362 TRAIN/7 DEV sessions are size-safe—too little for accepted full SFT.

## Bounded live target-wire probe (2026-09-25)

The Chris-owned exact-base route (model UID `54beeb64-c498-4e7d-a504-05a8d694808b`,
Pod UID `6a7308b4-4e8b-4683-a21f-9c32147dfd0a`) served two frozen TRAIN task
anchors through pinned OpenCode 1.18.27 and the fixed proxy. Both first requests
matched the reviewed system/user and two-tool hashes; both returned complete
HTTP-200 SSE streams with distinct response IDs. The synthetic local MCP did
not execute task tools. The first probe entered a tool loop: 10 bounded live
chat requests total, each capped at 128 output tokens. The route was paused
and independently read back at zero Pods/replicas and routing disabled.
Private sealed diagnostic: `/private/tmp/cpt-q38-live-qwen-wire.YKisC9/LIVE_WIRE_DIAGNOSTIC.json`
(file SHA-256 `c6d475ba5b54bb2a34b1f2f68ee4aa4e10964873de0208d7cc4ae326e81eeeed`).
This proves current target first-request wire only, **not** the historical
teacher tool-result renderer or corpus acceptance; `training_ready` stays false.

An isolated OpenCode 1.18.27 mock confirmed that two one-block MCP text results
reached its model request byte-for-byte. A separate current TRAIN task/version
was then probed with one exact historical bash call: the stored teacher result
was 2,451 characters, while the fresh MCP text was 2,444 characters and marked
as a tool error. The fresh instance was independently confirmed stopped with
its durable claim (sealed terminal
`sha256:53d52f465cc5428da273428d7551d1131b7aa137cd4a4fe25459f75ad18b7294`).
Different runtime output may explain the difference; it does **not** prove a
renderer bug or equivalence. The historical corpus remains unaccepted. An
earlier attempt against an obsolete TRAIN task version was rejected before an
instance was created; its durable claim was independently absent. Fresh teacher
collection through Fleet OpenCode was also rejected before job creation with
`sales_product_required`; the correct cyber-study billing label needs review.
Another current TRAIN version returned one successful MCP text block but differed
from its stored result by one character (2,251 versus 2,252); line-ending
equivalence was **not** established. A bounded follow-up timed out during
creation after the server materialized an instance. Its exact durable claim
enabled cleanup, and independent readback confirmed that instance `stopped`.
No further retry of that probe is planned without a changed diagnostic.

A read-only check of one exact strict TRAIN session found 53 stored tool-result
strings but neither raw MCP response blocks nor a separately captured next
model-facing tool message. Its harness metadata lacks a renderer binary/image
digest, and its instance lacks environment-version/image pins; exact session
and instance detail readbacks returned 404. A result string alone cannot prove
Grok-to-OpenCode rendering equivalence. The 943 strict TRAIN/27 DEV historical
sessions and one-block salvage remain provisional. Admit none as scientific
training data without immutable renderer and same-call wire evidence; otherwise
collect fresh verifier-backed OpenCode teacher successes on TRAIN families.

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
  At 2026-09-25 12:42 UTC it remained unadmitted, holding zero GPUs: the
  training queue had 25 GPUs below its 192-GPU quota, short of this job's 32.
  Kueue displayed a `cpu-head` topology fallback error; that alone does not
  establish a job-config defect while the B300 flavor lacks quota.
  A restore-only adapter and synthetic test exist, but no source step-1 seal,
  exact-image restore preflight, or GPU reload acceptance exists yet.
- **262k corrected scientific lane (not launchable):** freeze independent
  name/output/data/steps after source/capacity gates. Require transitive roles,
  verified successes, exact OpenCode 1.18.27 anchors/tools, visible complete
  rounds, unique targets, ≥20M **actual** TRAIN masked tokens and disjoint
  contiguous teacher DEV. Final families never train. Old runtime binds a
  capacity-only corpus/no-CE mode; qualify a new binding.

Initial full262 hypothesis after capacity/reload pass: full-weight Qwen3.8-27B,
four 8-B300 nodes, 262,144 context, global batch 32, one epoch, LR 3e-6,
group-1/GDN-512/chunk-1024. Freeze `max_steps = ceil(train_rows / 32)`, ~10–20
spaced CE/checkpoints; retain **all** native saves through eval and budget SFS.
Validate zero-step recovery with a new output/W&B identity. Each checkpoint
needs BF16 readback and matched base/candidate Fleet DEV and OpenCode
WebExploitBench pass@4. Keep final families sealed; train/teacher loss is not lift.

Before full262 POST, server-preview root `fleet.ai/failure-alerts: "off"`,
c1/q1, four-node shape/release; recheck duplicates, empty output and capacity.
This draft authorizes no full POST. Apply [AGENTS.md](../AGENTS.md). Until
Chris clarifies the wording, use stricter **>10 percentage-point absolute**
final-task lift with statistical evidence; never iterate on the final panel.
