# Qwen3.8 cyber SFT study — live gates (2026-09-25)

The objective is a **measured improvement on new Fleet blackbox tasks**, not
completion of a training job. The already-running 96k Teacher3K job
`chris-q38-t3k96-b8-v3-e7d5c0f8` is evidence that the hardware recipe works;
it is **not** the corrected-data experiment below.

## Why a new corpus is required

The recorded Teacher3K source contains 2,886 verifier-backed successful
sessions from 496 task keys and 1,176 task versions, about 57.4 million
supervised tokens. These are not 2,886 independent tasks: the historical
source audit found 370 components when matching exact atom-version locators;
a stricter, leakage-safe union across versions of the same atom has 356.
Current task breakage does not by itself
invalidate an older success at its exact historical version.

The earlier derived windows were not suitable for a clean capability study:
11,794 of 14,693 rows were raw-token clipped; many lost the system/task/tool
anchor, and the training tool names differed from the OpenCode tool names used
at evaluation. Among 149 prefix-salvaged sessions, 148 no longer retained the
successful final report. The old step-1000 baseline/checkpoint comparison also
had output-limit and task-family leakage problems. Its reported difference is
not a defensible estimate of model lift.

The successor must re-read structured source sessions, verify exact-version
success and transcript integrity, bind the model-facing OpenCode tool contract,
and build whole-message windows with the original task/tool anchor. Cover
early, middle, and late useful actions, but never duplicate target tokens or
train mostly on final submission. Keep all accepted sources in the provenance
ledger and temper sampling from dominant task components or teachers. Freeze
family roles before windowing, and seal separate train and teacher-validation
files. The **actual** trainer-ready token/mask files and exact digests must be
checked; a structured JSONL preview is not a completed corpus.

## Two different held-out measurements

1. Teacher-validation loss measures next-token prediction on successful
   teacher sessions from task families absent from training. It diagnoses
   optimization and overfitting, not blackbox exploit success.
2. Fleet task outcomes measure whether baseline and each trained checkpoint
   solve the **same currently runnable tasks** under the same OpenCode tools,
   budgets, four attempts per task, and verifier. Development tasks can guide
   checkpoint selection; a separately frozen final set is used once for the
   confirmatory lift claim. Infrastructure-invalid attempts remain invalid,
   never silent model failures.

The live catalog has 1,217 blackbox task versions, but only 75 have the
previous exact-version/runtime receipt proof, and just 20 (13 development,
7 final) are both currently qualified and family-disjoint from Teacher3K.
Four attempts on one task are **not** four independent tasks. Twenty families
cannot reliably resolve a modest 10-point lift. Expand the runnable,
lineage-clean roster from the unreviewed pool before a final significance
claim; verify startup, tools, verifier, outcome, and cleanup on exact versions.

The 2026-09-25 read-only discovery found 671 current, not-yet-analyzed
versions on task keys absent from Teacher3K. Exact-version metadata checks
passed for all 671; shared-atom lineage removes 238, leaving 433 candidate
versions in 327 independent components across 14 apps. **These are candidates,
not 327 qualified runnable tasks.** Model-free startup, tool, verifier,
negative-control, and cleanup checks must still be completed before assigning
any to the outcome holdout. After also excluding the separately protected
pending task roster, 408 versions in 311 independent components remained;
the first frozen model-free qualification wave contains 16 components across
14 applications (14 medium, 2 hard). Its metadata status is not runtime
acceptance.

The first runtime-qualification canary did start an environment, but its
client expected a deterministic create-request ID while the deployed
version-scoped API minted a different ID. The exact environment was stopped
and deletion verified. This is a qualification-client defect, **not** evidence
that the task is broken or runnable. No later wave cell was launched. The
client must use a supported create-once and recovery route before resuming.

An additional source-compatibility gate is open. In a stratified sample of
198 selected teacher sessions, 193 task/user messages explicitly referenced
older bare tool names, while none of the system messages did. All 198 full
transcript digests matched their selection records. A blanket exclusion would
discard nearly the whole corpus; treating those prompts as identical to the
target OpenCode model request would be equally unjustified. Preserve the
original transcript, capture the actual target model-facing request without
private content, compare exact task/version prompt identities, and admit only
a reviewed lossless transform or a demonstrated equivalent target contract.
With historical `use_tool` wrappers counted correctly, 184/198 sampled
sessions have exactly one complete report call, 10 have two, and 4 have none;
the one-report subset should be developed first. A same-task/version local
OpenCode 1.18.27 probe confirmed the target user and system anchors differ
from the source transcript, even though the Fleet task prompt itself matches
the source user text by hash. The tool definitions match the target names, but
an anchor transform remains unqualified until independently reproduced.

## Parallel execution lanes

- **96k fast lane:** use the proven one-node/eight-GPU full-weight recipe as a
  starting point, but with the new corpus and teacher-validation split. First
  prove an optimizer step, native checkpoint, reload, and matched tool contract.
  Then train with regular checkpoints. For **every** checkpoint, queue a
  teacher-loss measurement and matched Fleet development pass@4 immediately,
  without waiting for the full run. Keep its native state until the exported
  checkpoint has reloaded and the eval has accepted it; the historical
  `keep_checkpoints: 2` setting did not achieve this.
- **262k long-context lane:** the eight-node one-step result proves useful
  mechanics but not a four-node run or near-maximum examples. Independently
  qualify four nodes with a real near-262k sample, one finite optimizer step,
  memory evidence, checkpoint, and zero-step reload. Only then train the same
  corrected corpus and evaluate every checkpoint as above.

These are **new** study runs with new output identities. The historical live
96k job cannot satisfy the corrected 96k data, checkpoint-evaluation, or lift
gates. The 262k CPU-only preflight passed on 2026-09-25 with zero GPUs and
released its Job and Pod; it cannot substitute for a four-node GPU update and
reload proof. The capacity-only 262k sample is not the final SFT corpus.

Before any paid submission, check live project capacity/duplicates, use c1,
and verify the **rendered root Job/RayJob** has
`fleet.ai/failure-alerts: "off"`. A request flag is insufficient. New runs
must use immutable output paths, W&B, exact model/data/runtime bindings, and
prompt release of stopped or failed allocations.

The completion threshold (10 percentage points absolute versus 10% relative)
is being confirmed with Chris. Neither threshold may be claimed from training
loss or a small, repeatedly reused final task set.
