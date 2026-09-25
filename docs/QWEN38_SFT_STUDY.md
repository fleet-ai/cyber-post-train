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
and build whole-message windows with an explicitly labeled OpenCode target
anchor while retaining the original anchor privately for audit. Cover
early, middle, and late useful actions, but never duplicate target tokens or
train mostly on final submission. Keep all accepted sources in the provenance
ledger and temper sampling from dominant task components or teachers. Freeze
family roles before windowing, and seal separate train and teacher-validation
files. The **actual** trainer-ready token/mask files and exact digests must be
checked; a structured JSONL preview is not a completed corpus.

The reviewed source-family roles are now frozen in
`configs/data/qwen38-teacher3k-family-roles-20260925-v1.json`. Its canonical
payload SHA-256 is
`6c56b9b1ae9c0e5b21a36e48f0d5b6a451bda9da24ecab8262cb737acb1e2b68`.
Of the 1,176 selected source versions, 1,057 are training, 113 are separate
teacher-loss validation, and 6 are quarantined because their atom lineage
touches a protected Fleet task family. Their source sessions are respectively
2,671, 199, and 16; these counts are **not** packed training windows. This
split prevents known family leakage in teacher loss but does not qualify any
current Fleet task as runnable or prove that the new model-facing corpus is
ready to train.

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
7 final) are provisionally receipt-backed and family-disjoint from Teacher3K.
Those 20 still need fresh runtime/grade checks before they can support the
current lift claim.
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

### Full source compatibility census

All 2,886 selected transcript envelopes were fetched into a private 0700
cache and checked against their exact historical session/version and transcript
digests. The source importer keeps originals separate, strips hidden assistant
reasoning, substitutes the two-version-probed OpenCode instruction anchor,
and rewrites only proven tool calls. This is a **new target-anchor method**, not
the old original-anchor packer. The local probe does not yet attest the live
Fleet tool server or the final request received by the served Qwen model.

After the reviewed family split, 2,671 source sessions are assigned to train,
199 to teacher-validation, and 16 to protected test (excluded). The current
strict compatibility check provisionally retains **943 train sessions across
116 families** and **27 teacher-validation sessions across 9 families**. Its
train exclusions are 170 without one bound successful report, 867 with
unproven `use_tool` result wrappers, 652 with non-text tool results, and 39
with other tool/argument mismatches. Teacher-validation exclusions are
respectively 10, 65, 96, and 1. The narrow exact-discovery elision adds no
sessions once wrapper-dependent examples are correctly quarantined.

The 943 retained train sessions account for **20,675,618 historical source
supervised-token estimates**; the 27 validation sessions account for 480,294.
These are **not** Qwen's actual visible, masked target-token counts and may
overstate usable learning signal after the visible-only transform. The
required 20-million-token train threshold is therefore **not established**.
No corrected full training run should be launched until the native tokenizer
counts the sealed train/validation files and live target-tool/result equivalence
is proved. Fresh successful teacher traces using the exact target OpenCode
tools are the cleanest expansion path if this strict subset falls short.

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
