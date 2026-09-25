# Qwen3.8 cyber SFT study — live gates (2026-09-25)

The goal is measured lift on new Fleet blackbox tasks. Historical live 96k
run `chris-q38-t3k96-b8-v3-e7d5c0f8` proves hardware mechanics, **not** the
corrected-data study. Source and frozen-role evidence: [HOLDOUT.md](HOLDOUT.md).

## Why the old run cannot answer the question

Old packing raw-token-clipped **11,794 of 14,693** rows, often losing the
system/task/tool anchor; its tool names differed from OpenCode evaluation.
Of 149 prefix-salvaged sessions, 148 lost the final successful report. The
step-1000 comparison mixed output-limit failures and leaked task families:
its reported difference is **not** defensible evidence of lift.

Verify each success/transcript, retain originals privately, and bind an
explicit OpenCode target anchor and tool contract. Whole-message windows
should cover early/middle/late actions without duplicate target tokens or a
final-submission bias. Freeze family roles, cap dominant components/teachers,
and seal separate train/teacher-validation files. A JSONL preview is **not**
trainer-ready: verify actual token/mask files and digests.

## What the two held-out measurements mean

- **Teacher-validation loss:** next-token prediction on successful teacher
  sessions from unseen families diagnoses optimization, not exploit ability.
- **Fleet outcomes:** matched baseline/checkpoint pass@4 on exact runnable
  task versions measures ability. Development selects checkpoints; a separate
  final set confirms once. Infrastructure-invalid is not model failure.

The 20 provisional families (13 dev, 7 final) need fresh proof and cannot
resolve a modest 10-point lift. Discovery candidates are not qualified tasks;
see [HOLDOUT.md](HOLDOUT.md) for frozen order and API defects.

## Source-to-OpenCode compatibility

In a stratified sample of 198 sessions, all transcript digests matched;
193 task/user messages mentioned old bare tool names, no system message did.
Counting `use_tool` wrappers, 184 had one complete report call, 10 had two,
and 4 had none. Develop the one-report subset first. A same-task/version
OpenCode 1.18.27 probe found different target user/system anchors, although
the Fleet task prompt matched source user text by hash. Target tool definitions
matched; the anchor transform still needs live model-request proof.

All **2,886** envelopes passed exact session/version/transcript-digest checks
in a private 0700 cache. The importer keeps originals, strips hidden
reasoning, uses a two-version-probed target anchor, and rewrites proven calls.
Unlike the old packer, this method still needs live tool-server/model-request
attestation; a local probe cannot supply it.

Of 2,671 train-source and 199 teacher-validation sessions, strict checks
provisionally keep **943 / 116 families** and **27 / 9 families**; 16 sessions
in six versions are quarantined. Train exclusions: 170 unbound reports, 867
unproven `use_tool` wrappers, 652 non-text results, 39 other tool/argument
mismatches. Validation exclusions: 10, 65, 96, 1 respectively. Exact-
discovery elision adds none after wrapper quarantine.

The 867 wrapper exclusions are intentional. Fleet pins Grok CLI 1.0.13, but
individual sessions do not prove their CLI version or MCP block boundaries.
A local synthetic two-block probe produced Grok `OkayOutput` text of 48 bytes
(SHA prefix `5c56e82e`) versus OpenCode 1.18.27's 49 (`c25c3287`): one
versus two line breaks. None of **28,312** historical wrapped-bash strings
had a uniquely parseable final status block to recover those boundaries.
Blind unwrapping is not lossless proof; use attested original blocks or new
target-harness successes. No paid task was used for this probe.

Retained train/validation sessions estimate **20,675,618 / 480,294** source
supervised tokens, potentially more than Qwen's visible masked targets. The
20-million-token train floor is **unproved**. Full SFT requires native-tokenizer
counts on sealed files and live target-tool/result equivalence; if short,
collect fresh successes with exact target tools.

## Parallel execution lanes

- **96k fast lane:** start from proven one-node/eight-GPU full-weight mechanics
  with corrected corpus/teacher validation. Prove one update, native checkpoint,
  zero-step reload and tool match. At **each** checkpoint, queue teacher loss
  and matched Fleet dev pass@4; retain native state until export/reload/eval
  acceptance (`keep_checkpoints: 2` previously evicted some too early).
- **262k lane:** an eight-node step proved mechanics, not four-node feasibility.
  Qualify four nodes with a real near-262k sample, finite update, peak memory,
  checkpoint and zero-step reload; then use the same corrected corpus/eval rule.

Both need new run/output identities. The September 25 CPU-only 262k preflight
passed with zero GPUs and released its Job/Pod; its capacity-only example is
not the final corpus or a substitute for the four-node GPU gate. Apply
[AGENTS.md](../AGENTS.md) before paid work. The completion threshold (10
percentage points absolute versus 10% relative) still needs confirmation;
training loss and a repeatedly reused small final panel cannot satisfy it.
