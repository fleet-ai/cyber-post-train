# Qwen3.8 corrected Teacher3K minimum confirmation

Date: 2026-09-24
Status: proposal only; blocked; no corpus, training, serving, or evaluation work launched

The machine-readable authority for this proposal is
[`qwen38-corrected-teacher-sft-minimum-plan-v1.json`](../configs/qualification/qwen38-corrected-teacher-sft-minimum-plan-v1.json)
at logical SHA-256
`6277e8a3fe4117b80e9b9858cd88db8c90a8928b1eb956893d6d230edfd462f0`.
It was prepared from `cyber-post-train` main commit
`9859a27d09b4fd881dfe04bbf9a1dcf7413083b8` and is intentionally not a job
request.

## Decision

Run one combined correction, not another sweep:

1. admit a fully materialized message-aligned corpus;
2. qualify the proposed optimizer as the first 10% prefix of its full horizon;
3. resume the same run through one epoch; and
4. compare the exact base with the terminal corrected checkpoint on the
   lineage-heldout Fleet set under a successor matched protocol.

This answers whether the combined corrected-corpus and conservative-optimizer
treatment works. It does not attribute an effect among message boundaries,
formal tools, learning rate, or warmup. The already accepted step1000 export is
the negative diagnostic control; rerunning the malformed corpus would add no
decisive evidence.

The formal-tool mismatch is a confirmed structural defect: the retired dense
path rendered no tool schemas and bare `bash`/`submit_report` targets, while the
heldout OpenCode path presents full schemas and `fleet_bash`/
`fleet_submit_report` to the model. Synthetic exact-template comparison confirms
that this materially changes the rendered token stream, but this plan binds no
immutable receipt for that diagnostic and therefore makes no causal-magnitude
claim. A matched-schema-versus-mismatched ablation is a separate attribution
study, not part of the minimum sequence.

The sequence starts only after the base-versus-step1000 Fleet comparison has
valid paired cells. [PR #580](https://github.com/fleet-ai/cyber-post-train/pull/580)
supersedes the earlier aggregate because output-limit and process-error attempts
were incorrectly treated as capability zeros. Its replacement work is still
incomplete, so no old pass@4 value or effect estimate is an input to this recipe.

## Exact starting evidence

| Evidence | Immutable reference | Use |
|---|---|---|
| Root cause | [PR #577](https://github.com/fleet-ai/cyber-post-train/pull/577) at `feb5827cb1421899c210ab498235108f45e7c11e` | Establishes malformed mid-message windows and training/evaluation tool-contract drift. |
| Corpus repair | [PR #584](https://github.com/fleet-ai/cyber-post-train/pull/584) at `34f41b8f172e30e4e12c22b71bafa365bcf9a31f` | Supplies the proposed whole-message builder; remains blocked and non-launchable. |
| Weight audit | [PR #586](https://github.com/fleet-ai/cyber-post-train/pull/586) at `96e33180edc1464a4fc8b026ff19c57e812da674` | Rules out a no-op, wrong export, or wrong served weights; makes no quality claim. |
| Heldout lineage | [PR #578](https://github.com/fleet-ai/cyber-post-train/pull/578) at `942b332115a8bfecc48da76d197786ee86280cae` | Defines 20 lineage-clean families, but admits only the retired corpus manifests. |
| Eval adapter | [PR #576](https://github.com/fleet-ai/cyber-post-train/pull/576) at `4849fe357970247bc62df5f12b7770fb97cfbb2c` plus PR #580 at `4bab2b50ccfe5758327e340800a9fd1619a0265c` | Must be integrated before execution so restart handling and outcome validity cannot diverge. |

The exact base is `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, weight-manifest SHA-256
`06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352`,
and tokenizer-manifest SHA-256
`3938a9a8172f2738fed1be44efc11e2562059269d50d3721213f44802b53b4e1`.

## 1. Corpus canary

PR #584 must first clear all four of its declared blockers:

- a sealed exact identity-to-family-role roster;
- success evidence bound to the retained successful report call;
- an accepted OpenCode provider transform and sanitized request capture proving
  the exact Qwen-facing tools; and
- read-only rehash of the private normalized source and success evidence.

Then perform one zero-GPU, create-once full materialization and readback. A
sample build is insufficient. The accepted manifest must prove complete message
and tool-call/result boundaries, repeated system/task/formal tool schemas,
`fleet_bash` and `fleet_submit_report` as the exact model-facing names,
once-only target coverage, and alias-complete dev/final exclusion.

There is one additional admission gap: repository scientific policy requires
unique supervised target tokens, source and family counts,
distribution/concentration, and explicit per-source and per-family token caps
applied before packing. PR #584 currently records teacher session counts but
does not emit the complete per-task/session/teacher token concentration and cap
evidence. Those fields and all resulting counts/digests must be sealed before
training; retired corpus counts cannot fill them.

## 2. Optimizer canary

Freeze the accepted corpus into a new versioned runtime/config. Keep full-weight
BF16 FSDP, 32K maximum length, 8K copied context, global batch 8, one microbatch
per GPU on one eight-GPU node, seed `20260920`, and the pinned trainer image.
Use AdamW with betas `0.9/0.999`, weight decay `0.01`, maximum gradient norm
`1.0`, peak learning rate `1e-6`, and a constant schedule after 5% warmup.

For accepted corpus row count `R`:

```text
S = ceil(R / 8)                         # full one-epoch optimizer horizon
W = ceil(0.05 * S)                      # warmup updates
C = max(W + 1, ceil(0.10 * S)), C < S  # planned pause/checkpoint update
```

Compile the full `S`-step schedule first and pause at `C`; do not compile a
separate short schedule. Accept the canary only with exact input accounting,
finite loss/gradient/update/LR telemetry, changed full-weight model tensors, a
complete sealed native checkpoint, and an exact zero-step reload. Clipping
frequency is diagnostic, not a capability metric.

The current explicit optimizer wrapper is a useful pattern, not an executable
input: it is hard-coded to historical run/data identities, zero warmup, and
rejects pause/recovery. A new versioned successor and rendered-field test are
required.

## 3. Meaningful run

Resume the exact accepted pause checkpoint through step `S`. No scientific
field or schedule horizon changes. Only create-once recovery/output/W&B
identities, removal of `pause_after_step`, and exact recovery receipts may
differ. One epoch is the minimum complete exposure because every corrected
target appears once.

Select the terminal checkpoint in advance. Do not stop or select from imitation
loss. Before evaluation, require accepted native-checkpoint, BF16 export,
staging, serving-registration, and live-parity receipts.

## 4. Corrected matched heldout comparison

The fail-closed proposal is
[`qwen38-corrected-teacher3k-heldout20-pass4-successor-v1.json`](../configs/evaluation/qwen38-corrected-teacher3k-heldout20-pass4-successor-v1.json)
at logical SHA-256
`f66cbd9c02b3d33e0c8f07153f7af1262fb4971628c35bca673f485f5d8d35d4`.

Re-run lineage reconciliation after corpus materialization and issue a new
immutable protocol that binds its manifest. Preserve the exact 20-family roster
(13 development, 7 final-test) only if that reconciliation remains clean. Run
pass@4 with the four frozen attempt seeds and the exact OpenCode 1.18.27
harness, sampling, tools, images, and budgets in the proposal. The only arm
difference may be the accepted weight manifest and its direct serving identity.

Reuse historical base cells only if every protocol and outcome-validity binding
matches; otherwise run a fresh base arm. Preserve valid zeros. Hold output-limit
and process errors outside the capability denominator, retain originals, and
replace an invalid cell only as a score-blind whole base/candidate pair under a
new immutable successor. Report development and final-test strata separately;
the 20-family union is descriptive.

No WebExploitBench or other external evaluation is part of this plan. Every
future rendered root `Job` or `RayJob` must prove
`fleet.ai/failure-alerts: "off"` before creation. This document authorizes no
materialization, cluster workload, serving change, or scored session.
