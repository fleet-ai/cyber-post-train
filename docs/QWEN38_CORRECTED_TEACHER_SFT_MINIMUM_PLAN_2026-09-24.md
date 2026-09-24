# Qwen3.8 corrected Teacher3K minimum confirmation

Date: 2026-09-24
Status: proposal only; blocked; no corpus, training, serving, or evaluation work launched

The machine-readable authority for this proposal is
[`qwen38-corrected-teacher-sft-minimum-plan-v1.json`](../configs/qualification/qwen38-corrected-teacher-sft-minimum-plan-v1.json)
at logical SHA-256
`d60cbe89fa125cc7c88966e15f0e031682a599844f9869b3837d8679db9d56a8`.
It was prepared from `cyber-post-train` main commit
`0ae3e0923c1e74e65bea3a91f102c1ac7a12917d` and is intentionally not a job
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
formal tools, successful-report provenance, heldout exclusion, learning rate,
or warmup. The already accepted step1000 export is a **historical
flawed-treatment comparator**, not a negative control: its weight audit proves
real training and correct serving identity, but the earlier outcome and lineage
defects mean its capability direction is not yet established. It may be called
a negative treatment comparator only if a fresh outcome-valid, alias-clean,
exact-harness matched base-versus-step1000 comparison meets a preregistered
negative rule.

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
does not define a reviewed numeric cap policy, enforce both caps in versioned
builder code, regression-test that enforcement, or emit complete
per-task/session/teacher token concentration evidence. Training remains blocked
until immutable receipts bind the cap policy, implementation commit and file,
regression test, prepacking enforcement result, and admitted manifest. Every
field must be non-null; retired corpus counts cannot fill them.

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
`756b6cabf665dab351c0e8a661782cf33ee7a4fc785f04aeca32d4e23929e315`.

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
new immutable successor.

### Preregistered primary analysis

The primary stratum is the 13 clean development families. The unit is one
reviewed family and its binary endpoint is pass@4: at least one authoritative
success among four valid attempts. All 13 paired cells must be complete after
any predeclared whole-pair replacement. For family `i`, let `C_i` and `B_i` be
the corrected and base endpoints. The estimand is the absolute paired
task-family risk difference

```text
delta_dev = (1 / 13) * sum_i(C_i - B_i).
```

Let `b` count discordant corrected wins (`B_i=0, C_i=1`), let `c` count
discordant base wins, and let `D=b+c`. Test superiority with the one-sided exact
McNemar conditional-binomial p-value
`Pr[Binomial(D, 0.5) >= b]`, with no mid-p correction and `p=1` when `D=0`.
Report the two-sided 95% Clopper-Pearson exact interval for
`theta=Pr(corrected win | discordant pair)` and its conditional-on-observed-`D`
mapping `(D/13) * (2*theta - 1)` to the paired-risk-difference scale. When
`D=0`, report `delta_dev=0`, theta as not estimable, the theta interval as
`[0,1]`, and the mapped interval as `[0,0]`.

Call the primary result positive only if all 13 pairs are valid,
`delta_dev >= 3/13`, and the exact one-sided p-value is at most `0.05`. Call it
evidence of harm only if `delta_dev <= -3/13` and the reverse exact one-sided
p-value is at most `0.05`. Every other result is **not confirmed**, not evidence
of equivalence. This is the single confirmatory test.

Keep the seven final-test families sealed unless the development positive rule
passes. Then unseal them exactly once, with no tuning or checkpoint change, and
report the same paired delta, discordant counts, exact p-value, and conditional
interval separately. `delta_final >= 2/7` earns only the descriptive label
"directionally consistent"; it is not a second powered superiority claim. Do
not pool the final seven with development to rescue the primary decision. The
20-family union remains descriptive.

### Attribution and retained limits

A positive primary result supports only the combined treatment for this exact
checkpoint and fixed training seed. It cannot attribute an effect among message
alignment, formal tool schemas/names, successful-report provenance,
alias-complete exclusion, learning rate, or warmup. Any component claim requires
a separately preregistered factorial or single-factor ablation that holds source
eligibility, provenance, exclusions, supervised-token exposure, model, and
evaluation fixed.

If the fresh matched base-versus-step1000 comparison first confirms a historical
regression, the smallest bundle-separation follow-up is one separately
preregistered corrected-corpus arm using the historical optimizer and schedule
with matched supervised-target-token exposure. Exact base and step1000 cells may
be reused only if every binding remains compatible. That arm tests the corrected
corpus bundle at the historical optimizer; it still cannot distinguish message
alignment, tool contract, successful-report provenance, and alias exclusion from
one another. This plan does not authorize that arm.

The minimum run intentionally retains these limits:

- source system and task messages remain heterogeneous rather than normalized;
- targets remain action-only demonstrations without student-visible teacher
  reasoning;
- training uses a 32K maximum sequence and 8K copied-context budget while Fleet
  uses 262K native compaction/autocontinue;
- one training seed provides no training-seed variance estimate; and
- exact provider-request capture for the model-facing tool contract remains a
  pre-training blocker, not an assumed wire-level identity.

No WebExploitBench or other external evaluation is part of this plan. Every
future rendered root `Job` or `RayJob` must prove
`fleet.ai/failure-alerts: "off"` before creation. This document authorizes no
materialization, cluster workload, serving change, or scored session.
