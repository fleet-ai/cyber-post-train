# Qwen3.8 Teacher3K step-1000 causal synthesis

Date: 2026-09-24

Status: offline falsifiable hypothesis matrix; regression and causation not yet
identified; no launch

The machine-readable aggregate receipt is
[`2026-09-24-q38-step1000-causal-synthesis-v1.json`](evidence/qwen38-study/2026-09-24-q38-step1000-causal-synthesis-v1.json).
It is bound to current merged main and contains no task identity, prompt,
response, trace, answer, flag, private row, or external benchmark content.

## Observation and boundary

The historical dev17 aggregate reported base at 7/17 families at pass@4
(15/68 successful attempts) and Teacher3K32 step-1000 at 0/17 (0/68), with a
-41.18 percentage-point difference. Its immutable receipts call that aggregate
descriptively valid. A later merged plan says output-limit and process-error
attempts were zero-imputed, PR #580 supersedes the aggregate, and score-blind
whole-pair replacements are incomplete—but the cited correction code and
supersession receipt are not themselves on current main. Preserve this
disagreement and hold the historical numbers out of capability or causal
interpretation until the corrected classifier and replacements resolve it.

The old roster also contained four Teacher3K-exposed development families. The
corrected primary roster is clean dev13, with final7 kept sealed and separate.
Before any cause attribution, a fresh or completed exact-harness comparison must
have all 13 valid paired cells and meet the preregistered harm rule:
`delta(step1000-base) <= -3/13` with reverse one-sided exact McNemar `p <= 0.05`.
A nonnegative delta refutes the historical harm claim; every intermediate result
is “harm not confirmed,” not equivalence.

## Falsifiable cause matrix

| Suspected cause | Evidence and mechanism | Confidence | Exact matched-heldout discriminator |
|---|---|---|---|
| Model-facing tool contract | Training used bare `bash`/`submit_report` without the evaluation schema block; evaluation exposed `fleet_bash`/`fleet_submit_report`. A benign synthetic render diverged before the user message by 874 initial-context tokens. Step-1000 emitted 32 rejected bare calls in 24/68 attempts and completed no valid report, although it also produced 5,023 valid prefixed shell events (4,929 completed, 94 errors). The probe changes schema presence and names together. | Defect: high. Material contributor: medium. Sole explanation: low. | On clean dev13, cross base/step-1000 with current and training-compatible catalogs while underlying operations stay identical. Strong support requires at least five step-1000 0→1 family flips, zero reverse flips, a base gain at least four families smaller, zero rejected legacy-name calls, and at least one authoritative report. Zero positive flips after legacy names are accepted refutes a material effect at this resolution. |
| Raw clipping / anchor loss | 11,794/14,693 rows began their first target at the clip offset; step-1000 consumed 6,451/8,000 such rows. In the fixed sample, 254/256 clipped rows lacked the expected chat delimiter, versus 256/256 unclipped controls retaining it. The largest target was only 4,700 tokens. | Defect: high. Material contributor: medium. Sole explanation: low. | Train matched raw-clipped and message-aligned arms with the same source sessions, target-occurrence multiset, tool rendering, recipe, seed, token budget, and step count. At least five aligned 0→1 family flips with zero reverse flips supports a material cause; zero positive flips after proving zero anchorless rows refutes it at dev13 resolution. |
| Source concentration | One teacher supplied 49.14% of full-corpus tokens, the top five supplied 90.10%, one transitive lineage component supplied 13.27%, and one fractional application supplied 44.21%. The exact step-1000 prefix mix is unavailable. Token-mean loss can therefore overweight narrow modes, but the consumed prefix is not proven to match the full corpus. | Property: high. Step-1000 exposure and causal contribution: low-medium/unknown. | After fixing tools and anchors, compare frozen concentrated versus capped sampling with equal unique-token budget and treatment. At least five balanced-arm 0→1 flips with zero reverse flips supports a material cause; zero positive flips after verifying the caps refutes it at this resolution. |
| Family alias duplication / leakage | Exact key/version overlap was zero, but reviewed atom lineage exposed five of the old 25 families through six aliases. The corpus has 370 transitive components and 59 join multiple task keys. This invalidates the old heldout label and may bias SFT upward. The superseded zero-imputed aggregate cannot establish its effect on a valid comparison. | Contamination: high. Effect on an outcome-valid regression: unknown. | After corrected classification and any predeclared whole-pair replacements, compare clean13 with the separately reported exposed4 stratum. A changed paired effect confirms metric distortion; an identical effect refutes distortion of the number, but never repairs the original roster label. |
| Export, serving, or checkpoint stage | The native checkpoint/seal, zero-update BF16 export, reopened tensor equality, finite GPU reload, immutable stage, and fresh base/candidate live parity were accepted. This strongly disfavors obvious corruption or non-weight drift. The native FP32 step-1000 bytes were later retention-evicted, so a decisive replay is impossible. Correctly exported step-1000 weights may still encode destructive optimization. | Export/transport cause: low. Checkpoint-dose or optimizer cause: unresolved. | For a future preserved checkpoint, require two independent byte-identical exports and native/export finite-logit agreement under a predeclared BF16 tolerance. Identity refutes an export-bit cause. A different native-derived route that gains at least five families with zero reverse flips would support it. Separately, at least five families solved by a prospectively frozen earlier checkpoint and lost by step-1000, with no reverse flips and identical export/runtime gates, supports a checkpoint-dose effect. |

Only after historical harm is outcome-valid should the cause-specific ablations
run. For the three training-ablation rows, five wins and zero losses gives a
one-sided exact McNemar conditional p-value of 0.03125. One to four wins, any
material reverse movement, invalid pairs, or an unfrozen second difference is
inconclusive—not evidence of equivalence.

## Remaining offline work

The highest-value next action needs no job or model call: apply the corrected
terminal-event classifier to preserved records and publish aggregate counts of
valid outcomes, output-limit holds, process errors, and exact paired cells still
requiring replacement. Bind clean13 versus exposed4 membership to that validity
ledger, but do not calculate a capability effect from incomplete pairs. Then
aggregate invalid tool-name events and report completions by validity class and
stratum.

Other offline gaps are: reconstruct the deterministic step-1000 sampler prefix
into identity-free teacher/application/component shares; add derived-row digest
and message-anchor invariants; split the benign tool-render probe into schema
presence versus function-name effects; and inventory whether any accepted
intermediate-checkpoint results already satisfy the clean protocol before
proposing new execution.

The strongest current structural explanation is compound rather than singular:
the tool contract is a direct action-level mismatch, anchor loss affects most
optimizer rows, and concentration may amplify both. None has yet been linked to
an outcome-valid clean-dev13 regression or isolated by an ablation. Family
leakage is a separate validity defect, and the accepted artifact chain makes
export corruption a low-priority explanation.
