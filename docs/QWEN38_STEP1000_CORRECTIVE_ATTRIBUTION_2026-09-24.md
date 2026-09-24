# Qwen3.8 step-1000 corrective attribution package

Date: 2026-09-24

Status: blocked preparation only; no corpus, training, serving, or evaluation work launched

The machine-readable plan is
[`qwen38-step1000-corrective-attribution-v1.json`](../configs/qualification/qwen38-step1000-corrective-attribution-v1.json).
It reduces the next causal work to two experiments and does not create another evaluation or training operator.

## A. Test the historical tool names without training

Run the clean 13-family development set at pass@4 in a 2×2 comparison:

| Model | Current names | Historical bare aliases |
|---|---:|---:|
| Exact base | `fleet_bash`, `fleet_submit_report` | `bash`, `submit_report` |
| Exact step-1000 | `fleet_bash`, `fleet_submit_report` | `bash`, `submit_report` |

Only the two names may change between catalogs. The descriptions, JSON argument
schemas, challenge-side operations, tasks, seeds, runtime, context policy, and
budgets stay identical. This makes the result a test of the name mismatch, not
of tool schemas plus names together.

The current catalog first provides a fresh 104-session base-versus-step-1000
check. Stop if it does not reproduce the preregistered harm on all 13 valid
pairs. Only after that gate passes, run the 104 historical-alias sessions. The
historical aliases materially support the name hypothesis only if at least five step-1000
families change from unsolved to solved, none changes in reverse, the base gains
at least four fewer families, no legacy call is rejected, and step-1000
completes at least one authoritative report.

This phase is blocked because OpenCode's current Fleet MCP registration always
prefixes names. A minimal alias adapter must prove it forwards to the same two
operations and changes nothing except names. The merged strict heldout operator
should then render two ordinary two-model campaigns plus one immutable join;
its implementation must not be copied. PR604's current adapter only accepts a
16-family, 160-cell wave, so a reviewed successor must first accept this
13-family, 104-cell-per-catalog plan. Both catalog campaigns and their join are
sealed before any A0 result is read.

## B. Isolate raw clipping in 64 optimizer steps

Select exactly 512 verified-success training sessions and exactly one complete,
non-report assistant target from each. Both arms use the exact current formal
tool schemas and names. Each source target appears once and in the same order.
Before selection, cap each teacher model at 80 sessions and each reviewed task
family at eight sessions so no single source or family can dominate the test.

- **FIXED** renders the complete system, task, formal-tool anchor and whole
  message rounds. It permits no anchorless row.
- **CLIPPED** takes the historical raw slice beginning 8,192 tokens before the
  same target, even when that offset falls inside a message. Every selected row
  must have a nonzero clip offset.

Both arms have an 8,192-token pre-target budget. FIXED spends that budget first
on the full anchor, then keeps the largest recent suffix of complete
assistant-plus-tool-result rounds that fits. CLIPPED uses the raw 8,192 tokens
immediately before the same target. Per-row receipts record both lengths and
digests. Both arms then start from fresh copies of the exact base model and run
full-weight BF16 SFT for 64 steps at global batch eight. Model, target tokens,
optimizer, seed, row order, and exposure are fixed. The one intended difference
is the input context construction before each target.

The fixed arm supports a material clipping explanation only with at least five
clean-development families changing from unsolved under CLIPPED to solved under
FIXED and zero reverse changes. Anything weaker is inconclusive.

Preparation depends on the reviewed message-aligned builder, an exact private
512-session selection manifest, one paired-row builder, and a small versioned
successor of the already-qualified full-weight runtime. That successor changes
the corpus, run, output, and W&B identities, the 64-step/checkpoint horizon, and
the recovery deadline while preserving the qualified FSDP path, seed, and
optimizer checks. No heldout or external benchmark content may enter either
corpus.

## Exact stopping boundary

The plan contains no launch authority. Its blocker list is executable work:
each null identity must be replaced by an accepted immutable receipt, every
future rendered root Job or RayJob must contain
`fleet.ai/failure-alerts: "off"`, and both server previews must agree before any
create. Until then the package is an offline preregistration, not a result.
