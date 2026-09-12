# Qwen teacher exposure-matched control

## Result

The balanced teacher treatment previously changed two things at once: family
weighting and episode/window/supervised-output dose. This control removes that
specific confound. It keeps
the availability-weighted teacher pool as its candidate universe, keeps every
covered family, and selects only complete native episodes until it exactly
matches the balanced arm's episode, dense-window, and supervised-token totals.

The immutable aggregate lock is
[`configs/studies/qwen-blackbox-teacher-exposure-matched-control-v1.json`](../configs/studies/qwen-blackbox-teacher-exposure-matched-control-v1.json).
It is metadata only and cannot launch a job.

## Exact dose match

| Split | Available parent | Matched control | Balanced arm |
| --- | ---: | ---: | ---: |
| A | 71 episodes / 602 windows / 700,359 tokens | 50 / 401 / 419,165 | 50 / 401 / 419,165 |
| B | 74 episodes / 598 windows / 684,014 tokens | 53 / 402 / 437,537 | 53 / 402 / 437,537 |

At global batch 8, both A treatments have 51 nominal batches per epoch, as do
both B treatments. Matching rows matters because token equality alone could
still change the number of batches. Matching episode count keeps the
number of independent native trajectories equal. No target is duplicated and
the epoch count is not changed to manufacture the match.

The private A and B corpora have `train.parquet` as their only training-data
payload, use mode `0600`, have no teacher-CE development rows, and use the exact
frozen tokenizer, chat template, ordered tool interface, and dense
assistant-turn target policy.
Post-materialization acceptance also requires a non-symlink directory at mode
`0700`, the exact five-file inventory, regular non-symlink files at mode
`0600`, sealed selection/split/interface bindings, aggregate counts, and the
builder fingerprint. The verifier rereads `train.parquet` only as opaque byte
chunks to recompute SHA-256; it never opens Parquet or decodes token arrays.
Their manifests retain the outcome-protocol digest current when those source
corpora were built as provenance. The comparison itself separately binds the
current v2 A/B Fleet-dev protocols; it does not pretend an older corpus label
is the current evaluation contract.

## Deterministic selection

For each covered family, the selector enumerates all nonempty subsets of the
at-most-three available episodes. It prefers the subset nearest that family's
share of the parent distribution, in this order:

1. supervised-token share;
2. dense-window share;
3. episode share;
4. a sealed seeded digest for exact ties.

It then chooses the lexicographically first family-local preference vector
that has an exact joint completion. Suffix reachability bitsets prune the
search, but a memoized joint search proves the final episode/window/token
triple. It fails closed if the exact answer is infeasible or if its 200,000
state bound is exhausted.

This is not a global minimum-distance optimizer or an unbiased random sample.
It is a small, deterministic, auditable way to retain availability-shaped
exposure while exactly holding episode/window/supervised-output dose fixed.

## What changed

Total variation is measured against the original available teacher
distribution over covered families. Lower means closer to that parent.

| Split | Treatment | Episode TV | Window TV | Token TV | Token Gini |
| --- | --- | ---: | ---: | ---: | ---: |
| A | matched control | 0.1361 | 0.1417 | 0.1795 | 0.2796 |
| A | balanced | 0.1834 | 0.1800 | 0.1978 | 0.1878 |
| B | matched control | 0.1392 | 0.1504 | 0.1695 | 0.3286 |
| B | balanced | 0.1820 | 0.1884 | 0.1951 | 0.1941 |

The control is closer to the availability-weighted parent on all three
measures, while the balanced arm has substantially lower token Gini. That is
the intended contrast: the same episode/window/supervised-output dose, with a
different distribution across families.

## Evidence and interpretation boundary

The selection construction read only sealed study splits, sanitized episode
metadata, aggregate coverage receipts, and corpus manifests. It did not read
prompts, traces, flags, answers, scores, token arrays, held-out outcomes, or
external benchmark data. The separate materialization check performed the
opaque SHA-256 byte read described above, without decoding or inspecting the
private training payload.

Only supervised output tokens are available in the aggregate receipts. Input
and context-token exposure is therefore not matched, so this must be called an
**episode/window/supervised-token matched** control, not a fully compute-matched
control.

The two arms must otherwise use fresh identical model initialization, learning
rate, batch size, epoch count, seed, and Fleet development evaluation. Split B
is a later fresh-split/seed confirmation. No outcome claim is valid until the
exact training layout passes development-cluster qualification and a fresh
matched base control is available. Production launch remains closed.
