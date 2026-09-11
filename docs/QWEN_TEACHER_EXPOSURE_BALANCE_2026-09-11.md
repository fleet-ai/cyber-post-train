# Qwen teacher-exposure balance ablation

## Result

The existing A/B teacher corpora are dense and progress-bearing, but their
supervised-token exposure is materially availability-weighted across task
families and applications. A frozen comparison treatment now selects complete
certified episodes toward 16,384 supervised tokens per covered family. It does
not read scores or payloads, split an episode, rewrite messages or masks, repeat
a target within an epoch, or change the tokenizer, chat template, or tool
interface.

This is a **teacher-availability treatment**, not a claim that equal family
exposure is intrinsically optimal. Its public, self-digested lock is
[`configs/studies/qwen-blackbox-teacher-balanced-exposure-v1.json`](../configs/studies/qwen-blackbox-teacher-balanced-exposure-v1.json).

## Evidence boundary

The audit read only the sealed study inventory/splits and sanitized certified
source-coverage and selection metadata. It did not read prompts, traces, flags,
answers, scores, token arrays, held-out outcomes, or WebExploitBench data.

| Split | Qualified candidates | Covered / train families | Available selection | Balanced selection |
| --- | ---: | ---: | ---: | ---: |
| A | 84 | 29 / 59 | 71 episodes, 700,359 tokens | 50 episodes, 419,165 tokens |
| B | 89 | 30 / 59 | 74 episodes, 684,014 tokens | 53 episodes, 437,537 tokens |

Thirty A families and twenty-nine B families have no qualified successful
teacher episode. Balancing cannot manufacture coverage for them; both arms keep
the same covered-family set.

## Family imbalance

The coefficient of variation and Gini coefficient are calculated over total
supervised tokens per covered task family.

| Split / treatment | Min | Median | Max | Max / min | CV | Gini |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A available | 4,630 | 21,990 | 48,544 | 10.48 | 0.570 | 0.326 |
| A balanced | 4,630 | 15,224 | 30,126 | 6.51 | 0.350 | 0.188 |
| B available | 2,721 | 20,071.5 | 47,841 | 17.58 | 0.585 | 0.334 |
| B balanced | 2,721 | 15,901.5 | 30,126 | 11.07 | 0.365 | 0.194 |

The treatment lowers family-token Gini by 42% in both splits. It cannot make
exposure equal because the unit is an indivisible native episode: some families
only have a long episode and others only a short one. In split A, the largest
family's share therefore moves slightly from 6.93% to 7.19% even while the full
distribution becomes substantially less unequal. This residual is explicit,
not silently corrected with duplicated or partial targets.

## Application imbalance

Applications are the verified inventory labels. Shares are of each treatment's
own supervised-token total; means normalize for the different number of covered
families per application.

| Split | Application | Families | Available mean/family (share) | Balanced mean/family (share) |
| --- | --- | ---: | ---: | ---: |
| A | current | 6 | 23,088.7 (19.78%) | 14,879.0 (21.30%) |
| A | fakelook | 5 | 32,082.2 (22.90%) | 19,034.2 (22.70%) |
| A | fentry | 4 | 45,993.5 (26.27%) | 14,805.8 (14.13%) |
| A | fira | 11 | 14,096.8 (22.14%) | 12,128.5 (31.83%) |
| A | fubspot | 3 | 20,792.3 (8.91%) | 14,028.0 (10.04%) |
| B | current | 6 | 21,358.8 (18.74%) | 15,043.8 (20.63%) |
| B | fakelook | 6 | 34,415.0 (30.19%) | 18,917.8 (25.94%) |
| B | fentry | 4 | 39,298.2 (22.98%) | 16,135.8 (14.75%) |
| B | fira | 11 | 10,639.7 (17.11%) | 10,775.9 (27.09%) |
| B | fubspot | 3 | 25,047.0 (10.99%) | 16,896.3 (11.59%) |

The family-level policy reduces the very high mean exposure for `fentry` and
`fakelook`. It does not force equal application shares: doing so would create a
second treatment and would overweight each `fira` family merely because more
covered `fira` families are available.

## Frozen selection rule

For each reviewed training family, the selector:

1. starts from certified, verified-success episode metadata that already passes
   the dense-target and non-submission-progress gates;
2. chooses the whole episode whose family total is nearest 16,384 supervised
   tokens;
3. adds another whole episode only when it strictly moves the total nearer the
   target, up to three episodes and the existing hard ceiling of 49,152 tokens;
4. prefers a new source model only on equal-distance ties, then uses the sealed
   seed and episode identity as the deterministic tie-breaker.

The 16,384 target was frozen before model outcomes and matches one maximum SFT
sequence-length budget. It is a convenient exposure scale, not a tuned optimum.
The available and balanced arms should otherwise keep model initialization,
split, optimizer recipe, epoch count, W&B scalar policy, and Fleet development
evaluation protocol identical.

## Interpretation and next gate

At one epoch the balanced arm has 40.1% fewer supervised tokens in A and 36.0%
fewer in B. Therefore it tests the joint effect of less availability dominance
and less total teacher exposure; it is not a pure importance-weighting estimate.
Matching total exposure by replaying selected targets would violate this
treatment's no-duplication rule, while changing epoch count would introduce a
different optimization treatment. Report the token-budget difference directly.

The private source-selection receipts are materialized with mode `0600` and
bound by the public config, but balanced Parquet corpora have deliberately not
been built from this metadata-only audit. Their source records must be joined by
the existing digest-checking corpus builder, then exact counts, file digest,
tokenizer identity, target coverage, CPU preflight, and a dev GPU canary must be
qualified before any study arm can launch.
