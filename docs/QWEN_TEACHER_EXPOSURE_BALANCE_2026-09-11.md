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
| A | 84 | 29 / 59 | 71 episodes, 602 windows, 700,359 tokens | 50 episodes, 401 windows, 419,165 tokens |
| B | 89 | 30 / 59 | 74 episodes, 598 windows, 684,014 tokens | 53 episodes, 402 windows, 437,537 tokens |

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

The private source-selection receipts and A/B train-only Parquet corpora are now
materialized with mode `0600` and bound by the public config. There is no dev
Parquet and no teacher-cross-entropy target. The available controls were also
rematerialized under the same current target-policy fingerprint; their Parquet
digests are byte-identical to the earlier controls. This isolates the corpus
change to whole-episode selection while preserving the exact tokenizer, chat
template, tool interface, dense target policy, and outcome-evaluation protocols.

| Split / treatment | Corpus manifest | Train Parquet |
| --- | --- | --- |
| A available | `sha256:c4945ddb7cd8fd0b7b9ebbd9bad9b96215b0905c059ef1b22d71d4f214b4eb8e` | `sha256:a230b82655ccfea299ba8be12ac271706af89e98f81120e8289eefeeec9d2869` |
| A balanced | `sha256:b8aa6d6f9ae19fd509a90c3d2b6343631386d6bc14b019e620810da274bd9cf8` | `sha256:e59e05912bdadbbd0f147a02a5da500fc8242bf565fb0206c0e36c0394289d9b` |
| B available | `sha256:7b43bc25aec1390252b579043199d8a1e3295d29272b565d4c45dddde3cd15de` | `sha256:c0575a063dbbf1dee09a6ac27d5428f80d38e83248d8b10b0ab689822ec594f7` |
| B balanced | `sha256:1650edbb53ee33439311a8338207d95923f72283087e0069b89eb8114fafb194` | `sha256:a763aabcf3a16aeb761a7798df268096d9eb10eee788f924bfa0726b91b6272a` |

The common current target-policy digest is
`sha256:f371e87d02e6a32d32b511d9439caff97540709f17f346830d7989781c550cc1`.
An initial split-B materialization bound an obsolete protocol digest; its
create-once output is intentionally excluded. The locked split-B manifests
above bind the current sealed protocol instead.

The [study input](../configs/studies/qwen-blackbox-teacher-balanced-v1.json) and
[frozen plan](../configs/studies/qwen-blackbox-teacher-balanced-v1.plan.json)
remain metadata-only and explicitly nonlaunchable. CPU/image preflight and a
development-cluster GPU canary must qualify the exact corpus/runtime bindings
before any production study arm can launch.
