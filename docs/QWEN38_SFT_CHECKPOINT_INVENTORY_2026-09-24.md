# Qwen3.8 SFT checkpoint evaluation inventory

Date: 2026-09-24

Status: read-only sanitized inventory; no launch or serving mutation authorized

The exact machine-readable receipt is
[`qwen38-sft-checkpoint-inventory-20260924.json`](evidence/qwen38-sft-checkpoint-inventory-20260924.json).
It contains no task identity, prompt, trace, answer, flag, score, or credential.

## Decision

Use this promotion and evaluation order:

1. Finish the exact base-versus-Teacher3K32-step1000 matched pair. It is the
   only current pair with both routes Ready and a fresh accepted live-parity
   receipt. This is an operationally qualified diagnostic, not an accepted
   capability result.
2. Evaluate the prospectively frozen Teacher3K96 step400 successor only after
   step400 exists and passes the complete seal, BF16 export, CPU validation,
   zero-update GPU reload, immutable stage, paused registration, and fresh live
   parity chain. At this observation, the run was at signed step372, step350 was
   the newest complete native checkpoint, and step400 did not exist.
3. Then consider the remaining frozen top-five artifacts. Their exports,
   zero-update reloads, immutable stages, and paused zero-allocation
   registrations are accepted; each still needs fresh matched live parity.

Do not substitute step350 for the predeclared step400 successor or silently
retarget a frozen top-five arm after seeing evaluation outcomes.

## Actionable inventory

| Artifact | Exact source | Export/stage identity | Serving state | First remaining gate |
|---|---|---|---|---|
| Base control | revision `1d4bf0f...7c60c0`, weights `06c94e47...00f352` | `/models/qwen3.8-27b/<revision>` | `chris-q38-base-pass4-v1`, Ready | Whole-wave independent audit and authorization |
| Teacher3K32 step1000 | run `34148b91...8935b`, step1000, 32K manifest `a8d08609...8d6f5` | payload `023c5f8b...26db5` | `chris-q38-t3k32-s1000-v1`, Ready | Same as base; complete matched valid cells |
| Dense-b16 step200 | run `c9bbf581...26ae0`; availability fallback for evicted step300 | payload `2e519ef7...9982` | paused, zero Pods | Fresh base/candidate live parity |
| Teacher3K64 step225 | run `f690d588...4ce17`; availability fallback for evicted step285 | payload `bfadb6b6...80e0` | paused, zero Pods | Fresh base/candidate live parity |
| Teacher3K32-lr1 step500 | run `6b3765ec...0f2a`; availability fallback for evicted step700 | payload `65286cb3...6c900` | paused, zero Pods | Fresh base/candidate live parity |
| Teacher3K96 step300 | run `e7d5c0f8...c1fe`; frozen top-five checkpoint | payload `7e610d80...c51d` | paused, zero Pods | Fresh base/candidate live parity |
| Teacher3K96 step350 | same run; receipt `12c92a50...20e3e` | native checkpoint only | no route | Not selected; preserve as inventory evidence |
| Teacher3K96 step400 | same run; prospectively frozen successor | absent at observation | no route | Complete checkpoint receipt and seal |

Only the base and step1000 routes have current accepted matched live parity.
The four paused candidate registrations must not be called evaluation-ready
from export or registration receipts alone.

## 96K progression

The active full-weight run was healthy and productive at the observation:
one Ready/no-restart eight-GPU Pod, 93.975% 15-minute utilization, signed
optimizer step372, and newest complete checkpoint step350.

- Checkpoint receipts persist for steps 50, 100, 150, 200, 250, 300, and 350.
- Native retention is two; only steps 300 and 350 remained as native directories.
- Immutable seals and BF16 exports exist for steps 50, 100, and 300.
- Steps 50 and 300 passed zero-update finite GPU reload; step100 has no
  accepted GPU-reload receipt.
- Only step300 is staged and registered, and its route is paused.
- Step350 has no seal, export, reload, stage, or route. Step400 was absent.

Step50 and step100 remain fallback evidence, not additional default evaluation
arms. Steps150, 200, and 250 are receipt-only after native eviction.

## Heldout eligibility

The corrected Teacher3K lineage map resolves 496 training task keys, 1,176
exact versions, and 2,886 source sessions across the exact 32K, 64K, and 96K
manifests. Of the former 25-family roster, five families are exposed through
six training aliases. The valid roster is therefore 20 families: development13
plus final7. Exact task-key and task-version overlap was already zero in the
original 25; shared-atom alias lineage nevertheless exposes those five
families.

This clean20 contract applies to checkpoints trained only from the three bound
Teacher3K manifests, including the five actionable SFT candidates and the 96K
native progression. It means training-lineage heldout, not globally untouched:
the tasks have prior execution certification and the final stratum has
historical exposure. Report development13 and final7 separately; the union20 is
descriptive. The current QA33 pool adds zero executable tasks because none of
its 33 versions has a runtime-qualification receipt. Those 33 versions collapse
to 26 stable-atom components: 17 versions in 10 components are exposed, while
16 versions are 16 unexposed singleton components.

Historical Fresh75, teacher-v5, self-SFT, LR30, and Teacher3K262 step10 routes
remain paused. Do not reuse clean20 for them without a distinct exact training-
lineage audit and fresh live parity. Earlier Teacher3K32 step600-900 and
Teacher3K64 step195 routes are superseded learning-curve points, not default
arms.

## Current resource note

At the 12:22Z census, the base and step1000 servers were both Ready/no-restart
but used 0% GPU across the 15-minute window and three direct samples. They
occupied 16 GPUs. No resource change was made. Their OpenAI HTTPRoutes also
reported a rule-overlap condition with another route, while explicit model-
header routing remained bound; every campaign must still verify the exact
served model. Apply the repository's dedicated-capacity lifecycle if no
authorized consumer is imminent.

This inventory authorizes no checkpoint promotion, route resume, evaluator,
model call, or scoring call.
