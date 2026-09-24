# Qwen3.8 development comparison: incomplete, not a pass@4 result

The earlier report said that base Qwen3.8 solved 7 of 17 Fleet development
tasks while the step-1000 checkpoint solved none. That result is now
**superseded**. It treated attempts that reached OpenCode's output limit, or
whose process exited with an error, as if they were ordinary model failures.
The scientific protocol requires both kinds of attempt to be held out of the
score.

## What remains valid

The intended comparison used the same 17 tasks and four seeds—48, 54, 61 and
62—for both models. The checkpoint-selection chronology also remains valid.
What failed was the outcome-validity gate after the attempts had run.

| Model | Planned attempts | Normally completed | Output-limit holds | Process errors |
|---|---:|---:|---:|---:|
| Base Qwen3.8 | 68 | 52 | 15 | 1 |
| Step 1000 | 68 | 18 | 49 | 1 |

Only a normally completed OpenCode run can be a valid model outcome. The
score-blind lifecycle audit therefore leaves at most 52 eligible attempts for
the base arm and 18 for the step-1000 arm—not 68 each. The public correction
does not contain the private per-task-and-seed mapping needed to construct a
replacement roster.

## Scientifically correct result today

There is no complete pass@4 result yet. There is also no valid paired task
delta or confidence interval. Missing attempts differ greatly between the two
models, so treating them as zero would confound capability with output-limit
and process behavior. The old 41.18% versus 0%, the stated -41.18 percentage
point difference, and its bootstrap interval must not be cited as a valid
capability comparison.

The private attempt roster must next be joined score-blindly to the lifecycle
census. For every task and seed where either arm is invalid, a fresh paired
replacement seed must be recorded before execution. The original attempts stay
preserved. Scores may be opened only after every task has four valid paired
attempts; a held attempt is never silently replaced with zero.

## Evidence

- [Outcome-validity supersession](evidence/qwen38-dev17-pass4-outcome-validity-supersession-20260924.json)
- [Original validity receipt—superseded](evidence/qwen38-dev17-pass4-validity-20260923.json)
- [Original aggregate—superseded](evidence/qwen38-dev17-pass4-aggregate-20260923.json)
- [Original independent review—superseded](evidence/qwen38-dev17-pass4-independent-review-20260923.json)
- [Chronology correction—still valid](evidence/qwen38-dev17-pass4-chronology-correction-20260923.json)
