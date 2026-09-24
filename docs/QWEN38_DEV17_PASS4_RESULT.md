# Qwen3.8 development-set comparison: base model vs step 1000

This is a descriptive result on 17 Fleet development tasks. It is useful for
diagnosis, but it is **not** the final held-out result.

## What was compared

The base Qwen3.8 model and the step-1000 checkpoint each attempted the same 17
tasks four times. The four attempts used seeds 48, 54, 61, and 62, for 68
attempts per model. Both models used OpenCode 1.18.27 and the same task list,
limits, sampling settings, and context-handling rules.

Every model-and-seed group finished with all 17 expected results accepted,
locally preserved, and free of unresolved review items. A score-blind validity
check confirmed this before any outcomes were opened.

## Aggregate result

| Model | Tasks solved at least once | Pass@4 | Successful attempts | Attempt success rate |
|---|---:|---:|---:|---:|
| Base Qwen3.8 | 7 / 17 | 41.18% | 15 / 68 | 22.06% |
| Step 1000 | 0 / 17 | 0.00% | 0 / 68 | 0.00% |

The step-1000 checkpoint was 41.18 percentage points lower on task-level
pass@4. A paired bootstrap over tasks gives a 95% interval from -64.71 to
-17.65 percentage points. “Paired” means each model was compared on the same
task, rather than treating the two task lists as unrelated samples.

This is strong evidence of a regression on this development set. It does not,
by itself, tell us whether the cause was the training data, the training
settings, checkpoint selection, or an interaction with this harness. Those
questions require targeted follow-up experiments.

## Why checkpoint selection was not influenced by these results

The six checkpoints to compare were recorded in pull request 560 at exact head
`731ef08d261e84a5e248d5964d38a2b839703a0f` before any outcome was opened.
That frozen plan contained: base, step1000, b16-step300, 64k-step285,
lr1-step700, and 96k-step100. The result reported here covers only base and
step1000; the other planned comparisons are separate work.

A manually entered time in the original local boundary receipt was inaccurate.
The preserved Git and file times establish the real order: the six-checkpoint
plan was committed first, the selection boundary was then finalized, the
aggregation program was independently reviewed, and only then was the aggregate
result written. The correction changes no model choice or metric.

## Important limits

- These are development tasks, not the final held-out benchmark.
- The four seeds are repeated evaluation attempts, not four independently
  trained models.
- Only aggregate counts are published here. Task-level outcomes and private
  rollout content remain excluded.

## Evidence

- [Validity receipt](evidence/qwen38-dev17-pass4-validity-20260923.json)
- [Aggregate receipt](evidence/qwen38-dev17-pass4-aggregate-20260923.json)
- [Chronology correction](evidence/qwen38-dev17-pass4-chronology-correction-20260923.json)
- [Independent review](evidence/qwen38-dev17-pass4-independent-review-20260923.json)
