# Qwen3.8 teacher-SFT: next development arms

This preparation chooses the **two smallest useful next runs**. Both continue
already accepted partial Qwen3.8-27B teacher-SFT runs; neither starts over or
repeats an accepted optimizer step. No cluster, Jobs API, or W&B call was made
while preparing them.

The sealed study definition is
[`qwen-blackbox-teacher-next-dev-completions-v1.json`](../configs/studies/qwen-blackbox-teacher-next-dev-completions-v1.json).

## Ranked runs

| Rank | Continuation | Learning rate | Accepted starting point | Work left | Why this rank |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | [LR3e-5 step 6 → 76](../configs/qualification/qwen38-teacher-lr30-step6-resume-to76-dev-v1.template.json) | 0.00003 | 6 finite updates plus a successful eight-rank, zero-update checkpoint reload | 70 updates | It has the strongest recovery evidence and is the lowest-risk path to the first complete checkpoint under the current outcome-only protocol. |
| 2 | [LR1e-4 step 21 → 76](../configs/qualification/qwen38-teacher-lr100-step21-resume-to76-dev-v1.template.json) | 0.0001 | 21 finite updates and two written checkpoints | 55 updates | It is a deliberately high learning-rate boundary, 3.33× the first run, so it gives a wide and useful contrast. It must wait for the existing checkpoint pair and step-21 reload checks. |

The 76-step total is `ceil(602 training windows / batch 8) × 1 epoch`.
Everything except learning rate stays fixed: model revision, training bytes,
one epoch, batch 8, one example per GPU, cosine schedule, 5% warmup, 16,384-token
limit, seed 42, outcome-only validation, and eight GPUs. Starting new batch-size,
epoch, scheduler, or data-mix runs now would make the comparison harder to
interpret before this learning-rate contrast has a held-out Fleet result.

## Data and held-out selection

The repository's reviewed split contains 89 task families: 59 training, 20
development, and 10 final-test families. Versions and aliases from one family
remain together. The split records representation across application,
environment, vulnerability family, and difficulty. The exact teacher corpus
contains 602 windows and 700,359 supervised tokens drawn from successful
sessions on 29 of the 59 training tasks. The other training tasks do not gain
fabricated demonstrations; this coverage limit must remain visible when results
are interpreted.

Model selection uses only the 20 held-out Fleet development tasks. Base and
candidate use the same OpenCode 1.18.27 harness, `bash` and `submit_report`
tools, context and compaction settings, temperature 0.6, top-p 0.95, and four
fixed seeds. The only ranking signal is the paired change in full-task success
for the exact same task and seed. Training loss is recorded only to diagnose
whether fitting is healthy.

WebExploitBench is **not** used to choose the learning rate, checkpoint, stopping
point, or tie-break. Its results remain sealed until the Fleet development
decision is frozen, after which it can be reported as an external matched-harness
measurement. This avoids tuning directly to the benchmark we want to report.

The exact source corpus bytes and target policy are frozen by digests. The
sanitized corpus receipt does not expose the original teacher-collection harness,
so this plan does not invent or claim one. The exact harness above is the
base-versus-candidate evaluation harness.

## Live gates still required

LR3e-5 is ready only for fresh live checks. Before its single dev submission,
an operator must refresh authentication; materialize the inert template; run
the exact-image CPU preflight and authenticated dev preview; prove the output,
W&B, API, Kubernetes, and journal identities are unused; recheck the source
checkpoint; confirm c1/q1 priority 10,000 with no automatic requeue; and count
the requested nodes inside the eight-node experiment limit.

LR1e-4 has two additional blockers: one read-only proof must fully rehash both
existing checkpoints, and the step-21 checkpoint must pass an exact one-node,
eight-GPU, zero-optimizer-step reload. Only then may its continuation template
be filled and subjected to the same fresh live checks.

For either completed run, acceptance requires:

- complete, finite per-step scalar history in a unique W&B run;
- a digest-valid terminal step-76 checkpoint and independent full rehash;
- an exact all-rank reload that performs no optimizer update;
- deterministic BF16 export plus model and tokenizer reload;
- create-once staging, serving registration, and live parity;
- release of every owned GPU allocation; and
- complete, authoritative matched base/candidate outcomes for all 80 Fleet
  development task-seed pairs.

These files are inert preparation. They authorize no production job and perform
no submission by themselves.
