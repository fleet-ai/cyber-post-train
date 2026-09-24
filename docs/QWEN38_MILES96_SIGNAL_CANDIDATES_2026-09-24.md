# Qwen3.8 Miles96 signal candidates — 2026-09-24

## Result

The current score-blind order is:

1. `7317` — phase1-A
2. `0756` — next, replacing the now-invalid `8095` choice
3. `f294`
4. `2c50`

The machine-readable packet is
[`configs/qualification/qwen38-miles96-signal-candidates-20260924-v2.json`](../configs/qualification/qwen38-miles96-signal-candidates-20260924-v2.json).
It is deliberately `launchable: false`. This review made no live API call,
created no environment, ran no verifier, previewed no job, and launched no
workload.

## What these candidates are for

Each candidate is a small reward-signal check, not a learning run. It starts
eight predeclared episodes, at most two at once, and performs zero optimizer
steps. Historical results are used only to choose tasks that are likely to
produce both successful and unsuccessful outcomes. They are not reused as new
training rewards.

A candidate qualifies only when all eight slots are accounted for and the new
run proves all of the following:

- at least two episodes finish normally and receive a finite grade;
- the new grades contain at least two distinct values;
- each grade has its own verifier execution identity;
- every task instance is released;
- the runtime preflight passes; and
- no optimizer update or checkpoint exists.

## Why this order

`7317` is first because it is the existing phase1-A implementation target. Its
sealed earlier pass@4 record had four completed episodes with a two/two split.

`0756` is second because it is the reviewed replacement for `8095`. A sanitized
session summary records four completed episodes with a two/two split, matching
the useful signal profile of the replaced task. It also adds an independent
Fira task family.

`f294` and `2c50` each have four sealed completed episodes with three passes
and one non-pass. They remain useful mixed-signal candidates, but follow the two
two/two tasks in the fixed reviewed order. The packet records only counts and
digests, never reward magnitudes.

## Split and contamination review

The current authority is the transitive shared-atom split in
`fleet-blackbox-lineage-safe-split-20260924-v2.json`, not the older split labels
inside `qwen-blackbox-eligible-v1.json`.

All four selected exact task versions are assigned to `train`. Each belongs to
a singleton shared-atom component, and none of those components or groups
appears in `dev` or `final_test`.

`8095` is excluded because the current split assigns its exact version and
component to `dev`. Older artifacts that called it a training task are stale.
It must not re-enter this reward-qualification queue.

All four selected families were already exposed to Teacher3K training data.
That is acceptable for training and reward-signal acquisition, but it means
none of these four may be presented as independent held-out evidence for a
Teacher3K checkpoint. The same warning applies to `8095`: its family was
Teacher3K-exposed through a composite alias, so its current `dev` label does not
make it a valid Teacher3K held-out result.

## Exact live checks still required

The packet freezes the expected task, environment, data, verifier, lineage,
and ordered-tool identities. This includes the exact cyber contract, the
32,400-second environment lifetime, the verifier entry point (`verify`), and
the pinned source digest of the raw-to-OpenAI tool transformation. A launcher
must still repeat a fresh task GET no more than 900 seconds before use and
compare every listed field exactly. At session open it must also prove that the
raw tools are exactly `bash` followed by `submit_report`, including their
schemas, and that the FTI/OpenAI projection matches the sealed projected
digest.

Operational checks are intentionally outside this task-selection packet. A
launch still needs fresh duplicate and output-absence checks, capacity proof,
server previews, the root `fleet.ai/failure-alerts: "off"` annotation,
create-once submission, and exact-UID terminal monitoring and release.

Any changed task, environment, verifier, tool schema, split role, family
component, or source digest requires a new reviewed packet. A stale live receipt
cannot be treated as launch approval.
