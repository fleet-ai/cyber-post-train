# Offline Qwen outcome-only study planner

`python -m training.study_plan study.json --output prepared/study.json`

This validates an explicit table and atomically writes a digest-bound treatment
manifest. It does **not** prepare executable requests, read task/source payloads,
contact W&B, allocate resources or launch jobs. The separate one-run CLI remains
the supported preparation/preflight/dev/production route.

## Fill only evidence-backed bindings

Start from [the gated template](../configs/studies/qwen-blackbox-sft-v1.template.json).
It binds the reviewed A/B split, train-only projection, dev-task-set and common
final-test identities. The exact variant seeds are named strings, distinct from
integer training RNG seeds. Each split has separate teacher/self corpus,
source-selection and qualification digests. Its `fleet_dev_protocol_sha256`
must match the one embedded by corpus preparation and runtime compilation.

The template intentionally **does not compile**: private corpus/protocol/decision
digests are placeholders, not invented evidence. Self sources are explicitly
`blocked_no_compatible_sources`, with zero qualified sessions. Teacher coverage
currently supplies 71 candidate sessions on A and 74 on B; this is not itself
a completed corpus or training qualification. Set `status: qualified` only after
the actual source receipt and nonempty corpus pass their checks, and bind those
exact receipt/manifest digests. The compiler rejects unqualified/empty sources.

For the first teacher-only wave, copy the template under a globally new study
name, keep only the four teacher LR rows in `a-lr-screen`, remove all later
stages, remove split/source B and both self-source declarations, and resolve
every remaining digest from evidence. Do not populate a fake digest merely to
pass syntax. Each new wave gets a new manifest/name; previously accepted arms
are referenced by their outcome decision, never copied into another launch.

## Staged experiment, not a Cartesian sweep

The written first-stage rows bracket LR at `1e-6`, `3e-6`, `1e-5`, `3e-5`, with
batch 8 and one epoch on A, separately for teacher and self if qualified. This
30-fold bracket is a proposed screen, not a claim that any LR is optimal.

After fresh Fleet dev outcomes, freeze a decision receipt and only then resolve
the follow-up LR placeholders. Follow-up rows explicitly check B plus RNG seed
43, batches 16/32, and epochs 2/4. They vary one factor relative to the screen's
selected batch-8/epoch-1 baseline, not every combination. Each follow-up stage
requires the preceding outcome-decision digest. The compiler rejects unresolved
LRs/decisions, repeated scientific rows and run/output/W&B identity reuse.
Tied dev outcomes stay tied for fresh confirmation; there is no loss tie-breaker.

Every arm initializes from the exact fresh Qwen3.8 base and uses one eight-GPU
node, microbatch 1/GPU, `task_outcomes_only` and `eval_interval: 0`. It exports
training loss/step diagnostics only; the separate outcome protocol controls
selection. Final Fleet and WebExploitBench outcomes remain sealed from HPO until
the selection decision is frozen. Distinct hashes do not prove task-family
disjointness—review the actual split/source qualification separately.

## Capacity and execution gates

Stages cannot overlap: all previous stages must be terminal and their allocations
released. Each stage plus declared `other_active_study_nodes` must fit
`max_active_nodes <= 8`. The four pre-existing inference endpoints are explicitly
outside this **study** ceiling; new serving/evaluation allocations for the study
are not automatically exempt. A queued job without an allocation uses zero
active nodes. Static declarations go stale: one submitter must recheck all
active/admitted study allocations across dev/prod before each actual launch.

This planner is not an admission lock and its output grants no new authority.
Corpora, real task executability, device/runtime compatibility, checkpoint
save/reload, telemetry and release must still pass the existing pinned-image
and bounded dev-cluster gates. Use distinct dev/prod run/output/W&B identities.
The runtime change has local regression coverage; local tests alone do not
qualify a new distributed GPU run.
