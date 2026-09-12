# Qwen3.8 blackbox SFT study

This is the execution contract for the next Qwen3.8-27B teacher- and self-SFT
study. It does not treat a published task, a tokenizer-ready transcript, or a
numeric model outcome as proof that a task is runnable.

## Eligible task universe

The frozen input is
[`qwen-blackbox-eligible-v1.json`](../configs/data/qwen-blackbox-eligible-v1.json).
It contains **89 exact task versions** with both:

1. a digest-valid completed execution receipt proving the exact environment,
   verifier, scoring intent, ingest and cleanup path; and
2. a current exact-version read proving that the same starting data, runtime
   seed, environment version and verifier version can still be resolved.

The other 71 members of the historical 160-version roster remain excluded. Nine
of them have clean historical executions but no longer expose a reproducible
starting-data binding. The allowlist is task validity evidence, not SFT-source
eligibility and not authority to launch a job.

The study inventory joins those 89 versions to each task's exact immutable atom
source metadata. It has five applications, six environment labels, reviewed
vulnerability class/category labels and easy/medium/hard difficulty labels. The
splitter uses the exact atom locator as the family boundary; it does not infer a
family from model success or a task-name hash.

## Outcome-blind representative splits

The common final-test lock contains 10 task families. Each of split A and split B
contains 59 train, 20 Fleet development and the same 10 final-test tasks. The two
development sets overlap on only 5/20 tasks; the two train sets overlap on 44/59.
All choices use taxonomy only—no model outcome, transcript, loss or score enters
split selection.

For both development variants, the maximum absolute group-share gap from the
89-task population is below 0.03 for application and environment, below 0.06 for
vulnerability class/category, and below 0.04 for difficulty. The exact manifests
are:

- [`qwen-blackbox-study-final-test-v1.json`](../configs/data/qwen-blackbox-study-final-test-v1.json)
- [`qwen-blackbox-study-split-a-v1.json`](../configs/data/qwen-blackbox-study-split-a-v1.json)
- [`qwen-blackbox-study-split-b-v1.json`](../configs/data/qwen-blackbox-study-split-b-v1.json)
- [`qwen-blackbox-study-train-a-v1.json`](../configs/data/qwen-blackbox-study-train-a-v1.json)
- [`qwen-blackbox-study-train-b-v1.json`](../configs/data/qwen-blackbox-study-train-b-v1.json)

The two smaller `train-*` files are byte-equivalent projections of each outer
split's embedded `training_split`. Corpus and coverage commands consume those
train-only schemas while binding the full outer study split separately.

The 20-task Fleet development set is the per-arm model-selection signal. The
10-task final set stays sealed until a configuration is selected. WebExploitBench
is run after each completed arm as requested, but it is never used to rank or
retune arms; repeated inspection would otherwise turn it into another development
set.

## Source policy

Teacher and self sources are separate treatments. Each source must be a verified
successful session on an exact train task version, with acceptance, trace and
normalized-record digests. Held-out identities are rejected before their message
payloads are accessed.

The dense compiler supervises every compatible visible assistant response once.
Earlier five-ending-window corpora overweighted late submission behavior. The new
coverage gate requires at least one completed non-submission tool round and one
non-submission tool or decision response, enforces per-family episode/token caps,
and blocks a corpus if `submit_report` exceeds 50% of either supervised responses
or supervised tokens. These are structural coverage checks, not a claim that an
individual shell command was useful.

Teacher and self arms are not interpreted as a matched comparison when their task
or token coverage differs. Each source treatment gets its own baseline and HPO
curve.

## Training and evaluation protocol

- Exact initialization: `Qwen/Qwen3.8-27B` at
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- W&B logs training loss and global optimizer step for fitting diagnostics. It
  receives no prompts, messages, traces, flags, token IDs or scores.
- Teacher-reference validation CE is not computed or used for checkpoint choice.
- Checkpoints are written at the configured interval and retained by recency.
- Arm selection uses the v2 Fleet development estimator only: all four fixed
  binary outcomes are averaged within task, candidate and base are paired by
  exact task+seed, and tasks receive equal weight. Ties are retained for a fresh
  confirmation rather than broken with pass@4, training loss or teacher CE.
- The selected configuration is evaluated on the common Fleet final set. Each
  completed checkpoint also runs the requested WebExploitBench protocol through
  the separately qualified Tensorlake adapter, but those results do not tune HPO.

The first wave is explicit, not an accidental Cartesian product: broad learning
rates and epochs are varied within teacher and self treatments, then the promising
region is narrowed in a later wave. Split A/B is a separate robustness factor.
Every arm has unique output and W&B identities and binds exact corpus, source,
split, model and evaluation digests.

The teacher study's current sequencing contract is the self-digested,
non-executable
[`qwen-blackbox-teacher-staged-search-v3.json`](../configs/studies/qwen-blackbox-teacher-staged-search-v3.json).
It supersedes, but does not rewrite, the frozen v2 artifact. The completed
`1e-6` and `3e-5` two-step checks used the legacy constant/no-warmup path and
remain numerical runtime-stability evidence only. V3 binds the separate paired
six-step scheduler/warmup gate before the exact one-node/eight-GPU production
layout. Only after those gates and a fresh matched split-A base control may it
open the literature-grounded split-A LR bracket
(`1e-6`, `1e-5`, `3e-5`, `1e-4`); the `1e-4` arm has an additional extended
development stability/reload gate. Global batches 16 and 32 and horizons of two
and four epochs remain blocked until the earlier outcome barriers resolve.
Stages are serial and each wave is at most four nodes, below the
eight-active-study-node ceiling.

This order is outcome-only: training loss is diagnostic, teacher-reference CE is
absent, and only fresh paired Fleet dev mean-success outcomes select an arm.
The frozen staged-search v2 artifact still binds the historical attempt-1
protocol and is retained as evidence. V3 binds both corrected v2 Fleet-dev
protocols: all four fixed attempts are averaged within each task and candidate
and base are paired by exact task+seed, with uncertainty bootstrapped by whole
task clusters. It also binds the immutable
[`exposure-matched control`](../configs/studies/qwen-blackbox-teacher-exposure-matched-control-v1.json):
the matched-control and balanced arms have identical whole-episode,
dense-window and supervised-output-token counts but different family exposure.
Input/context-token exposure is not measured or matched, so this is not a fully
compute-matched comparison or evidence that either weighting is intrinsically
optimal. The enhanced W&B contract exports only its fixed scalar allowlist,
uses cumulative supervised tokens as the series axis, and requires complete
finish/sync evidence. The current route audit proves the shared base endpoint
is operational, but not an exact matched causal control, so it does not open any
outcome or uplift gate. WebExploitBench remains sealed and ineligible for HPO.
No stage in v3 is presently launchable and the file is not a job request.

### Two-step learning-rate boundary canaries

The inert
[`qwen38-teacher-lr-extremes-dev-v1.template.json`](../configs/qualification/qwen38-teacher-lr-extremes-dev-v1.template.json)
freezes the next dev-only numerical checks at `1e-6` and `3e-5`. Each arm uses
one four-GPU worker, global batch eight (two accumulation rounds), the exact
available-A teacher corpus, per-step checkpoints, scalar-only W&B, and a planned
pause after exactly two optimizer steps. The names, output roots, prepared roots
and W&B IDs are distinct, but neither arm is currently a runnable CLI input.
Both inherited the old constant/no-warmup runtime. Two updates cannot exercise a
5%-of-76-step warmup boundary, estimate task quality, or identify an optimal LR.

The changed runtime has one narrow clean-rejection path. Only a plan that
explicitly opts into the bounded numeric policy can record non-finite loss or
gradient norm as `TRAINING_REJECTED.json` with controller exit zero. A rejected
update is never checkpointed or accepted and never opens the next study gate.
Loader/CUDA exceptions, invalid LR or clock state, storage/checkpoint faults,
W&B tracking or sync faults, watchdog expiry, and unknown terminal states still
write `FAILED.json` and exit nonzero. This prevents an expected scientific
boundary result from being reported as a broken cluster job without hiding an
actual runtime defect.

Before either config may be materialized, bind independently verified terminal
and clean-release evidence for the enhanced-metrics canary and its exact
four-rank zero-update checkpoint reload. Then run the changed runtime's pinned
image CPU preflight, create exact per-arm plan/request digests, preview each
against the dev Jobs API, and recheck duplicate destinations and the aggregate
eight-study-node ceiling. One reviewed POST per arm is the only launch path.

### Scheduler qualification canaries

The inert
[`qwen38-teacher-scheduler-dev-v1.template.json`](../configs/qualification/qwen38-teacher-scheduler-dev-v1.template.json)
prepares a paired dev-only runtime gate at fixed LR `1e-5`, data, seed, global
batch and full 76-step horizon. One arm explicitly selects the compatible
constant/no-warmup pair; the other selects cosine with `warmup_ratio: 0.05`.
The compiler converts that ratio to `ceil(76 × 0.05) = 4` warmup steps. Both
pause after step six, so the cosine arm covers four warmup and two post-warmup
updates without shortening its scheduler horizon.

These configs are not launchable. They still need exact pinned-image CPU
preflight, per-arm plan/request and dev-preview digests, unique destination/W&B
checks, and reviewed create-once submission. Acceptance requires complete scalar
tracking, checkpoint evidence, exact four-rank zero-step scheduler reload and
GPU release. Their losses must not be used to rank schedulers or learning rates.

## Operational gates

New or changed training, serving and Tensorlake paths qualify on the dev cluster
first. Production submission is allowed only after a clean dev receipt, checkpoint
reload and exact duplicate checks. At most eight nodes from this study may be
active at once; the four pre-existing inference endpoints are outside that cap.
Queued work with no allocation consumes zero nodes. Failed, ambiguous or stalled
work is never blindly repeated, and a broken owned allocation is released before
off-node repair.

No capability/HPO arm had been submitted when these manifests were frozen. Two
bounded four-GPU development canaries were subsequently run; they are runtime
qualification evidence, not model-quality observations. Source coverage,
checkpoint serving and the real Tensorlake lifecycle remain explicit
pre-production gates.

### Accepted enhanced-metrics canary

Development run `chris-q38-ta4m-dev1-c69c000a` completed at
`2026-09-11T22:45:37Z` with RayJob UID
`d00dcd6b-8c7e-4ce8-b09b-6e2fef1fadb7` and Workload UID
`2cf5b047-df8d-401d-a22b-d9a7fa9f2460`. The Workload finished successfully,
the RayCluster and Pod are absent, and all four GPUs were released.

A separate zero-GPU, read-only verifier (Pod UID
`de7d57a8-94c2-4b04-a1ae-fa152187daa3`, deleted after success) proved one real
optimizer step, one complete W&B scalar event over the eight-field allowlist,
finite loss/gradient/throughput scalars, acknowledged W&B sync, no evaluation
telemetry, and a nonempty 21-file, 324,621,264,731-byte checkpoint. The terminal,
W&B and checkpoint-receipt digests are respectively
`f2c3c73ee64bbe576c361adce48665fd150d6f18dfa4f498670bb62d38810207`,
`a9fa8f6a6419eff871ae84f36c5d2a0b4ba6a966967845c2fc484ae6a9c28d1f`,
and `d8d18835c8b3b36e2155c85caa1c40b2d09bd16983ab8bc006b8ccdac595fc59`.
No `FAILED.json` or tracking-incomplete receipt exists. This accepts scalar
tracking and checkpoint creation only; it is not evidence of task uplift.

### Four-GPU metrics-canary recovery gate

The create-once
[`qwen38-ta4m-dev1-reload-v1.template.json`](../configs/qualification/qwen38-ta4m-dev1-reload-v1.template.json)
binds the planned recovery check for dev metrics canary `chris-q38-ta4m-dev1`.
It is deliberately non-launchable: terminal Job/RayJob identities, the paused
receipt digests, a CPU checkpoint-seal digest, and unique recovery run/output/W&B
identities are unresolved. Do not fill them from names or chat summaries.

After the exact source Job has succeeded with a digest-valid step-one
`TRAINING_PAUSED.json` and released its allocation, seal `global_step_1` once on
CPU with the command recorded in the template. Materialize a private recovery
config by copying the source scientific plan, removing `pause_after_step`,
assigning new run/output/W&B identities, and adding the exact manifest file hash
under `recovery` with `mode: validate`. Run the normal pinned-image CPU preflight
and dev Jobs API preview before any submission.

`validate` is the only zero-additional-step recovery mode. It restores the model,
optimizer, scheduler, RNG and sampler on every original rank, then exits without
training or teacher-reference CE. The source checkpoint has world size four, so
one four-GPU worker is the exact and sufficient qualification shape; eight GPUs
would be a topology mismatch, while a true `resume` would execute later optimizer
steps and answer a different question.

The CPU seal subsequently completed successfully. Create-once helper Pod UID
`0fcd52a5-433e-4f46-8f26-a918c0d13884` verified all 21 files and
324,621,264,731 bytes twice, then was deleted with its ConfigMap and confirmed
absent. The canonical checkpoint-manifest file SHA-256 is
`306740464dd13cfb96691ea69ccd65f2ba835a2146aeca8b7934f3d5f6186b32` and
its embedded receipt SHA-256 is
`ae009aa5790bdff2f4025962facf7db57c09955a0fc09c72459e926d6a03ee07`.
This proves CPU integrity only: `gpu_reload_verified` remains false until the
four-rank, zero-update development reload finishes and releases its allocation.

### Step-20 checkpoint-cadence reload gate

The later eight-rank cadence source run is terminally successful in the Jobs
API, RayJob and Workload projections and released its RayCluster, Pods and all
eight GPUs. Its exact step-20 checkpoint seal contains 33 files and
324,627,486,731 bytes. The zero-update reload
`chris-q38-ta8-cad20-reld6-db29f2b4` also succeeded and released all eight GPUs.
A separate zero-GPU verifier rehashed the full source inventory, confirmed all
eight ranks plus sampler and scheduler restoration, zero optimizer updates and
no training artifacts, then exited successfully and was deleted.

The sanitized, self-digested evidence is
[`2026-09-12-checkpoint-cadence-reload-dev-v1.json`](evidence/qwen38-study/2026-09-12-checkpoint-cadence-reload-dev-v1.json).
It binds digest-valid source start, step-21 pause, W&B sync and step-20/21
checkpoint receipts as well as the independently verified step-20 reload. The
source cadence RayCluster and Pods were TTL-cleaned before their immutable UIDs
and restart counts were captured, so full terminal provenance remains incomplete
and no zero-restart claim is made for that source run. The combined operational
gate in
[`qwen-blackbox-teacher-staged-search-v5.json`](../configs/studies/qwen-blackbox-teacher-staged-search-v5.json)
is accepted narrowly enough to open offline preflight and reviewed dev preview
for the bounded LR qualification cells. It does not authorize a submission by
itself. Production remains blocked on the split-A base control and exact
production preview.
