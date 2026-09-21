# Qwen3.8 high-throughput trajectory collection plan

## Status and purpose

This is an execution plan, not a launch authorization.  It specifies how to
grow two useful Qwen3.8 training-data sources without changing the task
contract, leaking held-out tasks into training, or turning private reasoning
into training data.

The two sources stay separate from collection through corpus publication:

1. **strong-teacher successes** — successful sessions from an explicitly named
   stronger model; and
2. **Qwen self-successes** — successful sessions from the exact Qwen3.8 model
   revision being studied.

That separation makes the later questions answerable: whether a result came
from teacher imitation, Qwen's own successful behavior, or a deliberately
defined mixture.  It also prevents a small self-success pool from being made to
look large by repetition.

The existing broad teacher corpus already contains 57,384,881 distinct visible
assistant target tokens from 2,886 verified successful sessions.  It is already
above the 20-million-token scale target.  The near-term collection priority is
therefore a **new, large Qwen self-success corpus**: reach at least 20,000,000
distinct supervised visible-action target tokens before treating self-SFT as a
standalone production comparison.  Teacher collection remains useful only for
new, proven training families or an explicitly defined source/harness contrast;
it should not repeat old attempts simply to inflate a count.

This plan intentionally reuses the existing components:

- `evals/fleet/evaluate.py` and its PostgreSQL-backed OpenCode workers for
  exact rollout collection;
- `training/cli.py export` and `training/cli.py normalize` for private session
  export and normalization;
- `training/fleet_teacher_corpus.py`, `training/dense.py`, and
  `training/corpus_subset.py` for evidence-bound dense SFT construction and
  duplicate removal; and
- `training/task_family_split.py` plus the locked current study split for
  family-safe train/held-out separation.

It does not add a second queue, a new rollout harness, or a trace database.
There is one deliberately small integration task before a new scale-up corpus
can be launched: an adapter from a sealed generic collection campaign to the
existing corpus-admission inputs.  That adapter is described below.  It should
be a narrow evidence translator, not a second collection framework.

## The scientific unit: a task family, not a task version or a window

One vulnerability family can have several task versions.  Those versions can
share the same underlying weakness, environment, or solution path.  They must
all have one role: training, development, or final test.  Splitting versions
or windows from the same family across roles would make held-out scores
misleadingly easy.

The current study split is immutable.  Its existing training families remain
training families, and its development and final-test families remain excluded
from every collection, export, corpus, and training request.  In particular,
the current 17 development and 8 final-test task families must never become
training data, even if a new version of one appears later.

For a scale-up study, create a reviewed metadata-only inventory of *new*
families.  Before splitting it:

1. attach every new version to its application and task-family identifier;
2. remove any family already assigned to the current development or final-test
   split;
3. make a version of an already-known training family inherit its existing
   training role; and
4. keep every genuinely new family together, then use the existing splitter to
   make an 80% training, 10% development, and 10% final-test split.

The new development/final partitions are a **new scale-up study**, not a way to
rewrite results from the locked study.  The split must be stratified over the
reviewed metadata fields already expected by `task_family_split.py` (application,
environment, difficulty, and vulnerability family).  A group with more than
25% of the exact task versions should make the split command fail for review;
the cap must not be silently relaxed.

The command shape is intentionally already in the repository:

```sh
uv run --locked python -m training.task_family_split \
  --inventory <private-reviewed-new-family-inventory.json> \
  --output <private-create-once-scale-up-split.json> \
  --seed qwen38-scale-up-v1 \
  --ratios-json '{"train":0.8,"dev":0.1,"final_test":0.1}' \
  --max-group-task-version-fraction 0.25
uv run --locked python -m training.task_family_split \
  --inventory <same-private-inventory.json> \
  --output <same-create-once-scale-up-split.json> \
  --seed qwen38-scale-up-v1 \
  --ratios-json '{"train":0.8,"dev":0.1,"final_test":0.1}' \
  --max-group-task-version-fraction 0.25 \
  --check
```

The inventory and split stay private: they identify task instances.  Published
receipts may contain only their digests and aggregate counts.

### The one missing bridge before a scale-up launch

The generic collector and the current broad-teacher builder are both sound, but
they do not yet plug together for a new scale-up split without an explicit
adapter:

- `data-fleet-teachers` intentionally requires the frozen 75-task
  `study_split_v2` format; it must not be tricked into treating a new study as
  that old one.
- `task_family_split.py` correctly emits the newer parameterized family-split
  format, which the old teacher builder does not accept directly.
- generic OpenCode campaign attempts keep their own canonical session/attempt
  evidence.  The job-ID transcript exporter is useful for an explicitly named
  job-backed source, but does not automatically turn generic campaign attempts
  into SFT inputs.

Before collecting a large new wave, add one small, tested
**collection-to-admission adapter**.  Its only job is to take an exact campaign
receipt, the matching parameterized family split, and the canonical private
attempt/session evidence; then emit private normalized records plus an
evidence-bound inventory in the schema expected by the existing dense corpus
admission code.  It must:

1. preserve every source session, exact task/version, model, harness, template,
   tool, and authoritative-grading binding;
2. map only training-family records to train candidates and prove that
   development/final families map to zero candidates;
3. reject any missing or ambiguous attempt rather than selecting a replacement;
4. produce fixed-reason aggregate rejection counts and digests only in its
   public receipt; and
5. have fixtures for a family-version boundary, a held-out exclusion, a
   duplicate session, an opaque-compaction rejection, and a private-reasoning
   rejection.

The adapter must not mutate a split, create a rollout, score a session, or
expose text.  Once it exists, `training/dense.py` remains the single windowing
and loss-mask implementation.  The existing 75-task builder remains the right
path for its already-locked teacher corpus; the adapter is the incremental path
for a new scale-up study and for Qwen self-success data.

## Collection treatment shared by both sources

Both sources use the same qualified OpenCode treatment unless a later experiment
explicitly changes one field and calls that out as the intervention:

| Item | Frozen treatment |
| --- | --- |
| Agent harness | OpenCode 1.18.27, exact released binary digest |
| Context | 262,144 tokens, native compaction with automatic continuation |
| Context reserve | 20,000 tokens of compaction headroom |
| Response budget | 32,768 tokens per model response |
| Tools | exactly `bash` and `submit_report` in the ordered qualified schema |
| Task binding | exact task, environment, data, and verifier version tuple |
| Model binding | repository, immutable revision, served identity, and observed server profile |
| Sampling | a predeclared temperature, top-p, and seed family; attempt number changes the seed |
| Attempts | four attempts per selected training task version in one predeclared wave |

The collection configuration must set `training_data_eligible: true`; ordinary
evaluation configurations default to false and remain sealed evaluation
material.  This is an explicit distinction, not a label added after a rollout.

Use the existing lifecycle rather than a handwritten worker:

```sh
cyber-post-train eval prepare <reviewed-collection-config.yaml> --output <private-wave-root>
cyber-post-train eval preflight <private-wave-root>
cyber-post-train eval init <private-wave-root>
cyber-post-train eval run <private-wave-root> <route> <unique-worker-id> --limit <bounded-count>
cyber-post-train eval status
```

`prepare` freezes the plan; `preflight` checks exact task, runtime, serving,
image, and OpenCode bindings without creating scored sessions; `init` creates
an empty dedicated PostgreSQL ledger; and `run` claims only the frozen pending
cells.  Each wave gets a different create-once name and its own database.  One
designated creator owns initialisation.  Multiple workers may run in parallel,
but each has a different worker identifier and uses the same ledger.

The PostgreSQL ledger already provides the needed duplicate protection:

- a cell is the immutable combination of plan, source route, exact task version,
  and attempt;
- workers claim pending cells atomically, so two workers cannot knowingly run
  the same cell;
- expired or ambiguous work moves to review and is never automatically replayed;
  and
- each live successor wave has a new sealed operation authorization, canonical
  private root, ordered identity-map digest, exclusive pre-mutation intent, and
  dedicated empty ledger; a same-path, alternate-path, or ambiguous create is
  never replayed.

Do not use a GPU-only Jobs API just to run this controller.  The qualified v2
rail is one exact amd64 CPU Kubernetes Job with Docker and durable private SFS
storage.  It binds the source, images, database, operation root, namespace,
queue, and PVC; proves two stable server previews and the top-level
`fleet.ai/failure-alerts: "off"` annotation; performs one create request; and
cleans up only the exact returned UIDs.  This narrow wrapper does not authorize
another campaign or a general-purpose cluster submitter.

## Source rules and scale-up sequence

### 1. Strong-teacher successes

The existing 57.38M-token corpus is the teacher anchor.  New teacher sessions
are admitted only when all of the following are bound in the source receipt:

- the source model is an explicitly approved stronger teacher with an immutable
  model identity;
- the session completed normally, authoritative grading verifies success, and
  infrastructure/session ingestion is complete;
- it belongs to a training family under the relevant immutable split;
- the exact OpenCode/tool/template treatment is known; and
- the transcript passes the task-rich visible-action policy below.

Collect additional teacher data only from newly qualified training families or
from a predeclared treatment comparison.  It is not acceptable to select
sessions because their outputs look especially elegant, short, or convenient.

### 2. Qwen self-successes

Start with all qualified training families, not merely the easiest cases.  A
wave supplies exactly four independently seeded attempts for every selected
training task version.  The next wave is justified only when the private
aggregate receipt shows one of the following:

- fewer than 20,000,000 distinct supervised visible-action target tokens have
  been admitted;
- a reviewed training stratum has no qualifying success coverage; or
- a separately named model, harness, or sampling treatment is being compared.

This is high throughput without outcome cherry-picking: task supply is balanced
before outcomes are known, then all qualified successes are admitted once.  A
hard family can yield no success and still be represented in collection
statistics.  A family that happens to yield many successes is not duplicated to
match a target token count.

The old self-success pool is useful for validation of the path but is too small
and narrow to anchor a production self-SFT result.  Do not repeat its examples
to manufacture a large corpus.

### Quantity and balance are separate corpus treatments

The primary corpus is **all qualified, unique successes**.  It maximizes real
evidence: no accepted session is repeated, upweighted, or removed because its
family is hard.  Equal attempt supply before outcomes prevents an easy task from
being run more often simply because it succeeds more often.

After that primary corpus exists, inspect only its aggregate concentration
metrics.  If one family provides more than 10% of target tokens or one
application provides more than 60%, create a second, explicitly named
family-balanced ablation.  It admits at most 100,000 distinct visible-action
target tokens per family, keeps the first deterministic digest-ordered sessions
needed to reach that cap, and never upsamples a smaller family.  This produces
a useful answer to a real question—whether a broad but uneven corpus transfers
worse than a less concentrated one—without relabelling the balanced subset as
more data.

The cap, digest-ordering rule, threshold, and source manifest digest belong in
the subset receipt.  The all-qualified corpus stays available as the default
quantity treatment.

### 3. Teacher/self mixtures

Do not mix the source pools during collection.  After each has an immutable
manifest, a mixture is a separately declared corpus treatment with:

- exact parent manifests and token counts;
- a deterministic source-weight rule;
- deduplication across parents before sampling; and
- the same held-out family exclusions as either parent.

The default first comparisons are teacher-only, self-only (only after the
20M-token gate), and then a predeclared mixture.  A mixture never changes a
source's task split or turns held-out success into training material.

## What makes a trajectory useful SFT data

The goal is to teach general black-box investigation and exploitation behavior,
not just the final report action.  Existing teacher-corpus rules already encode
that principle and should be applied equally to self-successes:

- require verified success and at least one non-`submit_report` tool response;
- require at least one completed non-submit tool round;
- require more assistant actions than report submissions; and
- cap report-submission assistant actions and target tokens at 50% of their
  respective totals.

Accepted rows contain visible assistant content and tool calls.  Tool results
and copied history are context, not prediction targets.  Every retained visible
assistant action is supervised once; no target action is silently truncated.
The existing dense builder also removes only an exactly redundant final answer
after a completed `submit_report`, rather than removing useful work that led to
the result.

`training/fleet_teacher_corpus.py` is the implementation reference for the
teacher lane.  The self lane should reuse its acceptance receipt shape, task-rich
coverage checks, and evidence binding, while recording `source_kind: self` and
the exact Qwen revision.  Export and normalisation remain in the existing
`training.cli` path; do not copy raw transcripts into Git, W&B, issue comments,
or public receipts.

## Reasoning and long context

### Written reasoning

The current collection lanes are **visible-action-only**.  They must not put
provider fields such as `thinking`, `reasoning`, or `reasoning_content` into
the SFT target.  The current policy deliberately treats these fields as private
or unknown, and the aggregate reasoning census has no approved source schema.

Visible reasoning may become a third, separate lane only after all of these
exist:

1. an explicit authorization that the reasoning is trainable and visible to the
   student at serving time;
2. an aggregate-only source inventory classifying every candidate as
   student-visible, private-or-unknown, or absent;
3. a frozen Qwen template and thinking-mode treatment that round-trips to the
   same tokenization at collection, training, and inference; and
4. tests proving that hidden reasoning is absent and no supervised span is cut
   off.

That lane must use a new corpus schema and a matched action-only control.  It
is not an `include_thinking` switch on either collection lane.

### Compaction

OpenCode can use native compaction during a live collection episode.  This lets
an agent keep working through long tool trajectories.  Offline action SFT is
different: it only trains on a static, exact prompt history.  A source episode
that contains opaque `context_compaction` is therefore rejected by the existing
builder, because the true later prompt cannot be reconstructed safely.

This is a quality gate, not an instruction to disable useful online compaction.
Record aggregate counts for:

- episodes with no compaction;
- episodes with opaque compaction and therefore rejected from offline SFT; and
- episodes with an exactly recorded continuation, should a future approved
  continuation-aware corpus contract exist.

Do not pretend that the original, uncompressed history was the prompt after a
summary.  A future continuation-aware lane must record the pre-summary prompt,
the summary's token identity, the post-summary prompt, and the later action's
actual prompt before it can admit such data.

## Corpus publication and duplicate prevention

After a wave finishes, export only the exact session set named by the sealed
collection receipt.  For an explicitly job-backed source, use the existing
private `training export --job-ids-file <explicit-list>` followed by
`training normalize`; for a generic campaign, use the small
collection-to-admission adapter above instead of assuming the job exporter owns
those attempts.  Then build one create-once corpus and a sanitized receipt.
The corpus process must reject rather than repair:

- a session outside the training split;
- a missing or mismatched model, harness, task, or verifier binding;
- incomplete grading/ingestion evidence;
- malformed tool/action structure;
- private-or-unknown reasoning; and
- opaque compaction where offline SFT cannot reconstruct the true prompt.

At corpus build time deduplicate, in this order:

1. exact source-session identity;
2. full normalized-trajectory digest; and
3. exact packed-window payload digest.

Count each unique supervised target token occurrence once.  Different window
sizes are alternative context treatments over the same target actions; they are
not extra independent data.  A parent-corpus subset or a teacher/self mixture
must verify all parent digests before it makes a new selection.

## Required private and sanitized metrics

Every wave and corpus receipt needs these exact aggregate measurements.  They
allow progress decisions without disclosing a prompt, tool output, flag,
credential, trace, token sequence, or per-session identity.

| Stage | Required measurements |
| --- | --- |
| Plan | plan digest; source kind; exact model/harness/template/tool digests; split digest; number of training families and task versions; planned cells; four-attempt seed family |
| Collection | pending/claimed/completed/review cells; valid-success count; infrastructure-invalid count; aggregate reasons for rejection; counts by application, environment, difficulty, and vulnerability-family stratum |
| Long-context quality | no-compaction, opaque-compaction, and exactly-recorded-continuation counts; input/output budget treatment digest |
| Corpus | candidate, admitted, and rejected-session counts by fixed reason code; unique trajectory and packed-window duplicates removed; source sessions; assistant actions; non-submit tool rounds; report-action/report-token share; distinct supervised target tokens |
| Balance | target-token share of the largest family, top 10% of families, and each application; whether the 10%/60% threshold required the deterministic capped ablation |
| Split protection | zero admitted development/final families and versions; family-closure check; counts of versions inheriting an old family assignment versus newly split families |
| Scale decision | whether self data is at least 20,000,000 distinct supervised visible-action target tokens; stratum coverage gaps; parent/child manifest digests |

The self-SFT readiness gate is exact: no split leak, no unresolved binding,
no opaque-compaction row admitted, all corpus deduplication checks pass, and at
least 20,000,000 distinct visible-action target tokens.  The teacher anchor
already meets the token threshold; new teacher data still must pass every other
gate.

## Operational checklist for an authorized operator

1. Build or refresh the reviewed task inventory without reading or publishing
   prompts/traces.  Inherit old family roles and create the new-family split.
2. Create one collection configuration per source kind.  Bind the exact source
   model revision, task split, OpenCode treatment, image digests, sampling seed
   family, four attempts, and `training_data_eligible: true`.
3. Run the existing `prepare` and `preflight` commands.  Stop if any exact
   binding, server profile, tool schema, image, or task runtime differs.
4. Initialise one fresh PostgreSQL ledger per wave.  Perform an exact duplicate
   census before the one designated creator starts workers.
5. Run enough independent CPU workers to consume the pre-authorized capacity.
   Workers only claim frozen pending cells; monitoring remains score-blind.
6. Reconcile terminals from receipts and authoritative grading, preserving
   ambiguous cells for review rather than retrying them automatically.
7. Run the collection-to-admission adapter for generic campaign records (or the
   existing explicit job exporter for job-backed records), then normalize and
   build a create-once, evidence-bound corpus using the existing dense policy.
8. Publish sanitized aggregate receipts and immutable manifest digests.  Keep
   source exports private.
9. Start another self wave only through the stated scale/coverage/treatment
   gate.  Use the immutable corpus manifests—not raw source directories—as the
   input to SFT experiment generation.

This gives Qwen collection a simple throughput loop: broad, balanced supply of
training-family attempts; strict evidence and action-quality admission; a clear
20M unique-token self-data target; and no hidden reasoning or held-out-task
leakage.
