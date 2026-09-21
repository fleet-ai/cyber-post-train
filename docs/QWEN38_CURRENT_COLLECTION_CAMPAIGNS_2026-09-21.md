# Qwen3.8 Fleet collection campaigns — 2026-09-21

## Outcome

The source-only base-Qwen action campaign is frozen and reproducible.  Its v1
packet is preserved as historical evidence but is not launchable because no
maintained authority can issue the global duplicate-census receipt it requires.
The immutable v2 successor preserves the same tasks, model, OpenCode treatment,
sampling, split, and 200 scientific cells while replacing only that unavailable
gate with a canonical create-once operation.  No Fleet job, environment, model
request, trace, or score was created or read while preparing either packet.

The production inputs are:

- source authorization/spec:
  `configs/collection/qwen38-base-current75-actions-pass4-v1.source.json`;
- materialized campaign:
  `configs/collection/qwen38-base-current75-actions-pass4-v1/`;
- create-once execution-safety successor:
  `configs/collection/qwen38-base-current75-actions-pass4-v2.source.json` and
  `configs/collection/qwen38-base-current75-actions-pass4-v2/`;
- stronger-teacher gate:
  `configs/collection/stronger-teacher-current75-actions-pass4-v1.requirements.json`;
- separate student-visible-reasoning gate:
  `configs/collection/qwen38-self-visible-reasoning-current75-pass4-v1.requirements.json`.

The v1 materializer is `training.current_collection_campaigns`; the v2
successor is `training.current_collection_campaigns_v2`.  They adapt the
existing sealed current-inventory and representative-split formats into the
generic collection renderer without changing a role, family, or exact task
version.  Its split is rooted at
`fleet-blackbox-current-study-20260914-v2` /
`sha256:48350b8fc23143abe297db3b2364590c72d564553ebb4e9a349fa06ec246a26b`
and emits the separate role-anchor and protected-family-lock artifacts required
by admission and private materialization.

Continuous growth uses `training.fleet_collection_roster`: a sealed qualified
subset of the 1,093-version supply produces a larger anchored inventory, exact
runtime bindings, child split, and protected-family lock while every historic
role stays fixed.  The current 75-task wave is the first frozen campaign, not a
ceiling and not permission to infer eligibility for the rest of the catalog.
The operational order is base-Qwen visible actions first, then the same sealed
train roster under an approved stronger-teacher action profile, then the
separately qualified Qwen student-visible-reasoning arm.  Every new sealed
roster regenerates its own exact packet and cell budget; it never enlarges an
already frozen packet in place.

## Exact supply and selection

The current production blackbox catalog contains 1,093 exact versions:

| Disjoint class | Exact versions | Collection disposition |
| --- | ---: | --- |
| Current, exact receipt, not known broken | 75 | admitted inventory |
| Current, exact receipt, currently broken | 5 | excluded |
| QA-cleared, awaiting exact receipt | 17 | qualification candidates only |
| Additional currently broken, no exact receipt | 47 | excluded |
| Not analyzed, no exact receipt | 949 | unqualified |

The admitted 75 contain 42 prior exact successes and 33 exact canonical
failures.  A canonical failure is still a valid task version for a fresh
collection attempt; it is not a successful training trajectory.  The immutable
family split is exactly 50 train / 17 dev / 8 final-test, with 75 distinct
application-plus-task-family groups and zero exact-version or reviewed-family
overlap.  Every one of the 25 dev/final families stays out of collection.
Because every present family is inherited, the anchored replay truthfully
records the narrow `all_current_groups_inherited` representativeness exception;
any child split containing a new family returns to the strict representative
coverage/repair gate.

The only tracked runtime catalog broad enough to supply exact environment/data
tuples has 89 rows.  The adapter takes the exact 75-identity intersection and
rejects its 14 extras.  It never treats membership in that older catalog as a
task-validity receipt.

## Frozen base-Qwen action wave

The wave uses all 50 admitted train families, one exact version per family and
four predeclared attempts per version: **200 cells**.  It binds:

- `Qwen/Qwen3.8-27B` at commit
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- Fleet session identity `qwen/qwen3.8-27b` and the exact recorded serving
  profile;
- OpenCode 1.18.27 and its release digest;
- exactly `bash` and `submit_report` and the reviewed tool-catalog digest;
- 262,144 context tokens, 20,000 compaction headroom, and
  `opencode_1.18.27_native_compaction_autocontinue_v2`;
- `chat_template_kwargs.enable_thinking=false` forced by the collection-only
  fixed proxy, an OpenCode model declaration with reasoning disabled, and an
  OpenCode command with its thinking-display flag absent;
- temperature 0.6, top-p 0.95, base seed 43, concurrency 8, no automatic
  retry, and `training_data_eligible: true`.

The packet admits only verifier-confirmed successful visible actions.  It
excludes private/unknown reasoning and opaque or otherwise unapproved offline
compaction.  Online 262K native-compaction metadata remains part of the exact
treatment; it does not make an opaque compacted trajectory valid offline.

Admission is capped at four sessions per exact task version/family and a 25%
maximum family target-token share.  Deduplication is ordered by exact source
session, normalized trajectory, then packed-window payload.

The v1 packet required a fresh census over every authoritative collection
ledger and Fleet session.  No maintained issuer can currently produce that
receipt from the public Fleet session interface, so v1 remains non-launchable.
It must not be weakened, relabeled, or reused as launch authority.

The v2 packet authorizes one new experiment with deterministic identities.  It
binds all 200 ordered cells three ways: the PostgreSQL ledger UUID, the
scientific cell SHA-256, and the execution SHA-256.  It also binds one canonical
private operation-directory name and one dedicated empty database identity.
Before initializing that database, the runtime writes an exclusive,
fsync-backed intent file.  If initialization returns an uncertain result, the
intent stays in place and neither the same path nor an alternate path may retry.
Historical sessions have different cell identities and remain separate
experiments; v2 never adopts or replays them.

The qualified v2 wrapper is exactly one amd64 Kubernetes `Job` at priority
class `c1`, with no GPU request or limit in any regular or init container.  The
root Job—not merely its Pod template—must carry
`fleet.ai/failure-alerts: "off"`.  It uses `completions: 1`, `parallelism: 1`,
`backoffLimit: 0`, `restartPolicy: Never`, and
`activeDeadlineSeconds: 768600`.  That deadline covers 25 waves of eight-way
concurrency at the eight-hour per-cell ceiling, bounded cleanup per wave, and
staging/preflight.  Before its only create call, the launcher must obtain two
identical normalized server previews and durably record a separate cluster
create intent.  An uncertain create is never repeated.  Terminal handling must
observe the exact Job name and UID, clean up that exact UID in the foreground,
and prove no owned child or idle allocation remains.

The historical runtime is `evals.fleet.visible_action_collection`; the v2
successor is `evals.fleet.visible_action_collection_v2`.  Both use the
standalone `collection_fixed_proxy.py` and import the unchanged historical
evaluator for task, route, ledger, and lifecycle checks.  Runtime file hashes
are sealed in every plan, avoiding changes to bytes pinned by earlier receipts.

## Stronger-teacher action wave

Do not relabel `q38-teacher-dense-v5-step186` as a stronger teacher.  It is a
Qwen student trained on stronger-teacher data, and its serving-readiness receipt
explicitly makes no capability claim.

The matching 50 × 4 teacher wave is therefore fail-closed.  The tracked corpus
proves historical action sources named `gpt-5.6-sol` and `grok-4.5`, but the
repository has no approved immutable collection profile for either.  The gate
lists the missing issuer-owned fields exactly: immutable model/provider revision
and session identity; complete serving route; exact non-thinking template;
source-use authorization issuer/time/scope/model binding; and a teacher-strength
issuer/time/model/comparison/capability receipt.

There is substantial genuine historical supply, but it is a mixed-source
aggregate and not a launch receipt.  The sealed manifest records **2,886**
verifier-success source sessions across **496** task keys and **1,176** exact
task versions, rechunked into **14,693** 32K windows with **176,654** visible
assistant responses and **57,384,881** supervised target tokens.  It excluded
25 held-out families across all versions.  Of those sessions, 645 are labeled
`gpt-5.6-sol` and 273 are labeled `grok-4.5` (**918** total); the remaining
1,968 came from eight other teacher labels.  The source manifest preserves a
262,144-token source maximum, while the tracked training artifact is a 32,768-
token rechunk, so it is not evidence that a new collection used the frozen
OpenCode 262K native-compaction treatment.

Historical GPT/Grok successes may enter the private materializer only when each
record supplies the exact packet/campaign, model alias and identity, template,
tool/harness, authoritative-success, ingestion, reasoning-visibility, and
compaction evidence required by the metadata admission boundary.  Aggregate
manifest counts alone do not supply those per-record bindings.  Preparing this
requirements packet read only that aggregate manifest; it did not open a raw
trace, session transcript, or score.

## Separate student-visible-reasoning wave

Student-visible Qwen reasoning is a third arm, never an option on the action
corpus.  The requirements packet binds the known base revision, Qwen template,
OpenCode binary, tools, 50/17/8 family boundary, and 200-cell ceiling, but it is
not launchable.  It reserves distinct source-profile, packet, selection,
record, and corpus schemas under the
`cyber_qwen_opencode_student_visible_reasoning_*_v1` namespace (with output
`cyber_qwen_opencode_visible_reasoning_sft_corpus_v1`); none may reuse the
visible-action packet.  The future packet retains the rooted family boundary,
20M-token minimum, 25% family-token cap, session/trajectory/window dedupe, and
explicit rejection of private/unknown reasoning, opaque compaction, unknown
serialization, and held-out families.  It remains blocked until all of these
receipts exist:

1. an explicit reasoning-corpus schema allowlist entry and source authorization;
2. a reasoning-specific immutable model/harness profile declaring
   `enable_thinking=true`, exact `preserve_thinking`, and student visibility;
3. exact template serialization/token round-trip evidence;
4. continuation and reload token-equivalence evidence, including the actual
   post-summary prompt after compaction;
5. zero-heldout-admission and source/session/trajectory/window dedupe tests.
6. a pinned immutable success-evidence mapping that binds source-session,
   normalized record/trajectory, exact task/version/family, and completed
   verifier success, followed by an aggregate reasoning census and selection
   receipt before training.

Every serialization fixture must carry the prompt token IDs produced locally
from `messages_without_target` with `add_generation_prompt=true`, plus the full
collection, training, and serving token IDs.  The prompt IDs must match that
local template call exactly and be an exact prefix of every full serialization;
this is the explicit continuation boundary, not an inferred text boundary.

Private or unknown teacher chain of thought is never eligible, and no missing
reasoning may be inferred, summarized, or reconstructed.

## Parallel task-quality qualification wave

The 1,013 current production blackbox versions without an exact execution
receipt are a qualification backlog, not training inventory.  Their source-only
gate is
`configs/qualification/fleet-blackbox-unproven-task-quality-wave-v1.requirements.json`.
It separates 17 promising QA rows (7 clean and 10 agent-failure) for the first
bounded roster, 949 not-analyzed rows for later batches, and 47 rows already
classified broken that remain excluded pending repair.

The first qualification wave contains the 17 QA-prioritized candidates in one
batch (7 `clean`, 10 `agent_failure`).  The 949 `not_analyzed` candidates then
fit exactly into 15 bounded waves: 14 × 64 plus a final 53.  Thus 966 is the
non-broken, unproven candidate upper bound, and 1,041 is the current 75 plus
that candidate pool.  Neither number is an admitted-task count: lineage review,
runtime binding, family assignment, final-family exclusion, execution evidence,
and the one-version-per-family cap can only reduce it.

No qualification roster is launchable from the current files.  The production
inventory has exact task/version and QA status but no reviewed application/task
family or runtime tuple.  Deriving a family from a task-key string would be an
unsafe inference.  Each selected successor roster must first bind reviewed
lineage, exact environment/data/verifier/starting-data identities, and prove
zero intersection with the eight immutable final-test family group IDs.  It is
capped at one version per family and 64 versions per bounded batch.

Qualification checks environment startup, exact `bash`/`submit_report` tool
reachability, verifier completion, finite authoritative outcome recording,
ingestion, and cleanup.  Either a genuine success or genuine model failure can
prove task health; neither is a capability claim or automatic training
admission.  Public evidence is aggregate-only, contains no task/session IDs,
private content, or numeric scores, and has `training_data_eligible: false`.
Any Kubernetes implementation must prove the top-level failed-job alert
annotation before create under the same fail-closed rule as collection.

After each qualification snapshot is sealed, the roster tool preserves every
historic role and assigns only reviewed new families.  Only then does the
renderer compute `admitted_train_task_versions × 4` and freeze a new base-Qwen
visible-action packet over every admitted train family.  A stronger-teacher
packet is generated separately only after its exact source profile is approved;
the reasoning packet remains a third contract.  This formula scales collection
without guessing how many of the 966 candidates will qualify or be assigned to
train rather than dev/final.

## Offline verification and next commands

Reproduce the committed files without any external call:

```sh
uv run --locked python -m training.current_collection_campaigns \
  --spec configs/collection/qwen38-base-current75-actions-pass4-v1.source.json \
  --root . \
  --output configs/collection/qwen38-base-current75-actions-pass4-v1 \
  --check

uv run --locked python -m training.current_collection_campaigns_v2 \
  --spec configs/collection/qwen38-base-current75-actions-pass4-v2.source.json \
  --root . \
  --output configs/collection/qwen38-base-current75-actions-pass4-v2 \
  --check
```

After qualification produces a sealed supply/qualified/root-anchor request,
grow the admitted roster without changing historic roles:

```sh
uv run --locked cyber-post-train data-fleet-roster <reviewed-roster-request.json>
```

That command is metadata-only.  The new child split must include at least one
new qualified family, pass strict representative coverage, and preserve the
same trusted root ID/digest before it can feed another campaign packet.

The v1 preparation below is retained only for historical reproduction.  It
cannot advance without an issuer that does not currently exist.  Do not use it
for a new launch.

Prepare the v2 canonical private evaluator directory locally (still no Fleet
call and no database mutation):

```sh
uv run --locked python -m evals.fleet.visible_action_collection_v2 prepare \
  configs/collection/qwen38-base-current75-actions-pass4-v2/eval-config.json \
  configs/collection/qwen38-base-current75-actions-pass4-v2/operation-authorization.json \
  --private-root <approved-private-parent>
```

The command exclusively creates the operation-root name already recorded in the
authorization.  A pre-existing path is a hard stop.  `preflight` performs
read-only Fleet metadata and image checks with zero model calls.  For the direct
CLI, `init` requires `ROLLOUT_DATABASE_URL` to name the already-provisioned,
dedicated empty database; it writes the no-retry intent first and initializes
the bound ledger there.  The qualified Job entrypoint instead writes that same
intent, creates its exact dedicated database, and initializes the ledger as one
ordered operation.  `run` creates environments and model traffic.  The Job
launcher is a separate one-way create operation and must satisfy the complete
checked-in packet before running these commands:

```sh
uv run --locked python -m evals.fleet.visible_action_collection_v2 preflight <wave-root>
uv run --locked python -m evals.fleet.visible_action_collection_v2 init <wave-root>
uv run --locked python -m evals.fleet.visible_action_collection_v2 run \
  <wave-root> base <unique-worker-id> --limit <bounded-count>
```

Actual base execution is blocked on fresh Fleet-team account/task/runtime,
serving-profile, image, OpenCode-startup, and non-thinking-template preflight;
a new canonical operation root and dedicated empty ledger; the reviewed
create-once Job launcher; and explicit operator launch authority.  Teacher and
student-visible-reasoning execution have the additional source/gate blockers
above.  Task-quality qualification is blocked on the reviewed lineage/runtime
roster and aggregate receipt rail just described; it must never borrow the
training campaign's eligibility bit.
