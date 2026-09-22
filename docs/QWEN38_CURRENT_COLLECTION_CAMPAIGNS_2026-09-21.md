# Qwen3.8 Fleet collection campaigns — 2026-09-21

## Outcome

The base-Qwen action campaign now has a frozen, launchable v2 successor.  Its
scientific identity is the same 50-train-family pass@4 wave as v1, while its
execution identity is a new exactly-once operation and an exact amd64 CPU Job.
No Fleet job, environment, model request, trace, or score was created or read
while preparing either source packet.

The production inputs are:

- source authorization/spec:
  `configs/collection/qwen38-base-current75-actions-pass4-v1.source.json`;
- materialized campaign:
  `configs/collection/qwen38-base-current75-actions-pass4-v1/`;
- launchable exactly-once successor:
  `configs/collection/qwen38-base-current75-actions-pass4-v2/` and
  `configs/collection/qwen38-base-current75-actions-pass4-v2.source.json`;
- stronger-teacher gate:
  `configs/collection/stronger-teacher-current75-actions-pass4-v1.requirements.json`;
- separate student-visible-reasoning gate:
  `configs/collection/qwen38-self-visible-reasoning-current75-pass4-v1.requirements.json`;
- separate teacher-visible-rationale gate:
  `configs/collection/stronger-teacher-visible-rationale-current75-v1.requirements.json`.

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
train roster under an approved stronger-teacher action profile, followed by
separately qualified Qwen-self reasoning and teacher-visible-rationale arms.
The latter two are different sources and may never share an identity.  Every new sealed
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

The immutable v1 artifact retains its unavailable global-history census
requirement and remains non-launchable historical evidence.  It was not
rewritten.  V2 instead proves the first uniquely named campaign at the boundary
we control: one authorization binds the ordered 200-cell ledger/scientific/
execution identity map, canonical operation root, dedicated empty ledger, and
no same-path, alternate-path, or ambiguous replay.  The exact authorization is
`sha256:1597d141b236a9f8fea16e92bf0b4a4f9b16045677931bf72bb5ca26618e4684`;
the v2 plan is
`sha256:fb0d54ecca6bb5a8942016dbf339e7b51c902ec39458eb610b4d1eae2585154f`.

The supported v2 controller is
`evals.fleet.visible_action_collection_job`.  It packages the exact clean
merged source and inputs into an immutable ConfigMap and renders one root
`Job`: amd64 CPU, priority class `c1`, zero GPU requests and limits,
`backoffLimit: 0`, `restartPolicy: Never`, and a derived 768,600-second active
deadline for 25 concurrency-8 waves.  The root Job itself carries
`fleet.ai/failure-alerts: "off"`.  The launcher requires two byte-stable server
previews, writes an exclusive local intent, sends one create request, and never
retries an ambiguous response.  Terminal cleanup reopens the returned exact
Job and ConfigMap UIDs, performs UID-preconditioned foreground deletion, and
proves owned Pod and Workload absence without reading logs, traces, or scores.

Inside that Job, `evals.fleet.visible_action_collection_job_entry` verifies the
exact agent archive/build receipt and proxy image, exclusively creates the
authorized SFS root, performs the Fleet-team/route/task preflight, writes the
scientific create intent, then creates the dedicated database and bound ledger.
Only after all those checks does it execute the 200 cells.  V2 remains
non-thinking visible-action collection; it does not widen the separate
student-visible-reasoning schema.

### Completion-budget v3 successor

The first v2 live prefix exposed one deterministic runtime defect without
opening prompts, traces, logs, or scores: OpenCode discovery (`/v1/models`) and
chat completions shared `FIXED_MAX_REQUESTS=600`. The sanitized attempt census
showed the same exit-1/process-error lifecycle across every reviewed failure:
599 completed model rounds followed by the same scoring-stage runtime
classification. V2 and its packet remain immutable evidence; no digest is
rewritten.

V3 binds `collection_fixed_proxy_v2.py`, which retains a bounded total-request
ceiling but counts only `/v1/chat/completions` against the exact 600-completion
budget. On request 601 it writes one content-free, create-once exhaustion
receipt before returning 429. `collection_completion_budget.py` reclassifies a
nonzero OpenCode exit as `output_limit` only when that exact receipt proves 600
completed chats and the rejected 601st chat; altered, missing, discovery-only,
or premature receipts remain `process_error`. Output-limit attempts are never
accepted as training trajectories.

The checked-in bounded gate is one admitted train family × pass@4 (4 cells):
`configs/collection/qwen38-base-current75-actions-pass4-v3-canary`. Only after
that Job has four valid accepted lifecycles may the separate 50-family × pass@4
packet at `configs/collection/qwen38-base-current75-actions-pass4-v3` be
launched. Both remain non-thinking visible-action campaigns; neither may be
mixed with student-visible reasoning or teacher-visible rationale.

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

## Separate teacher-visible-rationale wave

The teacher-visible-rationale arm now has a source-only renderer and a
metadata-only admission checker. It is still not a launcher or a training
input. The exact contract is
`configs/collection/stronger-teacher-visible-rationale-current75-v1.requirements.json`
and is explained in
[`TEACHER_VISIBLE_RATIONALE_COLLECTION.md`](TEACHER_VISIBLE_RATIONALE_COLLECTION.md).

This arm asks an exactly authorized stronger teacher for one to four concise,
ordinary visible sentences before each tool call. It rejects provider-private
thinking fields and never reconstructs missing reasoning. It binds OpenCode
1.18.27, the pinned Qwen3.8 tokenizer and chat template, visible ordinary
assistant content with exact OpenCode tool calls, `enable_thinking=false`, and
262K native online compaction. A compacted success is eligible only when each
visible summary and the actual next prompt are bound reproducibly.

The initial packet keeps the same 50-train / 17-dev / 8-final anchored roles
and 200-cell ceiling, while the eventual private corpus must contain at least
20M unique supervised tokens, cover at least 40 and 80% of selected families,
and respect the 25% per-family token ceiling. The source authorization, actual
collector, private token materializer, final unique-token receipt, and training
permit remain separate required gates. This identity cannot be mixed with the
action-only teacher arm or the Qwen-self visible-reasoning arm.

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

uv run --locked python -m training.current_collection_campaigns_v3 \
  --spec configs/collection/qwen38-base-current75-actions-pass4-v3.source.json \
  --root . \
  --output configs/collection/qwen38-base-current75-actions-pass4-v3 \
  --check

uv run --locked python -m training.collection_budget_canary \
  --spec configs/collection/qwen38-base-current75-actions-pass4-v3-canary.source.json \
  --root . \
  --output configs/collection/qwen38-base-current75-actions-pass4-v3-canary \
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

The v1 preparation path is retained only for historical reproduction.  It
cannot advance without an issuer that does not currently exist and must not be
used for a new launch.

After the v2 launcher PR is merged, use a fresh clean checkout of that exact
main commit.  Package the merged source and frozen inputs into a new private
directory (still no Fleet or Kubernetes call):

```sh
uv run --locked python -m evals.fleet.visible_action_collection_job prepare \
  --repo-root . \
  --config configs/collection/qwen38-base-current75-actions-pass4-v2/eval-config.json \
  --task-selection \
    configs/collection/qwen38-base-current75-actions-pass4-v2/task-selection.json \
  --authorization \
    configs/collection/qwen38-base-current75-actions-pass4-v2/operation-authorization.json \
  --collection-packet \
    configs/collection/qwen38-base-current75-actions-pass4-v2/collection-packet.json \
  --output <new-private-launch-packet-directory> \
  --expected-source-commit <exact-merged-main-sha>
```

For the completion-budget repair, substitute the four files from
`qwen38-base-current75-actions-pass4-v3-canary` in that same prepare command.
The launcher derives a distinct four-cell operation root/database and still
requires a clean exact merged commit, two byte-identical server previews,
absence checks, one create intent/call, root alert annotation `off`, c1/amd64,
and zero GPUs. Do not prepare or launch the 200-cell v3 packet until the canary
has four accepted cells and exact-UID cleanup evidence.

Validate locally, then launch once.  `launch` performs the two server previews
and exact-name absence checks before its single mutation.  The Job's internal
preflight is Fleet metadata-only and makes zero model calls; database/output
creation and model traffic occur only afterward:

```sh
uv run --locked python -m evals.fleet.visible_action_collection_job validate \
  <new-private-launch-packet-directory>/launch-packet.json

uv run --locked python -m evals.fleet.visible_action_collection_job launch \
  <new-private-launch-packet-directory>/launch-packet.json \
  --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 \
  --journal <new-private-launch-packet-directory>/KUBERNETES_CREATE.jsonl
```

The returned Job and ConfigMap UIDs are the only cleanup authority.  After an
exact terminal observation, inspect only aggregate Kubernetes state with:

```sh
uv run --locked python -m evals.fleet.visible_action_collection_job status \
  <new-private-launch-packet-directory>/launch-packet.json \
  --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 \
  --journal <new-private-launch-packet-directory>/KUBERNETES_CREATE.jsonl
```

That command reads no logs, traces, scores, or credentials.  Once it reports
`terminal_ready_for_cleanup`, release the exact bound resources with:

```sh
uv run --locked python -m evals.fleet.visible_action_collection_job cleanup \
  <new-private-launch-packet-directory>/launch-packet.json \
  --context nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6 \
  --journal <new-private-launch-packet-directory>/KUBERNETES_CREATE.jsonl \
  --receipt <new-private-launch-packet-directory>/KUBERNETES_CLEANUP.json
```

The Kubernetes create journal can never authorize a second create.  Cleanup
uses a separate sealed intent and is intentionally resumable only for the same
returned UIDs: after a lost delete response it re-reads the exact name/UID and
may reissue the UID-preconditioned foreground delete, while a reused name with
a different UID hard-fails.  An uncertain create response remains a
reconciliation event, never permission to submit again.  Teacher and
student-visible-reasoning execution retain their separate source/gate blockers.
Task-quality qualification still requires reviewed lineage/runtime evidence and
must never borrow the training campaign's eligibility bit.
