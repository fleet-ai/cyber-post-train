# Qwen3.8 student-visible reasoning collection design, v1

## Status and boundary

This is a source-only design and private-materialization contract for a
possible future Qwen3.8 training-data lane. The checked-in
`data-fleet-visible-reasoning-materialize` command can validate sealed inputs
and construct a private token-and-mask corpus locally. It creates no rollout,
evaluation, training job, cluster object, or model endpoint, and it does not
authorize collection. A corpus request without the required authorization and
evidence files fails before private records are read.

The current lane remains **action-only**. It admits verified task successes from
training task families and trains visible assistant actions and tool calls. It
does not train written reasoning. The current action-only admission boundary is
documented in [Fleet collection admission for action SFT](FLEET_COLLECTION_TO_SFT_ADMISSION.md).

This design describes a different corpus type,
`cyber_student_visible_reasoning_sft_corpus_v1`. It is deliberately separate so
that a later change cannot silently add reasoning to an existing action-only
corpus or change the scientific claim of an existing run.

## Question the new lane could answer

The question is narrow: when Qwen is explicitly configured to write reasoning
that a Qwen user can see, does training on that written reasoning in addition to
the same successful actions improve performance on held-out task families?

It is **not** a way to copy a stronger model's private chain of thought. Private
teacher `thinking`, `reasoning`, or `reasoning_content` fields are prohibited,
even if they happen to be present in an export. Stronger teachers may continue
to supply the existing action-only corpus. They cannot supply reasoning to this
lane unless a future, separately reviewed contract proves that the exact text
was both student-visible and authorized for training; this v1 design does not
provide that exception.

## Source and authorization gates

The initial allowed source is a fresh Qwen3.8 rollout. Before its content can
be read for a corpus, a sealed, aggregate-only source inventory must bind all
of the following:

1. Exact Qwen repository and immutable revision, tokenizer file inventory,
   tokenizer-backend digest, and chat-template bytes digest. The offline
   builder loads those local tokenizer bytes itself; it does not trust a
   caller-supplied rendering helper.
2. One explicit request treatment: `enable_thinking: true` and
   `preserve_thinking: true`. A parser name, endpoint label, or model name
   alone is not proof of that treatment.
3. The request format, exact `bash` / `submit_report` tool schema digest,
   tool-call parser, harness version, context size, context reserve, and
   sampling treatment.
4. A synthetic, non-task template test which proves that the pinned template
   renders a declared student-visible reasoning segment, preserves its token
   boundaries through the serving request, and reproduces the same token IDs
   when the corpus renderer uses the template.
5. A source-authorization and safety receipt whose scope names this model,
   template treatment, task-family split, and corpus purpose.
6. Counts and digests only: candidate sessions, verified successes, selected
   sessions, available/selected visible-reasoning spans, and rejected private
   or unknown-reasoning records. The inventory must not print prompts,
   reasoning, tool results, flags, credentials, token IDs, or session IDs.

The source inventory uses one of three visibility values for every candidate:
`student_visible`, `private_or_unknown`, or `absent`. Only
`student_visible` may proceed. `private_or_unknown` is an exclusion, never a
hint to infer, summarize, rewrite, or expose hidden reasoning.

The existing aggregate census command intentionally authorizes no reasoning
data by itself. The separate schema, local-tokenizer boundary, masking
contract, and tests now live with the private materializer; future collection
code must emit those exact sealed artifacts. Changing a census allowlist by
itself remains invalid.

## Broad family-safe data split

Use the parameterized family splitter over the reviewed, valid task catalog
before collecting or packing windows. Its three roles are:

| Role | Permitted use |
| --- | --- |
| `train` | Collection and training data only. |
| `dev` | Development evaluation only; never a source of training rows. |
| `final_test` | Final evaluation only; never a source of training rows or tuning. |

Every version and session of a reviewed task family must have exactly one role.
Different families may share an application, but a family cannot appear in both
training and either held-out role. The sealed selection binds its exact catalog
digest, split digest, and protected-heldout-family digest before any source
content is materialized.

This is intended for the broad validated catalog, not the historical fixed
75-task study split. The action-only and reasoning treatments for one comparison
must use the same frozen broad family split.

## Exact student-visible serialization and loss mask

The future private materializer must normalize only the source's exact Qwen
conversation representation. It must not translate a provider-specific hidden
reasoning field into Qwen text or manufacture a summary.

For every accepted assistant turn, the source record must carry an immutable
token-boundary proof with two supervised span kinds:

| Span or message | Loss mask | Rule |
| --- | --- | --- |
| System, user, tool result, or copied earlier assistant context | `0` | Context may condition later actions but is never a target. |
| Declared student-visible Qwen reasoning | `1` | Only when the source inventory, template proof, and authorization receipt all match. |
| Visible assistant text and `bash`/`submit_report` calls | `1` | These remain supervised action targets, as in the action-only corpus. |
| Compaction summary or unknown provider field | `0` or reject | It is never silently reclassified as student-visible reasoning. |

More exactly:

- A target begins at the exact assistant token boundary emitted by the pinned
  Qwen template and ends at that source turn's exact stop boundary. Template
  control tokens are targets only if they were part of that observed assistant
  serialization.
- The reasoning and action spans are ordered, non-overlapping, and together
  account for every supervised token in the turn. Their source-token digests
  are retained privately for verification.
- The private source uses the explicit `student_visible_reasoning` field. The
  materializer maps only that authorized field to the pinned Qwen template's
  reasoning slot, independently re-renders an action-free copy of the same
  structured assistant turn, and derives the reasoning/action token boundary
  from those two renders. Caller-supplied labels and offsets are evidence to
  compare, never boundary authority.
- Every selected target span appears once across all packed windows. Copied
  context is always zero-masked. No target may be cut, padded into a different
  template, or silently truncated to fit a window.
- The action-only paired corpus is derived from the **same selected turns** by
  changing student-visible reasoning spans from `1` to `0`; its visible action
  spans stay exactly the same. This makes the intervention legible.

The materializer must fail closed if the template digest, explicit thinking
setting, locally rendered token IDs, or target boundaries differ between
collection, corpus construction, and serving.

## Native online compaction and exact continuation

Live collection may use the qualified
`opencode_1.18.27_native_compaction_autocontinue_v2` treatment: a 262,144-token
context with 20,000 reserved tokens of headroom. Native compaction is useful
online because it lets a long task continue. It does **not** automatically make
the resulting history valid offline training data.

For each compaction boundary, the private source record must prove all of these
exact identities:

1. The ordered post-plugin `selected.head` projection, with every supported
   user/assistant/tool part bound to exact prior normalized message indices.
2. The exact OpenCode serialization of that head and the native `buildPrompt`
   request derived from it and the last completed prior summary. The template
   system anchor is not SessionV1 history and must be excluded.
3. The synthetic request payload: one user text part, `system=[]`, `tools={}`,
   plus its exact Qwen rendered prompt-token identity. Any nonidentity message
   transform or compaction prompt/context plugin is rejected.
4. The serialized pre-compaction prompt token digest and length.
5. The generated continuation or summary token digest and length.
6. The serialized post-compaction prompt token digest and length.
7. The exact prompt-token digest used to generate the next assistant target,
   equal to item 6 at the boundary.

The offline renderer must use the actual post-compaction prompt, not replay the
old uncompressed history. The compaction text is zero-masked unless it is
separately proved to be a Qwen student-visible target span; this design does
not assume that native compaction summaries are trainable reasoning.

If any one of those identities is missing, the affected target and its later
continuation are rejected. If the exporter cannot make native compaction
reconstructable, the entire compacted episode remains useful only as online
operational evidence and is excluded from offline SFT. This is stricter than
the current action-only lane, which rejects opaque compaction outright.

## Required provenance and aggregate metrics

The future corpus manifest must bind these private inputs by digest:

- Qwen source/model/template/thinking-treatment proof;
- source authorization and safety receipts;
- reviewed task catalog, family split, and protected-family lock;
- collection plan, exact tool/harness treatment, and terminal success evidence;
- private selected-session list and normalized-record digests; and
- every continuation proof for a compacted selected target.

Its public or broadly shared receipt may report only aggregates. At minimum:

- candidate, verified-success, selected, and rejected session counts;
- rejection counts by fixed reason, including private/unknown reasoning,
  template mismatch, and non-reconstructable compaction;
- visible-reasoning target tokens, action target tokens, and total supervised
  tokens, each counted once, plus the packing-independent total and 20M gate
  for each emitted matched arm;
- complete reasoning-to-action turn count;
- duplicate counts at source-session, normalized-trajectory,
  source-assistant-target, and packed-window levels;
- per-family token distribution and maximum family share; and
- no-compaction, exactly-recorded-compaction, and rejected-opaque-compaction
  target counts.

No aggregate receipt may contain a prompt, reasoning text, tool result, flag,
credential, token ID sequence, or session identifier.

## Matched comparison with action-only SFT

The first comparison is paired, not a replacement for the action-only corpus.
Both arms use the same base checkpoint, tokenizer/template, task-family split,
selected sessions, visible action targets, window limits, optimizer recipe, and
held-out Fleet/WebExploitBench evaluation protocol. The only intended change is
whether the already-authorized student-visible reasoning spans receive loss.

Report reasoning and action target-token exposure separately. Do not call a
drop in imitation loss a capability improvement. The comparison advances only
after both arms complete their frozen evaluations on the same held-out family
set; external benchmarks remain evaluation-only and cannot feed this corpus.

## Implementation sequence and stop conditions

1. The Qwen-self reasoning-only schemas, local-tokenizer boundary, aggregate
   final-selection gate, and private materializer are implemented without
   altering the current action-only admission command; see [the materialization
   contract](FLEET_VISIBLE_REASONING_MATERIALIZATION.md). Materialization still
   emits `sft_ready: false`; no current SFT launcher accepts the resulting
   aggregate `source_only_qualified` handoff.
2. Synthetic tests cover template/thinking proof, source authorization,
   masking, exact continuation, split isolation, and private-teacher rejection.
3. Produce only the aggregate census and sealed handoff. If no authorized
   student-visible source is
   available, stop; action-only data collection remains the correct lane.
4. Review the schema and tests. Only a separately authorized operator may then
   create a collection campaign.

The first large source-only campaign candidate is documented in
[First large Qwen visible-reasoning collection plan](QWEN38_VISIBLE_REASONING_LARGE_COLLECTION_V1.md).
It covers all 50 currently admitted training families in balanced waves and
targets at least 20 million unique supervised tokens. It remains explicitly
unsubmitted and independently unreviewed; its presence does not satisfy step 4
or authorize collection.

The design stops rather than falling back to hidden teacher reasoning, opaque
compaction, a mixed family split, or an unproven Qwen thinking template.

## Teacher-visible rationale is a separate arm

This v1 implementation deliberately accepts **only Qwen self traces**. That is
not a judgment that teacher-visible reasoning cannot be useful; it is a refusal
to mistake a provider's hidden reasoning field for user-visible text.

The repository now has a source-only renderer and metadata-only admission
boundary for this arm.  They enforce the following as a new, separately
reviewed contract:

1. An immutable source artifact that identifies the teacher output version and
   proves the reasoning was delivered as ordinary visible conversation text to
   the Qwen student, rather than exposed through a private provider field.
2. An explicit authorization that permits that exact visible text to be used
   for training.
3. A distinct source-profile and collection-packet schema bound to the teacher
   artifact, Qwen target/template, task-family split, and success evidence.
4. The same local Qwen template round trip, span labels, family isolation,
   source/session/window deduplication, and compaction lineage required above.
5. A separate corpus identity and matched action-only comparison. It must not
   be mixed into the Qwen-self corpus or relabel existing teacher action traces.

Its implemented schema names are distinct:

- `cyber_teacher_visible_rationale_source_profile_v1`
- `cyber_teacher_visible_rationale_source_authorization_v1`
- `cyber_teacher_visible_rationale_packet_v1`
- `cyber_teacher_visible_rationale_attempt_metadata_v1`
- `cyber_teacher_visible_rationale_admission_request_v1`
- `cyber_teacher_visible_rationale_selection_v1`
- `cyber_teacher_visible_rationale_admission_receipt_v1`

The private token-level record and corpus names remain reserved for a future
materializer.  The implemented metadata boundary requires exact source fields
and rejects every unlisted field.  In particular, it cannot map a provider's
`thinking`, `reasoning`, `reasoning_content`, or `analysis` property into the
ordinary visible-text field.  See
[the teacher-visible rationale collection contract](TEACHER_VISIBLE_RATIONALE_COLLECTION.md)
for the exact requirements, commands, and remaining gates.

The source-only renderer and admission checker do not authorize an external
collection or create trainable data.  Until an issuer supplies the exact
teacher authorization, an approved collector writes sealed metadata, a private
Qwen token materializer reaches the 20M unique-token gate independently in
each emitted matched arm, and a separate
training permit is accepted, teacher runs can add only visible actions to the
established action-only lane.
