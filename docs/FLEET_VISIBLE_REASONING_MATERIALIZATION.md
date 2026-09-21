# Private, separate reasoning-data materialization

`data-fleet-visible-reasoning-materialize` is a separate private, CPU-only
corpus builder. It does not alter `data-fleet-materialize`, which remains the
visible-action-only lane. It never contacts Fleet, starts a workload, or prints
source text, token IDs, prompts, tool output, flags, or session IDs. Its command
output is aggregate evidence only; its create-once output is private token IDs
and loss masks, never raw text.

This is not approval to collect a campaign. An operator still needs an
authorized collection and sealed successful-session selection.

## Three deliberately separate data tracks

| Track | What can be supervised | Current state | What is always rejected |
| --- | --- | --- | --- |
| Teacher/action-only | Successful teacher actions and tool calls | Existing action-only builder | Teacher hidden reasoning, unknown prose, opaque compaction |
| Qwen self/visible reasoning | Qwen reasoning explicitly shown to the Qwen user, plus the same visible actions | The builder in this document, source-only and not yet connected to a training launcher | Any generic teacher reasoning field, private/unknown reasoning, opaque compaction |
| Teacher/visible rationale | Teacher rationale supplied as ordinary, explicitly authorized text that a Qwen user could see | Design only; no code path accepts it yet | Provider-private thinking, inferred/reconstructed rationale, and all records until this separate contract is reviewed |

These are different scientific treatments. They use the same family-safe task
roles when they are compared, but their records, source proof, corpus identity,
and training permits never mix.

## Safety contract

The action-only materializer rejects assistant prose because it cannot know
whether prose is student-visible or private provider reasoning. This separate
materializer has no `include_thinking` switch. It accepts a source only when a
sealed profile proves all of the following:

- exact `Qwen/Qwen3.8-27B` target revision, tokenizer, and chat-template;
- OpenCode 1.18.27, the approved `bash`/`submit_report` tool surface, 262K
  context, and 20K headroom;
- `enable_thinking: true` and `preserve_thinking: true` for a Qwen self source;
- an independently materialized immutable authorization artifact that binds the
  reasoning treatment and calls it student-visible; and
- the exact local, digest-pinned Qwen tokenizer and template whose re-rendered
  token IDs, assistant start boundary, and labelled target spans equal the
  collected evidence.

`qwen_self` is the **only** v1 source. The implementation rejects a generic
teacher source rather than guessing whether teacher text was public reasoning
or provider-private state. A later teacher-visible-rationale arm needs its own
source profile, packet, evidence authority, corpus identity, and review. It
must prove that the exact text was visible to the Qwen student and licensed for
training; an ordinary `thinking`, `reasoning`, `reasoning_content`, `analysis`,
or unknown field can never satisfy that proof.

Before the builder opens private records, it verifies the sealed packet,
catalog, anchored family split, source-derived September root anchor,
protected-heldout lock, runtime bindings, and success selection. A generic
re-split, replacement root, held-out family, missing grading evidence, or
per-task cap violation stops before source content is read.

The builder is intentionally offline. It verifies that each locally materialized
authorization and success-evidence document is self-digesting and agrees with
the immutable Artifact Registry locator and payload digest it names. It cannot
ask the Registry whether that locator is real. The collection authority must
therefore materialize those files directly from the Registry before calling this
tool; this source-only builder never treats a locator string alone as a live
authentication check, and no training launcher accepts its output today.

## Targets and compaction

Each accepted window must include both a `student_visible_reasoning` target
span and a `visible_action` target span. All earlier context, tool results, and
compaction summaries are loss-masked. Spans are hashed, ordered,
non-overlapping, and deduplicated alongside source sessions, trajectories, and
packed-window payloads.

Opaque compaction is forbidden. Qwen self records may use only
`student_generated_exact_continuation_v1`, with the parent window, prior
history, actual continuation token IDs/digest/length, the exact prompt used to
generate its assistant summary, summary-message digest, pre/post prompt digests
and lengths, and the next true target prompt. The builder re-renders the
summary boundary with the pinned Qwen template; a self-consistent hash of
caller-provided continuation IDs is not enough. The post prompt must equal the
prompt used for the next target. Teacher-visible-rationale records cannot use
this exception in v1.

The token-only result has schema
`cyber_qwen_opencode_visible_reasoning_sft_corpus_v1`. Materialization itself
is **never** a training permit: its receipt records `sft_ready: false` even at
the target. The separate `data-fleet-visible-reasoning-authorize` command checks
the aggregate source census, corpus manifest, and coverage receipt together.
It writes `source_only_qualified`, not a collection or training approval, after
at least 20M unique supervised tokens and the 25% maximum task-family share are
met. No current SFT launcher accepts that qualification, so this remains a
source-only admission boundary rather than authorization to train.

## Collection handoff schemas

Future Qwen-self collection code must emit distinct immutable artifacts, not
modify an action packet:

- `cyber_qwen_opencode_student_visible_reasoning_source_profile_v1`
- `cyber_qwen_opencode_student_visible_reasoning_packet_v1`
- `cyber_qwen_opencode_student_visible_reasoning_success_evidence_v1`
- `cyber_qwen_opencode_student_visible_reasoning_census_v1`
- `cyber_qwen_opencode_student_visible_reasoning_selection_v1`
- `cyber_qwen_opencode_student_visible_reasoning_record_v1`
- `cyber_qwen_opencode_visible_reasoning_training_selection_v1`

The packet must carry `root_role_anchor_id`
`fleet-blackbox-current-study-20260914-v2` and a non-null
`family_role_anchor_sha256`. This keeps a reasoning arm scientifically distinct
while allowing a later action-only paired arm to use the same selected turns
with only reasoning loss masked.

The action-only schemas and materializer remain unchanged. A record with no
explicit Qwen-visible reasoning belongs in that action-only lane, not here.

The planned teacher-visible-rationale arm will use a different set of schema
names, ending in `teacher_visible_rationale_*`. It cannot reuse any schema
listed above or rename a teacher `thinking` field. Its separate gate is defined
in [the collection design](QWEN38_STUDENT_VISIBLE_REASONING_COLLECTION_DESIGN_V1.md#teacher-visible-rationale-is-a-separate-future-arm).
