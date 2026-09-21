# Private Qwen student-visible reasoning materialization

`data-fleet-visible-reasoning-materialize` is a separate private, CPU-only
corpus builder. It does not alter `data-fleet-materialize`, which remains the
visible-action-only lane. It never contacts Fleet, starts a workload, or prints
source text, token IDs, prompts, tool output, flags, or session IDs. Its command
output is aggregate evidence only; its create-once output is private token IDs
and loss masks, never raw text.

This is not approval to collect a campaign. An operator still needs an
authorized collection and sealed successful-session selection.

## Safety contract

The action-only materializer rejects assistant prose because it cannot know
whether prose is student-visible or private provider reasoning. This separate
materializer has no `include_thinking` switch. It accepts a source only when a
sealed profile proves all of the following:

- exact `Qwen/Qwen3.8-27B` target revision, tokenizer, and chat-template;
- OpenCode 1.18.27, the approved `bash`/`submit_report` tool surface, 262K
  context, and 20K headroom;
- `enable_thinking: true` and `preserve_thinking: true` for a Qwen self source;
- an authorization receipt that calls the reasoning student-visible; and
- a digest-pinned serialization adapter whose re-rendered token IDs, assistant
  start boundary, and labelled target spans equal the collected evidence.

`qwen_self` is the initial source. `teacher_visible` requires two additional
receipts: one for student-visible status and one for training authorization.
This never authorizes generic teacher `thinking`, `reasoning`,
`reasoning_content`, `analysis`, or any unknown field. Those fields cannot be
represented in the private record schema and fail closed.

Before the builder opens private records, it verifies the sealed packet,
catalog, anchored family split, source-derived September root anchor,
protected-heldout lock, runtime bindings, and success selection. A generic
re-split, replacement root, held-out family, missing grading evidence, or
per-task cap violation stops before source content is read.

## Targets and compaction

Each accepted window must include both a `student_visible_reasoning` target
span and a `visible_action` target span. All earlier context, tool results, and
compaction summaries are loss-masked. Spans are hashed, ordered,
non-overlapping, and deduplicated alongside source sessions, trajectories, and
packed-window payloads.

Opaque compaction is forbidden. Qwen self records may use only
`student_generated_exact_continuation_v1`, with the parent window, prior
history, actual continuation token IDs/digest/length, summary-message digest,
pre/post prompt digests and lengths, and the next true target prompt. The post
prompt must equal the prompt used for the next target. Teacher-visible records
cannot use this exception in v1.

The result has schema `cyber_qwen_opencode_visible_reasoning_sft_corpus_v1`.
It is SFT-ready only after it reaches the packet's immutable target (at least
20M unique supervised tokens) and no task family contributes more than 25%.
Smaller outputs are preserved as `collection_pending_target`, not training
corpora.

## Collection handoff schemas

Future collection code must emit distinct immutable artifacts, not modify an
action packet:

- `cyber_qwen_opencode_student_visible_reasoning_source_profile_v1`
- `cyber_qwen_opencode_student_visible_reasoning_packet_v1`
- `cyber_qwen_opencode_student_visible_reasoning_selection_v1`
- `cyber_qwen_opencode_student_visible_reasoning_record_v1`

The packet must carry `root_role_anchor_id`
`fleet-blackbox-current-study-20260914-v2` and a non-null
`family_role_anchor_sha256`. This keeps a reasoning arm scientifically distinct
while allowing a later action-only paired arm to use the same selected turns
with only reasoning loss masked.
