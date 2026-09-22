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
| Teacher/visible rationale | Teacher rationale supplied as ordinary, explicitly authorized text that a Qwen user could see | Separate source-profile/packet renderer and metadata-only admission contract; no collection or training launcher | Provider-private thinking, inferred/reconstructed rationale, and any record without exact source, verifier, Qwen-serialization, family, and compaction evidence |

These are different scientific treatments. They use the same family-safe task
roles when they are compared, but their records, source proof, corpus identity,
and training permits never mix.

## Safety contract

The action-only materializer rejects assistant prose because it cannot know
whether prose is student-visible or private provider reasoning. This separate
materializer has no `include_thinking` switch. It accepts a source only when a
sealed profile proves all of the following:

- exact `Qwen/Qwen3.8-27B` target revision, tokenizer, and chat-template;
- OpenCode 1.18.27, the approved `bash`/`submit_report` MCP surface rendered as
  the exact ordered `fleet_bash`/`fleet_submit_report` OpenCode wire schemas,
  262K context, and 20K headroom;
- `enable_thinking: true` and `preserve_thinking: true` for a Qwen self source;
- an independently materialized immutable authorization artifact that binds the
  reasoning treatment and calls it student-visible; and
- the exact local, digest-pinned Qwen tokenizer and template whose re-rendered
token IDs, assistant start boundary, and labelled target spans equal the
collected evidence. Every ordinary Qwen render receives those digest-bound
tool definitions; a missing, reordered, or schema-different catalog is rejected
before private records are opened.

Private records keep authorized reasoning in the distinct
`student_visible_reasoning` field. Raw provider fields such as
`reasoning_content`, `thinking`, or `analysis` are still rejected. Only after
that check does the builder map the authorized field into Qwen's pinned
`reasoning_content` template slot. It then independently renders the same
assistant turn without its visible action text or tool calls. The full and
action-free renders must yield one unambiguous token boundary; the caller's two
span labels, offsets, and token digests must equal that template-derived result.
Swapped labels or a self-consistent resegmentation are rejected.

`qwen_self` is the **only** v1 source. The implementation rejects a generic
teacher source rather than guessing whether teacher text was public reasoning
or provider-private state. A later teacher-visible-rationale arm needs its own
source profile, packet, evidence authority, corpus identity, and review. It
must prove that the exact text was visible to the Qwen student and licensed for
training; an ordinary `thinking`, `reasoning`, `reasoning_content`, `analysis`,
or unknown field can never satisfy that proof.

Before the builder opens private records, it verifies the sealed campaign plan,
wave plan, immutable operation authorization, collection packet, exact train
task selection, catalog, anchored family split, source-derived September root
anchor, protected-heldout lock, runtime bindings, verifier-success evidence,
and private selection. Every success and private record must name the exact
campaign cell, wave, attempt, and seed. The builder independently reconstructs
all 20,000 planned cell identities from the exact 50 training task versions and
rejects any different wave or cell universe. A generic re-split, replacement
root, held-out family, missing grading evidence, or replayed cell stops before
source content is read.

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
non-overlapping, and deduplicated alongside source sessions and trajectories.
The unique-token count additionally keys every assistant target to the exact
trajectory, target message, and supervised spans using offsets relative to the
assistant boundary. Window ids, sequence numbers, prompt length, and other
packing metadata cannot make one source target count twice. Packed-window
payloads remain a separate integrity check.

Opaque compaction is forbidden. Qwen self records may use only
`student_generated_exact_continuation_v1`, with the parent window, prior
history, actual continuation token IDs/digest/length, the exact prompt used to
generate its assistant summary, summary-message digest, pre/post prompt digests
and lengths, and the next true target prompt. The builder re-renders the
summary boundary with the pinned Qwen template; a self-consistent hash of
caller-provided continuation IDs is not enough. OpenCode 1.18.27 generates that
summary from a separate synthetic user request, not from the ordinary session
prefix. The pinned release resolves to commit
`b04697366f05419e9bd7a92f841813dd976161c9`; the active CLI compaction source
and imported core prompt source are digest-bound separately so the core
package's different serializer cannot be substituted. Each boundary therefore
captures the ordered, post-plugin
`selected.head` projection, binds every projected part back to exact prior
message indices, and independently reserializes the supported
reasoning→text→completed-tool subset. The template system anchor is excluded;
tool results are embedded beside their calls. Nonidentity message transforms,
plugin prompt/context changes, attachments, pruned/error tool states, and any
other native part order are rejected. The validator derives the exact
OpenCode `buildPrompt` text using the last completed prior summary, then
requires the captured request `messages`, `system=[]`, `tools={}`, payload
digest, and rendered prompt token IDs to match it byte for byte. The post prompt must
equal the prompt used for the next target, and every later training window must
retain an unbroken ancestry through the declared post-compaction root until a
new declared boundary replaces it. Pre-boundary parents and unrelated later
windows are rejected. Teacher-visible-rationale records cannot use this
exception in v1.

The token-only result has schema
`cyber_qwen_opencode_visible_reasoning_sft_corpus_v1`. It contains two paired
Parquet corpora: one supervises student-visible reasoning plus visible actions;
the other uses the exact same source records, packed windows, and token IDs but
masks the reasoning targets. Each arm has its own digest-sealed manifest, and a
cross-arm digest binds both manifests to the same ordered window identities.
It also binds the ordered packing-independent source-target identities, so the
reasoning and action-only arms cannot silently select different assistant turns.
Materialization itself
is **never** a training permit: its receipt records `sft_ready: false` even at
the target. The separate `data-fleet-visible-reasoning-authorize` command checks
the aggregate source census, corpus manifest, and coverage receipt together.
It writes `source_only_qualified`, not a collection or training approval, only
after **each** emitted arm independently has at least 20M packing-independent
unique supervised tokens, verified successes come from at least 20 training
families, and the 25% maximum task-family share is met. The reasoning-plus-action
count cannot compensate for an underfilled matched action-only arm. No current
SFT launcher accepts that qualification, so this remains a
source-only admission boundary rather than authorization to train.

## Collection handoff schemas

Future Qwen-self collection code must emit distinct immutable artifacts, not
modify an action packet:

- `cyber_qwen_opencode_student_visible_reasoning_source_profile_v1`
- `cyber_qwen_opencode_student_visible_reasoning_source_authorization_v1`
- `cyber_qwen_opencode_student_visible_reasoning_operation_authorization_v1`
- `cyber_qwen_opencode_student_visible_reasoning_packet_v1`
- `cyber_qwen_opencode_student_visible_reasoning_success_evidence_v1`
- `cyber_qwen_opencode_student_visible_reasoning_census_v1`
- `cyber_qwen_opencode_student_visible_reasoning_selection_v1`
- `cyber_qwen_opencode_student_visible_reasoning_record_v1`
- `cyber_qwen_opencode_visible_reasoning_sft_arm_manifest_v1`
- `cyber_qwen_opencode_visible_reasoning_training_selection_v1`

The packet must carry `root_role_anchor_id`
`fleet-blackbox-current-study-20260914-v2` and a non-null
`family_role_anchor_sha256`. This keeps a reasoning arm scientifically distinct
and requires the materializer to emit the matched action-only arm from the same
selected turns with only reasoning loss masked.

The action-only schemas and materializer remain unchanged. A record with no
explicit Qwen-visible reasoning belongs in that action-only lane, not here.

The teacher-visible-rationale arm uses a different set of schema names ending
in `teacher_visible_rationale_*`. It cannot reuse any schema listed above or
rename a teacher `thinking` field. Its source-only renderer and metadata
admission boundary are documented in
[the teacher-visible-rationale collection contract](TEACHER_VISIBLE_RATIONALE_COLLECTION.md).
It still has no private token materializer or training permit.
