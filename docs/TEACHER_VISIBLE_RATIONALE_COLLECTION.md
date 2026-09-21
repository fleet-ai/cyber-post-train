# Stronger-teacher visible-rationale collection

## What this lane is for

This is a separate data treatment for a simple question: does Qwen learn more
useful cyber behavior when a stronger teacher writes a short, user-visible
explanation before each action, compared with learning only the teacher's
actions?

It is not provider chain-of-thought collection. The teacher is explicitly
asked to write one to four ordinary, visible sentences before a tool call:
what it learned, what it will test next, and why. The exact instruction is
[`visible-rationale-instruction.txt`](../configs/collection/stronger-teacher-visible-rationale-current75-v1/visible-rationale-instruction.txt).
Fields named `thinking`, `reasoning`, `reasoning_content`, `analysis`, or any
other private-reasoning field are rejected. Missing explanations are not
reconstructed or inferred.

This lane remains separate from:

- the current stronger-teacher **action-only** corpus; and
- the Qwen-self **student-visible reasoning** corpus.

Their packets, source authorizations, selections, records, and eventual corpus
identities cannot be reused or mixed.

## Current implementation boundary

The checked-in requirements file and two CPU-only commands implement the
source contract and metadata admission boundary:

```sh
uv run --locked cyber-post-train data-fleet-teacher-visible-rationale-render \
  configs/collection/stronger-teacher-visible-rationale-current75-v1.requirements.json \
  /private/path/to/exact-source-authorization.json \
  /private/output/teacher-visible-rationale-contract

uv run --locked cyber-post-train data-fleet-teacher-visible-rationale-admit \
  /private/path/to/admission-request.json
```

Both commands are offline. They do not contact Fleet, call a model, launch a
job, read source text, or authorize external submission. The renderer will not
produce a packet until an immutable source authorization proves the exact
teacher deployment, route, training-use permission, teacher-strength receipt,
and ordinary visible-output surface. The current repository does not contain
that issuer-owned artifact, so no collection is launchable from the committed
files alone.

Admission consumes sealed metadata only. Its private selection carries opaque
record and task identities needed by the next local step. Its public receipt
contains counts and digests only. It always says `sft_ready: false`; the next
gate is a private Qwen tokenizer/template round trip, exact target-window
deduplication, and final token coverage verification.

## Frozen initial comparison

The initial source-only requirements bind the same anchored 50-train / 17-dev /
8-final family roles as the current action collection. All 25 held-out families
remain excluded. The first packet is sized as 50 training task versions times
four attempts, or 200 planned cells. This is a first wave, not a statement that
200 attempts will yield enough successful data.

The eventual corpus gate is at least **20 million unique supervised token
occurrences** after private deduplication, with no task family contributing more
than 25%. It also requires successful data from at least 40 families and at
least 80% of the packet's selected families. These are candidate breadth gates;
the private materializer still has to prove the final unique-token count and
packed-window deduplication. If the first wave falls short, a later packet can use a larger
qualified anchored roster or more predeclared attempts. It must be a new packet;
the frozen packet is never enlarged in place.

The clean experiment is paired:

1. collect one sealed set of successful teacher trajectories;
2. render the same selected turns with rationale and action targets; and
3. render the matched action-only arm by masking the already-visible rationale
   spans, without changing selected sessions or actions.

This makes rationale supervision the intended difference. Training loss is a
health signal; held-out Fleet outcomes select the treatment and matched
OpenCode WebExploitBench results confirm it.

## Exact source and success proof

The teacher source authorization must be an immutable Artifact Registry object
that binds:

- provider, model name, immutable provider/deployment revision, session model,
  and route-profile digest;
- a separate teacher-strength receipt;
- the exact visible-instruction file digest;
- permission to use that ordinary visible text to train the exact pinned
  Qwen3.8-27B target; and
- proof that provider-private reasoning was not ingested.

An admitted record must bind one predeclared campaign cell, the exact source
profile and packet, an authoritative verifier execution and receipt, and a
positive completed score. It must also bind an exact local-Qwen serialization
round-trip receipt. Admission recomputes the profile and packet from the exact
requirements and immutable source authorization rather than trusting their
self-declared digests alone. A success label without the verifier identities, or a
transcript with no ordinary visible rationale, is excluded.

The serialization contract uses ordinary Qwen assistant content plus the exact
OpenCode `bash` and `submit_report` tool-call representation. It pins
`enable_thinking=false`: the rationale is visible assistant text, not a hidden
thinking channel. For every fixture, locally rendered prompt tokens from
`messages_without_target` with `add_generation_prompt=true` must be the exact
prefix of collection, training, and serving serializations, and those three
full serializations must match. This metadata admission step binds the
round-trip receipt; the private materializer must repeat the token-level check.

Records are deduplicated in this order: record identity, source session,
normalized trajectory, and then—at the private materializer—packed-window
target occurrence. Versions in development or final-test families can never
enter the selection.

## Context and compaction

Collection uses OpenCode 1.18.27 with a 262,144-token context window, 20,000
tokens of reserved headroom, and native compaction/automatic continuation.
That lets a long cyber attempt continue, but it does not make an opaque summary
valid training context.

For every selected compacted trajectory, metadata must bind each boundary:

- the prompt before compaction;
- the prompt used to generate the summary;
- the exact ordinary, student-visible summary message;
- the summary re-rendered through the pinned Qwen tokenizer/template;
- the prompt after compaction; and
- the exact prompt used for the next supervised target.

The last two prompt digests must be equal. The visible summary is retained as
zero-loss context; rationale and actions after it are trained from the prompt
that was actually used. Opaque summaries and summaries containing private
provider reasoning are rejected.

## What remains before collection or training

1. An authorized issuer must publish the exact teacher source authorization
   and teacher-strength receipt.
2. A collection launcher must consume the rendered packet without changing its
   task, prompt, model, OpenCode, context, or retry bindings. Any Kubernetes
   wrapper still needs the mandatory root `fleet.ai/failure-alerts: "off"`
   annotation before creation.
3. A private exporter must emit the metadata schema and keep raw text private.
4. The teacher-specific Qwen token materializer must re-render ordinary visible
   text locally, prove all compaction boundaries, deduplicate target windows,
   and confirm the final 20M-token and family-concentration gates.
5. A separate reviewed training permit must bind that corpus. Neither rendering
   nor admission is permission to train.
