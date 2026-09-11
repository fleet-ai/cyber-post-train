# Qwen3.8 historical self-SFT interface audit — 2026-09-11

Status: **blocked before corpus compilation**. No training, evaluation, serving,
or cluster job was submitted by this audit.

## Metadata-only re-audit at 23:30 UTC

The additive, self-digesting
[`v2 receipt`](2026-09-11-self-sft-interface-audit-v2.json) rechecked every
currently eligible exact task version after this audit. The exact ledger has
87 clean Qwen3.8 successes on 43/89 task versions. Split A can use 63 successes
on 29/59 training versions, split B can use 58 on 28/59, and their 74-version
training union has 76 successes on 37 versions. All 87 are corroborated by the
session catalog. The catalog has 22 additional positive Qwen3.8 summaries, but
its summary route cannot bind their exact task version, model revision or
request interface, so they remain candidates only.

No exact ledger or catalog success was created after commit `e77a4445`. All 87
exact sources bind the same OpenCode 1.18.27 harness. The original 35-record
private audit proves that surface was namespaced and omitted the original
system prompt; the remaining records were deliberately not opened to invent
missing evidence. Therefore the native-compatible source count remains zero.

The nonlaunchable
[`direct recollection plan`](../../../configs/studies/qwen-blackbox-self-recollection-v1.json)
uses only the 74-version union of the representative A/B training splits and
keeps every held-out version excluded. It remains blocked on an immutable
direct-collector image, one exact base-route certificate, a frozen direct
system-prompt digest and synthetic recorder/compiler token parity. Shared and
dedicated endpoint sources are explicit separate treatments and may not be
silently pooled.

## Conclusion

The 35 previously prepared, verified-success Qwen3.8 records cannot be
faithfully translated into the current direct `bash`, `submit_report` dense-SFT
contract from the preserved evidence. The records are valid historical model
outcomes, but outcome validity does not prove training-interface parity.

The old records came from OpenCode. OpenCode namespaced the Fleet MCP server, so
the model-visible calls were primarily `fleet_bash` and
`fleet_submit_report`. The preserved records also omit the system prompt that
OpenCode placed in the actual model request. Renaming a target call changes its
Qwen token sequence, and inventing a system message changes its prefix. Neither
operation is byte-identical to the historical request or proven semantically
equivalent to the direct training interface.

The current compiler therefore behaves correctly by rejecting these records.
Do not reuse the older 175-window self-SFT Parquet as a workaround: it was built
under the older approximation of "original task plus visible actions" and its
manifest explicitly says the original system prompt was not reconstructed.

## Metadata-only evidence

This audit projected only schemas, role names, tool names, argument key/type
shapes, counts, and digests. It did not emit or inspect prompt text, assistant
or tool content, tool argument values, flags, answers, credentials, or model
scores.

- Private normalized source SHA-256:
  `cee5e6bf71989cba3affc4181422af95ada5d07aa57b263d231a5ea61c675e00`.
- Source-manifest self-digest:
  `41b1e1099de0ec92a0eb07d03ca6738fc017c0f00408771fc319faa6334abed3`.
- All 35 records start with `user, assistant`; none contains a `system`
  message. The source manifest records
  `original_system_prompt_reconstructed: false`.
- Across the 35 records, the normalized transcript has 4,261 tool calls:
  4,213 `fleet_bash`, 35 `fleet_submit_report`, two direct `bash`, and 11
  calls under other names. Eleven records contain at least one such other
  name.
- Twenty-two records use only the two OpenCode-namespaced Fleet tool names.
  However, only two records have argument key/type shapes that would satisfy
  the direct compiler after a namespace-only rename. Thirty-two records have at
  least one `fleet_bash` call whose arguments would additionally require fields
  to be deleted or renamed. The preserved evidence does not prove those edits
  are semantics-preserving.
- The source records do not bind the complete historical model-request system
  prompt or the complete model-visible OpenCode tool schemas. A current runtime
  schema cannot retroactively establish the historical one.
- Split A coverage receipt
  `acd68cad89861ad10f1ac5135ac95d1b06e2ccb806ea6873155f63e15f8c8a83`
  reviewed 20 sources and emitted zero. Split B receipt
  `aff9b06e862dcf31da7365ffbe9eba286f9e020cc803313d4e0527d4c874aeee`
  reviewed 19 and emitted zero. Every reviewed source was excluded as
  `unsupported_or_unproven_tool_interface` under target-policy digest
  `4ff62695661da24d100e4eccce93060f952e3241d713486e29fcb4a33de81fb1`.
  Their source-selection receipts remain blocked with
  `no_qualified_progress_bearing_sources`.

Repository code independently explains the mismatch:

- `evals/fleet/opencode_self_hosted.py` gives the remote MCP server the name
  `fleet`, permits `fleet_*`, and records OpenCode's model-visible tool name.
- The remote runtime itself is required to advertise ordered task tools
  `bash`, `submit_report`; the OpenCode namespace is an adapter-level model
  interface, not evidence that the target tokens are interchangeable.
- `training/dense.py` intentionally accepts only direct `bash`,
  `submit_report`, exact supported argument shapes, and an original
  `system,user` anchor. It does not unwrap or rename another harness's calls.

## Safe recollection plan

For the requested direct-interface self-SFT treatment, recollect successful
Qwen episodes rather than rewriting historical targets:

1. Freeze the exact Qwen revision, representative training-only task versions,
   environment/data/verifier bindings, direct harness, ordered tool schemas,
   budgets, and retry policy before execution.
2. Run the student through the direct `bash`, `submit_report` interface used by
   the intended trainer/evaluator. Do not route this treatment through an
   OpenCode MCP namespace and later strip the namespace.
3. Privately retain the complete canonical model-request prefix for every
   assistant target, including the actual system and user messages, complete
   ordered tool schemas, tool results, and any context transition. Public
   receipts should contain only their digests and structural counts.
4. Bind exact model revision, tokenizer/chat-template digest, harness image and
   code digest, task/runtime/verifier identities, tool-catalog digest,
   authoritative acceptance, raw trace digest, and normalized-record digest.
5. Exclude compaction unless the recorder preserves the exact post-compaction
   model request. Exclude malformed/unknown tool calls as targets; do not repair
   them after observing success.
6. Before corpus publication, render a fixed synthetic direct-tool exchange
   through both the recorder and dense compiler and require exact Qwen token-ID
   and loss-mask equality through the first observation. Then require every
   real record to satisfy the unchanged compiler and source-coverage gates.
7. Compile only representative-split training records, preserve held-out tasks,
   and run the normal bounded dev-cluster optimizer/checkpoint/reload canary
   before any production sweep.

If a future study instead intends to evaluate through OpenCode, define that as
a separate treatment and train on its actual `fleet_bash`,
`fleet_submit_report` surface with the exact captured OpenCode request prefix.
It must not be relabelled as direct-interface parity.
