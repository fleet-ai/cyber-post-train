# Deriving safe SFT source-coverage metadata

`python -m training.source_coverage --config <local-config.json>` derives the
episode metadata required by [the representative study selector](REPRESENTATIVE_STUDY_DATA.md).
It is local CPU preparation only: no network, Jobs API, GPU allocation, model
forward pass, new rollout, or held-out reference session.

The extractor reads private normalized records in memory. Its **outputs are
metadata-only**, not a second transcript or training dataset. It never emits
message content, tool arguments/results, prompts, private thinking, token IDs,
flag values, score values, credentials or library diagnostic text. Use a fresh
process and immutable input files for production preparation; synthetic tests do
not replace the exact-image/native-tokenizer qualification gate.

## Inputs

The JSON config has exactly these fields:

| Field | Meaning |
| --- | --- |
| `source` | `{path, sha256}` for the private normalized JSONL |
| `reviewed_evidence` | `{path, sha256}` for the reviewed metadata mapping below |
| `split` | Path to self-digested, **train-only** `cyber_task_split_v2` |
| `model_lock` | Exact model/tokenizer lock used by the corpus compiler |
| `tokenizer_root` | Existing local tokenizer directory; no download or remote code |
| `native_helper` | Exact pinned SkyRL helper source, verified by `dense.native_helper` |
| `max_length`, `context_tokens` | Exact integer corpus bounds, with no silent defaults |
| `output` | New create-once output directory |

Paths resolve relative to the config file. Source/evidence file digests are
mandatory. Tokenizer files are verified against the exact model lock through
`corpus.local_tokenizer`; helper, split, model lock, tokenizer and private source
files are checked for stability before metadata publication. A destination that
already exists is rejected, including a partial prior attempt. Output files use
mode `0600` inside a `0700` directory. A failure does not overwrite prior data.

### Reviewed acceptance/trace mapping

`cyber_sft_reviewed_source_evidence_v1` is self-digested with
`training.study_data.seal`. It contains only `schema`, `sha256`,
`origin: fleet_cyber`, `review_receipt_sha256`, and `episodes`.
Each episode entry contains exactly:

- `episode_id`, exact `task_key`, `task_version_id`, exact `model_id`;
- `source_kind: teacher|self`;
- `validity: valid`, `verified_success: true`;
- `acceptance_sha256`, `trace_sha256`, `normalized_record_sha256`.

The canonical normalized-record hash is `training.io.digest_json(record)` over
the **complete current record**. A historical hash computed before later fields
were added is not silently reused. Every mapped identity must be in the supplied
training split. No numeric score or trace payload belongs in the mapping.

The reviewer must obtain the acceptance/trace digests from exact authoritative
evidence and bind that work to the review receipt. The extractor does **not**
invent acceptance, infer success from an answer/report, or independently verify
the upstream grader. It checks the explicit mapping against exact private
record/model/task/version identity and the corpus compiler's existing
verified-success eligibility predicate. A disagreement, duplicate, missing
record or stale digest blocks publication.

Unmapped records and non-training records are skipped before accessing their
messages, outcome or model. JSONL parsing necessarily parses each input row, so
prefer a private input already restricted to reviewed train sessions; no held-out
record is tokenized, selected or emitted.

## One dense policy, one target count

The implementation directly calls the existing:

1. `corpus.local_tokenizer` and `dense.native_helper`;
2. `dense.compatible_messages`, preserving the proven tool surface and rejecting
   unsupported wrappers/compaction/incomplete tool transitions;
3. `dense.encode_record` and `dense.segment_record`;
4. `sft_runtime.dense_rows`, independently validating token bounds, positive mask
   counts, exact target spans and exhaustive eligible/excluded ordinal coverage.

There is no independent tokenization, overlapping-window unmasking or parallel
eligibility algorithm. Only the retained native assistant target spans contribute
to coverage. Copied context and tool observations contribute zero supervised
tokens; overlength targets contribute only explicit exclusion counts/hashes.
Each retained source reports its original, eligible and excluded assistant
inventory. A wholly incompatible source is listed by exact session/record hash
and fixed exclusion reason; it does not receive a fabricated certified episode.

`target_policy(...)` produces self-digested `cyber_sft_target_policy_v1`, binding
the exact tokenizer file/template/backend identity, native helper digest,
Transformers/tokenizers versions, `max_length`, `context_tokens`, and the exact
corpus/dense/runtime/coverage source files. Qualification must compare this digest
with the policy used to compile the final train corpus. Changing any bound
component requires new coverage evidence; an old receipt cannot justify a new
compiler policy. The function is available for the corpus qualification path to
compute the same identity.

## Categories are structural, not semantic

Categorization uses the already-normalized **retained assistant response**:

| Priority | Category |
| --- | --- |
| 1 | Any `submit_report` call → submission, including mixed submit/bash responses |
| 2 | Other supported tool calls → non-submission tool response |
| 3 | No tool call, nonempty visible content → decision response |
| 4 | Otherwise → other response, including an empty visible response after thinking is omitted |

The categories partition the retained assistant responses. Submission-token
count is the sum of their exact native positive-mask target spans, including
supervised terminators. A completed non-submission tool round means the original
response has all unique tool-call results in causal order, as proved by
`compatible_messages`. It does not mean that the shell command succeeded, an
exploit progressed, or the following assistant response survived the context
limit. Tool output is never itself a target.

## Outputs and integration

- `episodes.jsonl`: self-digested `cyber_sft_episode_metadata_v1` rows consumed by
  `study_data.select_sources`, including its exact required coverage counters.
- `coverage-receipts.jsonl`: per-source digest-bound counts, original/excluded
  inventory summaries and hashes of eligible targets/dense rows. No token values.
- `target-policy.json`: the immutable policy above.
- `manifest.json`: source/review/split/policy bindings, emitted/reviewed source
  counts, aggregate counters, explicit skips and excluded identities, file hashes.

Coverage `status: certified` means the token/mask/category counts were checked
against the bound dense policy and record; the success claim still comes from
the reviewed authoritative mapping. This does not certify model capability,
source semantic quality, or a complete representative task taxonomy.

Feed the emitted metadata and exact policy digest to
`study_data.select_sources`, with predeclared model filters, family episode/token
ceilings and seed. It then enforces structural quality and submission-dominance
gates before the selected unchanged private records enter `corpus.build`.
The extractor deliberately derives metadata for all explicitly reviewed sources
before those budget/diversity choices; it does not choose an HPO arm itself.

## Tests

`tests/test_source_coverage.py` exercises the real dense compiler and runtime
validator with a tiny synthetic tokenizer/helper. It checks retained-vs-raw
coverage, masked copied context, mixed-call classification, unsupported tools,
compaction, missing tool results, evidence/digest conflicts, held-out-field
noninspection, safe exception handling, create-once private files, stability and
policy drift. No real private record or model result is included in fixtures.
