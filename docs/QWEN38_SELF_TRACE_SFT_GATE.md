# Qwen3.8 self-trace SFT gate

Status: **blocked before data creation on three external bindings**. No
self-trace collection or SFT job is ready to preview or submit, and this
preparation performed no cluster, API, W&B, or registry mutation.

## Compatibility decision

The 87 exact, positive Qwen3.8 baseline sessions are valid evaluation outcomes,
but they are not a valid corpus for the direct-interface self-SFT arm. They cover
43 of the 89 eligible task versions and all use the OpenCode 1.18.27 harness.
The durable evidence does not bind their complete original model-request prefix,
and the audited historical subset used namespaced tool names rather than the
direct `bash`, `submit_report` contract. Reconstructing or renaming those records
would silently train a different interface.

The binding evidence is
[`2026-09-11-self-sft-interface-audit-v2.json`](evidence/qwen38-study/2026-09-11-self-sft-interface-audit-v2.json),
whose self-digest is
`sha256:b478d515e08ce371101d27208e49ac212f03388135e2c131eabbbc8c5b75fa2f`.
It admits zero native-compatible historical sessions. The existing split-A and
split-B self-coverage artifacts independently contain zero episodes. The old
175-window historical corpus remains prohibited.

## Offline recorder-to-dense parity

The deterministic synthetic-only recorder-to-dense gate is closed by
[`2026-09-12-self-trace-recorder-dense-parity-v1.json`](evidence/qwen38-study/2026-09-12-self-trace-recorder-dense-parity-v1.json).
Its file SHA-256 is
`sha256:43dd6942e6f5b62c175e7b7ec06c088dd319270601b9b8ecaf971551b5966e44`
and its self-digest is
`sha256:4903ed1580cb8ebc060d859ef746ea12af4d24399e5d2cbc44ba28b771396418`.
The upstream study plan and downstream collection request both bind that
receipt; the request additionally binds the exact nine-file local recorder,
episode, parser, hashing, Fleet-helper, and dense-adapter code closure.

The fixture exercises the SkyRL recorder adapter, direct two-tool episode path,
token-ID and loss-mask preservation across the complete two-turn synthetic
sequence, including the first masked observation, under the requested dense
policy (`max_length=16384`, `context_tokens=4096`). The fixture digest is
derived from the recorded/reference hashes, counts, observation boundary, and
dense policy rather than from a free-form label. It is deliberately not a
model/runtime certificate: it loads no Qwen weights, real chat-template bytes,
pinned native-helper file, collector image, base route, task prompt, trace,
flag, answer, score, or credential. Those limits are machine-checked in the
receipt so the synthetic gate cannot be mistaken for live qualification.

Three external blockers remain explicit in the sealed request:

- an immutable collector image digest;
- the exact fresh-base route certificate digest;
- the direct system-prompt digest obtained from its authoritative source.

The read-only follow-up
[`2026-09-12-self-trace-external-binding-audit-v1.json`](evidence/qwen38-study/2026-09-12-self-trace-external-binding-audit-v1.json)
checked the current candidate evidence without rewriting that immutable request.
Its self-digest is
`sha256:a9617d4643d5e722a889ba79114145c49ba0a9bd7a2ba130668392903acb6d1f`.
It found useful components, but none can truthfully fill one of the three
request fields yet:

- The exact OpenCode build context and a reproducible local OCI manifest exist,
  but a local manifest is not a pullable registry identity. Theseus
  [PR #31835](https://github.com/fleet-ai/theseus/pull/31835) was open and green
  at the observation time; the live dev image-build OpenAPI still lacked
  `rewrite_timestamp`, and neither merge/deployment nor the dedicated ECR/IAM
  binding was proven. A green PR is therefore not a collector-image receipt.
- The shared Qwen route component proves the exact model payload, tokenizer,
  chat template, serving runtime and historical readiness. Its own schema says
  `passed_component_only`, `launchable=false`. It does not bind the eventual
  pull-qualified collector, collector-visible request prefix, or a fresh
  collection-time route observation, so its digest must not be relabelled as
  the direct-route certificate.
- A tracked prompt artifact has stable bytes and a known digest, but its
  provenance is one GPT teacher-session variant and a proposed Qwen3.6
  diagnostic choice. That evidence explicitly does not establish the
  Qwen3.8 self-recollection treatment. Selecting it may be reasonable, but it
  is a scientific treatment decision, not a metadata recovery operation.

The remaining path is consequently precise: deploy and qualify the immutable
collector image; explicitly freeze the Qwen3.8 direct prompt source; then issue
one new self-digesting direct-route certificate from those bindings and a fresh
read-only route observation. Only a new versioned collection request may bind
that certificate. The sealed v1 request remains a fail-closed historical input.

This offline closure does not add a live collector/index command:
`training.self_trace_collection` remains a validator and private-review
library. Resolving the three external bindings therefore does not itself
authorize execution. A separately reviewed create-once orchestration must call
the prepared-attempt and source-review boundaries and bind every qualified
source receipt into the downstream index before any collection can start.

## Qualified data contract

Fresh collection must start from exact Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` through the SkyRL direct recorder.
Each source must have a digest-valid accepted receipt, positive authoritative
reward, confirmed environment release, exact task/version/model/template
bindings, the complete bounded episode limits and scoring/provisioning routes,
and the ordered direct tools plus their exact catalog digest. Instance IDs must
be canonical DNS-safe opaque identifiers; evidence-run, verifier and
verifier-execution identities must be canonical nonzero UUIDs. The sealed
binding retains the rendered-prefix digest; the recorded prompt-token digest
and raw task-prompt digest are independently recomputed at admission.
Compaction is disabled. The recorded token IDs and native loss mask are
authoritative; text is never retokenized or rewritten.

[`training/self_trace_corpus.py`](../training/self_trace_corpus.py) compiles only
sources on the 59-task split-A training projection. It rejects dev/final tasks,
source drift, missing or mismatched prompt-token evidence, invalid native token
records, non-positive outcomes, resource leaks, unsupported tools, and traces
that retain no non-submission action. Every fitting assistant response is
supervised exactly once, while copied assistant context and tool observations
remain masked. Deterministic per-task episode/token caps and 50% maximum terminal
submission response/token shares prevent a few tasks or the final submission
step from dominating the corpus.

The resulting manifest selects `task_outcomes_only`: the trainer reports W&B
training loss, optimizer step, learning rate, gradient norm/finite state,
supervised-token counts and throughput, but never loads a held-out teacher-CE
artifact. Model selection is by separately launched Fleet held-out outcomes.

## Materialization and dev-canary gate

The inert templates are:

- `configs/qualification/qwen38-self-trace-index-a-v1.template.json`
- `configs/qualification/qwen38-self-trace-corpus-a-v1.template.json`
- `configs/qualification/qwen38-self-trace-sft-a-dev-v1.template.json`

They deliberately contain `PENDING` values and cannot be submitted. After a
bounded fresh-base collection succeeds, independently review and seal the source
index. Copy the corpus template outside Git, replace only the exact source index,
image, receipt, and create-once output bindings, then run:

```sh
uv run python -m training.self_trace_corpus --config <private-corpus-config.json>
uv run cyber-post-train train <materialized-self-sft-config.json> \
  --output <new-private-prepared-run-directory>
uv run cyber-post-train preflight <new-private-prepared-run-directory>
uv run cyber-post-train preview <new-private-prepared-run-directory> --cluster dev
```

The first SFT canary remains one node, eight GPUs, priority `c1`, W&B enabled,
outcome-only validation, checkpoints every two steps, and a planned pause after
step 6. It is admissible only when the frozen corpus yields more than six
optimizer steps. The next paid action is therefore **not an SFT submit**: it is
an explicitly reviewed, bounded direct-collection canary after the collector
image, exact base route, and direct system-prompt digest are certified. Only
after local compilation, zero-GPU preflight, dev preview, and an explicit
duplicate/capacity check may an operator run
`cyber-post-train submit ... --cluster dev`.
