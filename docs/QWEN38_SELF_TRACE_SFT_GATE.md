# Qwen3.8 self-trace SFT gate

Status: **v2 is blocked before data creation on two external artifact
bindings**. The conservative direct-interface prompt policy is resolved; the
immutable direct collector qualification and collection-time base-route
certificate do not yet exist. No self-trace collection or SFT job is ready to
preview or submit, and this preparation performed no cluster, API, W&B, or
registry mutation. The sealed v1 request remains unchanged and historically
blocked on its original three fields.

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

## Conservative v2 prompt policy

The scientifically conservative treatment is the repository-defined direct
default: one `user` message containing the exact bound task prompt and **no
explicit system message**. This is not a hash of an empty prompt. The message is
absent. `training/rl_data.py` constructs this user-only prefix for normal
`skyrl_direct` data, and `training/rl_episode.py` uses the same user-only shape
when no override is supplied. Adding the tracked GPT-derived anti-refusal system
prompt would introduce a new treatment rather than recover a missing Qwen3.8
identity.

The decision is sealed in
[`qwen-blackbox-self-recollection-v2.json`](../configs/studies/qwen-blackbox-self-recollection-v2.json)
(self-digest
`sha256:79da6a71d7fc077cbc67048d32125408def99601763c98a73a38718f1a6e3d2a`)
and
[`qwen38-self-trace-collection-request-v2.json`](../configs/qualification/qwen38-self-trace-collection-request-v2.json)
(self-digest
`sha256:2c3040c28f6fc474908841e3582fb47cb9a69a5ce6993f2f9bc33ad702c96c22`).
It preserves the v1 roster, model, limits, sampling, and task/runtime bindings
by immutable reference. It pins:

- Qwen3.8-27B revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and chat-template SHA-256
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`;
- ordered direct tools `bash`, `submit_report` and canonical catalog digest
  `sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a`;
- both exact template invocations: the string render uses
  `tokenize=false, add_generation_prompt=true`; the token render uses
  `tokenize=true, return_dict=false, add_generation_prompt=true`. The optional
  Qwen knobs `enable_thinking`, `reasoning_effort`, and `preserve_thinking` are
  deliberately omitted in both calls, preserving the pinned template's
  defaults rather than inventing another intervention.

The template may emit its own template-defined system/tool framing even though
the input message list has no system role. Those emitted bytes are controlled by
the pinned template and invocation identities. For every attempt, the raw task
prompt SHA-256 must match the frozen live task binding, the UTF-8 rendered-prefix
SHA-256 must be stored as `initial_prompt_sha256`, and the canonical token-ID
sequence SHA-256 must be stored independently as
`initial_prompt_tokens_sha256`. Re-encoding the rendered bytes must equal those
token IDs, and the recorder prefill must equal the same token IDs. Text
rewriting or later retokenization is prohibited.

Consequently, no true human scientific choice remains for the baseline
self-trace arm. A future explicit-system-message arm is possible, but it must be
a separately named treatment; the GPT-derived candidate is not silently reused.

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

The equivalent user-only v2 fixture is
[`2026-09-12-self-trace-recorder-dense-parity-v2.json`](evidence/qwen38-study/2026-09-12-self-trace-recorder-dense-parity-v2.json).
Its file SHA-256 is
`sha256:1a31a533ee1c12d8e8cb88ccbf81f73281835ca3a1234e9dc664165854a314b2`
and its self-digest is
`sha256:9943b7e8e01199be1e582b8ffcc5a729c3caa7b41e6addeee72ba0e853f764dd`.
It changes the required conversation anchor from `system,user` to `user` and
binds the v2 prompt-policy, template-invocation, and rendered-request contract
digests. It remains synthetic and makes no collector-image or live-route claim.

The three external blockers below describe the frozen v1 request:

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
  The later immutable
  [live-gate recheck](evidence/qwen38-study/2026-09-12-opencode-agent-image-live-gate-v1.json)
  reached the same conclusion after binding the newer deployed API image and
  current PR/main state; no context upload or build was attempted.
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

V2 supersedes only that unresolved prompt treatment. It does not relabel either
runtime component as qualified. V2 requires full immutable artifact references
(`path`, file SHA-256, and self-digest), not a bare image regex match or a bare
route-certificate hash:

- The **collector qualification artifact** must bind a pullable linux/amd64
  `repository@sha256` and matching runtime image ID to the exact v2 source
  closure, SkyRL native helper, model revision, template, prompt policy,
  template invocations, rendered-byte/token contract, and ordered tool catalog.
  It must record a successful local synthetic check and clean-pull zero-GPU dev
  Pod with `imagePullPolicy: Always`, zero restarts, matching image ID, deletion,
  and confirmed absence. It must not load target model weights, make task or
  scoring requests, or publish an image during qualification.
- The **direct-route certificate artifact** must cross-bind that qualification's
  self-digest and exact collector image. It must bind the exact model lock,
  rehashed weights/tokenizer/template, immutable serving runtime identity,
  normalized arguments, ready route identity, and the same v2 interface. A
  fresh synthetic collector-to-route probe must prove the native
  `/inference/v1/generate` prompt-token-ID plus sampling-parameter request and
  response contract, including response token IDs and logprobs, without
  prompt/tool rewriting, target task content, or scoring/task requests.

No such artifacts are checked in. Null bindings remain fail-closed, so tests use
synthetic temporary receipts only to verify the schema and cross-binding logic.
Theseus [PR #31835](https://github.com/fleet-ai/theseus/pull/31835) concerns an
OpenCode evaluation image. Even after merge and deployment, that image is not a
SkyRL-direct collector qualification and cannot fill the v2 collector slot.

The remaining path is consequently precise: build/publish and qualify the exact
SkyRL-direct collector, then issue one self-digesting direct-route certificate
from that qualification and a fresh route observation/probe. Populate those two
artifact references only in a successor of the checked-in blocked v2 request;
never rewrite v1 or claim the null-bearing v2 request is ready.

This offline closure does not add a live collector/index command:
`training.self_trace_collection_v2` remains a validator and private
preparation/review library. Resolving the two external artifacts therefore does
not itself authorize execution. A separately reviewed create-once orchestration
must call the prepared-attempt and source-review boundaries and bind every
qualified source receipt into the downstream index before any collection can
start.

## Qualified data contract

Fresh collection must start from exact Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` through the SkyRL direct recorder.
Each source must have a digest-valid accepted receipt, positive authoritative
reward, confirmed environment release, exact task/version/model/template
bindings, the complete bounded episode limits and scoring/provisioning routes,
and the ordered direct tools plus their exact catalog digest. The first recorded
message must be the sole user task prompt and no explicit system message may
appear. Instance IDs must be canonical DNS-safe opaque identifiers; evidence-run,
verifier, and verifier-execution identities must be canonical nonzero UUIDs.
The sealed
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
qualification and exact base-route certificate are bound in a sealed ready v2
successor. Only
after local compilation, zero-GPU preflight, dev preview, and an explicit
duplicate/capacity check may an operator run
`cyber-post-train submit ... --cluster dev`.
