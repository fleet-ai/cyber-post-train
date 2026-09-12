# Qwen3.8 self-trace SFT gate

Status: **blocked before data creation**. No self-trace SFT job is ready to
preview or submit, and this preparation performed no cluster mutation.

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
image and exact base route are certified. Only after local compilation,
zero-GPU preflight, dev preview, and an explicit duplicate/capacity check may an
operator run `cyber-post-train submit ... --cluster dev`.
