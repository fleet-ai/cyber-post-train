# First Qwen3.8-27B LoRA configuration set

Status: **trainer image and leak-free V2 one-step corpus qualified and bound;
CPU preflight pending; GPU canary not launched** (2026-09-20)

This set freezes the first exact-model SFT gate, the first literature-informed
SFT anchor, and the two matched evaluation contracts. The exact one-step
template now binds the independently qualified source/image pair recorded in
[`qwen38-lora-megatron-trainer-image-2026-09-20-v1.json`](../configs/qualification/qwen38-lora-megatron-trainer-image-2026-09-20-v1.json).
No training or evaluation was launched by that qualification. The one-step run
still requires the normal exact-image CPU preflight, authenticated preview,
duplicate, output-root and resource checks. The production anchor and both
evaluation contracts remain fail-closed. The earlier Fresh75 corpus cannot be
used for this study as-is: two of its 37 task families are in the later frozen
ten-family final test set. A CPU-only, create-once filter has now published and
independently verified the leak-free V2 successor. Its exact public manifest is
[`qwen38-fresh75-teacher-sft-final-lock-free-v2.manifest.json`](../configs/data/qwen38-fresh75-teacher-sft-final-lock-free-v2.manifest.json),
and its sanitized qualification record is
[`qwen38-lora-one-step-corpus-2026-09-20-v1.json`](../configs/qualification/qwen38-lora-one-step-corpus-2026-09-20-v1.json).
The one-step template now points to that exact V2 manifest and staged root. The
exact runtime gate binds its payload-derived dataset, split and corpus digests,
866-row full-epoch ceiling, and planned pause after optimizer step 1. The
exact-image CPU preflight, authenticated preview and real training acceptance
remain pending. Corpus qualification and binding are not training acceptance.

## Training templates

| Purpose | File | Frozen treatment |
|---|---|---|
| Exact-model one-step gate | `configs/runs/qwen38-27b-lora-sft-r64-a32-one-step-v3.template.json` | exact Qwen3.8 revision; BF16 Megatron LoRA rank 64 / alpha 32 over all linear layers; TP8 / PP1 / CP1 / DP1; learning rate `3e-5`; global batch 1; one 8-GPU node; 16,384-token final-test-family-free V2 Fresh75 corpus; 866 rows and payload-derived `max_steps=866`; planned pause after optimizer step 1; c1; W&B. V1 and V2 operational identities are retired after infrastructure-invalid runs. V3 preserves the V2 runtime-path repair and additionally wraps the immutable-parquet rows in SkyRL's native dataset contract; the exact setup probe now exercises that contract before a paid run. |
| First production anchor | `configs/runs/qwen38-27b-lora-sft-r64-a32-anchor-v1.template.json` | exact Qwen3.8 revision; BF16 LoRA rank 64 / alpha 32; learning rate `3e-5`; global batch 8; one 8-GPU node; one epoch; 32,768-token broad teacher-action corpus; c1; W&B |

The one-step gate reuses every eligible row it safely can from the immutable
Fresh75 teacher corpus, after removing complete source sessions from every task
family represented in the frozen final test set. It tests exact-model forward,
backward, optimizer, adapter save, and frozen-base behavior without pretending
to be the broad-data experiment. A
separate zero-optimizer-step job must reload that checkpoint, merge/export it,
and reload the complete model and tokenizer. The anchor instead names a new
create-once broad corpus that does
not yet exist. That corpus must be built from the current 1,055-task inventory,
after lineage splitting, using every usable verified-success teacher trajectory
from the training lineages. Every eligible assistant action is a target once;
copied context and tool results are masked. WebExploitBench and the Fleet final
set are excluded before selecting sessions or windows.

## Final-test-family exclusion before the canary

The source corpus contains 916 windows, 115 successful sessions and 37 exact
task versions. A deterministic join of its task identities with the reviewed
study inventory found two protected families:

- `fakelook / cyber/atoms/fakelook/ambient-identity-header-trust@0`;
- `fentry / cyber/atoms/fentry/session-revocation-not-enforced@3`.

Both are exact tasks in
`configs/data/qwen-blackbox-study-final-test-v1.json`. Filtering only their raw
task keys would be too weak because another version or alias of the same family
would teach the model about the final test. The sealed V2 request
`configs/data/qwen38-fresh75-teacher-sft-final-lock-filter-v2.request.json`
therefore joins both source rows and final-test tasks to
`configs/data/qwen-blackbox-study-inventory-v1.json` and compares the reviewed
pair `(application, task_family)`. Unknown or ambiguous identities fail. If any
row from one source session is excluded, the whole session is excluded so its
remaining windows cannot reveal the same task.

The request is create-once and atomic. Its file digest is
`sha256:a8546703b5b3e949dd5691e8b7036a90fbea815c05d3fbe48665da32233a55e6`;
its logical request digest is
`sha256:af74f0440529f417fb09d11b9466d690696a253e2b9f0527a4018122a01a78a8`;
and the reviewed selection digest is
`sha256:b7b536940995a0d4b8674a4bdadd6240ef88dc1c07200a93f27b592ac8c046ea`.
It binds the exact source manifest and Parquet, the private-selection digest,
the final-test lock and the inventory without publishing private selection
contents, row data or session identifiers.

The CPU materializer `chris-q38-f75-filter-v4` (Pod UID
`d666d0b3-378d-4083-afa3-c838b964222e`) ran in `fleet-train-jobs` from
2026-09-20 05:46:34Z through 05:46:45Z on
`computeinstance-e04fw0f2anvp0de74v`. It used exact image
`ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317`,
plan digest
`sha256:f178cb9125b3babb2b7095a51ab84d3f1c9196f005df44052ad1636b6b8ce65b`
and runtime-bundle digest
`sha256:7f91b942e38968edafe677fdd6f8dbe6f07ce19f8766085f72c14b859bc11ed2`.
It exited zero with no restart and no GPU request.

The resulting V2 corpus contains 866 supervised windows from 108 successful
sessions over 35 task versions and 998,652 supervised tokens. Seven complete
source sessions and two source task versions were excluded. The public manifest
file digest is
`sha256:9e144c94b6ad85dc715e100ac5ae6689d6d385972fdfa378d39d8da1b0ccbed5`;
its logical digest is
`sha256:5db5600ac9f403fd147b2ecbf6068b514d075d07ac683d97bc83319c6e89d149`;
the Parquet digest is
`sha256:9bac7eef01ff7dfff5de82f139a3ccaaabacf26fe070c713396874355b8ecfbc`;
and the materialization receipt has file digest
`sha256:5225a353eedf9ee060c066a7965224c11afb4157eb65fd64436d6b8db1d168a5`
and logical digest
`sha256:e8d37fd414f0c60b61d8eb5d9bedf7660866d41a24419f44ba795941c66c63e0`.

The first verifier Pod, `chris-q38-f75-filter-v4-verify` (UID
`5f40798b-db9f-47ac-bbe6-5b463f122173`), never started because its Pod omitted
the `ghcr-pull` secret and anonymous GHCR authentication returned HTTP 401. It
requested no GPU, produced no scientific output and was removed after its UID
was checked. Its plan digest was
`sha256:663a3b9684d69ae995110362fbe56da5c0e57de5f0e5066a82f356cf7d75e375`.
The corrected read-only successor,
`chris-q38-f75-filter-v4-verify-v2` (UID
`3da2136a-e78c-4e6f-a285-c6821b5fce53`), used `ghcr-pull`, the same exact
image, plan digest
`sha256:f471892f9672e6dbe26b544cc28529c6a0d89a8cd10463a5474b0e43ebac17f7`
and verifier bundle digest
`sha256:061c6cb77620044dbde7ad7a8c13036a7434f76a0a421839869f4d67a74c13a4`.
It ran from 05:54:11Z through 05:54:20Z, exited zero with no restart or GPU,
proved that the source was unchanged, and found zero exact-task and zero
task-family overlap. After UID-bound cleanup, both verifier names and the
materializer name were absent and zero GPUs remained allocated.

These facts are sealed in the sanitized qualification receipt linked above.
The exact public manifest bytes in the repository match the independently
verified file digest. The one-step template binds manifest
`../data/qwen38-fresh75-teacher-sft-final-lock-free-v2.manifest.json` and root
`/mnt/sfs/jobs/chris-q38-study-corpora-v1/fresh75-teacher-final-lock-free-v2/data`.
The exact runtime gate binds dataset digest
`219efad0257ff29b3057c91de88144542f8617a124079c5a20c79cf80b071f72`,
split digest
`sha256:b7b536940995a0d4b8674a4bdadd6240ef88dc1c07200a93f27b592ac8c046ea`,
corpus-manifest digest
`sha256:5db5600ac9f403fd147b2ecbf6068b514d075d07ac683d97bc83319c6e89d149`
and `max_steps=866`. That is the full-epoch ceiling for 866 rows at global
batch 1; the qualification run still intentionally pauses after optimizer step
1. No training request was submitted by corpus qualification or binding.

The compiler recognizes only the exact one-step Qwen3.8 Megatron-LoRA
qualification shape. The image built from
`05b8e33062cff51a4c5e940d0021de3ce30481d3` omitted the Megatron dependency
set. The later `bd97c1e431511d70bed9a9b7ccc9093342d5466f` /
`sha256:89b8d99a...d6e29af7` pair proved those dependencies load, but it predates
the exact terminal evidence producer required by this gate and is retired from
launch binding. Exact evidence-capable source
`7e9356c8e02e7382e84b8484638baccdd1bbf680` was built into immutable image
`ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317`.
A separate zero-GPU dev Pod pulled that digest, reproduced the complete 14-file
load-bearing census, imported the required runtime APIs, exited zero with no
restart, and was deleted with zero GPU allocation. The compiler accepts only
that pair, not a merely well-formed digest. The broad anchor remains blocked:
the real one-step evidence and later zero-update reload/merge gate must first
be independently accepted. A template is not a workaround for either gate.

## Exact one-step source and image binding

The one-step template contains this exact immutable image reference:

`ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317`

Do not restore the retired
`bd97c1e4...` / `sha256:89b8d99a...` dependency-only pair, the superseded
`05b8e330...` source, a mutable tag, or an older full-weight image. The next
binding contains the exact evidence producer as well as the Qwen3.8 text
bridge, complete LoRA target census, strict adapter save/reload, exact per-turn
records, compaction fixes, and Megatron dependencies. The compiled request,
CPU preflight receipt, runtime Pod `imageID`, and terminal receipt must all
agree with that one reviewed source/image binding.

Because the future image is private GHCR content, the production Jobs API
request also binds the existing `ghcr-pull` image-pull secret. That field is
emitted only for this Qwen3.8 LoRA GHCR path; existing ECR SFT requests remain
unchanged. The separate development-cluster runtime check used that cluster's
own `skyrl-ghcr-pull` secret and does not alter the production request.

Only the one-step template binds this source/image pair. The production anchor
intentionally keeps source and image `null`. Passing the one-step training gate
does not qualify broad training: the separate zero-step checkpoint reload,
merge, export, and complete model/tokenizer reload must also be accepted first.

## Exact-model gate acceptance

The one-step run may advance only when one consistent evidence chain proves:

1. exact model, tokenizer, chat-template, corpus, source commit and image IDs;
2. exactly one finite forward loss, backward pass and optimizer update;
3. finite non-zero LoRA gradients and changed adapter tensors;
4. byte-identical frozen base tensors and no unexpected trainable parameter;
5. the complete intended Qwen3.8 attention, MLP and Gated-DeltaNet adapter
   inventory, with no missing or extra adapter keys;
6. a create-once adapter checkpoint and W&B scalar telemetry from the training
   job, followed by separate zero-step checkpoint reload, deterministic merge,
   export, and complete merged model/tokenizer reload evidence; and
7. terminal resource release with zero GPUs remaining.

This gate is followed by separate measured 32K, 96K, compacted multi-turn,
160K and 262K topology gates. A configured maximum is never reported as a
measured supported length.

### Artifact admission after the one-step gate

Qwen Megatron LoRA does not use the dense FSDP checkpoint layout and does not
use the GLM PEFT checkpoint layout. It now has a separate, fail-closed receipt
validator in `training/qwen38_lora_artifacts.py`. The validator accepts only:

- the exact TP8 adapter-rank file set plus disjoint, exhaustive optimizer/RNG
  and trainer/model-metadata inventories;
- the complete ordered all-linear target surface, reconciled Gated-DeltaNet,
  attention and MLP block counts, and an explicit per-parameter shard census;
- one finite optimizer update, changed adapters, an unchanged frozen base, and
  no unexpected trainable parameters;
- a later zero-update checkpoint reload and Megatron-Bridge merge whose
  complete BF16 tensor layout equals the base layout, contains no adapter
  payload files, records a deterministic merge, reopens every output tensor
  equal, reloads with finite logits, and leaves both inputs unchanged.

Those schemas are
`cyber_qwen38_megatron_lora_checkpoint_manifest_v1` and
`cyber_qwen38_megatron_lora_merged_hf_export_v1`. The WebExploitBench
provenance validator accepts this pair as a distinct branch and rejects every
dense/Qwen schema crossing. Existing dense Qwen and GLM validation is
unchanged. This is admission logic only: an exact-image producer still must
create the two receipts from a real accepted checkpoint before either
evaluation can launch.

## Matched evaluation contracts

The protocol templates are:

- `configs/evaluation/qwen38-lora-web-l0-opencode-pass1-v1.template.json`
- `configs/evaluation/qwen38-lora-fleet-final-opencode-pass1-v1.template.json`

Both freeze Qwen3.8's base model, tokenizer and chat template; OpenCode
1.18.27 at commit `4b7e19e315cca414121ba1d61523fef74bb3ae8b`;
temperature 0.6; top-p 0.95; 32,768 output tokens; one seed; 262,144-token
context with a 229,376-token input ceiling; native automatic compaction and
continuation; and pass@1. The WebExploitBench harness configures a 20,000-token
compaction reserve and a digest-bound retry proxy that makes at most five
attempts for transport errors or HTTP 429/502/503/504. The Fleet harness uses
the growth-aware v2 policy: 20,000 tokens of headroom plus one possible 32,768-
token response, so its configured compaction reserve is 52,768 tokens; it does
not retry model requests. Neither launcher automatically repeats a scientific
attempt. These context, compaction and retry controls are now fields in the
validated protocol schema and are independently re-derived by each concrete
matched launcher plan. The remaining `UNRESOLVED_*` values still prevent a
protocol digest until images, benchmark receipts and live model identities are
known.

The WebExploitBench arm uses all 15 Level-0 targets from exact source commit
`7f97d87fa8ab728260c0ba9b09b9c8f00bb82ad5`. Collection and scoring remain
separate. The primary metric is target-macro pass@1; micro pass@1 and
infrastructure-invalid counts are secondary. Benchmark inputs, traces and
outcomes remain evaluation-only and sealed while selecting an experiment.

The Fleet arm uses the ten exact task-family-held-out versions in
`configs/data/qwen-blackbox-study-final-test-v1.json`. The primary metric is
macro pass@1 across those ten lineages. It must use exact environment, data and
verifier versions, not mutable task keys. Development tasks, not this final
set, select checkpoints and hyperparameters.

The score-free launcher-ready task/environment/data projection is
`configs/evaluation/qwen38-lora-fleet-final-task-set-v1.json` at
`sha256:4f9a7c1edc317030981029e80176fb4f1196a470d9d602b676cc49cbc63ab1e4`.
It contains exactly the ten locked task versions and the seven exact fields
required by `cyber-post-train eval prepare`; it contains no outcomes or losses.
The live preflight must still fetch and freeze each exact verifier and rendered
task binding before any rollout.

For each benchmark, derive one `cyber_prepost_comparison_v1` manifest after an
adapter/merged checkpoint is accepted. Base and intervention arms may differ
only in the bound checkpoint or adapter bytes and harmless run names. Serving
engine, precision, KV-cache type, OpenCode image, prompt, tools, tasks,
verifiers, sampling, budgets and retry rules must be byte-identical.

## Unresolved bindings

No paid or mutating action is valid until all of these are exact:

- accepted one-step target-census, optimizer, frozen-base, adapter-checkpoint,
  and W&B evidence, plus separate zero-step checkpoint reload, deterministic
  merge/export, and complete merged model/tokenizer reload evidence (compiler
  support alone is not acceptance);
- complete Qwen3.8 adapter-target census and strict checkpoint/merge receipts;
- exact-image CPU data/runtime preflight, authenticated request preview,
  duplicate/output-root checks and immutable compiled one-step request;
- create-once 32K broad teacher-action corpus manifest and staged SFS root;
- qualified immutable matched serving image and serving-engine commit;
- matched base and merged-adapter serving registrations plus live-parity proof;
- qualified immutable OpenCode images for WebExploitBench and Fleet;
- WebExploitBench task/environment/verifier/prompt/tool manifest hashes;
- Fleet live verifier/prompt preflight receipts for the already-frozen exact
  task/environment/data tuples;
- final protocol digests; and
- absent output roots, absent duplicate run names, live c1 policy, and available
  goal-owned node/failure budget immediately before submission.

Template file hashes are intentionally not frozen while source, image, corpus,
serving, and evaluation bindings remain unresolved. Compute all four hashes
only when turning the reviewed templates into immutable plans; any later change
requires review and new run identities.
