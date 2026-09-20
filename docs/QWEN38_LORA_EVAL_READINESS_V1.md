# Qwen3.8 LoRA matched-evaluation readiness

Status: **fail closed; no evaluation was launched** (2026-09-19)

This audit maps the two frozen Qwen3.8 LoRA evaluation templates to the
currently tested launchers. The templates are scientific contracts, not job
inputs. Both intentionally retain unresolved values and fail protocol
validation until one accepted checkpoint, merged export, serving registration,
and live-parity receipt exist.

## Immutable bindings already frozen

Both arms bind the same base revision, tokenizer/chat template, OpenCode
version, sampling, context, compaction and pass@1 policy. The only permitted
base-versus-candidate intervention is the accepted model artifact and its
serving identity.

| Surface | Exact identity |
|---|---|
| Qwen3.8 base revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| base weights manifest | `sha256:06c94e47c0e31fd331ed410665c830ab1b657f90f15a1b11e7bc45e2de00f352` |
| tokenizer manifest | `sha256:3938a9a8172f2738fed1be44efc11e2562059269d50d3721213f44802b53b4e1` |
| chat template | `sha256:c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| OpenCode source | `4b7e19e315cca414121ba1d61523fef74bb3ae8b` |
| OpenCode release | `sha256:4af5494f9433f59db8c1e344198f0ee72a50c06ec009fb4a8aeab4c2d4abd702` |
| context/output ceilings | `262144` context, `229376` input, `32768` output tokens |
| Web compaction | native automatic continuation, `20000` configured reserve; renderer `sha256:949236e1031728f42db6c0ef0ebb89ceb0ef380e89459499c8c8e82552eb8520` |
| Fleet compaction | growth-aware v2 automatic continuation, `20000` headroom + `32768` response = `52768` configured reserve; renderer `sha256:428e9f2e4d4c758c4f682cbed314cf96b866d051b34fa0eea8757a3aad97aa1d` |
| Web request retry | transport or HTTP 429/502/503/504 only, at most 5 attempts, 2/4/8/16-second jittered backoff; proxy `sha256:d142a7ceb5c2e613bfa42269b7220bed08b50ad1ee28ceb0060c1691be4482c4` |
| Fleet request retry | exactly one model request, no retry; proxy `sha256:a65270cb021cbc8e2fe9b00d901b4b70e9ddc98dd58d93e9f695b1768dfb984d` |
| scientific-attempt retry | disabled in both protocols (`max_retries=0`) |
| Fleet tool schema | `sha256:85fad6bdc3a835bf52a11a99b3387740eb06eb3d1720ad9bb33f3feac215b44a` |
| Fleet system prompt | `sha256:8512560b0706f8859774bfe8e93c2c86ecf6a6a4603fe59db004fd04fcc738d1` |
| WebExploitBench revision | `7f97d87fa8ab728260c0ba9b09b9c8f00bb82ad5` |
| CAGE source | `09a191c565230cebb8255899d622d23c7ddeff33` |
| CAGE patch set | `sha256:baf629de2ea13a5648f9de792514efcd73c71667c1ffe375b8669cbc1ad72b51` |
| Level-0 target order | `sha256:d491b66885c8a379a887ee792d96ade76ffdc34cbfa3e75f9bc61c6975a29610` |
| Fleet final lineage lock | `sha256:dc625115be8c775de8ecf476bc0d8e06e71b6c7f4160ed50b25eb0308b30d8d7` |
| Fleet launcher task set | `sha256:4f9a7c1edc317030981029e80176fb4f1196a470d9d602b676cc49cbc63ab1e4` |

The Fleet task set is
`configs/evaluation/qwen38-lora-fleet-final-task-set-v1.json`. It is an exact,
outcome-free projection of the ten final held-out task versions into the seven
fields consumed by the Fleet worker.

The original Fresh75 training corpus is not eligible for the new comparison
unchanged. It contains two reviewed task families that are also in this final
set. The create-once V2 filter request is
`configs/data/qwen38-fresh75-teacher-sft-final-lock-filter-v2.request.json`.
It compares reviewed `(application, task_family)` identities, not only task-key
strings, and removes complete sessions for protected families. The verified
public successor is
`configs/data/qwen38-fresh75-teacher-sft-final-lock-free-v2.manifest.json`.
It contains 866 supervised windows from 108 successful sessions over 35 task
versions and 998,652 supervised tokens. Independent read-only verification
proved the source unchanged and found zero exact-task and zero task-family
overlap. The manifest file digest is
`sha256:9e144c94b6ad85dc715e100ac5ae6689d6d385972fdfa378d39d8da1b0ccbed5`,
its logical digest is
`sha256:5db5600ac9f403fd147b2ecbf6068b514d075d07ac683d97bc83319c6e89d149`,
and the Parquet digest is
`sha256:9bac7eef01ff7dfff5de82f139a3ccaaabacf26fe070c713396874355b8ecfbc`.
The sanitized terminal evidence is recorded in
`configs/qualification/qwen38-lora-one-step-corpus-2026-09-20-v1.json`.

The materializer Pod UID was `d666d0b3-378d-4083-afa3-c838b964222e` and the
successful verifier Pod UID was `3da2136a-e78c-4e6f-a285-c6821b5fce53`.
An earlier verifier Pod, UID `5f40798b-db9f-47ac-bbe6-5b463f122173`, never
started because `ghcr-pull` was omitted; it requested no GPU and produced no
scientific output. UID-bound cleanup was confirmed for all three exact names,
with zero GPUs remaining. The current one-step template and exact runtime gate
now bind the V2 public manifest and staged root
`/mnt/sfs/jobs/chris-q38-study-corpora-v1/fresh75-teacher-final-lock-free-v2/data`.
The runtime gate also binds
dataset digest
`219efad0257ff29b3057c91de88144542f8617a124079c5a20c79cf80b071f72`,
split digest
`sha256:b7b536940995a0d4b8674a4bdadd6240ef88dc1c07200a93f27b592ac8c046ea`,
corpus-manifest digest
`sha256:5db5600ac9f403fd147b2ecbf6068b514d075d07ac683d97bc83319c6e89d149`
and `max_steps=866`. This is the full-epoch ceiling for 866 rows at global
batch 1; the qualification run remains configured to pause after optimizer step
1. The CPU data/runtime preflight and real one-step acceptance are still
pending. Corpus qualification and binding do not authorize training or an
evaluation launch.

## Readiness matrix

| Gate | Local authority | State before the first accepted checkpoint |
|---|---|---|
| shared scientific controls | `configs/evaluation/qwen38-lora-web-l0-opencode-pass1-v1.template.json` and `configs/evaluation/qwen38-lora-fleet-final-opencode-pass1-v1.template.json` | frozen but deliberately unresolved; `training.science.validate_eval_protocol` rejects both |
| Fleet held-out task identity | `configs/evaluation/qwen38-lora-fleet-final-task-set-v1.json` | ready: ten exact task/environment/data tuples, no outcome-derived selection |
| WebExploitBench collection code | `evals/webexploitbench/tensorlake/collection_launcher.py`, `collection_pair.py`, `collection_supervisor.py` | locally tested; cannot seal a plan without a newly qualified snapshot, exact projects/images, and model identities |
| WebExploitBench scoring code | `evals/webexploitbench/tensorlake/deferred_score.py` | locally tested; cannot seal a score plan before accepted rollout collections and judge qualification exist |
| Fleet held-out launcher | `cyber-post-train eval`, implemented by `evals/fleet/evaluate.py` | locally tested; runnable config files still need exact live base/candidate routes, immutable worker images, verifier/prompt checks, and unique campaign storage |
| candidate training artifact | `training/qwen38_lora_artifacts.py::validate_checkpoint_receipt` | validator, exact source/image producer, and leak-free V2 one-step corpus are qualified and bound; CPU preflight, the real one-step checkpoint and accepted receipt are still missing |
| candidate merged model | `training/qwen38_lora_artifacts.py::validate_export_receipt` | validator present; separate zero-step reload/merge/export producer and accepted receipt still missing |
| candidate serving | `evals/webexploitbench/tensorlake/collection_provenance.py` | blocked until registration and live-parity receipts bind the exact merged model and route |
| matched evaluation launch | commands below | closed until every row above is accepted; no output root, database, sandbox, or scored attempt has been created by this work |

The two protocol templates are not TensorLake or Fleet launcher inputs. They
freeze the comparison. Concrete launcher drafts/configs are derived only after
the missing live identities exist; this avoids inventing a route, snapshot, or
receipt merely to make a config parse.

Before either paid launch, one final local acceptance step must compare each
sealed concrete launcher plan with its completed protocol template. The
existing TensorLake parity check proves that the base and candidate plans are
identical except for the model, but it does not by itself prove that both plans
used the values in this separate template. The Fleet launcher has the same
boundary. Until that plan-to-template receipt exists and binds the completed
protocol digest, the comparison remains deliberately non-runnable even if all
live routes and images have become available.

The previously tested trainer pair (`bd97c1e4` / image digest beginning
`89b8d99a`) does not expose the rank-local evidence API required by the strict
checkpoint receipt, so it is retired from the launch binding. The intended new
lineage is exact evidence-capable commit
`7e9356c8e02e7382e84b8484638baccdd1bbf680`, which contains the
non-scientific cache-hardening base plus the new evidence patch. Its complete
14-file census and immutable image
`ghcr.io/fleet-ai/skyrl-fleet-v2/trainer@sha256:7da4adba80d032509dba69fb4dd23bedca17fde3e2b88643815f80d6ee6c5317`
passed the bounded CPU qualification recorded in
`configs/qualification/qwen38-lora-megatron-trainer-image-2026-09-20-v1.json`.
That proves image identity and imports, not training. No evaluation artifact
may treat the older image, a placeholder digest, or the CPU qualification as
an accepted checkpoint.

## WebExploitBench Level-0

### Working path

The maintained path is score-free collection followed by independent scoring:

1. `evals.webexploitbench.tensorlake.collection_snapshot_qualification`
   creates, verifies, snapshots and releases one fresh score-free worker image.
2. `evals.webexploitbench.tensorlake.collection_launcher` seals one exact plan
   for each arm.
3. `evals.webexploitbench.tensorlake.collection_pair` binds the two plans and
   proves that only model identity differs.
4. `evals.webexploitbench.tensorlake.collection_supervisor` preserves and
   releases
   collection sandboxes without reading benchmark answers or scores.
5. `evals.webexploitbench.tensorlake.deferred_score` later scores the stored
   collections and validates the matched pair.

After every missing binding below is materialized, the exact command sequence
is:

```sh
python -m evals.webexploitbench.tensorlake.collection_snapshot_qualification \
  build-cage-bundle --repository <reviewed-cage-checkout> \
  --output <fresh-cage-source.bundle>
python -m evals.webexploitbench.tensorlake.collection_snapshot_qualification \
  seal-plan --draft <snapshot-qualification-draft.json> \
  --output <snapshot-qualification-plan.json>
python -m evals.webexploitbench.tensorlake.collection_snapshot_qualification \
  validate-plan --plan <snapshot-qualification-plan.json>
python -m evals.webexploitbench.tensorlake.collection_snapshot_qualification \
  run --plan <snapshot-qualification-plan.json> \
  --state <fresh-snapshot-qualification-state> --execute
python -m evals.webexploitbench.tensorlake.collection_launcher seal-plan \
  --draft <base-draft.json> --output <base-plan.json>
python -m evals.webexploitbench.tensorlake.collection_launcher seal-plan \
  --draft <candidate-draft.json> --output <candidate-plan.json>
python -m evals.webexploitbench.tensorlake.collection_pair seal \
  --base-plan <base-plan.json> --candidate-plan <candidate-plan.json> \
  --output <pair.json>
python -m evals.webexploitbench.tensorlake.collection_pair create \
  --pair <pair.json> --arm base --state <pair-state-dir> --task-index <N> --execute
python -m evals.webexploitbench.tensorlake.collection_pair create \
  --pair <pair.json> --arm candidate --state <pair-state-dir> --task-index <N> --execute
python -m evals.webexploitbench.tensorlake.collection_pair start-collection \
  --pair <pair.json> --arm base --state <pair-state-dir> --task-index <N> \
  --sandbox-id <base-sandbox-id-from-create> --execute
python -m evals.webexploitbench.tensorlake.collection_pair start-collection \
  --pair <pair.json> --arm candidate --state <pair-state-dir> --task-index <N> \
  --sandbox-id <candidate-sandbox-id-from-create> --execute
python -m evals.webexploitbench.tensorlake.collection_supervisor seal-plan \
  --draft <pair-state-dir>/base/<generated-base-terminal-draft.json> \
  --output <base-supervisor-plan.json>
python -m evals.webexploitbench.tensorlake.collection_supervisor run \
  --plan <base-supervisor-plan.json> --state <base-supervisor-state.json> --execute
python -m evals.webexploitbench.tensorlake.collection_supervisor seal-plan \
  --draft <pair-state-dir>/candidate/<generated-candidate-terminal-draft.json> \
  --output <candidate-supervisor-plan.json>
python -m evals.webexploitbench.tensorlake.collection_supervisor run \
  --plan <candidate-supervisor-plan.json> \
  --state <candidate-supervisor-state.json> --execute
python -m evals.webexploitbench.tensorlake.deferred_score seal-plan \
  --draft <score-draft.json> --output <score-plan.json>
python -m evals.webexploitbench.tensorlake.deferred_score score \
  --plan <score-plan.json>
python -m evals.webexploitbench.tensorlake.deferred_score score \
  --plan <score-plan.json> --execute
python -m evals.webexploitbench.tensorlake.deferred_score validate-pair \
  --base-plan <base-score-plan.json> --candidate-plan <candidate-score-plan.json>
```

`create` only allocates one sandbox. It does not run the agent. The separate
`start-collection` call is mandatory and writes the terminal-plan draft named
above. The terminal supervisor then verifies and snapshots the completed,
unscored rollout before releasing that sandbox. Skipping either step cannot
produce an accepted collection. Run the same sequence for every task index in
the frozen order; if a create response is uncertain, use `reconcile-create`
instead of repeating `create`.

### Missing bindings

- one fresh, fully qualified filesystem snapshot and its completion/release
  receipts;
- benchmark manifest, validation report and source-file digests;
- exact immutable evaluator, agent and network-proxy images;
- all fifteen target image and task-manifest digests;
- an agent-memory preflight for every TensorLake project;
- per-target project and create-once collection identities;
- exact runner/collector code digests and student-only model registries;
- the live base model identity;
- the candidate checkpoint, merged export, serving-registration and live-parity
  receipts described below;
- sealed base/candidate collection plans and their pair receipt; and
- before scoring, an accepted judge qualification, scorer digest, project and
  judge-model binding.

No qualifying snapshot, sealed collection plan/pair, or evaluation preflight
receipt was present in the checked local paths during this audit.

## Fleet final held-out set

### Working path

The maintained launcher is `cyber-post-train eval`, backed by
`evals/fleet/evaluate.py`:

```sh
cyber-post-train eval prepare <base.yaml> --output <base-prepared>
cyber-post-train eval preflight <base-prepared>
cyber-post-train eval init <base-prepared>
cyber-post-train eval run <base-prepared> shared <worker-id> --limit <N>

cyber-post-train eval prepare <candidate.yaml> --output <candidate-prepared>
cyber-post-train eval preflight <candidate-prepared>
cyber-post-train eval init <candidate-prepared>
cyber-post-train eval run <candidate-prepared> shared <worker-id> --limit <N>
```

The two YAML files must bind the same
`configs/evaluation/qwen38-lora-fleet-final-task-set-v1.json`, harness, prompt,
tools, sampling, budgets, retry policy and worker images. Only their exact live
model identity may differ.

### Missing bindings

- digest-pinned agent and proxy images staged on the CPU worker;
- live base and candidate route profiles, revisions, served-model IDs and
  session names;
- the accepted candidate registration/live-parity chain;
- live verifier and rendered-prompt preflight receipts for all ten already
  frozen task/environment/data tuples;
- Fleet credentials and PostgreSQL DSN supplied through the approved secret
  path, never a checked-in file;
- unique database, output and campaign identities; and
- a small matched-pair comparison receipt around the two descriptive campaign
  outputs before making a causal lift claim.

## Candidate artifact chain

`training/qwen38_lora_artifacts.py` defines a Qwen-only branch for the real
Megatron checkpoint shape:

- `cyber_qwen38_megatron_lora_checkpoint_manifest_v1` binds all eight TP adapter
  files, all optimizer/RNG and metadata files, the complete target/parameter
  census, one finite update and frozen-base identity.
- `cyber_qwen38_megatron_lora_merged_hf_export_v1` binds the exact checkpoint,
  exact base inventory, zero-update checkpoint/optimizer reload and merge,
  complete BF16 layout, absent adapter payloads, deterministic merge and finite
  model reload.

`evals/webexploitbench/tensorlake/collection_provenance.py` accepts that pair as
a distinct schema branch. Dense Qwen and GLM receipts retain their existing
validators, and crossing a dense receipt with a Qwen-LoRA receipt fails.

The validator and immutable producer image are ready. The exact-image one-step
run and a separate zero-update inventory/merge/reload job still must emit these
two create-once receipts. Until that happens, neither evaluation is runnable.
