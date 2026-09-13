# Exact post-training serving registration

This is a **prepared, unlaunched** SGLang/Qwen3.8 handoff. It replaces the old
fixed-checkpoint mutation path with one configurable transaction. No endpoint,
dev canary or scored evaluation was created while implementing it. Other serving
engines need separately tested argument/schema support; do not relabel an engine
to bypass a rejection.

Use `python -m training.register_post_sft` (or
`python -m training.serving_registration`). Historical V1 receipt validation
remains readable; the old direct `register(receipt, key)` mutation is retired.
The original source is retained as a test fixture solely to validate historical
ConfigMap evidence without changing its frozen hashes.

## Enforced boundaries

- The candidate clones the complete reviewed baseline desired spec. Only model
  ID, display name, checkpoint revision and model/cache paths change. Model
  revision is the selected export receipt SHA-256, not a mutable alias.
- The baseline requires BF16, a digest-pinned direct SGLang entrypoint, explicit
  bounded GPU resources and one replica. Shell wrappers, alternate/abbreviated
  model-identity arguments, literal credentials, unknown spec fields and automatic
  replica expansion are rejected.
- Export and qualification receipts are file-hash and self-digest checked. The
  SFT path requires its real GPU check. The Miles path instead reopens both the
  raw HF export and accepted zero-update HF reload/release through their public
  validators. Every staged payload is independently rehashed, with exact
  inventory and no symlinks. A staging receipt binds the
  PVC/observer/mount-to-catalog mapping, not merely an unrelated directory
  containing identical bytes.
- Preview validates locally and performs authenticated **GETs**: correct Fleet
  team, unchanged baseline, and no existing candidate ID, checkpoint revision or
  overlapping cache/model path. Unknown/partial pagination schemas fail closed.
- Execute requires reviewed exact dev-serving evidence, takes a single-writer
  lock, repeats preview, fsyncs a create-once intent and makes **one** POST. Any
  exception leaves the intent and blocks resubmission. Reconcile performs GETs
  and may write only a local receipt.
- Registration never means loaded serving readiness: results explicitly retain
  `serving_ready: false` and `live_parity_verified: false`.

This is not a new staging engine or dev-certification service. Producer receipts
must come from measured, reviewed evidence. A self-digest detects byte drift; it
is **not a signature authenticating the producer**. Never fill missing evidence
with assertions or copy an old receipt onto a new artifact.

## Prepare inputs

Use private JSON with schema `cyber_exact_serving_registration_v2` and exactly
these fields. All evidence paths are absolute; bare or `sha256:`-prefixed file
SHA-256 values are normalized.

| Field | Required value |
| --- | --- |
| `base_registration` | `{path, sha256}` for the exact observed base `{id, spec}` |
| `export` | `{path, sha256}` for the selected `cyber_native_checkpoint_hf_export_v1` `EXPORT.json` |
| `gpu_check` | `{path, sha256}` for matching passed `cyber_hf_export_check_v1`: one-GPU reload, finite synthetic forward, zero optimizer steps and unchanged source |
| `staging` | `{path, sha256}` for measured `cyber_inference_staging_check_v1`, below |
| `storage` | `namespace: inference`, exact `pvc_name`, canonical `pvc_uid`, absolute observed `staged_root`, exact `/models/...` `source_path` |
| `model_id` | New DNS-label identity, distinct from retained/active catalog models |
| `display_name` | Explicit descriptive label without private content |

The existing SFT schema and fields above are unchanged. A Miles RL artifact uses
schema `cyber_exact_miles_serving_registration_v1`, replaces `gpu_check` with
`reload_acceptance`, and keeps every other field identical:

| Miles field | Required value |
| --- | --- |
| `export` | `{path, sha256}` for the raw `cyber_miles_native_hf_export_v2` `EXPORT.json`; `inspect_export(...)` must reopen the exact source receipts, files, tensors, dtype and source-equivalence witness |
| `reload_acceptance` | `{path, sha256}` for `cyber_miles_hf_reload_accepted_v1`; `validate_reload_accepted(..., check_files=True)` must reopen its plan, submission, result, controller terminal and external GPU-release evidence |

The Miles plan records both file and self digests for the export and reload
acceptance, the complete staged `files`-manifest digest, and all public reload
validation digests. It derives `source_update_identity` only from the validated
training-plan digest, source-checkpoint receipt digest and exported tensor
inventory digest. It does **not** call the export/reload work an optimizer step:
both stages execute zero optimizer updates, while the source-checkpoint receipt
is the truthful identity of the selected trained update. The serving model
revision remains the raw export self digest so different export bytes cannot
collide under one checkpoint label. No `active_dev3`, `dev4`, or other mutable
canary name is part of this serving contract.

The staging receipt requires:

- `schema: cyber_inference_staging_check_v1`, `status: passed`;
- `namespace`, `pvc_name`, `pvc_uid`, `staged_root`, `source_path` matching the
  configuration, and actual canonical `observer_pod_uid`;
- `observed_mount_root`, `catalog_root: /models`; the source path equals
  `/models` joined with `staged_root` relative to the observed mount root;
- `mount_evidence_sha256` binding independent namespace/Pod/PVC/mount observations;
- exact `export_receipt_sha256`, `export_file_sha256`, `files_manifest_sha256`
  (the digest of the complete export `files` map);
- `destination_create_once: true`, `payload_rehashed: true`, supported by
  publication/rehash observations, not inferred from directory presence.

The operator must obtain and review real UID-bound mount evidence. Merely
creating a JSON receipt cannot establish that a desktop directory is the
inference PVC. This module verifies the binding and bytes but does not contact
Kubernetes or create a helper to discover a mount. No copy, repair, overwrite or
workload exec is performed here.

```bash
python -m training.register_post_sft prepare --config <private-config.json> --output <new-prepared-directory>
python -m training.register_post_sft preview <prepared-directory>
```

These examples do not execute automatically. Preparation freezes private
configuration, artifact/registration/contract hashes and registration-code
dependencies. Changed code/evidence needs a newly reviewed plan, not edited
prepared bytes. Preview requires `FLEET_API_KEY` through secret handling; never
put its value in command arguments, JSON or logs.

The supported control plane is
`https://inference.flt.build/fleet/v1/models`. A September 11 authenticated
read-only probe verified `{object: list, data: [...]}` and exact model GET with
`resource_version`, `spec`, `status`. No server-preview endpoint was verified;
preview explicitly returns `server_dry_run: false`. Unknown shapes stop the
transaction instead of selecting a fallback.

## Dev qualification before future execution

The production registration command accepts separately reviewed
`cyber_serving_dev_qualification_v1` evidence with `status: passed`, `cluster: dev`
and `api_base_url: https://api.ft.dev.flt.build`. That is the **actual bounded dev
qualification run** origin, not an invented dev inference-control-plane URL.
The test must exercise the selected serving engine and request/tool/cleanup path,
not merely the Transformers export checker.

Bind `execution_contract_sha256` and `export_receipt_sha256` from the prepared
plan, canonical controller/Pod UIDs, timezone-qualified `observed_at`, exact
observed `runtime_image_id` (a Kubernetes imageID or `repository@sha256:...` with
the requested digest) and `evidence_manifest_sha256`.
Its checks must establish all six of:

- `exact_model_identity`, `full_model_reload`, `finite_forward`;
- `structured_tool_call`, `context_continuation`, `cleanup_verified`.

For a Miles plan, the same dev qualification must additionally bind the plan's
exact `source_update_identity_sha256`, `reload_acceptance_receipt_sha256`, and
`staged_manifest_sha256`, the complete evidence-manifest digest, zero training/
rollout/verifier/benchmark work, and confirmed GPU release. The accepted
one-forward HF reload/release receipt is
an artifact gate, not a serving canary and not serving readiness. Missing any
generic staging or Miles dev-serving binding stops execution before a POST.

The execution contract permits dev/prod identity/path/revision and placement
differences; image, normalized runtime, resource shape, precision, parallelism,
scaling and other behavior stay exact. Review storage/placement separately.
For an accepted Miles export, the bounded producer/acceptor for this evidence is
the [Qwen3.8 Miles serving handoff](QWEN38_MILES_SERVING_HANDOFF.md). It runs one
dev-only one-GPU SGLang canary, contains no task/benchmark content and accepts
only after independent UID-bound terminal and GPU-release evidence. The generic
registration code still does not infer or manufacture dev evidence. Follow the
[dev-first policy](CLUSTER_ALERTS_AND_INFERENCE_SERVING.md), current authority,
live capacity/access checks and lifecycle bounds before:

```bash
python -m training.register_post_sft execute <prepared-directory> --dev-evidence <measured-dev-receipt.json> --dev-sha256 <receipt-file-sha256>
```

Do **not** execute merely because local tests pass. No POST was performed during
implementation. The module does not discover aggregate consumption or grant
resources: reconcile all owned training/serving/CPU allocations, the eight-node
ceiling, useful consumers and release conditions first.

## Ambiguous response and final acceptance

Keep one creator and one durable prepared directory across humans/automations.
Never copy/re-prepare the same identity elsewhere to bypass `intent.json`.
The journal prevents a second POST through this transaction; it is not a global
distributed lock across unrelated clients. A stable idempotency key is sent,
but server-side atomic create/CAS semantics were **not** verified. Do not infer
that guarantee from the header.

After uncertainty, use only:

```bash
python -m training.register_post_sft reconcile <prepared-directory>
```

This checks account, exact model ID and full desired spec. Missing/conflicting
objects remain held; no retry, PATCH, overwrite, delete or differently named
replacement is issued. A matching result records the control-plane resource
version. Existing local receipts require matching current readback before return.
Keep artifacts and claims intact during incident review.

Before scored consumers attach, independently bind live InferenceModel,
Deployment/Pod UIDs, actual loaded export/runtime, ready gateway and matched
base/post request/tool/decoding behavior. Freeze a new evaluation protocol;
select checkpoints on Fleet dev only, never WebExploitBench outcomes. After
useful consumption ends, drain and release only the owned route through its
supported lifecycle. See [evaluation readiness](QWEN_POST_SFT_EVAL_READINESS_2026-09-11.md)
and [GPU lifecycle](GPU_RESOURCE_LIFECYCLE.md).
