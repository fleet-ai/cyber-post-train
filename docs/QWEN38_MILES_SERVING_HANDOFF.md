# Qwen3.8 Miles export to evaluation endpoint

This is the narrow post-RL handoff from an accepted Miles HF export to a model
route that Fleet and Tensorlake may evaluate. It does not train, select a
checkpoint, deploy during preparation, or treat a registered model as ready.

## What must already exist

Use one exact pair from the same trained checkpoint:

- a `cyber_miles_native_hf_export_v2` `EXPORT.json`; and
- its accepted `cyber_miles_hf_reload_accepted_v1` receipt.

The reload must have executed zero optimizer updates and proved exact HF model,
tokenizer and finite-forward reload plus GPU release. The dev serving canary
reopens all of that evidence. A different export, reload, model revision or
tensor inventory requires a new prepared canary.

The reviewed production behavior is cloned from
`evals/fleet/serving/qwen38-27b-dedicated-v1.registration.json`: exact
Qwen3.8-27B revision, BF16, digest-pinned SGLang, Qwen reasoning/tool parsers,
decoding/context arguments and one-GPU resource shape. Only the served identity
and immutable export path/revision differ in dev.

## 1. Prepare the bounded dev canary

Copy `configs/qualification/qwen38-miles-serving-dev-v1.template.json` to
private durable experiment storage. Fill every `null` with the exact new name,
SFS output root, absolute base-registration path, accepted export path/file
SHA-256 and accepted reload path/file SHA-256. Do not commit the filled file.

```bash
uv run python -m cyber_post_train.cli miles-serving-dev \
  <private-dev-config.json> --output <new-prepared-directory>
uv run python -m cyber_post_train.cli preflight <new-prepared-directory>
uv run python -m cyber_post_train.cli preview <new-prepared-directory> --cluster dev
```

Preparation and preflight use no GPU. Preview is GET-only. Review the exact
one-worker/one-GPU `c1` (effective q1/10000), no-requeue render and point-in-time
dev capacity before the only POST:

```bash
uv run python -m cyber_post_train.cli submit <new-prepared-directory> --cluster dev
```

The job launches the exact SGLang command on localhost, rehashes every export
file, and performs only public synthetic probes: model identity, a short forward,
multi-turn continuation and a forced `identity` tool call. It records no response
text and includes no Fleet task, verifier, score or WebExploitBench content. It
then stops the SGLang process group. Startup is bounded at 30 minutes, requests
at 5 minutes and shutdown at 1 minute; the Jobs API releases the allocation on
exit. Never resubmit a directory with `SUBMISSION.jsonl`.

## 2. Accept real terminal and release evidence

While the exact job is live, preserve its API run ID, RayJob UID, Workload UID,
RayCluster UID, Pod UID and runtime image ID. After it succeeds, verify zero
restarts, exit 0 and that its RayCluster, Workload, GPU Pods and active GPU count
are all absent/zero. Copy
`configs/qualification/qwen38-miles-serving-dev-external-v1.template.json` to
private storage and replace every `null` from those observations.

The external observation is bound by its reviewed file SHA-256. It is not a
self-authenticating claim, so preserve how each field was read from the exact
API/Kubernetes objects. Acceptance performs no cluster read and cannot make an
unmeasured assertion true:

```bash
uv run python -m cyber_post_train.cli miles-serving-dev-accept \
  <new-prepared-directory> \
  --result <SFS-output>/DEV_SERVING_RESULT.json \
  --result-sha256 <file-sha256> \
  --external <reviewed-external-observation.json> \
  --external-sha256 <file-sha256> \
  --output <new-dev-qualification.json>
```

Acceptance reopens the exact export/reload, request, six serving checks, UIDs,
priority, zero-requeue policy, image and full GPU release. It emits the
`cyber_serving_dev_qualification_v1` consumed by production registration. It
still says `serving_ready: false` and performs no production mutation.

## 3. Stage and register in production

The production inference controller reads the `hf-cache-shared` PVC, not the
dev/training SFS path. Use a reviewed CPU-only, create-once transport to copy the
same accepted export into a new directory on that PVC. Preserve measured
namespace, PVC UID, observer Pod UID, mount-to-`/models` mapping and complete
payload rehash as `cyber_inference_staging_check_v1`. Do not reuse the historical
Qwen3.6 staging job: it hard-codes another source, destination and composition.
Do not point registration at SFS and do not synthesize a staging receipt.
`configs/qualification/qwen38-miles-inference-staging-evidence-v1.template.json`
lists the exact required observation fields; its `null` values and missing
self-digest deliberately make the template unacceptable as evidence.

Fill the private copy of
`configs/qualification/qwen38-miles-serving-prod-v1.template.json` with that
exact staging evidence, new non-overlapping model/cache identity and the same
accepted export/reload. Then:

```bash
uv run python -m training.serving_registration prepare \
  --config <private-prod-config.json> --output <new-prod-prepared-directory>
uv run python -m training.serving_registration preview <new-prod-prepared-directory>
uv run python -m training.serving_registration execute <new-prod-prepared-directory> \
  --dev-evidence <new-dev-qualification.json> --dev-sha256 <file-sha256>
```

`execute` repeats account, baseline and duplicate GETs, writes a create-once
intent and makes one idempotency-bound POST. On an ambiguous response, use only
`reconcile`; never rename and retry. Registration deliberately returns
`serving_ready: false`.

## 4. Qualify the live route before evaluations

Independently bind the created InferenceModel, Deployment/Pod UIDs, exact loaded
export and runtime image, health/gateway route and matched request/tool/decoding
behavior. Only that live-parity receipt may make the route usable by Fleet or
Tensorlake evaluation launchers.

Keep WebExploitBench sealed and evaluation-only. It may compare the already
selected checkpoint only after live route qualification; it must never select a
checkpoint, tune serving/training parameters, or enter any training corpus. Use
the [post-training WebExploitBench protocol](QWEN_POST_TRAINING_WEBEXPLOITBENCH_PROTOCOL.md)
to bind its Tensorlake plan to this exact live checkpoint identity.

At repository implementation time this document records no live dev canary,
staging copy, production registration or readiness receipt. Those remain
measured operations against the future accepted export.
