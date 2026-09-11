# Qwen Fleet-dev matched-base certification

The split-A base model route is operationally ready, but it is **not yet a
scientifically launchable matched control**. The read-only audit found the exact
Qwen3.8-27B revision and a healthy shared route, while also proving that current
base payload, harness-image and base/post live-pair receipts are absent. A Ready
replica and matching catalog label are not substitutes for those receipts.

The immutable preparation is
[`qwen38-blackbox-fleet-dev-a-base-certification-v1.json`](../configs/evaluation/qwen38-blackbox-fleet-dev-a-base-certification-v1.json).
It binds the split-A parent, task set, base-control template, model lock, route
audit, exact current route object identities, SGLang runtime and Fleet evaluator
source hashes. It authorizes no scored work and leaves the frozen base-control
template unchanged.

## What can be certified now

Two receipts have no post-SFT dependency and require no new GPU allocation:

1. **Existing base artifact and route.** Rehash the complete payload actually
   mounted by the current shared serving object and prove its weight, tokenizer
   and chat-template manifests equal the exact upstream lock. Bind that mount
   evidence to the current InferenceModel, Deployment, Pod, Service,
   EndpointSlice, image ID, normalized arguments, catalog, `model_info` and
   `server_info`. Gateway checks are GET-only; this receipt permits zero prompt,
   completion or scoring requests. If the live cache cannot be observed through
   an approved read-only mount or the current serving Pod, the shared route stays
   closed rather than inheriting the upstream hashes from its label.
2. **Agent and fixed-proxy images.** Build or reuse only digest-pinned
   linux/amd64 images, then clean-pull them on a safe CPU/dev worker. The receipt
   must bind the exact OpenCode 1.18.27 release and repository runtime files and
   prove writable private-home startup, ordered `bash`/`submit_report` tools,
   fixed temperature/top-p/seeds/32,768 output limit, native v2 compaction,
   automatic continuation, and proxy path/auth/size/request bounds. These are
   offline synthetic checks: they make no model, task, grader or score request.

Historical image IDs or the v1-compaction serving receipts do not pass. Existing
images may be reused only after their immutable digests and all current v2 checks
are captured in the new self-digesting receipt.

The current route metadata/readiness half can therefore be collected now. Do
not cache a behavioral “base half” as parity: behavior must be measured against
the post-SFT arm in the same bounded observation window.

## What must wait for post-SFT serving

The final live-pair receipt cannot exist until a selected checkpoint has all of:

- an accepted checkpoint seal and BF16 export/reload receipt;
- create-once staging with a complete rehashed payload;
- a current post-SFT serving registration cloned from the base runtime; and
- simultaneous healthy base and post-SFT routes.

Within one 15-minute window, the parity producer revalidates both registrations
and performs only non-scored probes: exact tokenizer encode/decode, structured
tool calling, a deterministic finite logit probe within each arm, and native-v2
compaction plus continuation. It creates no Fleet task instance and makes no
grading request. Base and post must match in endpoint block, engine, precision,
quantization, TP/DP, context, KV cache, parsers, serving image/arguments,
tokenizer/template and agent/proxy images. The post weight manifest must be the
only scientific difference.

## Offline gate

Validate the preparation without contacting Fleet or Kubernetes:

```sh
python -m evals.fleet.base_control_certification validate-plan \
  configs/evaluation/qwen38-blackbox-fleet-dev-a-base-certification-v1.json
```

After the three measured private receipts exist, assemble a create-once
sanitized certificate:

```sh
python -m evals.fleet.base_control_certification certify \
  configs/evaluation/qwen38-blackbox-fleet-dev-a-base-certification-v1.json \
  --base-route /restricted/base-route.json \
  --harness-images /restricted/harness-images.json \
  --live-pair /restricted/live-pair.json \
  --output /restricted/base-control-certificate.json
```

The validator rejects unknown receipt fields, digest drift, stale audited object
identity, incomplete payload hashing, mutable image references, a tokenizer or
template difference, any serving/harness mismatch, equal base/post weights, or
parity that created a task or contacted grading. It projects the exact base,
post and matched-runtime bindings required by the existing study-child sealing
gate.

Even a valid certificate does not submit or authorize an evaluation. A separate
reviewed, immutable split-A child must reference it, pass fresh exact task and
route preflight, use a new private result root, and preserve sealed outcome
handling. Evaluation scores remain outside W&B; teacher-reference CE remains
forbidden.
