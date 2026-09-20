# Fresh75 step-230 matched WebExploitBench plan

Date: 2026-09-20

Status: **prepared but not launchable**. No model was resumed, no TensorLake
sandbox was created or changed, and no evaluation was launched while preparing
this plan.

## What will be compared

The comparison changes only the model weights:

- base: `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`;
- candidate: the accepted Fresh75 checkpoint after optimizer step 230. Its
  payload was originally staged under `chris-q38-fresh75-step230-v1`, but the
  matched evaluation must use the new base-cloned route
  `chris-q38-fresh75-step230-wbe-v1`;
- harness: OpenCode;
- benchmark: all 15 WebExploitBench targets at the pinned benchmark and CAGE
  revisions in the sealed plan.

The exact machine-readable plan is
[`configs/evaluation/qwen38-fresh75-step230-opencode-wbe-matched-v1.json`](../../configs/evaluation/qwen38-fresh75-step230-opencode-wbe-matched-v1.json).
Its self-digest is
`sha256:3b37094868cc0dec65411cd552f67424eefbd9e01af1b5035b65bf54f535100d`.

## Two stages, in order

The first stage is a matched canary:

- one benchmark target;
- one rollout from the base model;
- one rollout from Fresh75;
- agent rollouts are saved without a judge call;
- both saved rollouts are scored later under the same judge settings.

The second stage is the full comparison:

- all 15 benchmark targets;
- four rollouts per target and per model;
- 60 rollouts for the base and 60 for Fresh75, 120 total;
- the same collection-first, score-later design.

The full stage cannot begin merely because the canary process exited. It needs
a receipt proving that both rollouts were fully saved, both were fully scored,
and the run was technically valid. The model's benchmark result is not used to
choose the checkpoint; WebExploitBench remains an external reporting benchmark.

## Why rollout collection and scoring are separate

An agent rollout must interact with a live challenge, so it is the expensive
and difficult-to-repeat part. The collector saves that completed interaction as
a read-only bundle and makes no judge call. The scoring command later reads that
same bundle.

This means:

- a judge outage cannot destroy a valid rollout;
- scoring can resume without running the agent again;
- a second judge can inspect the same rollout without changing the model's
  interaction; and
- a lost or uncertain scoring request is held for reconciliation rather than
  causing a duplicate paid call.

The existing `collection_launcher.py`, `collection_pair.py`,
`rollout_bundle.py`, and `deferred_score.py` implement those boundaries. The new
study-plan validator makes the canary-before-full order and duplicate exclusions
explicit.

## The 13 older suspended sandboxes

The read-only capacity audit found 13 suspended TensorLake sandboxes whose names
begin with `q38-s10-web-p4-v1-`. Historical local state identifies them as the
older `q38-t3k262-step10-opencode-web-p4-v1` campaign for the
`teacher3k-262k-step10` checkpoint. They are not Fresh75 work.

Their exact names and provider IDs are preserved in
[`qwen38-q38-s10-lineage-reconciliation-20260920.json`](qwen38-q38-s10-lineage-reconciliation-20260920.json).
They are reserved: the Fresh75 plan uses different campaign and sandbox names,
and no operator may resume, replace, or duplicate those old identities without
a fresh read-only provider check.

At launch time, a new `GET /sandboxes` inventory is mandatory. It must confirm
the current state of all 13 identities and prove that no equivalent Fresh75
base/candidate campaign is active or already accepted. The 2026-09-20 snapshot
is evidence for planning, not permission to launch later.

## Read-only serving discovery

At `2026-09-20T10:08:00Z`, authenticated reads of the serving control plane
showed:

- the shared base route `qwen3.8-27b` was ready with two replicas, base revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, resource version `30388276`,
  and full-spec digest
  `sha256:935294640178869a97ecacb2ac7aa7853a6a7d742f493acc918e0ae589652f0d`;
- the existing Fresh75 registration `chris-q38-fresh75-step230-v1` was safely
  paused with zero replicas and resource version `28721222`; and
- the proposed successor ID `chris-q38-fresh75-step230-wbe-v1` was absent
  (`GET` returned 404).

The live base `/model_info` and `/server_info` endpoints also responded. Their
full read-only response digests were
`sha256:c1c69bc87b0915cab1624d1024392fdeca9741bcd4611973396b3d0550a76a40`
and
`sha256:51fc9534d9edcfa77240b7a43f91c7edf897d3458afb2f6a4b4bdf30245b0fed`.
The server reported context length 262,144, data parallel 8, tensor parallel 1,
KV-cache format `fp8_e4m3`, Qwen reasoning and tool parsers, no weight
quantization, and SGLang `0.5.19`. Raw responses were not committed because
server metadata can contain private operational fields; these projected facts
must be freshly checked against the candidate during live parity.

The old Fresh75 registration must **not** simply be resumed. The shared base
route now runs data parallel 8 / tensor parallel 1 with image digest
`sha256:d6e7288627be8b02be88e4bba38e73f6d50e2826869f753c13a4c4385ab3eda9`,
the Qwen request compatibility shim and three additional runtime flags. The old
candidate registration records data parallel 1 / tensor parallel 8 metadata,
uses an older image and command, and lacks those flags. It also records the
payload-manifest digest as its model revision, while strict collection
provenance expects the accepted export-receipt digest. Resuming it would change
more than weights and invalidate the matched comparison.

The read-only preflight renderer in
`evals/webexploitbench/tensorlake/fresh75_canary_preflight.py` therefore builds a
new **paused c1** registration from the freshly observed base spec. It changes
only the route name, staged model paths, accepted export-receipt identity,
priority and zero-replica paused lifecycle. Its current preview spec digest is
`sha256:cdad3122886c0670f841169d2b2f69eb67d35a03b0c781a7f13847c8557eac19`.
That digest is planning evidence, not create permission: the base must be read
again before registration.

The independent Kubernetes read could not be refreshed in this pass because
the local Nebius login needed renewal. No Kubernetes fact was inferred from the
control-plane response. The operator must refresh access and prove that both
current `InferenceModel` objects match their control-plane reads before creating
or resuming anything.

## Exact remaining blockers

Before the paired canary can be sealed and launched, an authorized operator must
create the one paused successor (only if a fresh GET still says it is absent),
record its new UID in a successor study plan, temporarily serve it, and produce
one fresh receipt proving that:

1. both routes are ready;
2. the candidate route serves the exact accepted step-230 export;
3. tokenizer, chat template, numerical format, context limit and serving runtime
   are identical between the two routes; and
4. identical continuation, forward-generation and tool-call probes pass on
   both models.

The two previously missing local construction paths are now implemented:

- `qwen38_serving_evidence.py` binds the exact checkpoint, export, staging
  acceptance, model lock, new route UID, Deployment/ReplicaSet/Pod chain,
  runtime image and fixed content-free probes. It emits the two existing strict
  receipts and its tests pass those receipts back through the collection
  provenance validator. Raw probe text is discarded in memory and never enters
  either receipt.
- shared snapshot qualification accepts exactly one base project and one
  candidate project, proves that both are score-free OpenCode collection
  projects, and binds them to one qualified filesystem snapshot before the
  normal pair sealer compares the two arm plans.

The serving tool also supplies the bounded release monitor. It may perform one
resource-version-bound `pause` request only when the frozen startup, consumer
handoff, useful-request idle, maximum-runtime or terminal condition requires
release. It then polls until the API reports `paused`, the exact owned Pods are
absent and the derived GPU allocation is zero. It has no register, resume,
retire, TensorLake or evaluation-launch command.

The current process did not contain a TensorLake credential, so a fresh live
`GET /sandboxes` duplicate census was not possible. The 13 old `q38-s10`
sandboxes remain reserved, and the absence of equivalent Fresh75 campaigns must
be re-proved within ten minutes of launch.

Only after the serving proof, shared snapshot qualification and fresh duplicate
check may the exact score-free canary collection plans be sealed. The checked-in
study currently names the safely paused old registration. After the one allowed
successor registration returns a new UID, its model ID and UID must be bound in
a create-once successor study/live-gate artifact before launch; the old model ID
must never be substituted back into a collection plan. This work deliberately
did not register, resume or serve a model and did not create or change
TensorLake work.

## Exact canary arm blueprints

The preflight renderer freezes two concrete but deliberately unsealed arm
blueprints:

| Arm | Served model | Campaign | Sandbox prefix | Work |
|---|---|---|---|---|
| Base | `qwen3.8-27b` | `q38-base-f75p230-oc-wbe-c1-v1` | `q38-b-f75-c1-v1-` | target 0, pass@1, collect only |
| Candidate | `chris-q38-fresh75-step230-wbe-v1` | `q38-f75p230-oc-wbe-c1-v1` | `q38-c-f75-c1-v1-` | target 0, pass@1, collect only |

Both require OpenCode, the same new pair-qualified snapshot, deferred scoring,
fresh serving parity and a fresh duplicate census. They become executable plans
only after `collection_launcher.seal_plan` validates each arm and
`collection_pair.seal_pair` validates the pair.

## Next action

1. Refresh Nebius read access and capture the base and old candidate from both
   the serving control plane and Kubernetes.
2. Read the successor ID again. If it is still absent, create exactly one paused
   c1 registration from the fresh base spec; otherwise reconcile it and do not
   POST again.
3. Record the returned UID in a new sealed study plan, then qualify one shared
   TensorLake snapshot for both arm projects.
4. Reconcile TensorLake inventory read-only, seal the two canary collection
   arms and their pair, and only then resume the candidate so the consumer can
   start immediately.
5. Prove live parity, collect one score-free rollout per arm, pause the candidate
   and prove zero-GPU release, then score the two saved rollouts.

Do not prepare or launch the 15-by-4 stage until the exact canary acceptance
exists.

### Reproducible preflight command

The preflight tool performs no network calls and cannot mutate either system. It
accepts raw serving-control-plane responses plus raw `kubectl get ... -o json`
objects and writes one create-once local receipt:

```bash
uv run python -m evals.webexploitbench.tensorlake.fresh75_canary_preflight build \
  --study-plan configs/evaluation/qwen38-fresh75-step230-opencode-wbe-matched-v1.json \
  --base-api .private/fresh75-wbe/base.api.json \
  --candidate-api .private/fresh75-wbe/candidate.api.json \
  --base-kubernetes .private/fresh75-wbe/base.kube.json \
  --candidate-kubernetes .private/fresh75-wbe/candidate.kube.json \
  --observed-at <UTC-TIMESTAMP> \
  --output .private/fresh75-wbe/preflight.json
```

The output remains blocked by design. It renders the safe successor and the two
collection blueprints; it never registers, resumes, launches, scores or pauses
anything.

### Reproducible Qwen3.8 evidence capture

After the exact successor has been registered once, the collection plans and
consumer are ready to start, and the route has been resumed, capture fresh
serving evidence with:

```bash
uv run python -m evals.webexploitbench.tensorlake.qwen38_serving_evidence capture \
  --study-plan configs/evaluation/qwen38-fresh75-step230-opencode-wbe-matched-v1.json \
  --preflight .private/fresh75-wbe/preflight.json \
  --provenance .private/fresh75-wbe/checkpoint-export-references.json \
  --stage-acceptance docs/evidence/qwen38-fresh75-step230-inference-stage-v2-accepted-20260915.json \
  --model-lock configs/models/qwen38-27b-1d4bf0f2.lock.json \
  --kubernetes-context <PRODUCTION-CONTEXT> \
  --registration-output .private/fresh75-wbe/serving-registration.json \
  --live-parity-output .private/fresh75-wbe/live-parity.json
```

The provenance file contains only absolute local paths and file hashes for the
exact checkpoint and export receipts. The command first verifies that both
routes are ready, then checks model/server metadata before sending its three
fixed non-benchmark probes. Both output paths are create-once. It refuses the
old candidate route, any base/candidate runtime drift, a non-c1 candidate,
changed artifact bytes or a route that is not backed by one exact ready
Deployment/ReplicaSet/Pod chain.

The lifecycle policy is a small private JSON file with schema
`qwen38_checkpoint_serving_lifecycle_v1`. It binds the new model ID and UID,
registration/resume/ready/consumer/useful-request timestamps, terminal outcome
and the five deadline values already frozen in the preflight. A read-only check
is:

```bash
uv run python -m evals.webexploitbench.tensorlake.qwen38_serving_evidence monitor \
  --policy .private/fresh75-wbe/lifecycle.json \
  --kubernetes-context <PRODUCTION-CONTEXT>
```

It exits nonzero when a pause or further release wait is required. Add
`--execute` only for the independently owned monitor that is explicitly
authorized to pause this one exact successor. The pause path uses the resource
version from the immediately preceding GET and never retries an uncertain
write.
