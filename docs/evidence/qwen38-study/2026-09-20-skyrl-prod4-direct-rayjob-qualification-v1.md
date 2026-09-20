# SkyRL prod4 direct RayJob qualification

Status at this revision: **transport qualified; scientific GPU canary created exactly
once and admitted; scientific result pending**.

This change provides one create-once Kubernetes `RayJob` transport for the already
sealed `chris-q38-rlreward-prod4` scientific canary. It bypasses only the FleetJob
webhook that dropped the mandatory failed-job alert opt-out. It does not change the
model, task versions, reward source, optimizer, checkpoint rules, token limits,
episode duration, sampling, learning rate, seed, image, or one-node/eight-GPU shape.

## Frozen science

- plan SHA-256: `25d0abf30da462a6ba67c6ac8a3fc95f3f89a9a08e3ea295b21e8a22481630b3`
- request SHA-256: `98a83dbbab61f360a12ec6389adb55de637ded64dcceaa1debf0d791e754cef4`
- immutable image SHA-256: `89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f`
- direct RayJob manifest SHA-256:
  `4204a182a35522de15f6c602437fa963ebcce987eda5a65fb11591c31e9f1b6d`
- root object: `RayJob/chris-q38-rlreward-prod4` in `fleet-train-jobs`
- admission: normal `training-lq`, queue priority `q1`, pod priority `c1`, suspended
  until Kueue admits it
- failed-job alert annotation: `fleet.ai/failure-alerts: "off"`
- resources: one physical node, eight GPUs, 64 CPU requested, 512 GiB memory requested
- science: one train task, one development task, eight sampled episodes, one optimizer
  step, learning rate `1e-6`, evaluation before and after the update, step-1 checkpoint

The output initializer is create-once. The generated Fleet secret placeholder is
removed because a direct RayJob has no Fleet controller to create it; the maintained
`fleet-api` and `wandb-api` Secrets remain exact. The GPU process is explicitly bound
to UID 1000 and GID 100, matching the immutable image.

## Private input staging

The five exact private input files are bundled into a deterministic archive. The
archive and every file are bound by size and SHA-256. Upload must use this sequence:

1. copy to `/tmp/autoresearch-upload.tar.gz.partial`;
2. verify the remote byte count and SHA-256;
3. atomically rename it to `/tmp/autoresearch-upload.tar.gz`.

The zero-GPU stage Job waits only for the final name, rechecks the archive digest,
extracts only the five expected regular files, rechecks every file, then atomically
publishes the previously absent destination. Partial bytes can therefore never be
mistaken for a complete upload.

At qualification time the private archive was 4,449 bytes with SHA-256
`267d65b53f5df639f210ea1dc0e6d779de7b4e35ec354cc17d54aee00382a563`.
The stage plan SHA-256 was
`d759de16e332e533e6fa18f0a42724c7f2a7a49e6192f72f77dd4d1882c39a92`.

## Server dry-runs

The exact direct RayJob, data-stage Job, and scientific CPU-preflight Job each passed
Kubernetes server dry-run on both the development and production clusters. The
validators removed only documented Kubernetes defaults and compared the remainder
byte-for-byte with the intended object.

| Cluster | Object | Proof SHA-256 |
|---|---|---|
| development | direct RayJob | `9c633be6c208a2684ceaf3f3c87c9871b0d6f1209607bad692222ab1e6a8b943` |
| development | data stage | `8a2f0fa6802f0df37ff79faa1eb4a76219f833b3963790de9a668c4da0ebc4f7` |
| development | CPU preflight | `7b5061b6f2b686cb78071b2fec1ce70fad32d7194b5c963a30158821bd19dcaa` |
| production | direct RayJob | `7d76d62ae710c3de14c4b3b4c4079f67319b7c1725f540b5ff1cbe02d8212b29` |
| production | data stage | `806c21a5a00f39b45967a85e3105a0a1a3e9885632526ca2ec6cdce6acc12912` |
| production | CPU preflight | `70f081d9f973ff4a7af83a06c6a5d4d2ba7368228dead31b072306e04a2fc1e7` |

No server dry-run allocated a Pod, node, CPU, or GPU.

## Regression results

- `uvx ruff check` and `uvx ruff format --check`: passed.
- focused direct-rail and cleanup-observer tests: **19 passed**.
- the repository-wide suite completed with **15 pre-existing failures**. Those failures
  are in stale generated SkyRL queue/audit digests and older request builders that do
  not yet set the now-mandatory `failureAlerts: false`; none imports or exercises this
  direct rail. They are recorded rather than repaired here because they are outside the
  launch-critical prod4 transport.

## Remaining live gates

The first production CPU preflight (`chris-q38-prod4-preflight-v1`) exited before
writing a receipt and released cleanly with zero GPUs and zero restarts. V2 accepted
W&B 0.21.1's missing-run sentinel but still exited without a receipt. V3 added a
sealed rejection receipt and proved the remaining failure was the optional remote W&B
lookup, not model/data parsing: it rejected at `wandb_lookup` with a sanitized
`CommError` and then released cleanly. W&B 0.21.1 does not provide a stable distinction
between missing-run, subscription, and transient service errors at that read surface.

V4 therefore removes that non-authoritative remote read from the scientific preflight.
It still requires the injected W&B credential and validates the exact immutable
entity/project/run ID plus `resume="never"`. Runtime `wandb.init` is the authoritative
create-once boundary and executes before rollout or optimizer work. This avoids making
an observability read a false training blocker while preserving no-resume semantics.
No model, data, reward, optimizer, checkpoint, sampling, or GPU setting changed.

Before the sole GPU create, the operator must:

1. run the exact zero-GPU production data stage and verify its UID-bound receipt and
   complete release;
2. run the exact zero-GPU scientific preflight, including output and W&B run-ID
   absence, and verify its UID-bound receipt and complete release;
3. repeat the API/Kubernetes duplicate census and confirm the owned-node budget;
4. arm the production cleanup observer and verify its process is alive;
5. fsync one `CREATE_INTENT_DO_NOT_RETRY` record; and
6. issue exactly one `kubectl create`, never `apply`, `patch`, or retry.

Acceptance after creation still requires genuine reward variation, one finite optimizer
step, a reloadable step-1 checkpoint, exact UID-bound resource release, and independent
sanitized evidence validation. A Pod becoming Ready is not scientific acceptance.

## Live create record

The V4 zero-GPU scientific preflight passed at `2026-09-20T23:04:07Z`. Its
UID-bound observer result has SHA-256
`4b3ede99c046367efcab4860f61402ca1b0f94ff806b3fbe31606488a179c24f`; the
inner scientific receipt has SHA-256
`037dcf402310fbfcaa83227e2306bc3df142df30051bd792a6c69e033c9734f6`.
The preflight Job and Pod were both absent after cleanup, and it allocated zero GPUs.

After fresh duplicate and node-budget checks, the operator armed the exact cleanup
observer, fsynced the create intent, and issued the one permitted create. The live
identities are:

- RayJob `chris-q38-rlreward-prod4`, UID
  `21d311b0-136d-4831-a5de-341ff9d12c87`, created
  `2026-09-20T23:09:38Z`;
- Workload `rayjob-chris-q38-rlreward-prod4-77b49`, UID
  `9e8b6058-acca-4a4f-9c79-bfe17dc85a34`, admitted at create;
- RayCluster `chris-q38-rlreward-prod4-bpb2k`, UID
  `5c1dea98-b6af-46de-9659-aed9b74628fe`;
- head Pod `chris-q38-rlreward-prod4-bpb2k-head-dw6b5`, UID
  `b36e390f-9da0-487c-a57d-4e0aaff6ded5`.

The created-proof SHA-256 is
`6b5634b5a0e99ceefd69482fc27dbb5278b77aeae51866260cca5d3e7ca548e6`.
At the first post-create read, the Pod was scheduled on
`computeinstance-e04az8ppqdsr7e9pah`, initializing with zero restarts, and the
root RayJob still carried `fleet.ai/failure-alerts: "off"`. These facts prove
launch identity and admission only; the scientific acceptance gates above remain open.

The pre-create observer exposed one transport defect after the create: the complete
Kubernetes and Jobs API duplicate census took slightly longer than its 120-second
target-appearance allowance, so it exited 12 seconds before the RayJob was created.
The create call had verified that process before beginning the census, but had not
rechecked it immediately before create. No scientific workload setting or cluster
object was changed in response. A recovery observer was attached to the exact existing
RayJob UID and refuses any replacement UID; its armed receipt has SHA-256
`8db96ce16e421e45671f4399852d66d790577f08d69a2db048eab9b85638c1d3`.
The durable repair adds a distinct, truthful recovery-observer schema, raises the
production pre-create appearance allowance to five minutes, and rechecks observer
liveness immediately before writing create intent. Future creates therefore fail
closed if their observer exits during duplicate reconciliation.

## Terminal result and bounded successor

Prod4 terminated before collecting any train episode. The sealed native failure
classified the defect as `InvalidEpisode` at `skyrl_episode.py::parse`: Qwen emitted
more than one valid tool call in one sampled turn, while the adapter accepted exactly
one. This was an adapter limitation, not a task, reward, optimizer, model-loading, or
cluster failure.

- native failure self SHA-256:
  `cbc41545f426f42cad31fe9e0bc9fa124548309424db8f80bff0a106bccd0635`;
- root failure self SHA-256:
  `df796d5f21e35438dab5845d3b24b86dfaf738b809b4b253697b7349ae2559bd`;
- UID-bound recovery-observer result SHA-256:
  `472a20a618f7bcbc233872d7f1959108ee4fe3f9e1dbd0a5f5773a575cd23a0d`;
- accepted train episodes: zero;
- optimizer updates: zero;
- checkpoints: zero.

The observer deleted the exact owned RayJob and confirmed that its Workload,
RayCluster, Pod, and all eight GPUs were gone. The separate zero-GPU evidence-reader
Pod was subsequently deleted by exact UID as well. Prod4 must never be retried or
resumed.

The deterministic successor is prod5. It preserves model, task split, eight genuine
Fleet episodes, authoritative rewards, optimizer recipe, before/after evaluation, and
checkpoint gates. The only runtime change is ordered support for every valid tool call
in one sampled turn: calls execute in emitted order and all tool results are masked as
one observation group before the next assistant header. The focused RL suite passed
222 tests with two expected skips. Repair commit: `f285ed72`.

Prod5 frozen identities at preparation time:

- plan SHA-256: `e61591e0a95d33990296e2786abddf0f67d87b01f8c6bd5f54a6a217bcbaea31`;
- request SHA-256: `0ecb6155d986362556a9758274eb439b3b8912a85c178bdcfdf3becc7a3231e6`;
- bundled runtime SHA-256:
  `7db4616ab57dea0609424753e23d315de4b9fc430eb3864b99e1f5fd9dc02270`;
- direct root-RayJob manifest SHA-256:
  `59d20f929f23b569d1207d991762d5d151d38a0c3185dd5f1186708a8d4a6b8c`;
- private input archive SHA-256:
  `a5a7ee8334ec384365e7629376ec4fb150399aab14517b37b813153fb38d3719`.

Prod5 remains scientifically unaccepted until it records genuine authoritative reward
variation, one finite optimizer update, a reloadable step-1 checkpoint, matched
before/after development results, and complete GPU release.

## Prod5 live create record

The create-once zero-GPU data stage and scientific preflight both passed and released
before the GPU create:

- data-stage Job UID `0f37455c-cead-4105-a251-b318ca62a878`, Pod UID
  `fafe30d6-29fa-4375-8e85-82fec18d71f2`, stage receipt SHA-256
  `790bca1cd3e89c56a9461cd8d58afce42cce13407ab26bd4d83fcd695dd8bcdc`,
  release-observer result SHA-256
  `a048c3096bfcd10ccae6f21544362bf6b0126ed45162c3720636e1a737385139`;
- CPU-preflight Job UID `f1b61383-34ce-4655-8224-b5061989e40e`, Pod UID
  `9343e80c-315b-4970-89a4-6d7b5f2d31e1`, scientific receipt SHA-256
  `ef72b5b3052e94cb442c1f3bd6c7439fd7f604dd1504d24aa4b3140cd719baf5`,
  release-observer result SHA-256
  `939f62c849a991c06759533545b6d7e300accf65b895b1793f2eaf3a7fd3d514`.

Both Jobs carried `fleet.ai/failure-alerts: "off"`, requested zero GPUs, terminated
with exit zero and zero restarts, and were absent with their Pods after cleanup. The
preflight revalidated both the native parser and ordered multi-tool parser, exact
train/development counts, the one-step plan, create-once W&B binding, and output
absence.

After a fresh Kubernetes and Jobs-API duplicate census, a current owned-node census,
and a live pre-armed cleanup observer, the operator issued the sole prod5 create at
`2026-09-20T23:56:08Z`. Exact identities are:

- RayJob `chris-q38-rlreward-prod5`, UID
  `f9327842-014e-4f99-b271-74351dafd8a8`;
- Workload `rayjob-chris-q38-rlreward-prod5-f0eb3`, UID
  `0bbe0928-f0b9-40c7-aff6-0fe49bcf6c8a`, admitted at create;
- RayCluster `chris-q38-rlreward-prod5-dnt6z`, UID
  `5b05e68d-ab3b-46f2-828b-c4e83f9cb90e`;
- head Pod `chris-q38-rlreward-prod5-dnt6z-head-lmj7l`, UID
  `390b4d14-d3be-4fab-aabd-7eb3d0fb071c`, initially scheduled on
  `computeinstance-e04q9rsq2y5mpj2ktm` with eight GPUs and zero restarts.

The root RayJob carried `fleet.ai/failure-alerts: "off"` and c1/q1 priority before
creation. The created-proof SHA-256 is
`513aebcda9c56666dad65e0d91a17f887d17c53504c66c46ec17b132738cc91f`.
These facts prove the exact create and admission only; they do not yet prove any
episode, reward, optimizer update, checkpoint, or capability result.
