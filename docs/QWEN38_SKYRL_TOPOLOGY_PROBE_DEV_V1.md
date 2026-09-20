# Qwen3.8 SkyRL development topology probe

This is an operational check, not training. It answers one narrow question: can
the exact SkyRL image load the pinned Qwen3.8-27B model and start two tensor-parallel
inference engines on one eight-GPU development node, then stop them and release the
allocation cleanly?

The historical `v16` execution reached `SUCCEEDED` with zero restarts and
released all eight GPUs. Its cleanup observer did not capture the public
container receipt before teardown, so it proves execution and release but does
**not** pass this gate. The exact evidence is sealed in
`docs/evidence/qwen38-study/2026-09-20-skyrl-topology-probe-v16-qualification-summary.json`.

The create-once `v17` successor is prepared but has not been submitted. It keeps
`VLLM_USE_FLASHINFER_SAMPLER=0` and adds a fixed 30-second grace period for the
terminal receipt before cleanup. There is no failure-count gate. The development
FleetJob admission interface rejects the mandatory failed-job-alert opt-out
annotation, so `v17` now has one maintained direct-RayJob qualification transport.
That transport passed a non-creating development server dry-run on 2026-09-20.
No RayJob, FleetJob, Job, Pod, Workload, RayCluster, or GPU allocation was created.

It deliberately cannot read a training row, create an episode, call a verifier,
take an optimizer step, or write a checkpoint. Passing it does not qualify RL.
The separate 2,400-second scientific reward canary is unchanged and remains closed
until its own reward, optimizer, checkpoint and cleanup gates pass.

## Frozen shape

- next direct RayJob identity: `chris-q38-skyrl-probe-v17`
- development context: `nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb`
- namespace/project: `fleet-train-jobs` / `fleetjob-dev`
- priority: Kubernetes `c1`, queue `q1`
- image: `fleet/skyrl-train` at the exact digest in the config
- topology: one Ray head requesting all eight GPUs. The exact zero-replica worker
  group from the accepted `v17` embedded RayJob remains present, owns no Pod and
  owns no GPU
- Kueue topology: the head and dormant `gpu` worker group are explicitly q1. The
  runtime refuses acceptance unless Ray sees exactly one live GPU node with exactly
  eight GPUs before either TP4 engine is accepted
- GPU-head resources: 64 CPU / 512 GiB requested, 64 CPU / 768 GiB limited, plus
  eight GPUs. These CPUs must also be advertised to Ray so the two engine placement
  groups are schedulable
- dormant worker template: 2 CPU / 4 GiB requested, 4 CPU / 8 GiB limited, with
  zero replicas
- runtime identity on the GPU Pod: UID 1000, GID 100
- input model: read-only Fleet artifact
  `fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4`, mounted directly at `models/base`.
  The artifact is a create-once, hardlink-preserving staging of exact revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; the CPU and GPU gates verify all 28
  files against the exact size and SHA-256 inventory before engine startup
- zero-GPU preflight source:
  `models/fleetjob-dev/qwen38-27b-1d4bf0f2-skyrl-v4` is mounted directly at that
  same immutable runtime root. This exact PVC subpath must equal `models/` plus
  the FleetJob artifact path
- output: an init container in the same Ray head Pod creates the exact run-owned
  `models/fleetjob-dev/chris-q38-skyrl-probe-v17/models/run` SFS directory with
  `exist_ok=false`; only the sanitized probe receipt is accepted there. The main
  container mounts that create-once directory at the plan's unchanged `models/run`
  path
- vLLM sampler: `VLLM_USE_FLASHINFER_SAMPLER=0`; the exact-image CPU gate proves
  vLLM parses this as false and selects its native sampler rather than the
  unavailable FlashInfer top-k/top-p kernel on B300
- W&B: disabled; no task, benchmark or W&B credential is delivered
- process limit: 20 minutes for exact model verification and engine setup, then five
  minutes for engine cleanup
- RayJob limit: 30 minutes; an independent observer must still delete the exact
  root RayJob no later than 30 minutes after it was created
- retries: none

The probe reserves exactly one physical GPU node and all eight GPUs on its Ray head.

## Prepare and validate without creating a workload

Use a new directory. Preparation and server dry-run make no cluster object:

```sh
uv run --locked cyber-post-train rl-topology-probe \
  configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json \
  --output /absolute/new/probe-packet
uv run --locked cyber-post-train rl-topology-probe-preflight-preview \
  /absolute/new/probe-packet
uv run --locked cyber-post-train rl-topology-probe-preview \
  /absolute/new/probe-packet
uv run --locked cyber-post-train rl-topology-probe-receipt-preview \
  /absolute/new/probe-packet
uv run --locked cyber-post-train rl-topology-probe-rayjob-preview \
  /absolute/new/probe-packet
```

The last four commands use `kubectl create --dry-run=server` against the context and
namespace sealed into the plan. It fails if the server changes the image, model
mount, topology, resources, priority, deadline, command, environment or security
settings. It saves only a sanitized proof; the server response containing the
runtime bundle is not retained in the packet.

The direct renderer starts from the `v17` FleetJob's exact embedded RayJob, then
adds only the platform bindings that the FleetJob controller would otherwise add:
the existing read-only model artifact, the create-once output directory, Fleet
authentication by Secret reference, normal GPU node selection, and normal Kueue
admission. The root RayJob carries `fleet.ai/failure-alerts: "off"`, pod priority
`c1`, and queue priority `q1`. It never uses `apply`, `patch`, manual unsuspension,
or a retry loop.

The legacy Jobs API is intentionally not an alternate route. Its preview cannot
prove UID 1000/GID 100, so this profile rejects it. Moving the probe to production
or changing its topology requires a new config, plan and packet.

## Gates before the one allowed creation

Do not create the root RayJob until every item is true:

1. The source commit is clean, reviewed, pushed, and exactly matches the source
   used to prepare the packet.
2. `PREPARED.json`, `DIRECT_RAYJOB_PREPARED.json`, and
   `DIRECT_RAYJOB_PREVIEW.json` all validate and bind the same plan, source
   FleetJob, and direct RayJob digests.
3. A CPU-only Kubernetes Job has run in the exact image as UID 1000/GID 100 with
   zero requested GPUs, the exact read-only model mount, and a separate read-only
   view of the output registry. It must verify the runtime imports,
   native engine arguments, all model-file sizes and SHA-256 digests, prove the
   future RayJob output path is absent, and record zero visible GPUs. Its
   sanitized result comes only from the Kubernetes termination message; logs are
   never an evidence source.
4. The exact name and output are absent from every development and production
   RayJob, FleetJob, Job, and both Jobs API histories. The output destination
   contains no previous probe receipt, runtime directory, episode, checkpoint or
   export. The main Pod's init container repeats the create-once directory check.
5. The direct development RayJob server dry-run passes again immediately before
   creation.
6. An independent observer is already running. It has the exact context,
   namespace and name; after creation it records the RayJob UID, Workload UID,
   RayCluster UID and head Pod UID. It is authorized to delete only that exact
   RayJob.
7. The observer has a fixed deletion time no later than 30 minutes after RayJob
   creation. After terminal status it waits at most 30 seconds for the public
   receipt, then deletes; it also deletes at the fixed deadline for a stalled setup.
8. The operator has confirmed that adding eight development GPUs remains inside
   the current experiment-owned resource allowance.

The checked-in config always keeps `submission_authorized` false. Passing it does
not authorize a create by editing the plan. Instead, the launch command requires a
separate create-once authorization that embeds the released CPU result, all
three server-preview receipts and the already-armed GPU observer. Do not bypass
that evidence gate with a manual `kubectl create`.

Arm the independent observer before creating either Job. The foreground process
writes `ARMED.json` only after proving the exact name is absent. It then binds the
first observed UID, deletes only that UID after success, failure or the deadline,
and writes `RESULT.json` only after the workload and its recorded children are
absent:

```sh
uv run --locked python -m training.dev_cleanup_observer \
  --context nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb \
  --namespace fleet-train-jobs \
  --kind job \
  --name chris-q38-skyrl-probe-preflight-v29 \
  --maximum-seconds 1200 \
  --expected-gpus 0 \
  --plan-sha256 sha256:<exact-plan-digest> \
  --manifest-sha256 sha256:<exact-preflight-manifest-digest> \
  --armed /absolute/new/PREFLIGHT_OBSERVER_ARMED.json \
  --result /absolute/new/PREFLIGHT_OBSERVER_RESULT.json
```

For the GPU probe use `--kind rayjob`, the direct RayJob name and manifest digest,
`--maximum-seconds 1800`, and `--expected-gpus 8`. Keep that process alive before
the single create. Then authorize and create through the checked command surface:

```sh
uv run --locked cyber-post-train rl-topology-probe-rayjob-authorize \
  /absolute/new/probe-packet \
  --cpu-result /absolute/PREFLIGHT_OBSERVER_RESULT.json \
  --observer-armed /absolute/GPU_OBSERVER_ARMED.json
uv run --locked cyber-post-train rl-topology-probe-rayjob-create \
  /absolute/new/probe-packet
```

The create command revalidates every nested digest, confirms the observer process
is still alive, checks the exact name and output across both Kubernetes clusters
and both Jobs API histories, repeats the development server dry-run, fsyncs a
create-intent journal and its parent directory, then executes exactly one
non-retried `kubectl create`. It records the created UID. A missing or invalid
sanitized termination receipt makes the gate fail even when cleanup succeeds.
When Ray becomes terminal, the observer gives Kubernetes at most 30 seconds to
expose the declared public termination-message file before it performs the same
unconditional exact-UID delete. This fixes the receipt race observed on `v16`
without extending the 30-minute allocation bound. A separate zero-GPU, read-only
Job must still validate the durable SFS receipt after release; execution status
alone is never acceptance.

## Evidence while it runs

Record only sanitized operational facts:

- the exact object UIDs listed above;
- actual image ID, Pod node, priority class, restart count and eight-GPU request
  on the Ray head;
- timestamps for creation, admission, Pod readiness, process start and finish;
- the sealed `TOPOLOGY_PROBE.json` digest and its scalar fields; and
- whether the head Pod, RayCluster or Workload remains after deletion.

Do not read or publish raw model weights, the private engine log, runtime bundle,
credentials, task text, prompts, trajectories or benchmark data.

## Acceptance

The operational probe passes only when all of these are true:

- the GPU Pod ran as UID 1000/GID 100 with the exact image and zero restarts;
- the model mount matched the complete exact-revision inventory;
- exactly one live Ray GPU node exposed exactly eight GPUs;
- two inference-engine groups started, each with tensor parallel size four;
- the receipt says zero task rows, episodes, verifier calls, optimizer steps and
  checkpoints;
- internal Ray/engine cleanup completed within 25 minutes; and
- the external release proof below passes.

Anything else is a failed operational gate, never a zero-reward model result and
never permission to launch the scientific canary.

## Release proof

At terminal status—or at the fixed deadline—the observer waits no more than 30
seconds for the public termination receipt, then deletes the exact RayJob by
name only after confirming its recorded UID. It then waits until the RayJob,
Workload, RayCluster and head Pod identities it recorded are all absent. It also
verifies that the exact allocation uses zero GPUs.

The sanitized observation supplied to `validate_release` must contain exactly:

```json
{
  "schema": "cyber_dev_cleanup_observer_result_v1",
  "status": "released",
  "context": "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
  "namespace": "fleet-train-jobs",
  "kind": "rayjob",
  "name": "chris-q38-skyrl-probe-v17",
  "uid": "<RayJob UID>",
  "rayjob_name": "chris-q38-skyrl-probe-v17",
  "rayjob_uid": "<RayJob UID>",
  "workload_name": "<Workload name>",
  "workload_uid": "<Workload UID>",
  "raycluster_name": "<RayCluster name>",
  "raycluster_uid": "<RayCluster UID>",
  "pod_names": ["<head Pod name>"],
  "pod_uids": ["<head Pod UID>"],
  "expected_gpus": 8,
  "peak_gpus": 8,
  "restarts": 0,
  "terminal_status": "Succeeded",
  "receipt": {"status": "setup_and_internal_cleanup_passed"},
  "observer_error_class": "",
  "target_present": false,
  "rayjob_present": false,
  "workload_present": false,
  "raycluster_present": false,
  "pods_present": false,
  "active_gpus": 0,
  "created_at": "<UTC RFC3339 creation time>",
  "deletion_requested_at": "<UTC RFC3339 deletion time>",
  "release_observed_at": "<UTC RFC3339 absence-confirmation time>"
}
```

The validator accepts only `Succeeded`; it rejects deletion later than 30
minutes after the recorded creation time, any non-UUID resource identity, a
receipt from a different plan, any restart, failure to reach an eight-GPU peak,
or any recorded resource that remains present. A failed or deleted run can still
produce useful cleanup evidence, but it cannot pass the probe. Preserve that
sanitized failure evidence, do not create an automatic successor, and diagnose
off-cluster before proposing a new immutable identity.
