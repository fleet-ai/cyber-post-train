# Qwen3.8 SkyRL development topology probe

This is an operational check, not training. It answers one narrow question: can
the exact SkyRL image load the pinned Qwen3.8-27B model and start two tensor-parallel
inference engines on one eight-GPU development node, then stop them and release the
allocation cleanly?

It deliberately cannot read a training row, create an episode, call a verifier,
take an optimizer step, or write a checkpoint. Passing it does not qualify RL.
The separate 2,400-second scientific reward canary is unchanged and remains closed
until its own reward, optimizer, checkpoint and cleanup gates pass.

## Frozen shape

- FleetJob name: `chris-q38-skyrl-probe-v1`
- development context: `nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb`
- namespace/project: `fleet-train-jobs` / `fleetjob-dev`
- priority: Kubernetes `c1`, queue `q1`
- image: `fleet/skyrl-train` at the exact digest in the config
- topology: one CPU-only Ray head plus one worker Pod requesting all eight GPUs on
  one node
- head resources: 4 CPU / 16 GiB requested, 8 CPU / 32 GiB limited
- worker resources: 64 CPU / 512 GiB requested, 64 CPU / 768 GiB limited
- runtime identity on both Pods: UID 1000, GID 100
- input model: read-only Fleet model mount `Qwen/Qwen3.8-27B` at `models/base`;
  every file is checked against the exact repository revision and SHA-256 inventory
  before engine startup. Standard Hugging Face symlinks are allowed only when they
  resolve to regular files whose bytes match those exact digests
- output: the controller-created, run-owned `models/run` directory; only the sanitized probe receipt is
  accepted there
- W&B: disabled; no task, benchmark or W&B credential is delivered
- process limit: 20 minutes for exact model verification and engine setup, then five
  minutes for engine cleanup
- RayJob limit: 30 minutes; an independent observer must still delete the FleetJob
  no later than 30 minutes after the FleetJob was created
- retries: none

The CPU head does not count as a GPU node. The probe reserves exactly one GPU node
and eight GPUs.

## Prepare and validate without creating a workload

Use a new directory. Preparation and server dry-run make no cluster object:

```sh
uv run --locked cyber-post-train rl-topology-probe \
  configs/qualification/qwen38-skyrl-topology-probe-dev-v1.json \
  --output /absolute/new/probe-packet
uv run --locked cyber-post-train rl-topology-probe-preflight-preview \
  /absolute/new/probe-packet
uv run --locked cyber-post-train rl-topology-probe-preview \
  /absolute/new/probe-packet
```

The last two commands use `kubectl create --dry-run=server` against the context and
namespace sealed into the plan. It fails if the server changes the image, model
mount, topology, resources, priority, deadline, command, environment or security
settings. It saves only a sanitized proof; the server response containing the
runtime bundle is not retained in the packet.

The legacy Jobs API is intentionally not an alternate route. Its preview cannot
prove UID 1000/GID 100, so this profile rejects it. Moving the probe to production
or changing its topology requires a new config, plan and packet.

## Gates before the one allowed creation

Do not create the FleetJob until every item is true:

1. The source commit is clean, reviewed, pushed, and exactly matches the source
   used to prepare the packet.
2. `PREPARED.json`, `FLEETJOB_PREPARED.json`, and `FLEETJOB_PREVIEW.json` all
   validate and bind the same plan and manifest digests.
3. A CPU-only Kubernetes Job has run in the exact image as UID 1000/GID 100 with
   zero requested GPUs, the exact read-only model mount, and a separate read-only
   view of the FleetJob output registry. It must verify the runtime imports,
   native engine arguments, all model-file sizes and SHA-256 digests, prove the
   future FleetJob output path is absent, and record zero visible GPUs. Its
   sanitized result comes only from the Kubernetes termination message; logs are
   never an evidence source.
4. The exact name is absent from both development and production FleetJobs and
   from both legacy Jobs APIs. The output destination contains no previous probe
   receipt, runtime directory, episode, checkpoint or export.
5. The development server dry-run passes again immediately before creation.
6. An independent observer is already running. It has the exact context,
   namespace and name; after creation it records the FleetJob UID, Fleet job ID,
   RayJob UID, Workload UID, RayCluster UID and both Pod UIDs. It is authorized to
   delete only that exact FleetJob.
7. The observer has a fixed deletion time no later than 30 minutes after FleetJob
   creation. It deletes sooner on success, failure, irrelevance or a stalled setup.
8. The operator has confirmed that adding eight development GPUs remains inside
   the current experiment-owned resource allowance.

The checked-in config keeps `submission_authorized` false while items 3 and 6 are
open. Do not bypass that closed gate with a manual `kubectl create`.

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
  --name chris-q38-skyrl-probe-preflight-v6 \
  --maximum-seconds 1200 \
  --expected-gpus 0 \
  --plan-sha256 sha256:<exact-plan-digest> \
  --manifest-sha256 sha256:<exact-preflight-manifest-digest> \
  --armed /absolute/new/PREFLIGHT_OBSERVER_ARMED.json \
  --result /absolute/new/PREFLIGHT_OBSERVER_RESULT.json
```

For the GPU probe use `--kind fleetjob`, the FleetJob name and manifest digest,
`--maximum-seconds 1800`, and `--expected-gpus 8`. Keep that process alive before
the single create. A missing or invalid sanitized termination receipt makes the
gate fail even when cleanup succeeds.

## Evidence while it runs

Record only sanitized operational facts:

- the exact object UIDs listed above;
- actual image IDs, Pod nodes, priority classes, restart counts and eight-GPU
  request on the worker;
- timestamps for creation, admission, Pod readiness, process start and finish;
- the sealed `TOPOLOGY_PROBE.json` digest and its scalar fields; and
- whether either Pod, the RayCluster or the Workload remains after deletion.

Do not read or publish raw model weights, the private engine log, runtime bundle,
credentials, task text, prompts, trajectories or benchmark data.

## Acceptance

The operational probe passes only when all of these are true:

- both Pods ran as UID 1000/GID 100 with the exact image and zero restarts;
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

At terminal status—or at the fixed deadline—the observer deletes the exact
FleetJob by name only after confirming its recorded UID. It then waits until all
of the following recorded objects are absent: FleetJob, RayJob, Workload,
RayCluster, head Pod and GPU-worker Pod. It also verifies that the exact allocation
uses zero GPUs.

The sanitized observation supplied to `validate_release` must contain exactly:

```json
{
  "kubernetes_context": "nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb",
  "namespace": "fleet-train-jobs",
  "fleetjob_name": "chris-q38-skyrl-probe-v1",
  "job_id": "<Fleet job UUID>",
  "fleetjob_uid": "<FleetJob UID>",
  "rayjob_uid": "<RayJob UID>",
  "workload_uid": "<Workload UID>",
  "raycluster_uid": "<RayCluster UID>",
  "pod_uids": ["<head Pod UID>", "<GPU-worker Pod UID>"],
  "terminal_status": "Succeeded",
  "fleetjob_present": false,
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

The validator rejects deletion later than 30 minutes after the recorded creation
time, any non-UUID resource identity, a receipt from a different plan, or a
resource that remains present. `Failed` or observer-classified `Deleted` is
allowed only so cleanup can be proved; it does not make the probe accepted.
Preserve the sanitized failure evidence, do not create an automatic successor,
and diagnose off-cluster before proposing a new immutable identity.
