# SkyRL direct-RayJob qualification transport — v17

## Outcome

The maintained direct root-RayJob transport passed a real, non-creating
Kubernetes server dry-run on the development cluster on 2026-09-20. The server
accepted the exact `chris-q38-skyrl-probe-v17` identity, one-node/eight-GPU
topology, `c1` Pod priority, `q1` queue priority and root
`fleet.ai/failure-alerts: "off"` annotation.

The server preview created no cluster object. A later exact CPU preflight did
create one zero-GPU Job and Pod, captured below, and released both. No RayJob,
FleetJob, Workload, RayCluster or GPU allocation was created. Read-only
inventories on both the development and production clusters confirmed that the
exact `v17` identity remained absent. This evidence qualifies only the transport
and CPU gate; it does not claim that the GPU topology probe, reward canary or RL
training ran.

## Why this transport exists

The accepted v17 plan already contained the desired RayJob. The development
FleetJob admission interface rejected the required failed-job-alert opt-out on
that embedded object. The direct launcher therefore begins with the exact
embedded RayJob and adds only the cluster bindings normally supplied by the
FleetJob controller: normal GPU placement, the existing read-only model volume,
the create-once output volume, the Fleet secret reference and Kueue admission.

The plan, image, entrypoint, runtime bundle, model, tensor-parallel topology,
resource requests, deadlines and zero-scientific-work contract are unchanged.

## Immutable bindings

- plan SHA-256: `b398e73ba3124c4f22e53aceb9ac7924916ba2a3027a3f780261d184505ae678`
- source FleetJob manifest SHA-256:
  `0f8c7f3f5a6772fd4aaef706d8d2a638e6e7b24cb87edc6c3b0937d95bd6f389`
- direct RayJob manifest SHA-256:
  `761f249e9a050cb96503a927113ae0faff5df5917c652d97c71ee6d60c22d00c`
- server-rendered object SHA-256:
  `7c9ce6eb715942283c1811b40a2f79a45593b8a08afe989b8a4ac38f8add245a`
- sealed preview receipt SHA-256:
  `6735d21ce28de8687ab1bc8025dbc6def04b23bb508cb29256844063ef891bdf`
- sanitized receipt:
  [`2026-09-20-skyrl-direct-rayjob-preview-v17.json`](./2026-09-20-skyrl-direct-rayjob-preview-v17.json)

## Maintained safety properties

The checked command surface now enforces all of the following before its single
allowed create:

1. It regenerates the direct object from the sealed v17 plan and rejects byte
   drift in the source or projected manifests.
2. It requires a successful, released, exact-image CPU preflight and both CPU
   server previews.
3. It requires a fresh direct-RayJob server preview that retains the root alert
   opt-out and exact `c1`/`q1` scheduling.
4. It requires the UID-binding cleanup observer to be armed and alive first.
5. It checks the exact name, run directory, Fleet mount root and output subpath
   across RayJobs, FleetJobs and Jobs on both clusters, plus both Jobs API
   histories.
6. It performs one final server dry-run, then creates and fsyncs a durable intent
   journal and its parent directory.
7. It issues exactly one `kubectl create`. It never uses `apply`, `patch`, a
   manual unsuspend operation or a retry loop. Any ambiguous result must be
   reconciled from the journal rather than replayed.
8. Its direct observer binds the RayJob, Workload, RayCluster and Pod UIDs,
   deletes only the recorded RayJob UID and accepts release only after every
   recorded resource is gone and the active GPU count is zero.

Regression tests cover projection fidelity, preview drift, nested authorization,
duplicate rejection, journal-before-create ordering, no retry, direct observer
UID binding, terminal receipt capture and strict success/release acceptance.

## Accepted CPU preflight

The exact create-once CPU preflight `chris-q38-skyrl-probe-preflight-v29`
completed successfully and was deleted by its independent observer:

- final development / production server-preview receipts:
  `sha256:e10c17a7a5689de78bad20b35048c77da8b9fefde401e09952cc8c5095c7ab` /
  `sha256:384974aba4d142ac33fc0d646d72a83c79d0f688394955af809b41f202005a7f`
- duplicate check: the exact direct identity and output were absent across all
  RayJobs, FleetJobs and Jobs on both clusters and 1,093 Jobs API history rows;
  the CPU Job name was also absent on both clusters
- create-intent receipt:
  `sha256:b5527d246d0bcdf44bbd73c80ca31682804b2e185f9eb80ec973487622c869fa`
- root Job annotation / Pod priority / GPU request: failure alerts `off` / `c1` /
  no GPU resource key
- runtime transport: the digest-bound private bundle was embedded in the exact
  manifest; no mutable file-copy transport was used
- Job UID: `abfc31ed-e898-45b6-b8f3-3a47c8c8db6f`
- Pod UID: `9cb84709-1b74-4ff1-b7c2-9c49099b0ae8`
- Workload: none was created for this direct batch Job
- immutable image: `fleet/skyrl-train` at
  `sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f`
- restarts / peak GPUs / active GPUs after release: `0 / 0 / 0`
- receipt: `sha256:9d72e7bd1af5c09daa1e72d127d352961f6282949e0ab1dcf05b4a32663d925f`
- complete observer result:
  `sha256:6cfcbf2b4586a2e577c0215219994e652de353b37d4f9cbaeaba9b88e2b52e26`
- sanitized result:
  [`2026-09-20-skyrl-topology-probe-preflight-v29-result.json`](./2026-09-20-skyrl-topology-probe-preflight-v29-result.json)

The digest-bound validator accepted the exact image, UID 1000/GID 100, all
model-file checks, native sampler setting, zero task rows, zero episodes, zero
optimizer steps, future output absence, successful exit and complete release.

## Remaining live gates

Before any GPU creation, an operator must independently validate the durable CPU
receipt from its read-only verifier Job, re-run all non-creating server previews,
arm the direct RayJob cleanup observer, and create the one-time authorization
packet. Only then may the maintained launcher issue its single create. The CPU
result alone does not authorize that operation.

The finalized non-creating packet is at
`/private/tmp/q38-direct-rayjob-qualified.peT7WF/packet`. Its CPU manifest is
`preflight-job.json`, SHA-256
`64f06f9a9ed4db4de4cd2ce3afa2b84b25b8f7d4be471c32dd96e3842004ef14`.
Its CPU server-preview receipt is sealed at
`sha256:684a878bcbf85abb34e6d563d53d796c641313dd0815b06ecc925ff1b919dee6`.
The later read-only durable-receipt verifier manifest is
`receipt-verify-job.json`, SHA-256
`1409b89a9c113ff6ca2b7cb1afcfdc8428c0bd475b123d0820f185c684e8edfe`;
its server-preview receipt is sealed at
`sha256:4a5e0de1f77506c7c3435913794e28ab0cb0203fd2145d7954e898513cb91513`.

Prepare the CPU observer without changing the cluster:

```sh
uv run --locked python -m training.dev_cleanup_observer \
  --context nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb \
  --namespace fleet-train-jobs \
  --kind job \
  --name chris-q38-skyrl-probe-preflight-v29 \
  --maximum-seconds 1200 \
  --expected-gpus 0 \
  --plan-sha256 sha256:b398e73ba3124c4f22e53aceb9ac7924916ba2a3027a3f780261d184505ae678 \
  --manifest-sha256 sha256:64f06f9a9ed4db4de4cd2ce3afa2b84b25b8f7d4be471c32dd96e3842004ef14 \
  --armed /absolute/new/PREFLIGHT_OBSERVER_ARMED.json \
  --result /absolute/new/PREFLIGHT_OBSERVER_RESULT.json
```

For replay prevention, the exact CPU command that was issued once after its
observer wrote `ARMED.json` was:

```sh
kubectl \
  --context nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb \
  --namespace fleet-train-jobs \
  create \
  --filename /private/tmp/q38-direct-rayjob-qualified.peT7WF/packet/preflight-job.json
```

It must never be issued again. Before the eventual GPU create, arm the same observer with
`--kind rayjob`, name `chris-q38-skyrl-probe-v17`, maximum 1800 seconds,
expected GPUs 8, and direct manifest digest
`sha256:761f249e9a050cb96503a927113ae0faff5df5917c652d97c71ee6d60c22d00c`.

```sh
uv run --locked python -m training.dev_cleanup_observer \
  --context nebius-mk8s-fleetai-training-dev-e04p03enwk5c0va9tb \
  --namespace fleet-train-jobs \
  --kind rayjob \
  --name chris-q38-skyrl-probe-v17 \
  --maximum-seconds 1800 \
  --expected-gpus 8 \
  --plan-sha256 sha256:b398e73ba3124c4f22e53aceb9ac7924916ba2a3027a3f780261d184505ae678 \
  --manifest-sha256 sha256:761f249e9a050cb96503a927113ae0faff5df5917c652d97c71ee6d60c22d00c \
  --armed /absolute/new/GPU_OBSERVER_ARMED.json \
  --result /absolute/new/GPU_OBSERVER_RESULT.json
```
