# SkyRL prod4 direct RayJob qualification

Status at this revision: **transport qualified; scientific GPU canary not yet created**.

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
| development | direct RayJob | `8037ac77d42959ca5a02daafcfcdd2f03dcf37a939b8a53b3bef2223f2de6cae` |
| development | data stage | `fcb49e2a31ab667f3cd78879e629229471370dedb99c788a90483e6fee27457a` |
| development | CPU preflight | `b8c47c3705abb31197c93444da75cb7a611063d853c92ae45a224fef781d9ddc` |
| production | direct RayJob | `ef377ebf7b7adfd8f73bdfcf1ab1d45a5ae6fbccad4432415c475acc98e59f7c` |
| production | data stage | `39c1fec7fdca8a2f82ab943e6422dc032d9f9dd65f3c0b4feaf198676ca84c7c` |
| production | CPU preflight | `95b528540cb24377298fae34205102912c7c593fecf6396c4ade7d22f090abc0` |

No server dry-run allocated a Pod, node, CPU, or GPU.

## Regression results

- `uvx ruff check` and `uvx ruff format --check`: passed.
- focused direct-rail and cleanup-observer tests: **18 passed**.
- the repository-wide suite completed with **15 pre-existing failures**. Those failures
  are in stale generated SkyRL queue/audit digests and older request builders that do
  not yet set the now-mandatory `failureAlerts: false`; none imports or exercises this
  direct rail. They are recorded rather than repaired here because they are outside the
  launch-critical prod4 transport.

## Remaining live gates

The first production CPU preflight (`chris-q38-prod4-preflight-v1`) exited before
writing a receipt and released cleanly with zero GPUs and zero restarts. A bounded
read-only reproduction identified the exact defect: W&B 0.21.1 represents an absent
run as a `CommError` whose nested exception is the SDK's exact `ValueError` sentinel,
not as a response carrying HTTP status 404. The original check therefore rejected the
desired "run is absent" result. The create-once successor is
`chris-q38-prod4-preflight-v2`; it accepts only that exact SDK sentinel or an explicit
404 and still fails closed for authentication and service failures. No model, data,
reward, optimizer, checkpoint, sampling, or GPU setting changed.

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
