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
| development | direct RayJob | `65ab1dfafa608c6ec135a6555f1bcff9133dcdad2e2f55fd81a2d109486ff81d` |
| development | data stage | `bb6d2514e439ab4701de1d9a975a55604983f1c95c1a1d7e5c2eb7e3cb9f7194` |
| development | CPU preflight | `339979dc34b7d2b7ca94be828a9fe7c9199bc844e712ec7f4ac2dce76076990d` |
| production | direct RayJob | `f360235e960bd0925f096929a33c7c49552369d1419cd244d989642dca7c750a` |
| production | data stage | `f21d3b05728d26d709c65f68d2fe61ad9f0beef04588a7d6efbac30b3833c8f4` |
| production | CPU preflight | `ee1a2632534a85a7d8cc43864cf030b1f5e6db6c0406ee40bb89f9c5974254f9` |

No server dry-run allocated a Pod, node, CPU, or GPU.

## Regression results

- `uvx ruff check` and `uvx ruff format --check`: passed.
- focused direct-rail and cleanup-observer tests: **17 passed**.
- the repository-wide suite completed with **15 pre-existing failures**. Those failures
  are in stale generated SkyRL queue/audit digests and older request builders that do
  not yet set the now-mandatory `failureAlerts: false`; none imports or exercises this
  direct rail. They are recorded rather than repaired here because they are outside the
  launch-critical prod4 transport.

## Remaining live gates

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
