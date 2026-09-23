# Qwen3.8 SkyRL prod10 cluster-local controller boundary

Status: **source-only bootstrap prepared; no Job, RayJob, Pod, GPU allocation,
server preview, or target submission was created by this change.**

Prod10 must use the maintained prod9 direct SkyRL rail from a process that can
see the canonical SFS create-once root as UID 1000/GID 100. The generic Fleet
Jobs API cannot host that coordinator because its current request schema
requires at least one GPU per worker. An operator laptop is not a substitute:
it does not have the canonical SFS mount, and the prod9 create journal correctly
reports unavailable there.

[`skyrl_controller_job.py`](../cyber_post_train/skyrl_controller_job.py)
therefore defines the smallest safe cluster bootstrap:

- one suspended, queue-managed `batch/v1` Job at c1/q1;
- exact root and Pod-template `fleet.ai/failure-alerts: "off"` annotations;
- zero effective GPU requests across app and init containers;
- the pinned SkyRL image used by the prod10 recipe;
- exact UID 1000, GID/fsGroup 100;
- full `/mnt/sfs` read-only plus only
  `/mnt/sfs/jobs/chris-q38-study-corpora-v1/launch-controls` writable;
- no Kubernetes service-account token, Secret volume, `envFrom`, Fleet token,
  W&B token, or private task data;
- a byte-bound standard-library driver and a create-once SFS receipt; and
- a typed caller that requires duplicate absence, API-server dry-run validation,
  a canonical SFS fsynced do-not-retry intent, an exact Job UID response, and a
  clean terminal Pod/receipt binding. It has no `kubectl` fallback.

[`kubeconfig_job_api.py`](../cyber_post_train/kubeconfig_job_api.py) is the
concrete caller transport. It accepts only the exact production kubeconfig
context and its CA/exec-credential contract, uses Kubernetes HTTP server-side
dry-run with strict field validation, and exposes only the exact Job plus
UID/label-scoped Workload, Pod, ServiceAccount, and sanitized Pod-log reads.
It has no list-all, patch, apply, delete, shell, or `kubectl` method.

Terminal acceptance reopens the canonical SFS create journal, requires the
exact UID-owned Job, admitted Workload and one clean Pod, and requires the
Pod's termination message to equal its sole sanitized log receipt byte for
byte. The sealed terminal receipt retains the Workload UID/queues/priority,
resolved image ID, and both output digests for later audit.

The bootstrap is deliberately unable to submit the GPU target. Its sealed
packet fixes all three platform gates to false, and the runtime receipt records
`target_posts: 0`, `gpus: 0`, `private_rows_read: 0`, and `traces_read: 0`.
Flipping a JSON boolean cannot open the path because packet validation
reconstructs the entire closed contract.

## Why target submission remains closed

A read-only production audit on 2026-09-22 established three independent
blockers:

1. The namespace `default` ServiceAccount is bound to cluster-admin and must
   never be used as a controller identity. `fleet-trainer` disables token
   automount and has no controller RBAC. No dedicated prod10 controller account
   exists.
2. The deployed Jobs API deletes by mutable run name. It has no operation bound
   to both the immutable Jobs API run UUID and the exact RayJob UID. Granting a
   generic RayJob `delete` Role would broaden authority beyond the one owned
   target.
3. A correct project capacity census currently requires broad cluster object
   reads. No platform-signed, metadata-only capacity endpoint/receipt exists,
   and granting the controller cluster-wide Pod reads would expose unrelated
   specifications.

These are launch blockers, not reasons to weaken the prod9 guard. The reviewed
successor must supply all of the following before the non-submitting bootstrap
can evolve into a creator:

- a dedicated least-privilege identity whose server-rendered Job remains exact;
- a fresh, signed metadata-only 8-node/64-GPU project capacity receipt;
- an immutable run-UUID + RayJob-UID release operation with compare-and-delete
  semantics;
- fresh dev/prod server previews and collision receipts for the exact prod10
  identities;
- creator evidence binding the Jobs API run UUID and RayJob UID;
- a supervised exact-UID observer and accepted release receipt; and
- terminal evidence for reward variation, one finite update, checkpoint save,
  independent reload, and zero remaining owned allocation.

Until those capabilities exist, only the zero-GPU bootstrap may be previewed
after this change is independently reviewed, merged, and rebuilt from current
main. Creation additionally requires explicit operator authorization after a
fresh preview and duplicate check. Neither action authorizes the prod10 GPU
target.
