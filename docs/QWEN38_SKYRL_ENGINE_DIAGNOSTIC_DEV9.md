# Qwen3.8 SkyRL engine diagnostic dev9

Status: **fresh successor configured; not prepared, previewed, or submitted**.

Dev8 is terminal evidence and must never be edited, copied into a new prepared
directory, or replayed. Dev9 is the only reviewed successor identity. It changes
the runtime image and every create-once operational identity while preserving the
diagnostic science and safety boundaries.

## Exact closure

- Data config:
  [`qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v9.json`](../configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-data-dev-v9.json),
  file SHA-256
  `084bb9a65b77482e46f0218ff990a920abed2712de1e8cea4f8053dca594846e`.
- Run config:
  [`qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v9.json`](../configs/qualification/qwen38-rl-filtered-skyrl-engine-diagnostic-dev-v9.json),
  file SHA-256
  `6b17e668cde7eaded5bb5df5a010568702f8260d2c4f71e61cb9396383515538`.
- Image:
  `661864827319.dkr.ecr.us-east-1.amazonaws.com/fleet/skyrl-train@sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f`,
  built from Theseus commit
  `8d62868d6dc00eee793d83efe5738dc21e42758d`.
- Exact-image CPU qualification:
  [`2026-09-12-skyrl-worker-rpc-relay-image-cpu-qualification-v1.json`](evidence/qwen38-study/2026-09-12-skyrl-worker-rpc-relay-image-cpu-qualification-v1.json),
  embedded receipt SHA-256
  `28244d695896b8b9766df66caecd117a33fd5d9c2c5df35faba12bcc785d09f1`
  and file SHA-256
  `cb993198e84220a74ed77f471359b36149bb47eb7bf3380aea0e4edb98b55c19`.

The fresh identities are:

| Identity | Dev9 value |
| --- | --- |
| Config/data/W&B name | `chris-q38-rldiag-dev9` |
| Data root | `/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev9/data` |
| Prepared root and submission journal owner | `/mnt/sfs/jobs/chris-q38-study-corpora-v1/rldiag-inputs-dev9/prepared-v1` |
| Runtime output | `/mnt/sfs/jobs/chris-q38-rldiag-dev9` |

All four locations must be absent before their respective create-once step. A
partial identity mix is rejected. The request path also rejects any dev8 run,
output, data, or W&B identity before it can render or submit. The prior relay
image is globally engine-start disqualified for Qwen3.8, independent of run
name, so renaming an old plan cannot bypass the privacy correction.

## Preserved diagnostic contract

Dev9 remains dev-cluster-only and requests `c1`, no automatic requeue, two
workers with four GPUs each, and two TP4 engines. It carries no workload Secret,
initializes no W&B run, reads no task row, creates no challenge environment, and
performs no rollout, verifier call, optimizer update, or checkpoint write. The
request binds expected UID/GID `1000:100`. Preview accepts either explicit
`runAsUser=1000`, `runAsGroup=100`, and `runAsNonRoot=true`, or genuinely absent
identity fields only for image digest `89758df2...` under the immutable
default-user qualification receipt. Any explicit conflicting value remains
fatal, every other image remains ineligible for the exception, and the GPU
entrypoint rechecks its actual UID/GID before Ray initialization. Startup and
owned-resource cleanup remain bounded at 1,800 and 300 seconds.

Success still requires independent terminal evidence. It must cross-bind the
prepared plan/request/preflight/create-once journal, API and Kubernetes object
UIDs and owner chain, exact runtime image IDs, both Pods' runtime user and
effective security contexts, clean zero-work engine shutdown, terminal success,
and absence of the RayCluster/Pods with zero active GPUs. The dev8 terminal
receipt is preserved as historical evidence; it is not a dev9 receipt and cannot
open any later RL gate.

## Remaining pre-POST sequence

No step below was run by this change:

1. Create the new dev9 data root with `rl-data`; do not copy dev8 prepared bytes.
2. Prepare exactly the run config above into the declared dev9 prepared root.
3. In the pinned image as `1000:100`, run the zero-GPU engine-diagnostic CPU
   preflight and preserve its self-digested receipt.
4. Run `preview --cluster dev` and require the exact image, `2 x 4` shape,
   priority, no-requeue, no-secret, runtime-user, namespace, and output bindings.
5. Immediately recheck that the dev API has no matching name/output, all dev9
   SFS/W&B identities are absent, the prepared journal is absent, the intended
   two-node placement is available, and aggregate owned-node authorization is
   satisfied.
6. Only then may an authorized operator execute one dev POST. A timeout or any
   journal row is an ambiguous create that must be reconciled, never retried.

The new config is ready for data preparation and exact-image CPU preflight. It
is intentionally **not yet ready for a dev POST** because those fresh prepared,
preview, duplicate, capacity, authentication, and journal checks have not been
performed.
