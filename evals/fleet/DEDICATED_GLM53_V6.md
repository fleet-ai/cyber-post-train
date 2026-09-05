# HELD GLM5.3 dedicated serving v6

This package prepares two dedicated GLM5.3 endpoints without authorizing or
creating them. It exists to prevent GPU nodes from being admitted before there
is reconciled work ready to consume them.

The immutable authority is
`configs/glm53-dedicated-serving-v6-held.json`. Validate it and print the
sanitized capacity preview with:

```bash
evals/fleet/scripts/submit_glm53_dedicated_v6.sh held-preview
```

To inspect the exact current-schema Jobs API request locally, without making a
request, render one replica at a time:

```bash
python3 -m evals.fleet.glm53_dedicated_v6 server-preview --replica A
python3 -m evals.fleet.glm53_dedicated_v6 server-preview --replica B
```

`submit_glm53_dedicated_v6.sh submit` intentionally exits 78. A later
append-only release must replace that hold; do not edit the held receipt or
weaken the validators.

## Frozen serving treatment

Both requests use the already-proven immutable runtime image and exact staged
model:

- image: `ghcr.io/fleet-ai/cyber-post-train-glm53-runtime` at digest
  `ec93ba50613fd13fb4c0b0a9105767ab18209a1e0108dab0923aad694c0206ec`;
- model: `zai-org/GLM-5.3` at revision
  `30333038ada1f1dacb294a93270305a890b50c14`;
- SFS path: `/mnt/sfs/models/glm-5.3-30333038`;
- 262,144-token context and the proven TP8/DP8/EP8 SGLang arguments;
- one worker and eight GPUs per replica, at most two replicas;
- `fleet-serve-low` with `preemptionPolicy: Never` in the rendered Pod;
- at most two one-stream rollout controllers per endpoint.

The current Jobs API request schema does not expose the former
`image_pull_secrets` field. The GHCR image is private. Therefore a successful
JSON-schema validation is insufficient: an authenticated `/v1/runs/preview`
must show the exact image and `ghcr-pull` in the rendered Pod for each replica.
If it does not, do not submit. Publish the same immutable runtime to an
API-pullable registry or obtain a reviewed Jobs API image-pull contract first.

## No-idle lifecycle

The request embeds `scripts/glm53_dedicated_v6_lifecycle.sh` byte-for-byte; it
does not load a mutable script from SFS. Model loading is productive startup and
does not consume the idle budget. Once `/health` first returns HTTP 200, either
controller must keep one of these files fresh:

```text
<run_dir>/lifecycle/traffic-stream-1
<run_dir>/lifecycle/traffic-stream-2
```

With no fresh controller heartbeat for 600 seconds, the wrapper terminates the
server. Creating `<run_dir>/lifecycle/DRAIN` also terminates it cleanly. The
controller package must be fully rendered and ready before the GPU request is
submitted. If controller admission, networking, or scoring blocks after the
server becomes Ready, release the server before diagnosis; never leave a GPU
node idle while repairing CPU-side orchestration.

Each dedicated controller packages
`scripts/glm53_dedicated_v6_controller_heartbeat.sh` and runs `watch` against
only the child process owning its active cell. The helper accepts only replica
A/B and stream 1/2, writes atomically once a UID-bound server is Ready, and
removes its heartbeat when the child exits. A controller must not run the
watcher merely to reserve an idle endpoint.

## Explicit statistical blocks

The GLM generation-5 canary remains rank 13, attempt 1, on hosted inference.
The first dedicated scored canaries are fixed before release: replica A uses
rank 51, attempt 1, and replica B uses rank 76, attempt 1. Their acceptance
receipts must bind the exact task version, cell, session, verifier execution,
global claim, serving-object UIDs, ingestion, and cleanup; a summary Boolean is
not sufficient.
The other 399 cells are frozen as follows:

| Serving treatment | Task ranks | Streams | Remaining cells |
|---|---:|---:|---:|
| hosted | 1–25 (rank 13 attempts 2–4) | 1 | 99 |
| hosted | 26–50 | 1 | 100 |
| dedicated A | 51–75 | 2 | 100 |
| dedicated B | 76–100 | 2 | 100 |

The two dedicated streams split only at task boundaries: A uses ranks 51–63
and 64–75; B uses 76–88 and 89–100. No task may move between hosted and
dedicated serving after release. Persist the serving block and live endpoint
UIDs with every execution, report outcomes stratified by serving block, and do
not pool hosted and dedicated measurements without a later reviewed
equivalence analysis.

## Required release sequence

1. Require the exact GLM generation-5 hosted canary to be terminal accepted.
2. Run the fresh prebulk reconciliation and remove every accepted, active,
   claimed, or model-started cell from the executable inventory. Any newly
   discovered partial task stays entirely in its existing serving treatment.
3. Pre-render all controller packages and prove their CPU admission path.
4. Authenticated-preview replica A through the Jobs API. Normalize and seal a
   preview receipt, proving the exact private-image pull, shape, priority,
   non-preempting policy, command, and fresh run directory.
5. Perform exhaustive Jobs API, Kubernetes, SFS, and cell-claim duplicate
   checks. Create replica A once.
6. Bind API run, RayJob, RayCluster, head Pod, and Service UIDs. Require zero
   restarts, exact image/model/arguments/context, HTTP health, and content-blind
   structured `bash` plus `submit_report` parity.
7. Start the predeclared rank-51 attempt-1 scored cell while writing stream-1
   heartbeats. Only a digest-valid accepted canary releases A's second stream.
8. Repeat authenticated preview and create-once checks for B. Do not submit B
   before A's canary passes. Repeat UID-bound parity and the predeclared
   rank-76 attempt-1 scored B canary before B's second stream.
9. Maintain no more than two streams per endpoint and two nodes/16 GPUs total.
   Drain each replica immediately when its block is empty or its controllers
   become terminal.

The UID-bound runtime-gate validator lives in
`evals.fleet.glm53_dedicated_v6`; it requires the scored canary and structured
tool parity before declaring two streams released. It deliberately rejects
summary-only or non-UID-bound observations.

## Current blockers

The package remains HELD until the generation-5 canary and prebulk
reconciliation are accepted and authenticated preview proves private-image
pulling under the current Jobs API. Qwen dedicated serving is not part of this
package: only Qwen model staging is proven, so Qwen remains on its hosted
serving block until its own image, topology, arguments, and UID-bound canary are
independently qualified.
