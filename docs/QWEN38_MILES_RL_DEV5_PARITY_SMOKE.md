# Qwen3.8 Miles base-RL dev5 parity smoke

## Status

Dev5 is a **prepared, unsubmitted dev-cluster configuration**. It is one bounded
Qwen3.8-27B optimizer update over eight samples. Its create-once data derivation
reopens the already accepted dev4 task-isolated artifact and changes only the
embedded run ID plus its dependent configuration digest. It preserves every
prompt, task, split, tool, verifier, limit, and base-policy identity. Dev5 also
uses the same zero-step base native checkpoint. Preparing this tracked
configuration made no Jobs API POST, RayJob, or GPU allocation.

Tracked inputs:

- run: `configs/qualification/qwen38-miles-rl-reward-canary-dev-v5.json`;
- data derivation:
  `configs/qualification/qwen38-miles-rl-reward-canary-data-derive-dev-v5.json`;
- launch contract:
  `configs/qualification/qwen38-miles-rl-reward-canary-launch-dev-v5.json`;
- source data: the exact accepted dev4 manifest under
  `/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev4/data`, bound by both raw
  file and sealed self-digest;
- dev5 data destination:
  `/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev5-v3/data`;
- initial policy: the exact base Qwen3.8 model and digest-bound zero-step Miles
  checkpoint already used by dev4;
- fresh output and W&B identity: `chris-q38-miles-rlreward-dev5`.

The earlier `/mnt/sfs/jobs/chris-q38-miles-rlreward-inputs-dev5/data` path is
quarantined: a read-only verifier found that it is not the exact accepted
run-id-rebound artifact. Never overwrite, delete, or select it. The `-v3` root
is a new create-once identity.

## Peer-parity decisions

The settings deliberately copy the compatible parts of Neeraj's working
Qwen3.8 one-node safety envelope:

| Setting | Dev5 |
|---|---:|
| topology | 1 node × 8 GPUs |
| parallelism | TP4 / PP1 / CP2, sequence parallel |
| optimizer updates | 1 |
| generated training samples | 8 |
| dynamic token ceiling | 8,192 tokens/GPU |
| recomputation | full, uniform, one layer |
| learning rate | 2e-6 |
| KL coefficient | 0.001 |
| rollout temperature | 0.7 |
| CPU | request 32, limit 32 |
| memory | request 1,800 GiB, limit 2,400 GiB |
| priority | request `c1`; server renders `q1`, value 10,000 |
| requeue | disabled |

The native profile already supplies TP4/PP1/CP2 and sequence parallelism for
the 1×8 layout. The repository now rejects native-profile drift away from those
values or from full recomputation. Temperature, KL coefficient, and the dynamic
token ceiling are explicit configuration fields rather than hidden constants.

Task isolation is unchanged: optimization uses only the accepted Fleet train
row, Fleet dev remains evaluation-only, and external benchmarks are not training,
reward, tuning, retry, or checkpoint-selection inputs. The offline derivation
has no network path and fails unless the complete source artifact reopens, the
exact frozen task/split/tool/runtime bindings match, and restoring only `run_id`
and `config_sha256` gives byte-equivalent row content. Runtime checkpoint bytes
may live at their staged native path, but episode evidence continues to bind the
exact scientific base-policy identity.

## Jobs API constraint: shared memory

The requested peer envelope used 128 GiB at `/dev/shm`, but the current generic
Fleet Jobs API cannot express shared memory. Its schema exposes only CPU and
memory request/limit fields, and the server currently renders a fixed 64-GiB
memory-backed `/dev/shm` volume. A read-only inspection of an existing dev Jobs
API RayJob confirmed the same 64-GiB render. Unknown request fields are not a
safe workaround because the server may ignore them.

Therefore dev5 truthfully records 128 GiB as **not expressible** and retains the
Jobs API's 64-GiB render as one explicit dev-only parity deviation. It must not
be reported as a 128-GiB shared-memory run. If strict 128-GiB parity becomes an
acceptance requirement, extend the Jobs API schema and renderer first; do not
switch to an ad-hoc paid Kubernetes Job.

CPU 32 and memory 1,800/2,400 GiB are accepted by the current Jobs schema. The
local Miles boundary was lowered only for CPU request, from 64 to the peer-proven
32; its existing 1,536/2,048-GiB memory floors remain intact. Production's exact
previous resource contract is unchanged.

## Required launch sequence

1. Land the tracked code/config changes and rebuild an immutable source bundle
   from that exact commit.
2. Run the prepared zero-GPU, no-SFS, no-secret exact-image gate and require all
   focused imports/tests and the dev5 argument contract to pass.
3. Reopen the accepted dev4 data and create the dev5 run-ID-only successor at
   its fresh create-once data root. Independently verify its derivation receipt.
4. Reopen the derived dev5 data and base checkpoint by digest on a read-only
   dev helper; compile the exact plan/request without exposing task text.
5. Use the dev Jobs API's preview-only mode once. Require one 8-GPU worker,
   `c1`/rendered `q1`, no requeue, exact image, CPU/memory values above, and the
   documented 64-GiB shared-memory render.
6. Recheck fresh API name, output root, W&B ID, and the eight-node campaign cap.
   Only then may one create-once POST be considered.
7. Accept the run only with authoritative reward execution, one optimizer update,
   a value-level policy delta, a sealed step-1 native checkpoint, and terminal
   controller/resource-release evidence. A controller success alone is not
   scientific acceptance.

## Dev4 cancellation state

The exact queued dev4 identity was
`chris-q38-miles-rlreward-dev4-65a2598c`, API run ID
`65a2598c-13f5-46ce-a8dc-102075547b25`, RayJob UID
`74784bff-0fd4-423f-ba8d-2ca7e8a84496`, and Workload UID
`85a96f37-218c-4f88-8aad-3bca0ef42ec1`.

The operator first bound that exact live run in `Suspended`, unadmitted state,
then called `Jobs.cancel_once` exactly once. The Jobs API returned HTTP 204.
Independent reconciliation then proved API GET 404 and the exact RayJob,
Workload, RayCluster, Pods, and GPU allocation absent. It performed zero task
interactions and optimizer updates. The remaining history row is stale and is
not a live allocation. **Do not issue another DELETE.**

For any future exact queued run, cancellation is fail-closed: bind the live API
run and UID-matched Kubernetes objects first; write one create-once intent; call
`Jobs.cancel_once(exact_api_name)` exactly once; never retry an uncertain
transport result; then independently prove API 404 plus exact RayJob, Workload,
descendant, and GPU absence before sealing a sanitized release receipt.
