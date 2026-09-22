# Qwen3.8 SkyRL reward canary: fresh prod9 preparation

Prod9 is a **new one-node, eight-GPU, one-update canary**, not a retry or
resume of prod8. It keeps the reviewed Qwen3.8-27B image, task/version and
reward contract, 262,144-token context window, compaction, eight sampled
episodes, one optimizer update, and step-one checkpoint policy. Its new
identities and fresh prod9 source bindings for token-safe compaction, the
zero-GPU gates, and the bounded one-create rail are the only allowed changes.

The prior run's terminal result and GPU release cannot be proved: its launch
record contains neither a terminal receipt nor a verified release, while the
current Jobs API and both Kubernetes clusters no longer contain the named
RayJob. The sanitized reconciliation is
[`2026-09-21-skyrl-prod8-reconciliation-v1.json`](evidence/qwen38-study/2026-09-21-skyrl-prod8-reconciliation-v1.json).
Therefore prod8 is never resumed and none of its output or checkpoints may be
used.

## Frozen successor identity

- run and W&B ID: `chris-q38-rlreward-prod9`;
- output root: `/mnt/sfs/jobs/chris-q38-rlreward-prod9`;
- private data destination:
  `/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod9-v1/data`;
- zero-GPU data-stage Job: `chris-q38-prod9-data-v1`;
- zero-GPU exact-image preflight Job: `chris-q38-prod9-preflight-v1`.

The committed files are:

- [`qwen38-rl-reward-canary-prod-v9.json`](../configs/qualification/qwen38-rl-reward-canary-prod-v9.json)
  — the fresh run configuration;
- [`qwen38-rl-reward-canary-data-prod-v9.json`](../configs/qualification/qwen38-rl-reward-canary-data-prod-v9.json)
  — the unchanged scientific limits and fresh data destination;
- [`qwen38-rl-reward-canary-prod9-identity-v1.json`](../configs/qualification/qwen38-rl-reward-canary-prod9-identity-v1.json)
  — the sealed create-once identity.

Before any Fleet operation, the restricted operator can run the offline check
against the newly rebound **manifest only**:

```sh
uv run --locked python scripts/prepare_qwen38_skyrl_prod9_successor.py \
  --manifest <restricted-local-prod9-manifest.json>
```

It recompiles the real SkyRL plan and writes a sealed preparation receipt with
the current runtime, plan, request, zero-GPU rebind, and CPU-preflight digests. It also
verifies the fresh prod9 source closure for the token-safe recorder, direct
capacity gate, and terminal checkpoint/reload acceptance gate. It checks the
fresh identity, one-node shape, the exact 262,144-token context and
4,194,304-response-token episode limits, and
zero-GPU preflight's root alert opt-out. Those digests are fresh prod9 evidence;
they are never inherited from historical prod8. The command never reads task
rows, contacts external services, stages data, or authorizes a launch.

The exact staged rebind, absence, server-preview, and one-create sequence is in
[`QWEN38_SKYRL_PROD9_NONSUBMITTING_GATES.md`](QWEN38_SKYRL_PROD9_NONSUBMITTING_GATES.md).
It creates no Kubernetes object and deliberately omits every live create.

## Relationship to existing Qwen RL recipes

Prod9 is not a new trainer invented from scratch. It keeps the small, strict
one-update shape used to qualify Fleet's existing Qwen RL routes: one grouped
set of eight attempts, a real task grader, learning rate `1e-6`, a checkpoint
after the first update, and an independent reload before any scale-up. The
full comparison and source references are in
[`QWEN38_RL_RECIPE_ARCHAEOLOGY_2026-09-21.md`](QWEN38_RL_RECIPE_ARCHAEOLOGY_2026-09-21.md).

The narrow reason for using SkyRL here is long-horizon correctness, not a claim
that it is more mature than Fleet's maintained Miles route:

| Route | What prod9 reuses | What deliberately differs |
| --- | --- | --- |
| Maintained FTI/Miles 256K | Exact Qwen tool/template qualification, real grader IDs, grouped GRPO, `1e-6` first-update learning rate, and checkpoint/reload gates. | The supported 256K Miles shape needs four eight-GPU nodes and ends a full context without summarizing earlier turns. Prod9 instead uses one eight-GPU node and records a token-safe student-generated summary before it would overflow. |
| Neeraj Dataminer v003 sync Miles | Synchronous rollout → reward → update → weight-sync mechanics; reject malformed groups rather than inventing zero reward; qualify the tool template; and save/reload a checkpoint. | Its practical 96K route used truncation-with-reward and implemented no compaction. Its token-native IDs and per-turn prefix assertion mean a compacted objective would require separate design and qualification; the pinned source does not prove it impossible. Prod9 keeps the native Qwen 262,144-token limit and records each compacted model turn separately. |
| Prod9 SkyRL | The safeguards above, plus a direct root-RayJob capacity proof and root failure-alert annotation proof. | It remains only a one-update Fleet-cyber canary until it produces mixed real rewards, a finite non-zero update, a sealed checkpoint, an exact BF16 reload, and verified release. |

Neither prior route's task mix, reward distribution, transfer claim, or terminal
artifacts may be copied into this Fleet-cyber canary. They are implementation
lessons, not scientific evidence for this task set.

## Why this does not add a separate setup-module digest gate

The fresh training closure is `training.skyrl_prod9_training`, including its
explicitly bound historical helper modules. `training.skyrl_prod9_direct`
renders the matching zero-GPU rebind/CPU-preflight Jobs and GPU RayJob, and
contains the only no-retry create functions permitted for prod9. No source
change invokes those functions: live server previews, absence checks, a
pre-armed observer, receipts, and the capacity proof must still be fresh when
an authorized operator reaches each gate. The fresh source hashes and this
rail—not the historical direct rail—are therefore the relevant current gate.
The immutable image digest remains fixed by the qualification file.

## Exact private-data staging requirement

The only permitted source is the earlier **input** package, not any prod8
output:

`/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod8-v1/data`

The operator machine cannot mount this SFS path, so a local rebind/archive is
not a valid prod9 gate. The fresh zero-GPU SFS rebind/data-stage source uses
`rebind_private_source_for_identity` with the sealed prod9 identity. It may
change only each private row's embedded run ID and checksum, then rehash the
five-file package before publishing the new destination. Its root Job has the
failure-alert opt-out before preview; its one-create function requires dev and
prod previews, a pre-armed UID-bound observer, and a release receipt. This PR
does not invoke it. Do not commit or print private rows, prompts, task data, or
an archive.

## Required live sequence after review

The source now contains the exact stage, preflight, and one-create functions,
but no Job or RayJob has been created by this change. A real operation remains
blocked until this source is reviewed and all evidence below is fresh.

1. Confirm the prod9 name, output root, data destination, stage/preflight Job
   names and W&B ID are all absent. This must inspect the Jobs API and both
   clusters immediately before creation.
2. Server-preview the new rebind/data-stage and exact-image CPU-preflight Jobs
   in the required contexts. Each rendered root Job must have
   `fleet.ai/failure-alerts: "off"`.
3. Stage the data once through the fresh zero-GPU rail; independently verify
   its sanitized package digests and exact-UID zero-GPU release.
4. Run the exact-image CPU preflight once through the fresh renderer. It must
   prove compaction and both output-limit conditions and bind W&B to the new ID
   with `resume="never"`.
5. Construct the direct root RayJob from a fresh Jobs preview with the fresh
   renderer, server-preview it, and verify its root annotation is exactly
   `fleet.ai/failure-alerts: "off"` before any create.
6. Arm the existing UID-bound cleanup observer for exactly the new RayJob,
   plan and rendered-manifest digests. Immediately before the sole GPU create,
   prove all-namespace project capacity including the planned eight GPUs,
   recheck identities and W&B absence, record a no-retry create intent, and
   create exactly once. The observer may release only a terminal run, a
   resource-contract breach, or the maximum runtime.
7. Before `ACCEPTED.json` can be written, require the original eight-GPU
   observer's exact UID-bound creator handoff and release result, then require
   the separate one-GPU, zero-update reload and its release result. The
   acceptance receipt binds both sets of workload identities and receipt
   digests.

The create journals are not stored in a caller-selected working directory.
Stage, training, and reload roots are computed from their sealed immutable
identity below `/mnt/sfs/jobs/.cyber-post-train-prod9-create-once-v1`; a fresh
alternate directory cannot authorize or replay a create.

The preparation code does not create a workload. A separate authorized
operator must perform these live gates and one create-only call.
