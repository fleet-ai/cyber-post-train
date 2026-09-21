# Qwen3.8 SkyRL reward canary: fresh prod9 preparation

Prod9 is a **new one-node, eight-GPU, one-update canary**, not a retry or
resume of prod8. It keeps the reviewed Qwen3.8-27B image, task/version and
reward contract, 262,144-token context window, compaction, eight sampled
episodes, one optimizer update, and step-one checkpoint policy. The only
allowed changes are its new run, output, data-stage, CPU-preflight and W&B
identities.

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

It recompiles the real SkyRL plan and checks the fresh identity, one-node
shape, 262,144-token compaction settings, and zero-GPU preflight's root alert
opt-out. It never reads task rows, contacts external services, stages data, or
authorizes a launch.

The exact local rebind, read-only absence, and server-preview sequence is in
[`QWEN38_SKYRL_PROD9_NONSUBMITTING_GATES.md`](QWEN38_SKYRL_PROD9_NONSUBMITTING_GATES.md).
It creates no Kubernetes object and deliberately omits every live create.

## Why this does not add a separate setup-module digest gate

This direct training rail does not use the separate engine-setup module from
the collector diagnostic. Its relevant native modules are already pinned in
`training/skyrl_training.py`'s `NATIVE` map, checked by the exact-image CPU
preflight, and used again when the final RayJob is rendered and server-previewed.
The immutable image digest is also fixed by the qualification file. The
collector diagnostic has its own setup-module hash because it starts an engine
through that separate diagnostic path. Adding that unrelated check here would
not prove anything about this training path and would create a new, artificial
compatibility requirement.

## Exact private-data staging requirement

The only permitted source is the earlier **input** package, not any prod8
output:

`/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod8-v1/data`

In a restricted local working directory, use
`rebind_private_source_for_identity` with the sealed prod9 identity. It may
change only each private row's embedded run ID and that row's checksum. It must
then rehash the resulting private files and manifest, archive them, and publish
them create-once to the prod9 destination using the existing zero-GPU stage
Job. Do not commit or print the private rows, prompts, task data or archive.

## Before the one GPU create

1. Confirm the prod9 name, output root, data destination, stage/preflight Job
   names and W&B ID are all absent. This must inspect the Jobs API and both
   clusters immediately before creation.
2. Server-preview the stage and exact-image CPU-preflight Jobs in the required
   contexts. Each rendered root Job must have
   `fleet.ai/failure-alerts: "off"`.
3. Stage the data once; independently verify its digests and zero-GPU release.
4. Run the exact-image CPU preflight once. It must prove compaction and both
   output-limit conditions and bind W&B to the new ID with `resume="never"`.
5. Construct the direct root RayJob from a fresh Jobs preview, server-preview
   it, and verify its root annotation is exactly
   `fleet.ai/failure-alerts: "off"` before any create.
6. Arm the existing UID-bound cleanup observer for exactly the new RayJob,
   plan and rendered-manifest digests. It may release only a terminal run, a
   resource-contract breach, or the maximum runtime.

The preparation code does not create a workload. A separate authorized
operator must perform these live gates and one create-only call.
