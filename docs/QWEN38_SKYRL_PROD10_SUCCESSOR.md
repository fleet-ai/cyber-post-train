# Qwen3.8 SkyRL prod10 successor (no-launch packet)

Prod10 is the fresh successor to the terminally failed prod9 startup attempt.
It reuses the reviewed prod9 SkyRL schema, token-safe compaction runtime, task
versions, model, image, and one-update recipe. It does not resume prod9 or use
prod9 output. Its input is rebound again from the sealed prod8 input package;
only the embedded run identity and package checksums may change.

The create-once identities are:

- run and W&B ID: `chris-q38-rlreward-prod10`;
- output root: `/mnt/sfs/jobs/chris-q38-rlreward-prod10`;
- data root:
  `/mnt/sfs/jobs/chris-q38-study-corpora-v1/rlreward-inputs-prod10-v1/data`;
- data-stage Job: `chris-q38-prod10-data-v1`;
- exact-image preflight Job: `chris-q38-prod10-preflight-v1`.

The offline preparation check accepts only repository-owned qualification JSON
files. Their exact bytes are included in its sealed receipt; the launch gate
still requires those bytes to be merged and rechecked from current main:

```sh
uv run --locked python scripts/prepare_qwen38_skyrl_prod9_successor.py \
  --manifest <restricted-local-prod10-manifest.json> \
  --run configs/qualification/qwen38-rl-reward-canary-prod-v10.json \
  --data configs/qualification/qwen38-rl-reward-canary-data-prod-v10.json \
  --identity configs/qualification/qwen38-rl-reward-canary-prod10-identity-v1.json
```

This command reads no task rows, contacts no external service, allocates no
resource, and cannot authorize a launch. Before any create, require fresh
development and production server previews of both zero-GPU Jobs and the GPU
RayJob. Every rendered root Job or RayJob must contain exactly
`fleet.ai/failure-alerts: "off"`; the GPU request must remain `c1`/`q1`, one
node, and eight GPUs. Recheck Jobs API, both clusters, SFS output, and W&B
identity absence, then take a fresh all-namespace capacity census immediately
before the sole POST. An ambiguous POST is never retried.

Prod10 is scientifically accepted only after eight authoritative verifier
receipts show real reward variation, step one has finite optimizer telemetry
and changed model tensors, the step-one checkpoint and BF16 export are sealed,
an independent one-GPU zero-update reload produces finite logits, and both
exact-owned allocations are observed released. Missing evidence is a stopped
canary, not a zero reward and not permission to scale.
