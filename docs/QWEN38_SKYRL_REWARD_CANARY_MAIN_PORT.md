# Qwen3.8 SkyRL reward-canary main port

Status: **prepared, not submitted; external preview and submission are blocked.**

The independent offline launch-readiness audit is
[`2026-09-20-skyrl-launch-readiness-audit-v1.json`](evidence/qwen38-study/2026-09-20-skyrl-launch-readiness-audit-v1.json).
It recompiles the exact V17 and prod4 plan/request identities without reading
private payloads or contacting external services, and checks the five queued
production configurations. It is a fail-closed audit, not launch authority.

This is the current-main port of the scientifically valid one-node reward
canary from runtime commit `8b5e8a00521b4fe412e3e7d0bce9b8ea6855b005`,
with preflight provenance from
`6c0bc541e136fc114a3f7a17ce6ba07e9178bb78`. The prior preflight is
historical evidence only. It certifies neither these rebased runtime bytes nor
the new create-once identities.

The next immutable inputs are:

- topology config:
  [`qwen38-skyrl-topology-probe-dev-v2.json`](../configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json);
- data config:
  [`qwen38-rl-reward-canary-data-prod-v4.json`](../configs/qualification/qwen38-rl-reward-canary-data-prod-v4.json);
- run config:
  [`qwen38-rl-reward-canary-prod-v4.json`](../configs/qualification/qwen38-rl-reward-canary-prod-v4.json);
- source and submission closure:
  [`qwen38-rl-reward-canary-port-v4.json`](../configs/qualification/qwen38-rl-reward-canary-port-v4.json).

The port preserves the exact train and dev task versions, ordered
`bash`/`submit_report` surface, fractional partial reward contract,
98,304/81,920-token context and response limits, 600-turn ceiling,
2,400-second episode limit, 330-second tool limit, one node, eight GPUs, two
TP4 engines, eight samples, one optimizer step, learning rate `1e-6`,
pre/post evaluation, step-1 checkpoint, and seed 42. The only runtime repair is
canonical ordering of the already digest-verified live tool schema before Qwen
prompt rendering.

The v4 configuration selects immutable image
`sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f`
and sets `VLLM_USE_FLASHINFER_SAMPLER=0`. Its retained engine receipt proves
two-by-four engine startup and release only; it is not one-node training
qualification. The `v17` topology successor must independently produce and
validate `TOPOLOGY_PROBE.json` after its fixed 30-second terminal-receipt grace.

The scientific canary is production-routed because the reviewed experiment has
a 2,400-second episode limit. The development cluster's 30-minute workload
limit cannot contain that episode plus model startup and cleanup. The
development cluster remains the bounded place for the zero-update topology
check.

The exact prepared topology packet is bound to:

- plan SHA-256 `fbdd777447219623623fc9679aecc65f3b3c692aa20a9d49a4cc66ae29e43774`;
- request SHA-256 `87e3a0971f1d700d57194cc87fb1ec3b1df0e16fc1dfca9d8f634bf0a4209e37`;
- FleetJob manifest SHA-256 `1fc00898bbe35739db9896a2765bad734ee616d72c0caa26f1f73434208ba223`;
- CPU-preflight manifest SHA-256 `92a39c456d53f57e8ccbcc8a56b3427d815ed1c70e2c40464e0e7b1d2444aa37`;
- receipt-verifier manifest SHA-256 `255ae49210d50c285bf9ad4c4a330cc3ccc95070bd0dca9e0d19cc81420a4668`.

The locally compiled scientific packet is bound to plan SHA-256
`d6b36b07f1fcb9075e96f5d4918aecdf204c8ad2f3fa8d5d29527bfc9c777c9c`
and request SHA-256
`bb69d48afd9d6a1cafea678dbc408f803dda93e4e32bb42a23aac620da7c224c`.
Those digests do not authorize submission.

## Why submission is blocked

The command surface refuses external preview and submission before creating a
Jobs client. There is no longer a failure-count gate. Four unresolved gates
remain:

1. accepted `v17` topology receipt plus proof that its eight GPUs were released;
2. create-once staging and digest verification of the exact one-train/one-dev
   data manifest under the `prod4` data path;
3. an exact-image CPU preflight and Jobs API preview for `prod4`; and
4. execution of the encoded read-only Jobs/Kubernetes/SFS/W&B absence guard
   immediately before the one allowed create.

The current development FleetJob admission webhook rejects the mandatory
root `fleet.ai/failure-alerts: "off"` annotation on both the FleetJob and its
embedded RayJob. The zero-GPU preflight and receipt-verifier server dry-runs
pass with the annotation, but the GPU FleetJob cannot be created until the
FleetJob interface can preserve that annotation on the generated RayJob or the
probe moves to another alert-safe transport. The exact evidence is recorded in
[`2026-09-20-skyrl-alert-optout-launch-repair-v1.md`](evidence/qwen38-study/2026-09-20-skyrl-alert-optout-launch-repair-v1.md).

The offline defects for the five full arms are repaired. Each arm now has an
exact sanitized staged-manifest candidate, a runnable UID/mode/file-set/digest
validator, a full immutable plan, an exact request digest, a local no-submit
preview, and a two-phase UID-bound release-observer contract. These artifacts
contain counts and digests but no task prompts. They do not claim that the
private bytes have been copied to SFS or that a server preview has run.

The production-only runtime leaves the 2,400-second scientific episode limit
unchanged. Its plan-bound hard watchdog is 43,200 seconds for the four ten-step
arms and 172,800 seconds for the fifty-step arm. Those bounds exceed the legal
38,400/163,200-second episode schedules plus startup and drain allowance; the
shared one-step-canary runtime and its frozen request are unchanged.

The production submit path no longer relies on the generic duplicate check.
After the server preview and immediately before its create-once POST, it
requires a freshly armed release observer and rechecks the complete Jobs
history, the exact production Kubernetes namespace, the SFS staged tree and
absent output root, and the exact W&B run ID. The checks must finish within 120
seconds and are recorded in the durable no-retry POST intent. Historical clean
observations remain non-reusable.

External gates remain: prod4 acceptance, failure-budget reset, create-once SFS
staging, exact-image CPU preflights, authenticated server previews, and live
observer arming/release receipts. Nothing in the offline packet authorizes a
launch.

Review the deterministic packet without external access:

```sh
uv run --locked python scripts/prepare_qwen38_skyrl_production_queue.py --check
uv run --locked python scripts/audit_qwen38_skyrl_launch_readiness.py --check
```

After separately authorized SFS staging, preparing an arm through
`cyber-post-train rl` reopens that exact manifest and writes
`RELEASE_OBSERVER_CONTRACT.json` plus `OFFLINE_PREVIEW.json` beside the plan and
request. Running `cyber-post-train preflight` in the pinned CPU image invokes
the staged-data validator. `preview` and `submit` remain blocked by the sealed
qualification until the external gates above are replaced by accepted evidence.

After those gates pass, the scientific run may be created once. Acceptance
requires eight genuine rollouts, non-constant verifier rewards, one finite
optimizer update, a reloadable step-1 checkpoint, and complete resource release.
All-zero valid rewards are a truthful experiment outcome but do not establish a
learning update. Missing or truncated reward evidence is an infrastructure
failure, not a zero-reward model result.
