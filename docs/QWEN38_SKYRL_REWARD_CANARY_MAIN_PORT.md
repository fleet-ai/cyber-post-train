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

- plan SHA-256 `51accb1d9857254fb4bee013001c361785698c5c5b7b825a7b328ee785b717cd`;
- request SHA-256 `287f9a1a6cb58ae5b03a002be8f14109c53abcf9e097cb410e9d9fa1a3fa146b`;
- FleetJob manifest SHA-256 `047485386a653f3c8e6cb7b16b853b0bd12accf027b862c9fab58f0fbf5e0b99`;
- CPU-preflight manifest SHA-256 `b36a70da18448291771bddbeee9f17c007249bddfe402c5b0aef0bc283d60ddb`;
- receipt-verifier manifest SHA-256 `59c3c1f1fd22c523fd8860e4d61155ee884a1c1002dff7f23d57ffc893b33ee7`.

The locally compiled scientific packet is bound to plan SHA-256
`513e39ff78605f74ab08af22187cb7e2bb3ab2f9290da03a1402d7d282f03a65`
and request SHA-256
`07fc87fda635341352a1726cb061de874be5b4a1eabc82985e8b28ef85104473`.
Those digests do not authorize submission.

## Why submission is blocked

The command surface refuses external preview and submission before creating a
Jobs client. The blocked closure records five unresolved gates:

1. an explicit failure-budget reset (the current count is 10/10);
2. accepted `v17` topology receipt plus proof that its eight GPUs were released;
3. create-once staging and digest verification of the exact one-train/one-dev
   data manifest under the `prod4` data path;
4. an exact-image CPU preflight and Jobs API preview for `prod4`; and
5. a final read-only duplicate/output/W&B absence check immediately before the
   one allowed create.

The five full arms have additional unresolved gates. Their private manifests
are not staged, their production validator and exact plan/request seals do not
exist, and no independent UID-bound release observer is recorded. The shared
RL watchdog has an eight-hour hard bound, while the legal episode ceilings are
38,400 seconds for each ten-step arm and 163,200 seconds for the fifty-step arm,
before startup and optimizer time. A reviewed plan-bound watchdog/release
contract is therefore required before any full-arm preview or submission.

The generic submit path checks the Jobs history for name/output reuse and uses
an exclusive submission journal. It does not perform the required fresh
Kubernetes, SFS, or W&B absence checks. Those checks remain explicit blockers;
the historical clean observation in the queue receipt must not be reused.

After those gates pass, the scientific run may be created once. Acceptance
requires eight genuine rollouts, non-constant verifier rewards, one finite
optimizer update, a reloadable step-1 checkpoint, and complete resource release.
All-zero valid rewards are a truthful experiment outcome but do not establish a
learning update. Missing or truncated reward evidence is an infrastructure
failure, not a zero-reward model result.
