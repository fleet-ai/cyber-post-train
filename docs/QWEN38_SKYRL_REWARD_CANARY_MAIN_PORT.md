# Qwen3.8 SkyRL reward-canary main port

Status: **prepared source closure; external preview and submission are blocked.**

This is the current-main port of the scientifically valid one-node reward
canary from runtime commit `8b5e8a00521b4fe412e3e7d0bce9b8ea6855b005`,
with preflight provenance from
`6c0bc541e136fc114a3f7a17ce6ba07e9178bb78`. The prior preflight is
historical evidence only. It certifies neither these rebased runtime bytes nor
the new create-once identities.

The executable inputs are:

- data config:
  [`qwen38-rl-reward-canary-data-dev-v3.json`](../configs/qualification/qwen38-rl-reward-canary-data-dev-v3.json);
- run config:
  [`qwen38-rl-reward-canary-dev-v3.json`](../configs/qualification/qwen38-rl-reward-canary-dev-v3.json);
- source and submission closure:
  [`qwen38-rl-reward-canary-port-v3.json`](../configs/qualification/qwen38-rl-reward-canary-port-v3.json).

The port preserves the exact train and dev task versions, ordered
`bash`/`submit_report` surface, fractional partial reward contract,
98,304/81,920-token context and response limits, 600-turn ceiling,
2,400-second episode limit, 330-second tool limit, one node, eight GPUs, two
TP4 engines, eight samples, one optimizer step, learning rate `1e-6`,
pre/post evaluation, step-1 checkpoint, and seed 42. The only runtime repair is
canonical ordering of the already digest-verified live tool schema before Qwen
prompt rendering.

The v3 configuration selects immutable image
`sha256:89758df2b5f35cdb19efe948c7f6ef54f11e2e2ab47a45d600c25f36914e308f`
and disables the disqualified FlashInfer sampler path. Its retained engine
receipt proves two-by-four engine startup and release only; it is not one-by-eight
training qualification.

## Why submission is blocked

Preparation and local CPU preflight may proceed, but the command surface
refuses external preview and submission before creating a Jobs client. The
blocked closure records five unresolved gates:

1. a fresh main-based CPU preflight and preview;
2. fresh Jobs/SFS/W&B absence proofs for the new identities;
3. an explicit dev API target in current main;
4. resolution of the preserved 2,400-second episode horizon versus the current
   30-minute dev-workload deadline; and
5. one-by-eight topology qualification.

Reducing the horizon or silently sending this dev canary through the production
Jobs endpoint would change or misroute the experiment. Resolve those contracts,
mint a successor closure, regenerate every plan/request/preflight digest, and
only then reconsider submission.
