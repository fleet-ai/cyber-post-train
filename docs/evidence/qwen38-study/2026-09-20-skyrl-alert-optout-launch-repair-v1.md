# Qwen3.8 SkyRL alert-opt-out launch repair

Observed at 2026-09-20 on branch `codex/q38-skyrl-full-queue-v1` in
`/Users/christan/.codex/worktrees/skyrl-reward-canary-main-port-v2`.
The alert-enforcement base commit is
`ca7a3b65a6c9025465fead7ee346a2f9710ab371`. This was a preview-only pass:
no Kubernetes object, Fleet job, rollout, optimizer step, checkpoint, or GPU
allocation was created.

## Exact launch inputs

- Development topology configuration:
  `configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json`
- Production one-step reward canary:
  `configs/qualification/qwen38-rl-reward-canary-prod-v4.json`
- Production qualification closure:
  `configs/qualification/qwen38-rl-reward-canary-port-v4.json`
- Production queue closure:
  `configs/qualification/qwen38-skyrl-production-queue-v1.json`

The scientific recipe and data bindings did not change. The launch transport
now requires the exact root annotation
`fleet.ai/failure-alerts: "off"` on every generated Kubernetes Job, FleetJob,
and RayJob. Generic Jobs API requests bind `failureAlerts: false` and reject a
preview that does not render the root RayJob annotation.

The repaired development identities are:

- plan: `fbdd777447219623623fc9679aecc65f3b3c692aa20a9d49a4cc66ae29e43774`
- sealed request: `87e3a0971f1d700d57194cc87fb1ec3b1df0e16fc1dfca9d8f634bf0a4209e37`
- CPU preflight Job: `92a39c456d53f57e8ccbcc8a56b3427d815ed1c70e2c40464e0e7b1d2444aa37`
- FleetJob: `1fc00898bbe35739db9896a2765bad734ee616d72c0caa26f1f73434208ba223`
- receipt-verifier Job: `255ae49210d50c285bf9ad4c4a330cc3ccc95070bd0dca9e0d19cc81420a4668`

The repaired production canary compiles locally to:

- plan: `d6b36b07f1fcb9075e96f5d4918aecdf204c8ad2f3fa8d5d29527bfc9c777c9c`
- Jobs API request: `bb69d48afd9d6a1cafea678dbc408f803dda93e4e32bb42a23aac620da7c224c`

## Exact non-creating preview commands

The packet used in this pass was prepared with:

```sh
uv run --locked cyber-post-train rl-topology-probe \
  configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json \
  --output /private/tmp/q38-skyrl-alert-optout.uV0xth/packet
uv run --locked cyber-post-train rl-topology-probe-preflight-preview \
  /private/tmp/q38-skyrl-alert-optout.uV0xth/packet
uv run --locked cyber-post-train rl-topology-probe-preview \
  /private/tmp/q38-skyrl-alert-optout.uV0xth/packet
uv run --locked cyber-post-train rl-topology-probe-receipt-preview \
  /private/tmp/q38-skyrl-alert-optout.uV0xth/packet
```

All preview commands use `kubectl create --dry-run=server`; they do not create
an object.

## Server result

- The zero-GPU CPU preflight Job passed server dry-run. Its sanitized proof is
  `sha256:3dcb6833e9e0ebdf3790bbaf2e136bd380295b7669a03890ada8497ccf913569`.
  The server-rendered object digest is
  `e75aa6483e2c1f1355a95a824e85a626ce537cb10b33984c154abaa4c9ba1ca0`.
- The zero-GPU receipt-verifier Job passed server dry-run. Its sanitized proof
  is `sha256:e54ef36ae1f818c13110ab64950387bd0b980a2de2f4011564941cd6a063dc80`.
  The server-rendered object digest is
  `4e56e739d2548881301e2d15d462f93d4f68130b552e4ab4ef7e74283b16e0cc`.
- The exact annotated FleetJob was denied by the current development admission
  webhook. Removing both alert annotations makes the same server dry-run pass;
  retaining only the FleetJob annotation is denied because Fleet identity and
  queue metadata are controller-owned, and retaining only the embedded RayJob
  annotation is also denied. No FleetJob was created.

Therefore the current FleetJob interface does **not** support this mandatory
alert-safe launch. The development GPU topology probe remains blocked until the
FleetJob controller accepts and propagates the annotation, or the probe is
moved to a direct root-annotated RayJob transport that preserves the same model
mount and one-node topology.

The production generic Jobs API route is also fail closed. The request carries
`failureAlerts: false`, but launch remains prohibited until its exact server
preview proves the root RayJob annotation. That prod4 preview was not attempted
in this pass because the qualification gate is still closed on the topology,
data staging, CPU-preflight, and duplicate checks. A request field by itself is
not proof that the deployed server supports the annotation.

## Regression coverage

The focused suite asserts the annotation on the CPU preflight Job, FleetJob,
embedded RayJob, receipt-verifier Job, and generic SkyRL request. It removes
each annotation in turn and requires validation to fail. The suite result was
`246 passed, 1 skipped`.
