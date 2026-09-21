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
`fleet.ai/failure-alerts: "off"` on every generated Kubernetes Job and RayJob.
The outer FleetJob is a separate wrapper and is deliberately unannotated.
Generic Jobs API requests bind `failureAlerts: false` and reject a
preview that does not render the root RayJob annotation.

The repaired development identities are:

- plan: `b398e73ba3124c4f22e53aceb9ac7924916ba2a3027a3f780261d184505ae678`
- sealed request: `def61728802c80a304a5f2ffdc0c1c22dae785737f2f23c03e9455b1712bda83`
- CPU preflight Job: `64f06f9a9ed4db4de4cd2ce3afa2b84b25b8f7d4be471c32dd96e3842004ef14`
- FleetJob: `0f8c7f3f5a6772fd4aaef706d8d2a638e6e7b24cb87edc6c3b0937d95bd6f389`
- receipt-verifier Job: `1409b89a9c113ff6ca2b7cb1afcfdc8428c0bd475b123d0820f185c684e8edfe`

The repaired production canary compiles locally to:

- plan: `25d0abf30da462a6ba67c6ac8a3fc95f3f89a9a08e3ea295b21e8a22481630b3`
- Jobs API request: `98a83dbbab61f360a12ec6389adb55de637ded64dcceaa1debf0d791e754cef4`

## Exact non-creating preview commands

The packet used in this pass was prepared with:

```sh
uv run --locked cyber-post-train rl-topology-probe \
  configs/qualification/qwen38-skyrl-topology-probe-dev-v2.json \
  --output /private/tmp/q38-skyrl-rayjob-optout.dACZ9u/packet
uv run --locked cyber-post-train rl-topology-probe-preflight-preview \
  /private/tmp/q38-skyrl-rayjob-optout.dACZ9u/packet
uv run --locked cyber-post-train rl-topology-probe-preview \
  /private/tmp/q38-skyrl-rayjob-optout.dACZ9u/packet
uv run --locked cyber-post-train rl-topology-probe-receipt-preview \
  /private/tmp/q38-skyrl-rayjob-optout.dACZ9u/packet
```

All preview commands use `kubectl create --dry-run=server`; they do not create
an object.

## Server result

- The zero-GPU CPU preflight Job passed server dry-run. Its sanitized proof is
  `sha256:673afacf9c2ed5c6785f4ffbb1d886f4e2292e641858950267c4adc0e5ca86aa`.
  The server-rendered object digest is
  `cc413777ab8874719b650eba817b72a96c086ea9cf0cf6191c8f7b9592cbfe48`.
- The zero-GPU receipt-verifier Job passed server dry-run. Its sanitized proof
  is `sha256:2a6e66a19698220d9051b37b1858bd44f011340122b0d06ffc8d5d8e8324314a`.
  The server-rendered object digest is
  `a6801e09be3f5b62483ee66433c6894d13773ec2dfee92c3b04d5f5dd2ddefb8`.
- The outer FleetJob carried no alert annotation. Its embedded root RayJob did
  carry `fleet.ai/failure-alerts: "off"`, and the current development admission
  webhook denied that exact FleetJob. Removing only the embedded RayJob alert
  annotation makes the same server dry-run pass. No FleetJob was created.

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

The focused suite asserts the annotation on the CPU preflight Job, embedded
RayJob, receipt-verifier Job, and generic SkyRL request. It also asserts that
the outer FleetJob remains unannotated. It removes each required annotation in
turn and requires validation to fail. The suite result was
`246 passed, 1 skipped`.
