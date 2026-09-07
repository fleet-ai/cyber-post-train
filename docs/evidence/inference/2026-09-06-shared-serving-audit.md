# Shared Qwen3.8 27B and GLM-5.3 serving audit — 2026-09-06

This is a sanitized, mutable-state observation. It proves what Kubernetes exposed at the
observation time; it is not a durable guarantee of later readiness.

## Sources

- Kubernetes context: `nebius-mk8s-fleetai-training-e04zw4ye1k7wczqdw6`
- Namespace: `inference`
- Theseus source inspected at `8595b7bef23a4b98b1d2653e78dbb917268959e7`
- Deployed FTL monitor image source: `91212928e107ff949899f37a563d7cd0b9c9d123`

No workload, route, model, or peer resource was mutated during this audit.

## Qwen3.8 27B

- InferenceModel UID: `d06c0531-7181-41ad-a3fd-cd8e3e774ab9`
- Revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Runtime: SGLang BF16, TP1, one GPU, 262144 context, FP8 E4M3 KV cache
- Runtime image digest:
  `sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1`
- Observed Deployment replicas: `0`
- Observed ready Service endpoints: `0`
- Classification: registered but not live; stale InferenceModel status still said ready.

## GLM-5.3

- InferenceModel UID: `cedb0210-f3bd-4c56-97e9-aa53c08a0ce0`
- Revision: `30333038ada1f1dacb294a93270305a890b50c14`
- Runtime: SGLang FP8, TP8/DP8/EP8, eight GPUs, 262144 context, BF16 KV cache
- Runtime image digest:
  `sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1`
- Observed Deployment replicas: `1`
- Observed ready Service endpoints: `1`
- Observed model-container restarts: `0`
- Classification: live and ready at the observation time.

## Inference control plane

- Pod UID: `81438977-61cc-433a-a7fc-c2cc94edcd62`
- Process state: running with zero restarts
- Readiness: HTTP 503 `stale`
- Reconcile-error counter: `9`
- Last successful reconcile age: approximately 203826 seconds at observation
- Consequence: catalog and InferenceModel statuses were stale, and new registrations could
  not be expected to reconcile into live workloads.

## Capacity observation

Every schedulable node carrying either shared model's exact topology selector was occupied
at the observation time. The additional exact Qwen and GLM replicas would have waited for
normal capacity. No peer workload was changed or considered for preemption.

## Next gate

Diagnose and restore the shared inference-control-plane reconciliation loop, then prove the
shared Qwen route has a ready backend. Only after that should separately named exact-replica
registrations be previewed and submitted one at a time.

## Authorized repair follow-up

After the read-only audit, Chris authorized execution of the endpoint-recovery plan. The
stale inference-control-plane Pod was replaced through its owning Deployment; no
InferenceModel, serving Deployment, route, batch Job, RayJob, or peer workload was edited.

- Replacement control-plane Pod UID:
  `0a8c7d1e-9be1-4a1c-9820-c406caa434c0`
- Observed state: running, ready, zero restarts
- `/readyz`: `ready`
- Reconcile errors after replacement: `0`
- Catalog snapshot and successful-reconcile age: approximately five seconds at verification
- Fleet identity preflight: team
  `a1025f0b-ad67-49fc-a023-51800ab43e84` (`fleet`)

The fresh controller restored the Qwen Deployment's desired replica count to one without
changing its immutable serving contract. The recreated Pod UID is
`fc5e2f06-5ca9-4e4f-adb4-35181af69526`. It remained Pending because no schedulable GPU
matched its exact topology selector: the one matching schedulable node lacked free GPU
capacity, one matching node was unschedulable, and all other nodes did not match. Its
priority class has `preemptionPolicy=Never`, so it did not displace peer work.

The authenticated public catalog then reported:

| Model | Exact revision | Status | Ready replicas |
|---|---|---:|---:|
| `glm-5.3` | `30333038ada1f1dacb294a93270305a890b50c14` | `ready` | 1 |
| `qwen3.8-27b` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `starting` | 0 |

No additional exact replica was registered. Shared Qwen readiness is still the canary gate,
and the capacity census found no free schedulable B300 node for the exact Qwen or GLM
topology. The experiment's four-node/32-GPU ceiling is a usage ceiling, not permission to
uncordon maintenance nodes, provision infrastructure, or preempt another owner.
