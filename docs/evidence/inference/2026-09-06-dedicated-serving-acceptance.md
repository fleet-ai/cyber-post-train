# Qwen3.8 27B and GLM-5.3 dedicated serving acceptance — 2026-09-06

This is a sanitized serving acceptance record. It establishes endpoint identity and
operational readiness; it is not a scored evaluation result.

## Registration inputs

- Qwen: [`qwen38-27b-dedicated-v1.registration.json`](../../../evals/fleet/serving/qwen38-27b-dedicated-v1.registration.json)
- GLM: [`glm53-dedicated-v1.registration.json`](../../../evals/fleet/serving/glm53-dedicated-v1.registration.json)

Each input was normalized against the corresponding live shared `InferenceModel` and
matched exactly after changing only the public id, display name, and runtime
`--served-model-name`. The control plane accepted both request bodies without other
semantic changes; its sole added field was `desiredState: serving` and capability order
was not significant.

Registration used the authenticated inference-control-plane API with a stable
`Idempotency-Key` after both the public API and Kubernetes returned no matching model.
The credential was held only in a temporary interactive environment and then unset.

## Accepted endpoints

| Serving block | InferenceModel UID | Pod UID | Node | GPUs | Ready | Restarts |
|---|---|---|---|---:|---:|---:|
| Shared Qwen `qwen3.8-27b` | `d06c0531-7181-41ad-a3fd-cd8e3e774ab9` | `fc5e2f06-5ca9-4e4f-adb4-35181af69526` | `computeinstance-e04nnbyjvf87b5q53j` | 1 | 1/1 | 0 |
| Dedicated Qwen `chris-cyber-qwen38-27b-dedicated-v1` | `54beeb64-c498-4e7d-a504-05a8d694808b` | `6eaef829-001a-4ca4-8138-386238ab5c52` | `computeinstance-e04nnbyjvf87b5q53j` | 1 | 1/1 | 0 |
| Shared GLM `glm-5.3` | `cedb0210-f3bd-4c56-97e9-aa53c08a0ce0` | `c47907a6-0e11-45d6-82a5-b377c1a54903` | `computeinstance-e04f324s4mayq2tapm` | 8 | 1/1 | 0 |
| Dedicated GLM `chris-cyber-glm53-dedicated-v1` | `5801f9c0-db05-46be-a90a-72ba7a74d38f` | `f8a2bd8b-f0ce-414e-862f-0630f06d7e34` | `computeinstance-e04r0z7gp25s85eyb0` | 8 | 1/1 | 0 |

The four routes use 18 GPUs on three physical nodes, below the authorized ceiling of
32 GPUs on four nodes. Both Qwen replicas intentionally share one compatible node. The
priority class is `fleet-serve-low` with `preemptionPolicy=Never`; no peer resource was
cancelled, reprioritized, unsuspended, or preempted.

## Immutable identity

| Model | Revision | Runtime image digest | Precision | Parallelism | Context |
|---|---|---|---|---|---:|
| Qwen3.8 27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1` | BF16 | TP1 | 262144 |
| GLM-5.3 | `30333038ada1f1dacb294a93270305a890b50c14` | `sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1` | FP8 | TP8/DP8/EP8 | 262144 |

For each shared and dedicated route, the authenticated public catalog, native
`server_info`, and Kubernetes runtime image agreed on the exact binding. Qwen used
FP8 E4M3 KV cache, `qwen3` reasoning, and `qwen3_coder` tool parsing. GLM used BF16 KV
cache, `glm45` reasoning, and `glm47` tool parsing.

## Behavioral acceptance

Each route passed a non-scored forced structured-tool request through the public gateway.
The dedicated routes additionally passed four simultaneous forced structured-tool
requests, with all four requests returning exactly one expected call. No task prompt,
verifier, flag, trace, or score was read or persisted.

## Alert and lifecycle boundary

These are controller-owned Deployments in the `inference` namespace. No Kubernetes Job
or RayJob was created, so this rollout did not create a failed batch object for the
training-job Slack monitor. Persistent serving remains subject to inference health rules
and to the useful-consumer lifecycle in
[`GPU_RESOURCE_LIFECYCLE.md`](../../GPU_RESOURCE_LIFECYCLE.md). Attach scored consumers
only with explicit serving-block labels and release the dedicated endpoints when useful
rollout consumption ends or stalls outside its frozen allowance.
