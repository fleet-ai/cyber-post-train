# Fleet route and local-architecture parity finding — 2026-09-05

Status: frozen sanitized infrastructure finding; no workload was launched or
mutated while collecting this evidence.

## Exact GLM treatment question

The frozen target was `zai-org/GLM-5.3` at revision
`30333038ada1f1dacb294a93270305a890b50c14`, using OpenCode `1.18.27`, native
post-compaction continuation, a 262,144-token context, the frozen tool schema,
and exact `eval_task_version_id` bindings.

The `/v1/jobs` catalog model `z-ai/glm-5.3` is **not poolable** with that
treatment. At deployed Agent Runtime source commit
`01ad9a335ffcfe454f1c2d716766272b2e1c278b`, generic non-Fleet-prefixed models
route through OpenRouter. The catalog entry declares context/output limits but
does not attest `zai-org/GLM-5.3`, its immutable checkpoint revision, or an
immutable provider-route revision. OpenCode version, compaction, and context
agreement do not repair this missing model/route identity.

Relevant read-only sources in the Theseus repository:

- `shared/runtime/internal/mcpconfig/config.go`: generic OpenRouter routing and
  the distinct `fleet-glm/glm-5.3-fleet` route;
- `shared/runtime/Dockerfile`: OpenCode `1.18.27` pin;
- `shared/runtime/internal/opencode/opencode.go`: native post-compaction
  continuation;
- `k8s/platform/clusters/fleet-platform/apps/agent-runtime.yaml`: deployed
  Agent Runtime source and Fleet inference origin.

The generic catalog route may be reported as a separate exploratory provider
block. The distinct Fleet GLM route remains ineligible for pooling until live,
sanitized evidence binds the exact model revision, provider and endpoint route,
deployment revision, harness treatment, tools, and exact task versions.

## Local image architecture finding

The generation-2 local evaluator path paired Arm64 and Amd64 host/image
architectures. That mismatch failed at the executable boundary before a
scientifically valid rollout could be accepted. It is an infrastructure
failure, not a model outcome.

Scored local execution now requires native architecture equality for every
evaluator image. `evals.fleet.treatment_parity` normalizes
`x86_64`/`amd64` and `aarch64`/`arm64`, then fails closed when the image and host
differ. Cross-architecture emulation can still be studied separately, but its
results cannot enter a native treatment block.
