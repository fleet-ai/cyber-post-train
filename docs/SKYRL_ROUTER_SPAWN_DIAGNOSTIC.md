# SkyRL router process-boundary diagnostic

This is an operational compatibility fix, not RL or model qualification.

## What failed, and what is known

The previous repository wrapper rejected every multiprocessing start method
other than `fork` before entering Ray initialization. It also replaced the
native router target with a local closure, which would not be picklable under
`spawn` even if the guard alone were removed.

The sanitized pre-Ray CPU probe reported `fork` before native imports and
`spawn` afterward. It then stopped at the earlier native resource check because
a CPU-only probe cannot satisfy the requested GPU placement. Consequently that
probe establishes the context incompatibility but did not execute actor or
engine initialization. Without a diagnostic terminal receipt identifying the
exception site, attributing the earlier dev4 `ValueError` specifically to the
fork guard is a strong inference, not independently confirmed incident evidence.

The [exact upstream native router source](https://github.com/NovaSky-AI/SkyRL/blob/f5bc3b78dfddfb352870d5d7430cd226e5785838/skyrl/backends/skyrl_train/inference_servers/vllm_router.py)
constructs `multiprocessing.Process` using its module-global target. It therefore
uses the current process context; it does not force `fork`. The pinned-image
CPU test checks this file's SHA-256 before exercising its actual `start` and
`shutdown` methods.

## Narrow fix

- Preserve the native target as its original importable global function.
- Wrap only the router module's process constructor with a top-level picklable
  entrypoint. Preserve the native process arguments, name, daemon setting,
  lifecycle and returned process object.
- Blank credential-like environment variables during launch and restore the
  parent's exact values afterward. The child checks context and nonempty
  credential presence before invoking the native target. The shared proof
  contains an integer state only, never credential values.
- Support `fork` and `spawn`. Reject `forkserver`: an already-running server can
  retain an older environment, so its inheritance is not assumed safe.
- Preserve exact router/actor/placement-group tracking and cleanup checks.

A specifically recognized contract rejection before any Ray initialization may
return zero only after the empty owned-resource state and unchanged output
bundle are proven. Its receipt is `diagnostic_contract_rejected`, with
`engine_start_qualified=false`, all training qualification flags false and zero
tasks, rewards, optimizer updates and checkpoints. Arbitrary exceptions,
unproven cleanup and changed output still fail. No production failure is
relabelled or erased by this code change.

## Qualification scope

The tests exercise real local `fork` and `spawn` children, positive and negative
credential checks, parent restoration, target/context drift, typed pre-Ray
rejection, and failure of cleanup/output proof. The pinned-image-only test uses
the actual native router start/shutdown methods but a synthetic child target
and a substituted health check. It performs no Ray initialization, model load,
engine startup, rollout, task request, verifier call or optimization.

### Exact-image qualification, 2026-09-12

The root operator's standalone CPU self-test passed in the immutable training
image, with these exact bindings:

- Image SHA-256: `ba288751cd227c5be146d28f4a03237545d87d2cbd4c48464945b17fde566ff4`.
- Tested `training/skyrl_training.py` SHA-256:
  `30ff8db9f711665ad733449b5c23ae82c64be64bc7d1d1e6ec3abb8fd4126fb3`.
- Installed native router SHA-256:
  `f39f7b00125d670773cf1a461f2bc2e13e32dc40a84936da4d96b2db8e315a34`.
- CPU Pod: `chris-q38-rl-spawn-selftest-v1`, UID
  `ab400729-828f-418c-b3c7-740d5acc1d12`.

The observed native context was `spawn`. One picklable synthetic child passed
credential isolation and exited zero; the router was released, and the original
native target and multiprocessing module reference were restored. No GPUs,
Ray initialization, model loading or HTTP requests were used. This was a
standalone exact-image check, not a claim that every pytest case ran in that
image. Router release is also not a claim that the shared CPU helper Pod was
deleted; its remaining operator work and lifecycle are separate.

Local tests alone are not sufficient evidence for the pinned image. This
exact-image result qualifies only the tested process boundary. A subsequent
authorized GPU canary still has to prove real engine startup and cleanup; RL
remains unqualified until task/reward/update/checkpoint gates are independently
satisfied. Never edit or replay the old failed plan.
